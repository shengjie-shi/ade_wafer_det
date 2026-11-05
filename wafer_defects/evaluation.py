"""批量检测与本地对照报告。文件夹类别仅用于事后评估，不参与推理。"""

import csv
from dataclasses import asdict
import html
import json
from pathlib import Path
import re
from urllib.parse import quote

import cv2

from .detector import DetectorConfig, detect
from .io import read_image, save_result


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LABELS = {"明显缺陷": "positive", "不明显或无缺陷": "normal"}


def _natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.as_posix())]


def _image_label(relative: Path) -> str:
    """只供汇总使用。算法 detect 的输入始终只有像素及统一配置。"""
    for part in relative.parts[:-1]:
        if part in LABELS:
            return LABELS[part]
    return "unknown"


def _gallery(output: Path, cases: list[dict], metrics: dict, evaluate_labels: bool) -> Path:
    cards = []
    for case in cases:
        name = html.escape(case["relative_path"])
        category = case["label"]
        label_text = {"normal": "正常类", "positive": "缺陷类", "unknown": "未标注"}[category]
        if case["status"] == "error":
            content = f'<p class="error">{html.escape(case["error"])}</p>'
            count_text = "处理失败"
        else:
            directory = quote(case["output_relative"], safe="/")
            confirmation_link = (f'<a href="{directory}/confirmation.png">局部中值残差</a> · '
                                 if case.get("has_confirmation") else "")
            content = (
                f'<div class="images"><figure><img loading="lazy" src="{directory}/input.png">'
                f'<figcaption>输入</figcaption></figure><figure><img loading="lazy" src="{directory}/overlay.png">'
                f'<figcaption>候选标记（暗处为未检查区域）</figcaption></figure></div>'
                f'<p><a href="{directory}/report.json">坐标与参数</a> · '
                f'<a href="{directory}/residual.png">圆环残差</a> · '
                f'{confirmation_link}'
                f'<a href="{directory}/analysis_mask.png">检查范围</a></p>'
            )
            count_text = f'{case["defect_count"]} 个候选'
        cards.append(f'<article data-label="{category}"><h2>{name}</h2>'
                     f'<p class="meta">{label_text} · {count_text}</p>{content}</article>')
    summary = f'共 {metrics["images"]} 张，成功 {metrics["processed"]} 张，失败 {metrics["errors"]} 张。'
    if evaluate_labels:
        summary += (f' 正常类 {metrics["normal_images"]} 张，报出候选 {metrics["normal_flagged"]} 张；'
                    f'缺陷类 {metrics["positive_images"]} 张，检出候选 {metrics["positive_detected"]} 张。')
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>晶圆逐图检测对照</title>
<style>
body{{font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;margin:24px auto;max-width:1540px;padding:0 16px;color:#1d2935;background:#f5f6f8}}
h1{{font-size:26px}}h2{{font-size:17px;margin:0;overflow-wrap:anywhere}}article{{background:white;padding:16px;border-radius:8px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(600px,1fr));gap:18px}}.images{{display:flex;gap:8px}}figure{{flex:1;margin:0;min-width:0}}
img{{width:100%;height:auto;display:block;background:#101010}}figcaption,.meta{{font-size:13px;color:#52616f}}select{{padding:8px;margin:12px 0 20px}}
a{{color:#1261a0}}.error{{color:#9d251b}}.note{{max-width:1100px}}[hidden]{{display:none!important}}
@media(max-width:680px){{.grid{{grid-template-columns:1fr}}body{{margin:14px auto}}}}
</style><h1>晶圆逐图检测对照</h1><p>{html.escape(summary)}</p>
<p class="note">这是当前样例的调参回归结果，没有独立留出测试集。整图标签只用于事后汇总，不能证明所有缺陷点均被检出。
橙色框表示亮异常，蓝色框表示暗异常；同一小点的亮边与暗心可能合并为一个候选。
业务定义尚未确认的位置需要后续补充标注，当前没有完整逐点标注。</p>
<p><a href="summary.json">完整汇总</a> · <a href="summary.csv">逐图表格</a></p>
<label>显示 <select id="filter"><option value="all">全部</option><option value="positive">缺陷类</option><option value="normal">正常类</option><option value="unknown">未标注</option></select></label>
<main class="grid">{''.join(cards)}</main>
<script>document.getElementById('filter').addEventListener('change',e=>{{for(const a of document.querySelectorAll('article'))a.hidden=e.target.value!=='all'&&a.dataset.label!==e.target.value}});</script></html>'''
    path = output / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def run_batch(input_dir: str | Path, output_dir: str | Path,
              config: DetectorConfig | None = None, *, evaluate_labels: bool = False) -> dict:
    """逐图检测，保留目录结构，导出 JSON/CSV/HTML。

    同一个 config 实例用于所有图片。类别的读取发生在 detect 之后，
    更改文件名或移动到另一个类别文件夹不会改变单张图的检测结果。
    """
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not source.is_dir():
        raise ValueError(f"输入文件夹不存在：{source}")
    if output == source or output.is_relative_to(source):
        raise ValueError("输出文件夹必须位于输入文件夹之外，避免重复扫描生成的诊断图")
    files = sorted((p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                   key=_natural_key)
    if not files:
        raise ValueError("输入文件夹下没有支持的图片")
    config = config if config is not None else DetectorConfig.for_real_samples()
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    occupied = set()
    for image_path in files:
        relative = image_path.relative_to(source)
        destination = output / relative.with_suffix("")
        if destination in occupied:
            # 相同 stem、不同扩展名也不能互相覆盖。
            destination = destination.with_name(relative.name)
        occupied.add(destination)
        try:
            image = read_image(image_path)
            result = detect(image, config)
            save_result(image, result, destination)
            # 事后才读取目录标签，绝不把标签、文件名或路径传给 detect。
            label = _image_label(relative) if evaluate_labels else "unknown"
            case = {"relative_path": relative.as_posix(), "label": label, "status": "ok",
                    "defect_count": len(result.defects), "defects": [asdict(d) for d in result.defects],
                    "output_relative": destination.relative_to(output).as_posix(),
                    "has_confirmation": bool(config.confirmation_ksize),
                    "wafer_geometry": asdict(result.wafer_geometry),
                    "ring_center_xy": result.ring_center_xy}
        except (ValueError, OSError, cv2.error) as error:
            case = {"relative_path": relative.as_posix(),
                    "label": _image_label(relative) if evaluate_labels else "unknown",
                    "status": "error", "error": str(error)}
        cases.append(case)

    normal = [c for c in cases if c["label"] == "normal"]
    positive = [c for c in cases if c["label"] == "positive"]
    metrics = {
        "images": len(cases), "processed": sum(c["status"] == "ok" for c in cases),
        "errors": sum(c["status"] == "error" for c in cases),
        "normal_images": len(normal), "positive_images": len(positive),
        "normal_flagged": sum(c["status"] == "ok" and c["defect_count"] > 0 for c in normal),
        "normal_clear": sum(c["status"] == "ok" and c["defect_count"] == 0 for c in normal),
        "positive_detected": sum(c["status"] == "ok" and c["defect_count"] > 0 for c in positive),
        "positive_unflagged": sum(c["status"] == "ok" and c["defect_count"] == 0 for c in positive),
        "candidates": sum(c.get("defect_count", 0) for c in cases),
    }
    summary = {"evaluation_scope": "当前样例调参后的图像级回归，不是独立测试集或逐点召回率。",
               "labels_used_for_inference": False, "metrics": metrics,
               "config": asdict(config), "cases": cases}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("relative_path", "label", "status", "defect_count", "error"),
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cases)
    gallery = _gallery(output, cases, metrics, evaluate_labels)
    return {"report": str(gallery), "summary": str(output / "summary.json"), "metrics": metrics}
