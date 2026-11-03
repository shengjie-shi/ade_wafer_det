"""图像读写和诊断结果导出；无网络请求，支持中文文件路径。"""

from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from .detector import DetectionResult, _grayscale


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"图像不存在：{path}")
    content = np.fromfile(path, dtype=np.uint8)
    if not content.size:
        raise ValueError(f"图像文件为空：{path}")
    image = cv2.imdecode(content, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"无法解码图像：{path}")
    return image


def write_png(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"无法编码 PNG：{path}")
    encoded.tofile(path)


def save_result(image: np.ndarray, result: DetectionResult, output_dir: str | Path) -> Path:
    """导出可视图、精确浮点中间结果与 JSON。显示图不用于再次检测。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    gray = np.clip(np.rint(_grayscale(image)), 0, 255).astype(np.uint8)
    if gray.shape != result.labels.shape:
        raise ValueError("结果尺寸与图像不一致")
    write_png(output / "input.png", gray)
    write_png(output / "background.png", np.clip(np.rint(result.background), 0, 255).astype(np.uint8))

    # 有符号残差：中灰 128 表示 0；偏亮为正，偏暗为负，每灰度放大 3 倍。
    residual_view = np.clip(128 + 3 * result.residual, 0, 255).astype(np.uint8)
    residual_view[~result.analysis_mask] = 0
    write_png(output / "residual.png", residual_view)
    write_png(output / "analysis_mask.png", result.analysis_mask.astype(np.uint8) * 255)
    write_png(output / "defect_mask.png", result.binary_mask)
    score_view = np.clip(result.score_map * 16, 0, 255).astype(np.uint8)
    write_png(output / "score.png", score_view)

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[~result.analysis_mask] = (overlay[~result.analysis_mask] * .3).astype(np.uint8)
    for defect in result.defects:
        x, y, width, height = defect.bbox_xywh
        color = (20, 170, 250) if defect.polarity == "bright" else (250, 170, 20)
        cv2.rectangle(overlay, (max(0, x - 3), max(0, y - 3)),
                       (min(gray.shape[1] - 1, x + width + 2),
                        min(gray.shape[0] - 1, y + height + 2)), color, 1)
        cv2.putText(overlay, str(defect.id), (x + width + 4, max(12, y)),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
    write_png(output / "overlay.png", overlay)

    # NPZ 保留精确值；PNG 中的灰度截断及可视化增益不会丢失诊断数据。
    np.savez_compressed(output / "intermediates.npz", background=result.background,
                        residual=result.residual, noise_map=result.noise_map,
                        score_map=result.score_map, analysis_mask=result.analysis_mask,
                        labels=result.labels, radial_samples=result.radial_samples,
                        radial_profile=result.radial_profile)
    report = {
        "notice": "数学背景模型产生的图像异常候选，尚未在厂内真实 GaN 数据上验证。",
        "image_size_wh": [gray.shape[1], gray.shape[0]],
        "coordinate_convention": "x=列，y=行；bbox_xywh 为左上角 x/y 和宽/高，原点为左上角 0/0。",
        "wafer_geometry": asdict(result.wafer_geometry),
        "ring_center_xy": result.ring_center_xy,
        "inspected_pixels": int(result.analysis_mask.sum()),
        "coverage_notice": "请以 analysis_mask.png 的白色区域为实际检测范围；黑色区域未检查。",
        "visualization": {"residual_zero": 128, "residual_gain": 3, "score_gain": 16},
        "config": asdict(result.config),
        "defect_count": len(result.defects),
        "defects": [asdict(defect) for defect in result.defects],
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path
