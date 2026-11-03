"""离线命令行入口：检测实际图片，或运行包含真值的模拟演示。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import cv2
import numpy as np

from .detector import DetectorConfig, WaferGeometry, detect
from .io import read_image, save_result, write_png
from .synthetic import PROFILES, make_wafer


def _config(path: Path | None) -> DetectorConfig:
    if path is None:
        return DetectorConfig()
    values = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(values, dict):
        raise ValueError("配置文件必须是 JSON 对象")
    if "scales" in values:
        values["scales"] = tuple(values["scales"])
    return DetectorConfig(**values)


def _demo(output: Path, config: DetectorConfig) -> None:
    tiles = []
    overview = []
    for index, profile in enumerate(PROFILES):
        sample = make_wafer(profile=profile, seed=42 + index)
        result = detect(sample.image, config)
        destination = output / profile
        save_result(sample.image, result, destination)
        write_png(destination / "normal.png", sample.normal_image)
        write_png(destination / "truth_labels.png", sample.truth_labels)
        truth = {
            "notice": "注入真值，不是算法预测；模拟不代表真实 GaN 成像物理过程。",
            "spots": [asdict(spot) for spot in sample.spots],
        }
        (destination / "truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2), encoding="utf-8")
        overlay = read_image(destination / "overlay.png")
        tile = np.full((410, 390, 3), 245, dtype=np.uint8)
        tile[36:401, 14:375] = overlay
        cv2.putText(tile, f"{profile}: {len(result.defects)} candidates", (14, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (40, 40, 40), 1, cv2.LINE_AA)
        tiles.append(tile)
        overview.append({"profile": profile, "injected": len(sample.spots),
                         "detected_candidates": len(result.defects)})
    contact_sheet = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
    write_png(output / "overview.png", contact_sheet)
    (output / "summary.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "cases": overview}, ensure_ascii=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="近同心环晶圆的小缺陷检测，全程本地离线运行")
    commands = parser.add_subparsers(dest="command", required=True)
    detection = commands.add_parser("detect", help="检测一张灰度/BGR 图片")
    detection.add_argument("input", type=Path)
    detection.add_argument("--output", type=Path, default=Path("output/result"))
    detection.add_argument("--geometry", nargs=3, type=float, metavar=("X", "Y", "R"),
                           help="可选：手动指定晶圆外圆 x/y/半径")
    detection.add_argument("--mask", type=Path, help="可选：单通道有效晶圆掩膜，非零为有效")
    detection.add_argument("--config", type=Path, help="可选：JSON 参数文件")
    demonstration = commands.add_parser("demo", help="生成四种环形模拟数据并运行检测")
    demonstration.add_argument("--output", type=Path, default=Path("output/demo"))
    demonstration.add_argument("--config", type=Path)
    args = parser.parse_args(argv)

    try:
        config = _config(args.config)
        if args.command == "demo":
            _demo(args.output, config)
        else:
            # 禁止结果文件覆盖本次输入或掩膜。其他已有结果会被当前运行更新。
            reserved = {"input.png", "background.png", "residual.png", "analysis_mask.png",
                        "defect_mask.png", "score.png", "overlay.png", "report.json", "intermediates.npz"}
            for source in (args.input, args.mask):
                if (source is not None and source.resolve().parent == args.output.resolve()
                        and source.name.lower() in reserved):
                    raise ValueError("输出目录中的结果文件会覆盖输入，请选择其他输出目录")
            image = read_image(args.input)
            mask = read_image(args.mask) if args.mask is not None else None
            if mask is not None and mask.ndim != 2:
                raise ValueError("--mask 必须是单通道图像")
            geometry = WaferGeometry(*args.geometry) if args.geometry is not None else None
            result = detect(image, config, geometry=geometry, wafer_mask=mask)
            report = save_result(image, result, args.output)
            print(json.dumps({"defect_count": len(result.defects), "report": str(report.resolve())}, ensure_ascii=True))
    except (ValueError, TypeError, OSError, cv2.error) as error:
        parser.exit(2, f"error: {error}\n")
    return 0
