"""端到端测试：本地图片 → 命令行 → 可读报告与精确中间结果。"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from wafer_defects import detect
from wafer_defects.io import read_image, save_result, write_png
from wafer_defects.synthetic import make_wafer


class IOAndCLITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 临时文件放在项目内，便于在限制系统临时目录写权限的环境运行。
        cls.temp_root = Path(__file__).resolve().parents[1] / "output" / "test-temp"
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def test_chinese_path_image_round_trip(self):
        with tempfile.TemporaryDirectory(prefix="wafer-tests-", dir=self.temp_root) as directory:
            path = Path(directory) / "晶圆样例.png"
            image = make_wafer().image
            write_png(path, image)
            np.testing.assert_array_equal(read_image(path), image)

    def test_exported_report_and_npz_preserve_detection(self):
        sample = make_wafer()
        result = detect(sample.image)
        with tempfile.TemporaryDirectory(prefix="wafer-tests-", dir=self.temp_root) as directory:
            output = Path(directory)
            report_path = save_result(sample.image, result, output)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["defect_count"], 3)
            self.assertEqual(report["image_size_wh"], [361, 365])
            self.assertEqual(report["inspected_pixels"], int(result.analysis_mask.sum()))
            self.assertEqual({row["polarity"] for row in report["defects"]}, {"bright", "dark"})
            for name in ("input.png", "background.png", "residual.png", "analysis_mask.png",
                         "defect_mask.png", "overlay.png", "score.png"):
                self.assertTrue((output / name).is_file(), name)
            with np.load(output / "intermediates.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(data["labels"], result.labels)
                np.testing.assert_array_equal(data["residual"], result.residual)
                np.testing.assert_array_equal(data["analysis_mask"], result.analysis_mask)

    def test_command_line_detect_with_json_config(self):
        with tempfile.TemporaryDirectory(prefix="wafer-tests-", dir=self.temp_root) as directory:
            root = Path(directory)
            image_path = root / "待检测晶圆.png"
            write_png(image_path, make_wafer(profile="single_ring").image)
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"scales": [1, 3, 5, 7], "seed_score": 6}), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-m", "wafer_defects", "detect", str(image_path),
                 "--output", str(root / "result"), "--config", str(config_path)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["defect_count"], 3)
            self.assertTrue(Path(summary["report"]).is_file())

    def test_cli_refuses_to_overwrite_its_input(self):
        with tempfile.TemporaryDirectory(prefix="wafer-tests-", dir=self.temp_root) as directory:
            root = Path(directory)
            image_path = root / "input.png"
            write_png(image_path, make_wafer().image)
            original = image_path.read_bytes()
            completed = subprocess.run(
                [sys.executable, "-m", "wafer_defects", "detect", str(image_path), "--output", str(root)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
            self.assertEqual(completed.returncode, 2)
            self.assertNotIn("Traceback", completed.stderr)
            self.assertEqual(image_path.read_bytes(), original)

    def test_unreadable_images_have_clear_errors(self):
        with tempfile.TemporaryDirectory(prefix="wafer-tests-", dir=self.temp_root) as directory:
            root = Path(directory)
            (root / "empty.png").write_bytes(b"")
            (root / "invalid.png").write_bytes(b"this is not an image")
            for path in (root / "empty.png", root / "invalid.png", root / "missing.png"):
                with self.subTest(name=path.name):
                    with self.assertRaises(ValueError):
                        read_image(path)


if __name__ == "__main__":
    unittest.main()
