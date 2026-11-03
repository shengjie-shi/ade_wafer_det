"""近同心环晶圆的离线缺陷检测。所有处理均在本地完成。"""

from .detector import DetectionResult, DetectorConfig, Defect, WaferGeometry, detect

__all__ = ["DetectionResult", "DetectorConfig", "Defect", "WaferGeometry", "detect"]
