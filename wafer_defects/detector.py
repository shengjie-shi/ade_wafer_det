"""同半径稳健背景估计 + 多尺度双极性小点检测。

本模块不假设圆环的数量、宽度或间距，也不使用固定正弦模板。
背景沿半径可以快速变化，沿圆周则允许缓慢变化。
注释中的尺寸均为原图像素；灰度阈值使用 0～255 灰度单位。
"""

from dataclasses import dataclass, replace
from math import ceil, isfinite
from typing import Literal

import cv2
import numpy as np


@dataclass(frozen=True)
class WaferGeometry:
    """晶圆外轮廓的近似圆。坐标顺序为 x、y，而非数组的行、列。"""

    center_x: float
    center_y: float
    radius: float


@dataclass(frozen=True)
class DetectorConfig:
    """显式构造时保留通用灵敏参数；真实图请从 for_real_samples() 开始。

    angular_sigma_px: 沿圆周背景平滑的标准差，按弧长像素计量，不是窗口边长。
    radial_step_px: 背景径向采样间隔，不是圆环宽度。
    noise_floor: 噪声标准差的下限，防止平坦区域的分数异常放大。
    seed_score: 多尺度响应的种子阈值，分数不是缺陷概率。
    grow_score: 种子周围允许纳入连通域的较低像素阈值。
    center_exclusion_px/edge_margin_px: 明确不参与检测的范围。
    """

    radial_step_px: float = 0.5
    angular_samples: int = 0  # 0 表示按晶圆尺寸自动选择，保证外圈采样密度。
    angular_sigma_px: float = 12.0
    background_iterations: int = 4
    background_clip_sigma: float = 3.0
    noise_floor: float = 1.0
    noise_bin_width_px: float = 6.0
    refine_center: bool = True
    center_search_radius_px: float = 3.0
    center_exclusion_px: float = 4.0
    edge_margin_px: int = 4
    foreground_threshold: float = 3.0
    scales: tuple[int, ...] = (1, 3, 5, 7)
    seed_score: float = 6.0
    grow_score: float = 2.5
    min_pixel_contrast: float = 3.0
    min_mean_contrast: float = 2.5
    min_peak_contrast: float = 8.0
    min_area: int = 3
    max_area: int | None = 200
    # 0 关闭第二种背景的交叉检查；真实图配置使用 5×5 局部中值背景。
    confirmation_ksize: int = 0
    confirmation_min_mean: float = 0.0
    confirmation_min_snr: float = 0.0
    confirmation_min_core: float = 0.0
    merge_distance_px: float = 0.0

    @classmethod
    def for_real_samples(cls) -> "DetectorConfig":
        """真实样例的统一起始配置；没有按文件名、类别或坐标写检测规则。

        圆周背景的尺度缩小，以跟随真实窄环。局部中值残差提供独立证据，
        对其均值、噪声比及核心响应联合检查，抑制正常环形过渡与相位误差。
        """
        return cls(angular_sigma_px=4.0, edge_margin_px=6, seed_score=18.0,
                   min_peak_contrast=28.0, min_area=4, confirmation_ksize=5,
                   confirmation_min_mean=3.0, confirmation_min_snr=1.3,
                   confirmation_min_core=5.0, merge_distance_px=4.0)

    def __post_init__(self) -> None:
        positive = (
            "radial_step_px", "angular_sigma_px", "background_clip_sigma",
            "noise_floor", "noise_bin_width_px", "seed_score", "grow_score",
            "foreground_threshold",
        )
        for name in positive:
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是有限正数")
        for name in ("center_search_radius_px", "center_exclusion_px",
                     "min_pixel_contrast", "min_mean_contrast", "min_peak_contrast",
                     "confirmation_min_mean", "confirmation_min_snr", "confirmation_min_core",
                     "merge_distance_px"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是有限非负数")
        for name in ("edge_margin_px", "angular_samples", "background_iterations", "min_area"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} 必须是整数")
        if self.edge_margin_px < 0 or self.background_iterations < 1 or self.min_area < 1:
            raise ValueError("边缘留白不能为负，背景迭代次数和最小面积至少为 1")
        if self.angular_samples != 0 and self.angular_samples < 64:
            raise ValueError("angular_samples 必须为 0 或至少 64")
        if self.radial_step_px > 1:
            raise ValueError("radial_step_px 应不大于 1，避免漏采样窄环")
        if not self.scales or any(not isinstance(k, int) or isinstance(k, bool)
                                  or k < 1 or k % 2 == 0 for k in self.scales):
            raise ValueError("scales 必须包含正奇数，例如 (1, 3, 5, 7)")
        if len(set(self.scales)) != len(self.scales):
            raise ValueError("scales 不能重复")
        if self.max_area is not None and (
            not isinstance(self.max_area, int) or isinstance(self.max_area, bool)
            or self.max_area < self.min_area
        ):
            raise ValueError("max_area 必须为 None 或不小于 min_area 的整数")
        if self.confirmation_ksize not in (0, 3, 5):
            raise ValueError("confirmation_ksize 必须为 0、3 或 5")


@dataclass(frozen=True)
class Defect:
    """一处异常候选。bbox_xywh 中的宽高包含边界像素。"""

    id: int
    polarity: Literal["bright", "dark"]
    bbox_xywh: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    area: int
    peak_contrast: float
    mean_contrast: float
    peak_score: float
    touches_analysis_boundary: bool
    confirmation_mean: float = 0.0
    confirmation_snr: float = 0.0
    confirmation_core_mean: float = 0.0


@dataclass
class DetectionResult:
    """保留中间结果，便于在厂内检查漏检或误报原因。

    labels: 0 为未检出区域，正整数对应 defects 中的 id。
    analysis_mask: True 才表示本次实际检查的范围。
    background/residual/noise_map/score_map 在检查范围外填 0。
    wafer_geometry 是外轮廓；ring_center_xy 是微调后的明暗环中心。
    """

    defects: list[Defect]
    labels: np.ndarray
    analysis_mask: np.ndarray
    background: np.ndarray
    residual: np.ndarray
    noise_map: np.ndarray
    score_map: np.ndarray
    radial_samples: np.ndarray
    radial_profile: np.ndarray
    wafer_geometry: WaferGeometry
    ring_center_xy: tuple[float, float]
    config: DetectorConfig

    @property
    def binary_mask(self) -> np.ndarray:
        return np.where(self.labels > 0, 255, 0).astype(np.uint8)


def _grayscale(image: np.ndarray) -> np.ndarray:
    """接受灰度、BGR 或 BGRA 数组；不按单图最大值自动拉伸对比度。"""
    array = np.asarray(image)
    if array.ndim not in (2, 3) or min(array.shape[:2], default=0) < 32:
        raise ValueError("图像必须是至少 32×32 的灰度、BGR 或 BGRA 数组")
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        raise ValueError("彩色图像必须有 3 或 4 个通道，顺序为 BGR/BGRA")
    if array.dtype.kind not in "uif" or not np.isfinite(array).all():
        raise ValueError("图像必须是有限数值，不能包含 NaN 或 Inf")
    if np.min(array) < 0 or np.max(array) > 255:
        raise ValueError("第一版接受 0～255 灰度；16 位数据请按固定传感器量程转换")
    gray = array.astype(np.float32, copy=True)
    if gray.ndim == 3:
        code = cv2.COLOR_BGR2GRAY if gray.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        gray = cv2.cvtColor(gray, code)
    return gray


def _mad_sigma(values: np.ndarray, floor: float) -> float:
    """MAD 对少量异常稳健；1.4826 将高斯噪声下的 MAD 换算为标准差。"""
    if values.size == 0:
        return floor
    median = np.median(values)
    return max(floor, float(1.4826 * np.median(np.abs(values - median))))


def _circle_from_mask(mask: np.ndarray) -> WaferGeometry:
    """用最大外轮廓拟合圆，避免把晶圆内部的亮暗环分别当作晶圆。"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("找不到晶圆区域，请提供有效掩膜或手动圆心、半径")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.05 * mask.size or len(contour) < 8:
        raise ValueError("有效晶圆区域太小，请检查图像或手动提供几何信息")
    points = contour[:, 0, :].astype(np.float64)
    # 平移坐标后再最小二乘拟合，降低大坐标造成的数值误差。
    origin = points.mean(axis=0)
    centered = points - origin
    design = np.column_stack((2 * centered, np.ones(len(points))))
    solution, _, rank, _ = np.linalg.lstsq(design, (centered ** 2).sum(axis=1), rcond=None)
    if rank < 3:
        raise ValueError("外轮廓不能稳定拟合为圆，请手动指定晶圆几何信息")
    center = solution[:2] + origin
    radius = np.sqrt(solution[2] + np.dot(solution[:2], solution[:2]))
    return WaferGeometry(float(center[0]), float(center[1]), float(radius))


def _circle_from_outer_rays(gray: np.ndarray, config: DetectorConfig) -> WaferGeometry | None:
    """沿射线找最外侧可见边界，避免文字和彩色直边拉偏整个外轮廓。

    这里只利用尺寸作宽松搜索约束，不固定 361×356、固定圆心或固定半径。
    被图像裁掉的射线不参与拟合，剩余点通过迭代稳健圆拟合去掉离群点。
    """
    height, width = gray.shape
    initial_x, initial_y = (width - 1) / 2, (height - 1) / 2
    radii = np.arange(min(height, width) * .2, min(height, width) * .56, .25)
    angles = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    map_x = (initial_x + np.cos(angles)[:, None] * radii).astype(np.float32)
    map_y = (initial_y + np.sin(angles)[:, None] * radii).astype(np.float32)
    smoothed = cv2.GaussianBlur(gray, (3, 3), .65)
    samples = cv2.remap(smoothed, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # 图像最外 2 像素可能是直线边框，不能作为晶圆的圆形边界点。
    inside_frame = (map_x >= 2) & (map_x < width - 3) & (map_y >= 2) & (map_y < height - 3)
    patch = max(2, min(10, min(height, width) // 20))
    corners = np.concatenate((gray[:patch, :patch].ravel(), gray[:patch, -patch:].ravel(),
                               gray[-patch:, :patch].ravel(), gray[-patch:, -patch:].ravel()))
    black = float(np.median(corners))
    threshold = black + max(4.0, config.foreground_threshold, 6 * _mad_sigma(corners, 0.0))
    foreground = (samples > threshold) & inside_frame
    indices = np.arange(len(radii))
    last_foreground = np.max(np.where(foreground, indices, -1), axis=1)
    last_visible = np.max(np.where(inside_frame, indices, -1), axis=1)
    # 外边界后必须还能看到约 2 像素背景；否则该方向已被截图截断。
    usable = (last_foreground >= 0) & (last_foreground < last_visible - 8)
    if np.count_nonzero(usable) < 48:
        return None
    rows = np.flatnonzero(usable)
    points = np.column_stack((map_x[rows, last_foreground[usable]],
                               map_y[rows, last_foreground[usable]])).astype(np.float64)
    for _ in range(5):
        if len(points) < 32:
            return None
        origin = points.mean(axis=0)
        centered = points - origin
        design = np.column_stack((2 * centered, np.ones(len(points))))
        solution, _, rank, _ = np.linalg.lstsq(design, (centered ** 2).sum(axis=1), rcond=None)
        radius_squared = solution[2] + np.dot(solution[:2], solution[:2])
        if rank < 3 or radius_squared <= 0:
            return None
        center = solution[:2] + origin
        radius = float(np.sqrt(radius_squared))
        errors = np.hypot(points[:, 0] - center[0], points[:, 1] - center[1]) - radius
        median = np.median(errors)
        tolerance = max(.5, 3 * _mad_sigma(errors, 0.0))
        points = points[np.abs(errors - median) < tolerance]
    if not (0 <= center[0] < width and 0 <= center[1] < height):
        return None
    return WaferGeometry(float(center[0]), float(center[1]), radius)


def _prepare_roi(gray: np.ndarray, config: DetectorConfig,
                 geometry: WaferGeometry | None,
                 wafer_mask: np.ndarray | None) -> tuple[WaferGeometry, np.ndarray]:
    height, width = gray.shape
    if wafer_mask is not None:
        supplied = np.asarray(wafer_mask)
        if supplied.shape != gray.shape or supplied.dtype.kind not in "buif":
            raise ValueError("wafer_mask 必须是与图像同尺寸的数值或布尔数组")
        if not np.isfinite(supplied).all():
            raise ValueError("wafer_mask 不能包含 NaN 或 Inf")
        roi = supplied > 0
        if not roi.any():
            raise ValueError("wafer_mask 不能为空")
    elif geometry is not None:
        yy, xx = np.indices(gray.shape, dtype=np.float32)
        roi = np.hypot(xx - geometry.center_x, yy - geometry.center_y) <= geometry.radius
    else:
        outer_circle = _circle_from_outer_rays(gray, config)
        if outer_circle is not None:
            yy, xx = np.indices(gray.shape, dtype=np.float32)
            roi = np.hypot(xx - outer_circle.center_x, yy - outer_circle.center_y) <= outer_circle.radius
            return outer_circle, roi
        # 默认边角为背景。先估计背景灰度与噪声，再提取最外侧可见轮廓。
        # 填充轮廓内部的黑环，防止把正常暗环误认为有效区域中的洞。
        patch = max(2, min(height, width) // 20)
        corners = np.concatenate([gray[:patch, :patch].ravel(), gray[:patch, -patch:].ravel(),
                                  gray[-patch:, :patch].ravel(), gray[-patch:, -patch:].ravel()])
        black = float(np.median(corners))
        threshold = black + max(config.foreground_threshold, 6 * _mad_sigma(corners, 0.0))
        foreground = (gray > threshold).astype(np.uint8)
        contours, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            raise ValueError("找不到晶圆区域，请检查黑色边角或手动指定几何信息")
        largest = max(contours, key=cv2.contourArea)
        roi_u8 = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(roi_u8, [largest], -1, 1, thickness=cv2.FILLED)
        roi = roi_u8.astype(bool)
    if geometry is None:
        geometry = _circle_from_mask(roi)
    values = (geometry.center_x, geometry.center_y, geometry.radius)
    if not all(isfinite(v) for v in values) or geometry.radius < 8:
        raise ValueError("圆心和半径必须有限，半径至少为 8 像素")
    if not (0 <= geometry.center_x < width and 0 <= geometry.center_y < height):
        raise ValueError("圆心必须位于图像内")
    if geometry.radius > np.hypot(width, height):
        raise ValueError("半径明显超过图像尺寸")
    return geometry, roi


def _polar_sample(gray: np.ndarray, mask: np.ndarray, center: tuple[float, float],
                  radii: np.ndarray, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """只在这里把原图采样到极坐标，避免反复缩放削弱小缺陷。"""
    map_x = (center[0] + radii[:, None] * np.cos(angles)[None, :]).astype(np.float32)
    map_y = (center[1] + radii[:, None] * np.sin(angles)[None, :]).astype(np.float32)
    values = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # 使用线性采样的掩膜并要求接近 1，防止黑色边角参与插值。
    coverage = cv2.remap(mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return values, coverage > 0.999


def _refine_ring_center(gray: np.ndarray, roi: np.ndarray, geometry: WaferGeometry,
                        config: DetectorConfig) -> tuple[float, float]:
    """以外轮廓圆心为初值，寻找圆周灰度更一致的环中心；不改变晶圆掩膜。"""
    initial = (geometry.center_x, geometry.center_y)
    if not config.refine_center or config.center_search_radius_px == 0:
        return initial
    radii = np.arange(max(6.0, config.center_exclusion_px),
                      geometry.radius - config.edge_margin_px - 3, 2.0, dtype=np.float32)
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False, dtype=np.float32)
    if len(radii) < 4:
        return initial

    def objective(center: tuple[float, float]) -> float:
        values, valid = _polar_sample(gray, roi, center, radii, angles)
        # 只用整圈都有效的半径，避免圆边缘的背景黑色拉动圆心。
        rows = valid.all(axis=1)
        if rows.sum() < 4:
            return float("inf")
        values = values[rows]
        radial_median = np.median(values, axis=1, keepdims=True)
        # 中位绝对偏差抑制稀疏缺陷，允许不同半径有完全不同的灰度。
        dispersion = np.median(np.abs(values - radial_median), axis=1)
        return float(np.mean(dispersion))

    center, best = initial, objective(initial)
    # 粗到细搜索；搜索范围以外轮廓初值为准，不会逐轮无限漂移。
    for step in (1.0, 0.5, 0.25):
        for _ in range(max(1, ceil(config.center_search_radius_px / step))):
            previous = center
            for oy in (-step, 0.0, step):
                for ox in (-step, 0.0, step):
                    candidate = (previous[0] + ox, previous[1] + oy)
                    if (abs(candidate[0] - initial[0]) > config.center_search_radius_px
                            or abs(candidate[1] - initial[1]) > config.center_search_radius_px):
                        continue
                    if not (0 <= candidate[0] < gray.shape[1] and 0 <= candidate[1] < gray.shape[0]):
                        continue
                    score = objective(candidate)
                    if score < best - 1e-6:
                        center, best = candidate, score
            if center == previous:
                break
    return center


def _fill_profile(profile: np.ndarray) -> np.ndarray:
    finite = np.isfinite(profile)
    if not finite.any():
        raise ValueError("有效晶圆区域不足以估计圆环背景")
    return np.interp(np.arange(len(profile)), np.flatnonzero(finite), profile[finite]).astype(np.float32)


def _estimate_background(gray: np.ndarray, analysis_mask: np.ndarray, sampling_mask: np.ndarray,
                         center: tuple[float, float], radius_map: np.ndarray,
                         config: DetectorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_radius = float(radius_map[analysis_mask].max()) + 2
    radii = np.arange(0, max_radius + config.radial_step_px,
                      config.radial_step_px, dtype=np.float32)
    n_angles = config.angular_samples or 2 ** ceil(np.log2(max(512, 2 * np.pi * max_radius * 1.25)))
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False, dtype=np.float32)
    # 背景采样范围比实际检查范围稍大，给径向插值留出支撑。
    # 若直接用侵蚀后的检查掩膜采样，最外一圈可能因缺少插值邻点产生假残差。
    polar, valid = _polar_sample(gray, sampling_mask, center, radii, angles)

    # 第 1 步：同一半径的中位数给出任意形状的径向曲线。
    # 对完全无有效采样的行显式跳过，避免 nanmedian 产生空切片警告。
    profile = np.full(len(radii), np.nan, dtype=np.float32)
    for row in np.flatnonzero(valid.any(axis=1)):
        profile[row] = np.median(polar[row, valid[row]])
    profile = _fill_profile(profile)
    deviations = polar - profile[:, None]
    weights = valid.astype(np.float32)

    # 第 2 步：只沿圆周做高斯平滑，径向不使用会抹平窄环的大核。
    # 相同的原图弧长对应不同的角度宽度，因此每个半径使用不同带宽。
    # FFT 实现周期卷积，0°/360° 自然相连，不会在人为接缝处产生缺陷。
    frequencies = np.arange(n_angles // 2 + 1, dtype=np.float32)
    sigma_angle = config.angular_sigma_px / np.maximum(radii, 1.0)
    transfer = np.exp(-0.5 * (sigma_angle[:, None] * frequencies[None, :]) ** 2)
    denominator = np.fft.irfft(np.fft.rfft(weights, axis=1) * transfer, n=n_angles, axis=1)

    def circular_smooth(values: np.ndarray) -> np.ndarray:
        numerator = np.fft.irfft(np.fft.rfft(values * weights, axis=1) * transfer,
                                 n=n_angles, axis=1)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator),
                         where=denominator > 1e-6).astype(np.float32)

    correction = circular_smooth(deviations)
    for _ in range(config.background_iterations):
        # 第 3 步：将大残差限幅后重估背景，降低小点被吸收到背景的程度。
        # 这里限幅的是背景估计样本，实际检测始终保留原图灰度。
        error = deviations - correction
        sigmas = np.full(len(radii), config.noise_floor, dtype=np.float32)
        for row in np.flatnonzero(valid.any(axis=1)):
            sigmas[row] = _mad_sigma(error[row, valid[row]], config.noise_floor)
        limit = config.background_clip_sigma * sigmas[:, None]
        cleaned = correction + np.clip(error, -limit, limit)
        correction = circular_smooth(cleaned)
    model = (profile[:, None] + correction).astype(np.float32)

    # 第 4 步：背景仅映射回原始坐标一次；检测小点不在展开图上降采样。
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    angle_map = np.mod(np.arctan2(yy - center[1], xx - center[0]), 2 * np.pi)
    map_x = (angle_map * n_angles / (2 * np.pi)).astype(np.float32)
    map_y = np.clip(radius_map / config.radial_step_px, 0, len(radii) - 1).astype(np.float32)
    background = cv2.remap(model, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    background[~analysis_mask] = 0
    return background, radii, profile


def _estimate_noise(residual: np.ndarray, radius_map: np.ndarray,
                     mask: np.ndarray, config: DetectorConfig) -> np.ndarray:
    """从残差估计噪声，避免把原图正常环形梯度当成噪声。"""
    global_sigma = _mad_sigma(residual[mask], config.noise_floor)
    bin_ids = np.floor(radius_map / config.noise_bin_width_px).astype(np.int32)
    count = int(bin_ids[mask].max()) + 1
    sigmas = np.full(count, global_sigma, dtype=np.float32)
    for index in range(count):
        values = residual[mask & (bin_ids == index)]
        if values.size >= 24:
            sigmas[index] = _mad_sigma(values, config.noise_floor)
    centers = (np.arange(count) + 0.5) * config.noise_bin_width_px
    noise = np.interp(radius_map.ravel(), centers, sigmas).reshape(residual.shape).astype(np.float32)
    noise[~mask] = 0
    return noise


def _extract_defects(gray: np.ndarray, residual: np.ndarray, noise: np.ndarray, mask: np.ndarray,
                     config: DetectorConfig) -> tuple[list[Defect], np.ndarray, np.ndarray]:
    """多尺度产生种子，原图残差产生连通域，分别处理亮点与暗点。"""
    positive_seeds = np.zeros(mask.shape, dtype=bool)
    negative_seeds = np.zeros(mask.shape, dtype=bool)
    score_map = np.zeros(mask.shape, dtype=np.float32)
    weights = mask.astype(np.float32)
    variance = noise ** 2
    # 第二种背景只用于复核候选，不替代沿圆周的背景。
    # 中值滤波单独使用会破坏窄环，因此必须与圆环残差在同一区域相互印证。
    confirmation = (gray - cv2.medianBlur(gray, config.confirmation_ksize)
                    if config.confirmation_ksize else None)

    for size in config.scales:
        def box_sum(array: np.ndarray) -> np.ndarray:
            return cv2.boxFilter(array, -1, (size, size), normalize=False,
                                  borderType=cv2.BORDER_CONSTANT)

        count = box_sum(weights)
        complete = mask & (count >= size * size - 0.01)
        summed = box_sum(residual)
        sum_variance = box_sum(variance)
        # 这是独立噪声近似下的标准化响应，不是概率或严格显著性检验。
        response = summed / np.sqrt(np.maximum(sum_variance, config.noise_floor ** 2))
        mean = summed / np.maximum(count, 1)
        strong = complete & (np.abs(response) >= config.seed_score) & (
            np.abs(mean) >= config.min_mean_contrast)
        positive_seeds |= strong & (response > 0)
        negative_seeds |= strong & (response < 0)
        score_map = np.maximum(score_map, np.where(complete, np.abs(response), 0))

    labels = np.zeros(mask.shape, dtype=np.int32)
    defects: list[Defect] = []
    # 向外一像素是否离开分析范围，用来标识可能被截断的候选区域。
    interior = cv2.erode(weights.astype(np.uint8), np.ones((3, 3), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    threshold = np.maximum(config.grow_score * noise, config.min_pixel_contrast)
    for sign, polarity, seeds in ((1, "bright", positive_seeds), (-1, "dark", negative_seeds)):
        eligible = mask & (sign * residual >= threshold)
        count, components, stats, centroids = cv2.connectedComponentsWithStats(
            eligible.astype(np.uint8), connectivity=8)
        # 匹配滤波可能让种子响应扩散；仅保留实际像素残差连通域中包含种子的区域。
        seeded = set(np.unique(components[seeds & eligible]).tolist()) - {0}
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if component not in seeded or area < config.min_area:
                continue
            if config.max_area is not None and area > config.max_area:
                continue
            region = components == component
            values = sign * residual[region]
            peak = float(values.max())
            if peak < config.min_peak_contrast:
                continue
            confirmation_mean = confirmation_snr = confirmation_core = 0.0
            if confirmation is not None:
                # 使用整个连通域的有符号均值，正常相位错位的正负响应会抵消。
                # 只取最大像素或只看 bbox 内最大值，容易被细环、边框或邻近点误导。
                confirmation_mean = float(np.mean(sign * confirmation[region]))
                local_sigma = float(np.median(noise[region]))
                confirmation_snr = confirmation_mean / max(local_sigma, config.noise_floor)
                core = region & (sign * residual >= .5 * peak)
                confirmation_core = float(np.mean(sign * confirmation[core]))
                if (confirmation_mean < config.confirmation_min_mean
                        or confirmation_snr < config.confirmation_min_snr
                        or confirmation_core < config.confirmation_min_core):
                    continue
            defect_id = len(defects) + 1
            labels[region] = defect_id
            defects.append(Defect(
                id=defect_id, polarity=polarity,
                bbox_xywh=tuple(int(v) for v in stats[component, :4]),
                centroid_xy=tuple(float(v) for v in centroids[component]), area=area,
                peak_contrast=peak, mean_contrast=float(values.mean()),
                peak_score=float(score_map[region].max()),
                touches_analysis_boundary=bool(np.any(region & ~interior)),
                confirmation_mean=confirmation_mean, confirmation_snr=confirmation_snr,
                confirmation_core_mean=confirmation_core,
            ))
    if config.merge_distance_px > 0 and defects:
        # 同一个环状斑点可能同时有亮边和暗心。保留峰值对比度更强的候选，
        # 合并相距很近且 bbox 相交的重复响应，避免一处缺陷被重复计数。
        selected: list[Defect] = []
        for defect in sorted(defects, key=lambda d: (-d.peak_contrast, d.id)):
            x, y, width, height = defect.bbox_xywh
            duplicate = False
            for accepted in selected:
                ax, ay, aw, ah = accepted.bbox_xywh
                intersects = x < ax + aw and ax < x + width and y < ay + ah and ay < y + height
                distance = np.hypot(defect.centroid_xy[0] - accepted.centroid_xy[0],
                                     defect.centroid_xy[1] - accepted.centroid_xy[1])
                if intersects and distance <= config.merge_distance_px:
                    duplicate = True
                    break
            if not duplicate:
                selected.append(defect)
        merged_labels = np.zeros_like(labels)
        defects = []
        for new_id, defect in enumerate(selected, 1):
            merged_labels[labels == defect.id] = new_id
            defects.append(replace(defect, id=new_id))
        labels = merged_labels
    return defects, labels, score_map


def detect(image: np.ndarray, config: DetectorConfig | None = None, *,
           geometry: WaferGeometry | None = None,
           wafer_mask: np.ndarray | None = None) -> DetectionResult:
    """检测一张晶圆图，返回候选缺陷与所有关键中间结果。

    image: 0～255 灰度、BGR 或 BGRA，原图不被修改。
    config: 省略时使用真实样例配置；显式 DetectorConfig() 是通用灵敏配置。
    geometry: 可传入已知的晶圆外轮廓，跳过自动外圆估计。
    wafer_mask: 可选的有效晶圆区域，0 表示无效，非零表示有效。

    限制：单图背景模型不能可靠区分小型中心同心结构与正中心缺陷；
    真实配置排除中心半径 4 像素和外缘约 6 像素。请检查 analysis_mask。
    """
    config = config if config is not None else DetectorConfig.for_real_samples()
    gray = _grayscale(image)
    if max(config.scales) > min(gray.shape):
        raise ValueError("检测尺度不能大于图像尺寸")

    # 阶段 A：确定有效晶圆区域，外轮廓与明暗环中心分别处理。
    wafer_geometry, roi = _prepare_roi(gray, config, geometry, wafer_mask)
    ring_center = _refine_ring_center(gray, roi, wafer_geometry, config)
    yy, xx = np.indices(gray.shape, dtype=np.float32)
    radius_map = np.hypot(xx - ring_center[0], yy - ring_center[1])
    analysis_roi = roi.copy()
    if config.edge_margin_px:
        diameter = 2 * config.edge_margin_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        analysis_roi = cv2.erode(roi.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                                 borderValue=0).astype(bool)
    analysis_mask = analysis_roi & (radius_map >= config.center_exclusion_px)
    if np.count_nonzero(analysis_mask) < 64:
        raise ValueError("扣除中心和边缘后，有效检测区域不足 64 像素")

    # 阶段 B：估计任意径向曲线及缓慢角向变化，再从未平滑的原图相减。
    background, radial_samples, radial_profile = _estimate_background(
        gray, analysis_mask, roi, ring_center, radius_map, config)
    residual = np.where(analysis_mask, gray - background, 0).astype(np.float32)

    # 阶段 C：噪声标准化、多尺度点检测、双阈值连通域提取。
    noise_map = _estimate_noise(residual, radius_map, analysis_mask, config)
    defects, labels, score_map = _extract_defects(gray, residual, noise_map, analysis_mask, config)
    return DetectionResult(defects, labels, analysis_mask, background, residual,
                            noise_map, score_map, radial_samples, radial_profile,
                            wafer_geometry, ring_center, config)
