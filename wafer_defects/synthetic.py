"""可重复的数学模拟，用于测试算法行为；不代表 GaN 的真实成像物理过程。"""

from dataclasses import dataclass

import numpy as np

from .detector import WaferGeometry


# 每个节点为“相对半径、灰度”。节点间平滑过渡，不要求周期或等间距。
PROFILES: dict[str, tuple[tuple[float, float], ...]] = {
    "few_rings": ((0, 185), (.18, 45), (.39, 200), (.57, 60), (.81, 190), (1, 55)),
    "many_rings": ((0, 180), (.12, 40), (.21, 200), (.34, 50), (.42, 190),
                   (.61, 45), (.72, 205), (.88, 60), (1, 160)),
    "single_ring": ((0, 100), (.72, 100), (.87, 205), (1, 40)),
    "uneven_widths": ((0, 180), (.12, 180), (.16, 35), (.22, 35), (.29, 195),
                      (.64, 195), (.68, 45), (.84, 45), (.91, 190), (1, 70)),
}


@dataclass(frozen=True)
class SyntheticSpot:
    """支撑范围为 size×size 的局部灰度变化；amplitude 正值为亮点。"""

    x: int
    y: int
    amplitude: float
    size: int = 5


@dataclass
class SyntheticWafer:
    image: np.ndarray
    normal_image: np.ndarray
    true_background: np.ndarray
    truth_labels: np.ndarray
    geometry: WaferGeometry
    ring_center_xy: tuple[float, float]
    spots: tuple[SyntheticSpot, ...]


def _radial_profile(relative_radius: np.ndarray, knots: tuple[tuple[float, float], ...]) -> np.ndarray:
    xs, ys = np.asarray(knots, dtype=np.float64).T
    # 每两个节点之间采用 smoothstep，支持宽平台、窄环以及不等宽过渡。
    index = np.clip(np.searchsorted(xs, relative_radius, side="right") - 1, 0, len(xs) - 2)
    fraction = np.clip((relative_radius - xs[index]) / (xs[index + 1] - xs[index]), 0, 1)
    smooth = fraction ** 2 * (3 - 2 * fraction)
    return ys[index] + smooth * (ys[index + 1] - ys[index])


def make_wafer(*, profile: str = "uneven_widths", width: int = 361, height: int = 365,
               noise_std: float = 1.5, ellipticity: float = 0.004,
               illumination_xy: tuple[float, float] = (6.0, -3.0),
               ring_center_offset_xy: tuple[float, float] = (0.6, -0.4),
               seed: int = 42, spots: tuple[SyntheticSpot, ...] | None = None) -> SyntheticWafer:
    """创建正常图、含异常图及真值。spots=() 生成无缺陷负样本。

    spots=None 自动在暗处、亮处、交界处各注入一点。真值来自注入记录，
    不使用检测算法的任何中间结果。
    """
    if profile not in PROFILES:
        raise ValueError(f"未知 profile，可选值为 {tuple(PROFILES)}")
    if min(width, height) < 64 or not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("模拟尺寸至少为 64，噪声标准差必须为有限非负数")
    if not np.isfinite(ellipticity) or abs(ellipticity) >= .1:
        raise ValueError("ellipticity 必须有限且绝对值小于 0.1")
    if (len(illumination_xy) != 2 or len(ring_center_offset_xy) != 2
            or not np.isfinite((*illumination_xy, *ring_center_offset_xy)).all()):
        raise ValueError("照明和圆心偏移必须是两个有限数值")

    geometry = WaferGeometry((width - 1) / 2, (height - 1) / 2, min(width, height) / 2 - 5)
    center = (geometry.center_x + ring_center_offset_xy[0],
              geometry.center_y + ring_center_offset_xy[1])
    yy, xx = np.indices((height, width), dtype=np.float64)
    dx, dy = xx - center[0], yy - center[1]
    radius = np.sqrt((dx / (1 + ellipticity)) ** 2 + (dy * (1 + ellipticity)) ** 2)
    wafer_mask = np.hypot(xx - geometry.center_x, yy - geometry.center_y) <= geometry.radius
    background = _radial_profile(radius / geometry.radius, PROFILES[profile])
    background += (illumination_xy[0] * dx + illumination_xy[1] * dy) / geometry.radius
    background = np.where(wafer_mask, background, 0).astype(np.float32)
    rng = np.random.default_rng(seed)
    noisy = np.where(wafer_mask, background + rng.normal(0, noise_std, background.shape), 0)
    normal = np.clip(np.rint(noisy), 0, 255).astype(np.uint8)

    if spots is None:
        # 直接查询已知模拟背景，选择三种位置；与检测器实现无关。
        def choose(angle: float, kind: str) -> tuple[int, int]:
            distances = np.arange(.3 * geometry.radius, .9 * geometry.radius, .25)
            xs = np.rint(center[0] + distances * np.cos(angle)).astype(int)
            ys = np.rint(center[1] + distances * np.sin(angle)).astype(int)
            values = background[ys, xs]
            if kind == "dark":
                index = np.argmin(values)
            elif kind == "bright":
                index = np.argmax(values)
            else:
                # 去除四舍五入到重复像素带来的导数歧义，用真实连续径向曲线选位置。
                levels = _radial_profile(distances / geometry.radius, PROFILES[profile])
                index = np.argmax(np.abs(np.gradient(levels)))
            return int(xs[index]), int(ys[index])

        bright_xy = choose(-2.3, "dark")
        dark_xy = choose(.4, "bright")
        transition_xy = choose(1.9, "transition")
        spots = (SyntheticSpot(*bright_xy, 55), SyntheticSpot(*dark_xy, -55),
                 SyntheticSpot(*transition_xy, 24))

    image_f = normal.astype(np.float32)
    labels = np.zeros((height, width), dtype=np.uint16)
    for index, spot in enumerate(spots, 1):
        if (not isinstance(spot.size, int) or isinstance(spot.size, bool)
                or spot.size < 1 or spot.size % 2 == 0):
            raise ValueError("模拟缺陷尺寸必须为正奇数")
        if (not isinstance(spot.x, int) or not isinstance(spot.y, int)
                or not np.isfinite(spot.amplitude) or spot.amplitude == 0):
            raise ValueError("缺陷坐标必须为整数，幅度必须是有限非零值")
        half = spot.size // 2
        left, top = spot.x - half, spot.y - half
        right, bottom = left + spot.size, top + spot.size
        if left < 0 or top < 0 or right > width or bottom > height:
            raise ValueError("缺陷支撑范围超出图像")
        if not wafer_mask[top:bottom, left:right].all():
            raise ValueError("缺陷支撑范围超出晶圆")
        if labels[top:bottom, left:right].any():
            raise ValueError("模拟缺陷不能重叠")
        offsets = np.arange(-half, half + 1)
        ox, oy = np.meshgrid(offsets, offsets)
        sigma = max(0.6, spot.size * .27)
        bump = spot.amplitude * np.exp(-(ox ** 2 + oy ** 2) / (2 * sigma ** 2))
        image_f[top:bottom, left:right] += bump
        labels[top:bottom, left:right] = index
    image = np.clip(np.rint(image_f), 0, 255).astype(np.uint8)
    return SyntheticWafer(image, normal, background, labels, geometry, center, tuple(spots))
