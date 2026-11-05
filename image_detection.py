import cv2
import numpy as np
from pathlib import Path
import glob

MAX_RADIUS = 180
def split_horizontal(img_path: str, p1, p2, out_dir: str = "./output", method='ostu'):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[SKIP] unreadable PNG: {image_file}")
        return None
    (x1, y1), (x2, y2) = p1, p2
    w = (x2 - x1) // 3 * 3            # 截到 3 的倍数（1088 -> 1086）
    roi = img[y1:y2, x1:x1 + w]       # 得到 1086 x 361

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(img_path).stem
    for i, part in enumerate(np.split(roi, 3, axis=1), 1):
        out = out_dir / f"{stem}_part{i}.png"
        cv2.imwrite(str(out), part)
        print(f"part{i}: {part.shape[1]}x{part.shape[0]} -> {out}")
        detect_anomalies(str(out), circle_method="hough", boundary_margin=8)

def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(Path(path).suffix, img)
    if ok:
        buf.tofile(str(path))

# ========== 第一步：外圆检测 ==========
def detect_outer_circle_contour(gray, max_radius=MAX_RADIUS):
    """Canny + 轮廓：同心圆取面积最大的外接圆（最外环）"""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    edges = cv2.morphologyEx(
        edges, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = None
    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        (cx, cy), R = cv2.minEnclosingCircle(c)
        print("****R******",R)
        if best is None or R > best[2]:
            best = (int(round(cx)), int(round(cy)), int(round(R)))
    return best

def detect_outer_circle_hough(gray):
    """HoughCircles 备选：同心圆会检到多个，取半径最大那个"""
    blur = cv2.GaussianBlur(gray, (3, 3), 1.5)
    h, w = gray.shape[:2]
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(h, w) // 2,
        param1=100, param2=40,
        minRadius=int(min(h, w) * 0.2), maxRadius=int(min(h, w) * 0.5),
    )
    if circles is None:
        return None
    c = max(circles[0], key=lambda c: c[2])
    return int(c[0]), int(c[1]), int(c[2])

def detect_outer_circle(gray, method="contour"):
    return detect_outer_circle_hough(gray)


def circle_mask(shape, cx, cy, r):
    m = np.zeros(shape[:2], np.uint8)
    cv2.circle(m, (cx, cy), r, 255, -1)
    return m

# ========== 第二步：异常检测（全程约束在圆内） ==========
def detect_anomalies_legacy(
    img_path: str,
    out_dir: str = "./out",
    circle_method: str = "contour",   # contour / hough
    boundary_margin: int = 1,          # 距外圆至少这么多像素才算内区
    median_ksize: int = 21,
    diff_thresh: int = 25,
    min_area: int = 4,
    max_area: int = 1000,
    fill_ratio: float = 0.15,
    morph_ksize: int = 3,
):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape[:2]

    # --- 第一步：找最大外圆 ---
    circle = detect_outer_circle(gray, method=circle_method)
    if circle is None:
        raise RuntimeError("未检测到圆，请换 circle_method='hough' 或调 Canny 参数")
    cx, cy, R = circle
    R = 177
    print(f"[圆检测] center=({cx},{cy})  R={R}")
    R_eff = R-3
    # # 有效内区 mask：后续所有异常都必须在它内部
    eff_mask = circle_mask((H, W), cx, cy, R_eff)

    # --- 第二步：圆内做背景估计 + 差分 ---
    bg = cv2.medianBlur(gray, median_ksize if median_ksize % 2 else median_ksize + 1)
    diff = cv2.absdiff(gray, bg)
    _, mask = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    # 用 R（原圆，稍大）先粗遮，让贴边的异常能被完整"看见"，
    # 再由下一步的 R_eff 严格卡掉贴边结果。
    mask = cv2.bitwise_and(mask, circle_mask((H, W), cx, cy, R))

    if morph_ksize > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    # --- 第三步：连通域 + 三重边界校验 ---
    num, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 5)
    anomalies = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT]); y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH]); h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if w * h == 0 or area / (w * h) < fill_ratio:
            continue

        # ① bbox 在图像内
        if x < 0 or y < 0 or x + w > W or y + h > H:
            continue
        # ② bbox 内每个像素都在有效圆内（核心：杜绝贴边/出界）
        if np.any(eff_mask[y:y + h, x:x + w] == 0):
            continue
        # ③ 中心点到圆心距离也限制一次（双保险）
        ccx, ccy = int(cents[i][0]), int(cents[i][1])
        dist = float(np.hypot(ccx - cx, ccy - cy))
        if dist > R_eff:
            continue

        delta = int(gray[ccy, ccx]) - int(bg[ccy, ccx])
        kind = "白圈(亮异常)" if delta > 0 else "黑点(暗异常)"
        anomalies.append({"kind": kind, "bbox": (x, y, w, h),
                          "area": area, "center": (ccx, ccy),
                          "delta": delta, "dist_to_center": dist})
    # --- 第四步：可视化 ---
    vis = img.copy()
    cv2.circle(vis, (cx, cy), R,     (0, 200, 255), 2)                # 外圆(黄)
    for a in anomalies:
        x, y, w, h = a["bbox"]
        color = (0, 255, 0) if a["kind"].startswith("白") else (0, 0, 255)
        cv2.rectangle(vis, (x - 3, y - 3), (x + w + 3, y + h + 3), color, 2)
        cv2.putText(vis, a["kind"], (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stem = Path(img_path).stem
    imwrite_unicode(out / f"{stem}_anomaly.png", vis)
    #imwrite_unicode(out / f"{stem}_mask.png",    mask)
    # imwrite_unicode(out / f"{stem}_diff.png",    diff)
    # imwrite_unicode(out / f"{stem}_effmask.png", eff_mask)
    print(f"共检测到 {len(anomalies)} 个异常（全部完整位于有效圆内）：")
    for a in anomalies:
        print(f"  {a['kind']}  bbox={a['bbox']}  area={a['area']}  "
              f"Δ={a['delta']}  r={a['dist_to_center']:.1f}")
    return anomalies


def detect_anomalies(
    img_path: str,
    out_dir: str = "./out",
    circle_method: str = "auto",
    boundary_margin: int | None = None,
    *,
    config=None,
    diff_thresh: float | None = None,
    min_area: int | None = None,
    max_area: int | None = None,
    save_diagnostics: bool = True,
):
    """新版单张小图检测。split_horizontal 的切割逻辑保持原样。

    1. 自动稳健外圆定位，排除文字和截图直边的干扰；不再硬编码 R=177。
    2. 圆周背景残差与局部中值残差交叉复核，降低窄环处误报。
    3. 亮边/暗心的近邻重复响应合并为一个候选。

    circle_method 保留用于兼容原 split_horizontal 的调用，新版统一采用稳健定位。
    boundary_margin 未提供时使用真实图配置的 6 像素；显式提供时会生效。
    diff_thresh 在新版表示候选峰值对比度下限。复杂参数请传 DetectorConfig。
    原 median_ksize/fill_ratio/morph_ksize 等旧参数可调用 detect_anomalies_legacy。
    返回值为候选字典列表，坐标全部对应输入的小图。
    """
    from dataclasses import replace
    from wafer_defects import DetectorConfig, detect
    from wafer_defects.io import read_image, render_overlay, save_result, write_png

    if circle_method not in {"auto", "contour", "hough"}:
        raise ValueError("circle_method 必须为 auto、contour 或 hough")
    settings = config if config is not None else DetectorConfig.for_real_samples()
    overrides = {}
    if boundary_margin is not None:
        overrides["edge_margin_px"] = boundary_margin
    if diff_thresh is not None:
        overrides["min_peak_contrast"] = diff_thresh
    if min_area is not None:
        overrides["min_area"] = min_area
    if max_area is not None:
        overrides["max_area"] = max_area
    if overrides:
        settings = replace(settings, **overrides)

    image = read_image(img_path)
    result = detect(image, settings)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(img_path).stem
    # 保留原 demo 的文件命名；额外诊断图放入独立子目录，便于逐图复核。
    write_png(out / f"{stem}_anomaly.png", render_overlay(image, result))
    if save_diagnostics:
        save_result(image, result, out / stem)

    anomalies = []
    cx, cy = result.ring_center_xy
    for defect in result.defects:
        x, y = defect.centroid_xy
        sign = 1 if defect.polarity == "bright" else -1
        anomalies.append({
            "kind": "亮异常" if sign > 0 else "暗异常",
            "bbox": defect.bbox_xywh, "area": defect.area,
            "center": defect.centroid_xy, "delta": sign * defect.peak_contrast,
            "dist_to_center": float(np.hypot(x - cx, y - cy)),
            "confirmation_mean": defect.confirmation_mean,
            "confirmation_snr": defect.confirmation_snr,
        })
    print(f"[新版检测] {Path(img_path).name}：{len(anomalies)} 个候选；"
          f"外圆=({result.wafer_geometry.center_x:.2f}, {result.wafer_geometry.center_y:.2f}, "
          f"{result.wafer_geometry.radius:.2f})")
    return anomalies

if __name__ == "__main__":
    src_folder = r'/home/didev01/Production/ADC/demo/LV'
    png_files = glob.glob(f'{src_folder}/**/*.png', recursive=True)
    for image_file in png_files:
        print(image_file)
        split_horizontal(image_file, (3, 730), (1086, 1086))


