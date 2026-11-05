# GaN 晶圆近同心环图像缺陷检测 · 真实样例修订版

本项目在单张图内估计正常圆环背景，结合局部中值背景复核约 3～7 像素范围的亮斑、暗斑及细环缺口。环数、环宽和间距可以随图像变化，不使用固定周期模板，不需要训练神经网络。代码各阶段有中文注释。

详细参数说明和内网调参方法见 [参数说明与内网调参指南](docs/parameter-guide.zh-CN.md)，包含全部 25 项参数的单位、调整方向、相互约束、常见场景及分步配置示例。

所有图像处理和结果导出均在本地执行，不含上传、遥测、远程 API 或在线模型调用。首次安装依赖需要获取软件包；厂内可使用事先准备好的离线安装包。

默认配置已根据 `data` 中 54 张真实脱敏小图调整，实际尺寸为宽 361、高 356。所有图片使用同一组配置，算法不读取文件名、类别文件夹或预设缺陷坐标。

## 本批样例的核对结果

| 方法 | 正常类中报出候选的图片 | 缺陷类中报出候选的图片 |
| --- | --- | --- |
| 用户原始 `detect_anomalies` 默认参数 | 17 / 40 | 12 / 14 |
| 修订前的 `wafer_defects` 默认参数 | 38 / 40 | 14 / 14 |
| 本版统一真实图配置 | **0 / 40** | **14 / 14** |

当前共保留 32 个候选。`image34` 左上、右上两处黑环缺口均有对应候选；下方小黑圈的业务定义未确认，没有把它设为必须检出的标注。原始 demo 对照仅用完全相同的图片副本绕过 Windows 中文路径读取问题，没有修改旧检测算法。

**这是使用本批图片调参后的图像级回归结果，不是独立测试集表现，也不能证明每个真实缺陷点均无漏检。** 目前没有完整逐点标注；候选框还需要业务复核。真实回归测试会在缺少 `data` 时明确跳过，不会伪造结果。

逐图原图/检测结果、JSON、CSV 在运行后的 `output/real_samples/index.html`；原始 demo 对照记录在本次工作生成的 `output/real_samples/legacy_comparison/summary.json`。

## 安装与快速运行

需要 Python 3.10 或更新版本，依赖 NumPy、OpenCV。测试使用标准库 `unittest`，无需额外测试框架。

在项目目录的 PowerShell 中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .

# 生成四种 361×365 模拟图，并执行检测。
.\.venv\Scripts\python.exe -m wafer_defects demo --output output/demo

# 执行全部单元测试及命令行集成测试，警告也视为错误。
.\.venv\Scripts\python.exe -W error -m unittest discover -s tests -v

# 递归检测这批真实小图；类别只用于检测完成后的统计。
.\.venv\Scripts\python.exe -m wafer_defects batch data --evaluate-labels --output output/real_samples

# 检测自己的图片。中文文件路径同样支持。
.\.venv\Scripts\python.exe -m wafer_defects detect "D:\厂内数据\晶圆.png" --output output/result

# 指定参数；只需在 JSON 中写需要覆盖的参数。
.\.venv\Scripts\python.exe -m wafer_defects detect "D:\厂内数据\晶圆.png" --config config.real.json --output output/result

# 无类别标签的生产小图文件夹；不要求按“明显缺陷/正常”预先分组。
.\.venv\Scripts\python.exe -m wafer_defects batch "D:\厂内数据\切好的小图" --output output/production

# 如果知道晶圆的外圆，可手动指定 x、y、半径。
.\.venv\Scripts\python.exe -m wafer_defects detect "D:\厂内数据\晶圆.png" --geometry 180 182 175.5 --output output/result
```

安装后也可以用 `wafer-detect` 代替 `python -m wafer_defects`。

图像尺寸不强制固定；真实样例为 361×356，模拟样例仍为 361×365。输入支持灰度、BGR、BGRA，灰度范围为 0～255。RGB 数组请先转 BGR。16 位图请按**固定的传感器量程**转换，避免每张图独立拉伸改变阈值含义。不同照片不要先缩放到更低分辨率。

单图与批处理默认使用真实图配置 `real`；模拟演示默认使用保留弱点的 `sensitive` 配置。可通过 `--preset real` / `--preset sensitive` 显式选择，JSON 参数覆盖所选配置。`config.real.json` 与 `config.example.json` 均给出真实图配置。

原 `image_detection.py` 的 `split_horizontal` 切割实现未修改。其调用的新版 `detect_anomalies` 已接入当前算法，仍输出 `<文件名>_anomaly.png`，并返回候选字典列表。旧方法保留为 `detect_anomalies_legacy`；原 `__main__` 的 Linux 路径也未擅自替换，要处理现有小图文件夹可直接使用上面的 `batch` 命令。

```python
from image_detection import detect_anomalies

points = detect_anomalies("data/明显缺陷/image34.png", out_dir="output/single")
```

`boundary_margin` 现在会实际生效；新版函数不指定时为 6 像素。原切图函数显式传入的是 8 像素，其检测范围会更保守，与本批统一 6 像素配置的统计范围不同。

## 算法与代码对应

主要实现位于 `wafer_defects/detector.py`，公开入口为 `detect`。

1. **确定晶圆区域。** 利用黑色边角估计背景，沿射线找最外侧可见边界，并稳健拟合外圆。忽略截图的直边和被截断的方向，防止彩色竖线、文字把圆心拉偏十余像素。没有足够边界时回退到轮廓拟合；也可手动传入外圆或有效掩膜。没有固定半径 177。
2. **微调明暗环中心。** 以外轮廓圆心为起点，做粗到细的亚像素搜索，降低圆周灰度离散度。外轮廓与明暗环中心分别保存，中心微调不会移动晶圆掩膜。
3. **估计径向曲线。** 按 0.5 像素间隔取样，每个半径上的中位数形成独立径向曲线。允许单个外环加中央圆形、多层环、不等宽环及宽平台。
4. **估计缓慢角向变化。** 在极坐标中仅沿圆周平滑，按原图弧长控制尺度。高斯平滑使用周期卷积，0°/360° 接缝自然连接。通过迭代 MAD 限幅降低稀疏缺陷被拟合进背景的程度；实际原图不被限幅或去噪。这样也能适应轻微椭圆及照明渐变。
5. **回到原图计算残差。** 背景映射回原始像素，计算 `原图 − 背景`。背景采样范围保留插值所需的邻点，避免在检测边界制造假亮点。
6. **估计局部噪声。** 在残差上按半径分组估计 MAD，并设置噪声下限。正常圆环的原始明暗梯度不直接参与噪声估计。
7. **多尺度产生候选种子。** 对有符号残差做 1、3、5、7 像素窗口统计，分别寻找亮异常、暗异常。窗口完整位于检查范围内才有效。响应使用独立噪声近似标准化，**分数不是缺陷概率，也不是严格的统计显著性**。
8. **双阈值连通域提取。** 高分响应提供种子，较低的像素残差阈值确定实际区域，再按面积、峰值对比度筛选。候选区域来自原图残差；未强行要求恰好 25 像素。亮、暗点分开提取。
9. **第二种背景交叉验证。** 对未平滑的原图做 5×5 中值背景，计算第二种有符号残差。在圆环候选的真实连通域上检查其均值、相对于局部噪声的比值，以及高响应核心区域的均值。必须在同一位置、同一极性获得支持；不使用 bbox 里的最大值，以免被邻近环或亮边误导。中值滤波没有单独承担检测任务，避免把窄环整体当成异常。
10. **合并重复响应。** 对 bbox 相交且中心相距不超过 4 像素的响应，保留峰值对比度更强的一个。用于减少同一小点亮边、暗心被重复计数；需要分开非常近的独立缺陷时可将该参数减小或置 0。

## 在 Python 中调用

```python
from dataclasses import replace
from wafer_defects import DetectorConfig, WaferGeometry, detect
from wafer_defects.io import read_image, save_result

image = read_image("wafer.png")
config = DetectorConfig.for_real_samples()
# 修改真实配置中的个别参数，其余交叉验证步骤保留。
# config = replace(config, min_peak_contrast=26.0)
result = detect(image, config)

# 可选：手动指定晶圆外轮廓；实际检查范围还会扣除中心和边缘。
# result = detect(image, config, geometry=WaferGeometry(180, 182, 175.5))

for defect in result.defects:
    print(defect.polarity, defect.centroid_xy, defect.bbox_xywh,
          defect.area, defect.peak_contrast)

save_result(image, result, "output/result")
```

`detect` 不修改输入。坐标统一为 `x=列、y=行`，原点位于左上角；数组索引仍为 `[y, x]`。`bbox_xywh` 为左上角和宽高。

`detect(image)` 无配置调用默认选用 `for_real_samples()`。为了兼容原模拟试验，显式传入 `DetectorConfig()` 仍是第一版的灵敏参数组合，局部中值复核默认关闭。生产使用建议从 `DetectorConfig.for_real_samples()` 再用 `replace` 修改。

还可传入与图像同尺寸的 `wafer_mask`：非零为有效，内部的无效洞会被保留。命令行对应 `--mask mask.png`。

## 导出结果

| 文件 | 内容 |
| --- | --- |
| `input.png` | 转换后的灰度输入 |
| `background.png` | 估计的正常背景；检查范围外为黑 |
| `residual.png` | 有符号残差显示图；128 表示零，正负差异放大 3 倍，范围外为黑 |
| `score.png` | 多尺度响应显示图，数值放大 16 倍后截断显示 |
| `confirmation.png` | 真实配置下局部中值背景的有符号残差，显示增益与 `residual.png` 相同 |
| `analysis_mask.png` | **白色才是实际检查区域，黑色表示未检查** |
| `defect_mask.png` | 检出的异常区域，白色表示候选 |
| `overlay.png` | 候选框与编号；亮异常橙色、暗异常蓝色，未检查区域变暗 |
| `report.json` | 候选坐标、面积、极性、对比度、复核响应、实际配置及几何信息 |
| `intermediates.npz` | 精确浮点背景、残差、噪声、响应、区域标签和径向曲线 |

PNG 为显示图，存在取整或截断；定量排查请读取 NPZ。重复输出到同一目录会更新已有结果，命令行会拒绝覆盖本次输入或掩膜文件。

模拟演示额外输出正常图、注入标签和 `truth.json`。标签为注入真值，不能当作检测结果。`output/demo/overview.png` 为四种环形场景的检测总览。

## 参数调整顺序

| 参数 | 默认值 | 调整用途 |
| --- | --- | --- |
| `angular_sigma_px` | 4 | 圆周高斯背景标准差；真实窄环需要更小尺度以跟随正常角向变化，稳健限幅降低缺陷被吸收的程度 |
| `radial_step_px` | 0.5 | 径向采样间隔；保留窄环和交界，当前限制不大于 1 |
| `center_search_radius_px` | 3 | 环心相对外圆心的最大搜索偏移，单位像素 |
| `noise_floor` | 1 | 噪声标准差下限，防止低噪声区域分数被过度放大 |
| `seed_score` | 18 | 越低越容易产生候选，也会增加相关纹理的误报；这是经验响应，不是概率 |
| `grow_score` | 2.5 | 越低，连通域更容易覆盖弱边缘，也更容易粘连噪声 |
| `min_mean_contrast` | 2.5 | 模板窗口内平均残差的最低绝对值 |
| `min_peak_contrast` | 28 | 候选峰值残差的最低绝对值；更低对比度的新缺陷需要重新标注和校准 |
| `min_area` | 4 | 超阈值连通域最小面积，单位像素数 |
| `max_area` | 200 | 小点面积上限；需要保留大异常可设为 `null` / `None` |
| `center_exclusion_px` | 4 | 中心附近不检查的半径 |
| `edge_margin_px` | 6 | 晶圆有效区域向内收缩的距离，截图边界同样留出支撑 |
| `confirmation_ksize` | 5 | 局部中值背景窗口；0 关闭复核，另支持 3 |
| `confirmation_min_mean` | 3 | 候选连通域上同极性的中值残差均值下限 |
| `confirmation_min_snr` | 1.3 | 上述均值与局部残差噪声估计的比值下限，不是概率 |
| `confirmation_min_core` | 5 | 候选高响应核心区域的中值残差均值下限 |
| `merge_distance_px` | 4 | bbox 相交候选的近邻合并距离；0 关闭合并 |

建议先检查背景图和残差图。如果残差仍有连续整环或长圆弧，先检查环心、角向背景尺度和形变，不要直接通过升高阈值压掉它。背景合理后，再用厂内确认正常图校准每张晶圆的误报数量，用已标注缺陷检查召回情况，特别单独统计交界处的弱点。

## 测试范围及限制

目前 36 项测试通过。原来的 25 项模拟与接口测试显式使用灵敏配置，新增 9 项真实样例回归测试，覆盖正常类误报、正类整图漏检、image34 两个上方缺口、截图直边导致的圆心偏移、亮边暗心去重、新旧入口兼容、旋转及小幅亮度变化。另外增加 2 项批处理测试，检查类别标签不会影响推理，以及输出目录和报告链接是否正确。

模拟测试覆盖不同环数、单外环加内部圆形、不等宽圆环、轻微椭圆、照明变化、亮暗两种极性、交界处弱点、3/5/7 像素点、极坐标接缝、环心偏移、额外噪声种子、手动几何和掩膜、输入不被修改、中文路径、命令行和结果导出。

这些是有限的可复现测试用例，不代表独立生产批次的检出率或误报率。数学模拟器与真实成像过程存在差异。

- **真实配置默认排除中心半径 4 像素、晶圆外缘约 6 像素。** 正中心局部结构在同半径上缺少独立参考，可能与正常中央圆形无法区分；本版未声称覆盖该区域。外缘的完整模板还需要额外邻域支撑。请以实际 `analysis_mask` 为准。
- **近同心、局部异常稀疏是主要假设。** 明显扭曲、局部剧烈环宽变化、整圈异常、很大的异常区域可能被背景吸收或产生假残差，需要进一步建模。
- **外轮廓必须可辨。** 若最外一段晶圆和背景都是完全相同的黑色，仅凭图片无法恢复隐藏边界，应提供可信外圆或掩膜。
- **自动圆心是启发式估计。** 射线搜索以近似居中、近内接圆为起始假设；严重偏心或裁切时应手动指定外圆。中心结构很平、环很少或照明变化较强时，环心微调也未必精确；允许关闭 `refine_center`。
- **单张图不能证明异常来自材料。** 灰尘、坏点、压缩伪影等也可能成为候选，后续需结合厂内标准复核。
- **灰度动态范围必须一致。** 所有绝对阈值按 0～255 量程设置；输出分数也需要在正常生产图上校准。

参考运行环境：Python 3.12、NumPy 2.3.5、OpenCV 4.14.0。项目本身不依赖厂区数据或保密资料。
