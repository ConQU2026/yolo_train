# KFS Detect (ConQU Vision Group)

一个基于 Ultralytics YOLO 的目标检测训练与推理项目，使用 uv 管理 Python 环境与依赖，源代码位于 `src/`，配置位于 `config/`。

## 特性

- 使用 uv 管理环境与依赖，支持 GPU(CUDA 12.8) 与 CPU 安装路径
- 按比例拆分训练/验证集（符号链接），不破坏原始数据集
- 标签格式化工具，将原始标签转换为 YOLO 所需格式
- 数据增强示例：缩放、旋转、仿射、随机模糊，并支持 YOLO 标注随变换同步与可视化校验
- 一键训练与多种推理方式（图片/视频/摄像头）

---

## 目录结构（节选）

```
config/
	classes.txt         # 类别列表（每行一个）
	config.yaml         # 数据增强与通用生成参数
	data.yaml           # YOLO 数据集配置（path/train/val/names）
dataset/
	images/train        # 原始训练图像
	labels/train        # 原始训练标签（YOLO txt）
dataset_seperated/    # 由脚本生成：按比例划分的 train/valid（符号链接）
models/
	yolov8s.pt          # 训练起始权重（默认）
result/
	best.pt             # 训练得到的最佳模型（导出在此）
src/
	main.py             # 训练入口
	image_processor.py  # 数据增强 + YOLO 坐标同步变换与可视化校验
	parse_config.py     # YAML 配置解析工具
	seperate_valid.py   # 拆分 train/valid（符号链接）并校验一致性
	yolo_formatter.py   # 将原标签转为 YOLO 格式并构建 dataset_yolo
	copy_dir.py         # 目录复制/符号链接工具类
test/
	test_detect.py      # 推理脚本（图片/视频/摄像头）
```

---

## 环境准备（uv）

项目已在 `pyproject.toml` 里配置了 PyTorch CUDA 12.8 的索引（可选依赖 `cu128`）。建议使用 uv 创建隔离环境并安装依赖。

1. 安装 uv（二选一）：

```powershell
# 推荐：使用 pipx
pipx install uv

# 或使用 pip（若无 pipx）
python -m pip install -U uv
```

2. 创建虚拟环境（可选）：

```powershell
uv venv
# 可激活（也可不激活，使用 `uv run` 运行亦可）
.\.venv\Scripts\Activate.ps1
```

3. 安装依赖（按需选择 GPU 或 CPU）：

```powershell
# 安装项目定义的可选依赖（CUDA 12.8 的 torch/torchvision）
uv pip install -e .[cu128]

# 安装运行所需包（不修改 pyproject）
uv pip install ultralytics opencv-python pyyaml

# 若使用 CPU（无 NVIDIA GPU），可改为：
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# uv pip install ultralytics opencv-python pyyaml
```

提示：如果你倾向于把依赖写回 `pyproject.toml`，也可以使用 `uv add ultralytics opencv-python pyyaml`。

---

## 数据准备

1. 原始数据放在 `dataset/`：

- 图像：`dataset/images/train/*.jpg|png`
- 标签：`dataset/labels/train/*.txt`（YOLO txt：`class x y w h`，归一化坐标）

2. 将数据按比例拆分为 train/valid（创建符号链接，不复制大文件）：

```powershell
uv run python -m src.seperate_valid
```

生成的目录类似：

```
dataset_seperated/
	images/{train,valid}
	labels/{train,valid}
```

Windows 提示：创建符号链接可能需要管理员权限。若报错，请以管理员身份运行终端，或改为彻底复制（参考 `copy_dir.py`）。

3. 确认 `config/data.yaml` 中的 `path` 指向新的 `dataset_seperated` 目录；`names` 与 `config/classes.txt` 一致。

---

## 训练

默认使用 `models/yolov8s.pt` 作为预训练权重，数据配置来自 `config/data.yaml`，训练产物位于 `runs/` 与 `result/best.pt`。

```powershell
uv run python -m src.main
```

你可以在 `src/main.py` 中调整参数（如 `epochs`, `imgsz`, `batch`, `device` 等）。

---

## 推理（图片/视频/摄像头）

`result/best.pt` 作为默认推理权重：

```powershell
# 图片示例（默认读取 test/image/test3.jpg）
uv run python -m test.test_detect

# 或在代码中将 `detect_with_image(img_path=...)` 指向你的图片路径
```

运行后将在 `test/output/detected_image.jpg` 生成带框结果；也可在 `test_detect.py` 中启用 `detect_with_camera()` 或 `detect_with_video()`。

---

## 模块说明（src/）

- `main.py`：训练入口。加载 `models/yolov8s.pt` 与 `config/data.yaml`，调用 Ultralytics YOLO 的 `model.train(...)`。
- `image_processor.py`：
  - 随机缩放、旋转、仿射、模糊生成增强图像；
  - 提供 YOLO 标注随仿射矩阵的坐标同步变换；
  - `valid_result(...)` 可视化原/增广后图像与框对比，便于检查标注正确性。
- `parse_config.py`：通用 YAML 解析工具，读取 `config/config.yaml`。
- `seperate_valid.py`：按比例划分 `train/valid`，通过符号链接构建 `dataset_seperated/`；
  - 自检 image/label 一致性，输出详细日志；
  - Windows 下若无管理员权限，符号链接可能失败（提供提示）。
- `yolo_formatter.py`：
  - 读取 `config/classes.txt` 与标签文件名中的类别前缀，生成 YOLO 标签行；
  - 支持将原始 `dataset/` 复制/链接为 `dataset_yolo/` 并批量格式化标签。
- `copy_dir.py`：封装目录完整复制与“以符号链接代替复制”的两种模式，节省磁盘占用。

## 配置说明（config/）

- `config.yaml`
  - `IMG_SIZE`：输出画布尺寸（宽,高）
  - `ROTATION_ANGLE_RANGE` / `AFFINE_DISTORTION_RANGE` / `SCALE_MIN`：几何增强参数
  - `BLUR_PROBABILITY` / `KERNEL_RANGE`：随机模糊概率与核大小范围
- `data.yaml`
  - `path`：数据集根目录（通常指向 `dataset_seperated`）
  - `train` / `val`：相对于 `path` 的子路径
  - `names`：类别名列表，顺序即类别索引
- `classes.txt`
  - 每行一个类别名，需与 `data.yaml` 中 `names` 对应。

---

## 常见问题（FAQ）

- 符号链接报错（Windows）：请以管理员身份运行 PowerShell，或改用完整复制（`copy_dir.py`）。
- CUDA 版本不匹配：本项目的 `pyproject.toml` 已配置 CUDA 12.8 索引；若你的驱动/环境不匹配，请改用 CPU 安装命令或调整索引。
- OpenCV 窗口不显示：远程/无桌面环境下可去掉 `imshow` 并仅保存图片到磁盘。
- data.yaml 找不到数据：确认 `path` 指向 `dataset_seperated` 的绝对路径，或改为相对路径并保证工作目录正确。

---

## 日志与结果

2025-11-03：原始数据集未做透视泛化，透视图片识别效果较差，下一版考虑增加透视增强。

示例推理结果：

![result1](./test/output/detected_image.jpg)

# 数据集地址(local)

## http://pan.conqu.tech/f/8732
