# YOLOv8-OBB 工业缺陷感知系统

基于 YOLOv8-OBB (Oriented Bounding Box) 的工业级旋转目标检测系统，用于绝缘子缺陷检测。

---

## 项目简介

本项目基于 YOLOv8-OBB 实现绝缘子缺陷检测，支持 4 类缺陷识别：

| 类别 ID | 类别名 | 描述 |
|---------|--------|------|
| 0 | healthy_asset | 健康绝缘子 |
| 1 | defective_string | 缺陷串 |
| 2 | broken_disc | 破损盘 |
| 3 | flashover_disc | 闪络盘 |

项目特色：
- 自动化 **HBB → OBB** 数据标注转换（EPRI JSON → YOLO-OBB 格式）
- 基于 EPRI 电力数据集的多分类级联状态对齐
- 支持训练、评估、批量推理完整流水线

---

## 环境配置

### 1. 必备环境

- **Python**: 3.11
- **CUDA**: 11.8 / 12.1（需要 NVIDIA 显卡，训练必须）
- **包管理器**: conda（推荐）或 pip

### 2. 创建虚拟环境（推荐使用 conda）

```bash
# 创建 conda 环境
conda create -n yolo-obb python=3.11
conda activate yolo-obb
```

### 3. 安装依赖

项目没有 `requirements.txt`，需要手动安装以下依赖：

```bash
# 核心依赖：Ultralytics（包含 PyTorch + YOLO）
pip install ultralytics

# 如果 ultralytics 没有自动安装 PyTorch，手动安装 CUDA 版
# CUDA 12.1 版本：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8 版本：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 图像处理（EPRI 数据转换可视化需要）
pip install opencv-python

# （可选）用于 EPRI 数据转换的可视化
pip install matplotlib
```

### 4. 验证安装

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); from ultralytics import YOLO; print('Ultralytics OK')"
```

如果输出 `CUDA available: True` 说明环境配置成功。

---

## 项目结构

```
YOLOv8-OBB-Industrial-Defect-Perception/
│
├── train.py                          # 训练入口脚本
├── predict.py                        # 批量图片推理脚本
├── eval_test.py                      # 测试集评估脚本
├── convert_epri_to_obb.py            # EPRI 绝缘子数据转换（HBB JSON → OBB TXT）
│
├── insulator_obb.yaml                # 绝缘子数据集配置（4类）
├── yolov8n-obb.pt                    # YOLOv8n-OBB 预训练权重
│
│
├── datasets/                         # COCO8 示例数据（ultralytics 自动下载，非项目数据）
│   ├── coco8/
│   └── coco8-seg/
│
└── runs/                             # 训练/评估输出目录
    └── obb/insulator/
        ├── baseline_run/             # 训练运行结果（权重、曲线、日志）
        └── test_set_evaluation/      # 评估结果
```

---

## 使用指南

### 第一步：准备数据

1. 下载 EPRI 绝缘子数据集（IEEE 标准数据集）
2. 获取 `labels_v1.2.json` 和对应的图片文件夹
3. 修改 `convert_epri_to_obb.py` 中的路径：

```python
JSON_PATH = "your/path/labels_v1.2.json"
IMAGE_DIR = "your/path/Images/"
OUTPUT_DIR = "your/path/dataset_root/"
OUTPUT_VIS_DIR = "your/path/vis_check/"   # 可视化检查输出
```

4. 运行转换：

```bash
python convert_epri_to_obb.py
```

转换脚本会自动：
- 将 HBB (水平框) 转换为 OBB (旋转框)
- 分类为 4 种缺陷类型
- 生成可视化验证图像
- 输出 YOLO 格式的 TXT 标签

5. 确保数据目录结构如下：

```
dataset_root/
├── train/
│   ├── images/    # 训练图片
│   └── labels/    # 训练标签（.txt 文件，与图片同名）
├── val/
│   ├── images/    # 验证图片
│   └── labels/    # 验证标签
└── predict/
    ├── images/    # 测试图片
    └── labels/    # 测试标签
```

---

### 第二步：配置数据集 YAML

修改 `insulator_obb.yaml`：

```yaml
path: D:\你的路径\YOLOv8-OBB-Industrial-Defect-Perception\dataset_root  # 必须是绝对路径
train: train/images
val: val/images
test: predict/images
nc: 4
names:
  0: healthy_asset
  1: defective_string
  2: broken_disc
  3: flashover_disc
```

> ⚠️ **重要：** `path` 必须是**绝对路径**。

---

### 第三步：训练

修改 `train.py` 中的配置参数：

```python
YAML_PATH = r"insulator_obb.yaml"   # 数据集配置文件路径
MODEL_TYPE = "yolov8n-obb.pt"       # 预训练模型
# 可选模型：yolov8s-obb.pt, yolov8m-obb.pt, yolov8l-obb.pt（精度递增，速度递减）
```

可调训练参数：

```python
model.train(
    data=YAML_PATH,
    epochs=100,          # 训练轮数
    imgsz=640,           # 输入图像尺寸
    batch=8,             # 批大小（显存不够就减小，如 4 或 2）
    workers=4,           # 数据加载线程数
    device=0,            # GPU 设备号（CPU 训练用 "cpu"）
    lr0=0.001,           # 初始学习率
    patience=20,         # 早停 patience（验证集不再提升则提前停止）
)
```

运行训练：

```bash
python train.py
```

训练输出保存在 `runs/obb/` 目录下，包含：
- `weights/best.pt` — 最佳模型权重
- `weights/last.pt` — 最后一轮权重
- `results.csv` — 训练指标日志
- `confusion_matrix.png` — 混淆矩阵
- `results.png` — 训练曲线图

---

### 第四步：推理与评估

#### 批量图片推理

修改 `predict.py` 中的路径：

```python
model_path = r"runs/obb/insulator/baseline_run/weights/best.pt"  # 训练好的权重
source_path = r"dataset_root/predict/images"                      # 要推理的图片目录
```

运行：

```bash
python predict.py
```

结果保存在 `runs/obb/predict/` 中。

---

#### 测试集评估

修改 `eval_test.py` 中的路径：

```python
model_path = r"runs/obb/insulator/baseline_run/weights/best.pt"
data = r"insulator_obb.yaml"
```

运行：

```bash
python eval_test.py
```

输出 Precision、Recall、mAP50、mAP50-95 等指标。

---

## 可用的预训练模型

| 模型 | 大小 | 速度 | 适用场景 |
|------|------|------|----------|
| `yolov8n-obb.pt` | ~6.6 MB | 最快 | 实时检测、边缘设备 |
| `yolov8s-obb.pt` | ~22 MB | 快 | 精度略高，速度仍可接受 |
| `yolov8m-obb.pt` | ~52 MB | 中等 | 平衡精度与速度 |
| `yolov8l-obb.pt` | ~87 MB | 较慢 | 高精度需求 |

首次运行训练时，ultralytics 会自动下载所选模型。也可手动下载放入项目根目录。

---

## YOLO-OBB 标签格式说明

OBB（旋转框）标签与普通 YOLO 标签的区别：

**普通 YOLO（HBB，水平框）：**
```
class_id x_center y_center width height
```

**YOLO-OBB（旋转框）：**
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

所有坐标都是**归一化坐标**（0~1），相对于图像宽度和高度。四个点按顺序表示旋转矩形的四个顶点。

---

## 常见问题

### Q1: 训练时提示 CUDA out of memory
**解决方案：**
- 减小 `batch` 值（如改为 4 或 2）
- 减小 `imgsz`（如改为 320 或 416）
- 关闭其他占用显存的程序

### Q2: 找不到数据集文件
**检查清单：**
- YAML 中的 `path` 是否为绝对路径
- `train/images/` 和 `train/labels/` 目录是否存在
- 图片文件名与标签文件名是否一一对应（仅扩展名不同）

### Q3: 如何在 CPU 上训练/推理
将所有脚本中的 `device=0` 改为 `device="cpu"`，但速度会显著下降，不建议用于训练。

### Q4: 如何使用自己的分类标签
1. 修改 `convert_epri_to_obb.py` 中的类别映射逻辑
2. 在 YAML 文件中修改 `names` 列表，保持顺序一致
3. 修改 `nc` 数值为实际类别数

---

## 训练结果参考

| 数据集 | 模型 | Epochs | mAP50 | mAP50-95 |
|--------|------|--------|-------|----------|
| 绝缘子 (4类) | yolov8n-obb | 50 | 0.972 | 0.909 |

---

## 引用

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- EPRI 绝缘子数据集 (IEEE)(https://ieee-dataport.org/competitions/insulator-defect-detection)
