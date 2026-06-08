# YOLOv8-OBB 工业级自检与正式训练脚本

import os
import sys
import torch
from ultralytics import YOLO

# ==================== 【全局核心配置】 ====================
YAML_PATH = r"D:\code\python\YOLOv8-OBB-Industrial-Defect-Perception\角反识别项目\corner_reflector.yaml"
MODEL_TYPE = "yolov8n-obb.pt"
EPOCHS = 100
BATCH_SIZE = 8
# =========================================================

def run_sanity_check():

    cuda_available = torch.cuda.is_available()
    print(f"    - PyTorch 框架版本: {torch.__version__}")
    print(f"    - NVIDIA CUDA 加速: {'✅ 已经就位' if cuda_available else '❌ 未激活 (将降级为CPU)'}")
    if cuda_available:
        print(f"    - 核心参战独显型号: {torch.cuda.get_device_name(0)}")
    else:
        print("    🚨 警告: 强烈不建议使用 CPU 训练 OBB 模型！")
        sys.exit(1) # 硬核终止，防止浪费生命

    if os.path.exists(YAML_PATH):
        print(f"    -  YAML 配置文件: ✅ 完美捕获 -> {YAML_PATH}")
    else:
        print(f"    - ❌ 致命错误: 在指定路径下找不到 YAML 文件，请核对路径！")
        sys.exit(1)



def main():
    # 1. 触发自检锁（只要前面有一个地方不满足，直接原地拒绝执行，不触发后续大代码）
    run_sanity_check()

    # 2. 正式加载模型
    model = YOLO(MODEL_TYPE)

    # 3. 轰击数据集，开启正规军大训
    print(f"🚀 启动感知训练（总计 {EPOCHS} 轮）...")
    model.train(
        data=YAML_PATH,
        epochs=EPOCHS,
        imgsz=640,
        batch=BATCH_SIZE,
        workers=4,
        device=0,          # 0 代表直接死锁你的第一张 NVIDIA 独显 4060 Ti
        project="insulator",
        name="baseline_run"
    )

if __name__ == "__main__":
    main()