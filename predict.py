import os
from ultralytics import YOLO

def main():
    model_path = r"runs\obb\insulator\baseline_run-2\weights\best.pt"
    model = YOLO(model_path)

    source_path = r"D:\code\python\YOLOv8-OBB-Industrial-Defect-Perception\dataset_root\predict\images" #需要测试的盲盒图片目录

    print("\n绝缘子缺陷智能感知系统正在启动推理...")

    # 3. 开启预测
    results = model.predict(
        source=source_path,
        save=True,           # 自动把画好旋转框的成果图保存到硬盘上
        conf=0.25,           # 置信度门禁：AI 觉得概率大于 25% 的缺陷才画框，防止虚警
        device=0             # 调用 4060 Ti
    )
    print("\n🏁 测试完毕！")


if __name__ == "__main__":
    main()