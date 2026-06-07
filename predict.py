import os
from ultralytics import YOLO

def main():
    model_path = r"runs\obb\insulator\baseline_run\weights\best.pt"
    model = YOLO(model_path)

    source_path = r".\test_images" #需要测试的盲盒图片目录

    print("\n🔮 绝缘子缺陷智能感知系统正在启动推理...")
    print(f"正在读取 '{source_path}' 目录下的盲盒图片进行显卡加速计算...\n")

    # 3. 开启闪电预测
    results = model.predict(
        source=source_path,
        save=True,           # 自动把画好旋转框的成果图保存到硬盘上
        conf=0.25,           # 置信度门禁：AI 觉得概率大于 25% 的缺陷才画框，防止虚警
        device=0             # 调用 4060 Ti
    )

    print("\n🏁 盲盒测试完毕！")
    print("🌟 成果图已整整齐齐地保存在 'runs/obb/predict/' 目录下，快去开盲盒吧！")

if __name__ == "__main__":
    main()