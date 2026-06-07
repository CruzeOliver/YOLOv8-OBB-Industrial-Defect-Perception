from ultralytics import YOLO

def main():
    model_path = r"runs\obb\insulator\baseline_run-2\weights\best.pt"
    model = YOLO(model_path)

    # 2. 调用 val 函数，但通过 split='test' 强制它去跑 test 目录下的数据
    metrics = model.val(
        data=r"D:\code\python\YOLOv8-OBB-Industrial-Defect-Perception\insulator_obb.yaml",
        split='test',      # 这个参数告诉它去评测 test 集而不是默认的 val 集
        imgsz=640,
        device=0,          # 依旧动用 4060 Ti 闪电速度批改
        project="insulator",
        name="test_set_evaluation" # 它的结果会单独存放在这个文件夹里
    )

    # 3. 现场打印最核心的对账数据
    print(f"🎯 综合精确率 (Precision): {metrics.box.mp:.4f} ")
    print(f"🎯 综合召回率 (Recall):    {metrics.box.mr:.4f} ")
    print(f"📈 核心及格线 (mAP50):     {metrics.box.map50:.4f}")
    print(f"🏆 严苛工业线 (mAP50-95):  {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()