import cv2
import time
from ultralytics import YOLO

def main():
    # 1. 加载角反 YOLO-OBB 模型权重
    model_path = r"D:\code\python\YOLOv8-OBB-Industrial-Defect-Perception\runs\obb\insulator\baseline_run-4\weights\best.pt"
    model = YOLO(model_path)

    # 2. 初始化摄像头（优先使用 DSHOW 后端并执行预读暖机）
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.read()

    print("🔮 实时角反射器视觉定位模块已启动（按 ESC 键退出）...")

    prev_time = 0
    frame_stability_counter = 0
    STABILITY_THRESHOLD = 3 # 多帧一致性滤波阈值，防背景方框杂波误检

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 3. 显卡加速推理（verbose=False 关闭后台耗时日志打印，conf=0.65 过滤低分误检）
        results = model.predict(source=frame, show=False, verbose=False, conf=0.5, imgsz=640, device=0)

        best_detection = None
        for result in results:
            if result.obb is not None:
                conf_list = result.obb.conf.cpu().numpy()
                if len(conf_list) > 0:
                    # 每帧只捕获置信度最高的那一个目标
                    best_idx = conf_list.argmax()
                    best_detection = {
                        'corners': result.obb.xyxyxyxy.cpu().numpy()[best_idx],
                        'xywhr': result.obb.xywhr.cpu().numpy()[best_idx],
                        'conf': conf_list[best_idx]
                    }

        # 4. 时域滤波计数器更新
        if best_detection:
            frame_stability_counter = min(frame_stability_counter + 1, STABILITY_THRESHOLD * 2)
        else:
            frame_stability_counter = max(frame_stability_counter - 1, 0)

        # 5. 稳定通过滤波后，执行画面渲染
        if best_detection and frame_stability_counter >= STABILITY_THRESHOLD:
            corners = best_detection['corners']
            xywhr = best_detection['xywhr']
            conf = best_detection['conf']
            pts = corners.astype(int)

            # 画紧贴边缘的彩框 (绿色)
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # 计算并画出几何中心点 (红色) 以及打印坐标
            cx, cy = int(xywhr[0]), int(xywhr[1])
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # 在最靠上的顶点上方绘制标签 text
            pts_y_sorted = pts[pts[:, 1].argsort()]
            top_pt = pts_y_sorted[0]
            cv2.putText(frame, f"corner_reflector {conf:.2f}", (int(top_pt[0]), int(top_pt[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 6. 计算并渲染 FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Radar-Camera Calibration Assist Tool (Webcam OBB)", frame)

        # 7. 捕获 ESC 键安全退出 (ESC 的 ASCII 码为 27)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("系统已安全关闭。")

if __name__ == "__main__":
    main()