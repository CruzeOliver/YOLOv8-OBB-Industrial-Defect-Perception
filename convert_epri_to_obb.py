import os
import json
import cv2

# ==================== 【用户配置区域】 ====================
JSON_PATH = r"D:\code\python\YOLO_vision\IEEEdata\Train\labels_v1.2.json"        # EPRI的JSON标签文件路径
IMAGE_DIR = r"D:\code\python\YOLO_vision\IEEEdata\Train\Images"            # 存放JPG图片的文件夹路径
OUTPUT_DIR = r"D:\code\python\YOLO_vision\IEEEdata\Train\YOLOtxt"      # 转换后TXT标签的输出路径
OUTPUT_VIS_DIR = r"D:\code\python\YOLO_vision\IEEEdata\Train\visual_check" # 画好框的图片另存为路径
# =========================================================

# 定义4个类别的名称与对应的画框颜色 (OpenCV中颜色顺序为 BGR)
CLASS_INFO = {
    0: {"name": "Healthy", "color": (0, 255, 0)},       # 绿色：健康的单片或整串
    1: {"name": "Defect_String", "color": (0, 165, 255)},# 橙色：带缺陷的整串
    2: {"name": "Broken_Disc", "color": (0, 0, 255)},    # 红色：破损单片
    3: {"name": "Flashed_Disc", "color": (255, 0, 0)}  # 蓝色：闪络单片
}

def convert_bbox_to_obb(bbox, img_w, img_h):
    """将 [X, Y, W, H] 转换为 YOLOv8-OBB 规范的 4个顺时针顶点归一化坐标"""
    x, y, w, h = bbox
    x1, y1 = x, y
    x2, y2 = x + w, y
    x3, y3 = x + w, y + h
    x4, y4 = x, y + h

    x1_n, x2_n, x3_n, x4_n = x1 / img_w, x2 / img_w, x3 / img_w, x4 / img_w
    y1_n, y2_n, y3_n, y4_n = y1 / img_h, y2 / img_h, y3 / img_h, y4 / img_h
    return f"{x1_n:.6f} {y1_n:.6f} {x2_n:.6f} {y2_n:.6f} {x3_n:.6f} {y3_n:.6f} {x4_n:.6f} {y4_n:.6f}"

def process_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True) # 创建可视化检查文件夹

    print("⏳ 正在读取并解析 EPRI 官方 JSON 标签...")
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        data_list = json.load(f)

    print(f"✅ 成功加载 {len(data_list)} 张图片的标注信息。开始同步清洗与可视化渲染...")
    success_count = 0

    for item in data_list:
        filename = item.get("filename")
        labels_dict = item.get("Labels", {})
        objects = labels_dict.get("objects", [])

        if not objects:
            continue

        img_path = os.path.join(IMAGE_DIR, filename)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # 💡 创建一张专门用来画框的画布，不污染原图
        vis_img = img.copy()

        # 第一轮前置扫描：查明该图片中所有的单片，是否存在任何破损或闪络缺陷
        has_defect_in_image = False
        for obj in objects:
            if obj.get("string") == 0:
                conds = obj.get("conditions", {})
                if conds and "No issues" not in conds:
                    has_defect_in_image = True
                    break

        txt_lines = []

        # 第二轮正式扫描：分类、写TXT，并在画布上画框
        for obj in objects:
            is_string = obj.get("string")
            conds = obj.get("conditions", {})
            bbox = obj.get("bbox")

            if not bbox or len(bbox) != 4:
                continue

            class_idx = -1
            if is_string == 0:
                if "No issues" in conds:
                    class_idx = 0
                elif any("Broken" in str(v) for v in conds.values()):
                    class_idx = 2
                elif any("Flashover" in str(v) for v in conds.values()):
                    class_idx = 3
                else:
                    class_idx = 0
            elif is_string == 1:
                if has_defect_in_image:
                    class_idx = 1
                else:
                    class_idx = 0

            if class_idx != -1:
                # 1. 转换并记录 YOLO 文本数据
                obb_coords = convert_bbox_to_obb(bbox, img_w, img_h)
                txt_lines.append(f"{class_idx} {obb_coords}\n")

                # 2. ==================== 【新增：OpenCV 图像画框可视化】 ====================
                X, Y, W, H = bbox
                color = CLASS_INFO[class_idx]["color"]
                label_text = CLASS_INFO[class_idx]["name"]

                # 在画布上画出绝对像素矩形框 (线宽设为 3，确保大图里也能看清)
                cv2.rectangle(vis_img, (int(X), int(Y)), (int(X + W), int(Y + H)), color, 3)

                # 在框的左上角上方写上类别文字
                cv2.putText(vis_img, label_text, (int(X), int(Y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                # =========================================================================

        # 保存 YOLO 文本标签
        if txt_lines:
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            with open(os.path.join(OUTPUT_DIR, txt_filename), 'w', encoding='utf-8') as f_txt:
                f_txt.writelines(txt_lines)

            # 【新增】将画好框的验证图片另存到指定文件夹
            vis_img_path = os.path.join(OUTPUT_VIS_DIR, f"check_{filename}")
            cv2.imwrite(vis_img_path, vis_img)

            success_count += 1

    print(f"🏁 洗数据与可视化验票全部完成！")
    print(f"📄 YOLOv8-OBB 文本标签已存至: {OUTPUT_DIR}")
    print(f"🖼️ 画框对比图 已存至: {OUTPUT_VIS_DIR}")

if __name__ == "__main__":
    process_dataset()