# json_to_yolo_obb.py
# 工业级数据清洗：将 X-AnyLabeling 的 OBB JSON 账本一键转化为标准 YOLO-OBB 文本

import os
import json

# ==================== 【路径配置】 ====================
# 1. 存放你那 196 个原始 JSON 文件的文件夹路径
JSON_DIR = r"D:\code\python\YOLOv8-OBB-Industrial-Defect-Perception\角反识别项目\Data\jsons"
# 2. 转换后生成的标准 .txt 标签存放的基地
OUTPUT_TXT_DIR = r"D:\code\python\YOLOv8-OBB-Industrial-Defect-Perception\角反识别项目\Data\txts"
# =========================================================

def convert_json_to_yolo():
    if not os.path.exists(OUTPUT_TXT_DIR):
        os.makedirs(OUTPUT_TXT_DIR)

    json_files = [f for f in os.listdir(JSON_DIR) if f.endswith('.json')]
    success_count = 0

    print("🚚 正在启动数据管线，开始像素归一化重构...")

    for j_file in json_files:
        json_path = os.path.join(JSON_DIR, j_file)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 1. 抓取图片绝对宽高（用于归一化分母）
        img_w = data.get('imageWidth')
        img_h = data.get('imageHeight')

        if not img_w or not img_h:
            print(f"⚠️ 警告: {j_file} 未包含宽高信息，跳过。")
            continue

        # 2. 创建对应的 TXT 写入流
        txt_name = os.path.splitext(j_file)[0] + ".txt"
        txt_path = os.path.join(OUTPUT_TXT_DIR, txt_name)

        with open(txt_path, 'w', encoding='utf-8') as txt_f:
            # 3. 遍历所有的标注形状
            for shape in data.get('shapes', []):
                label = shape.get('label')

                # 锁死我们的核心资产：corner_reflector
                if label == 'corner_reflector':
                    class_id = 0
                    points = shape.get('points', [])

                    # OBB旋转框必须有完美的4个顶点坐标
                    if len(points) == 4:
                        normalized_points = []
                        for pt in points:
                            # 数学本质：像素坐标 / 总尺寸 = 归一化浮点数
                            nx = pt[0] / img_w
                            ny = pt[1] / img_h
                            normalized_points.append(f"{nx:.6f} {ny:.6f}")

                        # 拼接成 YOLO 要求的：0 x1 y1 x2 y2 x3 y3 x4 y4
                        line = f"{class_id} " + " ".join(normalized_points) + "\n"
                        txt_f.write(line)

        success_count += 1

    print("\n" + "="*20 + " 🎉 标签重构大功告成 " + "="*20)
    print(f"🔥 成功将 {success_count} 个 JSON 账本洗成标准 YOLO-OBB 文本。")
    print(f"📂 成果落盘路径: {OUTPUT_TXT_DIR}")
    print("="*61)

if __name__ == "__main__":
    convert_json_to_yolo()