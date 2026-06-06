import os
import random
import shutil

# ==================== 【用户配置区域】 ====================
IMAGE_SRC = r"D:\code\python\YOLOv8-OBB-Industrial-Defect-Perception\dataset_root\root\Images"         # 原版JPG图片文件夹
LABEL_SRC = r"D:\code\python\YOLOv8-OBB-Industrial-Defect-Perception\dataset_root\root\YOLOtxt"     # TXT标签文件夹
OUTPUT_ROOT = r"D:\code\python\YOLOv8-OBB-Industrial-Defect-Perception\dataset_root"     # 划分后标准YOLO数据集的存放总根目录
# =========================================================

def copy_file_group(label_list, split_name):
    copied_count = 0
    # 支持的图片后缀（按优先级排序）
    IMG_SUFFIXES = [".JPG", ".jpg", ".jpeg"]

    for label_name in label_list:
        base_name = os.path.splitext(label_name)[0]
        src_label = os.path.join(LABEL_SRC, label_name)
        src_img = None

        # 自动匹配存在的图片格式 ✅
        for suffix in IMG_SUFFIXES:
            img_name = base_name + suffix
            img_path = os.path.join(IMAGE_SRC, img_name)
            if os.path.exists(img_path):
                src_img = img_path
                break

        # 复制文件
        if src_img is not None:
            dst_img = os.path.join(OUTPUT_ROOT, split_name, 'images', os.path.basename(src_img))
            dst_label = os.path.join(OUTPUT_ROOT, split_name, 'labels', label_name)

            shutil.copy(src_img, dst_img)
            shutil.copy(src_label, dst_label)
            copied_count += 1
        else:
            print(f"🚨 警报: 找到了标签 {label_name}，但图片不存在！")

    print(f"✅ 完成 【{split_name}】 集划分，共同步复制 {copied_count} 组图片与标签文件。")

def split_data(train_ratio=0.8):
    # 1. 创建标准的YOLO目标检测目录架构
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, split, 'labels'), exist_ok=True)

    # 2. 扫描并获取所有TXT标签的文件名（以标签为基准，防止图片有漏标的情况）
    all_labels = [f for f in os.listdir(LABEL_SRC) if f.endswith('.txt')]

    # 💡 强行打乱顺序，保证训练集和验证集的数据分布是随机公平的
    random.seed(42)  # 固定随机种子，确保可复现性
    random.shuffle(all_labels)

    # 3. 计算切分节点
    split_idx = int(len(all_labels) * train_ratio)
    train_labels = all_labels[:split_idx]
    val_labels = all_labels[split_idx:]

    print(f"📊 资产盘点: 发现有效标注样本 {len(all_labels)} 组。")
    print(f"⏳ 正在按照 {train_ratio*100}% : {(1-train_ratio)*100}% 分配数据...")

    # 执行复制
    copy_file_group(train_labels, 'train')
    copy_file_group(val_labels, 'val')
    print(f"\n🏁 数据集构建成功！标准的YOLO架构已落盘至: {OUTPUT_ROOT}")

if __name__ == "__main__":
    split_data()