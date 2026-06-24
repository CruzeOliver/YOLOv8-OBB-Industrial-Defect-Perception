import os
import random
import re
import shutil

# ==================== 【用户配置区域】 ====================
IMAGE_SRC = r"D:\code\python\YOLO_train_data\IEEEdata\Train\Images"         # 原版JPG图片文件夹
LABEL_SRC = r"D:\code\python\YOLO_train_data\IEEEdata\Train\YOLOtxt"     # TXT标签文件夹
OUTPUT_ROOT = r"D:\code\python\YOLO_train_data\dataset_root"     # 划分后标准YOLO数据集的存放总根目录
# =========================================================


def get_group_key(filename):
    """
    提取文件的"内容组"标识：去掉文件名末尾的字母后缀（如 h/v 表示不同角度）。
    例如: 100213h.txt → 100213,  100213v.txt → 100213,  12345.txt → 12345
    这样同一内容、不同角度的图片会被分到同一个集合中。
    """
    stem = os.path.splitext(filename)[0]
    # 去掉末尾的所有字母字符（a-z, A-Z），保留核心内容ID
    group_key = re.sub(r'[a-zA-Z]+$', '', stem)
    # 如果整个 stem 都是字母（极端情况），保留原值
    return group_key if group_key else stem

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

def split_data(train_ratio=0.7, predict_ratio=0.1):
    # 1. 创建标准的YOLO目标检测目录架构
    for split in ['train', 'val', 'predict']:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, split, 'labels'), exist_ok=True)

    # 2. 扫描并获取所有TXT标签的文件名（以标签为基准，防止图片有漏标的情况）
    all_labels = [f for f in os.listdir(LABEL_SRC) if f.endswith('.txt')]

    # 💡 按"内容组"分组：比如 100213h.txt 和 100213v.txt 归为同一组 (key="100213")
    groups = {}  # group_key -> [label_filename, ...]
    for label in all_labels:
        group_key = get_group_key(label)
        groups.setdefault(group_key, []).append(label)

    group_keys = list(groups.keys())

    # 💡 在"组"级别上强行打乱顺序，保证同一内容的图片不会被分散到不同集合
    random.seed(42)  # 固定随机种子，确保可复现性
    random.shuffle(group_keys)

    # 3. 计算切分节点：先预留 predict，剩余再按 train_ratio 划分
    total_groups = len(group_keys)
    predict_count = max(1, int(total_groups * predict_ratio))  # 至少保留 1 组
    remaining_count = total_groups - predict_count
    train_count = int(remaining_count * train_ratio)

    predict_groups = group_keys[:predict_count]
    remaining_groups = group_keys[predict_count:]
    train_groups = remaining_groups[:train_count]
    val_groups = remaining_groups[train_count:]

    # 展开分组 -> 标签列表
    train_labels = [label for g in train_groups for label in groups[g]]
    val_labels = [label for g in val_groups for label in groups[g]]
    predict_labels = [label for g in predict_groups for label in groups[g]]

    print(f"📊 资产盘点: 发现有效标注样本 {len(all_labels)} 组 (共 {total_groups} 个内容组)。")
    print(f"⏳ 正在分配: predict {predict_ratio*100}% ({len(predict_labels)}组标签, {len(predict_groups)}个内容组) | "
          f"train/val 按照 {train_ratio*100}%:{(1-train_ratio)*100}% 分配剩余 {remaining_count} 个内容组...")

    # 执行复制
    copy_file_group(train_labels, 'train')
    copy_file_group(val_labels, 'val')
    copy_file_group(predict_labels, 'predict')
    print(f"\n🏁 数据集构建成功！标准的YOLO架构已落盘至: {OUTPUT_ROOT}")

if __name__ == "__main__":
    split_data()