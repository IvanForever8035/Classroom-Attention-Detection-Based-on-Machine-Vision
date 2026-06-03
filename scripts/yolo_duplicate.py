#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# YOLO 数据集类别复制工具

import os
import shutil
import glob
from pathlib import Path


def find_category_mapping(labels_dir):
    """扫描所有label文件，找到所有出现的类别ID"""
    categories = set()
    for lf in glob.glob(os.path.join(labels_dir, "*.txt")):
        with open(lf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if parts:
                        try:
                            categories.add(int(parts[0]))
                        except ValueError:
                            continue
    return sorted(categories)


def find_category_counts(labels_dir, categories):
    """
    返回两种统计：
      box_counts   - 每个类别的标注框数量（每个框都算）
      file_counts  - 包含该类别的文件数量（每个文件只算一次）
      file_categories - 每个文件包含的类别集合
    """
    box_counts = {c: 0 for c in categories}
    file_counts = {c: 0 for c in categories}
    file_categories = {}
    for lf in glob.glob(os.path.join(labels_dir, "*.txt")):
        file_cats = set()
        with open(lf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if parts:
                        try:
                            cat_id = int(parts[0])
                            if cat_id in box_counts:
                                box_counts[cat_id] += 1
                                file_cats.add(cat_id)
                        except ValueError:
                            continue
        file_categories[lf] = file_cats
        for c in file_cats:
            file_counts[c] += 1
    return box_counts, file_counts, file_categories


def find_image_path(label_path, images_dir):
    """根据label文件路径，找到对应的图片文件"""
    label_name = Path(label_path).stem
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
                ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP"]:
        img_path = os.path.join(images_dir, label_name + ext)
        if os.path.exists(img_path):
            return img_path
    return None


def clean_path(p):
    """去除路径两端的空白和引号"""
    return p.strip().strip('"').strip("'").strip()


def print_distribution(box_counts, file_counts, target_categories=None):
    """打印类别分布表格"""
    total_boxes = sum(box_counts.values())
    categories = sorted(box_counts.keys())
    max_count = max(box_counts.values())
    print()
    print("-" * 60)
    print("  类别ID    标注框数   文件数    占比      可视化")
    print("  " + "-" * 55)
    for cat_id in categories:
        bc = box_counts[cat_id]
        fc = file_counts[cat_id]
        pct = bc / total_boxes * 100 if total_boxes > 0 else 0
        bar_len = int(bc / max_count * 22) if max_count > 0 else 0
        bar = "█" * bar_len
        marker = ""
        if target_categories and cat_id in target_categories:
            marker = " <-- 目标"
        print("  " + str(cat_id).ljust(10) + str(bc).ljust(10) + str(fc).ljust(10) + "{:>5.2f}%   ".format(pct) + bar + marker)
    print("  " + "-" * 55)
    print("  总计：" + str(total_boxes) + " 个标注框")
    print("-" * 60)


def main():
    print("=" * 60)
    print("         YOLO 数据集类别复制工具")
    print("=" * 60)
    print()

    # 第一步：输入数据集路径
    dataset_path = clean_path(input("请输入数据集的根目录路径："))
    dataset_path = os.path.abspath(dataset_path)

    if not os.path.exists(dataset_path):
        print("\n[错误] 路径不存在 -> " + dataset_path)
        return

    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")

    if not os.path.exists(images_dir):
        print("\n[错误] 找不到 images 文件夹 -> " + images_dir)
        return
    if not os.path.exists(labels_dir):
        print("\n[错误] 找不到 labels 文件夹 -> " + labels_dir)
        return

    # 第二步：扫描所有类别
    print("\n正在扫描数据集: " + dataset_path)
    categories = find_category_mapping(labels_dir)

    if not categories:
        print("\n[错误] 未在任何 label 文件中找到类别信息")
        return

    box_counts, file_counts, file_categories = find_category_counts(labels_dir, categories)

    # 显示复制前的分布
    print("\n[ 复制前的类别分布 ]")
    print_distribution(box_counts, file_counts)

    # 第三步：输入要复制的类别
    print()
    category_input = input("请输入要复制的类别ID（多个用逗号分隔，如 6 或 6,7）：").strip()

    try:
        target_categories = [int(x.strip()) for x in category_input.split(",")]
    except ValueError:
        print("\n[错误] 请输入有效的数字ID")
        return

    invalid = [c for c in target_categories if c not in categories]
    if invalid:
        print("\n[错误] 以下类别ID不存在 -> " + str(invalid))
        print("   可用类别：" + str(categories))
        return

    # 第四步：输入复制份数
    print()
    copy_times = input("请输入每个文件要复制的份数：").strip()
    try:
        copy_times = int(copy_times)
        if copy_times < 1:
            print("\n[错误] 复制份数必须大于 0")
            return
    except ValueError:
        print("\n[错误] 请输入有效的正整数")
        return

    # 第五步：查找匹配文件
    print("\n正在查找包含类别 " + str(target_categories) + " 的文件...")

    matched_labels = []
    for lf, cats in file_categories.items():
        if cats & set(target_categories):
            matched_labels.append(lf)

    if not matched_labels:
        print("\n[错误] 未找到包含类别 " + str(target_categories) + " 的任何文件")
        return

    print("找到 " + str(len(matched_labels)) + " 个包含目标类别的文件")
    print()

    # 预览
    total_new_files = len(matched_labels) * copy_times
    print("操作预览：")
    print("   匹配文件数：" + str(len(matched_labels)))
    print("   每个复制份数：" + str(copy_times))
    print("   将新增文件数：" + str(total_new_files * 2) + "（图片+标注各 " + str(total_new_files) + " 个）")
    print()

    confirm = input("确认执行？(y/n)：").strip().lower()
    if confirm != "y":
        print("\n已取消操作")
        return

    # 第六步：执行复制
    print("\n开始复制...")
    success_count = 0
    fail_count = 0

    for lf in matched_labels:
        img_path = find_image_path(lf, images_dir)
        if img_path is None:
            print("   [警告] 找不到图片文件：" + Path(lf).stem + "，跳过")
            fail_count += 1
            continue

        img_ext = Path(img_path).suffix
        label_ext = Path(lf).suffix
        base_name = Path(lf).stem

        for i in range(1, copy_times + 1):
            new_img_name = base_name + "_copy" + str(i) + img_ext
            new_label_name = base_name + "_copy" + str(i) + label_ext

            new_img_path = os.path.join(images_dir, new_img_name)
            new_label_path = os.path.join(labels_dir, new_label_name)

            # 如果已存在则跳过，防止重复运行时重复复制
            if os.path.exists(new_img_path):
                continue

            try:
                shutil.copy2(img_path, new_img_path)
                shutil.copy2(lf, new_label_path)
                success_count += 1
            except Exception as e:
                print("   [错误] 复制失败：" + base_name + " -> " + str(e))
                fail_count += 1

    # 第七步：结果汇总
    print()
    print("=" * 60)
    print("   操作完成！")
    print("   成功复制：" + str(success_count) + " 对文件")
    if fail_count > 0:
        print("   失败/跳过：" + str(fail_count) + " 个")
    print("=" * 60)

    # 重新统计并显示复制后的分布
    new_categories = find_category_mapping(labels_dir)
    new_box_counts, new_file_counts, _ = find_category_counts(labels_dir, new_categories)
    print("\n[ 复制后的类别分布 ]")
    print_distribution(new_box_counts, new_file_counts, target_categories)


if __name__ == "__main__":
    main()
