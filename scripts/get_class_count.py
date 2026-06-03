#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Label 类别统计工具
功能：
  1. 统计指定文件夹中所有 YOLO 格式 label 文件内各个类别的目标数量
  2. 可选择删除空标注文件及其对应的图片文件
YOLO label 格式：每行 "class_id x_center y_center width height"
"""

import os
import sys
import glob
from collections import Counter


# 支持的常见图片扩展名
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_empty_label(file_path: str) -> bool:
    """
    判断一个 label 文件是否为"空"（无有效标注行）

    Args:
        file_path: label 文件路径

    Returns:
        True 表示该文件是空的（没有有效目标）
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    int(parts[0])
                    return False  # 找到至少一行有效标注，不是空文件
                except ValueError:
                    continue
    return True


def find_corresponding_image(label_file: str, images_dir: str) -> str | None:
    """
    根据 label 文件路径，在同级的 images 文件夹中查找对应的图片文件。

    Args:
        label_file: label 文件的完整路径，如 /data/labels/001.txt
        images_dir: images 文件夹路径，如 /data/images

    Returns:
        找到的图片完整路径，找不到则返回 None
    """
    stem = os.path.splitext(os.path.basename(label_file))[0]

    for ext in IMAGE_EXTENSIONS:
        image_path = os.path.join(images_dir, stem + ext)
        if os.path.isfile(image_path):
            return image_path

    # 再试一次小写扩展名
    for ext in IMAGE_EXTENSIONS:
        image_path = os.path.join(images_dir, stem + ext.lower())
        if os.path.isfile(image_path):
            return image_path

    return None


def count_yolo_labels(label_dir: str) -> tuple[Counter, int, int, list[str]]:
    """
    统计 YOLO 格式 label 文件夹中各类别的目标数量

    Args:
        label_dir: 包含 .txt label 文件的文件夹路径

    Returns:
        class_counter, total_objects, file_count, empty_label_files
    """
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))

    if not label_files:
        print(f"⚠ 在目录 '{label_dir}' 中未找到任何 .txt 文件。")
        return Counter(), 0, 0, []

    class_counter = Counter()
    total_objects = 0
    file_count = 0
    empty_label_files = []

    for file_path in sorted(label_files):
        file_count += 1

        if is_empty_label(file_path):
            empty_label_files.append(file_path)
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(parts[0])
                except ValueError:
                    continue
                class_counter[class_id] += 1
                total_objects += 1

    return class_counter, total_objects, file_count, empty_label_files


def print_statistics(class_counter: Counter, total_objects: int,
                     file_count: int, empty_file_count: int, label_dir: str):
    """打印统计结果"""
    print("=" * 60)
    print(f"  📂 目录: {os.path.abspath(label_dir)}")
    print(f"  📄 label 文件总数: {file_count}")
    print(f"    其中空文件(无目标): {empty_file_count}")
    print(f"  🎯 目标总数:        {total_objects}")
    print("=" * 60)

    if total_objects == 0:
        print("  (没有任何目标被统计)")
        print("=" * 60)
        return

    bar_max_len = 30
    print(f"  {'类别ID':>8}  {'数量':>8}  {'占比':>10}  {'柱状图'}")
    print("  " + "-" * 50)

    for class_id in sorted(class_counter.keys()):
        count = class_counter[class_id]
        percentage = count / total_objects * 100
        bar_len = int(count / max(class_counter.values()) * bar_max_len)
        bar = "█" * bar_len
        print(f"  {class_id:>8}  {count:>8}  {percentage:>8.2f}%  {bar}")

    print("  " + "-" * 50)
    print(f"  {'合计':>8}  {total_objects:>8}  {'100.00%':>10}")
    print("=" * 60)

    if class_counter:
        all_ids = sorted(class_counter.keys())
        missing_ids = [i for i in range(all_ids[0], all_ids[-1] + 1)
                       if i not in class_counter]
        if missing_ids:
            print(f"\n  💡 提示: 类别 ID 范围 {all_ids[0]}~{all_ids[-1]} 中，"
                  f"以下 ID 没有出现: {missing_ids}")


def delete_empty_files(empty_label_files: list[str], label_dir: str):
    """
    询问用户是否删除空标注文件及对应图片，并执行删除操作。

    Args:
        empty_label_files: 空 label 文件路径列表
        label_dir: label 文件夹路径
    """
    if not empty_label_files:
        print("\n  ✅ 没有空标注文件，无需删除。")
        return

    print(f"\n  📋 发现 {len(empty_label_files)} 个空标注文件：")
    for f in empty_label_files:
        print(f"     - {os.path.basename(f)}")

    # 推断 images 文件夹路径
    parent_dir = os.path.dirname(label_dir)
    label_folder_name = os.path.basename(label_dir).lower()

    # 常见的 images 文件夹命名
    candidates = ["images", "img", "imgs", "JPEGImages", "train", "val", "test"]
    images_dir = None

    # 优先尝试与 labels 同级的 images 文件夹
    for name in candidates:
        candidate_path = os.path.join(parent_dir, name)
        if os.path.isdir(candidate_path):
            images_dir = candidate_path
            break

    # 也尝试路径中包含 images 的文件夹
    if images_dir is None:
        for entry in os.listdir(parent_dir):
            entry_path = os.path.join(parent_dir, entry)
            if os.path.isdir(entry_path) and "image" in entry.lower():
                images_dir = entry_path
                break

    if images_dir:
        print(f"\n  🖼 自动检测到图片文件夹: {images_dir}")
    else:
        print(f"\n  ⚠ 未在同级目录 {parent_dir} 中找到图片文件夹。")
        print("    请手动输入图片文件夹路径（直接回车则只删除 label 文件）：")
        user_input = input("    图片文件夹路径: ").strip().strip('"').strip("'")
        images_dir = user_input if user_input and os.path.isdir(user_input) else None

    # 询问用户确认
    print("\n  ⚠ 即将执行以下删除操作：")
    print(f"     - 删除 {len(empty_label_files)} 个空标注文件 (.txt)")
    if images_dir:
        print(f"     - 删除对应的图片文件 (在 {images_dir})")

    print()
    confirm = input("  是否确认删除？(y/n): ").strip().lower()

    if confirm != "y":
        print("  🚫 已取消删除操作。")
        return

    # 执行删除
    deleted_labels = 0
    deleted_images = 0
    not_found_images = []

    for label_file in empty_label_files:
        # 删除 label 文件
        try:
            os.remove(label_file)
            deleted_labels += 1
        except OSError as e:
            print(f"  ❌ 删除 label 失败: {os.path.basename(label_file)} ({e})")

        # 查找并删除对应图片
        if images_dir:
            image_path = find_corresponding_image(label_file, images_dir)
            if image_path:
                try:
                    os.remove(image_path)
                    deleted_images += 1
                except OSError as e:
                    print(f"  ❌ 删除图片失败: {os.path.basename(image_path)} ({e})")
            else:
                not_found_images.append(os.path.basename(label_file))

    # 打印结果
    print("\n" + "=" * 60)
    print("  🗑 删除完成！")
    print(f"     ✅ 已删除空标注文件: {deleted_labels} 个")
    if images_dir:
        print(f"     ✅ 已删除对应图片:   {deleted_images} 个")
        if not_found_images:
            print(f"     ⚠ 未找到对应图片:   {len(not_found_images)} 个")
            for name in not_found_images:
                print(f"        - {name}")
    print("=" * 60)


if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║     YOLO Label 类别统计工具              ║")
    print("╚══════════════════════════════════════════╝\n")

    label_dir = input("请输入 label 文件夹路径: ").strip().strip('"').strip("'")

    if not os.path.isdir(label_dir):
        print(f"\n❌ 错误: '{label_dir}' 不是一个有效的文件夹路径。")
        sys.exit(1)

    print()
    class_counter, total_objects, file_count, empty_label_files = count_yolo_labels(label_dir)

    if file_count == 0:
        sys.exit(0)

    print_statistics(class_counter, total_objects, file_count, len(empty_label_files), label_dir)
    delete_empty_files(empty_label_files, label_dir)
