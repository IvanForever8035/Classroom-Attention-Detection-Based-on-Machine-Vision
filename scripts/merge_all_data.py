#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO 多组标注数据合并工具 v2
支持两种目录结构，按子组 (front / rear) 分类合并
(已适配 Windows 复制路径带来的引号及格式问题)
"""

import os
import sys
import shutil
from collections import defaultdict

# ============================================================
#                       工具函数
# ============================================================

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_class_file(filepath):
    """解析类别映射文件"""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()

    stripped = content.strip()
    if stripped.startswith("["):
        try:
            classes = eval(stripped)
            if isinstance(classes, (list, tuple)):
                return [str(c) for c in classes]
        except Exception:
            pass

    return [line.strip() for line in content.splitlines() if line.strip()]


def detect_structure(root):
    """检测目录结构类型，返回 1 / 2 / None"""
    if (os.path.isdir(os.path.join(root, "images", "all"))
            and os.path.isdir(os.path.join(root, "labels", "all"))
            and os.path.isfile(os.path.join(root, "classes.txt"))):
        return 1

    if (os.path.isdir(os.path.join(root, "front", "images", "all"))
            and os.path.isdir(os.path.join(root, "rear", "images", "all"))
            and os.path.isdir(os.path.join(root, "skip", "images", "all"))
            and os.path.isfile(os.path.join(root, "classes.txt"))):
        return 2

    return None


def get_image_files(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    )


def get_txt_files(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(f for f in os.listdir(folder) if f.endswith(".txt"))


def remap_label_lines(lines, old_to_new):
    """将标注行中的旧类别ID映射为新ID"""
    result = []
    skipped = 0
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        old_id = int(parts[0])
        if old_id in old_to_new:
            parts[0] = str(old_to_new[old_id])
            result.append(" ".join(parts) + "\n")
        else:
            skipped += 1
    return result, skipped


def fix_path(p):
    """适配 Windows 路径：去首尾引号、去空白、统一斜杠"""
    p = p.strip().strip('"').strip("'")
    return os.path.normpath(p)


def divider(char="=", length=65):
    print(char * length)


# ============================================================
#                       主流程
# ============================================================

def main():
    divider()
    print("  YOLO 多组标注数据合并工具 v2")
    divider()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # ----------------------------------------------------------
    # 1. 交互式输入各组数据
    # ----------------------------------------------------------
    groups = []

    while True:
        idx = len(groups) + 1
        print(f"\n{'─' * 40}")
        print(f"  【第 {idx} 组】（直接回车结束输入）")
        folder = input("  数据根目录路径: ").strip()
        if not folder:
            break
        
        # ✅ 自动适配 Windows 复制过来的路径（去引号、规范化）
        folder = fix_path(folder)

        if not os.path.exists(folder):
            print(f"  ✗ 路径不存在: {folder}")
            continue
        if not os.path.isdir(folder):
            print(f"  ✗ 路径存在但不是文件夹: {folder}")
            continue

        structure = detect_structure(folder)
        if structure is None:
            print("  ✗ 无法识别目录结构（需为结构1或结构2）")
            print("     结构1: root/images/all/  root/labels/all/  root/classes.txt")
            print("     结构2: root/{front,rear,skip}/images/all/  root/classes.txt")
            continue

        class_file = os.path.join(folder, "classes.txt")
        classes = parse_class_file(class_file)
        if not classes:
            print("  ✗ 类别文件为空")
            continue

        groups.append({"folder": folder, "structure": structure, "classes": classes})
        print(f"  ✓ 结构{structure}  |  {len(classes)} 个类别: {classes}")

    if not groups:
        print("\n未输入任何数据，退出。")
        sys.exit(0)

    # ----------------------------------------------------------
    # 2. 构建统一类别列表
    # ----------------------------------------------------------
    unified_classes = []
    for g in groups:
        for cls in g["classes"]:
            if cls not in unified_classes:
                unified_classes.append(cls)

    print("\n")
    divider()
    print(f"  合并后类别顺序（共 {len(unified_classes)} 个）")
    divider()
    for i, cls in enumerate(unified_classes):
        sources = []
        for gi, g in enumerate(groups):
            if cls in g["classes"]:
                sources.append(f"组{gi + 1}[{g['classes'].index(cls)}]")
        print(f"    {i:>3d} : {cls:<35s}  ← {', '.join(sources)}")

    # ----------------------------------------------------------
    # 3. 为每组构建 old_id -> new_id 映射
    # ----------------------------------------------------------
    for g in groups:
        g["id_map"] = {}
        for old_id, cls_name in enumerate(g["classes"]):
            g["id_map"][old_id] = unified_classes.index(cls_name)

    # ----------------------------------------------------------
    # 4. 选择输出路径
    # ----------------------------------------------------------
    print("\n")
    default_output = os.path.join(script_dir, "merged_yolo")
    output_dir = input(f"输出根目录路径 (默认 {default_output}): ").strip()
    if not output_dir:
        output_dir = default_output
    else:
        output_dir = fix_path(output_dir)  # ✅ 同样做路径适配

    # ----------------------------------------------------------
    # 5. 确定每个输出子组的数据来源
    # ----------------------------------------------------------
    source_map = {"front": [], "rear": []}

    for gi, g in enumerate(groups):
        if g["structure"] == 1:
            source_map["front"].append((gi, "common"))
            source_map["rear"].append((gi, "common"))
        else:
            source_map["front"].append((gi, "front"))
            source_map["front"].append((gi, "skip"))
            source_map["rear"].append((gi, "rear"))
            source_map["rear"].append((gi, "skip"))

    divider()
    print("  合并方案")
    divider()
    for out_sub, sources in source_map.items():
        desc = []
        for gi, src_sub in sources:
            desc.append(f"组{gi + 1}.{src_sub}")
        print(f"  {out_sub:8s} ← {' + '.join(desc)}")

    # ----------------------------------------------------------
    # 6. 逐输出子组合并
    # ----------------------------------------------------------
    print("\n")
    divider()
    print("  开始合并...")
    divider()

    grand_images = 0
    grand_labels = 0

    for out_sub in ["front", "rear"]:
        # img_out = os.path.join(output_dir, out_sub, "images", "all")
        # lbl_out = os.path.join(output_dir, out_sub, "labels", "all")
        img_out = os.path.join(output_dir, out_sub, "images")
        lbl_out = os.path.join(output_dir, out_sub, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        img_collected = {}
        lbl_collected = defaultdict(list)

        for gi, src_sub in source_map[out_sub]:
            g = groups[gi]
            if src_sub == "common":
                src_img_dir = os.path.join(g["folder"], "images", "all")
                src_lbl_dir = os.path.join(g["folder"], "labels", "all")
            else:
                src_img_dir = os.path.join(g["folder"], src_sub, "images", "all")
                src_lbl_dir = os.path.join(g["folder"], src_sub, "labels", "all")

            for fname in get_image_files(src_img_dir):
                if fname not in img_collected:
                    img_collected[fname] = os.path.join(src_img_dir, fname)

            for fname in get_txt_files(src_lbl_dir):
                stem = os.path.splitext(fname)[0]
                lbl_collected[stem].append(
                    (os.path.join(src_lbl_dir, fname), g["id_map"])
                )

        img_count = 0
        for fname in sorted(img_collected):
            shutil.copy2(img_collected[fname], os.path.join(img_out, fname))
            img_count += 1

        lbl_count = 0
        merge_count = 0
        for stem in sorted(lbl_collected):
            sources = lbl_collected[stem]
            merged_lines = []
            total_skipped = 0

            for src_path, id_map in sources:
                with open(src_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                remapped, skipped = remap_label_lines(lines, id_map)
                merged_lines.extend(remapped)
                total_skipped += skipped

            if total_skipped > 0:
                print(f"    ⚠ {out_sub}/{stem}.txt: 跳过 {total_skipped} 行（类别ID不在映射中）")

            out_path = os.path.join(lbl_out, stem + ".txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.writelines(merged_lines)

            lbl_count += 1
            if len(sources) > 1:
                merge_count += 1

        print(f"\n  ▶ {out_sub:8s}")
        print(f"    图片文件: {img_count}")
        print(f"    标注文件: {lbl_count}  （其中 {merge_count} 个由多来源合并）")

        grand_images += img_count
        grand_labels += lbl_count

    # ----------------------------------------------------------
    # 7. 保存统一类别文件
    # ----------------------------------------------------------
    class_out = os.path.join(output_dir, "classes.txt")
    with open(class_out, "w", encoding="utf-8") as f:
        for cls in unified_classes:
            f.write(cls + "\n")

    class_list_out = os.path.join(output_dir, "classes_list.txt")
    with open(class_list_out, "w", encoding="utf-8") as f:
        f.write(str(unified_classes))

    # ----------------------------------------------------------
    # 8. 最终汇总
    # ----------------------------------------------------------
    print("\n")
    divider()
    print("  ✅ 合并完成！")
    divider()
    print(f"  输入组数          : {len(groups)}")
    print(f"  合并图片总数      : {grand_images}")
    print(f"  合并标注文件总数  : {grand_labels}")
    print(f"  合并类别总数      : {len(unified_classes)}")
    print(f"  输出目录          : {os.path.abspath(output_dir)}")
    print(f"  类别文件          : {class_out}")
    print(f"  类别列表          : {class_list_out}")
    divider()
    print("  合并后的类别顺序:")
    divider()
    for i, cls in enumerate(unified_classes):
        print(f"    {i:>3d} : {cls}")
    divider()


if __name__ == "__main__":
    main()
