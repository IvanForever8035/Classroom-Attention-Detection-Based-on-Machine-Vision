#!/usr/bin/env python3
"""删除 YOLO 标注中的指定类别（支持 train/val 子文件夹结构，支持一次删除多个类别）"""
from pathlib import Path
import shutil

def process_dir(labels_dir, target_classes, renumber):
    """处理单个标注文件夹"""
    processed = 0
    for txt in labels_dir.glob("*.txt"):
        lines = []
        for line in txt.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cid = int(parts[0])
            
            # 如果当前类别在待删除的集合中，直接跳过
            if cid in target_classes:
                continue
                
            if renumber:
                # 计算在当前类别之前有多少个类别被删除了，用于位移新的类别 ID
                offset = sum(1 for t in target_classes if t < cid)
                new_cid = cid - offset
                lines.append(f"{new_cid} " + " ".join(parts[1:]))
            else:
                lines.append(line)
                
        txt.write_text("\n".join(lines) + "\n")
        processed += 1
    return processed

def main():
    # 运行时提问
    labels_dir = Path(input("标注文件夹路径: ").strip().strip('"').strip("'"))
    
    # 修改输入提示，允许输入多个ID
    classes_input = input("要删除的类别 ID (多个以空格分隔，如 1 3 5): ").strip()
    try:
        target_classes = set(map(int, classes_input.split()))
    except ValueError:
        print("❌ 输入的类别 ID 格式错误，请输入数字并用空格分隔")
        return
        
    if not target_classes:
        print("❌ 未输入要删除的类别 ID")
        return
        
    renumber = input("是否重新编号剩余类别？(y/n, 默认 y): ").strip().lower() != 'n'
    backup = input("是否备份原文件到 labels_backup？(y/n, 默认 y): ").strip().lower() != 'n'

    if not labels_dir.exists():
        print("❌ 路径不存在")
        return

    # 备份
    if backup:
        backup_dir = labels_dir.parent / "labels_backup"
        backup_dir.mkdir(parents=True, exist_ok=True)
        for txt in labels_dir.rglob("*.txt"):
            rel_path = txt.relative_to(labels_dir)
            dest = backup_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(txt, dest)
        print(f"✅ 备份完成: {backup_dir}")

    # 统计改前
    before_count = {}
    for txt in labels_dir.rglob("*.txt"):
        for line in open(txt):
            parts = line.strip().split()
            if parts:
                cid = int(parts[0])
                before_count[cid] = before_count.get(cid, 0) + 1
    print(f"改前各类别数量: {before_count}")

    # 处理：优先检查 train/val 子文件夹，否则直接处理当前目录
    total = 0
    subdirs_found = [s for s in ["train", "val"] if (labels_dir / s).exists()]
    if subdirs_found:
        for subdir in subdirs_found:
            subpath = labels_dir / subdir
            count = process_dir(subpath, target_classes, renumber)
            print(f" {subdir}/ 处理了 {count} 个文件")
            total += count
    else:
        count = process_dir(labels_dir, target_classes, renumber)
        print(f" 直接处理 {labels_dir.name}/ 下的文件，共 {count} 个")
        total += count

    # 统计改后
    after_count = {}
    for txt in labels_dir.rglob("*.txt"):
        for line in open(txt):
            parts = line.strip().split()
            if parts:
                cid = int(parts[0])
                after_count[cid] = after_count.get(cid, 0) + 1
    print(f"改后各类别数量: {after_count}")

    print(f"✅ 处理完成，共 {total} 个文件。已移除类别 ID: {sorted(list(target_classes))}")

if __name__ == "__main__":
    main()
