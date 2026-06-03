#!/usr/bin/env python3
"""检查多个图片文件夹之间是否有重复图片（按文件名）"""

from pathlib import Path
from collections import defaultdict

def collect_images(folder_path):
    """收集文件夹下所有图片文件名（不含后缀）"""
    folder = Path(folder_path)
    if not folder.exists():
        return set(), {}
    
    # 支持的图片扩展名
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif', '.tiff'}
    names = set()
    name_to_ext = {}
    
    for f in folder.rglob("*"):
        if f.is_file() and f.suffix.lower() in exts:
            name = f.stem  # 不含后缀的文件名
            names.add(name)
            name_to_ext[name] = f.suffix.lower()
    
    return names, name_to_ext

def main():
    print("=== 图片重复检查工具 ===\n")
    print("输入图片文件夹路径（Windows路径），输入空行结束输入：\n")
    
    folders = []
    while True:
        raw = input(f"文件夹 {len(folders) + 1} 路径: ").strip().strip('"').strip("'")
        if not raw:
            break
        folders.append(raw)
    
    if len(folders) < 2:
        print("\n❌ 需要至少两个文件夹才能比较")
        return
    
    print(f"\n正在收集 {len(folders)} 个文件夹的图片...\n")
    
    # 收集每个文件夹的图片名
    folder_names = []
    all_names = []
    for i, fp in enumerate(folders):
        names, _ = collect_images(fp)
        folder_names.append(names)
        all_names.extend(names)
        print(f"  [{i}] {fp}")
        print(f"      图片数量: {len(names)}")
    
    # 统计重复：哪些名字在多个文件夹中都出现
    name_to_folders = defaultdict(list)
    for i, names in enumerate(folder_names):
        for name in names:
            name_to_folders[name].append(i)
    
    # 找出有重复的
    duplicates = {name: idxs for name, idxs in name_to_folders.items() if len(idxs) > 1}
    
    if not duplicates:
        print("\n✅ 未发现重复图片")
        return
    
    print(f"\n⚠️  发现 {len(duplicates)} 张图片在多个文件夹中重复：\n")
    
    # 按文件夹对统计
    pair_count = defaultdict(set)  # (i,j) -> set of duplicate names
    
    for name, idxs in duplicates.items():
        for ii in range(len(idxs)):
            for jj in range(ii + 1, len(idxs)):
                i, j = idxs[ii], idxs[jj]
                key = (min(i, j), max(i, j))
                pair_count[key].add(name)
    
    # 输出详情
    for (i, j), names in sorted(pair_count.items()):
        print(f"文件夹 [{i}] 与 文件夹 [{j}] 重复: {len(names)} 张")
        print(f"  路径A: {folders[i]}")
        print(f"  路径B: {folders[j]}")
        print(f"  重复图片名（部分）: {list(names)[:5]}")
        if len(names) > 5:
            print(f"  ... 共 {len(names)} 张")
        print()
    
    # 按文件夹统计重复数量
    print("=== 各文件夹重复统计 ===")
    dup_count_per_folder = defaultdict(int)
    for name, idxs in duplicates.items():
        for idx in idxs:
            dup_count_per_folder[idx] += 1
    
    for idx in sorted(dup_count_per_folder):
        print(f"文件夹 [{idx}]: {dup_count_per_folder[idx]} 张重复（与其他文件夹共享）")
        print(f"  {folders[idx]}")

if __name__ == "__main__":
    main()
