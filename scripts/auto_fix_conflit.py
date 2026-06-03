#!/usr/bin/env python3
"""
YOLO 冲突自动修复脚本 (基于全局数量的动态平衡策略)
功能：
0. 统计 Front/Rear 各自的初始类别数量
1. 删除完全重复的框
2. 扫描冲突：只要组内存在互斥类别则跳过，否则无论多少个类别都纳入处理池
3. 对于多类别同框冲突，保留当前全局数量最少的那个类别，删除其余所有类别
"""

import pathlib
import random
import shutil
import ast
import os
import glob
from collections import defaultdict, Counter

# ═══════════════════════════════════════════════════════════
# 互斥表
# ═══════════════════════════════════════════════════════════
MUTEX_PAIRS = {
    ('read', 'write'), ('discuss', 'read'), ('discuss', 'write'),
    # ('hand-raising', 'read'), ('hand-raising', 'write'),
    ('BowHead', 'TurnHead'), ('BowHead', 'discuss'), ('BowHead', 'hand-raising'),
    ('blackBoard', 'screen'),
}
# ═══════════════════════════════════════════════════════════

def parse_yolo(path):
    boxes = []
    if not path or not pathlib.Path(path).exists(): return boxes
    for line in pathlib.Path(path).read_text(encoding='utf-8', errors='replace').splitlines():
        parts = line.strip().split()
        if len(parts) != 5: continue
        cls, xc, yc, w, h = map(float, parts)
        boxes.append((int(cls), xc, yc, w, h))
    return boxes

def save_yolo(path, boxes):
    lines = [' '.join(str(x) for x in [b[0],b[1],b[2],b[3],b[4]]) for b in boxes]
    pathlib.Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')

def load_classes(labels_dir):
    p = pathlib.Path(labels_dir)
    for name in ['classes.txt', 'classes']:
        f = p.parent / name
        if f.exists():
            content = f.read_text(encoding='utf-8', errors='replace')
            last = None
            for line in content.splitlines():
                line = line.strip()
                if not line: continue
                try:
                    arr = ast.literal_eval(line)
                    if isinstance(arr, list): last = arr
                except: pass
            if last: return last
    return []

def is_mutex(class_name_a, class_name_b):
    return tuple(sorted((class_name_a, class_name_b))) in MUTEX_PAIRS

def has_any_mutex(class_names_list):
    """检查一个列表中是否存在任意两个互斥的类别"""
    for i in range(len(class_names_list)):
        for j in range(i+1, len(class_names_list)):
            if is_mutex(class_names_list[i], class_names_list[j]):
                return True
    return False

def deduplicate_boxes(boxes):
    seen = {}
    for b in boxes:
        key = (b[0], round(b[1], 6), round(b[2], 6), round(b[3], 6), round(b[4], 6))
        if key not in seen: seen[key] = b
    return list(seen.values())

def get_class_counts(label_dir):
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    counter = Counter()
    for file_path in label_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try: counter[int(parts[0])] += 1
                    except ValueError: pass
    return counter

def main():
    print('='*60)
    print('YOLO Auto Conflict Fixer (Dynamic Balance Strategy)')
    print('='*60)

    dataset = pathlib.Path(input('Dataset root: ').strip().strip('"').strip("'"))
    front_labels = dataset / 'Front' / 'labels'
    rear_labels = dataset / 'Rear' / 'labels'
    if not front_labels.exists() or not rear_labels.exists():
        print('Error: Front/labels or Rear/labels not found'); return

    front_classes = load_classes(front_labels)
    rear_classes = load_classes(rear_labels)
    CLASS_NAMES_PER_SIDE = {'Front': front_classes, 'Rear': rear_classes}
    LABEL_DIRS = {'Front': front_labels, 'Rear': rear_labels}

    # ━━━ 第零阶段：统计初始数量 ━━━
    print('\n[1/5] Counting initial class distribution...')
    init_counts = {}
    for side, ldir in LABEL_DIRS.items():
        counts_id = get_class_counts(str(ldir))
        class_names = CLASS_NAMES_PER_SIDE[side]
        print(f'\n  [{side}] 初始类别数量:')
        for cls_id, cnt in sorted(counts_id.items()):
            name = class_names[cls_id] if cls_id < len(class_names) else f'Unknown({cls_id})'
            init_counts[(side, name)] = cnt
            print(f'    - {name}: {cnt}')

    # ━━━ 第一阶段：去重 ━━━
    print('\n[2/5] Removing exact duplicate boxes...')
    dedup_files_count = 0
    for side, ldir in LABEL_DIRS.items():
        for lbl_file in pathlib.Path(ldir).glob('*.txt'):
            boxes = parse_yolo(str(lbl_file))
            if not boxes: continue
            new_boxes = deduplicate_boxes(boxes)
            if len(new_boxes) < len(boxes):
                backup = str(lbl_file) + '.dedup_bak'
                if not pathlib.Path(backup).exists(): shutil.copy2(lbl_file, backup)
                save_yolo(str(lbl_file), new_boxes)
                dedup_files_count += 1
    print(f'  -> Cleaned {dedup_files_count} files.')

    # ━━━ 第二阶段：扫描冲突 ━━━
    # 新的数据结构：支持任意长度的类别组合
    # Key: (side, tuple(sorted_class_names)), Value: [{'file':..., 'indices_dict':{name: idx}}, ...]
    pending_non_mutex = defaultdict(list)
    skip_mutex_count = 0

    print('\n[3/5] Scanning for conflicts...')
    for side, ldir in LABEL_DIRS.items():
        class_names = CLASS_NAMES_PER_SIDE[side]
        files = list(pathlib.Path(ldir).glob('*.txt'))
        print(f'  Scanning {side}...', end=' ', flush=True)
        
        for lbl_file in files:
            boxes = parse_yolo(str(lbl_file))
            coord_map = defaultdict(list)
            for i, b in enumerate(boxes):
                k = (round(b[1],4), round(b[2],4), round(b[3],4), round(b[4],4))
                coord_map[k].append(i)
            
            for coord, indices in coord_map.items():
                if len(indices) < 2: continue
                
                # 获取这组重叠框涉及的所有类别名称
                classes_in_group = [boxes[i][0] for i in indices]
                unique_cls_ids = list(set(classes_in_group))
                
                names_in_group = []
                for cid in unique_cls_ids:
                    n = class_names[cid] if cid < len(class_names) else f'Unknown({cid})'
                    names_in_group.append(n)
                
                # 核心改动：只要组内存在任意一对互斥，就跳过整个组
                if has_any_mutex(names_in_group):
                    skip_mutex_count += 1
                    continue
                
                # 不互斥，加入处理池
                indices_dict = {}
                for box_idx, cls_id in zip(indices, classes_in_group):
                    n = class_names[cls_id] if cls_id < len(class_names) else f'Unknown({cls_id})'
                    indices_dict[n] = box_idx
                
                sorted_tuple = tuple(sorted(names_in_group))
                pending_non_mutex[(side, sorted_tuple)].append({
                    'file': str(lbl_file),
                    'indices_dict': indices_dict
                })
        print('Done.')

    print(f'\n  -> Skipped (Contains Mutex Pair): {skip_mutex_count}')
    print(f'  -> To Balance: {sum(len(v) for v in pending_non_mutex.values())} conflict groups')

    if not pending_non_mutex:
        print('\nNo conflicts to fix.'); return

    # ━━━ 第三阶段：贪心平衡策略计算 ━━━
    print('\n[4/5] Calculating optimal balance strategy...')
    remove_plan = defaultdict(set)

    for (side, cls_tuple), instances in pending_non_mutex.items():
        # 获取该组内所有类别的当前全局数量
        cls_counts = {c: init_counts.get((side, c), 0) for c in cls_tuple}
        
        # 策略：保留当前数量【最少】的类别，删除其他所有的。
        # 这样能最大程度缩小大类和小类之间的差距。
        # 如果数量一样少，按字母顺序选第一个保证确定性。
        keep_class = min(cls_counts, key=lambda c: (cls_counts[c], c))
        
        # 统计信息
        counts_str = ", ".join([f"{c}({cls_counts[c]})" for c in cls_tuple])
        
        for inst in instances:
            for cls_name, idx in inst['indices_dict'].items():
                if cls_name != keep_class:
                    remove_plan[inst['file']].add(idx)
                    
        print(f'  [{side}] Conflict: [{", ".join(cls_tuple)}]')
        print(f'         Counts: {counts_str}')
        print(f'         Action: Keep ONLY "{keep_class}", delete others. (Total {len(instances)} cases)\n')

    # ━━━ 第四阶段：执行 ━━━
    print('[5/5] Applying fixes...')
    modified_files = 0
    total_removed = 0
    for file_path, indices_to_remove in remove_plan.items():
        boxes = parse_yolo(file_path)
        new_boxes = [b for i, b in enumerate(boxes) if i not in indices_to_remove]
        if len(new_boxes) != len(boxes):
            backup = file_path + '.balance_bak'
            if not pathlib.Path(backup).exists(): shutil.copy2(file_path, backup)
            save_yolo(file_path, new_boxes)
            modified_files += 1
            total_removed += len(indices_to_remove)

    print(f'\n✅ Done! Modified {modified_files} files, removed {total_removed} boxes in total.')

if __name__ == '__main__':
    main()
