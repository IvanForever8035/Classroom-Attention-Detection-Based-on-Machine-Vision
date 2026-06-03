#!/usr/bin/env python3
""" YOLO Annotation Conflict Fixer - 支持多元组冲突解决 
- 二元冲突：红/绿交替，选A或B
- 多元冲突(>=3)：红/绿/蓝多色区分，只需点选保留其中一个，其余自动删除
Requires: PyQt5, matplotlib 
Install: pip install PyQt5 matplotlib 
"""

import shutil, ast, gc, sys, ctypes, pathlib, os, json, numpy as np

# ── Dependency check ──────────────────────────────────────
def check_deps():
    missing = []
    for mod in ['PyQt5', 'matplotlib']:
        try: __import__(mod.lower().replace('pyqt5', 'PyQt5'))
        except ImportError: missing.append(mod)
    return missing

def install_deps():
    print('[INFO] Installing: PyQt5 matplotlib...')
    os.system(f'{sys.executable} -m pip install PyQt5 matplotlib -q')
    print('[INFO] Done. Run the script again.')
    sys.exit(0)

if check_deps(): install_deps()

from PyQt5 import QtWidgets, QtCore, QtGui
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

# ── Config ───────────────────────────────────────────────
IMG_EXTS = ['.jpg', '.png', '.jpeg']
CONFLICT_COLORS = ['#FF4444','#44FF44','#4444FF','#FFFF00','#FF00FF'] # 用于多元冲突的调色板

# ── Disable matplotlib keymap ────────────────────────────
for _k in list(plt.rcParams):
    if _k.startswith('keymap.'): plt.rcParams[_k] = []

# ── Force window to foreground (Windows) ────────────────
def force_foreground(hwnd):
    try:
        u32 = ctypes.windll.user32
        cur_tid = ctypes.windll.kernel32.GetCurrentThreadId()
        fg_tid = u32.GetWindowThreadProcessId(u32.GetForegroundWindow(), None)
        attached = fg_tid and fg_tid != cur_tid and u32.AttachThreadInput(cur_tid, fg_tid, True)
        u32.SetForegroundWindow(hwnd); u32.BringWindowToTop(hwnd)
        if attached: u32.AttachThreadInput(cur_tid, fg_tid, False)
    except Exception: pass

# ── Helpers ─────────────────────────────────────────────
def load_classes(front_labels_dir):
    p = pathlib.Path(front_labels_dir)
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

def find_conflicts(boxes):
    """返回冲突列表。
    二元冲突格式: (ia, ib, cla, clb)
    多元冲突格式: ((ia, ib, ic), (cla, clb, clc))  # 用元组嵌套表示
    """
    from collections import defaultdict
    mp = defaultdict(list)
    for i, b in enumerate(boxes):
        k = (round(b[1],4), round(b[2],4), round(b[3],4), round(b[4],4))
        mp[k].append((b[0], i))
    
    conflicts = []
    for k, entries in mp.items():
        if len(entries) < 2: continue
        # 去除同坐标同类的重复框
        unique_entries = {}
        for c, idx in entries:
            if c not in unique_entries:
                unique_entries[c] = idx
                
        if len(unique_entries) < 2: continue
        
        if len(unique_entries) == 2:
            (c1, i1), (c2, i2) = list(unique_entries.items())
            conflicts.append((i1, i2, c1, c2)) # 保持旧的二元格式
        else:
            # 3个或更多类别：打包成多元组
            indices = tuple(unique_entries.values())
            cls_list = tuple(unique_entries.keys())
            conflicts.append((indices, cls_list))
    return conflicts

def get_img_path(stem, img_dirs):
    for idir in img_dirs:
        for ext in IMG_EXTS:
            p = pathlib.Path(idir) / f'{stem}{ext}'
            if p.exists(): return p
    return None

# ── Draw conflict boxes ─────────────────────────────────
def draw_conflict_boxes(ax, boxes, conflicts, selections, class_names, W, H):
    box_info = {}
    for ci, conf in enumerate(conflicts):
        if isinstance(conf[0], tuple): # 多元冲突
            indices, cls_list = conf
            for k, idx in enumerate(indices):
                which = chr(65 + k) # 'A', 'B', 'C'...
                box_info[idx] = (ci, indices, cls_list, which, 'multi')
        else: # 二元冲突
            ia, ib, cla, clb = conf
            box_info[ia] = (ci, (ia, ib), (cla, clb), 'a', 'dual')
            box_info[ib] = (ci, (ia, ib), (cla, clb), 'b', 'dual')
            
    conflict_set = set(box_info.keys())

    for i, b in enumerate(boxes):
        cls_idx, xc, yc, bw, bh = int(b[0]), b[1], b[2], b[3], b[4]
        x1, y1 = (xc - bw/2)*W, (yc - bh/2)*H
        x2, y2 = (xc + bw/2)*W, (yc + bh/2)*H

        if i in conflict_set:
            ci, indices, cls_list, which, ctype = box_info[i]
            sel = selections.get(ci)
            
            if sel is None: # 未选择状态
                if ctype == 'multi':
                    color_idx = ord(which) - 65
                    inner_color = CONFLICT_COLORS[color_idx % len(CONFLICT_COLORS)]
                else:
                    inner_color = '#FF3333' if ci % 2 == 0 else '#33FF33'
                glow_color = '#FFFFFF'; lw_inner = 3; lw_glow = 8
            elif sel == which: # 被保留
                inner_color = '#00FF00'; glow_color = '#FFFFFF'; lw_inner = 3; lw_glow = 8
            else: # 被删除
                inner_color = '#990000'; glow_color = '#FF6666'; lw_inner = 2; lw_glow = 5

            ax.add_patch(Rectangle((x1, y1), bw*W, bh*H, linewidth=lw_glow, edgecolor=glow_color, facecolor='none'))
            ax.add_patch(Rectangle((x1, y1), bw*W, bh*H, linewidth=lw_inner, edgecolor=inner_color, facecolor='none'))
            
            if sel is not None and sel != which:
                cy = (y1 + y2) / 2
                ax.plot([x1, x2], [cy, cy], color='#FF0000', linewidth=2, linestyle='--')
        else:
            color = '#999999'; lw = 1
            ax.add_patch(Rectangle((x1, y1), bw*W, bh*H, linewidth=lw, edgecolor=color, facecolor='none'))

        name = class_names[cls_idx] if cls_idx < len(class_names) else f'c{cls_idx}'
        lbl_color = '#888888'
        
        if i in conflict_set:
            ci, indices, cls_list, which, ctype = box_info[i]
            sel = selections.get(ci)
            if sel is None:
                mark = ''
                if ctype == 'multi':
                    color_idx = ord(which) - 65
                    lbl_color = CONFLICT_COLORS[color_idx % len(CONFLICT_COLORS)]
                else:
                    lbl_color = '#FF3333' if ci % 2 == 0 else '#33FF33'
            elif sel == which:
                mark = ' [KEEP]'; lbl_color = '#00FF00'
            else:
                mark = ' [DEL]'; lbl_color = '#990000'
            
            ci_label = box_info[i][0]
            which_lbl = box_info[i][3]
            prefix = f'[{ci_label+1}{which_lbl.upper()}] '
            name = prefix + name + mark
            
        ax.text(x1+2, y1+10, name, color=lbl_color, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

# ── Main window ─────────────────────────────────────────
def show_conflict_window(img_path, stem, class_names, lbl_path, boxes, conflicts, selections, seq_num, remaining, qapp):
    choice = []; history = []; show_boxes = True; _closing = False
    fig, ax = plt.subplots(1, figsize=(14, 9))
    img_data = mpimg.imread(str(img_path))
    if img_data.ndim == 2: img_data = np.stack([img_data]*3, axis=-1)
    elif img_data.shape[-1] == 4: img_data = img_data[:,:,:3]
    H, W = img_data.shape[:2]
    ax.imshow(img_data)
    draw_conflict_boxes(ax, boxes, conflicts, selections, class_names, W, H)

    def status_text():
        done = len(selections); total = len(conflicts); pending = total - done
        if pending == 0: return f' ALL DONE — close window to save and go next'
        return (f' {done}/{total} done | Left-click=keep | Right-click=undo | s=skip q=quit')

    ax.set_title(f'#{seq_num} (remaining: {remaining}) {stem}', fontsize=11, loc='left')
    ax.axis('off'); fig.tight_layout(); fig.canvas.draw()
    win = QtWidgets.QMainWindow(); win.setCentralWidget(fig.canvas)
    win.setWindowTitle(f'Conflict Fix #{seq_num} - {stem} [{img_path}]')
    win.resize(1200, 800); win.statusBar().showMessage(status_text())

    def refresh():
        if _closing: return
        ax.cla(); ax.imshow(img_data)
        if show_boxes: draw_conflict_boxes(ax, boxes, conflicts, selections, class_names, W, H)
        box_hint = ' [boxes hidden]' if not show_boxes else ''
        ax.set_title(f'#{seq_num} (remaining: {remaining}) {stem}{box_hint}', fontsize=11, loc='left')
        ax.axis('off'); fig.tight_layout(); fig.canvas.draw()
        win.statusBar().showMessage(status_text())

    def hit_test_all(xdata, ydata):
        hits = []
        for ci, conf in enumerate(conflicts):
            if isinstance(conf[0], tuple): # multi
                indices, cls_list = conf
                for k, idx in enumerate(indices):
                    which = chr(65 + k)
                    b = boxes[idx]
                    x1, y1 = (b[1]-b[3]/2)*W, (b[2]-b[4]/2)*H
                    x2, y2 = (b[1]+b[3]/2)*W, (b[2]+b[4]/2)*H
                    if x1 <= xdata <= x2 and y1 <= ydata <= y2: hits.append((ci, which))
            else: # dual
                ia, ib, cla, clb = conf
                for box_idx, which in [(ia, 'a'), (ib, 'b')]:
                    b = boxes[box_idx]
                    x1, y1 = (b[1]-b[3]/2)*W, (b[2]-b[4]/2)*H
                    x2, y2 = (b[1]+b[3]/2)*W, (b[2]+b[4]/2)*H
                    if x1 <= xdata <= x2 and y1 <= ydata <= y2: hits.append((ci, which))
        return hits

    def select_box(ci, which):
        selections[ci] = which
        conf = conflicts[ci]
        if isinstance(conf[0], tuple): 
            cls_kept = conf[1][ord(which) - 65]
        else: 
            cls_kept = conf[2] if which == 'a' else conf[3]
        name = class_names[cls_kept] if cls_kept < len(class_names) else f'c{cls_kept}'
        history.append(dict(selections))
        print(f'   conflict{ci+1}: keep [{which.upper()}] {name}', flush=True)
        refresh()
        if len(selections) == len(conflicts):
            _closing = True; choice.append('done'); win.close()

    def on_click(event):
        if event.xdata is None or event.ydata is None: return
        x, y = event.xdata, event.ydata
        if event.button == 3:
            if history:
                selections.clear(); selections.update(history.pop())
                print(f'   undo', flush=True); refresh()
            return

        hits = hit_test_all(x, y)
        if not hits: return

        if len(hits) == 1:
            select_box(*hits[0])
        else:
            dialog = QtWidgets.QDialog(win)
            dialog.setWindowTitle('Choose option')
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(QtWidgets.QLabel(f'<b>Found {len(hits)} overlapping boxes, choose one to KEEP:</b>'))
            for ci, which in hits:
                conf = conflicts[ci]
                if isinstance(conf[0], tuple): cls_idx = conf[1][ord(which) - 65]
                else: cls_idx = conf[2] if which == 'a' else conf[3]
                name = class_names[cls_idx] if cls_idx < len(class_names) else f'c{cls_idx}'
                btn = QtWidgets.QPushButton(f'Conflict {ci+1} [{which.upper()}] {name}')
                btn.clicked.connect(lambda _, c=ci, w=which: [select_box(c, w), dialog.accept()])
                layout.addWidget(btn)
            cancel_btn = QtWidgets.QPushButton('Cancel')
            cancel_btn.clicked.connect(dialog.reject)
            layout.addWidget(cancel_btn)
            dialog.setLayout(layout); dialog.exec_()
            refresh() # 修复原版这里不刷新的Bug
            if len(selections) == len(conflicts):
                _closing = True; choice.append('done'); win.close()

    def on_key(event):
        key_str = event.key.lower()
        if key_str == 'q': _closing = True; choice.append('quit'); win.close()
        elif key_str == 's': _closing = True; choice.append('skip'); win.close()
        elif key_str == 'h':
            nonlocal show_boxes; show_boxes = not show_boxes; refresh()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.setFocus()
    win.show(); win.raise_(); win.activateWindow()
    QtCore.QTimer.singleShot(50, lambda: force_foreground(int(win.winId())))
    print(f'\n#{seq_num} (remaining: {remaining}) {stem}', flush=True)
    qapp.exec_(); plt.close(fig); gc.collect()
    return choice[0] if choice else 'skip'

def main():
    print('='*60); print('YOLO Annotation Conflict Fixer'); print('='*60)
    dataset = pathlib.Path(input('Dataset root (contains Front/Rear subdirs): ').strip().strip('"').strip("'"))
    conflict_json_input = input('Conflict JSON path (Enter to skip/rescan): ').strip().strip('"').strip("'")
    front_labels = dataset / 'Front' / 'labels'; rear_labels = dataset / 'Rear' / 'labels'
    front_images = dataset / 'Front' / 'images'; rear_images = dataset / 'Rear' / 'images'
    if not front_labels.exists() or not rear_labels.exists(): print('Error: Labels not found'); return

    front_classes = load_classes(front_labels); rear_classes = load_classes(rear_labels)
    if not front_classes: front_classes = ['discuss','hand-raising','read','write','BowHead','TurnHead', 'answer','On-stage interaction','stand','screen','blackBoard']
    if not rear_classes: rear_classes = front_classes
    print(f'Front classes ({len(front_classes)}): {front_classes}'); print(f'Rear classes ({len(rear_classes)}): {rear_classes}\n')
    CLASS_NAMES_PER_SIDE = {'Front': front_classes, 'Rear': rear_classes}
    LABEL_DIRS = {'Front': str(front_labels), 'Rear': str(rear_labels)}
    IMG_DIRS = {'Front': str(front_images), 'Rear': str(rear_images)}

    # ── Scan / load conflicts ────────────────────────────
    conflict_pair_counts = {'Front': {}, 'Rear': {}}
    if conflict_json_input and pathlib.Path(conflict_json_input).exists():
        print(f'Loading from JSON: {conflict_json_input}')
        raw = json.loads(pathlib.Path(conflict_json_input).read_text(encoding='utf-8'))
        for item in raw:
            side = item.get('side', '')
            if side in conflict_pair_counts:
                for conf in item.get('conflicts', []):
                    if isinstance(conf[0], tuple):
                        cls_list = conf[1]
                        for i in range(len(cls_list)):
                            for j in range(i+1, len(cls_list)):
                                pair = tuple(sorted((cls_list[i], cls_list[j])))
                                conflict_pair_counts[side][pair] = conflict_pair_counts[side].get(pair, 0) + 1
                    else:
                        pair = tuple(sorted((conf[2], conf[3])))
                        conflict_pair_counts[side][pair] = conflict_pair_counts[side].get(pair, 0) + 1
    else:
        print('Scanning conflicts...'); raw = []
        for side, ldir in LABEL_DIRS.items():
            files = sorted(pathlib.Path(ldir).glob('*.txt'))
            print(f' {side}: {len(files)} files...', end=' ', flush=True); count = 0
            for lbl_file in files:
                boxes = parse_yolo(str(lbl_file))
                if len(boxes) >= 2:
                    confs = find_conflicts(boxes)
                    if confs:
                        raw.append({'fname': lbl_file.name, 'side': side, 'label_path': str(lbl_file), 'conflicts': confs})
                        count += 1
                        for conf in confs:
                            if isinstance(conf[0], tuple):
                                cls_list = conf[1]
                                for i in range(len(cls_list)):
                                    for j in range(i+1, len(cls_list)):
                                        pair = tuple(sorted((cls_list[i], cls_list[j])))
                                        conflict_pair_counts[side][pair] = conflict_pair_counts[side].get(pair, 0) + 1
                            else:
                                pair = tuple(sorted((conf[2], conf[3])))
                                conflict_pair_counts[side][pair] = conflict_pair_counts[side].get(pair, 0) + 1
            print(f'found {count}')
    print(f'Total: {len(raw)} conflict files')

    # ── 打印冲突详细统计 ────────────────────────────────
    print('\n' + '='*50); print('Conflict Pair Statistics (Which vs Which):'); print('='*50)
    for side in ['Front', 'Rear']:
        stats = conflict_pair_counts.get(side, {})
        if not stats: continue
        class_names = CLASS_NAMES_PER_SIDE.get(side, [])
        print(f'\n[{side}]')
        for (c1, c2), cnt in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            name1 = class_names[c1] if c1 < len(class_names) else f'Unknown(c{c1})'
            name2 = class_names[c2] if c2 < len(class_names) else f'Unknown(c{c2})'
            print(f' - "{name1}" <--> "{name2}": {cnt} 个冲突')
    print('='*50 + '\n')

    # ── Build working list ───────────────────────────────
    all_items = []
    for item in raw:
        fname = item['fname']; side = item['side']; ldir = LABEL_DIRS.get(side)
        if not ldir: continue
        label_path = pathlib.Path(ldir) / fname
        if not label_path.exists(): continue
        boxes = parse_yolo(str(label_path)); confs = find_conflicts(boxes)
        if not confs: continue
        stem = fname.replace('.txt', '')
        img_path = get_img_path(stem, [IMG_DIRS[side]])
        all_items.append({'fname': fname, 'side': side, 'stem': stem, 'label_path': str(label_path),
                          'img_path': str(img_path) if img_path else None, 'boxes': boxes, 'conflicts': confs})
    all_items.sort(key=lambda x: (-len(x['conflicts']), x['fname']))
    total = len(all_items); print(f'{total} conflict files to process\n')
    if total == 0: print('No conflict files found.'); return

    # ── PyQt5 app ───────────────────────────────────────
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    idx = 0; done = 0
    while idx < total:
        item = all_items[idx]; boxes = item['boxes']; conflicts = item['conflicts']
        label_path = item['label_path']; img_path = item['img_path']
        if img_path is None or not pathlib.Path(img_path).exists():
            print(f'Warning: image not found for {item["stem"]}, skipping'); idx += 1; continue
        selections = {}
        side_names = CLASS_NAMES_PER_SIDE.get(item['side'], front_classes)
        result = show_conflict_window(img_path, item['stem'], side_names, label_path, boxes, conflicts, selections, done+1, total - idx, qapp)
        
        if result == 'quit': print('\nQuit.'); return
        elif result == 'skip': print(f' -> skip\n'); idx += 1; continue
        elif result == 'done':
            keep_set = set(); conflict_box_set = set()
            for ci, conf in enumerate(conflicts):
                which = selections.get(ci)
                if isinstance(conf[0], tuple): # multi
                    indices, cls_list = conf
                    for idx_in_conf in indices: conflict_box_set.add(idx_in_conf)
                    if which: keep_set.add(indices[ord(which) - 65])
                else: # dual
                    ia, ib, cla, clb = conf
                    conflict_box_set.add(ia); conflict_box_set.add(ib)
                    if which == 'a': keep_set.add(ia)
                    elif which == 'b': keep_set.add(ib)
            
            for i, b in enumerate(boxes):
                if i not in conflict_box_set: keep_set.add(i)
            new_boxes = [boxes[i] for i in sorted(keep_set)]
            backup = label_path + '.bak'
            if not pathlib.Path(backup).exists(): shutil.copy2(label_path, backup)
            save_yolo(label_path, new_boxes)
            print(f' saved: {len(boxes)} -> {len(new_boxes)} backup: {backup}')
            done += 1; idx += 1
            
            new_boxes_p = parse_yolo(label_path); new_confs = find_conflicts(new_boxes_p)
            all_items[idx-1]['boxes'] = new_boxes_p; all_items[idx-1]['conflicts'] = new_confs
            if new_confs:
                print(f' warning: still {len(new_confs)} new conflicts, re-editing...')
                conflicts = new_confs # loop will reprocess
    print(f'\nDone! {done} files processed. Backup: *.bak')

if __name__ == '__main__': main()
