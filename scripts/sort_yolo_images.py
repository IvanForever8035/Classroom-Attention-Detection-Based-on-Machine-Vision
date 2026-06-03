#!/usr/bin/env python3
"""YOLO dataset image sorting tool: classify images into front/rear, show bboxes and class labels"""

import shutil
import ast
import gc
import sys
import ctypes
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SPLITS     = ["train", "val"]
CATEGORIES = ["front", "rear", "skip"]
IMG_EXTS   = [".jpg", ".png", ".jpeg"]
BOX_COLORS = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00',
              '#FF00FF', '#00FFFF', '#FF8000', '#8000FF']
VALID_KEYS = {'f', 'r', 's', 'n', 'b'}


# ── 禁用 matplotlib 快捷键（只需执行一次） ────────────────────────────────────
for _k in list(plt.rcParams):
    if _k.startswith("keymap."):
        plt.rcParams[_k] = []


def force_foreground(hwnd: int) -> None:
    """借用前台线程权限，将窗口可靠地提升到最前（仅 Windows 有效）。"""
    try:
        u32     = ctypes.windll.user32
        cur_tid = ctypes.windll.kernel32.GetCurrentThreadId()
        fg_tid  = u32.GetWindowThreadProcessId(u32.GetForegroundWindow(), None)
        attached = fg_tid and fg_tid != cur_tid and u32.AttachThreadInput(cur_tid, fg_tid, True)
        u32.SetForegroundWindow(hwnd)
        u32.BringWindowToTop(hwnd)
        if attached:
            u32.AttachThreadInput(cur_tid, fg_tid, False)
    except Exception:
        pass


def _parse_classes_text(content: str) -> list:
    """从文本内容解析类名列表（支持 Python 列表格式或每行一个）。"""
    content = content.strip()
    if content.startswith("["):
        try:
            return ast.literal_eval(content)
        except Exception:
            pass
    return [l.strip() for l in content.splitlines() if l.strip()]


def load_classes(dataset_path: Path) -> list:
    """从数据集目录加载类名。"""
    for name in ["classes.txt", "classes", "classes.json"]:
        f = dataset_path / name
        if f.exists():
            result = _parse_classes_text(f.read_text())
            if result:
                return result
    return []


def iter_items(labels_base: Path):
    """逐条生成数据集条目 {stem, split, img, lbl}。"""
    images_base = labels_base.parent / "images"
    for split in SPLITS:
        split_dir = labels_base / split
        if not split_dir.exists():
            continue
        for lbl_file in sorted(split_dir.glob("*.txt")):
            stem = lbl_file.stem
            img_file = next(
                (p for ext in IMG_EXTS if (p := images_base / split / (stem + ext)).exists()),
                None
            )
            yield {"stem": stem, "split": split, "img": img_file, "lbl": lbl_file}


def get_processed_stems(dataset_path: Path) -> set:
    """收集已处理图片的 stem 集合。"""
    processed = set()
    for cat in CATEGORIES:
        for split in SPLITS:
            img_dir = dataset_path / cat / "images" / split
            if img_dir.exists():
                processed.update(
                    f.stem for f in img_dir.glob("*")
                    if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}
                )
    return processed


def get_label_classes(lbl_path: Path, class_names: list) -> list:
    """返回 label 文件中出现的类名列表。"""
    if not lbl_path or not lbl_path.exists():
        return []
    cids = {int(line.split()[0]) for line in lbl_path.open() if line.strip()}
    return [class_names[c] if class_names and c < len(class_names) else str(c)
            for c in sorted(cids)]


def draw_boxes(ax, lbl_path: Path, class_names: list, w: int, h: int) -> None:
    """在 axes 上绘制 YOLO 格式的 bounding box。"""
    if not lbl_path or not lbl_path.exists():
        return
    for line in lbl_path.open():
        parts = line.strip().split()
        if not parts:
            continue
        cid = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])
        x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
        color = BOX_COLORS[cid % len(BOX_COLORS)]
        ax.add_patch(Rectangle((x1, y1), bw * w, bh * h,
                                linewidth=2, edgecolor=color, facecolor='none'))
        label = class_names[cid] if class_names and cid < len(class_names) else str(cid)
        ax.text(x1 + 3, y1 + 12, label, color='white', fontsize=8,
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.85))


def show_and_choose(img_path, stem, class_labels, lbl_path, class_names,
                    seq_num, total_count, qapp) -> str:
    """显示图片窗口，等待用户按键，返回选择 (f/r/s/n/b)。"""
    from PyQt5 import QtWidgets, QtCore

    choice = []   # 用单元素列表代替 queue，exec_() 阻塞期间写入后关窗即可

    fig, ax = plt.subplots(1, figsize=(14, 9))
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    ax.imshow(img)
    del img
    draw_boxes(ax, lbl_path, class_names, w, h)

    title_parts = [f"#{seq_num} (remaining: {total_count})  {stem}  —  {img_path.name}"]
    if class_labels:
        title_parts.append(f"Classes: {', '.join(class_labels)}")
    title_parts.append("(F)ront  (R)ear  (S)kip  (N)othing  (B)ack")
    ax.set_title("\n".join(title_parts), fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw()

    win = QtWidgets.QMainWindow()
    win.setCentralWidget(fig.canvas)
    win.setWindowTitle(title_parts[0])
    win.resize(1200, 800)
    win.statusBar().showMessage(
        "  F = Front    R = Rear    S = Skip    N = Nothing    B = Back (undo)"
    )

    class KeyFilter(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.KeyPress:
                key = event.text().lower()
                if key in VALID_KEYS:
                    choice.append(key)
                    win.close()
                    return True
            return False

    key_filter = KeyFilter()
    fig.canvas.installEventFilter(key_filter)
    fig.canvas.setFocusPolicy(QtCore.Qt.StrongFocus)

    win.show()
    win.raise_()
    win.activateWindow()
    fig.canvas.setFocus()
    QtCore.QTimer.singleShot(50, lambda: force_foreground(int(win.winId())))

    print(f"\n#{seq_num} (remaining: {total_count})  {stem}  —  press a key in the image window", flush=True)
    qapp.exec_()

    plt.close(fig)
    gc.collect()
    return choice[0] if choice else "n"  # 直接关窗视为 n


def main():
    dataset_path = Path(
        input("YOLO dataset path (parent of images/ and labels/): ")
        .strip().strip('"').strip("'")
    )
    labels_base = dataset_path / "labels"

    class_names = load_classes(dataset_path)
    print(f"Classes loaded: {class_names}\n", flush=True)

    for cat in CATEGORIES:
        for split in SPLITS:
            (dataset_path / cat / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_path / cat / "labels" / split).mkdir(parents=True, exist_ok=True)

    all_items = list(iter_items(labels_base))
    if not all_items:
        print("Error: no label files found.")
        return

    processed = get_processed_stems(dataset_path)
    remaining = len(all_items) - len(processed)
    print(f"Total: {len(all_items)}, already processed: {len(processed)}, remaining: {remaining}\n",
          flush=True)
    if remaining <= 0:
        print("All images already processed.")
        return

    from PyQt5 import QtWidgets
    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    done    = 0
    history = []   # (stem, split, cat, idx)  cat="_skip_" 表示 n 操作
    idx     = 0

    while idx < len(all_items):
        item  = all_items[idx]
        stem  = item["stem"]
        split = item["split"]
        img_path = item["img"]
        lbl_path = item["lbl"]

        if stem in processed:
            idx += 1
            continue
        if img_path is None or not img_path.exists():
            print(f"Warning: image not found for {stem}, skipping", flush=True)
            idx += 1
            continue

        class_labels = get_label_classes(lbl_path, class_names)
        choice = show_and_choose(img_path, stem, class_labels, lbl_path, class_names,
                                 done + 1, len(all_items) - len(processed), qapp)

        if choice == "b":
            if not history:
                print("  Nothing to undo\n", flush=True)
                continue
            last_stem, last_split, last_cat, last_idx = history.pop()
            if last_cat != "_skip_":
                for ext in IMG_EXTS:
                    (dataset_path / last_cat / "images" / last_split / (last_stem + ext)).unlink(missing_ok=True)
                (dataset_path / last_cat / "labels" / last_split / (last_stem + ".txt")).unlink(missing_ok=True)
                done -= 1
            processed.discard(last_stem)
            idx = last_idx
            print(f"  -> undone: {last_stem} (was {last_cat})\n", flush=True)
            gc.collect()
            continue

        if choice == "n":
            history.append((stem, split, "_skip_", idx))
            print(f"  -> nothing (n)\n", flush=True)
            idx += 1
            continue

        cat = {"f": "front", "r": "rear", "s": "skip"}[choice]
        shutil.copy2(img_path, dataset_path / cat / "images" / split / img_path.name)
        shutil.copy2(lbl_path, dataset_path / cat / "labels" / split / lbl_path.name)
        print(f"  -> {cat}/{split}/\n", flush=True)
        done += 1
        processed.add(stem)
        history.append((stem, split, cat, idx))
        idx += 1

    print(f"\nDone. Total classified this session: {done} images", flush=True)


if __name__ == "__main__":
    main()