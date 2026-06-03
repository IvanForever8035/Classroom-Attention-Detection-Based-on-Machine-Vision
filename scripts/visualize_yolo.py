#!/usr/bin/env python3
"""可视化 YOLO 标注，实时显示 bbox"""

import random
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CLASS_NAMES = ['guide', 'answer', 'On-stage interaction', 'blackboard-writing', 'stand', 'screen', 'blackBoard']

COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00',
    '#FF00FF', '#00FFFF', '#FF8000', '#8000FF',
]

def parse_class_names(raw):
    """解析类别名称列表输入"""
    raw = raw.strip()
    # 支持 "['a','b','c']" 或 "['a', 'b', 'c']" 或 "a, b, c" 等格式
    if raw.startswith('['):
        import ast
        try:
            return ast.literal_eval(raw)
        except:
            pass
    # 兜底：按逗号分隔
    return [x.strip().strip("'\"") for x in raw.split(',')]

def find_image(label_path, images_dir):
    """根据标注文件路径查找对应的图片"""
    img_name = label_path.stem + ".jpg"
    
    # 构建镜像路径结构：labels/train/xxx.txt -> images/train/xxx.jpg
    # label_path 如 labels/train/xxx.txt，images_dir 如 images/
    # 需要找 images/train/xxx.jpg
    
    # 获取相对于 labels_dir 的路径部分
    # 假设 labels 和 images 在同级目录
    label_parents = label_path.parts
    labels_idx = None
    for idx, p in enumerate(label_parents):
        if p == 'labels':
            labels_idx = idx
            break
    
    if labels_idx is not None:
        # 重建图片路径：把 labels 换成 images
        img_parts = list(label_parents)
        img_parts[labels_idx] = 'images'
        img_path = Path(*img_parts).with_suffix('.jpg')
        if img_path.exists():
            return img_path
    
    # 兜底：直接用 images_dir / stem
    img_path = images_dir / (label_path.stem.replace("labels", "images") + ".jpg")
    if img_path.exists():
        return img_path
    
    return None

def visualize(labels_dir, images_dir, class_names=None, sample=5):
    if class_names is None:
        class_names = []
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)

    label_files = list(labels_dir.rglob("*.txt"))
    random.shuffle(label_files)
    label_files = label_files[:sample]

    print(f"共 {len(label_files)} 张，随机抽 {sample} 张\n")

    for i, label_path in enumerate(label_files, 1):
        img_path = find_image(label_path, images_dir)
        
        if img_path is None or not img_path.exists():
            print(f"[{i}/{len(label_files)}] ⚠️  未找到图片: {label_path.stem}")
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        fig, ax = plt.subplots(1)
        ax.imshow(img)
        ax.set_title(f"{img_path.name}\n标注文件: {label_path.name}", fontsize=10)

        for line in open(label_path):
            parts = line.strip().split()
            if not parts:
                continue
            cid = int(parts[0])
            x_center, y_center, bw, bh = map(float, parts[1:])

            x1 = (x_center - bw / 2) * w
            y1 = (y_center - bh / 2) * h
            box_w = bw * w
            box_h = bh * h

            color = COLORS[cid % len(COLORS)]
            rect = patches.Rectangle(
                (x1, y1), box_w, box_h,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            name = class_names[cid] if cid < len(class_names) else f"c{cid}"
            ax.text(x1 + 3, y1 + 12, name,
                    color='white', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))

        ax.axis('off')
        print(f"[{i}/{len(label_files)}] {img_path.name}")
        plt.draw()
        plt.pause(0.5)
        print("    按图形窗口的 X 关闭，或等下张自动显示...")
        plt.show(block=True)
        plt.close()

    print("\n✅ 检查完成")

def main():
    labels_dir = input("标注文件夹路径: ").strip().strip('"').strip("'")
    images_dir = input("图片文件夹路径（与标注结构对应）: ").strip().strip('"').strip("'")
    class_names_raw = input("类别名称列表（如 ['guide','answer',...]）: ").strip()
    class_names = parse_class_names(class_names_raw) if class_names_raw else []
    sample = int(input("随机抽检数量（默认 5）: ").strip() or "5")

    print()
    print(f"类别: {class_names}\n")
    visualize(labels_dir, images_dir, class_names, sample)

if __name__ == "__main__":
    main()
