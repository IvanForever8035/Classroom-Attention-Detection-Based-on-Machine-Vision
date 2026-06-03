#!/usr/bin/env python3
"""
YOLO 数据集 Train/Val 分割工具
用法：运行脚本后输入数据集根目录路径
数据集结构：
    <root>/
        images/  *.jpg
        labels/  *.txt (忽略 *.txt.bak)
"""

import random
import shutil
from pathlib import Path


def split_dataset(root: Path, val_ratio: float = 0.2, seed: int = 42):
    images_dir = root / "images"
    labels_dir = root / "labels"

    # 校验目录
    if not images_dir.exists():
        raise FileNotFoundError(f"找不到 images 目录：{images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"找不到 labels 目录：{labels_dir}")

    # 收集所有有对应 label 的图片（忽略 .txt.bak）
    all_images = sorted(images_dir.glob("*.jpg"))
    paired = []
    skipped = []
    for img in all_images:
        label = labels_dir / (img.stem + ".txt")
        if label.exists():
            paired.append(img.stem)
        else:
            skipped.append(img.name)

    if not paired:
        raise RuntimeError("未找到任何图片与标注的配对文件，请检查数据集。")

    if skipped:
        print(f"\n⚠️  以下 {len(skipped)} 张图片没有对应的 .txt 标注，已跳过：")
        for name in skipped[:10]:
            print(f"   {name}")
        if len(skipped) > 10:
            print(f"   ... 共 {len(skipped)} 个")

    # 随机打乱并分割
    random.seed(seed)
    random.shuffle(paired)
    val_count = max(1, round(len(paired) * val_ratio))
    val_stems = set(paired[:val_count])
    train_stems = set(paired[val_count:])

    print(f"\n📊 数据集总计：{len(paired)} 个样本")
    print(f"   train：{len(train_stems)}  |  val：{len(val_stems)}")
    print(f"   val 比例：{val_count / len(paired):.1%}\n")

    # 创建输出目录
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)

    # 复制文件
    def copy_pair(stem: str, split: str):
        shutil.copy2(images_dir / f"{stem}.jpg",
                     root / split / "images" / f"{stem}.jpg")
        shutil.copy2(labels_dir / f"{stem}.txt",
                     root / split / "labels" / f"{stem}.txt")

    for stem in train_stems:
        copy_pair(stem, "train")
    for stem in val_stems:
        copy_pair(stem, "val")

    print("✅ 分割完成！输出目录结构：")
    print(f"   {root}/")
    print(f"   ├── train/")
    print(f"   │   ├── images/  ({len(train_stems)} 张)")
    print(f"   │   └── labels/  ({len(train_stems)} 个)")
    print(f"   └── val/")
    print(f"       ├── images/  ({len(val_stems)} 张)")
    print(f"       └── labels/  ({len(val_stems)} 个)")


def main():
    print("=" * 50)
    print("   YOLO 数据集 Train/Val 分割工具")
    print("=" * 50)

    # 输入路径
    raw = input("\n请输入数据集根目录路径：").strip().strip('"').strip("'")
    root = Path(raw).expanduser().resolve()
    if not root.exists():
        print(f"❌ 路径不存在：{root}")
        return

    # 输入 val 比例
    ratio_input = input("请输入验证集比例（默认 0.2）：").strip()
    val_ratio = float(ratio_input) if ratio_input else 0.2
    if not 0 < val_ratio < 1:
        print("❌ 比例必须在 (0, 1) 之间")
        return

    # 输入随机种子
    seed_input = input("请输入随机种子（默认 42）：").strip()
    seed = int(seed_input) if seed_input else 42

    try:
        split_dataset(root, val_ratio=val_ratio, seed=seed)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"❌ 错误：{e}")


if __name__ == "__main__":
    main()
