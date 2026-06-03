#!/usr/bin/env python3
"""
YOLO 标注空文件检测工具
用法：运行脚本后输入 labels 目录路径（或数据集根目录）
"""

from pathlib import Path


def check_empty_labels(labels_dir: Path):
    # 只检测 .txt，忽略 .txt.bak
    all_labels = sorted(labels_dir.glob("*.txt"))

    if not all_labels:
        print("⚠️  该目录下未找到任何 .txt 文件。")
        return

    empty = []
    for label in all_labels:
        if label.stat().st_size == 0:
            empty.append(label)

    total = len(all_labels)
    print(f"\n📂 共检测 {total} 个 .txt 文件")

    if not empty:
        print("✅ 全部正常，没有空标注文件。")
    else:
        print(f"❌ 发现 {len(empty)} 个空标注文件（占 {len(empty)/total:.1%}）：\n")
        for f in empty:
            print(f"   {f.name}")


def main():
    print("=" * 50)
    print("   YOLO 空标注文件检测工具")
    print("=" * 50)

    raw = input("\n请输入 labels 目录路径（或数据集根目录）：").strip().strip('"').strip("'")
    path = Path(raw).expanduser().resolve()

    if not path.exists():
        print(f"❌ 路径不存在：{path}")
        return

    # 如果输入的是数据集根目录，自动定位 labels/
    if (path / "labels").exists():
        labels_dir = path / "labels"
        print(f"📁 自动定位到：{labels_dir}")
    elif path.name == "labels" or any(path.glob("*.txt")):
        labels_dir = path
    else:
        print("❌ 未找到 labels 目录，请直接输入 labels 文件夹路径。")
        return

    check_empty_labels(labels_dir)


if __name__ == "__main__":
    main()
