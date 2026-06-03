from pathlib import Path
import shutil
import random
import os

def undersample_stand(label_dir, target_count=4000, seed=42):
    """
    把 stand 类样本减少到 target_count 个左右
    """
    random.seed(seed)
    label_dir = Path(label_dir)
    
    stand_files = []
    other_files = []
    
    for txt_file in label_dir.rglob("*.txt"):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        is_stand_only = True
        for line in lines:
            if line.strip():
                cls = int(line.strip().split()[0])
                if cls != 7:  # 假设 stand 是第8个类别（索引7）
                    is_stand_only = False
                    break
        
        if is_stand_only:
            stand_files.append(txt_file)
        else:
            other_files.append(txt_file)
    
    print(f"原始 stand-only 文件数: {len(stand_files)}")
    print(f"其他文件数: {len(other_files)}")
    
    # 如果 stand 文件过多，进行随机欠采样
    if len(stand_files) > target_count:
        keep_files = random.sample(stand_files, target_count)
        remove_files = set(stand_files) - set(keep_files)
        
        for f in remove_files:
            f.unlink()  # 删除标签文件
            img_file = f.parent.parent / "images" / f.with_suffix('.jpg').name
            if img_file.exists():
                img_file.unlink()  # 删除对应图片
        
        print(f"已删除 {len(remove_files)} 个 stand-only 文件")
    else:
        print("stand 文件数量已在目标范围内，无需欠采样")
    
    print(f"欠采样后 stand 文件数 ≈ {min(len(stand_files), target_count)}")

# ====================== 使用示例 ======================
# 请修改为你的实际路径
print("=== 处理 Front 训练集 ===")
undersample_stand(r"I:/ProgramData/课堂专注度项目/SCB-YOLO/data/Front/train/labels", target_count=3800)

print("\n=== 处理 Rear 训练集 ===")
undersample_stand(r"I:/ProgramData/课堂专注度项目/SCB-YOLO/data/Rear/train/labels", target_count=3800)