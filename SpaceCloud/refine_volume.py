# refine_volume.py (Final Refinement Version)

import numpy as np
import argparse
import sys
from pathlib import Path
from scipy.spatial import ConvexHull, QhullError
from tqdm import tqdm
import math

# --- 核心模块1: 体积计算 (无变化) ---
def calculate_volume(points_array):
    """
    根据numpy点数组计算凸包体积。
    """
    if points_array.shape[0] < 4:
        return None
    try:
        hull = ConvexHull(points_array)
        return hull.volume
    except (QhullError, ValueError):
        return None

# --- 核心模块2: 点云文件读写与缩放 (无变化) ---
def scale_and_overwrite_point_cloud(file_path: Path, coord_multiplier: float):
    """
    读取点云文件，缩放其XYZ坐标，然后覆盖保存。
    """
    try:
        lines = file_path.read_text(encoding='utf-8').splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    xyz = np.array(list(map(float, parts[:3])))
                    rgb_etc = parts[3:]
                    xyz_scaled = xyz * coord_multiplier
                    scaled_line = f"{xyz_scaled[0]:.6f} {xyz_scaled[1]:.6f} {xyz_scaled[2]:.6f}"
                    if rgb_etc:
                        scaled_line += " " + " ".join(rgb_etc)
                    new_lines.append(scaled_line)
                except ValueError:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        file_path.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
        return True
    except Exception as e:
        print(f"\n[错误] 处理文件 '{file_path}' 时失败: {e}", file=sys.stderr)
        return False

# --- 主逻辑 ---
def main():
    parser = argparse.ArgumentParser(
        description="精调点云数据集体积，将所有在 [100, 140] 范围外的点云体积精确缩放到 120。"
    )
    parser.add_argument(
        "root_dir", type=str, nargs='?', default='.',
        help="包含 Area_* 文件夹的根目录。默认为当前目录。"
    )
    args = parser.parse_args()
    root_path = Path(args.root_dir).resolve()

    if not root_path.is_dir():
        print(f"错误: 目录 '{root_path}' 不存在。", file=sys.stderr)
        sys.exit(1)

    print(f"--- 开始扫描目录进行体积精调: {root_path} ---")

    all_txt_files = list(root_path.glob('Area_*/*/*.txt'))
    main_point_clouds = [f for f in all_txt_files if f.stem == f.parent.name and 'alignmentAngle' not in f.name]

    if not main_point_clouds:
        print("未找到任何符合 'Area_*/Object_Name/Object_Name.txt' 结构的主体点云文件。")
        sys.exit(0)

    print(f"发现 {len(main_point_clouds)} 个主体点云对象。开始检查并精调...")
    print("=" * 70)

    # 定义新的体积控制参数
    SAFE_MIN = 100.0
    SAFE_MAX = 140.0
    TARGET_VOLUME = 120.0
    
    for main_pc_path in tqdm(main_point_clouds, desc="精调进度", unit="个"):
        try:
            points = np.loadtxt(main_pc_path, usecols=(0, 1, 2))
        except Exception as e:
            print(f"\n[跳过] 无法读取主体文件 '{main_pc_path}': {e}")
            continue

        original_volume = calculate_volume(points)
        if original_volume is None or original_volume <= 1e-9:
            print(f"\n[跳过] '{main_pc_path.parent.name}' 体积为0或无法计算。")
            continue
        
        # =================================================================
        # ========== 核心优化：精调逻辑 ==========
        # =================================================================
        
        # 检查体积是否在可接受的范围内
        if SAFE_MIN <= original_volume <= SAFE_MAX:
            print(f"\n[保留] '{main_pc_path.parent.name}' 体积 {original_volume:.4f} 在 [{SAFE_MIN}, {SAFE_MAX}] 范围内, 无需操作。")
            continue

        # 如果不在范围内，则计算精确的缩放系数以达到目标体积
        # S = (V_target / V_orig) ^ (1/3)
        coord_multiplier = (TARGET_VOLUME / original_volume) ** (1/3.0)
        new_volume = original_volume * (coord_multiplier ** 3) # 理论上应非常接近120

        print(f"\n[精调] '{main_pc_path.parent.name}': 体积 {original_volume:.4f} 超出范围, 重新缩放至 {TARGET_VOLUME}")
        print(f"    - 理论体积变化: {original_volume:.4f}  ->  {new_volume:.4f}")

        # =================================================================
        
        # 收集所有需要缩放的文件 (主体 + 部件)
        files_to_scale = [main_pc_path]
        annotations_dir = main_pc_path.parent / 'Annotations'
        if annotations_dir.is_dir():
            files_to_scale.extend(list(annotations_dir.glob('*.txt')))
        
        print(f"    - 将对 {len(files_to_scale)} 个文件 (主体+部件) 应用缩放...")
        all_successful = True
        for file_to_scale in files_to_scale:
            if not scale_and_overwrite_point_cloud(file_to_scale, coord_multiplier):
                all_successful = False
        
        if all_successful:
            print("    - ✅ 精调成功。")
        else:
            print("    - ❌ 精调过程中发生错误，请检查日志。")

    print("=" * 70)
    print("所有主体点云精调处理完毕。")

if __name__ == "__main__":
    main()
