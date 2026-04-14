# scale_point_clouds.py (Optimized Version)

import numpy as np
import argparse
import sys
from pathlib import Path
from scipy.spatial import ConvexHull, QhullError
from tqdm import tqdm
import math # 引入math库以使用向上取整 (ceil)

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
        description="高效批量缩放点云数据集，使其体积落入目标范围 [50, 120]。"
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

    print(f"--- 开始扫描目录: {root_path} ---")

    all_txt_files = list(root_path.glob('Area_*/*/*.txt'))
    main_point_clouds = [f for f in all_txt_files if f.stem == f.parent.name and 'alignmentAngle' not in f.name]

    if not main_point_clouds:
        print("未找到任何符合 'Area_*/Object_Name/Object_Name.txt' 结构的主体点云文件。")
        sys.exit(0)

    print(f"发现 {len(main_point_clouds)} 个主体点云对象。开始处理...")
    print("=" * 60)

    for main_pc_path in tqdm(main_point_clouds, desc="处理主体对象", unit="个"):
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
        # ========== 核心优化：替换while循环为直接数学计算 ==========
        # =================================================================
        TARGET_MIN, TARGET_MAX = 20.0, 120.0
        volume_factor = 1.0
        action = "无需缩放"
        coord_multiplier = 1.0

        if original_volume > TARGET_MAX:
            # 体积太大，直接计算缩小倍数
            # F >= V_orig / 120  => F = 10 * ceil(V_orig / 1200)
            volume_factor = 10 * math.ceil(original_volume / (TARGET_MAX * 10))
            coord_multiplier = 1 / (volume_factor ** (1/3.0))
            action = f"缩小 {int(volume_factor)} 倍"
            
        elif original_volume < TARGET_MIN:
            # 体积太小，直接计算放大倍数
            # F >= 20 / V_orig => F = 10 * ceil(2 / V_orig)
            volume_factor = 10 * math.ceil((TARGET_MIN / original_volume) / 10)
            coord_multiplier = volume_factor ** (1/3.0)
            action = f"放大 {int(volume_factor)} 倍"
        
        # =================================================================
        # ====================== 优化部分结束 =========================
        # =================================================================

        if action == "无需缩放":
            print(f"\n[保留] '{main_pc_path.parent.name}' 体积 {original_volume:.4f} 在目标范围内，无需缩放。")
            continue

        new_volume = original_volume * (coord_multiplier ** 3)
        print(f"\n[处理] '{main_pc_path.parent.name}': 应 {action}")
        print(f"    - 体积变化: {original_volume:.4f}  ->  {new_volume:.4f}")

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
            print("    - ✅ 缩放成功。")
        else:
            print("    - ❌ 缩放过程中发生错误，请检查日志。")

    print("=" * 60)
    print("所有主体点云处理完毕。")

if __name__ == "__main__":
    main()
