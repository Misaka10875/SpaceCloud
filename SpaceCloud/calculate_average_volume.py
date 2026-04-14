# calculate_average_volume.py

import numpy as np
import argparse
import sys
import time
from pathlib import Path
from scipy.spatial import ConvexHull, QhullError
from tqdm import tqdm

def calculate_point_cloud_volume(file_path):
    """
    读取点云文件，计算其凸包体积。 (核心计算函数，与之前基本相同)

    Args:
        file_path (str or Path): 点云txt文件的路径。

    Returns:
        float: 计算出的点云凸包体积。如果无法计算则返回None。
    """
    points = []
    try:
        # 首先获取总行数以初始化内部进度条
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)
        
        # 使用tqdm显示单个文件的读取进度
        with open(file_path, 'r', encoding='utf-8') as f:
            # disable=True 可以让这里的进度条在总进度条运行时不那么混乱
            # 如果想看每个文件的详细进度，可以设为 False
            with tqdm(total=total_lines, desc=f"读取 {Path(file_path).name}", unit="点", leave=False, disable=True) as pbar:
                for line in f:
                    try:
                        coords = list(map(float, line.strip().split()[:3]))
                        if len(coords) == 3:
                            points.append(coords)
                    except (ValueError, IndexError):
                        pass # 在批量处理中，静默跳过格式错误的行
                    finally:
                        pbar.update(1)

    except FileNotFoundError:
        print(f"错误: 文件未找到 '{file_path}'", file=sys.stderr)
        return None
    except Exception as e:
        print(f"读取文件 '{file_path}' 时发生错误: {e}", file=sys.stderr)
        return None

    if len(points) < 4:
        # 点数不足，无法计算体积
        return None

    points_array = np.array(points)
    
    try:
        # Scipy的核心功能：计算凸包
        hull = ConvexHull(points_array)
        return hull.volume
    except QhullError:
        # 点集可能是共面或共线的，无法形成3D体
        return None
    except Exception as e:
        print(f"为文件 '{file_path}' 计算体积时发生未知错误: {e}", file=sys.stderr)
        return None


def main():
    """
    主函数，用于遍历目录、执行计算并报告平均值。
    """
    parser = argparse.ArgumentParser(
        description="批量计算指定目录下两层子文件夹内点云的体积，并输出平均值。"
    )
    parser.add_argument(
        "root_dir",
        type=str,
        nargs='?', # '?' 表示这个参数是可选的
        default='.', # 如果不提供，则默认为当前目录
        help="要扫描的根目录。默认为当前目录。"
    )
    args = parser.parse_args()

    # 使用 pathlib 定位根目录
    root_path = Path(args.root_dir)
    if not root_path.is_dir():
        print(f"错误: 指定的路径 '{root_path}' 不是一个有效的目录。", file=sys.stderr)
        sys.exit(1)

    print(f"--- 开始在目录 '{root_path}' 中搜索点云文件 ---")
    
    # 使用 glob 模式匹配两层子目录下的 .txt 文件
    # * 代表第一层文件夹 (n)
    # */* 代表第二层文件夹 (m)
    # */*/*.txt 代表第二层文件夹中的所有txt文件
    target_files = sorted(list(root_path.glob('*/*/*.txt')))

    if not target_files:
        print("未找到任何符合 '*/*/*.txt' 结构的点云文件。请检查您的目录结构。")
        sys.exit(0)

    print(f"发现 {len(target_files)} 个待处理的点云文件。")
    print("-" * 30)

    all_volumes = []
    failed_files = []
    start_time = time.time()

    # 使用tqdm创建总进度条
    for file_path in tqdm(target_files, desc="总体进度", unit="文件"):
        volume = calculate_point_cloud_volume(file_path)
        
        if volume is not None:
            # 实时打印每个文件的结果
            print(f"  [成功] 文件: {str(file_path):<50} 体积: {volume:.6f}")
            all_volumes.append(volume)
        else:
            # 如果计算失败，也进行记录
            print(f"  [失败] 文件: {str(file_path):<50} 无法计算体积 (点数不足或点共面)")
            failed_files.append(str(file_path))

    end_time = time.time()
    
    # --- 最终结果报告 ---
    print("\n" + "="*40)
    print("           批量计算结果摘要")
    print("="*40)
    
    total_processed = len(target_files)
    successful_count = len(all_volumes)
    failed_count = len(failed_files)

    print(f"总处理时间: {end_time - start_time:.2f} 秒")
    print(f"总计文件数: {total_processed}")
    print(f"成功计算数: {successful_count}")
    print(f"失败计算数: {failed_count}")

    if successful_count > 0:
        average_volume = sum(all_volumes) / successful_count
        min_volume = min(all_volumes)
        max_volume = max(all_volumes)
        
        print("-" * 20)
        print(f"最大体积: {max_volume:.6f}")
        print(f"最小体积: {min_volume:.6f}")
        print(f"平均体积: {average_volume:.6f} 立方单位")
        print("-" * 20)

    if failed_count > 0:
        print("\n计算失败的文件列表:")
        for f in failed_files:
            print(f" - {f}")

    print("="*40)


if __name__ == "__main__":
    main()
