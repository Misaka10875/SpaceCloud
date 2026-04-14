# -*- coding: utf-8 -*-
"""
S3DIS 数据集点云分布均匀度评价（卡方检验）。
数据格式：TXT 点云文件。目录结构为 data_root/Area_*/xxx/xxx.txt，
即每个 Area_X 下有一层子目录（如房间名），点云 .txt 在该子目录内。
每个 .txt 每行一个点，前 3 列为 x y z（可用空格或逗号分隔），后续列可为 rgb、label 等，程序只取前 3 列。
"""
import os
import sys
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from chi2_utils import compute_chi2_for_pointcloud


def load_txt_pointcloud(txt_path, delimiter=None):
    """
    从 txt 加载点云，取前 3 列为 xyz。
    delimiter=None 时按空白分隔，否则用指定分隔符（如 ','）。
    """
    try:
        data = np.loadtxt(txt_path, delimiter=delimiter, dtype=np.float64, ndmin=2)
    except Exception:
        data = np.loadtxt(txt_path, delimiter=',', dtype=np.float64, ndmin=2)
    if data.shape[1] < 3:
        raise ValueError('至少需要 3 列: {}'.format(txt_path))
    return data[:, 0:3]


def collect_s3dis_txt_files(data_root, max_files=None):
    """
    收集 S3DIS 的 txt 点云文件路径。
    目录结构：data_root/Area_*/xxx/xxx.txt，即 Area 下有一层子目录，.txt 在该子目录内。
    """
    data_root = os.path.abspath(data_root)
    subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    area_like = [d for d in subdirs if 'Area_' in d or 'area_' in d.lower()]

    txt_files = []
    for area in sorted(area_like):
        area_path = os.path.join(data_root, area)
        for subname in sorted(os.listdir(area_path)):
            subpath = os.path.join(area_path, subname)
            if not os.path.isdir(subpath):
                continue
            for fn in sorted(os.listdir(subpath)):
                if fn.lower().endswith('.txt'):
                    txt_files.append(os.path.join(subpath, fn))
                if max_files is not None and len(txt_files) >= max_files:
                    return txt_files[:max_files]
        if max_files is not None and len(txt_files) >= max_files:
            break
    return txt_files


def main():
    parser = argparse.ArgumentParser(description='S3DIS 点云卡方均匀度评价（TXT 格式）')
    parser.add_argument('--data_root', type=str, required=True,
                        help='S3DIS 数据根目录（内含 Area_*/xxx/xxx.txt）')
    parser.add_argument('--n_bins', type=int, default=10,
                        help='每维度体素数，总格数 n_bins^3')
    parser.add_argument('--min_expected', type=float, default=5.0,
                        help='体素期望点数下限，不足时自动减少 n_bins')
    parser.add_argument('--max_files', type=int, default=None,
                        help='最多加载的 txt 文件数，默认全部')
    parser.add_argument('--delimiter', type=str, default=None,
                        help='列分隔符，默认按空白；若为逗号可填 comma 或 ,')
    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    if not os.path.isdir(data_root):
        print('数据根目录不存在:', data_root)
        return

    txt_list = collect_s3dis_txt_files(data_root, max_files=args.max_files)
    if not txt_list:
        print('未找到任何 .txt 点云文件。请确认目录结构为 data_root/Area_*/xxx/xxx.txt。')
        return

    delim = ',' if (args.delimiter in (',', 'comma')) else None
    n_bins_cube = args.n_bins ** 3
    print('已发现 {} 个 txt 文件，体素格数 B={}，正在计算卡方...'.format(len(txt_list), n_bins_cube))
    chi2_values = []
    n_valid_bins_list = []
    total_points = 0
    for path in txt_list:
        try:
            xyz = load_txt_pointcloud(path, delimiter=delim)
        except Exception as e:
            print('  跳过 {}: {}'.format(path, e))
            continue
        if xyz.shape[0] < 10:
            print('  跳过 {}: 点数过少 ({})'.format(path, xyz.shape[0]))
            continue
        c, n_used = compute_chi2_for_pointcloud(xyz, n_bins=args.n_bins, min_expected=args.min_expected)
        chi2_values.append(c)
        n_valid_bins_list.append(n_used)
        total_points += xyz.shape[0]
        print('  {} 点数={}  χ²={:.2f} 参与体素数={}'.format(os.path.basename(path), xyz.shape[0], c, n_used))

    if not chi2_values:
        print('没有成功计算任何文件的卡方。')
        return
    chi2_values = np.array(chi2_values)
    n_valid_bins_list = np.array(n_valid_bins_list)
    avg_voxel_bins = float(np.mean(n_valid_bins_list))
    print('---')
    print('S3DIS 卡方汇总: 均值={:.2f} 中位数={:.2f} 标准差={:.2f}'.format(
        np.mean(chi2_values), np.median(chi2_values), np.std(chi2_values)))
    print('总点数={}  平均体素划分块（非空体素均值）={:.2f}  体素格数 B={}'.format(
        total_points, avg_voxel_bins, n_bins_cube))


if __name__ == '__main__':
    main()
