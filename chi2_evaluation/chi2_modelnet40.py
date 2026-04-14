# -*- coding: utf-8 -*-
"""
ModelNet40 (modelnet40_resampled) 点云分布均匀度评价（卡方检验）。
数据格式：root 下 modelnet40_shape_names.txt、modelnet40_train.txt、modelnet40_test.txt，
以及类别子目录（如 airplane_0001/）内 .txt 文件，每行 CSV：x,y,z,...，取前 3 列为 xyz。
"""
import os
import sys
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from chi2_utils import compute_chi2_for_pointcloud


def load_modelnet40_shape(txt_path):
    """单样本：CSV，列 x,y,z,...，返回 (N, 3) xyz。"""
    data = np.loadtxt(txt_path, delimiter=',', dtype=np.float64)
    return data[:, 0:3]


def collect_modelnet40_files(root, split='train', num_category=40, max_samples=None):
    """
    split: 'train' | 'test'
    """
    if num_category == 10:
        train_file = os.path.join(root, 'modelnet10_train.txt')
        test_file = os.path.join(root, 'modelnet10_test.txt')
    else:
        train_file = os.path.join(root, 'modelnet40_train.txt')
        test_file = os.path.join(root, 'modelnet40_test.txt')
    path_list = []
    for sp in (split,) if split in ('train', 'test') else ('train', 'test'):
        fpath = train_file if sp == 'train' else test_file
        if not os.path.isfile(fpath):
            continue
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 例如 airplane_0001
                shape_name = line
                category = '_'.join(shape_name.split('_')[:-1])
                txt_path = os.path.join(root, category, shape_name + '.txt')
                if os.path.isfile(txt_path):
                    path_list.append(txt_path)
        if max_samples is not None and len(path_list) >= max_samples:
            break
    if max_samples is not None:
        path_list = path_list[:max_samples]
    return path_list


def main():
    parser = argparse.ArgumentParser(description='ModelNet40 点云卡方均匀度评价')
    parser.add_argument('--data_root', type=str, default='../data/modelnet40_normal_resampled',
                        help='ModelNet40 数据集根目录')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--num_category', type=int, default=40, choices=[10, 40])
    parser.add_argument('--n_bins', type=int, default=10)
    parser.add_argument('--min_expected', type=float, default=5.0)
    parser.add_argument('--max_samples', type=int, default=50000000)
    args = parser.parse_args()

    data_root = os.path.abspath(os.path.join(SCRIPT_DIR, args.data_root))
    if not os.path.isdir(data_root):
        print('数据根目录不存在:', data_root)
        return

    files = collect_modelnet40_files(data_root, split=args.split, num_category=args.num_category, max_samples=args.max_samples)
    if not files:
        print('未找到任何样本 .txt（请检查 modelnet40_train.txt / modelnet40_test.txt 及类别子目录）')
        return

    n_bins_cube = args.n_bins ** 3
    chi2_values = []
    n_valid_bins_list = []
    total_points = 0
    for i, path in enumerate(files):
        xyz = load_modelnet40_shape(path)
        c, n_used = compute_chi2_for_pointcloud(xyz, n_bins=args.n_bins, min_expected=args.min_expected)
        chi2_values.append(c)
        n_valid_bins_list.append(n_used)
        total_points += xyz.shape[0]
        print('  {} 点数={}  χ²={:.2f} 参与体素数={}'.format(os.path.basename(path), xyz.shape[0], c, n_used))

    chi2_values = np.array(chi2_values)
    n_valid_bins_list = np.array(n_valid_bins_list)
    avg_voxel_bins = float(np.mean(n_valid_bins_list))
    print('---')
    print('ModelNet40 卡方汇总 ({} 样本): 均值={:.2f} 中位数={:.2f} 标准差={:.2f}'.format(
        len(chi2_values), np.mean(chi2_values), np.median(chi2_values), np.std(chi2_values)))
    print('总点数={}  平均体素划分块（非空体素均值）={:.2f}  体素格数 B={}'.format(
        total_points, avg_voxel_bins, n_bins_cube))


if __name__ == '__main__':
    main()
