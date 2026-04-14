# -*- coding: utf-8 -*-
"""
ShapeNet Part (shapenetcore_partanno_segmentation_benchmark_v0_normal) 点云分布均匀度评价（卡方检验）。
数据格式：root 下按 synsetoffset2category.txt 的类别建子目录，每样本为 .txt，每行 x y z nx ny nz seg_label，取前 3 列为 xyz。
"""
import os
import sys
import json
import argparse
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from chi2_utils import compute_chi2_for_pointcloud


def load_shapenet_part_file(txt_path):
    """单文件：空格或逗号分隔，列 x y z [nx ny nz] seg，返回 (N, 3) xyz。"""
    data = np.loadtxt(txt_path, dtype=np.float64)
    return data[:, 0:3]


def collect_shapenet_part_files(root, split='train', max_total=None):
    """
    split: 'train' | 'val' | 'test' | 'trainval'
    """
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    if not os.path.isfile(catfile):
        return []
    cat = {}
    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            if len(ls) >= 2:
                cat[ls[0]] = ls[1]

    train_ids = set()
    val_ids = set()
    test_ids = set()
    for name, ext in [('shuffled_train_file_list.json', train_ids),
                      ('shuffled_val_file_list.json', val_ids),
                      ('shuffled_test_file_list.json', test_ids)]:
        path = os.path.join(root, 'train_test_split', name)
        if os.path.isfile(path):
            with open(path, 'r') as f:
                for d in json.load(f):
                    ext.add(str(d.split('/')[2]))

    if split == 'trainval':
        split_ids = train_ids | val_ids
    elif split == 'train':
        split_ids = train_ids
    elif split == 'val':
        split_ids = val_ids
    elif split == 'test':
        split_ids = test_ids
    else:
        split_ids = train_ids | val_ids | test_ids

    files = []
    for synset, folder in cat.items():
        if max_total is not None and len(files) >= max_total:
            break
        dir_point = os.path.join(root, folder)
        if not os.path.isdir(dir_point):
            continue
        for fn in sorted(os.listdir(dir_point)):
            if not fn.endswith('.txt'):
                continue
            token = fn[:-4]
            if token not in split_ids:
                continue
            files.append(os.path.join(dir_point, fn))
            if max_total is not None and len(files) >= max_total:
                break
    return files


def main():
    parser = argparse.ArgumentParser(description='ShapeNet Part 点云卡方均匀度评价')
    parser.add_argument('--data_root', type=str,
                        default='../data/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                        help='ShapeNet Part 数据集根目录')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test', 'trainval'])
    parser.add_argument('--n_bins', type=int, default=10)
    parser.add_argument('--min_expected', type=float, default=5.0)
    parser.add_argument('--max_samples', type=int, default=50000000,
                        help='最多评价样本数')
    args = parser.parse_args()

    data_root = os.path.abspath(os.path.join(SCRIPT_DIR, args.data_root))
    if not os.path.isdir(data_root):
        print('数据根目录不存在:', data_root)
        return

    files = collect_shapenet_part_files(data_root, split=args.split, max_total=args.max_samples)
    if not files:
        print('未找到任何 .txt 样本文件')
        return

    n_bins_cube = args.n_bins ** 3
    chi2_values = []
    n_valid_bins_list = []
    total_points = 0
    for i, path in enumerate(files):
        xyz = load_shapenet_part_file(path)
        c, n_used = compute_chi2_for_pointcloud(xyz, n_bins=args.n_bins, min_expected=args.min_expected)
        chi2_values.append(c)
        n_valid_bins_list.append(n_used)
        total_points += xyz.shape[0]
        print('  {} 点数={}  χ²={:.2f} 参与体素数={}'.format(os.path.basename(path), xyz.shape[0], c, n_used))

    chi2_values = np.array(chi2_values)
    n_valid_bins_list = np.array(n_valid_bins_list)
    avg_voxel_bins = float(np.mean(n_valid_bins_list))
    print('---')
    print('ShapeNet Part 卡方汇总 ({} 样本): 均值={:.2f} 中位数={:.2f} 标准差={:.2f}'.format(
        len(chi2_values), np.mean(chi2_values), np.median(chi2_values), np.std(chi2_values)))
    print('总点数={}  平均体素划分块（非空体素均值）={:.2f}  体素格数 B={}'.format(
        total_points, avg_voxel_bins, n_bins_cube))


if __name__ == '__main__':
    main()
