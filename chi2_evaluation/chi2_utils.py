# -*- coding: utf-8 -*-
"""
卡方检验工具：用于评价点云在空间中的分布均匀程度。
公式: χ² = Σ (O_b - E_b)² / E_b
其中 O_b 为体素内观测点数，E_b = N/B 为均匀时的期望点数。
χ² 越小表示分布越接近均匀。
"""
import numpy as np


def compute_chi2_for_pointcloud(points_xyz, n_bins=10, min_expected=5.0):
    """
    对单一点云计算卡方统计量，评价空间分布均匀度。

    Parameters
    ----------
    points_xyz : np.ndarray, shape (N, 3)
        点云坐标，每行为 (x, y, z)。
    n_bins : int
        每个维度上的体素划分数，总体素数 B = n_bins^3。
    min_expected : float
        期望点数过小的体素会与相邻体素合并，直至 E_b >= min_expected（经验上 E_b >= 5）。

    Returns
    -------
    chi2 : float
        卡方统计量，越小越均匀。
    n_valid_bins : int
        参与计算的体素个数（合并后）。
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    if points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
        raise ValueError("points_xyz 应为 (N, 3) 的数组")
    xyz = points_xyz[:, :3].copy()
    N = xyz.shape[0]
    if N == 0:
        return np.nan, 0

    # 包围盒
    min_vals = xyz.min(axis=0)
    max_vals = xyz.max(axis=0)
    extent = max_vals - min_vals
    extent[extent < 1e-9] = 1e-9  # 避免除零

    # 归一化到 [0, n_bins)，再取整得到体素索引
    indices = np.floor((xyz - min_vals) / extent * n_bins).astype(np.int32)
    indices = np.clip(indices, 0, n_bins - 1)

    # 线性化体素索引
    stride = np.array([n_bins * n_bins, n_bins, 1], dtype=np.int32)
    voxel_flat = (indices * stride).sum(axis=1)

    # 统计每个体素的点数
    B_total = n_bins ** 3
    obs, _ = np.histogram(voxel_flat, bins=np.arange(B_total + 1))

    # 等体积网格下 E_b = N / B
    E = N / B_total
    if E < min_expected:
        # 体素过大时仍按当前网格算，但给出警告；或可减少 n_bins
        pass

    # 只对期望足够大的体素计算（或合并稀疏体素）：这里采用简单做法——忽略 O_b 和 E_b 都为 0 的格子，且对 E 很小的做合并
    # 常见做法：仅保留 obs > 0 的格子，期望为 E；若 E < min_expected 则先减少 n_bins 使 E >= min_expected
    if E < min_expected:
        # 减少 n_bins 使平均每格点数 >= min_expected
        n_bins_new = max(2, int((N / min_expected) ** (1.0 / 3.0)))
        return compute_chi2_for_pointcloud(points_xyz, n_bins=n_bins_new, min_expected=min_expected)

    # 等体积网格下 E_b = E 恒定；空格贡献 (0-E)²/E = E
    chi2 = np.sum((obs - E) ** 2 / E)
    n_valid_bins = np.sum(obs > 0)
    return float(chi2), int(n_valid_bins)


def compute_chi2_batch(points_list, n_bins=10, min_expected=5.0, aggregate='mean'):
    """
    对多帧/多个点云分别计算卡方，再聚合。

    Parameters
    ----------
    points_list : list of np.ndarray
        每项为 (N_i, 3) 的点云。
    n_bins, min_expected : 同 compute_chi2_for_pointcloud
    aggregate : 'mean' | 'median' | 'all'
        聚合方式：均值、中位数或返回全部值。

    Returns
    -------
    result : float or np.ndarray
        聚合后的卡方值或全部卡方值数组。
    """
    chi2_values = []
    for pts in points_list:
        c, _ = compute_chi2_for_pointcloud(pts[:, :3], n_bins=n_bins, min_expected=min_expected)
        if not np.isnan(c):
            chi2_values.append(c)
    chi2_values = np.array(chi2_values)
    if len(chi2_values) == 0:
        return np.nan if aggregate != 'all' else np.array([])
    if aggregate == 'mean':
        return float(np.mean(chi2_values))
    if aggregate == 'median':
        return float(np.median(chi2_values))
    return chi2_values
