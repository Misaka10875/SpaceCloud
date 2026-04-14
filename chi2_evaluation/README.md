# 点云分布均匀度评价（卡方检验）

基于论文中的卡方统计量 χ² = Σ (O_b - E_b)² / E_b 评价点云在包围盒内规则网格上的分布均匀程度：**χ² 越小表示分布越接近均匀**。

## 依赖

- Python 3
- NumPy

## 公共模块

- **chi2_utils.py**：`compute_chi2_for_pointcloud(points_xyz, n_bins=10, min_expected=5.0)`  
  对单一点云 `(N, 3)` 计算卡方值；`min_expected` 不足时会自动减少 `n_bins`。

## 四个数据集的脚本

| 脚本 | 数据集 | 数据格式与默认路径 |
|------|--------|--------------------|
| **chi2_s3dis.py** | S3DIS | **TXT 格式**：目录结构为 `data_root/Area_*/xxx/xxx.txt`（Area 下有一层子目录，点云 .txt 在该子目录内）。每行前 3 列为 x y z。需用 `--data_root` 指定根目录 |
| **chi2_semantic_kitti.py** | Semantic KITTI | `sequences/<seq>/velodyne/*.bin`，每帧 N×4 float32，取 x,y,z。默认 `--data_root` 为当前目录 |
| **chi2_shapenet_part.py** | ShapeNet Part (shapenetcore_partanno_segmentation_benchmark_v0_normal) | 根目录含 `synsetoffset2category.txt`、`train_test_split/*.json` 及类别子目录下 `.txt` 样本，取前 3 列 xyz |
| **chi2_modelnet40.py** | ModelNet40 (modelnet40_resampled) | 根目录含 `modelnet40_*_train/test.txt` 及类别子目录下 `.txt`（CSV），取前 3 列 xyz |

## 使用示例

```bash
# S3DIS（TXT 格式，指定数据根目录）
python chi2_s3dis.py --data_root /path/to/s3dis_txt_root --n_bins 10

# Semantic KITTI（需先下载数据集到 data_root）
python chi2_semantic_kitti.py --data_root /path/to/semantic_kitti --max_total_frames 200

# ShapeNet Part
python chi2_shapenet_part.py --data_root /path/to/shapenetcore_partanno_segmentation_benchmark_v0_normal --split train --max_samples 500

# ModelNet40
python chi2_modelnet40.py --data_root /path/to/modelnet40_normal_resampled --split train --max_samples 500
```

各脚本均支持 `--n_bins`（每维度体素数）、`--min_expected`（体素期望点数下限）。输出包括：每样本/每帧的 χ² 与参与体素数；汇总的卡方均值、中位数、标准差；**总点数**（所有参与计算的点云点数之和）、**平均体素划分块**（各样本非空体素数的均值）、**体素格数 B**（n_bins³）。
