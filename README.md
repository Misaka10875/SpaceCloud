# SpaceCloud 论文配套代码与数据说明

本目录收录深圳大学本科毕业论文《**SpaceCloud（太空云）：从大规模太空物体点云中学习语义**》相关的**可复现代码与数据集**，与正文实验流程对应，便于审阅者或后续研究者复现与扩展。  


---

## 1. 内容概览

| 组件 | 路径 | 说明 |
|------|------|------|
| **SpaceCloud 数据集** | `SpaceCloud/` | 面向太空目标点云语义分割的数据，按 **S3DIS 风格**组织（`Area_*`、场景目录、`Annotations/*.txt`），与论文中的类别与划分一致 |
| **Pointcept 实验代码** | `Pointcept_10lebels/` | 基于开源框架 [Pointcept](https://github.com/Pointcept/Pointcept) 的点云语义分割基准；**已将 S3DIS 数据通道改为 10 类 SpaceCloud 语义标签**，配置文件见 `configs/s3dis/` |
| **均匀度评价（χ²）** | `chi2_evaluation/` | 论文中用于对比点云空间分布均匀程度的卡方统计脚本（亦支持 S3DIS 等其它数据集） |

> **命名说明**：`Pointcept_10lebels` 表示本 fork 中数据集按 **10 个类别（labels）**配置（`lebels` 为历史拼写，迁移时请勿随意重命名，以免影响已有脚本与路径引用）。

---

## 2. SpaceCloud 数据集

### 2.1 简介

- **来源**：三维几何取自 [NASA 相关 3D 模型资源](https://nasa3d.arc.nasa.gov/)，经表面采样与人工标注得到逐点语义标签；**几何为主**（与正文一致，不依赖真实 RGB 纹理进行分割学习）。
- **规模与划分**：共 **7 大类目标**，**10 类细粒度语义标签**，约 **513** 个场景样本；按 `Area_1` … `Area_6` 划分（与 S3DIS「按区域划分」习惯一致，具体 train/val 以配置文件为准）。
- **使用数据前请自行遵守 NASA 模型与衍生数据的许可与引用要求。**

### 2.2 语义类别（10 类）

与 `Pointcept_10lebels/configs/_base_/dataset/s3dis.py` 中 `class_names` 一致（顺序即网络标签索引）：

`mainBody`, `attachment`, `partBody`, `partTip`, `planetBody`, `planetCrater`, `solarPanel`, `tip`, `terrainBody`, `terrainCrater`

中文含义可对照论文正文中的标签说明（如太阳能板、主体、天体/地形相关类别等）。

### 2.3 目录结构（S3DIS 规范）

典型布局如下（示例）：

```text
SpaceCloud/
├── Area_1/
│   ├── Area_1_alignmentAngle.txt    # 各场景全局对齐角（预处理需要）
│   ├── <场景名>/
│   │   └── Annotations/
│   │       ├── mainBody_1.txt
│   │       ├── solarPanel_1.txt
│   │       └── ...
│   └── ...
├── Area_2/
│   └── ...
└── ...
```

- 每个 `Annotations/*.txt` 为**单个物体/部件**的点集，行格式与 S3DIS / Pointcept 预期一致：至少 **6 列**为 `x y z R G B`（本数据集中颜色列可为占位，模型以几何为主）。
- 各 `Area_*` 下含 `Area_*_alignmentAngle.txt`，列出该 Area 内全部场景名及对齐角，供 **`preprocess_s3dis.py`** 遍历场景列表使用。

### 2.4 SpaceCloud 目录内 Python 工具脚本

以下脚本位于数据集根目录 `SpaceCloud/`（个别为历史实验时拷贝的**单场景副本**，与根目录脚本逻辑相同）。用于**凸包体积统计、尺度归一化、标注名检查与数据质检**；与论文中「基于凸包体积的归一化」思路一致。

**环境配置**（按脚本而异，缺省时按需安装）：

```bash
pip install numpy scipy tqdm
```

**数据路径（与预处理脚本一致）**：

- 主体点云：`Area_* /<场景名>/<场景名>.txt`
- 部件标注：`Area_* /<场景名>/Annotations/<语义标签>_<序号>.txt`
- 行格式：至少三列 `x y z`，其后可为 `R G B` 等（脚本会保留第 4 列及以后的内容）
---

#### 脚本一览

| 文件名 | 作用概述 |
|--------|-----------|
| `scale.py` | 按**固定除数**统一缩小/放大坐标，结果写入新文件 `*_scaled.txt`（不覆盖原文件） |
| `edit_volume.py` | **批量粗调**：以每个场景**整体点云凸包体积**为指标，将体积过大/过小的场景缩放到约 **[20, 120] 区间**（**直接覆盖**主体 `*.txt` 与对应 `Annotations/*.txt`） |
| `refine_volume.py` | **二次精调**：对体积不在 **[100, 140]** 的场景，按比例缩放使凸包体积接近 **120**（**直接覆盖**同套文件） |
| `calculate_average_volume.py` | 遍历 `*/*/*.txt`，逐文件计算凸包体积并输出**最大/最小/平均值**（只读） |
| `find_labels.py` | 递归扫描所有 `Annotations` 目录，列出符合 `标签_数字.txt` 形式的**不重复标签前缀**（只读） |
| `check_flat_device_files_generic.py` | 在非 `Annotations` 路径下检测**近乎共面**的主体点云（极差过小），用于发现退化几何（只读） |
| `check_lines.py` | 查找行数超过 **30 万** 的 `.txt`（排除文件名含 `Angle` 的对齐角文件），用于发现异常大文件（只读） |
| `Area_4/.../scale.py` | 与根目录 `scale.py` **同逻辑**的历史/局部拷贝；修改 `root_directory` 与 `scale_factor` 后可在该子树单独试运行 |

---

#### `scale.py`（按比例缩放，输出新文件）

- **用途**：递归遍历目录下全部 `.txt`，将每行前 3 列坐标**除以** `scale_factor`，后面列原样拼接；结果保存为**同目录下** `原名_scaled.txt`，**不修改**原始文件。
- **修改方式**：编辑脚本末尾的 `root_directory`（默认 `'./'`）与 `scale_factor`（如 `1000.0` 表示坐标缩小为原来的 1/1000，凸包体积约为原来的 1/10⁹）。
- **运行**：

```bash
cd SpaceCloud
python scale.py
```

---

#### `edit_volume.py`（粗调体积区间，覆盖写回）

- **用途**：在数据集根下查找形如 `Area_*/*/<场景名>.txt` 的主体文件，计算其凸包体积；若体积 **> 120** 则整体缩小，若 **< 20** 则整体放大（部件与主体的相对关系通过对**同一乘子**缩放所有相关 `*.txt` 保持）。
- **注意**：脚本首行注释仍写作 `scale_point_clouds.py`，逻辑上以**实际代码**为准；`argparse` 文案写为 [50,120]，实现中为 **[20, 120]**。
- **运行**：

```bash
cd SpaceCloud
python edit_volume.py
python edit_volume.py "G:/path/to/SpaceCloud"
```

---

#### `refine_volume.py`（精调到目标体积，覆盖写回）

- **用途**：仅当主体点云凸包体积**不在 [100, 140]** 时，按 \(s = (120/V)^{1/3}\) 缩放坐标，使体积接近 **120**；**在 [100, 140] 内则跳过**。通常接在 `edit_volume.py` 之后做精细对齐。
- **运行**：

```bash
cd SpaceCloud
python refine_volume.py
python refine_volume.py /path/to/SpaceCloud
```

---

#### `calculate_average_volume.py`（统计体积分布，只读）

- **用途**：匹配 `root_dir/*/*/*.txt`（两层子目录下的所有 txt），逐文件计算凸包体积，汇总成功数、失败数及**最大/最小/平均**体积。
- **运行**：

```bash
cd SpaceCloud
python calculate_average_volume.py
python calculate_average_volume.py .
```

---

#### `find_labels.py`（枚举标注文件名中的类别前缀，只读）

- **用途**：在所有名为 `Annotations` 的文件夹中，匹配 `语义_序号.txt` 形式，输出不重复的**语义前缀**（如 `solarPanel`、`mainBody`），用于核对是否与 Pointcept 配置中 `class_names` 一致。
- **修改**：默认 `start_dir="."`；若从其它工作目录运行，可编辑 `find_unique_labels(start_dir=".")`。
- **运行**：

```bash
cd SpaceCloud
python find_labels.py
```

---

#### `check_flat_device_files_generic.py`（共面/薄片状点云筛查，只读）

- **用途**：跳过 `Annotations` 目录，对其余 `.txt`（且文件名不含 `alignmentAngle`）计算 XYZ 各轴极差；若任一小于阈值 `TOLERANCE = 0.4`，则报告为 X/Y/Z 方向**过薄**的疑似平面物体，便于人工复查。
- **运行**：

```bash
cd SpaceCloud
python check_flat_device_files_generic.py
```

（默认从当前目录扫描；若需封装 CLI，可自行改为 `argparse` 传入根路径。）

---

#### `check_lines.py`（超大文本文件筛查，只读）

- **用途**：递归查找行数 **> 300000** 的 `.txt`，排除文件名含 **`Angle`** 的文件（用于跳过对齐角等小文件命名特例）。用于发现点数异常膨胀或重复导出的文件。
- **运行**：

```bash
cd SpaceCloud
python check_lines.py
python check_lines.py "G:/path/to/SpaceCloud"
```

---

## 3. Pointcept 基准实验（`Pointcept_10lebels/`）

本仓库的 Pointcept 子目录基于官方 Pointcept，用于在 **SpaceCloud（按 S3DIS 管线）**上训练/测试 PointNet++/SPUNet/Point Transformer 等配置。  
**完整安装、依赖版本、Docker、各数据集通用说明**请参阅同目录下的 **`Pointcept_10lebels/README.md`**（上游官方文档，务必先读）。

以下仅概括与 **本论文 / SpaceCloud** 强相关的步骤。

### 3.1 环境

- **系统**：推荐 **Ubuntu 18.04+**，**CUDA 11.3+**，**PyTorch 1.10+**（或 README 中推荐的 conda 组合，如 `environment.yml`）。
- 需按官方说明安装 **`spconv`**、编译 **`libs/pointops`** 等（见 `Pointcept_10lebels/README.md` → *Installation*）。

### 3.2 数据准备：预处理 + 链接到 `data/s3dis`

在我们提供的源代码中，已经包含预处理好的数据集并放在 data 目录中，可直接前往[3.3 训练部分](#33-训练)。

Pointcept 训练使用**预处理后的** S3DIS 格式数据。原始 SpaceCloud 为 Stanford 风格的 `Area_*/*/Annotations/*.txt`，需用本 fork 自带的预处理脚本生成中间格式，再放入 `data/s3dis`。

1. 进入代码目录并设置 `PYTHONPATH`（以 Linux/macOS 为例）：

   ```bash
   cd Pointcept_10lebels
   export PYTHONPATH=./
   ```

2. 对 **SpaceCloud 根目录**执行预处理（按需选择 `Area_*`；与官方 S3DIS 用法相同）：

   ```bash
   python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py \
     --dataset_root /绝对路径/到/SpaceCloud \
     --output_root /绝对路径/到/输出目录/spacecloud_processed \
     --splits Area_1 Area_2 Area_3 Area_4 Area_5 Area_6
   ```

   可选参数（与官方一致）：`--align_angle`、`--parse_normal`（需 `--raw_root` 等，SpaceCloud 若以几何为主可仅用基础流程）。  
   预处理脚本中的 **10 类名称**已与 SpaceCloud 标注一致，无需再改类别表。

3. 将预处理结果链接（或复制）为 Pointcept 期望的 **`data/s3dis`**：

   ```bash
   mkdir -p data
   ln -s /绝对路径/到/spacecloud_processed data/s3dis
   ```

   **Windows**：无 `ln -s` 时可用**管理员** `mklink /J` 创建目录联接，或直接复制目录到 `Pointcept_10lebels/data/s3dis`。

### 3.3 训练

官方推荐脚本（详见 `Pointcept_10lebels/README.md` → *Quick Start* → *Training*）：

```bash
export CUDA_VISIBLE_DEVICES=0   # 按需
export PYTHONPATH=./
sh scripts/train.sh -p python -g 1 -d s3dis -c <配置文件名> -n <实验名>
```

示例（SPUNet 基线，配置文件在 `configs/s3dis/` 下）：

```bash
sh scripts/train.sh -p python -g 1 -d s3dis -c semseg-spunet-v1m1-0-base -n semseg-spunet-spacecloud
```

或直接调用：

```bash
python tools/train.py --config-file configs/s3dis/semseg-spunet-v1m1-0-base.py \
  --options save_path=exp/s3dis/my_exp
```

实验输出默认位于 `exp/` 下（含日志、TensorBoard、`wandb` 等，可关闭 `wandb`，见上游 README）。

### 3.4 测试与精评

训练结束后可按上游说明使用 `scripts/test.sh` 或 `tools/test.py`；精评（Precise Evaluation）等行为以 **Pointcept 当前版本默认逻辑**为准，见 `Pointcept_10lebels/README.md` → *Quick Start* → *Testing*。

### 3.5 配置文件位置

- **SpaceCloud 10 类语义名**：`configs/_base_/dataset/s3dis.py`
- **各模型训练配置**：`configs/s3dis/*.py`（如 `semseg-pt-v3m1-0-base.py`、`semseg-spunet-v1m1-0-base.py` 等）

修改 batch size、点采样数、路径等时，请以论文实验设置与显存条件为准。

---

## 4. 点云均匀度评价（`chi2_evaluation/`）

用于复现论文中基于 **χ² 统计量**的「规则网格内分布均匀度」对比。依赖 **Python 3**、**NumPy**。

详见子目录 **`chi2_evaluation/README.md`**。对 **与 S3DIS 相同 txt 目录结构**的数据，示例：

```bash
cd chi2_evaluation
python chi2_s3dis.py --data_root /绝对路径/到/SpaceCloud --n_bins 10
```

（若仅统计原始 txt、未做 Pointcept 体素化，该脚本与论文中「原始点云分布」分析一致；参数含义见该 README。）

---


