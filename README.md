# TADA_T2 GPU 加速版

> 基于 [TADA_T2](https://github.com/ryanemenecker/TADA_T2) 的 GPU 加速修改版

---

## 目录

- [项目简介](#项目简介)
- [与原版差异](#与原版差异)
- [系统要求](#系统要求)
- [安装](#安装)
  - [一键安装](#一键安装)
  - [手动安装](#手动安装)
- [使用方法](#使用方法)
  - [基本用法](#基本用法)
  - [查找表管理](#查找表管理)
  - [命令行参数](#命令行参数)
- [输出格式](#输出格式)
- [文件结构](#文件结构)
- [性能对比](#性能对比)
- [常见问题](#常见问题)
- [许可证](#许可证)

---

## 项目简介

TADA_T2 是一个用于预测**转录激活域 (Transcriptional Activation Domain, TAD)** 的深度学习工具，基于 TensorFlow 2 的卷积-注意力-双向 LSTM 模型。输入蛋白质序列，输出每个序列的 TAD 预测分数（0-1，越高越可能是 TAD）。

本项目是 TADA_T2 的 **GPU 加速修改版**，在原始仓库基础上：

- 将所有可在GPU上进行的操作全部重写，显著提高推理速度


### 工作原理

```
输入 FASTA 序列
    │
    ▼
┌─────────────────────┐
│  1. 解析 FASTA       │  提取序列名称与氨基酸序列
│  2. 序列清洗         │  剔除 *, X, U 等非标准氨基酸
│  3. 滑动窗口切分     │  步长 1，窗口 40aa（>40aa 序列）
│  4. 特征计算 (GPU)   │  每窗口 42 维特征向量
│  5. 特征缩放 (GPU)   │  StandardScaler + MinMaxScaler
│  6. 模型推理 (GPU)   │  Conv1D → Attention → BiLSTM → Softmax
│  7. 取最高分窗口     │  每条序列输出最佳 TAD 分数
└─────────────────────┘
    │
    ▼
输出 TSV（序列名、长度、最高分、最佳片段、位置）
```

### 42 维特征构成

| 索引 | 特征 | 来源 | 级别 |
|------|------|------|------|
| 0 | kappa | NumPy 矢量化计算 | 序列级 |
| 1 | omega | NumPy 矢量化计算 | 序列级 |
| 2-9 | 理化性质 ×8 | 查找表（hydropathy, WW_hydropathy, ncpr, disorder_prom, fcr, charge, frac_neg, frac_pos） | 窗口级 |
| 10-20 | 氨基酸类别 ×11 | one-hot 计数（aliphatics, aromatics, branching, charged, ...） | 窗口级 |
| 21 | 二级结构 | alphaPredict 查找表 | 窗口级 |
| 22-41 | 20 种氨基酸计数 | one-hot 计数（训练集顺序） | 窗口级 |

---


## 系统要求

| 项目 | 最低要求 | 推荐 |
|------|---------|------|
| Python | ≥ 3.8 | 3.10+ |
| TensorFlow | ≥ 2.10 | 2.15+（GPU 版） |
| GPU 显存 | 4 GB | 16 GB+ |
| 系统内存 | 8 GB | 32 GB+ |
| 磁盘空间 | 2 GB | 5 GB+（含查找表） |

**依赖包：**
- `tensorflow` ≥ 2.10（模型推理 + GPU 特征计算）
- `numpy`（矢量化计算）
- `alphaPredict`（二级结构预测，生成查找表时需要）
- `protfasta`（FASTA 解析辅助）
- `tqdm`（进度条，可选）
- `localcider`（仅生成 cider 查找表时需要，运行预测不需要）

---

## 安装

### 一键安装

```bash
# 1. 下载安装包
本分支下的TADA_V2_GPU目录

# 2. 定向到目录
cd TADA_T2_GPU

# 3. 运行安装脚本
chmod +x setup.sh
./setup.sh
```

安装脚本自动完成以下步骤：

1. 克隆原始 [TADA_T2](https://github.com/ryanemenecker/TADA_T2) 仓库
2. 安装 Python 依赖（tensorflow, numpy, alphaPredict, protfasta, tqdm）
3. `pip install -e .` 安装 TADA_T2 包
4. 用修改后的 `features.py`、`model.py`、`predictor.py` 覆盖安装包中的原文件
5. 安装完成 ✓

### 手动安装

如果自动安装遇到问题，可以手动执行：

```bash
# 1. 克隆原始仓库
git clone --depth 1 https://github.com/ryanemenecker/TADA_T2.git
cd TADA_T2

# 2. 安装依赖
pip install tensorflow numpy alphaPredict protfasta tqdm

# 3. 安装 TADA_T2 包
pip install -e .

# 4. 替换修改后的文件
#    找到安装路径：
python3 -c "import TADA_T2; import os; print(os.path.join(os.path.dirname(TADA_T2.__file__), 'backend'))"
#    将 src/TADA_T2/backend/ 下的 features.py、model.py、predictor.py 复制到上述路径

# 5. 复制 predict_tad.py 到工作目录
cp /path/to/TADA_T2_GPU/src/predict_tad.py /your/work/dir/
```

---

## 使用方法

### 基本用法

```bash
cd /your/work/dir
python3 predict_tad.py sequences.fasta -o results.tsv
```

首次运行时，如果 `alpha_5mer_lookup.npy` 和 `cider_5mer_lookup.npy` 不存在，脚本会**自动生成**（分别约 2-3 分钟和 10 分钟）。后续运行直接加载缓存，秒级启动。

### 查找表管理

predict_tad.py 按以下优先级查找两个查找表文件：

| 优先级 | 查找方式 | 示例 |
|--------|---------|------|
| 1 | CLI 参数 | `--alpha_lookup ./alpha.npy` |
| 2 | 环境变量 | `export TADA_ALPHA_LOOKUP=/path/to/alpha.npy` |
| 3 | 包数据目录 | `TADA_T2/data/alpha_5mer_lookup.npy` |
| 4 | 同目录 | 与 `features.py` 同级 |
| 5 | 当前工作目录 | `./alpha_5mer_lookup.npy` |
| 6 | 自动生成 | 都找不到就自动重新生成 |

**指定已有文件：**

```bash
python3 predict_tad.py seq.fa \
  --alpha_lookup /data/alpha_5mer_lookup.npy \
  --cider_lookup /data/cider_5mer_lookup.npy
```

**仅生成查找表（不预测）：**

```bash
# 生成 alphaPredict 查找表（约 2-3 分钟，~13 MB）
python3 predict_tad.py --gen_alpha

# 生成 localCider 查找表（约 10 分钟，~100 MB，需要 localcider）
python3 predict_tad.py --gen_cider
```

**通过环境变量指定（适合批量脚本）：**

```bash
export TADA_ALPHA_LOOKUP=/shared/alpha_5mer_lookup.npy
export TADA_CIDER_LOOKUP=/shared/cider_5mer_lookup.npy
python3 predict_tad.py sequences.fasta -o results.tsv
```

### 命令行参数

```
positional arguments:
  fasta                 输入 FASTA 文件路径

optional arguments:
  -h, --help            显示帮助信息
  -o, --output          输出 TSV 文件路径（不指定则打印到终端）
  --batch_size          GPU 每批处理的序列窗口数（默认 10000）
                        显存不足时减小，大显存可增大到 20000-30000
  --overlap             >40aa 序列的滑动窗口重叠数（默认 39，即步长 1）
                        增大步长可加速但降低分辨率
  --alpha_lookup        指定 alphaPredict 5-mer 查找表路径
  --cider_lookup        指定 localCider 5-mer 查找表路径
  --gen_alpha           仅生成 alphaPredict 查找表并退出
  --gen_cider           仅生成 localCider 查找表并退出
```

---

## 输出格式

输出为 TSV（Tab 分隔）格式，包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| 序列名称 | string | FASTA 文件中的序列标识符 |
| 序列长度 | int | 清洗后的序列氨基酸数 |
| TAD最高分 | float | 滑动窗口中最高的 TAD 预测分数（0-1） |
| 最佳40aa片段 | string | 得分最高的 40 氨基酸窗口序列 |
| 起始位置 | int | 最佳窗口在原序列中的起始位置（1-indexed） |
| 终止位置 | int | 最佳窗口在原序列中的终止位置（含） |

**示例输出：**

```
序列名称	序列长度	TAD最高分	最佳40aa片段	起始位置	终止位置
Seq001	256	0.892341	DEDEDEKRKRKRDEDEDEKRKRKRDEDEDEKRKRKRDEDE	45	84
Seq002	40	0.123456	MFQILRKKKRVKLESLIMNRKSFAQSIENLFALSLLVKDG	1	40
Seq003	35	0.000000	XXXX...	1	35
```

**分数解读：**
- **≥ 0.5**：预测为 TAD（转录激活域）
- **< 0.5**：预测为非 TAD
- **= 0.0**：序列长度 < 40aa，无法预测

**终端汇总输出：**

```
📊 汇总
  总序列数:     698
  TAD (≥0.5):   234 (33.5%)
  非TAD (<0.5): 464 (66.5%)
  分数范围:     0.0012 ~ 0.9876
  平均分数:     0.2345
  中位数分数:   0.1234
  GPU 总耗时:   12.3s
  吞吐量:       57 seqs/s (19,180 windows/s)
```

---

## 文件结构

```
TADA_T2_GPU/
│
├── README.md                           # 本文档
├── setup.sh                            # 一键安装脚本
│
└── src/
    │
    ├── predict_tad.py                  # 主预测脚本（独立运行）
    │   ├── FASTA 解析 + 序列清洗
    │   ├── 查找表自动检测/生成
    │   ├── 滑动窗口切分
    │   ├── kappa/omega 矢量化计算
    │   ├── GPU 批量推理
    │   └── TSV 结果输出
    │
    ├── generate_cider_lookup.py        # localCider 5-mer 查找表生成器
    │   └── 需要 localcider 包，生成约 10 分钟
    │
    └── TADA_T2/
        └── backend/
            ├── features.py             # GPU 特征计算模块（替换原版）
            │   ├── _encode_sequences_to_tensor()   # 序列 → 整数张量
            │   ├── _compute_all_features_gpu()     # GPU 特征计算
            │   ├── _compute_kappa_omega_cpu()      # NumPy 矢量化 kappa/omega
            │   ├── scale_features_predict()        # GPU 特征缩放
            │   └── 查找表自动加载（支持环境变量路径）
            │
            ├── model.py                # CNN-Attention-BiLSTM 模型定义（替换原版）
            │   ├── Attention           # 自定义注意力层（TF2 兼容 + get_config 序列化）
            │   │   ├── build()         # 初始化权重矩阵 W + 偏置 b
            │   │   ├── call()          # tanh → softmax → 加权求和
            │   │   └── get_config()    # 支持模型保存/加载
            │   └── TadaModel           # 模型构建器
            │       └── create_model()  # Input → Conv1D → Dropout → Conv1D →
            │                            #  Dropout → Attention → BiLSTM → BiLSTM → Dense(softmax)
            │
            └── predictor.py            # GPU 加速推理器（替换原版）
                ├── predict_tada()      # 全链路 GPU 推理入口
                │   ├── create_features()   # 特征计算（GPU Tensor）
                │   ├── scale_features()    # 特征缩放（GPU Tensor）
                │   ├── model.predict()     # 模型推理（GPU Tensor）
                │   └── 返回结果时才拷回 CPU
                ├── _model_cache        # 全局模型缓存，避免重复加载
                └── get_model_path()    # 获取预训练权重路径
```

安装后（通过 setup.sh）：

```
<python_env>/lib/python3.x/site-packages/TADA_T2/
├── __init__.py
├── TADA.py                             # 原版 API（predict, predict_from_fasta）
├── backend/
│   ├── features.py                     # ← 已替换为 GPU 版
│   ├── model.py                        # ← 已替换为 TF2 兼容版
│   ├── predictor.py                    # ← 已替换为全链路 GPU 版
│   └── utils.py                        # 滑动窗口、padding 工具
└── data/
    ├── tada.14-0.02.hdf5               # 预训练模型权重（1.7 MB）
    ├── scaler_metric.npy               # StandardScaler + MinMaxScaler 参数
    ├── alpha_5mer_lookup.npy           # alphaPredict 查找表（首次运行自动生成）
    └── cider_5mer_lookup.npy           # localCider 查找表（可选）
```


## 修改文件详情

### features.py — GPU 特征计算模块

**修改要点：**

- 移除 localCider 强依赖，kappa/omega 改用 NumPy `searchsorted` 矢量化计算（快 50-100x）
- 特征计算全流程使用 TF Tensor，数据保持在 GPU 上
- 查找表加载支持环境变量路径（`TADA_ALPHA_LOOKUP`、`TADA_CIDER_LOOKUP`）
- 5-mer 编码使用显式 slice + base-20 索引替代 conv1d
- scaler 参数读取列号
- 序列自动清洗，剔除非标准氨基酸

### model.py — CNN-Attention-BiLSTM 模型定义

**修改要点：**

- `Attention` 自定义层更新为 TF2 兼容写法，添加 `get_config()` 方法支持模型序列化/保存/加载
- 使用显式 `layers.Input(shape=...)` 层替代旧版隐式 input_shape，消除 TF2 弃用警告
- `Attention` 权重初始化改用 `glorot_uniform`，提升训练稳定性
- 模型架构：`Input(36,42) → Conv1D(100,2,gelu) → Dropout(0.3) → Conv1D(100,2,gelu) → Dropout(0.3) → Attention → BiLSTM(100) → BiLSTM(100) → Dense(2,softmax)`

### predictor.py — 全链路 GPU 推理器

**修改要点：**

- 数据流完全保持在 GPU：`sequences → create_features (GPU) → scale_features (GPU) → model.predict (GPU)`，仅最终结果拷回 CPU
- 引入全局模型缓存 `_model_cache`，避免每次调用重复加载模型权重
- 使用 `importlib.resources.files` 获取预训练权重路径，兼容现代 Python 包管理
- 支持 `return_both_values` 参数，可获取 softmax 两个输出值（TAD 分数 + 非 TAD 分数）

---

## 性能对比

### kappa/omega 计算

| 方法 | 698 条序列 | 速度提升 |
|------|-----------|---------|
| localCider（原版） | ~60-120 秒 | 1× |
| NumPy `searchsorted`（本版） | ~1-2 秒 | **50-100×** |

### 端到端推理（698 条序列，236K 窗口）

| 阶段 | 耗时 |
|------|------|
| FASTA 解析 + 序列清洗 | < 1 秒 |
| 滑动窗口切分 | < 1 秒 |
| kappa/omega 预计算 | ~1 秒 |
| GPU 批量推理 | ~10-15 秒 |
| **总计** | **~15 秒** |

吞吐量：约 50-60 条序列/秒，约 15,000-20,000 窗口/秒。

在RTX PRO6000（96G 显存）上为植物蛋白质组的每条序列（共约60000条序列）预测TAD_Max，共耗时约1小时

---

## 常见问题

### Q: 首次运行很慢？

首次运行需要生成 `alpha_5mer_lookup.npy`（约 2-3 分钟）。生成后缓存在磁盘，后续运行秒级启动。

### Q: 没有 GPU 能用吗？

可以。TensorFlow 会自动回退到 CPU。速度会慢约 10-50 倍，但仍可正常运行。

### Q: cider_5mer_lookup.npy 是什么？必须有吗？

cider 查找表存储了所有 3,200,000 个 5-mer 的 8 个理化性质（疏水性、电荷等），用于窗口级特征计算。**预测时必须存在**。如果缺失，脚本会尝试自动生成（需要安装 localcider：`pip install localcider`，约 10 分钟）。

### Q: 序列中有非标准氨基酸怎么办？

脚本会自动剔除以下字符：`*` `X` `U` `B` `Z` `J` `O`，仅保留 20 种标准氨基酸（ACDEFGHIKLMNPQRSTVWY）。清洗后如果序列长度 < 40aa，该序列的 TAD 分数会输出 0.0。

### Q: batch_size 怎么选？

- **4GB 显存**：`--batch_size 2000`
- **8GB 显存**：`--batch_size 5000`
- **16GB 显存**：`--batch_size 10000`（默认）
- **24GB+ 显存**：`--batch_size 20000-30000`

### Q: 输出结果和原版一致吗？

特征计算逻辑与原版 localCider 版本等价（使用预计算的 5-mer 查找表，数值精度 < 1e-6）。kappa/omega 使用相同公式的 NumPy 实现，理论上结果一致。

### Q: 如何用 Python API 调用？

安装后可以直接 import：

```python
from TADA_T2.TADA import predict, predict_from_fasta

# 预测单条序列
result = predict("MFQILRKKKRVKLESLIMNRKSFAQSIENLFALSLLVKDGFQILRKKK")

# 从 FASTA 文件预测
result = predict_from_fasta("sequences.fasta")
```

---

## 许可证

原始 TADA_T2 采用 [MIT License](https://github.com/ryanemenecker/TADA_T2/blob/main/LICENSE)。

本修改版遵循相同的 MIT 许可证。

---

## 致谢

- [TADA_T2](https://github.com/ryanemenecker/TADA_T2) — 原始仓库，Ryan Emenecker, Holehouse Lab WUSM
- [alphaPredict](https://github.com/holehouse-lab/alphaPredict) — 二级结构预测
- [localCider](https://github.com/Pappulab/localCider) — 理化性质计算
## 联系方式
xiaouyuxiaofen@petalmail.com
