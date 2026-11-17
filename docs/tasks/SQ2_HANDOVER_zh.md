# SQ2 交接说明（组员 C → 组长 A / 组员 D）

组员 C 为 Dongqing Zhao。英文第 4 节及图 2 已并入[报告 PDF](../../report/main.pdf)（当前第 3–4 页）。[单页 PowerPoint](../../report/sq2_presentation.pptx) 带英文讲者备注，可直接并入组内 Slide 4。[主表](../../outputs/tables/sq2_noise_and_horizon.csv)记录 C0–C7 在各自完整测试集上的表现。

## 1. 给组长的英文摘要句

Excluding training returns below 5% in magnitude reduced test AUC from 0.5322 to 0.5170, while retaining all charts and smoothing labels reduced calibration error from 0.0166 to 0.0137.

## 2. 核心数字表

### 2.1 八配置主表（完整 OOS 测试集，`sq2_noise_and_horizon.csv`）

| 编号 | 训练条件 | 训练数（保留率） | Acc % | CE | AUC | ECE | LS Sharpe (t) | 平均持有期 LS % | 最优轮 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 20d 全样本（复用 seed42） | 482,926 (100%) | 52.33 | 0.6924 | 0.5322 | 0.0166 | 1.70 (7.59) | 1.78 | 10 |
| C1 | 20d \|Retx\|≥1% | 429,880 (89.02%) | 52.13 | 0.6940 | 0.5303 | 0.0265 | 1.80 (8.05) | 1.61 | 7 |
| C2 | 20d \|Retx\|≥3% | 349,785 (72.43%) | 51.46 | 0.6987 | 0.5259 | 0.0454 | 1.76 (7.89) | 1.40 | 6 |
| C3 | 20d \|Retx\|≥5% | 283,161 (58.63%) | 51.34 | 0.7131 | 0.5170 | 0.0786 | 1.35 (6.04) | 0.97 | 3 |
| C4 | 20d \|t-stat\|≥0.5 | 455,654 (94.35%) | 51.68 | 0.6942 | 0.5283 | 0.0262 | 1.58 (7.08) | 1.59 | 7 |
| C5 | 20d 全样本 + 平滑 0.1 | 482,926 (100%) | 52.35 | 0.6921 | 0.5324 | 0.0137 | 1.74 (7.77) | 1.80 | 10 |
| C6 | 5d 全样本 | 485,346 (100%) | 52.28 | 0.6976 | 0.5299 | 0.0419 | 2.24 (10.02) | 1.07 | 4 |
| C7 | 60d 全样本 | 475,880 (100%) | 52.52 | 0.6926 | 0.5296 | 0.0190 | 1.23 (5.49) | 1.62 | 14 |

测试集：20d 1,491,887 / 5d 1,498,816 / 60d 1,471,746；验证集：206,968 / 208,005 / 203,949。每月生成一次信号，按预测概率排序；最高与最低十分位等权收益之差使用本周期收益。Sharpe 使用 √12 年化。

### 2.2 共同样本三周期对比（1,471,746 张图键交集，`sq2_common_sample_horizon.csv`）

| 周期 | 上涨占比 % | Acc % | CE | AUC | ECE | LS Sharpe | 单期 LS % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5d (C6) | 50.46 | 52.27 | 0.6976 | 0.5298 | 0.0420 | 2.20 | 1.04 |
| 20d (C0) | 50.55 | 52.28 | 0.6925 | **0.5315** | 0.0170 | 1.66 | **1.73** |
| 60d (C7) | 51.51 | **52.52** | 0.6926 | 0.5296 | 0.0190 | 1.23 | 1.62 |

### 2.3 市值分组训练保留率（C0 训练池正市值五分位，`sq2_filter_retention_by_cap.csv`）

| 组 | C1 τ=1% | C2 τ=3% | C3 τ=5% | C4 \|t\|≥0.5 |
|---|---:|---:|---:|---:|
| Q1（小盘） | 90.57 | 81.45 | 70.95 | 91.64 |
| Q5（大盘） | 89.11 | 69.49 | 52.98 | 97.86 |

原始 482,926 个训练样本中有 37,285 个负市值，数据说明没有解释负号。本次仅用其余 445,641 个正值排市值五分位。5% 过滤在小盘组保留 70.95%，在大盘组保留 52.98%。这一筛选只用于市值诊断。

### 2.4 成对分块 bootstrap（240 个月、块长 3、2000 次，`sq2_paired_bootstrap.csv`）

| 对比 | 月均 LS 差（百分点） | 95% CI（百分点） | 结论 |
|---|---:|---|---|
| C2−C0 | **−0.373** | [−0.603, −0.179] | **显著为负** |
| C5−C0 | +0.023 | [−0.108, +0.164] | 不显著 |

⚠️ 表中 `obs_sharpe_diff`（C2−C0 为 −0.74）是**收益差序列自身的 Sharpe**（√12·mean(diff)/std(diff)），**不是**两个模型 Sharpe 之差（那会是 1.76−1.70=+0.07）。报告与讲稿必须写清这个区别，否则与主表 Sharpe 列自相矛盾。

## 3. 解读边界

- 四个固定收益阈值下，Accuracy、CE、AUC 和 ECE 随阈值升高逐步变差。Sharpe 依次为 1.70、1.80、1.76、1.35。早停轮次和保留率可描述，无法据此判定欠拟合或分解偏差与方差。
- C2 相对 C0 的 AUC 低 0.0062，Sharpe 高 0.07，月均多空收益低 0.373 个百分点。C5 的 CE 与 ECE 分别低 0.00030、0.00286，月均收益差为 +0.023 个百分点。上表区间只检验月均收益差；单种子实验未估计跨训练种子的变异。
- 跨周期比较使用 §2.2 的共同样本。60d 上涨占比也最高（51.51%）；20d 的 AUC 和单期收益最高。5d 每月仅取一个信号，60d 持有期重叠。按图像日期切分，1999 年末图像的远期标签可能延伸到 2000 年。原基线与新训练的初始化不保证相同。

## 4. 文件清单（交 A/D）

| 文件 | 用途 |
|---|---|
| `outputs/predictions/sq2_c{1..7}_*.parquet`（7 个） | C1–C7 样本外测试预测（标准 11 列，跨周期另加 2 列，C6/C7 含本周期收益列） |
| `outputs/predictions/baseline_seed42.parquet` | 兼作 C0 测试预测（勿改） |
| `outputs/predictions/val_sq2_c{0..7}_*.parquet`（8 个） | 验证集预测（未过滤；C0 由既有权重重建；供 A 做 SQ3 校准/后续分析） |
| `outputs/checkpoints/sq2_*.pt`（7 个）+ `baseline_seed42.pt` | 最优验证权重（C0 为原始基线；供 D 重推理/Grad-CAM 对照） |
| `outputs/tables/sq2_run_manifest.csv` | 8 行运行元数据（训练数、保留率、最优轮、耗时；每 exp 取最后一行） |
| `outputs/tables/sq2_noise_and_horizon.csv` | 主表（§2.1 来源） |
| `outputs/tables/sq2_common_sample_horizon.csv` | 共同样本三周期表（§2.2） |
| `outputs/tables/sq2_filter_retention_by_cap.csv` | 市值分组保留率（§2.3） |
| `outputs/tables/sq2_paired_bootstrap.csv` | 成对 bootstrap（§2.4，注意 obs_sharpe_diff 定义） |
| `outputs/figures/fig2_sq2_noise_horizon.pdf` / `.png`(300dpi) | 图 2(a) 完整 20d 测试集过滤对照；(b) 共同样本周期比较与完整 20d 测试集平滑对照 |
| `logs/sq2_gpu0.log`、`logs/sq2_gpu1.log`、`logs/sq2_smoke_c2.log` | 双 RTX 3090 训练全程日志（含逐轮 loss 与 OOS 行） |
| `experiments/exp_sq2_label_noise.py`、`tests/test_sq2_horizon_fallback.py`、`tests/conftest.py` | 实验脚本与跨周期单测 |

## 5. 复跑命令

本地已有原始数据包 `data/img_data.zip` 与服务器归档 `data/sq2_outputs_20261117.zip`。在新 checkout 中将前者的 `monthly_20d/` 解到 `data/`，从后者恢复 `outputs/predictions/` 和 `outputs/checkpoints/`；保留当前分支里的表、图与 manifest。要复核已有结果，安装依赖并执行第 ③ 步。

```bash
# ① 数据准备 + 环境自检（详见 SQ2_EXECUTION_PLAN_zh.md §1）
bash scripts/prepare_sq2_data.sh
uv sync --locked --python 3.12 --extra dev
uv run python scripts/verify_env.py && uv run pytest -q

# ② 全矩阵训练（双 GPU 并行；已有预测会被跳过，加 --force 才重训）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/exp_sq2_label_noise.py \
    --train-only --exp C1,C2,C3,C4 > logs/sq2_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp_sq2_label_noise.py \
    --train-only --exp C5,C6,C7 > logs/sq2_gpu1.log 2>&1 &
wait

# ③ 免训练重建全部表与图 2（从已存预测 + manifest）
.venv/bin/python experiments/exp_sq2_label_noise.py --summarize

# ④ 单实验强制重跑（示例 C2）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/exp_sq2_label_noise.py --exp C2 --force
```

## 6. 复核记录（2026-11-17）

- 本地自检验证 27 年的 2,196,994 条数据。仓库原有 7 项测试通过。
- 逐行比较全部 16 份预测与原始切分。年份、图像行号、标签、日期和同周期收益一致；收益按预测文件存储精度对齐。主键唯一，概率与 logits 一致。八组训练行数、正例率、验证 CE 和 checkpoint 最优轮次匹配。C0 测试预测的 SHA-256 与准备脚本一致。
- 从已存预测、运行清单和原始标签独立重建八组主表、共同样本表及配对 bootstrap 表，与服务器版本一致。另重算正市值分组。初版图 2(b) 使用三个不同的样本总体，现统一取共同样本；(a) 的 Sharpe 与 AUC 分开标度。
- 双 GPU 的运行清单更新已加锁。两个训练进程可使用 `--train-only`，结束后统一执行一次 `--summarize`。

## 7. 英文讲稿（117 词，约 55 秒）

For SQ2, we kept the same stock charts and CNN and changed how the model learned from small returns. If a stock barely rises or falls, its direction might be hard to predict. We removed those cases from training and tested on every eligible chart. As the cutoff rose to five percent, we kept only fifty-nine percent of training charts, and ranking accuracy, measured by AUC, fell from 0.532 to 0.517. Keeping the charts and softening the labels improved calibration, with no clear gain in average portfolio return. Finally, on the same charts, sixty days gave the highest directional accuracy. Twenty days gave the best ranking and the largest average long–short spread, 1.73 percent per holding period.
