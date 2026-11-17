# SQ2 交接说明（组员 C → 组长 A / 组员 D）

> 交付日期 2026-11-17。分支 `feat/sq2-label-noise`，实验脚本、单测、表格、图与日志已随本文档一并提交；预测 parquet 与 checkpoint 按仓库约定不入 Git（见 §4/§5 复跑）。全部产物已落盘，可从预测 parquet 免训练重建表与图（`--summarize` 已验证）。执行矩阵与验收标准见 [SQ2_EXECUTION_PLAN_zh.md](./SQ2_EXECUTION_PLAN_zh.md)。

## 1. 一段话结论（给组长 A）

- **EN**: On the full 1,491,887-image OOS test set (2000–2019), margin-filtering training samples monotonically hurt Accuracy (52.33%→51.34%), CE (0.6924→0.7131), AUC (0.5322→0.5170) and ECE (0.0166→0.0786), and C2−C0 monthly long-short returns are significantly negative (−0.37%, 95% CI [−0.60, −0.18]); label smoothing 0.1 is the only variant improving both CE (−0.0003) and ECE (0.0166→0.0137) with slightly better Sharpe (1.70→1.74, not significant); on 1,471,746 common images, accuracy ranks 60d (52.52%) > 20d (52.28%) ≈ 5d (52.27%) while 20d has the best AUC (0.5315) and per-period long-short return (1.73% vs 1.62%/1.04%).
- **中文**：149 万张样本外图上，按收益幅度过滤训练样本单调损害 Acc/CE/AUC/ECE 全部四指标（C2−C0 月均多空收益差 −0.37%，95% CI [−0.60, −0.18]，显著为负）；label smoothing 0.1 是唯一同时改善 CE 与 ECE 的变体（ECE 0.0166→0.0137，Sharpe 1.70→1.74，不显著）；共同样本 1,471,746 张图上 Acc 60d(52.52%) > 20d(52.28%) ≈ 5d(52.27%)，但 AUC 与单期多空收益均以 20d 最高（0.5315、1.73%）。

## 2. 核心数字表

### 2.1 八配置主表（完整 OOS 测试集，`sq2_noise_and_horizon.csv`）

| 编号 | 训练条件 | 训练数（保留率） | Acc % | CE | AUC | ECE | LS Sharpe (t) | 月均 LS % | 最优轮 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 20d 全样本（复用 seed42） | 482,926 (100%) | 52.33 | 0.6924 | 0.5322 | 0.0166 | 1.70 (7.59) | 1.78 | 10 |
| C1 | 20d \|Retx\|≥1% | 429,880 (89.02%) | 52.13 | 0.6940 | 0.5303 | 0.0265 | 1.80 (8.05) | 1.61 | 7 |
| C2 | 20d \|Retx\|≥3% | 349,785 (72.43%) | 51.46 | 0.6987 | 0.5259 | 0.0454 | 1.76 (7.89) | 1.40 | 6 |
| C3 | 20d \|Retx\|≥5% | 283,161 (58.63%) | 51.34 | 0.7131 | 0.5170 | 0.0786 | 1.35 (6.04) | 0.97 | 3 |
| C4 | 20d \|t-stat\|≥0.5 | 455,654 (94.35%) | 51.68 | 0.6942 | 0.5283 | 0.0262 | 1.58 (7.08) | 1.59 | 7 |
| C5 | 20d 全样本 + 平滑 0.1 | 482,926 (100%) | 52.35 | 0.6921 | 0.5324 | 0.0137 | 1.74 (7.77) | 1.80 | 10 |
| C6 | 5d 全样本 | 485,346 (100%) | 52.28 | 0.6976 | 0.5299 | 0.0419 | 2.24 (10.02) | 1.07 | 4 |
| C7 | 60d 全样本 | 475,880 (100%) | 52.52 | 0.6926 | 0.5296 | 0.0190 | 1.23 (5.49) | 1.62 | 14 |

测试集：20d 1,491,887 / 5d 1,498,816 / 60d 1,471,746；验证集：206,968 / 208,005 / 203,949。月均 LS = 逐调仓月 top10%−bottom10% 等权多空（20d 组按月，含 √12 年化的 Sharpe）。

### 2.2 共同样本三周期对比（1,471,746 张图键交集，`sq2_common_sample_horizon.csv`）

| 周期 | 上涨占比 % | Acc % | CE | AUC | ECE | LS Sharpe | 单期 LS % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 5d (C6) | 50.46 | 52.27 | 0.6976 | 0.5298 | 0.0420 | 2.20 | 1.04 |
| 20d (C0) | 50.55 | 52.28 | 0.6925 | **0.5315** | 0.0170 | 1.66 | **1.73** |
| 60d (C7) | 51.51 | **52.52** | 0.6926 | 0.5296 | 0.0190 | 1.23 | 1.62 |

### 2.3 市值分组训练保留率（C0 训练池市值五分位，`sq2_filter_retention_by_cap.csv`）

| 组 | C1 τ=1% | C2 τ=3% | C3 τ=5% | C4 \|t\|≥0.5 |
|---|---:|---:|---:|---:|
| Q1（小盘） | 87.16 | 75.73 | 64.28 | 90.00 |
| Q5（大盘） | 89.11 | 69.52 | 53.11 | 97.78 |

要点：固定收益阈值 τ 越严，大盘股被删越多（C3 从 64.28% 降到 53.11%——大盘波动小、|Retx| 小）；t-stat 过滤方向相反，大盘保留 97.78%（按波动标准化后信噪比高）。过滤同时改变样本量与股票构成。

### 2.4 成对分块 bootstrap（240 个月、块长 3、2000 次，`sq2_paired_bootstrap.csv`）

| 对比 | 月均 LS 差 % | 95% CI | 结论 |
|---|---:|---|---|
| C2−C0 | **−0.373** | [−0.603, −0.179] | **显著为负** |
| C5−C0 | +0.023 | [−0.108, +0.164] | 不显著 |

⚠️ 表中 `obs_sharpe_diff`（C2−C0 为 −0.74）是**收益差序列自身的 Sharpe**（√12·mean(diff)/std(diff)），**不是**两个模型 Sharpe 之差（那会是 1.76−1.70=+0.07）。报告与讲稿必须写清这个区别，否则与主表 Sharpe 列自相矛盾。

## 3. 结果解读要点（供报告第 4 节）

1. **边际过滤：单调损害。** τ 从 0→5%，Acc 52.33→51.34、CE 0.6924→0.7131、AUC 0.5322→0.5170、ECE 0.0166→0.0786 单调变差；样本越少，早停越早（ep 10→3），欠拟合加重。C1/C2 的 LS Sharpe（1.80/1.76）略高于 C0（1.70），但成对分块 bootstrap 显示 C2−C0 月均多空收益差 −0.37%（CI [−0.60, −0.18]）显著为负——Sharpe 点值的小优势不成立，过滤未带来更好的概率、排序或收益。C4（t-stat，保留 94.35%）与 C1–C3（89%/72%/59%）保留率不可比，且同样全面差于 C0。
2. **label smoothing 0.1：唯一一致的小改善。** CE 0.6924→0.6921、ECE 0.0166→0.0137（−0.0029），Acc/AUC/Sharpe/月均收益同向微升（1.70→1.74，1.78→1.80），bootstrap CI 跨 0 不显著。方向上"软化硬标签"优于"删疑似噪声样本"。
3. **周期：准确率与经济收益不同序。** 共同样本上 Acc 60d(52.52%) > 20d(52.28%) ≈ 5d(52.27%)（60d 上涨占比也最高 51.51%，部分反映基率差异）；AUC 20d 最高（0.5315）；单期多空收益 20d(1.73%) > 60d(1.62%) > 5d(1.04%)。跨周期 Sharpe（2.24/1.66/1.23）只作描述：5d 每月仅一次信号（月内不重复调仓）、60d 相邻持有期重叠（收益自相关），不可解释为可实施策略的优劣排序。
4. **局限。** 单种子（seed 42）；固定阈值同时改变训练样本量与构成（见 2.3），差异不能归因于"去噪"单一机制；C4 保留率与 C1–C3 不可比；60d 前瞻窗口跨越 1999/2000 分界（训练池尾部图像的标签部分落在测试期），切分按图像日期为准；跨周期 Sharpe 的可比性见第 3 点。

## 4. 文件清单（交 A/D）

| 文件 | 用途 |
|---|---|
| `outputs/predictions/sq2_c{1..7}_*.parquet`（7 个） | C1–C7 样本外测试预测（标准 13+2 列契约，C6/C7 含本周期收益列） |
| `outputs/predictions/baseline_seed42.parquet` | 兼作 C0 测试预测（勿改） |
| `outputs/predictions/val_sq2_c{0..7}_*.parquet`（8 个） | 验证集预测（未过滤；C0 由既有权重重建；供 A 做 SQ3 校准/后续分析） |
| `outputs/checkpoints/sq2_*.pt`（7 个）+ `baseline_seed42.pt` | 最优验证权重（C0 为原始基线；供 D 重推理/Grad-CAM 对照） |
| `outputs/tables/sq2_run_manifest.csv` | 8 行运行元数据（训练数、保留率、最优轮、耗时；每 exp 取最后一行） |
| `outputs/tables/sq2_noise_and_horizon.csv` | 主表（§2.1 来源） |
| `outputs/tables/sq2_common_sample_horizon.csv` | 共同样本三周期表（§2.2） |
| `outputs/tables/sq2_filter_retention_by_cap.csv` | 市值分组保留率（§2.3） |
| `outputs/tables/sq2_paired_bootstrap.csv` | 成对 bootstrap（§2.4，注意 obs_sharpe_diff 定义） |
| `outputs/figures/fig2_sq2_noise_horizon.pdf` / `.png`(300dpi) | 报告图 2： 过滤权衡 + (b) 周期与平滑 |
| `logs/sq2_gpu0.log`、`logs/sq2_gpu1.log`、`logs/sq2_smoke_c2.log` | 双 RTX 3090 训练全程日志（含逐轮 loss 与 OOS 行） |
| `experiments/exp_sq2_label_noise.py`、`tests/test_sq2_horizon_fallback.py`、`tests/conftest.py` | 实验脚本与单测（已提交至本分支） |

## 5. 复跑命令

```bash
# ① 数据准备 + 环境自检（详见 SQ2_EXECUTION_PLAN_zh.md §1）
bash scripts/prepare_sq2_data.sh
uv sync --locked --python 3.12 --extra dev
uv run python scripts/verify_env.py && uv run pytest -q

# ② 全矩阵训练（双 GPU 并行；已有预测会被跳过，加 --force 才重训）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/exp_sq2_label_noise.py \
    --exp C1,C2,C3,C4 > logs/sq2_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp_sq2_label_noise.py \
    --exp C5,C6,C7 > logs/sq2_gpu1.log 2>&1 &
wait

# ③ 免训练重建全部表与图 2（从已存预测 + manifest）
.venv/bin/python experiments/exp_sq2_label_noise.py --summarize

# ④ 单实验强制重跑（示例 C2）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/exp_sq2_label_noise.py --exp C2 --force
```

## 6. 自检记录

**交付前校验（2026-11-17，本次执行）：129/129 PASS，0 FAIL。**
脚本逻辑：16 个 parquet（8 实验 × test/val）× 7 项逐文件检查 + 17 项跨文件/数据池重建检查。逐项结果：

| 检查项 | 范围 | 结果 |
|---|---|---|
| 1 主键 (year,row_idx) 唯一 | 16 个文件 | PASS（0 重复） |
| 2 prob_up 有限且 ∈[0,1]；logit_down/logit_up 存在且全有限 | 16 个文件 | PASS |
| 3 y_true ∈{0,1} | 16 个文件 | PASS |
| 4 标签对齐原始 feather（每年随机 ≥500 行；C6 用 Retx_5d_label、C7 用 Retx_60d_label、其余 Retx_20d_label） | 16 个文件（test 每个抽样 1 万行 / val 3,500 行） | PASS，0 错配 |
| 5 C0–C5 test 键集完全一致且 = baseline_seed42 键；val 键集一致（设计上未过滤） | 6×2 文件 | PASS |
| 6 C6/C7 含本周期 Retx_{h}/Ret_{h} 列 | 4 个文件 | PASS |
| 7 行数 = 预期 = manifest（test 1,491,887/1,498,816/1,471,746；val 206,968/208,005/203,949） | 16 个文件 | PASS |
| X 训练池嵌套与划分复算：从原始 feather 经公共 `ChartDataset` + seed42 划分重建训练池键（482,926），按阈值重建 C1/C2/C3/C4 键集——数量与 manifest 逐一相等（429,880/349,785/283,161/455,654），且 C3⊆C2⊆C1⊆池、C4⊆池；三周期 val/test 键集与重建划分完全一致 | 原始数据 | PASS |

说明：val parquet 按设计未过滤、六组键全同，无法单独证明训练池嵌套，故嵌套验证采用上述"原始 feather 重建键集"法（比间接验证更强）。

**此前环境自检（2026-11-17，训练前）**：`verify_env.py` 6/6 PASS（CUDA / 双 RTX 3090）；`pytest` 7/7 通过（含跨周期收益回退单测）；C0 复算五指标与交付基线一致（52.33225% / CE 0.692360 / AUC 0.532183 / ECE 0.016573 / LS Sharpe 1.696175）；分组样本数与执行计划零偏差。

**图 2**：PNG 3110×1190 @300dpi；从当前主表 CSV 重渲染 PNG 与仓库文件 **sha256 逐位一致**，mtime（18:06:33）晚于最后落盘的预测（18:05:31）——图与当前数据完全同步。视觉层面（文字重叠、双轴可读性）本次校验 Agent 不支持图像输入未做，需 A/D 人工目检一次。

## 7. 给 D 的展示素材（5 页内 1 页 + 约 55 秒讲稿提纲）

**一页要点**（标题建议："SQ2 — 监督信号：过滤、平滑与周期"）：
- 左：图 2(a) 缩略——τ↑ 时保留率(灰柱)↓，AUC/Sharpe 双线同步下滑；C4 空心点单列。
- 右：图 2(b) 缩略 + 一行字"smoothing: ECE 0.0166→0.0137"。
- 底部三条结论：① 过滤单调损害四指标，C2−C0 月均 LS −0.37%（CI [−0.60, −0.18]）显著为负；② 平滑 0.1 唯一改善 CE/ECE，收益微升不显著；③ 共同样本 Acc 60d>20d≈5d，但 AUC 与单期 LS 收益 20d 最高；跨周期 Sharpe 仅描述。
- 局限一行：单种子；固定阈值同时改样本量与构成；60d 窗口跨 1999/2000 界（按图像日期切分）。

**约 55 秒讲稿提纲**：
1. (≈10s) 问题：图像与网络固定，监督信号怎么选？三个旋钮——训练样本过滤、标签平滑、预测周期。
2. (≈10s) 设计：同一 CNN、同一 1993–1999 训练/验证切分，只改监督：|Retx_20d|≥1/3/5%、|t|≥0.5、平滑 0.1、5d/60d 标签；验证与测试集永不过滤。
3. (≈12s) 结果一：过滤越狠，Acc/CE/AUC/ECE 全线越差；C2 的 Sharpe 看起来略高，但逐月成对 bootstrap 显示 C2−C0 多空收益差 −0.37%、显著为负——删"疑似噪声"样本没有收益。
4. (≈10s) 结果二：把硬标签软化到 0.95/0.05，是唯一同时改善 CE 和校准（ECE 0.0166→0.0137）的改动，收益略优但不显著。
5. (≈10s) 结果三：换周期，60d 方向最准（52.52%），但 20d 排序最强（AUC 0.5315）、单期赚得最多（1.73%）；5d 月内只有一次信号、60d 持有期重叠，Sharpe 只作描述。
6. (≈3s) 收尾：结论——全样本 + 轻度平滑最稳；单种子，小差异只作描述。
