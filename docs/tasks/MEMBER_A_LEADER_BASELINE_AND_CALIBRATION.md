# 组员 A（组长）专属任务书：公共底座 + Baseline 复现 + SQ3 概率校准

> **负责人**：组员 A（组长）  
> **前置必读**：[DATA_AND_CONTRACTS.md](../DATA_AND_CONTRACTS.md) ｜ [REPORT_BLUEPRINT.md](../REPORT_BLUEPRINT.md)  
> **工程分支**：`feat/core-and-sq3`

---

## 1. 你的角色定位：全组“基建底座提供者” + SQ3 研究负责人

你的工作分为前后两个阶段：
1. **阶段一（最优先，解锁全组并行）**：编写精简、健壮的公共包 `src/visual_alpha/`（约 250–300 行代码），并跑通 **Baseline 20-Day CNN**（单模型 `seed=42` + 5 种子集成 `seeds=[42..46]`），将最优权重存入 `outputs/checkpoints/`、样本外预测表存入 `outputs/predictions/`。一旦交付，组员 B、C、D 即可完全独立并行推进！
2. **阶段二（你自己的科研实验 SQ3）**：不需要再训练任何新神经网络！直接基于你阶段一跑出的 5 个种子的验证集与测试集 Logits，完成**深度模型概率校准（Temperature Scaling & Deep Ensemble）与高置信度选择性预测（Selective Prediction）**实验。

---

## 2. 阶段一：公共底座 (`src/visual_alpha/`) 与 Baseline 复现清单

请严格按照 [DATA_AND_CONTRACTS.md](../DATA_AND_CONTRACTS.md) 第 4–5 节规格实现以下模块：

1. **`src/visual_alpha/utils.py`**
   * `get_device()`：自动适配 `cuda` $\to$ `mps` $\to$ `cpu`。
   * `seed_everything(seed)`：锁定 `random`、`numpy`、`torch` 随机种子。
2. **`src/visual_alpha/dataset.py`**
   * 使用 `np.memmap(..., dtype=np.uint8, mode="r").reshape((-1, 64, 60))` 懒加载 `data/monthly_20d/`（切记在 Dataset `__getitem__` 或 worker 初始化中按需打开 memmap，返回 `(1, 64, 60)` 的 `uint8` 张量，在训练循环搬上 GPU 后再 `.float().div_(255.0)`，兼顾零内存膨胀与最高吞吐量）。
   * 默认过滤 `Retx_20d_label != 2`；支持可选参数 `label_col="Retx_20d_label"`、`min_abs_retx: float = 0.0`（供组员 C 用）、`image_mask_mode: str | None = None`（供组员 B 用）以及自定义年份范围 `years`（供组员 D 用）。
   * 标准切分函数：`1993–1999` 有效样本按固定 `split_seed=42` 划分为 70% Train (`482,925`) / 30% Val (`206,969`)；`2000–2019` 全量有效样本 (`1,491,887`) 作为 OOS Test。
3. **`src/visual_alpha/models.py`**
   * 实现精确对齐原论文 `Figure 2` 的 `CNN20D`（3 个 `(5, 3)` 卷积块 + BatchNorm + LeakyReLU(0.01) + `(2, 1)` MaxPool + Dropout(0.5) + `Linear(46080, 2)`，Xavier 初始化）。
4. **`src/visual_alpha/evaluate.py`**
   * 校验预测 DataFrame 是否符合 `.parquet` 契约，并统一计算：`accuracy`、`ce_loss`、`auc`、`ece`（10 等宽分箱）以及基于 `Ret_20d` 按月截面排序的等权十分位多空年化夏普比率 `ls_sharpe`（及 $t$-stat、多头/空头年化收益）。
5. **`experiments/exp_baseline.py`**
   * 训练 `seeds = [42, 43, 44, 45, 46]` 共 5 个独立初始化的 `CNN20D`（`Adam(lr=1e-5)`，`batch_size=256`，`max_epochs=15`，验证集 Cross-Entropy Loss `patience=2` 早停）。
   * 保存权重 `outputs/checkpoints/baseline_seed{s}.pt`、验证集预测 `outputs/predictions/val_baseline_seed{s}.parquet`、样本外预测 `outputs/predictions/baseline_seed{s}.parquet` 以及 5 模型等权概率平均 `outputs/predictions/baseline_ensemble5.parquet`。
   * 生成 **`outputs/tables/tab1_baseline.csv`**（汇报 5 个单模型的 `Mean ± Std` 与 5-Seed Ensemble 的完整指标）。

---

## 3. 阶段二：SQ3 科学问题与实验设计（Probability Calibration & Selective Prediction）

### 核心科学问题 (RQ3)
> 在极低信噪比（样本外整体准确率仅 ~52%–53%）的股票涨跌预测中，现代 CNN 输出的上涨概率 $P(y=1\mid x)$ 是否准确反映了真实的经验胜率？后处理**温度缩放（Temperature Scaling）**与**多模型深度集成（Deep Ensemble）**能否改善概率校准度（ECE）？当我们只交易模型最确信的尾部样本（Selective Prediction）时，准确率与经济收益如何演化？

### 受控实验矩阵 (`experiments/exp_sq3_calibration.py`)

| 实验组 | 方法定义 | 评估对象 (`2000–2019` Full OOS) |
| :--- | :--- | :--- |
| **M1: Uncalibrated Single CNN** | 单模型 (`seed=42`) 原始 Softmax 概率 $p = \sigma(z_1 - z_0)$ | OOS Accuracy, CE, AUC, ECE, LS Sharpe |
| **M2: Temp-Scaled Single CNN** | 在 `1993–1999` **30% 验证集**上通过 L-BFGS 最小化 NLL 拟合单一标量温度 $T > 0$（严禁触碰测试集！），测试集输出 $\text{Softmax}(z / T)$ | 检验参数化温度校准对单模型过度自信的修正效果（注：单调变换不改变排序 AUC/Sharpe，但显著改善 CE 与 ECE） |
| **M3: 5-Seed Deep Ensemble** | 5 个随机初始化模型测试集预测概率的算术平均 $\bar{p} = \frac{1}{5}\sum_{k=1}^5 p^{(k)}$ | 检验方差缩减（Variance Reduction）是否同时提升校准度（ECE）与尾部选股夏普比率（Sharpe） |
| **M4: Selective Prediction 曲线** | 按置信度 $|p - 0.5|$ 从高到低排序，在覆盖率 $\text{Coverage} \in \{100\%, 80\%, 50\%, 30\%, 20\%, 10\%, 5\%\}$ 下计算条件准确率 | 揭示“整体微弱优势（~53%）如何在高置信尾部转化为显著预测优势（如 56%+）”的统计机制 |

---

## 4. 你的交付物清单

- [ ] 公共包代码：`src/visual_alpha/{__init__,utils,dataset,models,trainer,evaluate}.py` 及基础单元测试
- [ ] 实验脚本：`experiments/exp_baseline.py`、`experiments/exp_sq3_calibration.py`
- [ ] 表格：`outputs/tables/tab1_baseline.csv`、`outputs/tables/sq3_calibration_summary.csv`
- [ ] 论文插图 **`outputs/figures/fig3_sq3_calibration.pdf` (& `.png`)**：
  * 左图 **(a) Reliability Diagram & Probability Histogram**：对比 Single CNN、Temp-Scaled CNN 与 5-Seed Ensemble 的校准曲线与预测概率窄分布直方图；
  * 右图 **(b) Accuracy–Coverage Curve (Selective Prediction)**：展示 Single vs. Ensemble 在不同置信度覆盖率下的样本外准确率递增曲线。
- [ ] 报告撰写：Section 1（引言与科学问题）、Section 2（数据、架构与评估协议 + Table 1）、Section 5（RQ3 校准与选择性预测 + Figure 3）。

---

## 5. 给你的 AI 编程助手的一键启动提示词（直接复制发送即可）

```text
请先完整阅读根目录下的 AGENTS.md、docs/DATA_AND_CONTRACTS.md 和 docs/tasks/MEMBER_A_LEADER_BASELINE_AND_CALIBRATION.md。
我是组员 A（组长）。现在我们开始正式开发：
1. 请先按照 docs/DATA_AND_CONTRACTS.md 的精确规格实现 src/visual_alpha/ 公共底座包（utils.py, dataset.py, models.py, trainer.py, evaluate.py），确保严格使用 np.memmap 懒加载、自动适配 cuda/mps/cpu 设备、正确验证 CNN20D 的 FC 46080 维度，并写一个简短的 pytest 测试验证数据切片、模型前向传播和 evaluate_predictions 指标计算无误。
2. 验证通过后，再帮我编写 experiments/exp_baseline.py 和 experiments/exp_sq3_calibration.py。
```
