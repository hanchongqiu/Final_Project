# 组员 C 专属任务书：SQ2 低信噪比标签噪声、边际过滤与预测周期分析

> **负责人**：组员 C  
> **前置必读**：[README.md](../../README.md)（先配好环境）｜ [DATA_AND_CONTRACTS.md](../DATA_AND_CONTRACTS.md) ｜ [REPORT_BLUEPRINT.md](../REPORT_BLUEPRINT.md)  
> **工程分支**：`feat/sq2-label-noise`

---

## 1. 你的任务是什么？为什么它最体现《统计机器学习》功底？

在猫狗图像分类里，猫就是猫、狗就是狗，标签几乎 100% 确定；但在金融市场里，原论文用 `Retx_20d > 0` 把连续收益率一刀切成 `0/1` 二分类硬标签：
* 股票甲未来 20 天涨了 **`+0.02%`** $\to$ 标签为 **`1`（看涨）**；
* 股票乙未来 20 天跌了 **`-0.02%`** $\to$ 标签为 **`0`（看跌）**。

从统计学习视角看，零附近的微弱涨跌几乎全是市场微观结构随机噪声（Aleatoric Noise）！强迫高容量神经网络用交叉熵损失去死记硬背这些边界模糊样本，极易导致过拟合。

你的模块（报告 **Section 4**）专门研究**“监督信号层面的去噪与正则化”**：
1. **训练集大间隔边际过滤（Margin Filtering）**：如果在**训练集**中剔除 $|\text{Retx}_{20d}| < \tau$ 的零附近微弱波动样本，只让 CNN 学习走势明确的样本，样本外泛化会变好还是变差？（注：当 $\tau$ 从小变大时，噪声偏差减小，但有效训练样本量也随之减少——这正是教科书级的 **偏差-方差权衡 Bias-Variance Trade-off** 实验！无论结果是单调变化还是“倒 U 型先升后降”，都是极佳的科学结论！）
2. **软标签正则化（Label Smoothing）**：不删样本，而是将硬标签 `[0, 1]` 软化为 `[0.05, 0.95]`（`label_smoothing=0.1`），防止模型对含噪金融标签过度自信。
3. **预测前瞻周期对比（5d vs. 20d vs. 60d Horizon）**：同一张过去 20 天的 K 线图，用来预测未来 **5 天、20 天、60 天** 的涨跌（`.feather` 文件中已自带 `Retx_5d_label` 和 `Retx_60d_label`！），视觉信号的半衰期有多长？

---

## 2. ⚠️ 必须严守的两条实验红线（防逻辑 Bug）

1. **过滤字段必须用 `Retx_20d`，绝不能用 `Ret_20d`**：如 [DATA_AND_CONTRACTS.md](../DATA_AND_CONTRACTS.md) 第 2 节实测指出的，标签 `Retx_20d_label` 是由除息收益 `Retx_20d` 定义的。做阈值过滤时必须写 `df["Retx_20d"].abs() >= tau`（或波动率标准化版 `df["Retx_20d_tstat"].abs() >= tau_t`）。
2. **只过滤训练集，验证集与 `2000–2019` 测试集绝对禁止过滤！**
   * 真实交易部署时你不可能提前知道未来 20 天收益绝对值大不大，且不同模型必须在**完全相同的 1,491,887 条样本外测试集**上评估，指标才具有可比性！

---

## 3. 你的受控实验矩阵 (`experiments/exp_sq2_label_noise.py`)

全部固定使用组长写好的标准架构 `CNN20D`、`seed = 42`、`Adam(lr=1e-5)` 与验证集早停 `patience = 2`：

| 实验编号 | 实验组名称 | 训练集 (`1993–1999` 70% Train) 改动 | 监督标签列 | 目的说明 |
| :--- | :--- | :--- | :---: | :--- |
| **C0 (复用)** | **Baseline ($\tau = 0\%$)** | 不剔除（100% 训练样本，`482,925` 条） | `Retx_20d_label` | 直接复用组长跑好的 `baseline_seed42.parquet` |
| **C1** | **Margin $\tau = 1\%$** | 仅保留 `abs(Retx_20d) >= 0.01` 的训练样本 | `Retx_20d_label` | 轻度剔除零附近极微弱噪声 |
| **C2** | **Margin $\tau = 3\%$** | 仅保留 `abs(Retx_20d) >= 0.03` 的训练样本 | `Retx_20d_label` | 中度边际过滤 |
| **C3** | **Margin $\tau = 5\%$** | 仅保留 `abs(Retx_20d) >= 0.05` 的训练样本 | `Retx_20d_label` | 激进过滤（观察样本量缩减与高波动小盘股偏移带来的方差效应） |
| **C4** | **Vol-Adjusted Margin** | 仅保留波动率标准化 `abs(Retx_20d_tstat) >= 0.5` 的训练样本 | `Retx_20d_label` | **亮点实验**：固定百分比阈值会误删低波动大盘蓝筹股，用 $t$-stat 过滤实现跨股票波动率公平去噪！ |
| **C5** | **Label Smoothing** | 保留 100% 训练样本，损失改为 `CrossEntropyLoss(label_smoothing=0.1)` | `Retx_20d_label` | 检验软标签正则化对 CE / ECE / Sharpe 的改善 |
| **C6 & C7** | **Horizon 5d & 60d** | 保留 100% 样本，分别以 `Retx_5d_label`（对应回测列 `Ret_5d`）和 `Retx_60d_label`（对应回测列 `Ret_60d`）训练与评估 | `Retx_5d_label` / `Retx_60d_label` | 揭示 20 日视觉形态在短周期（5d 强反转）与长周期（60d 衰减）上的可预测性演化 |

*(注：每次训练仅需约 5–8 分钟，C1–C7 共 7 次轻量训练，总计不到 1 小时即可全部跑完！)*

---

## 4. 你的交付物清单

- [ ] 实验脚本：`experiments/exp_sq2_label_noise.py`
- [ ] 汇总数据表：`outputs/tables/sq2_noise_and_horizon.csv`（记录每个 $\tau$ 下保留的训练样本数、保留比例、正样本比例，以及样本外 Accuracy、CE、AUC、ECE、LS Sharpe）
- [ ] 论文插图 **`outputs/figures/fig2_sq2_noise_horizon.pdf` (& `.png`)**：
  * 左子图 **(a) Margin Filtering Trade-off**：横轴为阈值 $\tau \in \{0\%, 1\%, 3\%, 5\%\}$（附注 `t-stat >= 0.5` 点），左纵轴为 OOS LS Sharpe / AUC 折线，右纵轴背景柱状图为保留的有效训练样本比例（直观呈现 Bias-Variance 权衡）；
  * 右子图 **(b) Label Regularization & Horizon Decay**：展示 Label Smoothing 效果及同一 20d 图像在 `5d / 20d / 60d` 三个预测周期上的 OOS Accuracy 与多空单期收益/Sharpe 对比。
- [ ] 报告撰写：完成英文报告 **Section 4 (RQ2: Supervision under Low SNR，约 1.3 页)** 初稿，并在 [REPORT_BLUEPRINT.md](../REPORT_BLUEPRINT.md) 登记 AI 使用记录。

---

## 5. 给你的 AI 编程助手的一键启动提示词（直接复制发送即可）

```text
请先完整阅读根目录下的 AGENTS.md、docs/DATA_AND_CONTRACTS.md 和 docs/tasks/MEMBER_C_LABEL_NOISE_AND_LOSS.md。
我是组员 C，负责 SQ2（低信噪比标签噪声边际过滤、Label Smoothing 与 5d/20d/60d 多周期对比）。
请严格复用 src/visual_alpha/ 公共组件，帮我实现 experiments/exp_sq2_label_noise.py。特别注意两条铁律：
1) 阈值过滤仅作用于训练集，且必须基于除息收益 Retx_20d（及 Retx_20d_tstat），验证集与 2000-2019 样本外测试集必须保持 100% 未过滤；
2) 全程保持 np.memmap 内存安全并输出符合契约的汇总表与 Figure 2 双栏论文插图。
```
