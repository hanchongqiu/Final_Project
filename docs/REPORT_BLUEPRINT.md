# 8 页 NeurIPS 技术报告写作蓝图与图表规范 (`REPORT_BLUEPRINT.md`)

> **定位**：本文档规定期末 8 页英文技术报告（NeurIPS 格式）的**精确章节标题、页数预算、图表设计规格、5 分钟视频脚本大纲**以及课程强制要求的 **AI 使用记录表**。  
> **格式要求**：英文撰写（以下采用“英文标准章节标题 + 中文写作要点指导”），总篇幅**严格 $\le 8$ 页**（含参考文献与附录性声明，建议正文 + 图表控制在 7.2 页，参考文献 + 分工与 AI 声明占 0.8 页）。

---

## 1. 论文拟定标题 (Proposed Title)

> **Deconstructing Visual Alpha: An Empirical Audit of Inductive Bias, Label Noise, Calibration, and Concept Drift in Financial Chart CNNs**

*副标题/破题思路*：不把 CNN 图像选股当成神秘黑盒，而是从统计机器学习（Statistical Machine Learning）的四个基本维度——**函数空间归纳偏置、低信噪比监督信号、概率校准与时序分布漂移**——对其进行系统性拆解与验证。

---

## 2. 逐节篇幅分配与写作提纲（总计 8.0 页）

### Abstract (~150–180 words, 0.25 Page) — *负责人：组员 D 初稿 + 组长 A 定稿*
* **四句话结构**：
  1. **背景**：Jiang, Kelly, & Xiu (2023) 提出利用 2D 卷积神经网络从 $64 \times 60$ 股票 OHLC 价格与成交量图中提取价格趋势预测信号（Visual Alpha）。
  2. **问题**：在极低信噪比（Low SNR）与高度非平稳的金融市场中，这种视觉预测能力究竟源自何处、受何种统计瓶颈制约，仍缺乏系统性的机器学习诊断。
  3. **方法**：我们在 1993–2019 年超过 219 万张美股月度图像样本上构建了统一评测基准，围绕四个核心科学问题（2D 空间归纳偏置与遮挡消融、零收益附近标签噪声边际过滤、预测概率校准与选择性预测、跨时代概念漂移与 Grad-CAM 归因）展开受控实验。
  4. **核心发现**：用 2~3 个最亮眼的定量实验数字收尾（待实验跑出后填入，如 Ensemble 提升 Sharpe、近期像素主导预测、现代市场漂移幅度等）。

---

### Section 1: Introduction & Scientific Questions (~0.75 Page) — *负责人：组员 A*
* **1.1 Motivation**：简述传统量化动量/反转线性因子与图像化表征的区别；指出金融预测具有 **极低信噪比（样本外准确率仅略高于 50%）** 与 **机制演化（市场微观结构变迁）** 两大特征。
* **1.2 Four Research Questions (RQs)**：用清晰的项目符号列出我们沿着 ML Pipeline 提出的 4 个可证伪科学问题（RQ1–RQ4），并画一小段话点明全篇结构。

---

### Section 2: Dataset, Architecture & Unified Evaluation Protocol (~1.0 Page) — *负责人：组员 A*
* **2.1 Visual Representation & Temporal Split**：
  * 描述 $64 \times 60$ 二值图像构造（Rows `0:51` OHLC+MA，Row `51` 分隔带，Rows `52:64` 成交量；Cols `0:60` 对应 20 个交易日 $\times$ 3 列）。
  * 明确严格防穿越切分：`1993–1999`（有效样本 `689,894`，70/30 训练/验证）与 `2000–2019`（有效样本 `1,491,887` 样本外测试）；特别说明 `Retx_20d_label`（除息分类监督）与 `Ret_20d`（含息真实组合收益）的严谨对齐。
* **2.2 Baseline CNN & Evaluation Metrics**：
  * 简述 3 层卷积 + LeakyReLU + MaxPool + Dropout(0.5) + `FC 46080` 基线结构与早停机制。
  * 定义全篇统一四维指标：OOS Accuracy、Cross-Entropy / AUC、10-bin ECE、以及等权多空十分位组合年化夏普比率（Equal-Weighted Long-Short Decile Sharpe Ratio）。
* **插入 `Table 1`**：Baseline 单种子均值±标准差 vs. 5-Seed Ensemble 样本外复现基准表（验证我们的底座成功复现了原论文的样本外预测效应）。

---

### Section 3: RQ1 — Architectural Inductive Bias & Visual Component Ablation (~1.4 Pages) — *负责人：组员 B*
* **科学假设 (Hypothesis)**：如果预测信号仅仅是简单的线性动量或全局像素密度，传统扁平化统计学习模型（PCA + Logistic Regression 或手工像素分区统计 + LightGBM）应能匹敌 CNN；反之，若 2D 局部多尺度形态至关重要，2D-CNN 应显著优于扁平模型与纯时序 1D-CNN，但容量过大的通用视觉骨干（如 ResNet）可能在低信噪比下过拟合。
* **实验分析**：
  1. **跨模型族归纳偏置对比**（PCA+LR vs. Pixel-Stats LightGBM vs. 1D-CNN vs. Baseline 2D-CNN vs. ResNet-18）：结合课程所学的**模型复杂度与归纳偏置（Bias-Variance & Inductive Bias）**理论分析为何适度容量的定制非对称卷积核 `(5, 3)` 最契合金融图表。
  2. **输入成分遮挡消融（Zero-Shot / Retrained Masking）**：对比遮挡成交量柱（Rows `52:64`）、遮挡前 10 天（Cols `0:30`）与遮挡最近 10 天（Cols `30:60`）后的准确率与 Sharpe 衰减，定量回答“成交量重要还是价格重要？近期 K 线重要还是远期基底重要？”。
* **插入 `Table 2` + `Figure 1`**（双子图：左图为不同模型族 OOS AUC & Sharpe 对比；右图为不同区域遮挡后的性能衰减柱状图）。

---

### Section 4: RQ2 — Supervision under Low SNR: Margin Filtering, Loss Regularization & Horizon (~1.3 Pages) — *负责人：组员 C*
* **科学假设 (Hypothesis)**：在二值化方向标签 $y = \mathbb{I}(\text{Retx}_{20d} > 0)$ 下，$+0.05\%$ 与 $-0.05\%$ 本质上都是市场微观噪声却被赋予完全相反的硬标签（Aleatoric Label Noise）。在训练集中剔除 $|\text{Retx}_{20d}| < \tau$ 的模糊边界样本，能否起到类似大间隔分类器（Large-Margin Classifier）的去噪作用？
* **实验分析**：
  1. **偏差-方差权衡（Bias-Variance Trade-off）定量揭示**：展示训练集过滤阈值 $\tau \in \{0\%, 1\%, 3\%, 5\%\}$（及波动率标准化阈值 $|\text{Retx\_20d\_tstat}|$）在**全量未过滤 2000–2019 测试集**上的表现。无论是观察到适度 $\tau$ 提升泛化（倒 U 型峰值），还是大 $\tau$ 因有效训练样本锐减且丢弃低波动大盘股导致方差膨胀/分布偏移，都用统计学习理论给出严谨推演！
  2. **软标签正则化与预测周期衰减**：对比 Label Smoothing（$\varepsilon = 0.1$）对过拟合的抑制作用，以及固定同一 20 日图像预测 `5d / 20d / 60d` 标签时信号随前瞻窗口拉长的衰减规律。
* **插入 `Figure 2`**（双子图：左图为随阈值 $\tau$ 变化的保留训练样本比例 vs. OOS Sharpe/AUC 双轴曲线；右图为 `5d / 20d / 60d` 预测衰减与 Label Smoothing 对比）。

---

### Section 5: RQ3 — Probability Calibration, Ensembling & Selective Prediction (~1.2 Pages) — *负责人：组员 A*
* **科学假设 (Hypothesis)**：现代神经网络往往存在概率过度自信（Overconfidence），而在金融组合构建中，输出的预测概率直接决定仓位权重或尾部选股阈值。多随机种子深度集成（Deep Ensemble）与验证集温度缩放（Temperature Scaling）能否显著降低 ECE，并在“仅交易最高置信度 Top-$k\%$ 样本”的选择性预测（Selective Prediction）场景下释放更高胜率？
* **实验分析**：
  1. 对比 Single CNN、Temperature-Scaled CNN 与 5-Seed Ensemble 的预测概率直方图与可靠性曲线（Reliability Diagram），解释金融低信噪比下预测概率聚集在 `[0.42, 0.58]` 窄区间的统计合理性。
  2. **Risk-Coverage / Selective Prediction 曲线**：展示当只对 $|p - 0.5|$ 最大的前 `100% -> 50% -> 20% -> 10% -> 5%` 样本做判断时，单调递增的样本外分类准确率，直接从统计视角解释“为什么全市场整体准确率仅 ~53%，但多空十分位组合（仅取首尾各 10%）却能获得可观夏普比率”。
* **插入 `Figure 3`**（双子图：左图为 10-bin Reliability Diagram & ECE 对比；右图为 Coverage vs. Selective Accuracy 曲线）。

---

### Section 6: RQ4 — Temporal Concept Drift & Grad-CAM Visual Attribution (~1.4 Pages) — *负责人：组员 D*
* **科学假设 (Hypothesis)**：1993–1999 年美股处于分式报价（$1/8$ 美元最小报价单位）与人工做市商时代，而 2000 年十进制化改革与 2010 年代高频算法交易普及深刻改变了短期量价动力学（Temporal Concept Drift）。用现代数据（`2006–2012`）重训模型能否缓解 `2013–2019` 上的概念漂移？CNN 在像素层面究竟聚焦于何种经济规律？
* **实验分析**：
  1. **跨时代泛化矩阵**：在公共测试区间 `2013–2019` 上对比 `Model_1990s (train 1993–1999)`、`Model_Modern (train 2006–2012)` 及两者混合/对比表现，辅以按年份滚动的 OOS Accuracy / Sharpe 时序图（检验 2000 互联网泡沫、2008 次贷危机及 2010s 量化时代的阶段性表现）。
  2. **Grad-CAM 个案与群体统计归因**：不仅展示典型高置信看涨/看跌样本的 Grad-CAM 叠加图，更计算 **Top-1000 看涨预测与 Top-1000 看跌预测的全局平均 Grad-CAM 空间热力分布矩阵**，直观展示网络注意力在“最近几个交易日收盘位置 vs. 早期均线基底 vs. 底部放量柱”上的权重分配（与组员 B 的遮挡消融形成完美交叉印证！）。
* **插入 `Table 3`（跨时代漂移对比表）+ `Figure 4`**（四宫格组合图：逐年滚动表现曲线 + 典型个股 Grad-CAM + 看涨/看跌全局平均 Grad-CAM 热力图）。

---

### Section 7: Conclusion, Team Contributions & AI Usage Statement (~0.7 Page) — *负责人：组员 D 统稿 + 全体确认*
* **7.1 Conclusion & Limitations**：用 1 段话串联四个模块的发现（什么真正驱动了 Visual Alpha、它的统计边界在哪里），指出未计交易摩擦与换手率等局限性。
* **7.2 Team Member Contributions（课程硬性要求，清晰分点）**：明确写出 Member A / B / C / D 各自负责的代码模块、科学实验与报告章节。
* **7.3 Statement on AI / LLM Usage（课程硬性要求）**：诚实、规范披露全组如何使用 AI 工具辅助环境搭建、代码框架生成、Bug 排查及学术英语润色，并声明所有实验运行、数据核验与科学结论均由小组成员独立验证负责。
* **References**：精选 6–8 篇核心文献（Jiang et al. 2023; Guo et al. 2017 Calibration; Selvaraju et al. 2017 Grad-CAM; Gu, Kelly, & Xiu 2020 Empirical Asset Pricing via ML 等）。

---

## 3. 全文图表总控清单（严控 3 表 + 4 图，版面紧凑美观）

| 编号 | 归属章节 & 负责人 | 图表内容 | 建议尺寸与导出文件名 |
| :--- | :--- | :--- | :--- |
| **Table 1** | Sec 2 (组员 A) | Baseline Single-Seed ($\mu \pm \sigma$) vs. 5-Seed Ensemble 在 2000–2019 全样本上的完整指标基准 | `outputs/tables/tab1_baseline.csv` |
| **Table 2** | Sec 3 (组员 B) | 跨模型族（PCA+LR, LightGBM, 1D-CNN, 2D-CNN, ResNet）参数量、训练耗时与四维 OOS 指标对比 | `outputs/tables/tab2_inductive_bias.csv` |
| **Table 3** | Sec 6 (组员 D) | `Model_1990s (93-99)` vs. `Model_Modern (06-12)` 在 `2013–2019` 样本外的概念漂移对比 | `outputs/tables/tab3_concept_drift.csv` |
| **Figure 1** | Sec 3 (组员 B) | **(a)** 模型架构对比条形图；**(b)** 图像局部区域遮挡（成交量 / 前10天 / 近10天）性能衰减对比 | 双栏宽 `(10, 3.8)`，`fig1_sq1_ablation.pdf` |
| **Figure 2** | Sec 4 (组员 C) | **(a)** 噪声过滤阈值 $\tau$ 与保留样本占比/OOS 性能曲线；**(b)** Label Smoothing 与 `5d/20d/60d` 周期对比 | 双栏宽 `(10, 3.8)`，`fig2_sq2_noise_horizon.pdf` |
| **Figure 3** | Sec 5 (组员 A) | **(a)** 单模型 vs. 温度缩放 vs. 集成的 Reliability Diagram；**(b)** Selective Prediction 覆盖率-准确率曲线 | 双栏宽 `(10, 3.8)`，`fig3_sq3_calibration.pdf` |
| **Figure 4** | Sec 6 (组员 D) | **(a)** 2000–2019 逐年滚动表现与现代重训对比；**(b-d)** 个股示例与 Top-1000 看涨/看跌全局平均 Grad-CAM 热力图 | 双栏宽 `(10, 4.5)`，`fig4_sq4_drift_gradcam.pdf` |

---

## 4. $\le 5$ 分钟展示视频脚本结构（严格限时 4分30秒–4分50秒）

| 时间段 | 页码 | 讲述人建议 | 核心展示内容 |
| :---: | :---: | :---: | :--- |
| `0:00–0:45` | Slide 1–2 | 组员 A | **破题与底座**：什么是 K 线图像选股？为什么我们从统计机器学习的 4 个环节做系统诊断审计？展示复现基线表现。 |
| `0:45–1:45` | Slide 3 | 组员 B | **RQ1 归纳偏置与消融**：2D-CNN 为什么赢过传统统计模型与 1D-CNN？遮掉成交量或近期 K 线会发生什么？ |
| `1:45–2:40` | Slide 4 | 组员 C | **RQ2 低信噪比与标签噪声**：剔除 0 附近微弱波动样本的偏差-方差权衡曲线，以及不同预测周期的信噪比衰减。 |
| `2:40–3:35` | Slide 5 | 组员 A | **RQ3 概率校准与尾部选股**：为什么整体准确率仅 ~53% 却能支撑高夏普策略？用校准曲线与高置信覆盖率图一锤定音。 |
| `3:35–4:40` | Slide 6–7 | 组员 D | **RQ4 概念漂移与 Grad-CAM 总结**：1990s 模型在 2010s 失效了吗？展示惊艳的全局平均 Grad-CAM 热力图，总结全文核心 Takeaway。 |

---

## 5. AI / 大语言模型使用追踪登记表（AI Usage Log — 全组随手填）

> **说明**：根据 [Final Project.pdf](../Final%20Project.pdf) 第 1 页强制要求，每当组员或 AI Agent 使用大模型辅助完成实质性工作时，请在此表追加一行记录，最终汇总进报告 Section 7.3。

| 日期 | 成员 | 使用的 AI 工具 / 模型 | 使用环节与具体用途 | 人工核验与修正情况 |
| :--- | :---: | :--- | :--- | :--- |
| 2026-11-12 | 组员 A (组长) | GitHub Copilot (VS Code Agent) | 解读课程任务书、核验 1993–2019 数据集字段与像素结构、配置 `uv` 环境及起草项目工程规范文档体系 | 人工逐项确认研究切口、工程量边界与四人任务分工方案 |
| 2026-11-12 | 组员 A (组长) | GitHub Copilot (VS Code Agent) | 辅助实现 `src/visual_alpha/` 公共底座、单元测试、`exp_baseline.py` 5-seed 基线训练及 `exp_sq3_calibration.py` 校准分析与出图脚本 | 通过 `pytest` 校验网络维度与指标计算，并在 M5 Pro 上完整跑通 1993–2019 实验验证复现结果 |
| 2026-11-12 | 组员 A (组长) | GitHub Copilot (VS Code Agent) | 起草 NeurIPS 格式报告主框架 (`report/main.tex`) 及已完成模块（Abstract 框架、Sec 1、Sec 2、Sec 5、Sec 7 声明与参考文献） | 人工核对全部实测表格数字、公式定义与 PDF 编译排版 |
| *(待填)* | 组员 B | | 辅助编写 SQ1 跨模型对比与图像遮挡消融代码 | |
| *(待填)* | 组员 D | | 辅助编写 SQ4 现代重训、Grad-CAM 可视化及英文报告语法润色 | |
| 2026-11-17 | 组员 C | OpenAI Codex | 辅助拟定 SQ2 执行计划，核对课程要求、压缩包内标签与基线预测，复算分组样本数并检查跨周期收益回退逻辑 | 待组员 C 人工复核 |
| 2026-11-17 | 组员 C | OpenCode Agent (GLM-5.3) | 辅助实现并执行 SQ2 实验脚本（C1–C7 双 GPU 训练、汇总表与图 2），完成结果分析、交付前校验（129 项全部通过）与交接文档 | 训练与评估走公共代码，预测/键/行数经原始 feather 复核，待组员 C 人工抽查 |
| 2026-11-17 | 组员 C | OpenAI Codex | 复核服务器预测与原始标签、重算市值分组、修正图 2 比较样本，起草英文第 4 节、展示页与讲稿 | 已完成逐行数据对照及报告编译；待组员 C 人工审阅与录制 |
