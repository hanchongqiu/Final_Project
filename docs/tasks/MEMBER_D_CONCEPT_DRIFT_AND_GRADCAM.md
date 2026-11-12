# 组员 D 专属任务书：SQ4 跨时代概念漂移、Grad-CAM 可解释性与报告统稿

> **负责人**：组员 D  
> **前置必读**：[README.md](../../README.md)（先配好环境）｜ [DATA_AND_CONTRACTS.md](../DATA_AND_CONTRACTS.md) ｜ [REPORT_BLUEPRINT.md](../REPORT_BLUEPRINT.md)  
> **工程分支**：`feat/sq4-drift-gradcam`

---

## 1. 你的任务定位：最具视觉冲击力的收官章节 + 全组报告/展示统稿人

为了保证 4 人工作量绝对公平，你的**模型重训次数是全组最少的（仅需训练 1~2 个模型，耗时不到 15 分钟！）**，把宝贵精力投入到两件最能拔高整篇论文档次的事情上：
1. **科研实验 SQ4（报告 Section 6）**：
   * **跨时代概念漂移（Temporal Concept Drift）**：原论文用 **1993–1999 年**（美股还在用 $1/8$ 美元分数报价、人工做市商为主的时代）训练出的固定模型，一直预测到 2019 年。到了算法高频交易主导的 2010 年代，市场规律变了吗？如果改用更近期的 **`2006–2012` 年** 数据重新训练同一个 CNN，在 **`2013–2019` 公共测试期**上能否打败 90 年代老模型？
   * **Grad-CAM 视觉可解释性归因（Visual Attribution）**：打开 CNN 黑盒！不仅画出高置信看涨/看跌个股的 Grad-CAM 热力图，更创新性地统计 **Top-1000 看涨样本与 Top-1000 看跌样本的“全局空间平均 Grad-CAM 热力图”**，用一张一目了然的 $64 \times 60$ 概率注意力图告诉读者：CNN 到底在看最近几天的收盘位置、均线交叉，还是底部放量柱？
2. **报告与展示合稿统稿**：将 4 位组员的小节整合进 8 页 NeurIPS 模板（统一字体与图表风格），牵头汇总 Section 7（结论、分工声明、AI 使用披露）与 5 分钟展示 Slides。

---

## 2. 你的受控实验设计 (`experiments/exp_sq4_drift_gradcam.py`)

### 实验一：跨时代分布漂移与滚动稳定性审计（Temporal Concept Drift）

1. **逐年滚动表现拆解（1993–1999 训练模型的 20 年生命周期）**：
   * 直接读取组长跑好的 `baseline_ensemble5.parquet`（及 `baseline_seed42.parquet`），按年份 `year = 2000, 2001, ..., 2019` 逐年统计样本外 Accuracy、AUC 与年度多空组合收益/Sharpe。
   * 重点观察三个历史制度变迁阶段：`2000–2002`（互联网泡沫破裂与美股十进制报价改革 Decimalization）、`2008–2009`（全球金融危机高波动期）、`2013–2019`（现代量化交易成熟期）。
2. **新旧时代训练窗口正面交锋（在 `2013–2019` 共 `472,357` 条无重叠测试样本上公平PK）**：
   * **`Model_1990s`**：使用 `1993–1999` 训练的原基线模型（直接切片其在 `2013–2019` 上的预测结果，无需重训）。
   * **`Model_Modern`**：保持 `CNN20D` 网络结构与超参数 100% 不变，仅将训练/验证池替换为 **`2006–2012`**（共 `507,561` 条有效样本，70/30 切分，`seed=42`，可选跑 3 种子平均）。
   * **`Model_Combined`（可选加分项，仅多跑 8 分钟）**：联合使用 `1993–1999` + `2006–2012`（总样本翻倍）训练一个模型，在 `2013–2019` 上检验：**金融视觉预测中，“数据越新（新鲜度）”更重要，还是“跨越更多牛熊周期（样本规模）”更重要？**——无论谁赢，都是极具洞察力的统计机器学习发现！

### 实验二：Grad-CAM 个案与群体空间注意力归因

针对 `CNN20D` 第 3 个卷积块的最后一个 `Conv2d` 层（特征图空间尺寸 `(6, 60)`，双线性插值上采样回 `(64, 60)`）：
1. **经典个股案例（Case Study）**：从样本外测试集中挑选 2 张最高置信度看涨（`prob_up` 最高且真实上涨）和 2 张最高置信度看跌的 $64 \times 60$ 原图，叠加半透明 Grad-CAM 热力图，结合 [DATA_AND_CONTRACTS.md](../DATA_AND_CONTRACTS.md) 的像素坐标解读其 K 线形态（如“近期急跌至窗口底部缩量企稳 $\to$ 短期超跌反转看涨” vs. “短期急拉远离均线 $\to$ 超买回落看跌”）。
2. **群体统计热力图（Aggregate Spatial Attribution —— 全文视觉高光图！）**：
   * 在测试集中抽取预测上涨概率最高的 **Top-1000 Bullish 样本** 和最低的 **Top-1000 Bearish 样本**，分别对它们的归一化 Grad-CAM 矩阵 $(64 \times 60)$ 求像素级均值！
   * 标出 `row 51`（价格/成交量分界线）和横向 Day 1 $\to$ Day 20 坐标轴，定量计算热力能量落在“价格区 (`rows 0:51`) vs. 成交量区 (`rows 52:64`)”以及“前 10 天 (`cols 0:30`) vs. 后 10 天 (`cols 30:60`)”的百分比，与组员 B 的遮挡消融实验形成完美闭环互证！

---

## 3. 你的交付物清单

- [ ] 实验脚本：`experiments/exp_sq4_drift_gradcam.py`
- [ ] 汇总表格：**`outputs/tables/tab3_concept_drift.csv`**（`Model_1990s` vs. `Model_Modern` vs. `Model_Combined` 在 `2013–2019` 上的四维指标对比）及 `outputs/tables/sq4_yearly_performance.csv`
- [ ] 论文插图 **`outputs/figures/fig4_sq4_drift_gradcam.pdf` (& `.png`)**：
  * **(a)** 2000–2019 逐年滚动表现曲线（含 `2013–2019` 新旧模型对比阴影区）；
  * **(b)** 典型看涨/看跌个股原图 + Grad-CAM 叠加示例；
  * **(c) & (d)** Top-1000 Bullish 与 Top-1000 Bearish 全局平均 $64 \times 60$ 空间注意力热力图（标注价格区/成交量区虚线）。
- [ ] 报告撰写与全组统稿：
  * 撰写 **Abstract 初稿**、**Section 6 (RQ4: Temporal Concept Drift & Grad-CAM Attribution，约 1.4 页)** 与 **Section 7 (Conclusion, Team Contributions & AI Usage Statement，约 0.7 页)**；
  * 汇总全组文稿排版至 `report/`（严格 $\le 8$ 页 NeurIPS 格式）并统筹 5 分钟展示 Slides。

---

## 4. 给你的 AI 编程助手的一键启动提示词（直接复制发送即可）

```text
请先完整阅读根目录下的 AGENTS.md、docs/DATA_AND_CONTRACTS.md、docs/REPORT_BLUEPRINT.md 和 docs/tasks/MEMBER_D_CONCEPT_DRIFT_AND_GRADCAM.md。
我是组员 D，负责 SQ4（跨时代概念漂移、Grad-CAM 视觉可解释性归因）以及报告统稿。
请严格复用 src/visual_alpha/ 中的数据集、CNN20D 模型与统一评估器，帮我设计并实现 experiments/exp_sq4_drift_gradcam.py：
1) 实现 2000-2019 逐年指标拆解，并在 2006-2012（及 93-99+06-12 联合）数据上训练对比模型，统一在 2013-2019 公共测试集上生成 Table 3；
2) 针对 CNN20D 第三个卷积块的 Conv2d 实现干净无第三方黑盒依赖的 PyTorch Grad-CAM，导出典型个股案例与 Top-1000 看涨/看跌全局空间平均热力图（Figure 4）。
```
