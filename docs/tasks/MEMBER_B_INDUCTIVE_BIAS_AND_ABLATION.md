# 组员 B 专属任务书：SQ1 架构归纳偏置对比与图像遮挡消融

> **负责人**：组员 B  
> **前置必读**：[README.md](../../README.md)（先配好环境）｜ [DATA_AND_CONTRACTS.md](../DATA_AND_CONTRACTS.md) ｜ [REPORT_BLUEPRINT.md](../REPORT_BLUEPRINT.md)  
> **工程分支**：`feat/sq1-inductive-bias`

---

## 1. 你的任务是什么？为什么它在报告中极其出彩？

原论文直接把 $64 \times 60$ 的股票图表喂给了一个特定结构的 **2D-CNN**，拿到了不错的收益。但在《统计机器学习》课堂上，老师最关心的问题是：
> **“为什么非要用二维卷积神经网络不可？如果把这张 $64 \times 60 = 3840$ 像素的黑白图展平喂给传统统计学习模型（PCA + 逻辑回归 / LightGBM），或者喂给只沿时间滑动的 1D-CNN，又或者换成参数量更大的现代 ResNet，表现会怎样？此外，图像里的成交量柱、早期 K 线、近期 K 线到底哪一部分真正贡献了预测力？”**

你的模块（报告 **Section 3**）将直接利用课程核心概念——**归纳偏置（Inductive Bias）、特征表示与模型容量（Model Capacity）**，正面回答这两个极具说服力的问题！

* **前置依赖说明**：组长（组员 A）会提供现成的 `visual_alpha.dataset`（一行代码加载数据）和 `visual_alpha.evaluate`（一行代码算完所有指标），并提供训练好的 `baseline_seed42.pt` 权重供你直接做遮挡消融！

---

## 2. 你的受控实验设计 (`experiments/exp_sq1_inductive_bias.py`)

所有模型统一在 `1993–1999` 训练/验证，统一在**完全相同的 `2000–2019` 全量测试集（1,491,887 条样本）**上导出标准 `.parquet` 预测表并调用 `evaluate_predictions()`。

### 实验一：跨模型族归纳偏置对比（5 个代表性模型，其中 Baseline 直接复用组长结果！）

| 模型编号 | 模型家族 | 具体实现规格（简洁高效，严禁耗时网格搜索） | 检验的统计学习假设 |
| :--- | :--- | :--- | :--- |
| **B0 (复用)** | **2D-CNN Baseline (`CNN20D`)** | 直接读取组长跑好的 `baseline_seed42.parquet` | 二维非对称局部卷积核 `(5, 3)` 的基准表现 |
| **B1** | **线性统计基线 (`PCA + LogisticRegression`)** | 从 `1993–1999` 训练集随机子采样 `150,000` 张图展平为 `3840` 维拟合 `IncrementalPCA(n_components=64)` + L2 正则 `LogisticRegression(C=1.0)`，分块推理 `2000–2019`（全程仅需 ~2 分钟！） | 检验纯线性低秩像素投影能否捕捉涨跌信号 |
| **B2** | **树模型集成 (`Pixel-Summary + LightGBM`)** | 将每张 $64 \times 60$ 图按经济学意义提取约 **48 个紧凑统计特征**（如：将 20 天均分为 4 个 5天时间段 $\times$ 价格重心高度、最高/最低跨度、收盘位置、底部成交量柱高度均值与变化率），训练 `LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)` *(注：macOS 上若同一脚本同时使用 `lightgbm` 和 `torch`，请务必将 `import lightgbm` 写在 `import torch` 之前以避免 OpenMP 冲突)* | 检验非线性树模型配合人工分区统计特征能否逼近端到端 CNN |
| **B3** | **纯时序卷积 (`1D-CNN`)** | 将 `(B, 1, 64, 60)` 重排为 `(B, 192, 20)`（每天 3 列 $\times$ 64 行 = 192 维当日切片特征，沿 `T=20` 做 3 层 `Conv1d(kernel_size=3)` + BN + LeakyReLU + FC），优化器与早停同 Baseline | **关键对比**：保留时间平移不变性但打碎纵向价格局部几何结构，验证纵向卷积的价值 |
| **B4** | **高容量残差网络 (`ResNet-Tiny / ResNet18-1ch`)** | 单通道输入的轻量残差网络（如改第一层为 `Conv2d(1, 32, 3, padding=1)` 的 4-stage 残差网络，参数量显著大于窄通道层或去掉大池化），同 `lr` 与早停 | 检验在金融极低信噪比下，更深更复杂的现代视觉架构是否更容易发生过拟合（训练损失更低但验证/测试泛化不升反降） |

### 实验二：输入视觉要素遮挡消融（Input Component Masking）
利用 [DATA_AND_CONTRACTS.md](../DATA_AND_CONTRACTS.md) 第 3 节的精确像素坐标，对比不同视觉要素被置零（涂黑 `0`）后的样本外表现衰减（建议先对已训练好的 `CNN20D` 做**推理期遮挡归因（Occlusion Sensitivity）**，并可选对核心的 `No-Volume` 额外重训 1 次对比“模型能否只靠价格学会替代信号”）：

1. **`Mask-Volume`（抹除底部成交量柱）**：令 `img[:, 52:64, :] = 0`（检验成交量确认对趋势预测的边际贡献）。
2. **`Mask-Early10d`（抹除前 10 天，仅留最近 10 天）**：令 `img[:, :, 0:30] = 0`。
3. **`Mask-Recent10d`（抹除最近 10 天，仅留早期 10 天）**：令 `img[:, :, 30:60] = 0`（检验短期反转/动量的时间衰减规律：近端信息是否远重要于远端信息？）。
4. **`Retrain-NoVolume`（无成交量从头重训 1 次 `CNN20D`）**：训练和推理全程令 `img[:, 52:64, :] = 0`，区分“原模型依赖成交量”与“成交量本身包含不可替代增量信息”。

---

## 3. 你的交付物清单

- [ ] 实验脚本：`experiments/exp_sq1_inductive_bias.py`（支持命令行参数选择跑哪个子实验，保存各模型的 `.parquet` 预测结果）
- [ ] 汇总表格：**`outputs/tables/tab2_inductive_bias.csv`**（包含 B0–B4 的参数量、训练时间、OOS Acc、CE、AUC、ECE、LS Sharpe）与 `outputs/tables/sq1_masking_ablation.csv`
- [ ] 论文插图 **`outputs/figures/fig1_sq1_ablation.pdf` (& `.png`)**：
  * 左子图 **(a)**：5 种模型架构的 OOS AUC 与多空年化 Sharpe 双指标柱状对比图；
  * 右子图 **(b)**：不同图像区域遮挡（Full Image vs. No Volume vs. Keep Recent 10d vs. Keep Old 10d）相对于 Baseline 的 $\Delta \text{Accuracy}$ 与 $\Delta \text{Sharpe}$ 衰减图。
- [ ] 报告撰写：完成英文报告 **Section 3 (RQ1: Architectural Inductive Bias & Visual Component Ablation，约 1.4 页)** 初稿，并在 [REPORT_BLUEPRINT.md](../REPORT_BLUEPRINT.md) 登记 AI 使用记录。

---

## 4. 给你的 AI 编程助手的一键启动提示词（直接复制发送即可）

```text
请先完整阅读根目录下的 AGENTS.md、docs/DATA_AND_CONTRACTS.md 和 docs/tasks/MEMBER_B_INDUCTIVE_BIAS_AND_ABLATION.md。
我是组员 B，负责 SQ1（架构归纳偏置对比与图像遮挡消融）。
请严格复用 src/visual_alpha/ 中的数据加载器和统一评估函数 evaluate_predictions，帮我规划并实现 experiments/exp_sq1_inductive_bias.py（包含 PCA+LR、Pixel-Stats LightGBM、1D-CNN、轻量 ResNet 以及精确像素坐标遮挡消融实验），以及生成 Table 2 和 Figure 1 的绘图脚本。注意全程必须使用 np.memmap 按批读取，严禁全量将图像转为 float32 读入内存。
```
