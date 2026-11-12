# 数据字典、模型规格与全组统一接口契约 (`DATA_AND_CONTRACTS.md`)

> **定位**：本文档是全组 4 位成员与所有 AI Agent 的**唯一技术事实源（Single Source of Truth）**。  
> 所有统计数字、列名、像素坐标与网络维度均已在本地解压后的完整数据集上完成 **100% 实测校验**。任何组员在编写实验代码前必须严格遵守本契约，以确保全组实验结果可直接横向对比。

---

## 1. 数据集物理存储与样本规模（1993–2019 实测）

### 1.1 文件命名与读取方式
数据位于 `data/monthly_20d/`（共 27 个年份，`Y` $\in [1993, 2019]$，总体积约 **8.6 GB**），每年包含严格一一对应的两个文件：

1. **图像二进制矩阵**：`20d_month_has_vb_[20]_ma_{Y}_images.dat`
   * **必须使用 `np.memmap` 只读按需映射**（严禁全量转 `float32` 读入内存，否则将膨胀至 ~34 GB 导致 16G/24G 电脑内存溢出）：
     ```python
     import numpy as np
     images = np.memmap(dat_path, dtype=np.uint8, mode="r").reshape((-1, 64, 60))
     ```
   * 数据类型：`uint8`，像素取值**严格二值化为 `{0, 255}`**（0 = 黑色背景，255 = 白色前景线条）。
2. **标签与收益元数据表**：`20d_month_has_vb_[20]_ma_{Y}_labels_w_delay.feather`
   * 读取方式：`df = pd.read_feather(feather_path)`
   * 对齐保证：经全量校验，27 个年份全部满足 `images.shape[0] == len(df)`。

### 1.2 标准时间切分与实测样本量

| 数据子集 | 年份范围 | 原始总行数 | 有效样本数 (`Retx_20d_label != 2`) | 用途说明 |
| :--- | :---: | ---: | ---: | :--- |
| **全量数据集** | `1993–2019` | `2,196,994` | `2,181,781` | 27 年完整美股月频截面样本 |
| **Baseline 训练/验证池** | `1993–1999` | `694,871` | **`689,894`** | 按 `split_seed=42` 随机切分 **70% 训练集** (`482,925`) / **30% 验证集** (`206,969`) |
| **主实验样本外测试集 (Full OOS)** | `2000–2019` | `1,502,123` | **`1,491,887`** | Baseline、SQ1、SQ2、SQ3 统一在**完全相同的 1,491,887 条样本**上评估 |
| **SQ4 现代训练/验证池** | `2006–2012` | `511,240` | **`507,561`** | SQ4 概念漂移实验中的“现代量化时期”训练/验证池（70:30 切分） |
| **SQ4 现代公共测试集** | `2013–2019` | `474,732` | **`472,357`** | SQ4 公平对比 `Model_1990s (93-99)` 与 `Model_Modern (06-12)` 的无重叠样本外区间 |

---

## 2. `.feather` 35 列完整数据字典与**核心避坑指南**

每年 `.feather` 文件具有完全一致的 **35 列**：

| 类别 | 列名 | 类型 | 含义与使用规范 |
| :--- | :--- | :--- | :--- |
| **主键标识** | `Date` | `datetime64[ns]` | 20日窗口最后一天（即月末调仓/预测基准日 $t$） |
| | `StockID` | `object (str)` | CRSP `PERMNO` 股票唯一代码 |
| | `MarketCap` | `float32` | 基准日股票市值（用于市值分组或加权检验） |
| | `EWMA_vol` | `float64` | 指数加权移动平均波动率估计 |
| | `window_size` | `uint8` | 固定常数 `20` |
| **监督分类标签** | **`Retx_20d_label`** | `int8` | **本项目默认分类标签**：`1` (`Retx_20d > 0`)，`0` (`Retx_20d <= 0`)，`2` (`Retx_20d` 缺失) |
| | `Retx_5d_label`, `Retx_60d_label`, `Retx_label` | `int8` | 未来 5日 / 60日 / 次日除息收益方向标签（取值 `{0, 1, 2}`，供 SQ2 多周期实验使用） |
| **除息收益率 (`Retx`)** | **`Retx_20d`** | `float32` | **未来 20 个交易日累计除息收益率（不含分红）** —— 与 `Retx_20d_label` 100% 严格对应 |
| | `Retx`, `Retx_5d`, `Retx_60d`, `Retx_week`, `Retx_month`, `Retx_quarter` | `float32` | 其他窗口除息收益率 |
| | `Retx_tstat`, `Retx_5d_tstat`, `Retx_20d_tstat`, `Retx_60d_tstat` | `float32` | 波动率标准化后的未来收益 $t$-统计量（可供 SQ2 作为波动率自适应信噪比指标） |
| **含息总收益率 (`Ret`)** | **`Ret_20d`** | `float64` | **未来 20 个交易日含息总收益率（含现金分红）** —— **全组统一用于计算多空组合真实经济 Sharpe Ratio** |
| | `next_month_ret_0delay`, `next_month_ret_1delay` | `float64` | 下一自然月总收益（0天延迟 / 1天延迟） |
| | `Ret`, `log_ret`, `cum_log_ret`, `Ret_week`, `Ret_month`, `Ret_quarter`, `Ret_5d`, `Ret_60d`, `Ret_65d`, `Ret_180d`, `Ret_250d`, `Ret_260d` | `float64` | 其他窗口历史或前瞻总收益率字段 |

### ⚠️ 极重要实测发现：`Retx_20d` 与 `Ret_20d` 的差异（全组严禁混用！）
经代码逐行核查发现：数据集中官方标签 `Retx_20d_label` 是由**除息价格收益率 `Retx_20d`** 生成的（`Retx_20d > 0` 为 `1`，`<= 0` 为 `0`），而**不是**含分红总收益率 `Ret_20d`（每年约有 ~900 条分红股样本满足 `Retx_20d <= 0` 但 `Ret_20d > 0`）。因此全组严格执行以下三条铁律：
1. **缺失值过滤**：所有训练、验证、测试集第一步必须剔除 `Retx_20d_label == 2`（等价于 `Retx_20d.notna()`）。
2. **SQ2 阈值过滤对齐**：组员 C 在训练集做噪声边际过滤时，**必须使用 `abs(Retx_20d) >= tau`（或 `abs(Retx_20d_tstat) >= tau_t`）**，严禁使用 `abs(Ret_20d)`，否则会出现过滤准则与标签符号相矛盾的逻辑 Bug！同时，**验证集与样本外测试集（2000–2019）严禁做任何阈值剔除**，确保所有模型在同一测试宇宙中公平对比。
3. **经济组合回测对齐**：在 `evaluate.py` 计算多空十分位组合（Decile Long-Short Portfolio）收益与年化夏普比率（Sharpe Ratio）时，统一使用真实可交易的含息总收益 **`Ret_20d`**（若个别行 `Ret_20d` 为 `NaN` 则回退至 `Retx_20d`）。

---

## 3. $64 \times 60$ 图像像素解剖坐标（经全样本像素密度实测验证）

每张图像形状为 `(H=64, W=60)`，对应纵向 64 行（`row = 0..63`，自上而下）、横向 60 列（`col = 0..59`，自左向右对应过去第 1 天至第 20 天）：

```text
Row  0 ──────────────────────────────────────────────────────┐
  ...   价格与均线区 (Price OHLC + 20d Moving Average)        │ 高度 = 51 px (rows 0:51)
Row 50  row 0 = 窗口最高价, row 50 = 窗口最低价                │
Row 51 ════════════════ 全黑隔离带 (像素恒为 0) ═══════════════╡ 高度 =  1 px (row 51)
Row 52  成交量柱区 (Volume Bars, 仅在 col = 3*d + 1 列非零)    │ 高度 = 12 px (rows 52:64)
Row 63 ──────────────────────────────────────────────────────┘ row 63 = 零成交量基线
        Col: 0,1,2 (Day 1, 最早)  ......  Col: 57,58,59 (Day 20, 最新调仓日 Date)
```

* **横向时间坐标（20 个交易日 $\times$ 3 列/日 = 60 列）**：对于窗口内第 `d` 个交易日（`d = 0, 1, ..., 19`，`d=19` 为最新基准日）：
  * `col = 3*d`：当日开盘价横杠（Open tick）+ 均线连接点；
  * `col = 3*d + 1`：当日最高至最低价竖线（High-Low bar）+ 均线点 + **底部成交量竖柱**；
  * `col = 3*d + 2`：当日收盘价横杠（Close tick）+ 均线连接点。
* **标准图像消融切片公式（供组员 B / D 直接调用）**：
  * 遮挡成交量（No Volume Bars）：`x[..., 52:64, :] = 0`
  * 遮挡前 10 天、仅保留最近 10 天（Recent 10d Only）：`x[..., :, 0:30] = 0`
  * 遮挡最近 10 天、仅保留早期 10 天（Old 10d Only）：`x[..., :, 30:60] = 0`
  * 遮挡前 15 天、仅保留最近 5 天（Recent 5d Only）：`x[..., :, 0:45] = 0`
  * 1D 多变量时序重排（供 SQ1 `1D-CNN` 使用）：将 `(B, 1, 64, 60)` reshape 为 `(B, 64, 20, 3)` 再合并通道为 `(B, 192, 20)`，使卷积核仅沿 `T=20` 时间轴滑动，破坏二维空间局部几何结构但保留 100% 原始像素信息。

---

## 4. 标准 20-Day Baseline CNN 精确网络规格 (`CNN20D`)

严格对齐课程说明 [Final Project.pdf](../Final%20Project.pdf) 第 7 页 `Figure 2` 与 Jiang, Kelly, & Xiu (2023) Section 3.3（展开特征维度精确等于 **`FC 46080`**）：

| 层级 | PyTorch 精确配置 | 输出张量形状 `(C, H, W)` | 说明 |
| :--- | :--- | :---: | :--- |
| **Input** | `uint8 -> float32 / 255.0` | `(1, 64, 60)` | 值域 `[0.0, 1.0]` |
| **Block 1** | `Conv2d(1, 64, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1))`<br>`BatchNorm2d(64)` $\to$ `LeakyReLU(0.01)` $\to$ `MaxPool2d(kernel_size=(2,1))` | Conv: `(64, 27, 60)`<br>Pool: `(64, 13, 60)` | 第一层纵向带步长与空洞卷积，契合原论文宽视野扫描 |
| **Block 2** | `Conv2d(64, 128, kernel_size=(5,3), stride=(1,1), padding=(2,1))`<br>`BatchNorm2d(128)` $\to$ `LeakyReLU(0.01)` $\to$ `MaxPool2d(kernel_size=(2,1))` | Conv: `(128, 13, 60)`<br>Pool: `(128, 6, 60)` | 保持横向 60 列时间分辨率不变 |
| **Block 3** | `Conv2d(128, 256, kernel_size=(5,3), stride=(1,1), padding=(2,1))`<br>`BatchNorm2d(256)` $\to$ `LeakyReLU(0.01)` $\to$ `MaxPool2d(kernel_size=(2,1))` | Conv: `(256, 6, 60)`<br>Pool: **`(256, 3, 60)`** | **Grad-CAM 默认挂载目标层**：Block 3 的最后一个 `Conv2d` |
| **Head** | `Flatten()` $\to$ `Dropout(p=0.5)` $\to$ `Linear(46080, 2)` | Logits: `(2,)` | `256 * 3 * 60 = 46,080`，完全匹配原论文 `FC 46080`！ |

* **权重初始化**：所有 `Conv2d` 与 `Linear` 权重采用 `nn.init.xavier_uniform_` 初始化，偏置初始化为 `0`。
* **锁定训练超参数（全组禁止随意改动，确保单一变量控制）**：
  * 优化器：`Adam(lr=1e-5, weight_decay=0.0)`
  * Batch Size：`256`（或 `512`，由于使用 mean reduction 且 `lr=1e-5` 极稳，默认统一 `256`）
  * 最大训练轮数：`max_epochs = 15`，**Early Stopping `patience = 2`**（监控 `1993–1999` 30% 验证集的 Cross-Entropy Loss，保存最优验证集权重）
  * 随机种子规范：单模型实验默认固定 `seed = 42`；5 模型集成（Ensemble）固定使用 `seeds = [42, 43, 44, 45, 46]`（网络初始化随机种子不同，训练/验证集划分固定 `split_seed = 42` 以便验证集概率可直接平均）。

---

## 5. 公共底座接口与预测产物契约 (`src/visual_alpha/`)

组长（组员 A）将在第一阶段完成 `src/visual_alpha/` 下的公共模块，组员 B/C/D 直接 `import` 调用，**无需重复造轮子**：

1. `visual_alpha.utils`
   * `get_device() -> torch.device`：自动按 `cuda` $\to$ `mps` $\to$ `cpu` 优先级选择设备。
   * `seed_everything(seed: int)`：固定 `random`、`numpy`、`torch` 随机种子。
2. `visual_alpha.dataset`
   * `ChartDataset`：基于 `np.memmap` 的懒加载数据集，支持传入 `years: list[int]`、`indices`、可选图像变换 `transform`（如遮挡消融）、可选样本过滤掩码（如 SQ2 `abs(Retx_20d) >= tau`）以及标签列选择（默认 `Retx_20d_label`）。
   * `build_train_val_test_splits(...)`：返回标准 `1993–1999`（70/30，固定 `split_seed=42`）与 `2000–2019` OOS DataLoader。
3. `visual_alpha.models`
   * `CNN20D`：上表定义的精确基线模型。
4. `visual_alpha.trainer`
   * 通用训练、验证早停与批量推理循环，自动在 GPU/MPS 上执行 `x.to(device).float().div_(255.0)`。
5. `visual_alpha.evaluate`（**统一裁判函数**）

### 5.1 统一预测结果文件契约：`outputs/predictions/<exp_name>.parquet`
任何模型（无论是 PyTorch CNN、1D-CNN、ResNet，还是 Scikit-Learn 逻辑回归、LightGBM）完成样本外推理后，**必须保存为包含以下列的标准 `.parquet` 文件**：

| 列名 | 类型 | 必填 | 说明 |
| :--- | :--- | :---: | :--- |
| `year` | `int16` | ✅ | 样本所属年份（如 `2000..2019`） |
| `row_idx` | `int32` | ✅ | 该样本在当年原始文件中的行号（`0..N_year-1`），便于精确回溯原图（如画 Grad-CAM） |
| `Date` | `datetime64[ns]` | ✅ | 截面日期（按月分组计算多空组合收益） |
| `StockID` | `string` | ✅ | 股票代码 |
| `y_true` | `int8` | ✅ | 真实二分类标签 `{0, 1}` |
| `logit_down` | `float32` | 选填 | 类别 0 原始 logit（树模型/LR 可填 `np.nan`，神经网络必填以供温度缩放使用） |
| `logit_up` | `float32` | 选填 | 类别 1 原始 logit（神经网络必填） |
| `prob_up` | `float32` | ✅ | 模型预测上涨概率 $P(y=1 \mid x) \in [0, 1]$ |
| `Retx_20d` | `float32` | ✅ | 未来 20 日除息收益率 |
| `Ret_20d` | `float32` | ✅ | 未来 20 日含息总收益率 |
| `MarketCap` | `float32` | ✅ | 截面市值 |

### 5.2 全组统一四维评估指标（由 `evaluate_predictions(pred_df)` 一键输出）
所有实验表格统一汇报以下核心指标，兼顾**统计机器学习严谨性**与**金融实证意义**：

1. **OOS Accuracy (%)**：样本外分类准确率（阈值 `prob_up >= 0.50`，保留两位小数，如 `52.85%`；注意金融月度涨跌预测中 52%~54% 已极具经济显著性）。
2. **OOS Cross-Entropy Loss (`CE`) & ROC-AUC (`AUC`)**：对数似然损失与排序区分度（保留四位小数）。
3. **Expected Calibration Error (`ECE`, 10 Equal-Width Bins)**：衡量预测概率与真实上涨频率的校准偏差（保留四位小数）：
   $$\text{ECE} = \sum_{m=1}^{10} \frac{|B_m|}{N} \big| \text{acc}(B_m) - \text{conf}(B_m) \big|$$
4. **Equal-Weighted Long-Short Decile Sharpe Ratio (`LS_Sharpe`)**：
   * 在每个调仓截面 `Date` $t$ 内，按 `prob_up` 排序（使用 `method="first"` 等频分为 10 组，避免离散概率并列导致分桶为空），做多预测概率最高的 Top 10%（Decile 10），做空最低的 Bottom 10%（Decile 1），等权计算月度多空组合超额收益 $R_{LS, t} = \bar{R}_{10, t} - \bar{R}_{1, t}$（基于 `Ret_20d`）。
   * 汇报年化夏普比率 $\text{SR}_{\text{ann}} = \sqrt{12} \cdot \frac{\hat{\mu}(R_{LS})}{\hat{\sigma}(R_{LS})}$ 及对应均值显著性 $t$-statistic（保留两位小数）。
