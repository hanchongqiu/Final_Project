# Deconstructing Visual Alpha — MSDM 5054 Final Project

> **课程**：MSDM 5054 Statistical Machine Learning（统计机器学习，Instructor: Prof. Yuan Yao）  
> **截止时间**：2026 年 11 月 22 日（周日）23:59（Canvas 提交）  
> **选题**：Section 5.2 论文复现与统计机器学习审计 —— Jiang, Kelly, & Xiu (2023), *"(Re-)Imag(in)ing Price Trends"*, *Journal of Finance*

---

## 1. 项目简介：我们在做什么？

原论文提出了一种极具创意的量化预测范式：将每只美股过去 **20 个交易日** 的开高低收价格（OHLC）、20日均线（MA）和底部成交量柱（Volume Bars）绘制成一张 **$64 \times 60$ 的单通道二值图像**，利用轻量级 **卷积神经网络（CNN）** 预测未来 20 天的股票涨跌方向。

作为《统计机器学习》期末 4 人小组研究项目，我们**不做简单的代码照搬与刷分**，而是沿着**机器学习系统的完整生命周期**，对这一“视觉 Alpha”模型展开**四维度的科学机制审计（Scientific Audit）**：

| 研究模块 | 生命周期阶段 | 核心科学问题 (Scientific Question) | 负责人 |
| :--- | :--- | :--- | :--- |
| **Baseline & SQ3** | **公共底座 + 概率输出层** | **基线复现与概率校准（Calibration）**：CNN 在 2000–2019 样本外的预测能力是否稳健？其输出概率是否存在过度自信？模型集成（Ensemble）与温度缩放能否校准概率并提升高置信尾部交易胜率？ | **组员 A（组长）** |
| **SQ1** | **输入与模型架构层** | **归纳偏置与要素消融（Inductive Bias & Masking）**：预测力真的来自 2D 局部卷积吗？对比 PCA+逻辑回归、像素统计+LightGBM、1D-CNN 与 ResNet，以及遮挡成交量/远期窗口后的性能变化。 | **组员 B** |
| **SQ2** | **监督信号与损失层** | **低信噪比标签噪声（Label Noise & Margin）**：将 $+0.01\%$ 与 $-0.01\%$ 强行切成相反标签是否迫使模型拟合噪声？在训练集剔除 $|\text{Retx}| < \tau$ 模糊样本或采用标签平滑，能否改善泛化？ | **组员 C** |
| **SQ4** | **时间部署与可解释层** | **市场概念漂移与视觉归因（Concept Drift & Grad-CAM）**：1990 年代人工交易期学到的视觉规律在 2010 年代量化市场漂移了吗？通过 Grad-CAM 全局平均热力图揭示 CNN 究竟看懂了什么金融规律。 | **组员 D** |

---

## 2. 文档导航（请先阅读你对应的文档！）

为了让**每位组友**和**各位组友使用的 AI 编程助手（Copilot / Cursor / Claude Code）**都能零沟通成本无缝协作，项目已配备完整文档体系：

* 🤖 **AI 跨会话自动上下文**：[AGENTS.md](./AGENTS.md)（AI 会自动读取，包含内存安全红线与工程规范）
* 🧭 **组长统筹与 10 天推进计划**：[docs/PROJECT_CHARTER.md](./docs/PROJECT_CHARTER.md)
* 📐 **数据字典、CNN 规格与全组统一接口契约（必读）**：[docs/DATA_AND_CONTRACTS.md](./docs/DATA_AND_CONTRACTS.md)
* 📝 **8 页 NeurIPS 报告写作蓝图与 AI 使用登记表**：[docs/REPORT_BLUEPRINT.md](./docs/REPORT_BLUEPRINT.md)
* 📋 **4 位组员各自专属任务书（直接点开你自己的那一份即可开工）**：
  * [组员 A（组长）：公共底座 + Baseline 复现 + SQ3 概率校准](./docs/tasks/MEMBER_A_LEADER_BASELINE_AND_CALIBRATION.md)
  * [组员 B：SQ1 架构归纳偏置对比 + 图像遮挡消融](./docs/tasks/MEMBER_B_INDUCTIVE_BIAS_AND_ABLATION.md)
  * [组员 C：SQ2 标签噪声阈值过滤 + 损失函数与多周期对比](./docs/tasks/MEMBER_C_LABEL_NOISE_AND_LOSS.md)
  * [组员 D：SQ4 跨时代概念漂移 + Grad-CAM 可解释性 + 报告/视频统稿](./docs/tasks/MEMBER_D_CONCEPT_DRIFT_AND_GRADCAM.md)

---

## 3. 三分钟环境搭建与数据准备指南（新组员上手）

### 第一步：克隆代码并放入数据集
由于解压后图像数据达 **8.6 GB**，`.gitignore` 已屏蔽数据文件，请勿将 `data/` 提交至 Git。
1. 下载课程数据包 `img_data.zip`（802.9 MB，链接见 [Final Project.pdf](./Final%20Project.pdf) 第 5 页或向组长获取网盘包）。
2. 解压到项目根目录下的 `data/` 文件夹中，确保目录结构如下：
   ```text
   Final_Project/
   └── data/
       └── monthly_20d/
           ├── 20d_month_has_vb_[20]_ma_1993_images.dat
           ├── 20d_month_has_vb_[20]_ma_1993_labels_w_delay.feather
           ├── ...
           └── 20d_month_has_vb_[20]_ma_2019_labels_w_delay.feather
   ```
   *(macOS 解压命令：`unzip -q -o img_data.zip -d data/ -x "__MACOSX/*" "*/._*"`)*

### 第二步：安装 Python 环境（任选一种方式）

#### 方式 A：使用 `uv`（极速推荐，30 秒装完且版本 100% 锁定）
```bash
# 如未安装 uv：brew install uv (macOS) 或 pip install uv
uv sync --python 3.12 --extra dev
```

#### 方式 B：使用传统 `venv` / `conda` + `pip`
```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # 已包含 -e . 可编辑安装本仓库 visual_alpha 包
```

### 第三步：一键自检环境、GPU 加速与数据完整性
```bash
.venv/bin/python scripts/verify_env.py
```
看到全部输出 `[OK]` 即表示你的 Python 依赖、硬件加速（Apple Silicon `mps` / NVIDIA `cuda` / `cpu`）和 27 年共 **2,196,994** 条样本全部校验通过！

---

## 4. 仓库目录结构

```text
Final_Project/
├── AGENTS.md                        # AI Agent 全局跨上下文开发准则
├── pyproject.toml / uv.lock         # uv 项目依赖定义与锁文件
├── requirements.txt                 # 兼容传统 pip/conda 的依赖清单
├── Final Project.pdf                # 课程官方大作业要求原件
├── Image Demo.html                  # 原论文作者提供的图像读取示例
├── .github/instructions/            # 按文件类型自动挂载的 AI 细则（代码规范 / 报告规范）
├── docs/
│   ├── PROJECT_CHARTER.md           # 项目整体规划、里程碑与组长验收清单
│   ├── DATA_AND_CONTRACTS.md        # 实测数据字典、像素坐标、网络参数与统一评估契约
│   ├── REPORT_BLUEPRINT.md          # 8页 NeurIPS 报告逐节大纲、图表规划与 AI 披露日志
│   └── tasks/                       # 4 位组员各自独立的详细任务书与 AI 提示词模板
├── scripts/
│   ├── verify_env.py                # 环境、硬件与 1993-2019 数据完整性一键自检脚本
│   ├── prepare_sq2_data.sh          # SQ2 数据准备脚本
│   └── sq4_rebuild_figures.py       # SQ4 离线重绘 Figure 4 脚本（无需原始 8.6GB 数据）
├── src/visual_alpha/                # 【全组共用底座包】DataLoader / Baseline CNN / 统一评估器
├── experiments/                     # 各成员可复现的命令行实验脚本（Baseline / SQ1–SQ4）
├── notebooks/                       # 结果可视化与 SQ4 交互式复现 Notebook
├── outputs/
│   ├── checkpoints/                 # 模型权重（git-ignored）
│   ├── predictions/                 # 统一格式的样本外预测结果 .parquet（git-ignored）
│   ├── gradcam/                     # SQ4 Top-2000 Grad-CAM 数组与抽样图像包
│   ├── figures/                     # 导出的 300 DPI 论文插图（PDF / PNG）
│   └── tables/                      # 导出的汇总实验结果表格（CSV / JSON）
└── report/                          # 8页 NeurIPS 技术报告源码 (main.tex/main.pdf) 与展示 Slides
```

---

## 5. 一键复现与验证命令

```bash
# 1. 环境与数据自检 + 单元测试
.venv/bin/python scripts/verify_env.py
.venv/bin/pytest

# 2. Baseline (5-seed) & SQ3 校准与选择性预测 (Member A: Hanchong Qiu)
.venv/bin/python experiments/exp_baseline.py
.venv/bin/python experiments/exp_sq3_calibration.py

# 3. SQ1 架构归纳偏置对比与遮挡消融 (Member B: Minyi Xie)
.venv/bin/python experiments/exp_sq1_inductive_bias.py
.venv/bin/python experiments/plot_sq1.py

# 4. SQ2 标签噪声过滤、标签平滑与多周期对比 (Member C: Dongqing Zhao)
.venv/bin/python experiments/exp_sq2_label_noise.py

# 5. SQ4 跨时代概念漂移、Top-2000 Grad-CAM 与离线重绘 (Member D: Yan Ziyuan)
.venv/bin/python experiments/sq4_train_modern.py --repo . --data-dir data/monthly_20d --baseline outputs/predictions/baseline_seed42.parquet --out outputs/sq4_modern
.venv/bin/python experiments/sq4_run_top2000_gradcam.py --archive outputs/gradcam/SQ4_Top2000_Images.zip --checkpoint outputs/checkpoints/baseline_seed42.pt --repo . --output outputs/sq4_top2000
.venv/bin/python scripts/sq4_rebuild_figures.py --root .

# 6. 编译 8 页 NeurIPS 技术报告
cd report && tectonic main.tex   # 或 pdflatex main.tex（运行两次）
```

