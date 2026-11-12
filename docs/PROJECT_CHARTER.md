# 项目统筹章程与协作规范 (`PROJECT_CHARTER.md`)

> **适用对象**：组长（统领全局与进度验收）及全体 4 位小组成员  
> **课程与截止时间**：MSDM 5054 Statistical Machine Learning — **2026 年 11 月 22 日 23:59**

---

## 1. 项目战略定位：为什么我们的方案能拿高分且工作量可控？

课程说明（[Final Project.pdf](../Final%20Project.pdf) Section 1 & 4）明确指出：评分最看重**科学推理链条（问题定义 $\to$ 受控实验设计 $\to$ 深度统计解释）**，而非盲目堆砌庞大模型或刷排行榜精度。

原论文 Jiang, Kelly, & Xiu (2023) 已经证明了“2D K线图 + CNN 能赚钱”。如果只复现一遍，缺乏我们自己的思考；如果试图发明打败 *Journal of Finance* 顶刊的新金融因子，则超出了课程时间与算力边界。

因此，我们将整个项目定位为一次**体系化的统计机器学习诊断审计（Deconstructing Visual Alpha: A Statistical ML Audit）**——固定同一套高质量干净数据集与统一评测底座，沿着机器学习流水线的 **4 个关键环节**各自切入一个经典统计学习问题：

```text
[输入与架构层] ──▶ 组员 B (SQ1): 2D 空间卷积归纳偏置 vs. 传统统计学习(PCA+LR / LightGBM) & 遮挡消融
       │
[标签与损失层] ──▶ 组员 C (SQ2): 金融低信噪比 0 附近标签噪声过滤 (|Retx| >= τ) & 标签平滑
       │
[概率输出层]   ──▶ 组员 A (SQ3): 基线底座搭建 + 深度网络过度自信诊断、集成校准与高置信选择性预测
       │
[时间与解释层] ──▶ 组员 D (SQ4): 跨时代概念漂移 (1990s vs. 2010s) + Grad-CAM 群体平均视觉归因 + 统稿
```

### 核心优势：高度解耦，零阻塞并行！
* 一旦组长（组员 A）交付公共底座包 `src/visual_alpha/`（约 250~300 行代码）和跑通的 Baseline 预测文件，**组员 B、C、D 三条实验支线互不依赖**！哪怕某位组员进度稍慢或得出“负向结论”（例如过滤过多噪声反而因为样本量减少导致方差上升），也完全不影响其他章节，且“偏差-方差权衡下的倒 U 型/负向结论”本身就是极佳的统计机器学习分析素材。

---

## 2. 工程量护栏（Scope Guardrails —— 严防工作量失控！）

为了确保 4 位同学在期末季轻松、高质量完赛，全组严格遵守以下**“三做三不做”原则**：

1. **只用 `monthly_20d` 单一数据集，绝不去额外爬取外部宏观/财报数据**（SQ2 探索 5d/60d 周期直接复用当前 `.feather` 表中现成自带的 `Retx_5d_label` 和 `Retx_60d_label` 列即可，无需下载 5d/60d 图像包！）。
2. **锁定基线超参数，绝不做网格搜索（No Grid Search）**：所有 CNN 变体统一采用 `lr=1e-5`、`batch_size=256`、验证集早停 `patience=2`（通常 5~9 个 Epoch 即收敛，在 M 系列芯片或 RTX GPU 上跑完一次仅需 **5~10 分钟**）。全组加起来总共只需约 **12~15 次短训练**。
3. **代码精简重用**：全组总 Python 代码量控制在 **700~900 行**以内；所有组员复用 `visual_alpha.dataset` 与 `visual_alpha.evaluate`，每人只需编写 **1 个约 150~200 行的实验脚本**和 **1 张核心组合图**。

---

## 3. 四人角色分工总览

| 角色 | 工程开发交付物 | 报告与展示交付物 | 预计代码与运行耗时 |
| :--- | :--- | :--- | :--- |
| **组员 A（组长）** | 1. 公共底座 `src/visual_alpha/`<br>2. Baseline 5-seed 训练 (`exp_baseline.py`)<br>3. SQ3 校准与选择性预测 (`exp_sq3_calibration.py`) | • Section 1 (Introduction 前半) & Section 2 (Setup)<br>• Section 5 (SQ3 Calibration + Fig 3)<br>• Table 1 (Baseline 复现验证表) | 核心包 ~300 行 + 脚本 ~150 行；GPU 运行约 45 分钟 |
| **组员 B** | SQ1 传统模型对比与像素遮挡消融 (`exp_sq1_inductive_bias.py`)：PCA+LR、像素统计+LightGBM、1D-CNN、轻量 ResNet + 3组遮挡推理 | • Section 3 (SQ1 Inductive Bias & Input Masking)<br>• Table 2 + Fig 1 (架构对比与遮挡衰减柱状图) | 脚本 ~200 行；遮挡推理无需重训，总运行约 35 分钟 |
| **组员 C** | SQ2 标签噪声边际过滤与损失函数 (`exp_sq2_label_noise.py`)：`τ ∈ {0%, 1%, 3%, 5%}`、Label Smoothing `ε=0.1`、同图预测 5d/60d 标签 | • Section 4 (SQ2 Label Noise, Margin & Horizon)<br>• Fig 2 (阈值过滤偏差-方差曲线 + 多周期衰减图) | 脚本 ~180 行；6 次轻量训练，总运行约 45 分钟 |
| **组员 D** | SQ4 跨时代漂移与 Grad-CAM (`exp_sq4_drift_gradcam.py`)：`2006-2012` 重训模型对比 + 单样本与 Top-1000 平均 Grad-CAM 热力图 | • Section 6 (SQ4 Drift & Grad-CAM + Fig 4 + Table 3)<br>• Section 7 (Conclusion + Contributions + AI 声明)<br>• **8页报告排版统稿 + 5分钟展示 PPT/录制合成** | 脚本 ~180 行；仅 1~2 次重训，承担统稿核心工作 |

---

## 4. 建议 10 天冲刺里程碑（倒排时间表）

假设以截止日 `Day 10`（11 月 22 日）倒推（若提前启动可等比例拉宽，更从容）：

* **Phase 1：底座搭建与对齐（Day 1 – Day 2）**
  * **组长 (A)**：完成 `src/visual_alpha/` 公共代码编写与单元测试，跑通 1 个单种子 Baseline 并导出首份标准 `.parquet` 预测表，推送至主分支。
  * **组员 B / C / D**：克隆仓库、解压 `img_data.zip`、运行 `.venv/bin/python scripts/verify_env.py` 通过环境自检，阅读各自的任务书 `docs/tasks/MEMBER_*.md`。
* **Phase 2：四线并行实验推进（Day 3 – Day 6）**
  * **组员 A**：跑完 5-seed Ensemble Baseline，完成 SQ3 温度缩放、可靠性图（Reliability Diagram）与 Top-$k\%$ 高置信准确率曲线。
  * **组员 B**：完成 PCA+LR、像素特征 LightGBM、1D-CNN、ResNet18 对比，并直接加载组长训练好的 Baseline 权重完成 Volume / 时间窗口遮挡消融实验。
  * **组员 C**：跑完 4 组 `|Retx_20d| >= τ` 过滤实验、Label Smoothing 实验及 `5d/20d/60d` 多周期可预测性对比。
  * **组员 D**：跑完 `2006–2012` 现代模型训练，在 `2013–2019` 上对比 `1993–1999` 老模型，并编写 Grad-CAM 生成经典反转形态与全局平均热力图。
  * **Day 6 晚组会 Check-in（30 分钟）**：核对全组 `.parquet` 指标表与 4 张主图草稿，确认每个模块的科学故事线。
* **Phase 3：报告撰写与图表精修（Day 7 – Day 8）**
  * 每位成员按 [REPORT_BLUEPRINT.md](./REPORT_BLUEPRINT.md) 字数配额撰写自己负责的小节草稿，统一配色导出 `300 DPI` PDF/PNG 插图放入 `outputs/figures/`。
  * **组员 D + 组长 A**：在 NeurIPS 模板中合稿，严格压缩排版至 **8 页以内**。
* **Phase 4：5 分钟视频录制与终审提交（Day 9 – Day 10）**
  * 制作 6~7 页精炼 Slides，录制 **$\le 5$ 分钟**展示视频（建议语速自然、控制在 4 分 30 秒左右，严禁超时）。
  * 核对下方提交清单，于 **11 月 22 日 23:59 前**打包上传 Canvas。

---

## 5. Git 多人协作规范（简单无冲突工作流）

1. **分支命名**：每位成员在独立分支开发，避免直接在 `main` 上改同一文件：
   * 组员 A：`feat/core-and-sq3`
   * 组员 B：`feat/sq1-inductive-bias`
   * 组员 C：`feat/sq2-label-noise`
   * 组员 D：`feat/sq4-drift-gradcam`
2. **文件边界隔离**：
   * 公共目录 `src/visual_alpha/` 由组长维护；若组员 B/C/D 需要新增专属模型（如 `CNN1D`、`ResNet`）或工具函数，可放在自己的实验脚本中，或提 PR 由组长合入 `src/visual_alpha/`。
   * 每人的实验脚本独立命名（`experiments/exp_sq1_*.py` 等），导出的图表以 `fig1_sq1_*`、`fig2_sq2_*`、`fig3_sq3_*`、`fig4_sq4_*` 区分，**天然零合并冲突**。
3. **严禁提交大文件**：提交前执行 `git status`，确认没有误加 `.dat`、`.feather`、`.pt`、`.parquet` 等大体积二进制文件（`.gitignore` 已自动拦截）。

---

## 6. 组长最终提交验收清单（Canvas Submission Checklist）

根据 [Final Project.pdf](../Final%20Project.pdf) Section 1 规定，提交前逐项打钩确认：

- [ ] **小组人数合规**：1–4 人，封面写明全体成员姓名、学号与邮箱。
- [ ] **NeurIPS 格式 8 页技术报告 (`report.pdf`)**：严格不超过 8 页，包含问题背景、4 个科学子问题、实验图表与统计分析结论。
- [ ] **明确标注成员分工 (Member Contributions)**：在报告末尾清晰列出每位组员的具体贡献。
- [ ] **明确披露 AI / 大模型使用情况 (LLM Usage Disclosure)**：根据 [REPORT_BLUEPRINT.md](./REPORT_BLUEPRINT.md) 中的登记表，诚实、清晰说明如何使用 AI 辅助环境配置、代码调试与英文润色。
- [ ] **可复现源代码与说明**：附带干净的 `src/`、`experiments/`、`notebooks/`、`README.md` 与 `requirements.txt`（不含 8.6GB 原始数据，注明数据下载与放置路径）。
- [ ] **$\le 5$ 分钟展示视频**：MP4 格式，音画清晰，时长严格 $\le 5:00$。
