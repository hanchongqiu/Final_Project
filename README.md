# Deconstructing Visual Alpha — MSDM 5054 Final Project

> **Course**: MSDM 5054 Statistical Machine Learning (Instructor: Prof. Yuan Yao, HKUST, Fall 2026)  
> **Team**: Hanchong Qiu (21356531) · Minyi Xie (21319480) · Dongqing Zhao (21420966) · Ziyuan Yan (21219496)  
> **Topic**: Replication and four-stage statistical ML audit of Jiang, Kelly, & Xiu (2023), *"(Re-)Imag(in)ing Price Trends"*, *The Journal of Finance*

---

## 1. Project Overview

Jiang, Kelly, and Xiu (2023) render 20-day U.S. equity OHLC price bars, 20-day moving averages (MA), and volume bars into $64 \times 60$ binary chart images and train CNNs to forecast 20-day return direction. Rather than treating the CNN as a black box, we build a unified memory-mapped benchmark ($N = 2,196,994$ stock-months, 1993–2019; $1,491,887$ OOS stock-months, 2000–2019) and audit four statistical machine learning questions along the modeling lifecycle:

| Module | Pipeline Stage | Scientific Question | Lead |
| :--- | :--- | :--- | :--- |
| **Baseline & RQ3** | Core pipeline & calibration | Baseline 5-seed CNN20D replication, post-hoc temperature scaling vs. deep ensembling, and selective tail prediction | **Hanchong Qiu** |
| **RQ1** | Architecture & input space | 2D-CNN vs. PCA+LR, pixel-summary LightGBM, Temporal 1D-CNN, and ResNet; spatial/temporal occlusion & volume ablation | **Minyi Xie** |
| **RQ2** | Supervision & horizons | Near-zero return margin filtering ($\lvert \text{Retx}_{20d} \rvert \ge \tau$, $\lvert t\text{-stat} \rvert \ge 0.5$), label smoothing ($\varepsilon = 0.1$), and 5/20/60-day transfer | **Dongqing Zhao** |
| **RQ4** | Temporal transfer & attribution | 2000s vs. 2010s decay, 2006–2012 modern retraining on matched 2013–2019 holdout ($N = 472,357$), and global Top-1,000 Grad-CAM | **Ziyuan Yan** |

---

## 2. Key Deliverables

- **Assignment Specification**: [`Final Project.pdf`](./Final%20Project.pdf) & official image demo [`Image Demo.html`](./Image%20Demo.html)
- **8-Page Technical Report (NeurIPS Style)**: [`report/main.pdf`](./report/main.pdf) (source: [`report/main.tex`](./report/main.tex))
- **Data Schema, Pixel Coordinates & Evaluation Contract**: [`docs/DATA_AND_CONTRACTS.md`](./docs/DATA_AND_CONTRACTS.md)

---

## 3. Environment Setup & Data Preparation

### Step 1: Dataset Placement
Because the unpacked raw dataset is ~8.6 GB, `data/monthly_20d/` is git-ignored. Download `img_data.zip` (see `Final Project.pdf`, p. 5) and extract it into `data/`:
```bash
unzip -q -o img_data.zip -d data/ -x "__MACOSX/*" "*/._*"
```
Expected layout:
```text
data/monthly_20d/
├── 20d_month_has_vb_[20]_ma_1993_images.dat
├── 20d_month_has_vb_[20]_ma_1993_labels_w_delay.feather
├── ...
└── 20d_month_has_vb_[20]_ma_2019_labels_w_delay.feather
```

### Step 2: Python Environment
```bash
# Option A (recommended, locked versions via uv):
uv sync --python 3.12 --extra dev

# Option B (venv + pip):
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Verify Environment & Unit Tests
```bash
.venv/bin/python scripts/verify_env.py
.venv/bin/pytest
```

---

## 4. Repository Structure

```text
Final_Project/
├── Final Project.pdf                # Course assignment specification
├── Image Demo.html                  # Original paper image-loading demo
├── README.md                        # Project overview & reproduction guide
├── pyproject.toml / uv.lock         # Locked uv environment specification
├── requirements.txt                 # Pip-compatible dependency list
├── docs/
│   └── DATA_AND_CONTRACTS.md        # Verified data schema, 64x60 pixel anatomy & CNN20D spec
├── src/visual_alpha/                # Shared package: memmap Dataset, CNN20D, Trainer, Evaluator
├── experiments/                     # Reproducible CLI experiment scripts (Baseline, RQ1–RQ4)
├── scripts/
│   ├── verify_env.py                # Environment, hardware & 1993–2019 data integrity check
│   ├── prepare_sq2_data.sh          # SQ2 data preparation helper
│   └── sq4_rebuild_figures.py       # Offline Figure 4 rebuild from bundled Grad-CAM artifacts
├── notebooks/
│   └── SQ4_Results_Walkthrough.ipynb # Interactive RQ4 results walkthrough
├── tests/                           # Unit tests for core pipeline, RQ1, and RQ2 horizon logic
├── outputs/
│   ├── checkpoints/                 # Saved model weights (git-ignored)
│   ├── predictions/                 # Standardized OOS .parquet predictions (git-ignored)
│   ├── gradcam/                     # RQ4 Top-2000 Grad-CAM arrays & sampled charts
│   ├── figures/                     # Publication-ready 300 DPI figures (PDF / PNG)
│   └── tables/                      # Summary & diagnostic metric tables (CSV / JSON)
└── report/
    ├── main.tex / main.pdf          # Final 8-page NeurIPS technical report
    └── neurips_msdm5054.sty         # NeurIPS-style LaTeX stylesheet
```

---

## 5. Reproduction Commands

```bash
# 1. Baseline (5-seed) & RQ3 Calibration / Selective Prediction (Hanchong Qiu)
.venv/bin/python experiments/exp_baseline.py
.venv/bin/python experiments/exp_sq3_calibration.py

# 2. RQ1 Inductive Bias & Spatial Masking Ablation (Minyi Xie)
.venv/bin/python experiments/exp_sq1_inductive_bias.py
.venv/bin/python experiments/plot_sq1.py

# 3. RQ2 Label Noise Margin Filtering, Label Smoothing & Horizons (Dongqing Zhao)
.venv/bin/python experiments/exp_sq2_label_noise.py

# 4. RQ4 Temporal Transfer, Top-2000 Grad-CAM & Offline Replot (Ziyuan Yan)
.venv/bin/python experiments/sq4_train_modern.py --repo . --data-dir data/monthly_20d --baseline outputs/predictions/baseline_seed42.parquet --out outputs/sq4_modern
.venv/bin/python experiments/sq4_run_top2000_gradcam.py --archive outputs/gradcam/SQ4_Top2000_Images.zip --checkpoint outputs/checkpoints/baseline_seed42.pt --repo . --output outputs/sq4_top2000
.venv/bin/python scripts/sq4_rebuild_figures.py --root .

# 5. Recompile 8-Page Report
cd report && tectonic main.tex   # or pdflatex main.tex (run twice)
```

