# AGENTS.md — AI Agent Cross-Session Guide

## Project Context
**Course**: MSDM 5054 Statistical Machine Learning (Final Group Project, 4 members, Due: Nov 22, 2026).  
**Topic**: Replication and statistical ML audit of Jiang, Kelly, & Xiu (2023), *"(Re-)Imag(in)ing Price Trends"* (*Journal of Finance*).  
**Goal**: Predict 20-day US stock return direction from $64 \times 60$ binary OHLC+MA+Volume chart images (1993–2019, $N = 2,196,994$) and investigate four focused scientific questions (SQ1: Inductive Bias & Masking, SQ2: Label Noise & Margin Filtering, SQ3: Probability Calibration, SQ4: Concept Drift & Grad-CAM Interpretability).

## Essential Documentation Map (Read Before Acting)
On branch `feat/sq2-label-noise`, start with [Member C's GPU execution plan](./docs/tasks/SQ2_EXECUTION_PLAN_zh.md). It contains the data preparation command, verified sample counts, C0–C7 matrix, evaluation details, and deliverables. The assignment and the three relevant lecture PDFs are included in this checkout.

Do not guess schemas, architectures, or metrics. Always consult the authoritative doc for your current task:
- **Data schema, pixel coordinates, CNN spec & prediction output contract**: [docs/DATA_AND_CONTRACTS.md](./docs/DATA_AND_CONTRACTS.md)
- **Team milestones, scope guardrails & integration checklist**: [docs/PROJECT_CHARTER.md](./docs/PROJECT_CHARTER.md)
- **8-page NeurIPS report outline, figure specs & LLM usage log**: [docs/REPORT_BLUEPRINT.md](./docs/REPORT_BLUEPRINT.md)
- **Role-specific implementation specs**:
  - Member A (Shared Core + Baseline + SQ3 Calibration): [docs/tasks/MEMBER_A_LEADER_BASELINE_AND_CALIBRATION.md](./docs/tasks/MEMBER_A_LEADER_BASELINE_AND_CALIBRATION.md)
  - Member B (SQ1 Inductive Bias + Input Masking): [docs/tasks/MEMBER_B_INDUCTIVE_BIAS_AND_ABLATION.md](./docs/tasks/MEMBER_B_INDUCTIVE_BIAS_AND_ABLATION.md)
  - Member C (SQ2 Label Noise + Loss / Horizon): [docs/tasks/MEMBER_C_LABEL_NOISE_AND_LOSS.md](./docs/tasks/MEMBER_C_LABEL_NOISE_AND_LOSS.md)
  - Member D (SQ4 Concept Drift + Grad-CAM + Report Integration): [docs/tasks/MEMBER_D_CONCEPT_DRIFT_AND_GRADCAM.md](./docs/tasks/MEMBER_D_CONCEPT_DRIFT_AND_GRADCAM.md)

## Architecture & Directory Layout
- `src/visual_alpha/`: Shared Python package imported by all experiments (`dataset`, `models`, `evaluate`, `utils`). Do **not** modify shared interfaces without updating [docs/DATA_AND_CONTRACTS.md](./docs/DATA_AND_CONTRACTS.md).
- `experiments/`: Reproducible CLI scripts per workstream (`exp_baseline.py`, `exp_sq1_inductive_bias.py`, `exp_sq2_label_noise.py`, `exp_sq3_calibration.py`, `exp_sq4_drift_gradcam.py`).
- `notebooks/`: Clean analysis & plotting notebooks consuming saved artifacts from `outputs/`.
- `data/monthly_20d/`: Raw per-year `.dat` (`uint8` memmap) and `.feather` files (git-ignored, ~8.6 GB).
- `outputs/`: `checkpoints/` (git-ignored weights), `predictions/` (standardized `.parquet` dumps), `figures/` (PDF/PNG for report), `tables/` (CSV/LaTeX tables).
- `scripts/verify_env.py`: Environment, hardware accelerator, and data integrity check.

## Critical Invariants & Gotchas (Strictly Enforced)
1. **Memory Safety (`np.memmap` Mandatory)**: Raw `.dat` files total 8.6 GB (`uint8`). Converting the full dataset to `float32` in RAM requires ~34 GB and will OOM/swap on 16–24 GB laptops. Always open `.dat` files with `np.memmap(..., dtype=np.uint8, mode="r").reshape((-1, 64, 60))` and cast to `float32 / 255.0` **inside mini-batches only**.
2. **Cross-Platform Device Selection**: Teammates may run macOS Apple Silicon (`mps`), Linux/Windows NVIDIA (`cuda`), or CPU. Always use automatic device resolution (`cuda` -> `mps` -> `cpu`), never hardcode `.cuda()` or `.to("mps")`.
3. **Label & Return Alignment**: `Retx_20d_label` (`1` if `Retx_20d > 0`, `0` if `Retx_20d <= 0`, `2` if `NaN`) is defined on **ex-dividend return `Retx_20d`**, NOT `Ret_20d`. Always filter out `Retx_20d_label == 2` first, pair threshold filtering with `Retx_20d`, and keep the OOS test set **100% unfiltered** (except `label != 2`) across all experiments for fair comparison.
4. **Strict Temporal Split**: Never leak post-1999 data into baseline training/validation. Default split: `1993–1999` (70% train / 30% val, seeded) and `2000–2019` (OOS test).
5. **LLM Disclosure Tracking**: Course rules require documenting how AI/LLMs are used. Append a brief 1-line entry to the AI Usage Log in [docs/REPORT_BLUEPRINT.md](./docs/REPORT_BLUEPRINT.md) whenever AI generates substantial code or analysis.

## Environment & Commands
- Run Python via `.venv`: `.venv/bin/python <script.py>` (or `uv run python <script.py>`).
- Verify environment & dataset: `.venv/bin/python scripts/verify_env.py`
- Lint & format check: `.venv/bin/ruff check src/ experiments/ scripts/`
- Run unit tests: `.venv/bin/pytest`
