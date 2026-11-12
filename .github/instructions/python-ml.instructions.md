---
description: "Use when writing or modifying Python code, PyTorch models, data loaders, evaluation scripts, or Jupyter notebooks. Covers np.memmap memory safety, MPS/CUDA compatibility, pixel anatomy, and output schemas."
applyTo: "src/**/*.py, experiments/**/*.py, scripts/**/*.py, notebooks/**/*.ipynb"
---
# Python & Machine Learning Engineering Guidelines

## Data Loading & Memory Constraints
- **Never** load all years into a dense `np.ndarray` or convert whole-year arrays to `float32` in memory.
- Keep `np.memmap` handles lightweight (or open lazily per worker in PyTorch `DataLoader` to avoid multiprocessing pickling issues on macOS/Windows `spawn`).
- Normalize binary pixels `{0, 255}` via `x.float().div_(255.0)` after moving the `uint8` tensor batch to the target device (`mps`/`cuda`/`cpu`)—this cuts host-to-GPU memory bandwidth by 4x.
- **macOS ARM64 OpenMP Rule**: In any script importing both `lightgbm` and `torch`, **always `import lightgbm` BEFORE `import torch`** (or run LightGBM in a separate process / `n_jobs=1`) to prevent dual-`libomp` segmentation faults on Apple Silicon.

## Verified $64 \times 60$ Image Pixel Coordinates
- Shape: `(H=64, W=60)`, `dtype=uint8`, values strictly in `{0, 255}`.
- **Rows `0:51` (51 rows)**: OHLC price bars + 20-day Moving Average (MA) curve (`row 0` = window high, `row 50` = window low).
- **Row `51` (1 row)**: All-zero black separator line (`img[51, :] == 0` everywhere).
- **Rows `52:64` (12 rows)**: Volume bars rising upward from `row 63`; non-zero **only** at middle column `3*d + 1` for each day `d in 0..19`.
- **Columns `0:60` (20 days $\times$ 3 cols/day)**: Day `d` (`d=0` oldest, `d=19` most recent `Date`) occupies `col = 3*d` (Open), `3*d + 1` (High-Low + Volume), `3*d + 2` (Close).

## Reproducibility & Standardized Evaluation
- Seed Python `random`, `numpy`, and `torch` (`manual_seed`) explicitly; default ensemble seeds: `[42, 43, 44, 45, 46]`.
- No hyperparameter grid searches: lock baseline architecture and optimizer settings defined in [docs/DATA_AND_CONTRACTS.md](../../docs/DATA_AND_CONTRACTS.md) unless testing an explicit ablation variable.
- Every experiment must save OOS predictions to `outputs/predictions/<exp_name>.parquet` following the exact column schema in [docs/DATA_AND_CONTRACTS.md](../../docs/DATA_AND_CONTRACTS.md) and compute metrics exclusively via `visual_alpha.evaluate`.
