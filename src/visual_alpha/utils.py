"""Device selection, reproducibility seeds, paths, and image masking utilities."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "monthly_20d"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

IMG_H = 64
IMG_W = 60
DEFAULT_ENSEMBLE_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_SPLIT_SEED = 42


def ensure_output_dirs() -> None:
    """Ensure standard output subdirectories exist."""
    for directory in (CHECKPOINTS_DIR, PREDICTIONS_DIR, FIGURES_DIR, TABLES_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def get_device(prefer: str | None = None) -> torch.device:
    """Resolve compute accelerator automatically: cuda -> mps -> cpu."""
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def apply_image_mask(x: torch.Tensor, mask_mode: str | None) -> torch.Tensor:
    """Apply in-place-safe spatial masking to (..., 64, 60) chart tensors.

    Supported modes (defined in docs/DATA_AND_CONTRACTS.md Section 3):
    - None / "none": untouched full image
    - "no_volume": zero out volume bars rows 52:64
    - "price_only": alias for "no_volume"
    - "volume_only": zero out OHLC+MA rows 0:51
    - "recent_10d": zero out oldest 10 days cols 0:30, keep recent 10 days cols 30:60
    - "early_10d" / "old_10d": zero out recent 10 days cols 30:60, keep oldest 10 days cols 0:30
    - "recent_5d": zero out oldest 15 days cols 0:45, keep recent 5 days cols 45:60
    """
    if mask_mode is None or mask_mode == "none":
        return x
    out = x.clone()
    if mask_mode in ("no_volume", "price_only"):
        out[..., 52:64, :] = 0
    elif mask_mode == "volume_only":
        out[..., 0:51, :] = 0
    elif mask_mode == "recent_10d":
        out[..., :, 0:30] = 0
    elif mask_mode in ("early_10d", "old_10d"):
        out[..., :, 30:60] = 0
    elif mask_mode == "recent_5d":
        out[..., :, 0:45] = 0
    else:
        msg = f"Unsupported image mask_mode: {mask_mode!r}"
        raise ValueError(msg)
    return out
