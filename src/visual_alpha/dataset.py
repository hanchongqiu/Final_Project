"""Memory-safe np.memmap chart dataset and standardized temporal split builders."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from visual_alpha.utils import DEFAULT_DATA_DIR, DEFAULT_SPLIT_SEED, IMG_H, IMG_W

HORIZON_COLUMNS: dict[str, tuple[str, str, str, str]] = {
    "20d": ("Retx_20d_label", "Retx_20d", "Ret_20d", "Retx_20d_tstat"),
    "5d": ("Retx_5d_label", "Retx_5d", "Ret_5d", "Retx_5d_tstat"),
    "60d": ("Retx_60d_label", "Retx_60d", "Ret_60d", "Retx_60d_tstat"),
}


@dataclass(frozen=True)
class SplitDatasets:
    """Standardized Train / Val / OOS Test ChartDataset bundle."""

    train: ChartDataset
    val: ChartDataset
    test: ChartDataset


class ChartDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Memory-mapped 64x60 uint8 stock chart dataset across multiple years.

    Invariants (docs/DATA_AND_CONTRACTS.md):
    - Opens per-year `.dat` files lazily with `np.memmap(..., dtype=np.uint8, mode='r')`.
    - Filters out invalid labels (`label == 2`) first.
    - Supports vectorized slice/batch indexing (`get_batch`) for high-throughput training
      without multiprocessing pickling overhead or float32 RAM expansion.
    """

    def __init__(
        self,
        years: Iterable[int],
        *,
        data_dir: Path | str = DEFAULT_DATA_DIR,
        horizon: str = "20d",
        indices: np.ndarray | None = None,
        min_abs_retx: float = 0.0,
        min_abs_tstat: float = 0.0,
    ) -> None:
        if horizon not in HORIZON_COLUMNS:
            msg = f"Unsupported horizon {horizon!r}; expected one of {tuple(HORIZON_COLUMNS)}"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        self.years = sorted({int(y) for y in years})
        self.horizon = horizon
        self.label_col, self.retx_col, self.ret_col, self.tstat_col = HORIZON_COLUMNS[horizon]

        meta_frames: list[pd.DataFrame] = []
        self._year_Num_rows: dict[int, int] = {}

        cols_to_read = [
            "Date",
            "StockID",
            "MarketCap",
            self.label_col,
            self.retx_col,
            self.ret_col,
            self.tstat_col,
        ]
        # Always retain 20d return columns as well so prediction parquet contract is satisfied
        for extra_col in ("Retx_20d", "Ret_20d"):
            if extra_col not in cols_to_read:
                cols_to_read.append(extra_col)

        for year in self.years:
            dat_path = self._dat_path(year)
            feather_path = self._feather_path(year)
            if not dat_path.exists() or not feather_path.exists():
                msg = f"Missing data files for year {year} in {self.data_dir}"
                raise FileNotFoundError(msg)

            n_imgs = dat_path.stat().st_size // (IMG_H * IMG_W)
            df = pd.read_feather(feather_path, columns=cols_to_read)
            if len(df) != n_imgs:
                msg = f"Alignment error in {year}: {n_imgs} images != {len(df)} feather rows"
                raise RuntimeError(msg)

            self._year_Num_rows[year] = n_imgs
            df["year"] = np.int16(year)
            df["row_idx"] = np.arange(n_imgs, dtype=np.int32)

            valid_mask = df[self.label_col].to_numpy() != 2
            df = df.loc[valid_mask].reset_index(drop=True)
            meta_frames.append(df)

        full_meta = pd.concat(meta_frames, ignore_index=True)

        if indices is not None:
            full_meta = full_meta.iloc[np.asarray(indices, dtype=np.int64)].reset_index(drop=True)

        if min_abs_retx > 0.0:
            keep = full_meta[self.retx_col].abs() >= float(min_abs_retx)
            full_meta = full_meta.loc[keep].reset_index(drop=True)

        if min_abs_tstat > 0.0:
            keep = full_meta[self.tstat_col].abs() >= float(min_abs_tstat)
            full_meta = full_meta.loc[keep].reset_index(drop=True)

        self.metadata = full_meta
        self._years_arr = full_meta["year"].to_numpy(dtype=np.int16, copy=True)
        self._row_idx_arr = full_meta["row_idx"].to_numpy(dtype=np.int32, copy=True)
        self._labels_arr = full_meta[self.label_col].to_numpy(dtype=np.int64, copy=True)
        self._memmaps: dict[int, np.memmap] = {}

    def _dat_path(self, year: int) -> Path:
        return self.data_dir / f"20d_month_has_vb_[20]_ma_{year}_images.dat"

    def _feather_path(self, year: int) -> Path:
        return self.data_dir / f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"

    def _get_memmap(self, year: int) -> np.memmap:
        mm = self._memmaps.get(year)
        if mm is None:
            n_rows = self._year_Num_rows[year]
            mm = np.memmap(
                self._dat_path(year),
                dtype=np.uint8,
                mode="r",
                shape=(n_rows, IMG_H, IMG_W),
            )
            self._memmaps[year] = mm
        return mm

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_memmaps"] = {}
        return state

    def __len__(self) -> int:
        return len(self._labels_arr)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        year = int(self._years_arr[idx])
        row_idx = int(self._row_idx_arr[idx])
        img_np = np.array(self._get_memmap(year)[row_idx], dtype=np.uint8, copy=True)
        x = torch.from_numpy(img_np).unsqueeze(0)  # (1, 64, 60) uint8
        y = torch.tensor(int(self._labels_arr[idx]), dtype=torch.long)
        return x, y

    def get_batch(self, batch_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized uint8 batch gather across per-year memmaps.

        Groups requested indices by year so OS page reads are sequential/coalesced,
        returning `(B, 1, 64, 60)` `uint8` and `(B,)` `int64` tensors without float32 RAM bloat.
        """
        idx_arr = np.asarray(batch_indices, dtype=np.int64)
        bsz = len(idx_arr)
        batch_imgs = np.empty((bsz, 1, IMG_H, IMG_W), dtype=np.uint8)

        batch_years = self._years_arr[idx_arr]
        batch_rows = self._row_idx_arr[idx_arr]

        for year in np.unique(batch_years):
            pos = np.nonzero(batch_years == year)[0]
            mm = self._get_memmap(int(year))
            rows = batch_rows[pos]
            # Sort row offsets within year for sequential disk/page-cache access
            order = np.argsort(rows)
            batch_imgs[pos[order], 0] = mm[rows[order]]

        x = torch.from_numpy(batch_imgs)
        y = torch.from_numpy(self._labels_arr[idx_arr])
        return x, y

    def iter_batches(
        self,
        batch_size: int = 256,
        *,
        shuffle: bool = False,
        seed: int = 42,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Fast in-process vectorized batch iterator over memmapped uint8 images."""
        n = len(self)
        if shuffle:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n)
        else:
            perm = np.arange(n, dtype=np.int64)

        for start in range(0, n, batch_size):
            yield self.get_batch(perm[start : start + batch_size])

    def build_prediction_frame(
        self,
        prob_up: np.ndarray,
        *,
        logit_down: np.ndarray | None = None,
        logit_up: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Build standardized `.parquet` prediction DataFrame matching DATA_AND_CONTRACTS.md."""
        n = len(self)
        if len(prob_up) != n:
            msg = f"Length mismatch: got {len(prob_up)} predictions for dataset of size {n}"
            raise ValueError(msg)

        ld = (
            np.asarray(logit_down, dtype=np.float32)
            if logit_down is not None
            else np.full(n, np.nan, dtype=np.float32)
        )
        lu = (
            np.asarray(logit_up, dtype=np.float32)
            if logit_up is not None
            else np.full(n, np.nan, dtype=np.float32)
        )

        out = pd.DataFrame(
            {
                "year": self._years_arr.astype(np.int16, copy=False),
                "row_idx": self._row_idx_arr.astype(np.int32, copy=False),
                "Date": self.metadata["Date"].to_numpy(),
                "StockID": self.metadata["StockID"].astype(str).to_numpy(),
                "y_true": self._labels_arr.astype(np.int8, copy=False),
                "logit_down": ld,
                "logit_up": lu,
                "prob_up": np.asarray(prob_up, dtype=np.float32),
                "Retx_20d": self.metadata["Retx_20d"].to_numpy(dtype=np.float32, copy=False),
                "Ret_20d": self.metadata["Ret_20d"].to_numpy(dtype=np.float32, copy=False),
                "MarketCap": self.metadata["MarketCap"].to_numpy(dtype=np.float32, copy=False),
            }
        )
        if self.horizon != "20d":
            out[self.retx_col] = self.metadata[self.retx_col].to_numpy(dtype=np.float32, copy=False)
            out[self.ret_col] = self.metadata[self.ret_col].to_numpy(dtype=np.float32, copy=False)
        return out


def build_train_val_test_splits(
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    train_val_years: Iterable[int] = range(1993, 2000),
    test_years: Iterable[int] = range(2000, 2020),
    val_ratio: float = 0.30,
    split_seed: int = DEFAULT_SPLIT_SEED,
    horizon: str = "20d",
    train_min_abs_retx: float = 0.0,
    train_min_abs_tstat: float = 0.0,
) -> SplitDatasets:
    """Create standard temporal splits with deterministic 70/30 train/val partitioning.

    Crucially, margin filtering (`train_min_abs_retx` / `train_min_abs_tstat`) is applied
    ONLY to the training split; validation and OOS test splits remain 100% unfiltered
    (apart from dropping `label == 2`) to guarantee fair model selection and evaluation.
    """
    pool = ChartDataset(train_val_years, data_dir=data_dir, horizon=horizon)
    n_pool = len(pool)
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_pool)
    n_val = round(n_pool * val_ratio)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])

    train_ds = ChartDataset(
        train_val_years,
        data_dir=data_dir,
        horizon=horizon,
        indices=train_idx,
        min_abs_retx=train_min_abs_retx,
        min_abs_tstat=train_min_abs_tstat,
    )
    val_ds = ChartDataset(
        train_val_years,
        data_dir=data_dir,
        horizon=horizon,
        indices=val_idx,
    )
    test_ds = ChartDataset(
        test_years,
        data_dir=data_dir,
        horizon=horizon,
    )
    return SplitDatasets(train=train_ds, val=val_ds, test=test_ds)
