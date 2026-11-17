"""Member C — SQ2 Label Noise, Margin Filtering & Multi-Horizon matrix (C0–C7).

Implements the fixed experiment matrix of docs/tasks/SQ2_EXECUTION_PLAN_zh.md §2:
C0 reuses the delivered seed-42 baseline predictions; C1–C7 are fresh trainings that
vary only the training-sample margin filter, label smoothing, or the prediction
horizon (5d / 60d). All other hyperparameters are locked (split seed 42, train seed
42, Adam lr 1e-5, batch 256, max 15 epochs, patience 2; `seed_everything(42)` is
called BEFORE `CNN20D()` construction for C1–C7).

Usage:
    # Smoke test a single configuration (1 epoch):
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/exp_sq2_label_noise.py \
        --exp C2 --max-epochs 1

    # Full matrix, two GPUs in parallel:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/exp_sq2_label_noise.py \
        --exp C1,C2,C3,C4 > logs/sq2_gpu0.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/exp_sq2_label_noise.py \
        --exp C5,C6,C7 > logs/sq2_gpu1.log 2>&1 &
    wait

    # Rebuild every table & figure from saved predictions + manifest (no training):
    .venv/bin/python experiments/exp_sq2_label_noise.py --summarize

Outputs (per experiment `sq2_<suffix>.parquet`, `val_sq2_<suffix>.parquet`,
`sq2_<suffix>.pt`, plus a run-manifest row) and the aggregate artifacts:
    outputs/tables/sq2_run_manifest.csv
    outputs/tables/sq2_noise_and_horizon.csv
    outputs/tables/sq2_common_sample_horizon.csv
    outputs/tables/sq2_filter_retention_by_cap.csv
    outputs/tables/sq2_paired_bootstrap.csv
    outputs/figures/fig2_sq2_noise_horizon.pdf / .png (300 dpi)
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from visual_alpha import (
    CHECKPOINTS_DIR,
    CNN20D,
    FIGURES_DIR,
    PREDICTIONS_DIR,
    TABLES_DIR,
    SplitDatasets,
    build_train_val_test_splits,
    compute_decile_portfolio,
    ensure_output_dirs,
    evaluate_predictions,
    get_device,
    predict_dataset,
    save_predictions,
    seed_everything,
    train_model,
)

SPLIT_SEED = 42
TRAIN_SEED = 42

MANIFEST_PATH = TABLES_DIR / "sq2_run_manifest.csv"
BASELINE_TEST_PARQUET = PREDICTIONS_DIR / "baseline_seed42.parquet"
BASELINE_CKPT = CHECKPOINTS_DIR / "baseline_seed42.pt"
EXPECTED_COMMON_SAMPLE_N = 1_471_746

MANIFEST_COLUMNS = [
    "exp_id",
    "name",
    "horizon",
    "train_n",
    "retention_pct",
    "train_pos_rate",
    "val_n",
    "test_n",
    "best_epoch",
    "best_val_loss",
    "elapsed_sec",
    "timestamp",
]

HORIZON_RET_COLS: dict[str, tuple[str, str]] = {
    "5d": ("Ret_5d", "Retx_5d"),
    "20d": ("Ret_20d", "Retx_20d"),
    "60d": ("Ret_60d", "Retx_60d"),
}


@dataclass(frozen=True)
class ExpConfig:
    """One cell of the fixed SQ2 experiment matrix."""

    exp_id: str
    name: str  # file-name suffix
    horizon: str
    filter_rule: str  # human-readable rule for the main table
    min_abs_retx: float = 0.0
    min_abs_tstat: float = 0.0
    label_smoothing: float = 0.0


SQ2_EXPERIMENTS: dict[str, ExpConfig] = {
    "C0": ExpConfig("C0", "c0_baseline", "20d", "none (full sample; reuse seed-42 baseline)"),
    "C1": ExpConfig("C1", "c1_margin_tau001", "20d", "abs(Retx_20d) >= 0.01", min_abs_retx=0.01),
    "C2": ExpConfig("C2", "c2_margin_tau003", "20d", "abs(Retx_20d) >= 0.03", min_abs_retx=0.03),
    "C3": ExpConfig("C3", "c3_margin_tau005", "20d", "abs(Retx_20d) >= 0.05", min_abs_retx=0.05),
    "C4": ExpConfig("C4", "c4_tstat05", "20d", "abs(Retx_20d_tstat) >= 0.5", min_abs_tstat=0.5),
    "C5": ExpConfig(
        "C5", "c5_label_smooth010", "20d", "none + label_smoothing=0.1", label_smoothing=0.1
    ),
    "C6": ExpConfig("C6", "c6_horizon5d", "5d", "none (full sample; 5d labels)"),
    "C7": ExpConfig("C7", "c7_horizon60d", "60d", "none (full sample; 60d labels)"),
}
TRAINING_EXPS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]

# Module-level cache of unfiltered splits per horizon (retention denominators, C0 pool).
_UNFILTERED_SPLITS: dict[str, SplitDatasets] = {}


# ---------------------------------------------------------------------------
# Cross-horizon return fallback (Member C core fix; tested by tests/test_sq2_horizon_fallback.py)
# ---------------------------------------------------------------------------


def fill_horizon_returns(df: pd.DataFrame, *, ret_col: str, retx_col: str) -> pd.DataFrame:
    """Return an evaluation copy with same-horizon `Retx` filling missing `Ret`.

    `evaluate_predictions`'s internal decile-portfolio fallback always substitutes
    `Retx_20d`, which is WRONG for 5d/60d horizons. This helper fills the copy with
    the SAME-horizon ex-dividend return instead and raises if any gap remains.
    The original prediction parquet frame is never modified.
    """
    df_eval = df.copy()
    missing = df_eval[ret_col].isna()
    df_eval.loc[missing, ret_col] = df_eval.loc[missing, retx_col]
    n_still_nan = int(df_eval[ret_col].isna().sum())
    if n_still_nan:
        msg = f"{ret_col} still has {n_still_nan} NaN values after filling from {retx_col}"
        raise ValueError(msg)
    return df_eval


def evaluate_sq2(
    pred_df: pd.DataFrame, horizon: str, *, label: str = ""
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate one experiment under the execution plan §3 rules.

    - 20d (C0–C5): direct `evaluate_predictions(df, ret_col="Ret_20d")`.
    - 5d/60d (C6/C7): fill the evaluation copy with the SAME-horizon Retx, assert no
      NaN remains, evaluate, and cross-check the decile-portfolio summary against
      `compute_decile_portfolio(ret_col, fallback_ret_col=retx_col)` on the raw frame.

    Returns (metrics dict with an added `ls_mean_ret_pct`, monthly LS return frame).
    """
    ret_col, retx_col = HORIZON_RET_COLS[horizon]
    if horizon == "20d":
        metrics = evaluate_predictions(pred_df, ret_col=ret_col)
        port, monthly = compute_decile_portfolio(
            pred_df, ret_col=ret_col, fallback_ret_col=retx_col
        )
    else:
        n_missing = int(pred_df[ret_col].isna().sum())
        df_eval = fill_horizon_returns(pred_df, ret_col=ret_col, retx_col=retx_col)
        metrics = evaluate_predictions(df_eval, ret_col=ret_col)
        port, monthly = compute_decile_portfolio(df_eval, ret_col=ret_col)
        port_ref, _ = compute_decile_portfolio(
            pred_df, ret_col=ret_col, fallback_ret_col=retx_col
        )
        max_diff = max(abs(float(port[k]) - float(port_ref[k])) for k in port_ref)
        assert max_diff < 1e-9, f"fallback cross-check failed: max summary diff {max_diff:.3e}"
        print(
            f"  [fallback xcheck {label}] filled {n_missing:,} missing {ret_col} from "
            f"{retx_col} (no NaN left); decile summary matches direct "
            f"fallback call (max diff {max_diff:.2e}) — OK",
            flush=True,
        )
    metrics["ls_mean_ret_pct"] = float(monthly["ls_ret"].mean() * 100.0)
    return metrics, monthly


# ---------------------------------------------------------------------------
# Shared helpers: paths, splits, manifest
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Member C — SQ2 label-noise & multi-horizon experiment matrix (C0–C7)."
    )
    parser.add_argument(
        "--exp",
        action="append",
        default=None,
        help=(
            "Experiment ID(s): C0..C7 or 'all'; repeatable and/or comma-separated "
            "(e.g. --exp C1,C2 --exp C5). Default: C1..C7 ('all')."
        ),
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Only rebuild tables & figure from saved predictions + manifest; no training.",
    )
    parser.add_argument("--max-epochs", type=int, default=15, help="Max training epochs.")
    parser.add_argument("--patience", type=int, default=2, help="Early-stopping patience.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training mini-batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=1024, help="Inference batch size.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cuda:0", "cuda:1", "cpu"],
        help="Compute device (default: auto = cuda -> mps -> cpu).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain/re-infer even if the test parquet already exists.",
    )
    return parser.parse_args()


def resolve_exps(spec: list[str] | None) -> list[str]:
    """Expand repeatable/comma-separated --exp values into an ordered, de-duplicated list."""
    ids: list[str] = []
    for chunk in spec or ["all"]:
        for part in chunk.split(","):
            part = part.strip().upper()
            if not part:
                continue
            if part == "ALL":
                ids.extend(TRAINING_EXPS)
            elif part in SQ2_EXPERIMENTS:
                ids.append(part)
            else:
                msg = f"Unknown experiment {part!r}; choose from {list(SQ2_EXPERIMENTS)} or 'all'."
                raise SystemExit(msg)
    return list(dict.fromkeys(ids))


def exp_paths(cfg: ExpConfig) -> tuple[Path, Path, Path]:
    """Return (test parquet, val parquet, checkpoint) paths for one experiment."""
    return (
        PREDICTIONS_DIR / f"sq2_{cfg.name}.parquet",
        PREDICTIONS_DIR / f"val_sq2_{cfg.name}.parquet",
        CHECKPOINTS_DIR / f"sq2_{cfg.name}.pt",
    )


def get_unfiltered_splits(horizon: str) -> SplitDatasets:
    """Cached unfiltered standard splits for one horizon (retention denominator, C0 pool)."""
    if horizon not in _UNFILTERED_SPLITS:
        _UNFILTERED_SPLITS[horizon] = build_train_val_test_splits(
            split_seed=SPLIT_SEED, horizon=horizon
        )
    return _UNFILTERED_SPLITS[horizon]


def append_manifest_rows(rows: list[dict[str, object]]) -> None:
    """Append run-metadata rows to outputs/tables/sq2_run_manifest.csv (write header if new)."""
    new = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if MANIFEST_PATH.exists():
        new = pd.concat([pd.read_csv(MANIFEST_PATH), new], ignore_index=True)
    new.to_csv(MANIFEST_PATH, index=False)


def load_manifest() -> pd.DataFrame:
    """Read the manifest and keep only the LAST row per exp_id."""
    if not MANIFEST_PATH.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    manifest = pd.read_csv(MANIFEST_PATH)
    return manifest.groupby("exp_id", sort=False).tail(1).reset_index(drop=True)


def _parquet_len(path: Path) -> int:
    return len(pd.read_parquet(path, columns=["y_true"]))


def _ckpt_summary(path: Path) -> tuple[object, float]:
    """Read (best_epoch, best_val_loss) from a checkpoint; blanks if unavailable."""
    if not path.exists():
        return "", float("nan")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    return ckpt.get("best_epoch", ""), float(ckpt.get("best_val_loss", float("nan")))


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# C0: reuse delivered baseline test predictions (+ optional val inference)
# ---------------------------------------------------------------------------


def ensure_c0_ready(force: bool, eval_batch_size: int, device: torch.device) -> None:
    """Materialize C0: delivered test parquet, optional val inference, manifest row.

    The OOS test predictions ALWAYS come from the delivered
    `outputs/predictions/baseline_seed42.parquet`; only validation predictions may be
    rebuilt by loading `outputs/checkpoints/baseline_seed42.pt` (inference only).
    """
    cfg = SQ2_EXPERIMENTS["C0"]
    _, val_path, _ = exp_paths(cfg)
    manifest = load_manifest()
    has_row = cfg.exp_id in set(manifest["exp_id"])

    if (not val_path.exists() or force) and BASELINE_CKPT.exists():
        t0 = time.perf_counter()
        ckpt = torch.load(BASELINE_CKPT, map_location=device, weights_only=True)
        model = CNN20D()
        model.load_state_dict(ckpt["model_state_dict"])
        val_df = predict_dataset(
            model,
            get_unfiltered_splits("20d").val,
            batch_size=eval_batch_size,
            device=device,
        )
        save_predictions(val_df, val_path)
        print(
            f"[C0] Rebuilt validation predictions from {BASELINE_CKPT.name} in "
            f"{time.perf_counter() - t0:.1f}s -> {val_path.name} "
            f"(test set stays the delivered {BASELINE_TEST_PARQUET.name})",
            flush=True,
        )
    elif not val_path.exists():
        print(
            f"[C0] WARNING: {val_path.name} missing and {BASELINE_CKPT.name} not found; "
            "skipping val rebuild.",
            flush=True,
        )

    if not has_row:
        if not BASELINE_TEST_PARQUET.exists():
            msg = f"C0 requires the delivered {BASELINE_TEST_PARQUET}; run exp_baseline.py first."
            raise FileNotFoundError(msg)
        base = get_unfiltered_splits("20d")
        best_epoch, best_val_loss = _ckpt_summary(BASELINE_CKPT)
        append_manifest_rows(
            [
                {
                    "exp_id": cfg.exp_id,
                    "name": cfg.name,
                    "horizon": cfg.horizon,
                    "train_n": len(base.train),
                    "retention_pct": 100.0,
                    "train_pos_rate": float(base.train.metadata["Retx_20d_label"].mean()),
                    "val_n": _parquet_len(val_path) if val_path.exists() else len(base.val),
                    "test_n": _parquet_len(BASELINE_TEST_PARQUET),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "elapsed_sec": "",
                    "timestamp": _now(),
                }
            ]
        )
        print(f"[C0] Appended C0 manifest row (best_epoch={best_epoch} from checkpoint).", flush=True)


def ensure_manifest_row(cfg: ExpConfig) -> None:
    """Rebuild a manifest row from local metadata when reusing saved predictions."""
    if cfg.exp_id == "C0":
        ensure_c0_ready(force=False, eval_batch_size=1024, device=get_device())
        return
    if cfg.exp_id in set(load_manifest()["exp_id"]):
        return
    test_path, val_path, ckpt_path = exp_paths(cfg)
    if not test_path.exists():
        return
    base = get_unfiltered_splits(cfg.horizon)
    splits = (
        build_train_val_test_splits(
            split_seed=SPLIT_SEED,
            horizon=cfg.horizon,
            train_min_abs_retx=cfg.min_abs_retx,
            train_min_abs_tstat=cfg.min_abs_tstat,
        )
        if (cfg.min_abs_retx > 0.0 or cfg.min_abs_tstat > 0.0)
        else base
    )
    best_epoch, best_val_loss = _ckpt_summary(ckpt_path)
    append_manifest_rows(
        [
            {
                "exp_id": cfg.exp_id,
                "name": cfg.name,
                "horizon": cfg.horizon,
                "train_n": len(splits.train),
                "retention_pct": 100.0 * len(splits.train) / len(base.train),
                "train_pos_rate": float(splits.train.metadata[f"Retx_{cfg.horizon}_label"].mean()),
                "val_n": _parquet_len(val_path) if val_path.exists() else len(base.val),
                "test_n": _parquet_len(test_path),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "elapsed_sec": "",
                "timestamp": _now(),
            }
        ]
    )
    print(f"[{cfg.exp_id}] Rebuilt missing manifest row from saved artifacts.", flush=True)


def _maybe_rebuild_val(cfg: ExpConfig, eval_batch_size: int, device: torch.device) -> None:
    """If a reused experiment lacks its val parquet but has a checkpoint, infer it."""
    _, val_path, ckpt_path = exp_paths(cfg)
    if val_path.exists() or not ckpt_path.exists():
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = CNN20D()
    model.load_state_dict(ckpt["model_state_dict"])
    val_df = predict_dataset(
        model,
        get_unfiltered_splits(cfg.horizon).val,
        batch_size=eval_batch_size,
        device=device,
    )
    save_predictions(val_df, val_path)
    print(f"[{cfg.exp_id}] Rebuilt validation predictions from checkpoint -> {val_path.name}")


# ---------------------------------------------------------------------------
# Training experiments C1–C7
# ---------------------------------------------------------------------------


def run_training_experiment(cfg: ExpConfig, args: argparse.Namespace, device: torch.device) -> None:
    """Train, predict, save, evaluate, and log one C1–C7 experiment."""
    test_path, val_path, ckpt_path = exp_paths(cfg)
    if test_path.exists() and not args.force:
        print(
            f"\n--- {cfg.exp_id} ({cfg.name}): {test_path.name} already exists — reusing "
            "saved predictions (pass --force to retrain) ---",
            flush=True,
        )
        ensure_manifest_row(cfg)
        _maybe_rebuild_val(cfg, args.eval_batch_size, device)
        return

    t0 = time.perf_counter()
    splits = build_train_val_test_splits(
        split_seed=SPLIT_SEED,
        horizon=cfg.horizon,
        train_min_abs_retx=cfg.min_abs_retx,
        train_min_abs_tstat=cfg.min_abs_tstat,
    )
    base = get_unfiltered_splits(cfg.horizon)
    full_train_n = len(base.train)
    train_n = len(splits.train)
    label_col = f"Retx_{cfg.horizon}_label"
    pos_rate = float(splits.train.metadata[label_col].mean())

    print(
        f"\n--- {cfg.exp_id} ({cfg.name}) | horizon={cfg.horizon} | rule: {cfg.filter_rule} ---\n"
        f"  Train={train_n:,} ({100.0 * train_n / full_train_n:.2f}% of {full_train_n:,} "
        f"same-horizon pool) | pos_rate={pos_rate:.4f} | Val={len(splits.val):,} | "
        f"Test={len(splits.test):,}",
        flush=True,
    )

    seed_everything(TRAIN_SEED)  # BEFORE model construction (execution plan §2)
    model = CNN20D()
    result = train_model(
        model,
        splits.train,
        splits.val,
        seed=TRAIN_SEED,
        lr=args.lr,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        label_smoothing=cfg.label_smoothing,
        checkpoint_path=ckpt_path,
        device=device,
    )

    t_inf = time.perf_counter()
    val_df = predict_dataset(
        model, splits.val, batch_size=args.eval_batch_size, device=device
    )
    test_df = predict_dataset(
        model, splits.test, batch_size=args.eval_batch_size, device=device
    )
    save_predictions(val_df, val_path)
    save_predictions(test_df, test_path)
    elapsed = time.perf_counter() - t0

    metrics, _ = evaluate_sq2(test_df, cfg.horizon, label=cfg.exp_id)
    print(
        f"  Saved val/test predictions in {time.perf_counter() - t_inf:.1f}s | "
        f"total wall time {elapsed:.1f}s (best epoch {result.best_epoch}, "
        f"best val_loss {result.best_val_loss:.4f})",
        flush=True,
    )
    print(
        f"  [{cfg.exp_id} OOS 2000-2019] Acc={metrics['accuracy_pct']:.2f}% | "
        f"CE={metrics['ce_loss']:.4f} | AUC={metrics['auc']:.4f} | ECE={metrics['ece']:.4f} | "
        f"LS Sharpe={metrics['ls_sharpe']:.2f} (t={metrics['ls_tstat']:.2f}) | "
        f"meanLS/period={metrics['ls_mean_ret_pct']:.3f}%",
        flush=True,
    )
    append_manifest_rows(
        [
            {
                "exp_id": cfg.exp_id,
                "name": cfg.name,
                "horizon": cfg.horizon,
                "train_n": train_n,
                "retention_pct": 100.0 * train_n / full_train_n,
                "train_pos_rate": pos_rate,
                "val_n": len(splits.val),
                "test_n": len(test_df),
                "best_epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
                "elapsed_sec": elapsed,
                "timestamp": _now(),
            }
        ]
    )


# ---------------------------------------------------------------------------
# Aggregate outputs (tables 3–7 + figure), usable in both run & --summarize modes
# ---------------------------------------------------------------------------


def build_main_table(
    available: dict[str, pd.DataFrame], manifest: pd.DataFrame
) -> pd.DataFrame:
    """Table 3 — one row per available experiment on its full OOS test set."""
    rows: list[dict[str, object]] = []
    for exp_id, test_df in available.items():
        cfg = SQ2_EXPERIMENTS[exp_id]
        metrics, _ = evaluate_sq2(test_df, cfg.horizon, label=exp_id)
        mrow = manifest.loc[manifest["exp_id"] == exp_id]
        mrow = mrow.iloc[0] if len(mrow) else None
        rows.append(
            {
                "exp_id": exp_id,
                "name": cfg.name,
                "horizon": cfg.horizon,
                "filter_rule": cfg.filter_rule,
                "train_n": int(mrow["train_n"]) if mrow is not None else "",
                "retention_pct": float(mrow["retention_pct"]) if mrow is not None else "",
                "train_pos_rate": float(mrow["train_pos_rate"]) if mrow is not None else "",
                "val_n": int(mrow["val_n"]) if mrow is not None else "",
                "test_n": len(test_df),
                "best_epoch": (mrow["best_epoch"] if mrow is not None else ""),
                "accuracy_pct": metrics["accuracy_pct"],
                "ce_loss": metrics["ce_loss"],
                "auc": metrics["auc"],
                "ece": metrics["ece"],
                "ls_sharpe": metrics["ls_sharpe"],
                "ls_tstat": metrics["ls_tstat"],
                "ls_mean_ret_pct": metrics["ls_mean_ret_pct"],
                "eval_scope": "full OOS 2000-2019",
            }
        )
    return pd.DataFrame(rows)


def build_common_sample_table(available: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """Table 4 — C6/C0/C7 re-evaluated on the common (year, row_idx) intersection."""
    required = ("C0", "C6", "C7")
    if not all(exp_id in available for exp_id in required):
        print(
            f"  [common-sample] skipped: need C0+C6+C7 test predictions, "
            f"missing {[e for e in required if e not in available]}",
            flush=True,
        )
        return None
    keys = ["year", "row_idx"]
    common = available["C0"][keys].drop_duplicates()
    for exp_id in ("C6", "C7"):
        common = common.merge(available[exp_id][keys].drop_duplicates(), on=keys, how="inner")
    n_common = len(common)
    assert n_common == EXPECTED_COMMON_SAMPLE_N, (
        f"common-sample size {n_common:,} != expected {EXPECTED_COMMON_SAMPLE_N:,}"
    )

    rows: list[dict[str, object]] = []
    for exp_id in ("C6", "C0", "C7"):
        cfg = SQ2_EXPERIMENTS[exp_id]
        sub = available[exp_id].merge(common, on=keys, how="inner")
        assert len(sub) == n_common, f"{exp_id}: joined {len(sub):,} != {n_common:,}"
        metrics, _ = evaluate_sq2(sub, cfg.horizon, label=f"{exp_id}-common")
        rows.append(
            {
                "exp_id": exp_id,
                "horizon": cfg.horizon,
                "n_common": n_common,
                "up_rate_pct": float(sub["y_true"].mean() * 100.0),
                "accuracy_pct": metrics["accuracy_pct"],
                "ce_loss": metrics["ce_loss"],
                "auc": metrics["auc"],
                "ece": metrics["ece"],
                "ls_sharpe": metrics["ls_sharpe"],
                "ls_tstat": metrics["ls_tstat"],
                "ls_mean_ret_pct": metrics["ls_mean_ret_pct"],
                "eval_scope": "common sample (year,row_idx) of C6/C0/C7",
            }
        )
    return pd.DataFrame(rows)


def build_cap_retention_table() -> pd.DataFrame:
    """Table 5 — C1–C4 retention by C0 training-pool MarketCap quintile (Q1=small)."""
    pool = get_unfiltered_splits("20d").train.metadata
    meta = pool[["MarketCap", "Retx_20d", "Retx_20d_tstat"]].copy()
    # rank(method="first") guarantees unique qcut edges on float32 market caps
    meta["cap_quintile"] = pd.qcut(
        meta["MarketCap"].rank(method="first"), q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]
    )
    grouped = meta.groupby("cap_quintile", observed=True)
    out = pd.DataFrame(
        {
            "cap_quintile": ["Q1 (small)", "Q2", "Q3", "Q4", "Q5 (large)"],
            "n_total": grouped.size().to_numpy(),
            "mean_market_cap": grouped["MarketCap"].mean().to_numpy(),
        }
    )
    rules = {
        "c1": meta["Retx_20d"].abs() >= 0.01,
        "c2": meta["Retx_20d"].abs() >= 0.03,
        "c3": meta["Retx_20d"].abs() >= 0.05,
        "c4": meta["Retx_20d_tstat"].abs() >= 0.5,
    }
    for key, keep_mask in rules.items():
        rates = (
            meta.assign(keep=keep_mask.to_numpy())
            .groupby("cap_quintile", observed=True)["keep"]
            .mean()
            * 100.0
        )
        out[f"retention_{key}_pct"] = rates.to_numpy()
    return out


def moving_block_bootstrap(
    diff: np.ndarray, *, block: int = 3, n_boot: int = 2000, seed: int = 42
) -> dict[str, np.ndarray]:
    """Moving-block bootstrap of monthly LS-return differences (Sharpe & mean)."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    n_blocks = int(np.ceil(n / block))
    offsets = np.arange(block)
    sharpe = np.empty(n_boot)
    mean_ret = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        idx = (starts[:, None] + offsets[None, :]).ravel()[:n]
        sample = diff[idx]
        mu = float(sample.mean())
        sd = float(sample.std(ddof=1))
        mean_ret[b] = mu
        sharpe[b] = float(np.sqrt(12.0) * mu / sd) if sd > 0.0 else 0.0
    return {
        "sharpe_q": np.percentile(sharpe, [2.5, 50.0, 97.5]),
        "mean_q": np.percentile(mean_ret, [2.5, 50.0, 97.5]),
    }


def build_bootstrap_table(
    monthly_cache: dict[str, pd.DataFrame]
) -> pd.DataFrame | None:
    """Table 6 — paired moving-block bootstrap of C2−C0 and C5−C0 monthly LS returns."""
    rows: list[dict[str, object]] = []
    for exp_id in ("C2", "C5"):
        if exp_id not in monthly_cache or "C0" not in monthly_cache:
            print(f"  [paired-bootstrap] skipped {exp_id}−C0: missing predictions", flush=True)
            continue
        aligned = pd.concat(
            [monthly_cache[exp_id]["ls_ret"].rename("treat"), monthly_cache["C0"]["ls_ret"].rename("base")],
            axis=1,
            join="inner",
        ).sort_index()
        diff = (aligned["treat"] - aligned["base"]).to_numpy(dtype=np.float64)
        mu, sd = float(diff.mean()), float(diff.std(ddof=1))
        boot = moving_block_bootstrap(diff, block=3, n_boot=2000, seed=42)
        rows.append(
            {
                "comparison": f"{exp_id}-C0",
                "n_months": len(diff),
                "block_len": 3,
                "n_boot": 2000,
                "obs_sharpe_diff": float(np.sqrt(12.0) * mu / sd) if sd > 0 else 0.0,
                "sharpe_diff_q2_5": boot["sharpe_q"][0],
                "sharpe_diff_q50": boot["sharpe_q"][1],
                "sharpe_diff_q97_5": boot["sharpe_q"][2],
                "obs_mean_ret_diff_pct": mu * 100.0,
                "mean_ret_diff_q2_5_pct": boot["mean_q"][0] * 100.0,
                "mean_ret_diff_q50_pct": boot["mean_q"][1] * 100.0,
                "mean_ret_diff_q97_5_pct": boot["mean_q"][2] * 100.0,
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def plot_figure2(main: pd.DataFrame) -> None:
    """Figure 2 — (a) margin-filtering trade-off; (b) horizon & label smoothing."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    palette = sns.color_palette("colorblind")
    idx = main.set_index("exp_id")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1), dpi=300)

    # Panel (a): margin filtering trade-off (tau = 0/1/3/5% + C4 as separate hollow point)
    tau_points = [("C0", 0.0, "0%"), ("C1", 1.0, "1%"), ("C2", 3.0, "3%"), ("C3", 5.0, "5%")]
    pts = [(tau, idx.loc[exp_id], lab) for exp_id, tau, lab in tau_points if exp_id in idx.index]
    x_tau = [p[0] for p in pts]
    has_c4 = "C4" in idx.index
    x_c4 = 6.4

    ax_ret = ax1.twinx()
    ax_ret.bar(
        x_tau,
        [float(r["retention_pct"]) for _, r, _ in pts],
        width=0.85,
        alpha=0.22,
        color="0.55",
        label="Train retention (right)",
        zorder=1,
    )
    if has_c4:
        ax_ret.bar(
            [x_c4], [float(idx.loc["C4", "retention_pct"])],
            width=0.85, alpha=0.22, color="0.55", zorder=1,
        )
    ax_ret.set_ylabel("Train Retention (%)", color="0.35", fontsize=9.5)
    ax_ret.set_ylim(0, 108)
    ax_ret.tick_params(axis="y", colors="0.35", labelsize=8.5)
    ax_ret.grid(False)

    ax1.set_zorder(ax_ret.get_zorder() + 1)
    ax1.patch.set_visible(False)
    line_auc, = ax1.plot(
        x_tau, [float(r["auc"]) for _, r, _ in pts], marker="o", ms=5.5, lw=1.8,
        color=palette[0], label="AUC (left)",
    )
    line_sr, = ax1.plot(
        x_tau, [float(r["ls_sharpe"]) for _, r, _ in pts], marker="s", ms=5.5, lw=1.8,
        color=palette[3], label="LS Sharpe (left)",
    )
    if has_c4:
        r4 = idx.loc["C4"]
        ax1.plot([x_c4], [float(r4["auc"])], marker="o", ms=9, lw=0,
                 mfc="none", mec=palette[0], mew=1.8)
        ax1.plot([x_c4], [float(r4["ls_sharpe"])], marker="s", ms=9, lw=0,
                 mfc="none", mec=palette[3], mew=1.8)
        ax1.axvline(5.7, linestyle=":", color="0.6", lw=1.0)
        ax1.annotate(
            f"C4: |t-stat|≥0.5\nretention {float(r4['retention_pct']):.1f}%\n"
            f"AUC {float(r4['auc']):.4f}, Sharpe {float(r4['ls_sharpe']):.2f}",
            xy=(x_c4, float(r4["auc"])),
            xytext=(4.35, float(r4["ls_sharpe"])),
            fontsize=7.8,
            color="0.25",
            arrowprops={"arrowstyle": "-", "color": "0.5", "lw": 0.8},
        )
    xticks = x_tau + ([x_c4] if has_c4 else [])
    xlabels = [p[2] for p in pts] + (["|t|≥0.5"] if has_c4 else [])
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabels)
    ax1.set_xlim(-0.6, 7.6 if has_c4 else 5.6)
    ax1.set_xlabel("Margin filter threshold $\\tau$ on $|Retx_{20d}|$")
    ax1.set_ylabel("AUC / LS Sharpe")
    ax1.set_title("(a) Margin Filtering Trade-off", fontweight="bold")
    handles = [line_auc, line_sr] + ax_ret.get_legend_handles_labels()[0]
    ax1.legend(handles=handles, loc="lower left", frameon=True, fontsize=8.3)

    # Panel (b): horizon & label smoothing
    hz_points = [("C6", "5d", 5), ("C0", "20d", 20), ("C7", "60d", 60)]
    pts_b = [(x, idx.loc[exp_id], lab) for exp_id, lab, x in hz_points if exp_id in idx.index]
    x_hz = [p[0] for p in pts_b]
    line_acc, = ax2.plot(
        x_hz, [float(r["accuracy_pct"]) for _, r, _ in pts_b], marker="o", ms=5.5,
        lw=1.8, color=palette[0], label="Accuracy (left)",
    )
    ax_ret2 = ax2.twinx()
    line_ls, = ax_ret2.plot(
        x_hz, [float(r["ls_mean_ret_pct"]) for _, r, _ in pts_b], marker="^", ms=6,
        lw=1.8, color=palette[1], label="Mean LS return /period % (right)",
    )
    ax_ret2.set_ylabel("Mean Long-Short Return per Period (%)", color="0.3", fontsize=9.5)
    ax_ret2.tick_params(axis="y", colors="0.3", labelsize=8.5)
    ax_ret2.grid(False)
    ax2.set_xticks([5, 20, 60])
    ax2.set_xticklabels(["5d", "20d", "60d"])
    ax2.set_xlabel("Prediction Horizon (trading days; signals monthly)")
    ax2.set_ylabel("OOS Directional Accuracy (%)")
    ax2.set_title("(b) Horizon & Label Smoothing", fontweight="bold")
    ax2.legend(handles=[line_acc, line_ls], loc="upper left", frameon=True, fontsize=8.3)
    if "C5" in idx.index and "C0" in idx.index:
        ax2.text(
            0.97,
            0.04,
            f"Label smoothing: CE {float(idx.loc['C0', 'ce_loss']):.4f}"
            f"$\\rightarrow${float(idx.loc['C5', 'ce_loss']):.4f}, "
            f"ECE {float(idx.loc['C0', 'ece']):.4f}"
            f"$\\rightarrow${float(idx.loc['C5', 'ece']):.4f}",
            transform=ax2.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.3,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "0.7", "alpha": 0.9},
        )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_file = FIGURES_DIR / f"fig2_sq2_noise_horizon.{ext}"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Saved {out_file}")
    plt.close(fig)


def print_key_comparisons(
    main: pd.DataFrame, common: pd.DataFrame | None
) -> None:
    """Print the headline SQ2 differences required by the execution plan §3.5."""
    idx = main.set_index("exp_id")
    print("\n=== Key SQ2 comparisons ===", flush=True)
    if {"C0", "C2"} <= set(idx.index):
        print(
            f"  C2−C0 (margin τ=3%): ΔAUC = "
            f"{float(idx.loc['C2', 'auc']) - float(idx.loc['C0', 'auc']):+.4f} | "
            f"ΔLS Sharpe = "
            f"{float(idx.loc['C2', 'ls_sharpe']) - float(idx.loc['C0', 'ls_sharpe']):+.2f}",
            flush=True,
        )
    else:
        print("  C2−C0: missing C2 and/or C0 predictions", flush=True)
    if {"C0", "C5"} <= set(idx.index):
        print(
            f"  C5−C0 (label smoothing 0.1): ΔCE = "
            f"{float(idx.loc['C5', 'ce_loss']) - float(idx.loc['C0', 'ce_loss']):+.4f} | "
            f"ΔECE = {float(idx.loc['C5', 'ece']) - float(idx.loc['C0', 'ece']):+.4f}",
            flush=True,
        )
    else:
        print("  C5−C0: missing C5 and/or C0 predictions", flush=True)
    if common is not None:
        rank = common.sort_values("accuracy_pct", ascending=False)
        order = " > ".join(f"{r.exp_id}/{r.horizon} ({r.accuracy_pct:.2f}%)" for r in rank.itertuples())
        print(f"  Common-sample ({rank['n_common'].iloc[0]:,} imgs) accuracy ranking: {order}", flush=True)
    else:
        print("  Common-sample ranking: unavailable (needs C0+C6+C7)", flush=True)


def build_outputs(device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Rebuild tables 3–7 and Figure 2 from saved predictions + manifest (no training)."""
    print("\n=== Building SQ2 tables & figure from saved predictions ===", flush=True)
    ensure_c0_ready(force=False, eval_batch_size=1024, device=device)

    available: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for exp_id, cfg in SQ2_EXPERIMENTS.items():
        test_path = BASELINE_TEST_PARQUET if exp_id == "C0" else exp_paths(cfg)[0]
        if test_path.exists():
            available[exp_id] = pd.read_parquet(test_path)
            ensure_manifest_row(cfg)
        else:
            missing.append(exp_id)
    if missing:
        print(f"  Missing test predictions for: {missing} — tables/figure will be partial.", flush=True)

    manifest = load_manifest()
    monthly_cache: dict[str, pd.DataFrame] = {}
    for exp_id, test_df in available.items():
        _, monthly = evaluate_sq2(test_df, SQ2_EXPERIMENTS[exp_id].horizon, label=exp_id)
        monthly_cache[exp_id] = monthly

    main = build_main_table(available, manifest)
    main_path = TABLES_DIR / "sq2_noise_and_horizon.csv"
    main.to_csv(main_path, index=False)
    print(f"\nSaved main table to {main_path}:")
    print(main.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    common = build_common_sample_table(available)
    if common is not None:
        common_path = TABLES_DIR / "sq2_common_sample_horizon.csv"
        common.to_csv(common_path, index=False)
        print(f"\nSaved common-sample table to {common_path}:")
        print(common.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    cap_table = build_cap_retention_table()
    cap_path = TABLES_DIR / "sq2_filter_retention_by_cap.csv"
    cap_table.to_csv(cap_path, index=False)
    print(f"\nSaved filter-retention-by-cap table to {cap_path}:")
    print(cap_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    boot = build_bootstrap_table(monthly_cache)
    if boot is not None:
        boot_path = TABLES_DIR / "sq2_paired_bootstrap.csv"
        boot.to_csv(boot_path, index=False)
        print(f"\nSaved paired moving-block bootstrap table to {boot_path}:")
        print(boot.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if not main.empty:
        plot_figure2(main)
    print_key_comparisons(main, common)
    return main, common


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    device = get_device(None if args.device == "auto" else args.device)

    if args.summarize:
        print(f"=== SQ2 Summarize (no training) | Device: {device} ===", flush=True)
        build_outputs(device)
        return

    exp_ids = resolve_exps(args.exp)
    print(
        f"=== SQ2 Label Noise & Horizon Matrix | Device: {device} | "
        f"Experiments: {exp_ids} ===",
        flush=True,
    )
    for exp_id in exp_ids:
        if exp_id == "C0":
            ensure_c0_ready(force=args.force, eval_batch_size=args.eval_batch_size, device=device)
        else:
            run_training_experiment(SQ2_EXPERIMENTS[exp_id], args, device)
    build_outputs(device)


if __name__ == "__main__":
    main()
