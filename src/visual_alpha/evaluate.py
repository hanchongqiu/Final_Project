"""Standardized 4-metric evaluator, portfolio backtester, and calibration helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score

REQUIRED_PREDICTION_COLUMNS = (
    "year",
    "row_idx",
    "Date",
    "StockID",
    "y_true",
    "logit_down",
    "logit_up",
    "prob_up",
    "Retx_20d",
    "Ret_20d",
    "MarketCap",
)

PERIODS_PER_YEAR: dict[str, float] = {
    "Ret_20d": 12.0,
    "Ret_5d": 12.0,   # Monthly rebalance snapshotted once per month in monthly_20d
    "Ret_60d": 12.0,  # Monthly overlapping rebalance (report single-period & annualized)
}


def validate_prediction_frame(pred_df: pd.DataFrame) -> None:
    """Verify prediction DataFrame adheres to DATA_AND_CONTRACTS.md Section 5.1."""
    missing = [col for col in REQUIRED_PREDICTION_COLUMNS if col not in pred_df.columns]
    if missing:
        msg = f"Prediction DataFrame missing required contract columns: {missing}"
        raise ValueError(msg)
    if pred_df["prob_up"].isna().any():
        msg = "Prediction DataFrame contains NaN values in 'prob_up'"
        raise ValueError(msg)


def save_predictions(pred_df: pd.DataFrame, path: Path | str) -> Path:
    """Validate and save standardized OOS prediction DataFrame to `.parquet`."""
    validate_prediction_frame(pred_df)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(out_path, index=False)
    return out_path


def compute_ece(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    *,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    """Compute 10-bin equal-width Expected Calibration Error (ECE) on positive-class probs

    Also computes standard confidence-based ECE on `max(p, 1-p)` vs correctness, returning
    both the scalar confidence ECE and a detailed bin table for reliability diagrams.
    """
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(prob_up, dtype=np.float64)
    n = len(y)
    if n == 0:
        msg = "Cannot compute ECE on empty arrays"
        raise ValueError(msg)

    # Confidence-based ECE (Guo et al., 2017): predicted class confidence vs accuracy
    y_hat = (p >= 0.5).astype(np.float64)
    correct = (y_hat == y).astype(np.float64)
    conf = np.maximum(p, 1.0 - p)

    conf_edges = np.linspace(0.5, 1.0, n_bins + 1)
    conf_bin_ids = np.clip(np.digitize(conf, conf_edges[1:-1], right=False), 0, n_bins - 1)
    ece_val = 0.0
    for b in range(n_bins):
        mask = conf_bin_ids == b
        cnt = int(mask.sum())
        if cnt > 0:
            acc_b = float(correct[mask].mean())
            conf_b = float(conf[mask].mean())
            ece_val += (cnt / n) * abs(acc_b - conf_b)

    # Positive-class probability reliability bins across [0, 1] (plus finer view in [0.35, 0.65])
    prob_edges = np.linspace(0.0, 1.0, n_bins + 1)
    prob_bin_ids = np.clip(np.digitize(p, prob_edges[1:-1], right=False), 0, n_bins - 1)
    rows: list[dict[str, float | int]] = []
    for b in range(n_bins):
        mask = prob_bin_ids == b
        cnt = int(mask.sum())
        rows.append(
            {
                "bin": b,
                "bin_low": float(prob_edges[b]),
                "bin_high": float(prob_edges[b + 1]),
                "count": cnt,
                "weight": cnt / n,
                "mean_prob": float(p[mask].mean()) if cnt > 0 else float("nan"),
                "emp_freq": float(y[mask].mean()) if cnt > 0 else float("nan"),
            }
        )
    return float(ece_val), pd.DataFrame(rows)


def compute_decile_portfolio(
    pred_df: pd.DataFrame,
    *,
    ret_col: str = "Ret_20d",
    fallback_ret_col: str = "Retx_20d",
    n_quantiles: int = 10,
    annualization: float = 12.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Compute monthly cross-sectional equal-weighted Long-Short Decile portfolio metrics."""
    ret = pred_df[ret_col].to_numpy(dtype=np.float64).copy()
    if fallback_ret_col in pred_df.columns:
        nan_mask = np.isnan(ret)
        if nan_mask.any():
            ret[nan_mask] = pred_df[fallback_ret_col].to_numpy(dtype=np.float64)[nan_mask]

    work = pd.DataFrame(
        {
            "Date": pred_df["Date"].to_numpy(),
            "prob_up": pred_df["prob_up"].to_numpy(dtype=np.float64),
            "ret": ret,
        }
    )
    work = work.dropna(subset=["prob_up", "ret"])

    # Rank within each Date using method="first" to break exact ties cleanly into 10 equal deciles
    ranks = work.groupby("Date", sort=False)["prob_up"].rank(method="first")
    counts = work.groupby("Date", sort=False)["prob_up"].transform("count")
    work = work.loc[counts >= n_quantiles].copy()
    ranks = ranks.loc[work.index]
    counts = counts.loc[work.index]

    work["decile"] = np.floor((ranks - 1.0) * n_quantiles / counts).astype(np.int32) + 1
    work["decile"] = work["decile"].clip(1, n_quantiles)

    monthly = (
        work.groupby(["Date", "decile"], sort=True)["ret"]
        .mean()
        .unstack("decile")
        .dropna(subset=[1, n_quantiles])
        .sort_index()
    )
    monthly["long_ret"] = monthly[n_quantiles]
    monthly["short_ret"] = monthly[1]
    monthly["ls_ret"] = monthly[n_quantiles] - monthly[1]

    ls = monthly["ls_ret"].to_numpy(dtype=np.float64)
    n_months = len(ls)
    mean_mo = float(np.mean(ls))
    std_mo = float(np.std(ls, ddof=1)) if n_months > 1 else float("nan")
    sharpe_ann = float(np.sqrt(annualization) * mean_mo / std_mo) if std_mo > 0 else 0.0
    t_stat = float(mean_mo / (std_mo / np.sqrt(n_months))) if std_mo > 0 else 0.0

    summary = {
        "n_months": float(n_months),
        "long_ann_ret": float(mean_mo * 0.0 + np.mean(monthly["long_ret"]) * annualization),
        "short_ann_ret": float(np.mean(monthly["short_ret"]) * annualization),
        "ls_ann_ret": float(mean_mo * annualization),
        "ls_ann_vol": float(std_mo * np.sqrt(annualization)),
        "ls_sharpe": sharpe_ann,
        "ls_tstat": t_stat,
    }
    return summary, monthly


def evaluate_predictions(
    pred_df: pd.DataFrame,
    *,
    ret_col: str = "Ret_20d",
    annualization: float = 12.0,
) -> dict[str, float]:
    """Unified referee function computing all 4 standard metric families."""
    validate_prediction_frame(pred_df)
    y_true = pred_df["y_true"].to_numpy(dtype=np.int64)
    prob_up = np.clip(pred_df["prob_up"].to_numpy(dtype=np.float64), 1e-7, 1.0 - 1e-7)
    y_pred = (prob_up >= 0.5).astype(np.int64)

    acc_pct = float((y_pred == y_true).mean() * 100.0)
    ce_loss = float(log_loss(y_true, prob_up, labels=[0, 1]))
    auc = float(roc_auc_score(y_true, prob_up))
    ece, _ = compute_ece(y_true, prob_up, n_bins=10)
    port_metrics, _ = compute_decile_portfolio(
        pred_df, ret_col=ret_col, annualization=annualization
    )

    return {
        "n_samples": float(len(pred_df)),
        "accuracy_pct": acc_pct,
        "ce_loss": ce_loss,
        "auc": auc,
        "ece": ece,
        "ls_sharpe": port_metrics["ls_sharpe"],
        "ls_tstat": port_metrics["ls_tstat"],
        "ls_ann_ret_pct": port_metrics["ls_ann_ret"] * 100.0,
        "long_ann_ret_pct": port_metrics["long_ann_ret"] * 100.0,
        "short_ann_ret_pct": port_metrics["short_ann_ret"] * 100.0,
    }


def fit_temperature_scaling(
    logit_down: np.ndarray,
    logit_up: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Fit post-hoc scalar temperature T > 0 minimizing validation Cross-Entropy (NLL)."""
    logits = np.stack([logit_down, logit_up], axis=1).astype(np.float64)
    y = np.asarray(y_true, dtype=np.int64)
    diff = logits[:, 1] - logits[:, 0]

    def nll_from_log_temp(params: np.ndarray) -> tuple[float, np.ndarray]:
        log_t = float(params[0])
        temp = np.exp(log_t)
        scaled = diff / temp
        # Numerically stable binary cross-entropy on logit difference z1 - z0
        prob_1 = 1.0 / (1.0 + np.exp(-np.clip(scaled, -40.0, 40.0)))
        loss = float(np.mean(np.logaddexp(0.0, scaled) - y * scaled))
        # d(NLL)/d(log_t) = mean((y - prob_1) * scaled)
        grad = np.array([float(np.mean((y - prob_1) * scaled))], dtype=np.float64)
        return loss, grad

    res = minimize(
        nll_from_log_temp,
        x0=np.array([0.0], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-3.0, 3.0)],
    )
    return float(np.exp(res.x[0]))


def apply_temperature_to_logits(
    logit_down: np.ndarray,
    logit_up: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Compute calibrated P(y=1 | x) = Softmax(logits / T)[:, 1]."""
    logits = torch.from_numpy(np.stack([logit_down, logit_up], axis=1).astype(np.float32))
    probs = torch.softmax(logits / float(temperature), dim=1)[:, 1]
    return probs.numpy()


def compute_selective_prediction_curve(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    *,
    coverages: tuple[float, ...] = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05),
) -> pd.DataFrame:
    """Evaluate conditional accuracy and win rate across top-confidence coverage slices."""
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(prob_up, dtype=np.float64)
    margin = np.abs(p - 0.5)
    order = np.argsort(-margin)  # Highest confidence first
    y_sorted = y[order]
    p_sorted = p[order]
    pred_sorted = (p_sorted >= 0.5).astype(np.int64)
    correct_sorted = (pred_sorted == y_sorted).astype(np.float64)

    n = len(y)
    rows: list[dict[str, float]] = []
    for cov in coverages:
        k = max(1, round(n * cov))
        rows.append(
            {
                "coverage": float(cov),
                "coverage_pct": float(cov * 100.0),
                "n_selected": float(k),
                "min_confidence": float(0.5 + margin[order[k - 1]]),
                "accuracy_pct": float(correct_sorted[:k].mean() * 100.0),
                "auc": float(roc_auc_score(y_sorted[:k], p_sorted[:k]))
                if len(np.unique(y_sorted[:k])) == 2
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)
