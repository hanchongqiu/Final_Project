"""Member A — SQ3 Probability Calibration, Deep Ensembling & Selective Prediction.

Consumes the saved validation & OOS predictions from `exp_baseline.py` (zero extra neural net
training required!) to evaluate:
  M1: Uncalibrated Single CNN (seed=42)
  M2: Temperature-Scaled Single CNN (scalar T fit strictly on 1993-1999 validation set)
  M3: 5-Seed Deep Ensemble (equal-weighted probability average across seeds 42..46)
  M4: Temp-Scaled 5-Seed Ensemble (scalar T fit on validation mean logits)

Outputs:
  - `outputs/predictions/sq3_temp_scaled_seed42.parquet`
  - `outputs/tables/sq3_calibration_summary.csv`
  - `outputs/tables/sq3_selective_prediction.csv`
  - `outputs/figures/fig3_sq3_calibration.pdf` & `.png` (300 DPI publication figure)
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from visual_alpha import (
    DEFAULT_ENSEMBLE_SEEDS,
    FIGURES_DIR,
    PREDICTIONS_DIR,
    TABLES_DIR,
    apply_temperature_to_logits,
    compute_selective_prediction_curve,
    ensure_output_dirs,
    evaluate_predictions,
    fit_temperature_scaling,
    save_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SQ3 Calibration & Selective Prediction analysis.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_ENSEMBLE_SEEDS),
        help="Seeds included in the Deep Ensemble (default: 42 43 44 45 46).",
    )
    parser.add_argument(
        "--primary-seed",
        type=int,
        default=42,
        help="Reference single-model seed for M1 & M2 (default: 42).",
    )
    return parser.parse_args()


def compute_zoomed_reliability(
    y_true: np.ndarray,
    prob_up: np.ndarray,
    *,
    low: float = 0.35,
    high: float = 0.65,
    n_bins: int = 10,
    min_count: int = 200,
) -> pd.DataFrame:
    """Compute empirical positive rate vs mean predicted prob across active probability bins."""
    edges = np.linspace(low, high, n_bins + 1)
    bin_ids = np.digitize(prob_up, edges) - 1
    rows: list[dict[str, float]] = []
    for b in range(n_bins):
        mask = bin_ids == b
        cnt = int(mask.sum())
        if cnt >= min_count:
            rows.append(
                {
                    "bin_center": float(0.5 * (edges[b] + edges[b + 1])),
                    "mean_prob": float(prob_up[mask].mean()),
                    "emp_freq": float(y_true[mask].mean()),
                    "count": float(cnt),
                }
            )
    return pd.DataFrame(rows)


def plot_figure3(
    y_true: np.ndarray,
    probs_dict: dict[str, np.ndarray],
    selective_df: pd.DataFrame,
    metrics_map: dict[str, dict[str, float]],
    temp_single: float,
) -> None:
    """Generate two-panel publication Figure 3 for Section 5 (RQ3)."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    palette = sns.color_palette("colorblind")
    color_map = {
        "Single CNN (Uncalibrated)": palette[3],
        f"Single CNN + Temp Scaling (T={temp_single:.2f})": palette[0],
        "5-Seed Deep Ensemble": palette[2],
    }
    marker_map = {
        "Single CNN (Uncalibrated)": "o",
        f"Single CNN + Temp Scaling (T={temp_single:.2f})": "s",
        "5-Seed Deep Ensemble": "^",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.1), dpi=300)

    # Panel (a): Zoomed Reliability Diagram + Embedded Probability Density
    ax1.plot([0.36, 0.64], [0.36, 0.64], linestyle="--", color="0.35", lw=1.2, label="Perfect Calibration")
    for label, probs in probs_dict.items():
        rel = compute_zoomed_reliability(y_true, probs, low=0.35, high=0.65, n_bins=12, min_count=500)
        ece_val = metrics_map[label]["ece"]
        ax1.plot(
            rel["mean_prob"],
            rel["emp_freq"],
            marker=marker_map[label],
            ms=5.5,
            lw=1.8,
            color=color_map[label],
            label=f"{label} (ECE={ece_val:.4f})",
        )

    ax1_hist = ax1.twinx()
    ens_key = "5-Seed Deep Ensemble"
    single_key = "Single CNN (Uncalibrated)"
    ax1_hist.hist(
        probs_dict[single_key],
        bins=np.linspace(0.35, 0.65, 37),
        density=True,
        alpha=0.14,
        color=color_map[single_key],
    )
    ax1_hist.hist(
        probs_dict[ens_key],
        bins=np.linspace(0.35, 0.65, 37),
        density=True,
        alpha=0.18,
        color=color_map[ens_key],
    )
    ax1_hist.set_ylabel("Prediction Density", color="0.45", fontsize=9.5)
    ax1_hist.tick_params(axis="y", colors="0.45", labelsize=8.5)
    ax1_hist.grid(False)
    ax1_hist.set_ylim(0, ax1_hist.get_ylim()[1] * 2.2)

    ax1.set_xlim(0.36, 0.64)
    ax1.set_ylim(0.36, 0.64)
    ax1.set_xlabel("Mean Predicted Upward Probability $\\hat{p}$")
    ax1.set_ylabel("Empirical Upward Frequency $P(y=1 \\mid \\hat{p})$")
    ax1.set_title("(a) OOS Reliability Diagram & Narrow SNR Density", fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, fontsize=8.3)

    # Panel (b): Selective Prediction Accuracy vs Coverage
    for label in ("Single CNN (Uncalibrated)", "5-Seed Deep Ensemble"):
        sub = selective_df.loc[selective_df["model"] == label].sort_values(
            "coverage_pct", ascending=False
        )
        ax2.plot(
            sub["coverage_pct"],
            sub["accuracy_pct"],
            marker=marker_map[label],
            ms=5.5,
            lw=2.0,
            color=color_map[label],
            label=f"{label} (Sharpe={metrics_map[label]['ls_sharpe']:.2f})",
        )

    ax2.axhline(50.0, linestyle=":", color="0.45", lw=1.1, label="Random Guess (50.0%)")
    ax2.invert_xaxis()
    ax2.set_xlabel("Selected High-Confidence Coverage (%)  [100% $\\rightarrow$ Top 5%]")
    ax2.set_ylabel("Conditional OOS Directional Accuracy (%)")
    ax2.set_title("(b) Selective Prediction: Tail Confidence vs. Accuracy", fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, fontsize=8.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out_file = FIGURES_DIR / f"fig3_sq3_calibration.{ext}"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Saved {out_file}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    val_single_path = PREDICTIONS_DIR / f"val_baseline_seed{args.primary_seed}.parquet"
    test_single_path = PREDICTIONS_DIR / f"baseline_seed{args.primary_seed}.parquet"
    if not val_single_path.exists() or not test_single_path.exists():
        msg = f"Missing baseline predictions ({test_single_path}). Run exp_baseline.py first."
        raise FileNotFoundError(msg)

    val_single = pd.read_parquet(val_single_path)
    test_single = pd.read_parquet(test_single_path)

    # 1. Fit scalar Temperature T strictly on 1993-1999 validation set
    temp_single = fit_temperature_scaling(
        val_single["logit_down"].to_numpy(),
        val_single["logit_up"].to_numpy(),
        val_single["y_true"].to_numpy(),
    )
    print(f"Fitted validation temperature for seed {args.primary_seed}: T = {temp_single:.4f}")

    test_temp_df = test_single.copy()
    test_temp_df["prob_up"] = apply_temperature_to_logits(
        test_single["logit_down"].to_numpy(),
        test_single["logit_up"].to_numpy(),
        temp_single,
    )
    save_predictions(test_temp_df, PREDICTIONS_DIR / f"sq3_temp_scaled_seed{args.primary_seed}.parquet")

    # 2. Load all ensemble seeds for M3 & M4
    val_seed_dfs = [pd.read_parquet(PREDICTIONS_DIR / f"val_baseline_seed{s}.parquet") for s in args.seeds]
    test_seed_dfs = [pd.read_parquet(PREDICTIONS_DIR / f"baseline_seed{s}.parquet") for s in args.seeds]

    ens_prob_test = np.mean([d["prob_up"].to_numpy(dtype=np.float64) for d in test_seed_dfs], axis=0)
    test_ens_df = test_single.copy()
    test_ens_df["prob_up"] = ens_prob_test.astype(np.float32)

    # Optional M4: Temperature scaling on ensemble mean logits
    val_mean_ld = np.mean([d["logit_down"].to_numpy(dtype=np.float64) for d in val_seed_dfs], axis=0)
    val_mean_lu = np.mean([d["logit_up"].to_numpy(dtype=np.float64) for d in val_seed_dfs], axis=0)
    test_mean_ld = np.mean([d["logit_down"].to_numpy(dtype=np.float64) for d in test_seed_dfs], axis=0)
    test_mean_lu = np.mean([d["logit_up"].to_numpy(dtype=np.float64) for d in test_seed_dfs], axis=0)

    temp_ens = fit_temperature_scaling(val_mean_ld, val_mean_lu, val_single["y_true"].to_numpy())
    print(f"Fitted validation temperature for {len(args.seeds)}-seed ensemble: T_ens = {temp_ens:.4f}")
    test_ens_temp_df = test_single.copy()
    test_ens_temp_df["prob_up"] = apply_temperature_to_logits(test_mean_ld, test_mean_lu, temp_ens)
    save_predictions(test_ens_temp_df, PREDICTIONS_DIR / f"sq3_temp_scaled_ensemble{len(args.seeds)}.parquet")

    # 3. Evaluate all SQ3 methods
    temp_label = f"Single CNN + Temp Scaling (T={temp_single:.2f})"
    ens_label = f"{len(args.seeds)}-Seed Deep Ensemble"
    ens_temp_label = f"{len(args.seeds)}-Seed Ensemble + Temp Scaling (T={temp_ens:.2f})"

    candidates = [
        ("Single CNN (Uncalibrated)", 1.0, test_single),
        (temp_label, temp_single, test_temp_df),
        (ens_label, 1.0, test_ens_df),
        (ens_temp_label, temp_ens, test_ens_temp_df),
    ]

    summary_rows: list[dict[str, float | str]] = []
    metrics_map: dict[str, dict[str, float]] = {}
    selective_frames: list[pd.DataFrame] = []

    for label, temp_val, df_pred in candidates:
        m = evaluate_predictions(df_pred)
        metrics_map[label] = m
        summary_rows.append({"method": label, "temperature": temp_val, **m})

        sel = compute_selective_prediction_curve(
            df_pred["y_true"].to_numpy(),
            df_pred["prob_up"].to_numpy(),
        )
        sel.insert(0, "model", label)
        selective_frames.append(sel)

    summary_df = pd.DataFrame(summary_rows)
    sel_all_df = pd.concat(selective_frames, ignore_index=True)

    summary_path = TABLES_DIR / "sq3_calibration_summary.csv"
    sel_path = TABLES_DIR / "sq3_selective_prediction.csv"
    summary_df.to_csv(summary_path, index=False)
    sel_all_df.to_csv(sel_path, index=False)

    print(f"\nSaved SQ3 summary table to {summary_path}:")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved Selective Prediction table to {sel_path}:")
    print(
        sel_all_df.loc[sel_all_df["model"].isin(["Single CNN (Uncalibrated)", ens_label])].to_string(
            index=False, float_format=lambda v: f"{v:.2f}"
        )
    )

    plot_probs = {
        "Single CNN (Uncalibrated)": test_single["prob_up"].to_numpy(dtype=np.float64),
        temp_label: test_temp_df["prob_up"].to_numpy(dtype=np.float64),
        "5-Seed Deep Ensemble": test_ens_df["prob_up"].to_numpy(dtype=np.float64),
    }
    # Normalize ensemble key if custom seed count was passed
    if ens_label != "5-Seed Deep Ensemble":
        metrics_map["5-Seed Deep Ensemble"] = metrics_map[ens_label]
        sel_all_df["model"] = sel_all_df["model"].replace({ens_label: "5-Seed Deep Ensemble"})

    plot_figure3(
        y_true=test_single["y_true"].to_numpy(dtype=np.int64),
        probs_dict=plot_probs,
        selective_df=sel_all_df,
        metrics_map=metrics_map,
        temp_single=temp_single,
    )


if __name__ == "__main__":
    main()
