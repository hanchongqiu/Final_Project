"""Render SQ1 Figure 1 from completed full-panel experiment tables."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visual_alpha import FIGURES_DIR, PREDICTIONS_DIR, TABLES_DIR, compute_decile_portfolio


def masking_intervals(arms):
    """Paired circular block bootstrap, resampling all stocks in each selected month."""
    estimates = []
    for arm in arms:
        name = "baseline_seed42" if arm == "baseline" else f"sq1_{arm}"
        df = pd.read_parquet(
            PREDICTIONS_DIR / f"{name}.parquet",
            columns=["Date", "prob_up", "y_true", "Ret_20d", "Retx_20d"],
        )
        correct = ((df.prob_up >= 0.5) == df.y_true).groupby(df.Date).agg(["sum", "size"])
        _, portfolio = compute_decile_portfolio(df)
        pd.testing.assert_index_equal(correct.index, portfolio.index)
        if arm == "baseline":
            dates = correct.index
            blocks = np.random.default_rng(42).integers(len(dates), size=(2000, 20, 1))
            months = ((blocks + np.arange(12)) % len(dates)).reshape(2000, 240)
        pd.testing.assert_index_equal(correct.index, dates)
        acc = (
            100
            * correct["sum"].to_numpy()[months].sum(1)
            / correct["size"].to_numpy()[months].sum(1)
        )
        returns = portfolio.ls_ret.to_numpy()[months]
        sharpe = np.sqrt(12) * returns.mean(1) / returns.std(1, ddof=1)
        estimates.append(np.stack([acc, sharpe]))
    return np.quantile(np.array(estimates) - estimates[0], [0.025, 0.975], axis=2)


def main():
    arch = (
        pd.read_csv(TABLES_DIR / "tab2_inductive_bias.csv")
        .set_index("arm")
        .loc[["pca_lr", "pixel_lgbm", "temporal", "baseline", "resnet"]]
    )
    masks = (
        pd.read_csv(TABLES_DIR / "sq1_masking_ablation.csv")
        .set_index("arm")
        .loc[["baseline", "no_volume", "recent_10d", "early_10d"]]
    )
    if not (arch.n_samples.eq(1_491_887).all() and masks.n_samples.eq(1_491_887).all()):
        raise ValueError("Figure 1 requires full-panel results")
    intervals = masking_intervals(masks.index)
    fig, axes = plt.subplots(2, 2, figsize=(9, 3.8), layout="constrained", sharex="col")
    axes[0, 0].set_title("(a) Model architecture", loc="left")
    axes[0, 1].set_title("(b) Inference-time masking", loc="left")
    labels = ["PCA\n+ LR", "Pixel\n+ GBM", "1D\nCNN", "2D\nCNN", "ResNet"]
    mask_labels = ["Full\nchart", "Hide\nvolume", "Hide first\n10 days", "Hide last\n10 days"]
    for ax, data, ticks, ylabel, origin in (
        (axes[0, 0], arch.auc, labels, "ROC-AUC", 0.5),
        (axes[1, 0], arch.ls_sharpe, labels, "L/S Sharpe", 0),
        (axes[0, 1], masks.delta_accuracy_pct, mask_labels, "Δ accuracy (pp)", 0),
        (axes[1, 1], masks.delta_ls_sharpe, mask_labels, "Δ L/S Sharpe", 0),
    ):
        colors = ["#366682" if name == "baseline" else "#258270" for name in data.index]
        ax.bar(range(len(ticks)), data - origin, bottom=origin, color=colors, width=0.62)
        ax.axhline(origin, color=".45", linewidth=0.7)
        ax.set_xticks(range(len(ticks)), ticks, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
    for metric, ax in enumerate(axes[:, 1]):
        for i in range(1, len(masks)):
            lo, hi = intervals[:, i, metric]
            ax.vlines(i, lo, hi, color="#23323b", lw=1.1)
            ax.hlines([lo, hi], i - 0.08, i + 0.08, color="#23323b", lw=1.1)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"fig1_sq1_ablation.{suffix}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
