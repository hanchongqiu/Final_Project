"""Member A — Baseline 20-Day CNN 5-seed replication & Table 1 generator.

Usage:
    .venv/bin/python experiments/exp_baseline.py
    .venv/bin/python experiments/exp_baseline.py --seeds 42 --max-epochs 10
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from visual_alpha import (
    CHECKPOINTS_DIR,
    CNN20D,
    DEFAULT_ENSEMBLE_SEEDS,
    PREDICTIONS_DIR,
    TABLES_DIR,
    build_train_val_test_splits,
    count_parameters,
    ensure_output_dirs,
    evaluate_predictions,
    get_device,
    predict_dataset,
    save_predictions,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train & evaluate 20-day Baseline CNN ensemble.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_ENSEMBLE_SEEDS),
        help="Random seeds to train/evaluate (default: 42 43 44 45 46).",
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=256, help="Training mini-batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=1024, help="Inference batch size.")
    parser.add_argument("--max-epochs", type=int, default=15, help="Max training epochs.")
    parser.add_argument("--patience", type=int, default=2, help="Validation early-stopping patience.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing checkpoint & prediction parquets if already saved.",
    )
    return parser.parse_args()


def build_table1(seed_rows: list[dict[str, float | str]], ens_metrics: dict[str, float]) -> pd.DataFrame:
    """Format Table 1 comparing Individual Seeds, Single-Seed Mean±Std, and 5-Seed Ensemble."""
    metric_keys = [
        "accuracy_pct",
        "ce_loss",
        "auc",
        "ece",
        "ls_sharpe",
        "ls_tstat",
        "ls_ann_ret_pct",
        "long_ann_ret_pct",
        "short_ann_ret_pct",
    ]
    df_seeds = pd.DataFrame(seed_rows)
    n_samples = float(df_seeds["n_samples"].iloc[0])

    summary_rows: list[dict[str, float | str]] = list(seed_rows)

    if len(seed_rows) > 1:
        mean_row: dict[str, float | str] = {
            "model": f"Single CNN (Mean of {len(seed_rows)} seeds)",
            "n_samples": n_samples,
        }
        std_row: dict[str, float | str] = {
            "model": f"Single CNN (Std of {len(seed_rows)} seeds)",
            "n_samples": 0.0,
        }
        for k in metric_keys:
            mean_row[k] = float(df_seeds[k].mean())
            std_row[k] = float(df_seeds[k].std(ddof=1))
        summary_rows.append(mean_row)
        summary_rows.append(std_row)

    ens_row: dict[str, float | str] = {
        "model": f"Deep Ensemble ({len(seed_rows)} seeds)",
        "n_samples": n_samples,
    }
    for k in metric_keys:
        ens_row[k] = float(ens_metrics[k])
    summary_rows.append(ens_row)
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    device = get_device()

    print(f"=== Baseline 20-Day CNN Replication | Device: {device} ===", flush=True)
    t_load = time.perf_counter()
    splits = build_train_val_test_splits()
    print(
        f"Loaded splits in {time.perf_counter() - t_load:.1f}s | "
        f"Train={len(splits.train):,} | Val={len(splits.val):,} | OOS Test={len(splits.test):,}",
        flush=True,
    )

    sample_model = CNN20D()
    print(f"CNN20D trainable parameters: {count_parameters(sample_model):,}", flush=True)
    del sample_model

    seed_metric_rows: list[dict[str, float | str]] = []
    history_frames: list[pd.DataFrame] = []
    val_prob_list: list[np.ndarray] = []
    test_prob_list: list[np.ndarray] = []
    test_logit_down_list: list[np.ndarray] = []
    test_logit_up_list: list[np.ndarray] = []

    for seed in args.seeds:
        ckpt_path = CHECKPOINTS_DIR / f"baseline_seed{seed}.pt"
        val_pred_path = PREDICTIONS_DIR / f"val_baseline_seed{seed}.parquet"
        test_pred_path = PREDICTIONS_DIR / f"baseline_seed{seed}.parquet"

        if args.skip_existing and ckpt_path.exists() and val_pred_path.exists() and test_pred_path.exists():
            print(f"\n--- Seed {seed}: loading existing predictions ---", flush=True)
            val_df = pd.read_parquet(val_pred_path)
            test_df = pd.read_parquet(test_pred_path)
        else:
            print(f"\n--- Seed {seed}: training CNN20D ---", flush=True)
            model = CNN20D()
            if args.skip_existing and ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(ckpt["model_state_dict"])
                print(
                    f"  Loaded checkpoint {ckpt_path.name} (best_epoch={ckpt['best_epoch']}, "
                    f"val_loss={ckpt['best_val_loss']:.4f})",
                    flush=True,
                )
            else:
                res = train_model(
                    model,
                    splits.train,
                    splits.val,
                    seed=seed,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    eval_batch_size=args.eval_batch_size,
                    max_epochs=args.max_epochs,
                    patience=args.patience,
                    checkpoint_path=ckpt_path,
                    device=device,
                )
                hist = res.history.copy()
                hist.insert(0, "seed", seed)
                history_frames.append(hist)

            t_inf = time.perf_counter()
            val_df = predict_dataset(
                model, splits.val, batch_size=args.eval_batch_size, device=device
            )
            test_df = predict_dataset(
                model, splits.test, batch_size=args.eval_batch_size, device=device
            )
            save_predictions(val_df, val_pred_path)
            save_predictions(test_df, test_pred_path)
            print(f"  Saved Val & OOS predictions in {time.perf_counter() - t_inf:.1f}s", flush=True)

        metrics = evaluate_predictions(test_df)
        print(
            f"  [OOS 2000-2019 Seed {seed}] "
            f"Acc={metrics['accuracy_pct']:.2f}% | CE={metrics['ce_loss']:.4f} | "
            f"AUC={metrics['auc']:.4f} | ECE={metrics['ece']:.4f} | "
            f"LS Sharpe={metrics['ls_sharpe']:.2f} (t={metrics['ls_tstat']:.2f})",
            flush=True,
        )
        seed_metric_rows.append({"model": f"CNN20D (seed={seed})", **metrics})
        val_prob_list.append(val_df["prob_up"].to_numpy(dtype=np.float64))
        test_prob_list.append(test_df["prob_up"].to_numpy(dtype=np.float64))
        test_logit_down_list.append(test_df["logit_down"].to_numpy(dtype=np.float64))
        test_logit_up_list.append(test_df["logit_up"].to_numpy(dtype=np.float64))

    if history_frames:
        pd.concat(history_frames, ignore_index=True).to_csv(
            TABLES_DIR / "baseline_training_history.csv", index=False
        )

    # Build and save validation & OOS ensemble predictions
    ens_suffix = f"ensemble{len(args.seeds)}"
    val_ens_df = splits.val.build_prediction_frame(
        prob_up=np.mean(val_prob_list, axis=0).astype(np.float32),
    )
    test_ens_df = splits.test.build_prediction_frame(
        prob_up=np.mean(test_prob_list, axis=0).astype(np.float32),
        logit_down=np.mean(test_logit_down_list, axis=0).astype(np.float32),
        logit_up=np.mean(test_logit_up_list, axis=0).astype(np.float32),
    )
    save_predictions(val_ens_df, PREDICTIONS_DIR / f"val_baseline_{ens_suffix}.parquet")
    save_predictions(test_ens_df, PREDICTIONS_DIR / f"baseline_{ens_suffix}.parquet")

    ens_metrics = evaluate_predictions(test_ens_df)
    print(
        f"\n=== [OOS 2000-2019 {len(args.seeds)}-Seed Ensemble] ===\n"
        f"Acc={ens_metrics['accuracy_pct']:.2f}% | CE={ens_metrics['ce_loss']:.4f} | "
        f"AUC={ens_metrics['auc']:.4f} | ECE={ens_metrics['ece']:.4f} | "
        f"LS Sharpe={ens_metrics['ls_sharpe']:.2f} (t={ens_metrics['ls_tstat']:.2f})",
        flush=True,
    )

    tab1 = build_table1(seed_metric_rows, ens_metrics)
    tab1_path = TABLES_DIR / "tab1_baseline.csv"
    tab1.to_csv(tab1_path, index=False)
    print(f"\nSaved Table 1 summary to {tab1_path}")
    print(tab1.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
