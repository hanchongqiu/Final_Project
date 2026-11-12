"""Unit tests for the shared `visual_alpha` foundation package."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from visual_alpha import (
    CNN20D,
    ChartDataset,
    apply_image_mask,
    apply_temperature_to_logits,
    build_train_val_test_splits,
    compute_selective_prediction_curve,
    count_parameters,
    evaluate_predictions,
    fit_temperature_scaling,
    get_device,
    validate_prediction_frame,
)


def test_cnn20d_architecture_and_forward() -> None:
    model = CNN20D()
    assert model.FC_IN_FEATURES == 46080
    assert count_parameters(model) > 500_000

    device = get_device()
    model.to(device)
    x = torch.zeros((4, 1, 64, 60), dtype=torch.float32, device=device)
    b1 = model.block1(x)
    b2 = model.block2(b1)
    b3 = model.block3(b2)
    logits = model(x)

    assert tuple(b1.shape) == (4, 64, 13, 60)
    assert tuple(b2.shape) == (4, 128, 6, 60)
    assert tuple(b3.shape) == (4, 256, 3, 60)
    assert tuple(logits.shape) == (4, 2)


def test_image_masking_coordinates() -> None:
    x = torch.ones((2, 1, 64, 60), dtype=torch.float32)
    x[:, :, 51, :] = 0.0

    no_vol = apply_image_mask(x, "no_volume")
    assert float(no_vol[:, :, 52:64, :].sum()) == 0.0
    assert float(no_vol[:, :, 0:51, :].sum()) == float(2 * 51 * 60)

    rec_10 = apply_image_mask(x, "recent_10d")
    assert float(rec_10[:, :, :, 0:30].sum()) == 0.0
    assert float(rec_10[:, :, 0:51, 30:60].sum()) == float(2 * 51 * 30)

    old_10 = apply_image_mask(x, "early_10d")
    assert float(old_10[:, :, :, 30:60].sum()) == 0.0
    assert float(old_10[:, :, 0:51, 0:30].sum()) == float(2 * 51 * 30)


def test_memmap_dataset_and_margin_filtering() -> None:
    ds_1993 = ChartDataset([1993])
    assert len(ds_1993) > 80_000
    assert set(np.unique(ds_1993.metadata["Retx_20d_label"])).issubset({0, 1})

    x0, y0 = ds_1993[0]
    assert x0.dtype == torch.uint8
    assert tuple(x0.shape) == (1, 64, 60)
    assert int(y0.item()) in (0, 1)

    xb, yb = ds_1993.get_batch(np.array([0, 10, 25]))
    assert xb.dtype == torch.uint8
    assert tuple(xb.shape) == (3, 1, 64, 60)
    assert tuple(yb.shape) == (3,)
    assert torch.equal(xb[0], x0)

    splits = build_train_val_test_splits(
        train_val_years=[1993],
        test_years=[2000],
        train_min_abs_retx=0.03,
    )
    assert len(splits.train) < int(len(ds_1993) * 0.70)
    assert float(splits.train.metadata["Retx_20d"].abs().min()) >= 0.03 - 1e-6
    assert float(splits.val.metadata["Retx_20d"].abs().min()) < 0.01
    assert float(splits.test.metadata["Retx_20d"].abs().min()) < 0.01


def test_evaluate_predictions_and_temperature_scaling() -> None:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2015-01-31", periods=24, freq="ME")
    rows = []
    for d in dates:
        signal = rng.uniform(-1.5, 1.5, size=50)
        ret = 0.03 * signal + rng.normal(0.0, 0.05, size=50)
        y = (ret > 0).astype(np.int8)
        logit_down = (-1.5 * signal).astype(np.float32)
        logit_up = (1.5 * signal).astype(np.float32)
        prob_up = 1.0 / (1.0 + np.exp(-(logit_up - logit_down)))
        for i in range(50):
            rows.append(
                {
                    "year": np.int16(d.year),
                    "row_idx": np.int32(i),
                    "Date": d,
                    "StockID": f"S{i:03d}",
                    "y_true": y[i],
                    "logit_down": logit_down[i],
                    "logit_up": logit_up[i],
                    "prob_up": np.float32(prob_up[i]),
                    "Retx_20d": np.float32(ret[i]),
                    "Ret_20d": np.float32(ret[i]),
                    "MarketCap": np.float32(1e6),
                }
            )
    pred_df = pd.DataFrame(rows)
    validate_prediction_frame(pred_df)
    metrics = evaluate_predictions(pred_df)

    assert 55.0 < metrics["accuracy_pct"] <= 100.0
    assert 0.55 < metrics["auc"] <= 1.0
    assert metrics["ls_sharpe"] > 1.0

    temp = fit_temperature_scaling(
        pred_df["logit_down"].to_numpy(),
        pred_df["logit_up"].to_numpy(),
        pred_df["y_true"].to_numpy(),
    )
    assert temp > 1.0
    cal_probs = apply_temperature_to_logits(
        pred_df["logit_down"].to_numpy(),
        pred_df["logit_up"].to_numpy(),
        temp,
    )
    cal_df = pred_df.copy()
    cal_df["prob_up"] = cal_probs
    cal_metrics = evaluate_predictions(cal_df)
    assert cal_metrics["ce_loss"] <= metrics["ce_loss"] + 1e-6

    sel = compute_selective_prediction_curve(pred_df["y_true"].to_numpy(), cal_probs)
    assert float(sel.iloc[-1]["accuracy_pct"]) >= float(sel.iloc[0]["accuracy_pct"])
