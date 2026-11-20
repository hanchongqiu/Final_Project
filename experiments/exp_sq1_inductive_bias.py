"""SQ1 inductive bias and occlusion experiments. Run with --help for individual arms."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightgbm as lgb  # Import before torch on macOS.
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from torch import nn

from visual_alpha import (
    CHECKPOINTS_DIR,
    CNN20D,
    PREDICTIONS_DIR,
    TABLES_DIR,
    ChartDataset,
    build_train_val_test_splits,
    count_parameters,
    ensure_output_dirs,
    evaluate_predictions,
    get_device,
    predict_dataset,
    save_predictions,
    seed_everything,
    train_model,
)

ARCH = ("baseline", "pca_lr", "pixel_lgbm", "temporal", "resnet")
MASKS = ("no_volume", "recent_10d", "early_10d")
ARMS = (*ARCH, *MASKS, "retrain_no_volume")


class TemporalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for cin, cout in ((192, 64), (64, 128), (128, 256)):
            layers.extend(
                [nn.Conv1d(cin, cout, 3, padding=1), nn.BatchNorm1d(cout), nn.LeakyReLU(0.01)]
            )
        self.layers = nn.Sequential(*layers, nn.Flatten(), nn.Dropout(0.5), nn.Linear(5120, 2))

    def forward(self, x):
        # Columns 3*d through 3*d+2 belong to day d.
        daily = x.reshape(-1, 64, 20, 3).transpose(2, 3).reshape(-1, 192, 20)
        return self.layers(daily)


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
        )
        self.skip = (
            nn.Identity()
            if cin == cout
            else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        )

    def forward(self, x):
        return torch.relu(self.main(x) + self.skip(x))


def residual_net():
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        ResidualBlock(32, 32, 1),
        ResidualBlock(32, 64, 2),
        ResidualBlock(64, 128, 2),
        ResidualBlock(128, 256, 2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.5),
        nn.Linear(256, 2),
    )


def pixel_features(x):
    """Twelve observable price/volume summaries per successive five-day block."""
    img = x.numpy()[:, 0]
    center = img[:, :51, 1::3] > 0
    seen = center.any(axis=1)
    high = np.where(seen, 50 - center.argmax(axis=1), np.nan)
    low = np.where(seen, center[:, ::-1].argmax(axis=1), np.nan)
    volume = (img[:, 52:, 1::3] > 0).sum(axis=1)
    features = []
    for start in range(0, 20, 5):
        sl = slice(start, start + 5)
        h, lo, v = high[:, sl], low[:, sl], volume[:, sl]
        mid, span = (h + lo) / 2, h - lo
        features.extend(
            [
                np.nanmean(mid, axis=1),
                np.nanstd(mid, axis=1),
                mid[:, -1] - mid[:, 0],
                np.nanmax(h, axis=1),
                np.nanmin(lo, axis=1),
                np.nanmean(span, axis=1),
                np.nanmax(span, axis=1),
                v.mean(axis=1),
                v.std(axis=1),
                v.max(axis=1),
                v[:, -1] - v[:, 0],
                (img[:, :51, 3 * start : 3 * (start + 5)] > 0).mean(axis=(1, 2)),
            ]
        )
    return np.column_stack(features).astype(np.float32)


def flat_pixels(x):
    return x.numpy().reshape(len(x), -1).astype(np.float32) / 255


def fit_classical(arm, splits):
    start = time.perf_counter()
    train = splits.train
    if arm == "pca_lr":
        indices = np.random.default_rng(42).choice(len(train), 150_000, replace=False)
        batches = np.array_split(indices, 150)
        pca = IncrementalPCA(n_components=64)
        for batch in batches:
            pca.partial_fit(flat_pixels(train.get_batch(batch)[0]))

        def features(x):
            return pca.transform(flat_pixels(x))

        x_train = np.concatenate([features(train.get_batch(i)[0]) for i in batches])
        labels = train.metadata[train.label_col].to_numpy()[indices]
        model = LogisticRegression(C=1, max_iter=500)
    else:
        features = pixel_features
        x_train = np.concatenate([features(x) for x, _ in train.iter_batches(1024)])
        labels = train.metadata[train.label_col].to_numpy()
        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            n_jobs=4,
            verbosity=-1,
        )
    model.fit(x_train, labels)
    seconds = time.perf_counter() - start
    if arm == "pca_lr":
        size = pca.components_.size + pca.mean_.size + model.coef_.size + model.intercept_.size
        unit = "projection, centering and logistic coefficients"
    else:
        size = sum(t["num_leaves"] for t in model.booster_.dump_model()["tree_info"])
        unit = "tree leaves"
    prob = np.concatenate(
        [model.predict_proba(features(x))[:, 1] for x, _ in splits.test.iter_batches(1024)]
    )
    return splits.test.build_prediction_frame(prob), seconds, size, unit


def check_baseline(model, test, reference, device):
    if len(reference) != len(test):
        raise ValueError("Reference and test split have different lengths")
    for key in ("year", "row_idx"):
        np.testing.assert_array_equal(reference[key], test.metadata[key])
    indices = np.linspace(0, len(test) - 1, 128, dtype=int)
    x, _ = test.get_batch(indices)
    model.to(device).eval()
    with torch.inference_mode():
        p = model(x.to(device).float() / 255).softmax(1)[:, 1].cpu().numpy()
    ref = reference.prob_up.to_numpy()[indices]
    np.testing.assert_allclose(p, ref, rtol=0, atol=1e-4)
    print(f"Baseline replay: max probability difference {np.abs(p - ref).max():.2g}", flush=True)


def write_row(path: Path, row, order):
    records = pd.read_csv(path).set_index("arm").to_dict("index") if path.exists() else {}
    records[row["arm"]] = row
    frame = pd.DataFrame([dict(records[k], arm=k) for k in order if k in records])
    if "baseline" in frame.arm.values and "no_volume" in order:
        base = frame.set_index("arm").loc["baseline"]
        for key in ("accuracy_pct", "ls_sharpe"):
            frame["delta_" + key] = frame[key] - base[key]
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--eval-batch-size", type=int, default=256)
    args = parser.parse_args()
    ensure_output_dirs()
    device = get_device()
    needs_training = any(arm not in ("baseline", *MASKS) for arm in args.arms)
    splits = build_train_val_test_splits() if needs_training else None
    test = splits.test if needs_training else ChartDataset(range(2000, 2020))
    reference_path = PREDICTIONS_DIR / "baseline_seed42.parquet"
    baseline = CNN20D()
    weights = torch.load(
        CHECKPOINTS_DIR / "baseline_seed42.pt", weights_only=True, map_location="cpu"
    )
    baseline.load_state_dict(weights["model_state_dict"])
    if device.type == "cuda":
        baseline.to(memory_format=torch.channels_last)
    check_baseline(
        baseline,
        test,
        pd.read_parquet(reference_path, columns=["year", "row_idx", "prob_up"]),
        device,
    )
    for arm in dict.fromkeys(args.arms):
        print(f"Starting {arm} on {device}", flush=True)
        seed_everything(42)
        size = count_parameters(baseline)
        unit = "trainable weights and biases"
        seconds = 0.0
        if arm == "baseline":
            pred = pd.read_parquet(reference_path)
            hist = pd.read_csv(TABLES_DIR / "baseline_training_history.csv")
            seconds = hist.loc[hist.seed == 42, "epoch_sec"].sum()
        elif arm in ("pca_lr", "pixel_lgbm"):
            with threadpool_limits(limits=4, user_api="blas"):
                pred, seconds, size, unit = fit_classical(arm, splits)
        else:
            mask = arm if arm in MASKS else None
            model = baseline
            if arm not in MASKS:
                model = {
                    "temporal": TemporalCNN,
                    "resnet": residual_net,
                    "retrain_no_volume": CNN20D,
                }[arm]()
                if device.type == "cuda" and arm != "temporal":
                    model.to(memory_format=torch.channels_last)
                if arm == "retrain_no_volume":
                    mask = "no_volume"
                elif arm in ("temporal", "resnet"):
                    for layer in model.modules():
                        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                            nn.init.xavier_uniform_(layer.weight)
                            if layer.bias is not None:
                                nn.init.zeros_(layer.bias)
                result = train_model(
                    model,
                    splits.train,
                    splits.val,
                    device=device,
                    eval_batch_size=args.eval_batch_size,
                    image_mask_mode=mask,
                    checkpoint_path=CHECKPOINTS_DIR / f"sq1_{arm}.pt",
                )
                seconds = result.elapsed_sec
                size = count_parameters(model)
                result.history.to_csv(TABLES_DIR / f"sq1_{arm}_history.csv", index=False)
            pred = predict_dataset(
                model, test, device=device, image_mask_mode=mask, batch_size=args.eval_batch_size
            )
        if arm != "baseline":
            path = PREDICTIONS_DIR / f"sq1_{arm}.parquet"
            tmp = path.with_suffix(".tmp")
            save_predictions(pred, tmp)
            tmp.replace(path)
        row = dict(
            arm=arm,
            model_size=size,
            size_unit=unit,
            train_seconds=seconds,
            **evaluate_predictions(pred),
        )
        print(row, flush=True)
        if arm in ARCH:
            write_row(TABLES_DIR / "tab2_inductive_bias.csv", row, ARCH)
        if arm not in ARCH or arm == "baseline":
            write_row(
                TABLES_DIR / "sq1_masking_ablation.csv",
                row,
                ("baseline", *MASKS, "retrain_no_volume"),
            )
        del pred


if __name__ == "__main__":
    main()
