"""Reusable PyTorch training, validation early-stopping, and batched inference loop."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from visual_alpha.dataset import ChartDataset
from visual_alpha.utils import apply_image_mask, get_device, seed_everything


@dataclass
class TrainResult:
    """Summary artifact returned by `train_model`."""

    best_epoch: int
    best_val_loss: float
    best_val_acc_pct: float
    elapsed_sec: float
    history: pd.DataFrame
    checkpoint_path: Path | None


def _prepare_inputs(
    x_uint8: torch.Tensor,
    device: torch.device,
    image_mask_mode: str | None = None,
) -> torch.Tensor:
    """Move uint8 batch to target accelerator, cast to float32 in [0, 1], and mask if needed."""
    x = x_uint8.to(device, non_blocking=True).float().div_(255.0)
    if image_mask_mode is not None and image_mask_mode != "none":
        x = apply_image_mask(x, image_mask_mode)
    return x


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    dataset: ChartDataset,
    *,
    batch_size: int = 512,
    device: torch.device | None = None,
    image_mask_mode: str | None = None,
) -> pd.DataFrame:
    """Run batched inference over a `ChartDataset` and return a standardized DataFrame."""
    dev = device or get_device()
    model.to(dev)
    model.eval()

    n = len(dataset)
    logit_down = np.empty(n, dtype=np.float32)
    logit_up = np.empty(n, dtype=np.float32)
    prob_up = np.empty(n, dtype=np.float32)

    offset = 0
    for x_u8, _ in dataset.iter_batches(batch_size=batch_size, shuffle=False):
        bsz = len(x_u8)
        x = _prepare_inputs(x_u8, dev, image_mask_mode=image_mask_mode)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        logits_np = logits.detach().cpu().numpy()
        probs_np = probs[:, 1].detach().cpu().numpy()

        logit_down[offset : offset + bsz] = logits_np[:, 0]
        logit_up[offset : offset + bsz] = logits_np[:, 1]
        prob_up[offset : offset + bsz] = probs_np
        offset += bsz

    return dataset.build_prediction_frame(
        prob_up=prob_up,
        logit_down=logit_down,
        logit_up=logit_up,
    )


def evaluate_loss_and_acc(
    model: nn.Module,
    dataset: ChartDataset,
    *,
    criterion: nn.Module,
    batch_size: int = 512,
    device: torch.device,
    image_mask_mode: str | None = None,
) -> tuple[float, float]:
    """Compute exact sample-weighted average loss and accuracy (%) on a dataset."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for x_u8, y_cpu in dataset.iter_batches(batch_size=batch_size, shuffle=False):
            bsz = len(y_cpu)
            x = _prepare_inputs(x_u8, device, image_mask_mode=image_mask_mode)
            y = y_cpu.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            preds = logits.argmax(dim=1)

            total_loss += float(loss.item()) * bsz
            total_correct += int((preds == y).sum().item())
            total_count += bsz

    return total_loss / total_count, (total_correct / total_count) * 100.0


def train_model(
    model: nn.Module,
    train_ds: ChartDataset,
    val_ds: ChartDataset,
    *,
    seed: int = 42,
    lr: float = 1e-5,
    weight_decay: float = 0.0,
    batch_size: int = 256,
    eval_batch_size: int = 512,
    max_epochs: int = 15,
    patience: int = 2,
    label_smoothing: float = 0.0,
    image_mask_mode: str | None = None,
    checkpoint_path: Path | str | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> TrainResult:
    """Train a PyTorch classifier with validation Cross-Entropy early stopping."""
    seed_everything(seed)
    dev = device or get_device()
    model.to(dev)

    train_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    val_criterion = nn.CrossEntropyLoss()  # Always monitor true un-smoothed CE on validation
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history_rows: list[dict[str, float | int]] = []
    t0 = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for x_u8, y_cpu in train_ds.iter_batches(
            batch_size=batch_size, shuffle=True, seed=seed * 1000 + epoch
        ):
            bsz = len(y_cpu)
            x = _prepare_inputs(x_u8, dev, image_mask_mode=image_mask_mode)
            y = y_cpu.to(dev, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = train_criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * bsz
            running_correct += int((logits.argmax(dim=1) == y).sum().item())
            running_count += bsz

        train_loss = running_loss / running_count
        train_acc = (running_correct / running_count) * 100.0
        val_loss, val_acc = evaluate_loss_and_acc(
            model,
            val_ds,
            criterion=val_criterion,
            batch_size=eval_batch_size,
            device=dev,
            image_mask_mode=image_mask_mode,
        )
        ep_sec = time.perf_counter() - ep_t0

        improved = val_loss < (best_val_loss - 1e-6)
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if checkpoint_path is not None:
                ckpt = Path(checkpoint_path)
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val_loss,
                        "best_val_acc_pct": best_val_acc,
                    },
                    ckpt,
                )
        else:
            epochs_no_improve += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc_pct": train_acc,
                "val_loss": val_loss,
                "val_acc_pct": val_acc,
                "epoch_sec": ep_sec,
            }
        )
        if verbose:
            mark = "*" if improved else " "
            print(
                f"  [Epoch {epoch:02d}/{max_epochs:02d}]{mark} "
                f"train_loss={train_loss:.4f} ({train_acc:.2f}%) | "
                f"val_loss={val_loss:.4f} ({val_acc:.2f}%) | {ep_sec:.1f}s",
                flush=True,
            )

        if epochs_no_improve >= patience:
            if verbose:
                print(
                    f"  Early stopping triggered at epoch {epoch} "
                    f"(best epoch={best_epoch}, best val_loss={best_val_loss:.4f}).",
                    flush=True,
                )
            break

    model.load_state_dict(best_state)
    total_sec = time.perf_counter() - t0
    return TrainResult(
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_val_acc_pct=best_val_acc,
        elapsed_sec=total_sec,
        history=pd.DataFrame(history_rows),
        checkpoint_path=Path(checkpoint_path) if checkpoint_path is not None else None,
    )
