"""MSDM 5054 Final Project — Visual Alpha shared package."""

from visual_alpha.dataset import ChartDataset, SplitDatasets, build_train_val_test_splits
from visual_alpha.evaluate import (
    apply_temperature_to_logits,
    compute_decile_portfolio,
    compute_ece,
    compute_selective_prediction_curve,
    evaluate_predictions,
    fit_temperature_scaling,
    save_predictions,
    validate_prediction_frame,
)
from visual_alpha.models import CNN20D, count_parameters
from visual_alpha.trainer import TrainResult, predict_dataset, train_model
from visual_alpha.utils import (
    CHECKPOINTS_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_ENSEMBLE_SEEDS,
    DEFAULT_SPLIT_SEED,
    FIGURES_DIR,
    IMG_H,
    IMG_W,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
    PROJECT_ROOT,
    TABLES_DIR,
    apply_image_mask,
    ensure_output_dirs,
    get_device,
    seed_everything,
)

__version__ = "0.1.0"

__all__ = [
    "CHECKPOINTS_DIR",
    "CNN20D",
    "DEFAULT_DATA_DIR",
    "DEFAULT_ENSEMBLE_SEEDS",
    "DEFAULT_SPLIT_SEED",
    "FIGURES_DIR",
    "IMG_H",
    "IMG_W",
    "OUTPUTS_DIR",
    "PREDICTIONS_DIR",
    "PROJECT_ROOT",
    "TABLES_DIR",
    "ChartDataset",
    "SplitDatasets",
    "TrainResult",
    "__version__",
    "apply_image_mask",
    "apply_temperature_to_logits",
    "build_train_val_test_splits",
    "compute_decile_portfolio",
    "compute_ece",
    "compute_selective_prediction_curve",
    "count_parameters",
    "ensure_output_dirs",
    "evaluate_predictions",
    "fit_temperature_scaling",
    "get_device",
    "predict_dataset",
    "save_predictions",
    "seed_everything",
    "train_model",
    "validate_prediction_frame",
]

