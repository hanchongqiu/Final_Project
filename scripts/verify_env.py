"""Environment, hardware accelerator, and dataset integrity check for MSDM 5054 Final Project."""

from __future__ import annotations

import platform
import site
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "monthly_20d"
YEARS = list(range(1993, 2020))
IMG_H, IMG_W = 64, 60


def ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def ensure_editable_package_visible() -> None:
    """Work around macOS iCloud Desktop/Documents setting UF_HIDDEN on .venv/*.pth (CPython gh-113802)."""
    try:
        __import__("visual_alpha")
        return
    except ModuleNotFoundError:
        src_pkg = PROJECT_ROOT / "src" / "visual_alpha"
        site_dirs = [Path(p) for p in site.getsitepackages() if "site-packages" in p]
        if src_pkg.exists() and site_dirs:
            link_path = site_dirs[0] / "visual_alpha"
            if not link_path.exists():
                link_path.symlink_to(src_pkg, target_is_directory=True)
                ok("Linked src/visual_alpha into site-packages (macOS iCloud UF_HIDDEN .pth fix)")


def check_python_and_packages() -> None:
    print("=== 1. Python & Package Check ===")
    if (sys.version_info.major, sys.version_info.minor) < (3, 11):
        fail(f"Python >= 3.11 required, found {sys.version.split()[0]}")
    ok(f"Python {sys.version.split()[0]} ({platform.platform()})")

    required = [
        "numpy",
        "pandas",
        "pyarrow",
        "lightgbm",  # Must be imported before torch on macOS ARM64 to avoid libomp collision
        "torch",
        "sklearn",
        "matplotlib",
        "seaborn",
        "visual_alpha",
    ]
    for mod_name in required:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            ok(f"{mod_name:<14} {ver}")
        except (ImportError, OSError) as exc:
            fail(f"Cannot import '{mod_name}': {exc}")


def check_accelerators() -> None:
    print("\n=== 2. Compute Backend & LightGBM Sanity Check ===")
    import lightgbm as lgb
    import numpy as np
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        desc = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        desc = "Apple Silicon Metal (mps)"
    else:
        device = torch.device("cpu")
        desc = "CPU (no GPU backend detected)"

    x = torch.tensor([[0, 255], [255, 0]], dtype=torch.uint8, device=device).float().div_(255.0)
    if float(x.sum().item()) != 2.0:
        fail(f"Unexpected tensor op result on {device}")
    ok(f"PyTorch tensor forward test passed on: {desc}")

    rng = np.random.default_rng(42)
    X_dummy = rng.normal(size=(64, 8)).astype(np.float32)
    y_dummy = (X_dummy[:, 0] > 0).astype(np.int32)
    clf = lgb.LGBMClassifier(n_estimators=5, max_depth=2, verbosity=-1, random_state=42)
    clf.fit(X_dummy, y_dummy)
    _ = clf.predict_proba(X_dummy[:4])
    ok("LightGBM fit/predict sanity check passed")


def check_dataset() -> None:
    print("\n=== 3. Dataset Integrity Check (1993-2019) ===")
    import numpy as np
    import pandas as pd

    if not DATA_DIR.exists():
        fail(
            f"Missing {DATA_DIR}. Please unzip img_data.zip into data/ "
            "(see README.md Section 3)."
        )

    total_rows = 0
    valid_train_val = 0
    valid_oos = 0
    bytes_per_img = IMG_H * IMG_W

    for year in YEARS:
        dat_path = DATA_DIR / f"20d_month_has_vb_[20]_ma_{year}_images.dat"
        feather_path = DATA_DIR / f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"
        if not dat_path.exists() or not feather_path.exists():
            fail(f"Missing files for year {year} in {DATA_DIR}")

        file_bytes = dat_path.stat().st_size
        if file_bytes % bytes_per_img != 0:
            fail(f"Corrupted .dat size for {year}: {file_bytes} not divisible by {bytes_per_img}")
        n_imgs = file_bytes // bytes_per_img

        df = pd.read_feather(feather_path, columns=["Date", "StockID", "Retx_20d_label", "Ret_20d"])
        if len(df) != n_imgs:
            fail(f"Row mismatch in {year}: {n_imgs} images vs {len(df)} label rows")

        n_valid = int((df["Retx_20d_label"] != 2).sum())
        total_rows += n_imgs
        if year <= 1999:
            valid_train_val += n_valid
        else:
            valid_oos += n_valid

    # Spot-check memmap pixel invariants on 1993 and 2019
    for check_year in (1993, 2019):
        dat_path = DATA_DIR / f"20d_month_has_vb_[20]_ma_{check_year}_images.dat"
        mm = np.memmap(dat_path, dtype=np.uint8, mode="r").reshape((-1, IMG_H, IMG_W))
        sample = np.asarray(mm[:200])
        uniq = set(np.unique(sample).tolist())
        if not uniq.issubset({0, 255}):
            fail(f"Unexpected pixel values in {check_year}: {uniq}")
        if int(sample[:, 51, :].sum()) != 0:
            fail(f"Row 51 separator line is non-zero in {check_year}")

    ok(f"All 27 years (1993-2019) aligned: total={total_rows:,} samples")
    ok(f"1993-1999 valid train/val pool : {valid_train_val:,} samples")
    ok(f"2000-2019 valid OOS test pool  : {valid_oos:,} samples")
    ok("Memmap uint8 {0, 255} & row-51 separator invariants verified")


def main() -> None:
    ensure_editable_package_visible()
    check_python_and_packages()
    check_accelerators()
    check_dataset()
    print("\nAll checks passed! Your environment and dataset are 100% ready.")


if __name__ == "__main__":
    main()
