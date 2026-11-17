"""Member C — SQ2 cross-horizon return-fallback tests.

Validates the 5d/60d evaluation-copy fix in `experiments/exp_sq2_label_noise.py`:
`evaluate_predictions`'s internal decile-portfolio fallback always substitutes
`Retx_20d`, which is wrong for non-20d horizons. The fix fills the evaluation copy
with the SAME-horizon `Retx_{h}` from `Retx_{h}` before calling the public evaluator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.exp_sq2_label_noise import fill_horizon_returns
from visual_alpha import compute_decile_portfolio, evaluate_predictions


def _make_frame(n_months: int = 12, n_stocks: int = 10, seed: int = 7) -> pd.DataFrame:
    """Synthetic contract-compliant frame with 5d return columns.

    With `n_stocks=10` every cross-section splits into exactly 10 single-stock
    deciles, so the top-ranked stock IS the long leg of its rebalance date.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2015-01-31", periods=n_months, freq="ME"):
        signal = rng.uniform(-1.5, 1.5, size=n_stocks)
        ret5 = 0.04 * signal + rng.normal(0.0, 0.02, size=n_stocks)
        probs = 1.0 / (1.0 + np.exp(-2.5 * signal))
        for i in range(n_stocks):
            p = float(probs[i])
            logit = float(np.log(p / (1.0 - p)))
            rows.append(
                {
                    "year": np.int16(d.year),
                    "row_idx": np.int32(i),
                    "Date": d,
                    "StockID": f"S{i:03d}",
                    "y_true": np.int8(ret5[i] > 0),
                    "logit_down": np.float32(-logit),
                    "logit_up": np.float32(logit),
                    "prob_up": np.float32(p),
                    "Retx_20d": np.float32(0.10),
                    "Ret_20d": np.float32(0.10),
                    "MarketCap": np.float32(1e6),
                    "Retx_5d": np.float32(ret5[i]),
                    "Ret_5d": np.float32(ret5[i]),
                }
            )
    return pd.DataFrame(rows)


def test_same_horizon_fallback_uses_retx_5d_not_retx_20d() -> None:
    df = _make_frame()
    date2 = df["Date"].unique()[2]
    idx = df.loc[df["Date"] == date2, "prob_up"].idxmax()  # sole decile-10 member
    df.loc[idx, "Ret_5d"] = np.nan
    df.loc[idx, "Retx_5d"] = np.float32(0.03)
    df.loc[idx, "y_true"] = np.int8(1)  # consistent with Retx_5d > 0
    assert df.loc[idx, "Ret_20d"] == pytest.approx(0.10)
    assert df.loc[idx, "Retx_5d"] == pytest.approx(0.03)

    filled = fill_horizon_returns(df, ret_col="Ret_5d", retx_col="Retx_5d")
    # (1) evaluation runs and the missing sample's return becomes 0.03 (not 0.10)
    assert float(filled.loc[idx, "Ret_5d"]) == pytest.approx(0.03)
    assert not filled["Ret_5d"].isna().any()
    assert np.isnan(float(df.loc[idx, "Ret_5d"]))  # original frame untouched

    metrics = evaluate_predictions(filled, ret_col="Ret_5d")
    assert 0.5 < metrics["auc"] <= 1.0
    assert metrics["accuracy_pct"] > 50.0

    _, monthly = compute_decile_portfolio(filled, ret_col="Ret_5d")
    assert float(monthly.loc[date2, "long_ret"]) == pytest.approx(0.03, abs=1e-8)
    assert float(monthly.loc[date2, "long_ret"]) != pytest.approx(0.10)


def test_compute_decile_portfolio_fallback_matches_prefilled_evaluation() -> None:
    df = _make_frame(seed=11)
    idx = df.index[15]
    df.loc[idx, "Ret_5d"] = np.nan
    df.loc[idx, "Retx_5d"] = np.float32(-0.05)
    df.loc[idx, "y_true"] = np.int8(0)  # consistent with Retx_5d < 0

    filled = fill_horizon_returns(df, ret_col="Ret_5d", retx_col="Retx_5d")
    port_fill, monthly_fill = compute_decile_portfolio(filled, ret_col="Ret_5d")
    # (2) direct same-horizon fallback call on the raw frame agrees exactly
    port_fb, monthly_fb = compute_decile_portfolio(
        df, ret_col="Ret_5d", fallback_ret_col="Retx_5d"
    )
    for key, value in port_fb.items():
        assert float(port_fill[key]) == pytest.approx(float(value), abs=1e-12)
    pd.testing.assert_series_equal(monthly_fill["ls_ret"], monthly_fb["ls_ret"])


def test_fill_horizon_returns_raises_when_both_returns_missing() -> None:
    df = _make_frame(n_months=3)
    df.loc[0, ["Ret_5d", "Retx_5d"]] = np.nan
    with pytest.raises(ValueError, match="still has 1 NaN"):
        fill_horizon_returns(df, ret_col="Ret_5d", retx_col="Retx_5d")
