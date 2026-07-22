"""Feature construction — strictly as-of-signal-time.

Every feature at signal bar t may use data up to and including the close of
t and NOTHING later. test_trading_system.py asserts this by truncation
invariance: computing features on bars[:t+1] must equal computing them on
the full series. Feature set mirrors the reference fp_modeling.py (lagged
returns + regime) plus the spec's additions (vol, ATR ratio, MA distance,
day-of-week).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from trading_system.data import Bars, atr, realized_vol_daily

FEATURE_NAMES = [
    "ret_1", "ret_2", "ret_3",       # last 1/2/3-bar returns
    "vol_20",                        # 20-bar realized daily vol
    "atr_ratio",                     # ATR(14) / close
    "ma50_dist",                     # close / MA50 - 1
    "regime",                        # 1 if MA50 > MA200 else 0
    "dow",                           # day of week 0..4
    "dir_long",                      # primary signal direction (1 long)
]

_WARMUP = 210   # enough for MA200 + slack


def _sma(x: np.ndarray, w: int, t: int) -> float:
    if t + 1 < w:
        return float("nan")
    return float(np.mean(x[t + 1 - w: t + 1]))


def feature_vector(bars: Bars, t: int, direction: str) -> Optional[np.ndarray]:
    """Features at bar t (None during warm-up). Uses bars[<= t] only."""
    if t < _WARMUP or t >= len(bars):
        return None
    c = bars.close
    ret = lambda k: float(c[t] / c[t - k] - 1.0)  # noqa: E731
    vol20 = realized_vol_daily(c[: t + 1], 20)[t]
    atr14 = atr(bars.slice(0, t + 1), 14)[t]
    ma50 = _sma(c, 50, t)
    ma200 = _sma(c, 200, t)
    if not all(np.isfinite(v) for v in (vol20, atr14, ma50, ma200)):
        return None
    return np.array([
        ret(1), ret(2), ret(3),
        float(vol20),
        float(atr14 / c[t]),
        float(c[t] / ma50 - 1.0),
        1.0 if ma50 > ma200 else 0.0,
        float(bars.dates[t].dayofweek),
        1.0 if direction == "long" else 0.0,
    ], dtype=float)


def build_matrix(bars: Bars, ts: List[int], directions: List[str]
                 ) -> tuple[np.ndarray, List[int]]:
    """Feature matrix for the given signal bars; returns (X, kept_indices)."""
    rows, kept = [], []
    for i, (t, d) in enumerate(zip(ts, directions)):
        v = feature_vector(bars, t, d)
        if v is not None:
            rows.append(v)
            kept.append(i)
    if not rows:
        return np.empty((0, len(FEATURE_NAMES))), []
    return np.vstack(rows), kept
