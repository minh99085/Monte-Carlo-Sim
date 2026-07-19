#!/usr/bin/env python3
"""
make_demo_calibration.py — build a FAKE calibration table for local testing.

This writes ``calibration/DEMO_5d.json`` from synthetic price data that has a
deliberate, strong upward edge whenever the trend is bullish with high RSI.
It exists so you can watch the whole TradingView → decision pipeline work on
your own machine WITHOUT internet and WITHOUT waiting for a real 8-year
download — and so the demo reliably produces a TRADE verdict (real data
often, and correctly, produces NO_TRADE).

    python examples/make_demo_calibration.py

Then send a bullish DEMO alert to the webhook bridge and run
``run_weekly_from_tv.py`` (see deploy/README.md and the local-test guide).

DO NOT trade on this table — the edge in it is invented. For real use, run
``python signal_calibration.py AAPL --years 8`` against live market data.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# Make the repo root importable whether run from root or examples/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import signal_calibration as sc  # noqa: E402


def build(ticker: str = "DEMO", years: int = 10, seed: int = 42,
          calibration_dir: str = "calibration") -> Path:
    rng = np.random.default_rng(seed)
    n = 252 * years
    daily_sigma = 0.10 / math.sqrt(252)      # ~10% annual vol
    prices = [100.0]
    for _ in range(n):
        arr = np.asarray(prices)
        fast = sc.ema(arr, 9)[-1]
        slow = sc.ema(arr, 21)[-1]
        bullish = np.isfinite(fast) and np.isfinite(slow) and fast > slow
        # +25%/yr drift only while bullish, else flat — a clean, learnable edge.
        mu = (0.25 / 252) if bullish else 0.0
        prices.append(prices[-1] * math.exp(rng.normal(mu, daily_sigma)))
    table = sc.calibrate(ticker, years=years, horizon_days=5,
                         prices=np.asarray(prices))
    path = table.save(calibration_dir)
    return path


if __name__ == "__main__":
    out = build()
    print(f"Wrote fake practice calibration -> {out}")
    print("This is SYNTHETIC data for testing only. Do not trade on it.")
