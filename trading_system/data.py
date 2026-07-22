"""Market data adapter (HAND read side).

Target interface is the Robinhood MCP tool set (get_equity_historicals,
get_equity_quotes, ...) per the rebuild spec. That endpoint needs the bot's
OAuth session, which is not available to offline research runs, so the
default backend is yfinance with the SAME shape — swap ``fetch_bars`` when
the authenticated path exists. Everything downstream depends only on
:class:`Bars`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Bars:
    """Daily OHLC (adjusted) for one symbol."""

    symbol: str
    dates: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    def __len__(self) -> int:
        return int(self.close.size)

    def slice(self, lo: int, hi: int) -> "Bars":
        return Bars(self.symbol, self.dates[lo:hi], self.open[lo:hi],
                    self.high[lo:hi], self.low[lo:hi], self.close[lo:hi])


def fetch_bars(symbol: str, years: float = 8.0) -> Bars:
    """Daily adjusted OHLC via yfinance. Raises RuntimeError on no data."""
    import yfinance as yf

    period_years = max(1, int(math.ceil(years)))
    df = yf.download(symbol, period=f"{period_years}y", progress=False,
                     auto_adjust=True)
    if df is None or len(df) < 300:
        raise RuntimeError(f"insufficient history for {symbol!r}")
    cols = {}
    for name in ("Open", "High", "Low", "Close"):
        col = df[name]
        if hasattr(col, "columns"):
            col = col.iloc[:, 0]
        cols[name.lower()] = col
    frame = pd.DataFrame(cols).dropna()
    return Bars(
        symbol=symbol.upper(),
        dates=pd.DatetimeIndex(frame.index),
        open=frame["open"].to_numpy(float),
        high=frame["high"].to_numpy(float),
        low=frame["low"].to_numpy(float),
        close=frame["close"].to_numpy(float),
    )


FetchFn = Callable[[str, float], Bars]


def realized_vol_daily(close: np.ndarray, window: int) -> np.ndarray:
    """Trailing daily log-return std; NaN during warm-up. Value at index t
    uses returns up to and including t (known at the close of t)."""
    logret = np.diff(np.log(close), prepend=np.nan)
    out = np.full(close.size, np.nan)
    for t in range(window, close.size):
        seg = logret[t - window + 1: t + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size >= max(5, window // 2):
            out[t] = float(np.std(seg))
    return out


def atr(bars: Bars, window: int = 14) -> np.ndarray:
    """Average True Range (simple mean of TR); NaN during warm-up."""
    h, l, c = bars.high, bars.low, bars.close
    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.stack([h - l,
                             np.abs(h - prev_c),
                             np.abs(l - prev_c)]), axis=0)
    out = np.full(len(bars), np.nan)
    for t in range(window, len(bars)):
        seg = tr[t - window + 1: t + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[t] = float(np.mean(seg))
    return out
