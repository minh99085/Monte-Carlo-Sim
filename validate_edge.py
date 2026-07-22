#!/usr/bin/env python3
"""
validate_edge.py — brutal, paper-only edge validation for the signal-
conditioned drift strategy. Measures; does not tune. The honest result may
be "the edge does not survive."

WHAT THIS ANSWERS (see VALIDATION.md, produced by --write)
----------------------------------------------------------
1. EXECUTABLE FILLS   gross edge filled at the signal-bar close vs. the next
                      executable price (signal Friday close -> Monday open).
2. FULL COST MODEL    net edge after round-trip cost (0.10/0.20/0.40% a side)
                      and after a short-term cap-gains tax drag.
3. BENCHMARK          strategy (net of cost+tax) vs QQQ / SPY buy-and-hold on
                      Sharpe, Sortino, max drawdown, CAGR — with the honest
                      tax asymmetry (buy-hold is untaxed until sale).
4. REGIME / DOWNSIDE  results by calendar year and isolated to benchmark
                      drawdowns >10% (incl. 2022): is there any bear evidence?
5. STATISTICAL        (a) true correlation of the two "independent" lenses and
                      the agreement filter's marginal lift; (b) the effective
                      number of independent bets behind the 20-trade
                      kill-switch; (c) bootstrap CI on the edge and Kelly at
                      the lower bound / half-edge / zero-edge.
6. VERDICT            one honest paragraph: does the edge survive?

METHOD NOTES (why this is out-of-sample and not self-fulfilling)
----------------------------------------------------------------
* WALK-FORWARD. At each weekly decision the bucket table is re-fit using only
  data available up to that date (expanding window). A day is traded only if
  its bucket's shrunk drift is in the trade direction AND the 12-1 momentum
  lens agrees — exactly the deployed rule. No future information decides a
  trade. This is the honest test the live paper log cannot run (the paper log
  settles at clean closes and never re-derives point-in-time).
* NON-OVERLAPPING. Weekly cadence == the 5-day horizon, so trade windows do
  not overlap in time for a single name (overlap across names is handled
  explicitly in item 5b).
* NO TUNING. Every threshold is inherited from signal_calibration
  (shrinkage, hard-zero |t|<1). Nothing here is fit to improve the result.

DATA: requires real daily OHLC (Open needed for executable fills). Uses
yfinance when available; injectable for tests. This module does NOT place or
simulate any live order.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from signal_calibration import (
    EMA_SLOW_LEN,
    MOM_LOOKBACK_DAYS,
    TRADING_DAYS_PER_YEAR,
    compute_features,
    momentum_12_1,
    newey_west_stats,
    shrink_mu,
)

WEEK = 5  # trading days: horizon and rebalance cadence


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class OHLC:
    dates: pd.DatetimeIndex
    open: np.ndarray
    close: np.ndarray

    def __len__(self) -> int:
        return int(self.close.size)


def fetch_ohlc(ticker: str, years: float = 8.0) -> OHLC:
    """Daily adjusted Open/Close from yfinance. Raises on no data."""
    import yfinance as yf

    period_years = max(1, int(math.ceil(years)))
    data = yf.download(ticker, period=f"{period_years}y", progress=False,
                       auto_adjust=True)
    if data is None or len(data) < MOM_LOOKBACK_DAYS + 3 * WEEK:
        raise RuntimeError(f"insufficient OHLC for {ticker!r}")
    op = data["Open"]
    cl = data["Close"]
    if hasattr(op, "columns"):
        op = op.iloc[:, 0]
    if hasattr(cl, "columns"):
        cl = cl.iloc[:, 0]
    df = pd.DataFrame({"open": op, "close": cl}).dropna()
    return OHLC(dates=pd.DatetimeIndex(df.index),
                open=df["open"].to_numpy(float),
                close=df["close"].to_numpy(float))


# ---------------------------------------------------------------------------
# Point-in-time verified-bucket decision (mirrors the deployed rule)
# ---------------------------------------------------------------------------


def _bucket_shrunk_drift(fwd: np.ndarray, mask: np.ndarray,
                         horizon_days: int = WEEK) -> float:
    """Shrunk forward-return drift for the days selected by ``mask`` (uses the
    same Newey-West + shrinkage contract as calibration)."""
    sample = fwd[mask & np.isfinite(fwd)]
    st = newey_west_stats(sample, lag=horizon_days - 1)
    raw = st["mean"] if math.isfinite(st["mean"]) else 0.0
    return shrink_mu(raw, st["n_eff"], st["t_stat"])


@dataclass
class Trade:
    entry_date: pd.Timestamp
    side: str                       # "long" / "short"
    bucket: str
    r_signal_close: float           # signed simple return, signal-close fill
    r_executable: float             # signed simple return, next-open fill
    year: int
    both_lenses: bool               # agreement filter passed


def walk_forward_trades(
    ohlc: OHLC,
    *,
    horizon_days: int = WEEK,
    min_train_days: int = MOM_LOOKBACK_DAYS + 60,
    require_agreement: bool = True,
) -> List[Trade]:
    """Replay the strategy at the given holding horizon with point-in-time
    calibration; record each trade's signal-close vs. executable-fill return.
    Rebalance cadence == ``horizon_days`` so trades never overlap in time."""
    h = int(horizon_days)
    close = ohlc.close
    op = ohlc.open
    n = close.size
    feats = compute_features(close)            # trend/rsi/bucket per day (causal)
    buckets = feats["bucket"].to_numpy(dtype=object)
    trends = feats["trend"].to_numpy(dtype=object)
    mom = momentum_12_1(close)
    mom_sign = np.where(np.isfinite(mom), np.sign(mom), np.nan)

    log_c = np.log(close)
    trades: List[Trade] = []
    # Decide at t (a "Friday" close); need open[t+1] and open[t+1+h].
    t = max(min_train_days, EMA_SLOW_LEN + h + 2)
    while t + h + 1 < n:
        bkt = buckets[t]
        trend = trends[t]
        if isinstance(bkt, str) and isinstance(trend, str):
            side = "long" if trend == "bullish" else "short"
            # Point-in-time forward returns available strictly before t.
            fwd = np.full(t, np.nan)
            if t > h:
                fwd[:t - h] = log_c[h:t] - log_c[:t - h]
            bmask = np.array([buckets[i] == bkt for i in range(t)], dtype=bool)
            drift = _bucket_shrunk_drift(fwd, bmask, horizon_days=h)
            edge_dir = drift if side == "long" else -drift
            # Momentum lens agreement (point-in-time sign bucket at t).
            mom_ok = True
            if require_agreement:
                ms = mom_sign[t]
                mom_side = ("long" if ms > 0 else "short") if np.isfinite(ms) else None
                mom_ok = (mom_side == side)
            if edge_dir > 0.0 and (mom_ok or not require_agreement):
                # signal-close fill: close[t] -> close[t+h]
                rc = close[t + h] / close[t] - 1.0
                # executable fill: open[t+1] -> open[t+1+h]
                re = op[t + 1 + h] / op[t + 1] - 1.0
                if side == "short":
                    rc, re = -rc, -re
                trades.append(Trade(
                    entry_date=ohlc.dates[t], side=side, bucket=bkt,
                    r_signal_close=float(rc), r_executable=float(re),
                    year=int(ohlc.dates[t].year),
                    both_lenses=bool(mom_ok),
                ))
        t += h
    return trades


# ---------------------------------------------------------------------------
# Short-term reversal signal (a genuinely different bet from trend-following)
# ---------------------------------------------------------------------------
# Pre-specified, NOT tuned (fixing the parameters up front avoids fishing a
# threshold that flatters the backtest): enter LONG when a name's short
# lookback return is oversold by more than Z_THRESH trailing sigmas, hold
# HORIZON days, one position per name at a time (non-overlapping). Long-only
# to match a cash account. This is the "buy the dip, many small wins"
# profile — its risk lives in the tail (a dip that keeps falling), which the
# confirm report surfaces as avg/worst loss.
REVERSAL_LOOKBACK = 3        # days of the oversold move
REVERSAL_Z_THRESH = 1.5      # trailing-sigma threshold for "oversold"
REVERSAL_VOL_WINDOW = 60     # trailing window for the daily-vol estimate


def reversal_trades(
    ohlc: OHLC,
    *,
    horizon_days: int = 5,
    lookback: int = REVERSAL_LOOKBACK,
    z_thresh: float = REVERSAL_Z_THRESH,
    vol_window: int = REVERSAL_VOL_WINDOW,
    min_train_days: int = REVERSAL_VOL_WINDOW + 10,
    require_agreement: bool = False,   # signature parity with walk_forward_trades
) -> List[Trade]:
    """Long-only short-term reversal, point-in-time, non-overlapping."""
    h, k = int(horizon_days), int(lookback)
    close, op, n = ohlc.close, ohlc.open, ohlc.close.size
    logret = np.diff(np.log(close), prepend=np.log(close[0]))  # [0]=0
    trades: List[Trade] = []
    t = max(int(min_train_days), vol_window + k + 2)
    while t + h + 1 < n:
        rk = close[t] / close[t - k] - 1.0
        window = logret[t - vol_window:t]
        sd = float(np.std(window)) if window.size else 0.0
        z = rk / (sd * math.sqrt(k)) if sd > 0 else 0.0
        if z < -z_thresh:                       # oversold → long the bounce
            rc = close[t + h] / close[t] - 1.0
            re = op[t + 1 + h] / op[t + 1] - 1.0
            trades.append(Trade(
                entry_date=ohlc.dates[t], side="long", bucket="oversold",
                r_signal_close=float(rc), r_executable=float(re),
                year=int(ohlc.dates[t].year), both_lenses=True))
            t += h                              # hold to exit before re-entering
        else:
            t += 1
    return trades


TRADE_GENERATORS: Dict[str, Callable[..., List[Trade]]] = {
    "trend": walk_forward_trades,
    "reversal": reversal_trades,
}


# ---------------------------------------------------------------------------
# Item 1 + 2: edge, costs, tax
# ---------------------------------------------------------------------------


def _annualize_per_trade(mean_per_trade: float, trades_per_year: float) -> float:
    return mean_per_trade * trades_per_year


def cost_tax_table(
    trades: Sequence[Trade],
    *,
    cost_sides: Sequence[float] = (0.0005, 0.0010, 0.0020, 0.0040),
    tax_rate: float = 0.35,
    trades_per_year: float = 52.0,
    fill: str = "executable",
) -> List[Dict[str, float]]:
    """Per-trade gross -> net after round-trip cost -> after tax, per cost level.
    Returns one row per cost level with annualized figures."""
    r = np.array([t.r_executable if fill == "executable" else t.r_signal_close
                  for t in trades], dtype=float)
    rows: List[Dict[str, float]] = []
    for c in cost_sides:
        # multiplicative round trip (enter and exit each pay c)
        net = (1.0 + r) * (1.0 - c) * (1.0 - c) - 1.0
        after_tax = net - tax_rate * np.maximum(net, 0.0)
        rows.append({
            "cost_per_side": c,
            "gross_mean": float(np.mean(r)) if r.size else 0.0,
            "net_mean": float(np.mean(net)) if net.size else 0.0,
            "after_tax_mean": float(np.mean(after_tax)) if net.size else 0.0,
            "gross_annual": _annualize_per_trade(
                float(np.mean(r)) if r.size else 0.0, trades_per_year),
            "net_annual": _annualize_per_trade(
                float(np.mean(net)) if net.size else 0.0, trades_per_year),
            "after_tax_annual": _annualize_per_trade(
                float(np.mean(after_tax)) if net.size else 0.0, trades_per_year),
            "n_trades": int(r.size),
        })
    return rows


# ---------------------------------------------------------------------------
# Item 3: performance stats + benchmark
# ---------------------------------------------------------------------------


def resample_drawdown(period_returns: np.ndarray, *, n_boot: int = 5000,
                      seed: int = 13) -> Dict[str, Any]:
    """Reshuffle the trade/period return SEQUENCE many times to get the
    distribution of possible drawdowns (the dev.to 'Monte Carlo on trade
    results' method). Same total return, different path each time — the
    single backtest ordering is just one lucky draw. Reported drawdowns are
    negative; more-negative = worse.

    Reshuffling assumes the sampled units are exchangeable; here the units are
    per-period PORTFOLIO returns (already bundling correlated same-period
    names), so this captures sequence/streak risk. Regime clustering could
    make reality worse still — treat the tail here as a floor, not a ceiling.
    """
    r = np.asarray(period_returns, dtype=float)
    r = r[np.isfinite(r)]

    def maxdd(x: np.ndarray) -> float:
        eq = np.cumprod(1.0 + x)
        peak = np.maximum.accumulate(eq)
        return float((eq / peak - 1.0).min()) if x.size else 0.0

    if r.size < 3:
        bt = maxdd(r)
        return {"backtest_dd": bt, "median_dd": bt, "p95_dd": bt,
                "worst_dd": bt, "n": int(r.size)}
    backtest = maxdd(r)
    rng = np.random.default_rng(seed)
    dds = np.array([maxdd(rng.permutation(r)) for _ in range(n_boot)])
    return {
        "backtest_dd": backtest,
        "median_dd": float(np.median(dds)),
        "p95_dd": float(np.quantile(dds, 0.05)),   # exceeded only ~5% of the time
        "worst_dd": float(dds.min()),
        "n": int(r.size),
    }


def perf_stats(period_returns: np.ndarray, periods_per_year: float) -> Dict[str, float]:
    """Sharpe / Sortino / maxDD / CAGR from a series of per-period simple
    returns (rf = 0)."""
    r = np.asarray(period_returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"sharpe": float("nan"), "sortino": float("nan"),
                "max_drawdown": 0.0, "cagr": 0.0, "n": 0,
                "total_return": 0.0}
    mean = float(np.mean(r))
    sd = float(np.std(r, ddof=1)) if r.size > 1 else 0.0
    downside = r[r < 0.0]
    dsd = float(np.sqrt(np.mean(downside ** 2))) if downside.size else 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    total = float(equity[-1] - 1.0)
    years = r.size / periods_per_year
    cagr = (equity[-1] ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else 0.0
    return {
        "sharpe": (mean / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan"),
        "sortino": (mean / dsd * math.sqrt(periods_per_year)) if dsd > 0 else float("nan"),
        "max_drawdown": float(dd.min()),
        "cagr": float(cagr),
        "total_return": total,
        "n": int(r.size),
    }


def portfolio_weekly_returns(
    per_ticker_trades: Dict[str, List[Trade]],
    *,
    cost_side: float,
    tax_rate: float,
    fill: str = "executable",
) -> np.ndarray:
    """Equal-weight the (after-cost, after-tax) trade returns by entry week
    across names; weeks with no trade contribute 0 (cash)."""
    by_week: Dict[pd.Timestamp, List[float]] = {}
    for _, trades in per_ticker_trades.items():
        for t in trades:
            r = t.r_executable if fill == "executable" else t.r_signal_close
            net = (1.0 + r) * (1.0 - cost_side) ** 2 - 1.0
            at = net - tax_rate * max(net, 0.0)
            key = pd.Timestamp(t.entry_date).normalize()
            by_week.setdefault(key, []).append(at)
    if not by_week:
        return np.array([], dtype=float)
    weeks = sorted(by_week)
    return np.array([float(np.mean(by_week[w])) for w in weeks], dtype=float)


def benchmark_buy_hold(
    ohlc: OHLC,
    *,
    long_term_tax: float = 0.15,
) -> Dict[str, Any]:
    """Buy-and-hold daily-return stats, pre and post a single terminal
    long-term cap-gains tax on the total gain (the honest asymmetry vs a
    weekly-churn strategy taxed every trade at the ordinary rate)."""
    r = ohlc.close[1:] / ohlc.close[:-1] - 1.0
    pre = perf_stats(r, TRADING_DAYS_PER_YEAR)
    total_gain = ohlc.close[-1] / ohlc.close[0] - 1.0
    after_tax_total = total_gain - long_term_tax * max(total_gain, 0.0)
    years = len(ohlc) / TRADING_DAYS_PER_YEAR
    at_cagr = ((1.0 + after_tax_total) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    return {
        "pre_tax": pre,
        "after_tax_total_return": float(after_tax_total),
        "after_tax_cagr": float(at_cagr),
        "total_return": float(total_gain),
        "start": str(ohlc.dates[0].date()),
        "end": str(ohlc.dates[-1].date()),
    }


# ---------------------------------------------------------------------------
# Item 4: regime / downside
# ---------------------------------------------------------------------------


def drawdown_series(close: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(close)
    return close / peak - 1.0


def regime_segmentation(
    per_ticker_trades: Dict[str, List[Trade]],
    benchmark: OHLC,
    *,
    cost_side: float = 0.0020,
    tax_rate: float = 0.35,
    dd_threshold: float = -0.10,
) -> Dict[str, Any]:
    """Trade stats by calendar year and inside benchmark drawdowns > 10%."""
    all_trades = [t for ts in per_ticker_trades.values() for t in ts]
    # by year
    years: Dict[int, List[float]] = {}
    for t in all_trades:
        r = t.r_executable
        net = (1.0 + r) * (1.0 - cost_side) ** 2 - 1.0
        years.setdefault(t.year, []).append(net)
    by_year = {
        y: {"n": len(v), "mean_net": float(np.mean(v)),
            "hit_rate": float(np.mean(np.array(v) > 0)),
            "total": float(np.prod(1.0 + np.array(v)) - 1.0)}
        for y, v in sorted(years.items())
    }
    # benchmark drawdown mask by date
    dd = drawdown_series(benchmark.close)
    date_to_dd = {pd.Timestamp(d).normalize(): dd[i]
                  for i, d in enumerate(benchmark.dates)}
    down_returns: List[float] = []
    for t in all_trades:
        key = pd.Timestamp(t.entry_date).normalize()
        d = date_to_dd.get(key)
        if d is not None and d <= dd_threshold:
            r = t.r_executable
            down_returns.append((1.0 + r) * (1.0 - cost_side) ** 2 - 1.0)
    down = np.array(down_returns, dtype=float)
    return {
        "by_year": by_year,
        "down_tape": {
            "n": int(down.size),
            "mean_net": float(np.mean(down)) if down.size else None,
            "hit_rate": float(np.mean(down > 0)) if down.size else None,
            "total": float(np.prod(1.0 + down) - 1.0) if down.size else None,
            "dd_threshold": dd_threshold,
        },
    }


# ---------------------------------------------------------------------------
# Item 5a: lens correlation + agreement marginal lift
# ---------------------------------------------------------------------------


def lens_correlation(ohlc: OHLC) -> Dict[str, Any]:
    """Correlation of the two 'independent' lenses and sign-agreement rate.

    Lens 1 (trend): (ema_fast - ema_slow) / ema_slow — continuous.
    Lens 2 (momentum): 12-1 momentum — continuous.
    """
    feats = compute_features(ohlc.close)
    ef = feats["ema_fast"].to_numpy(float)
    es = feats["ema_slow"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s1 = (ef - es) / es
    s2 = momentum_12_1(ohlc.close)
    m = np.isfinite(s1) & np.isfinite(s2)
    s1m, s2m = s1[m], s2[m]
    if s1m.size < 30:
        return {"pearson": None, "sign_agreement": None, "n": int(s1m.size)}
    pear = float(np.corrcoef(s1m, s2m)[0, 1])
    sign_agree = float(np.mean(np.sign(s1m) == np.sign(s2m)))
    return {"pearson": pear, "sign_agreement": sign_agree, "n": int(s1m.size)}


def agreement_marginal_lift(trades_single: List[Trade],
                            trades_agree: List[Trade]) -> Dict[str, Any]:
    """Compare forward return + hit-rate of single-lens-verified trades vs.
    trades where both lenses agree. Marginal lift = agree - single."""
    def stat(ts: List[Trade]) -> Dict[str, float]:
        r = np.array([t.r_executable for t in ts], dtype=float)
        return {"n": int(r.size),
                "mean": float(np.mean(r)) if r.size else float("nan"),
                "hit_rate": float(np.mean(r > 0)) if r.size else float("nan")}
    a = stat(trades_agree)
    s = stat(trades_single)
    return {
        "single_lens": s,
        "both_agree": a,
        "marginal_mean_lift": (a["mean"] - s["mean"])
        if (a["n"] and s["n"]) else None,
        "marginal_hit_lift": (a["hit_rate"] - s["hit_rate"])
        if (a["n"] and s["n"]) else None,
    }


# ---------------------------------------------------------------------------
# Item 5b: effective number of independent bets
# ---------------------------------------------------------------------------


def effective_n(per_ticker_trades: Dict[str, List[Trade]],
                window: int = 20) -> Dict[str, Any]:
    """N_effective behind the last ``window`` trades, discounting cross-
    sectional correlation of concurrent names. Uses the average pairwise
    correlation of the tickers' trade-return series; for N correlated bets
    N_eff = N / (1 + (N-1) * rho_bar)."""
    # Build aligned per-week return matrix across tickers.
    weeks: Dict[pd.Timestamp, Dict[str, float]] = {}
    for tk, trades in per_ticker_trades.items():
        for t in trades:
            weeks.setdefault(pd.Timestamp(t.entry_date).normalize(), {})[tk] = t.r_executable
    tickers = sorted(per_ticker_trades)
    if len(tickers) < 2:
        return {"rho_bar": 0.0, "n_window": window, "n_eff": float(window),
                "note": "single ticker — cross-sectional correlation N/A"}
    week_keys = sorted(weeks)
    mat = np.full((len(week_keys), len(tickers)), np.nan)
    for i, wk in enumerate(week_keys):
        for j, tk in enumerate(tickers):
            if tk in weeks[wk]:
                mat[i, j] = weeks[wk][tk]
    # pairwise correlations on overlapping weeks
    corrs: List[float] = []
    for a in range(len(tickers)):
        for b in range(a + 1, len(tickers)):
            col_a, col_b = mat[:, a], mat[:, b]
            ok = np.isfinite(col_a) & np.isfinite(col_b)
            if ok.sum() >= 8:
                c = np.corrcoef(col_a[ok], col_b[ok])[0, 1]
                if math.isfinite(c):
                    corrs.append(float(c))
    rho_bar = float(np.mean(corrs)) if corrs else 0.0
    n = window
    n_eff = n / (1.0 + (n - 1) * max(rho_bar, 0.0)) if rho_bar > 0 else float(n)
    return {"rho_bar": rho_bar, "n_window": n, "n_eff": float(n_eff),
            "n_pairs": len(corrs)}


# ---------------------------------------------------------------------------
# Item 5c: bootstrap CI on edge + Kelly under uncertainty
# ---------------------------------------------------------------------------


def bootstrap_edge_ci(returns: np.ndarray, *, n_boot: int = 5000,
                      alpha: float = 0.05, seed: int = 7) -> Dict[str, float]:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 5:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n": int(r.size)}
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(r, r.size, replace=True))
                      for _ in range(n_boot)])
    return {
        "mean": float(np.mean(r)),
        "lo": float(np.quantile(means, alpha / 2)),
        "hi": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(r.size),
    }


def kelly_under_uncertainty(returns: np.ndarray,
                            *, current_fraction: float = 0.25) -> Dict[str, Any]:
    """Kelly fraction f* ≈ mean/variance for the per-trade return
    distribution, evaluated at the point estimate, the bootstrap lower bound,
    half-edge, and zero-edge. Compares to the deployed quarter-Kelly."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 5:
        return {"note": "too few trades", "n": int(r.size)}
    var = float(np.var(r, ddof=1))
    ci = bootstrap_edge_ci(r)
    mean = ci["mean"]

    def f_star(edge: float) -> float:
        return (edge / var) if var > 0 else 0.0

    f_point = f_star(mean)
    f_lo = f_star(ci["lo"])
    f_half = f_star(mean / 2.0)
    f_zero = 0.0
    deployed = current_fraction * f_point           # quarter-Kelly of point est
    # If the TRUE edge were half the point estimate, the deployed size is this
    # multiple of the correct quarter-Kelly:
    correct_if_half = current_fraction * f_half
    overbet_if_half = (deployed / correct_if_half) if correct_if_half > 0 else float("inf")
    return {
        "n": int(r.size),
        "variance": var,
        "edge_point": mean,
        "edge_ci_lo": ci["lo"],
        "edge_ci_hi": ci["hi"],
        "f_star_point": f_point,
        "f_star_ci_lo": f_lo,
        "f_star_half": f_half,
        "f_star_zero": f_zero,
        "deployed_quarter_kelly": deployed,
        "overbet_factor_if_true_edge_half": overbet_if_half,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(x: Optional[float], nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    return f"{100 * x:+.{nd}f}%"


def run_full(
    tickers: Sequence[str],
    *,
    benchmark: str = "QQQ",
    years: float = 8.0,
    tax_rate: float = 0.35,
    fetch: Callable[[str, float], OHLC] = fetch_ohlc,
) -> Dict[str, Any]:
    per_ticker_trades: Dict[str, List[Trade]] = {}
    ohlc_by_ticker: Dict[str, OHLC] = {}
    for tk in tickers:
        ohlc = fetch(tk, years)
        ohlc_by_ticker[tk] = ohlc
        per_ticker_trades[tk] = walk_forward_trades(ohlc)
    bench = fetch(benchmark, years)

    all_trades = [t for ts in per_ticker_trades.values() for t in ts]
    r_exec = np.array([t.r_executable for t in all_trades], dtype=float)
    r_close = np.array([t.r_signal_close for t in all_trades], dtype=float)

    # portfolio equity (net + tax) at the middle cost level
    wk = portfolio_weekly_returns(per_ticker_trades, cost_side=0.0020,
                                  tax_rate=tax_rate)
    strat = perf_stats(wk, 52.0)

    # 5a marginal lift: rebuild single-lens (no agreement) trade set
    single_trades: List[Trade] = []
    for tk in tickers:
        single_trades.extend(
            walk_forward_trades(ohlc_by_ticker[tk], require_agreement=False))

    return {
        "tickers": list(tickers),
        "benchmark": benchmark,
        "n_trades": len(all_trades),
        "edge": {
            "signal_close_mean": float(np.mean(r_close)) if r_close.size else 0.0,
            "executable_mean": float(np.mean(r_exec)) if r_exec.size else 0.0,
            "signal_close_annual": (float(np.mean(r_close)) * 52) if r_close.size else 0.0,
            "executable_annual": (float(np.mean(r_exec)) * 52) if r_exec.size else 0.0,
            "fill_decay": float(np.mean(r_close) - np.mean(r_exec)) if r_exec.size else 0.0,
        },
        "cost_tax": cost_tax_table(all_trades, tax_rate=tax_rate),
        "strategy_perf": strat,
        "benchmark_perf": benchmark_buy_hold(bench),
        "regime": regime_segmentation(per_ticker_trades, bench),
        "lens_corr": (lens_correlation(ohlc_by_ticker[tickers[0]])
                      if tickers else {}),
        "agreement_lift": agreement_marginal_lift(single_trades, all_trades),
        "effective_n": effective_n(per_ticker_trades),
        "kelly": kelly_under_uncertainty(r_exec),
    }


def tax_for_horizon(horizon_days: int, short_rate: float,
                    long_rate: float) -> float:
    """Long-term cap-gains rate applies only when the position is held longer
    than one year (~252 trading days); otherwise the ordinary short-term rate."""
    return long_rate if horizon_days >= 252 else short_rate


def run_turnover_sweep(
    tickers: Sequence[str],
    *,
    horizons: Sequence[int] = (5, 21, 63, 126, 252),
    benchmark: str = "QQQ",
    years: float = 8.0,
    cost_side: float = 0.0020,
    short_tax: float = 0.35,
    long_tax: float = 0.15,
    fetch: Callable[[str, float], OHLC] = fetch_ohlc,
) -> Dict[str, Any]:
    """Test the Option-B hypothesis: does the (real, pre-cost) signal survive
    once you trade LESS often? Sweeps holding horizons and reports, for each,
    the after-cost/after-tax annualized edge and the strategy Sharpe vs the
    benchmark. Trading frequency = 252/horizon, so cost drag falls with longer
    holds, and holds >1yr get the long-term tax rate.

    HONEST CAVEAT baked into the output: testing several horizons is itself a
    multiple-comparisons risk. A single horizon squeaking past is NOT proof.
    """
    ohlc_by_ticker = {tk: fetch(tk, years) for tk in tickers}
    bench = fetch(benchmark, years)
    bench_stats = benchmark_buy_hold(bench)
    rows: List[Dict[str, Any]] = []
    for h in horizons:
        per_ticker = {tk: walk_forward_trades(ohlc_by_ticker[tk], horizon_days=h)
                      for tk in tickers}
        all_tr = [t for ts in per_ticker.values() for t in ts]
        tpy = TRADING_DAYS_PER_YEAR / float(h)         # trades per year
        tax = tax_for_horizon(h, short_tax, long_tax)
        ct = cost_tax_table(all_tr, cost_sides=[cost_side], tax_rate=tax,
                            trades_per_year=tpy)[0]
        wk = portfolio_weekly_returns(per_ticker, cost_side=cost_side,
                                      tax_rate=tax)
        perf = perf_stats(wk, tpy)
        beats = (perf["sharpe"] is not None
                 and math.isfinite(perf["sharpe"])
                 and math.isfinite(bench_stats["pre_tax"]["sharpe"])
                 and perf["sharpe"] > bench_stats["pre_tax"]["sharpe"])
        rows.append({
            "horizon_days": h,
            "trades_per_year": tpy,
            "n_trades": ct["n_trades"],
            "tax_rate": tax,
            "gross_annual": ct["gross_annual"],
            "net_annual": ct["net_annual"],
            "after_tax_annual": ct["after_tax_annual"],
            "sharpe": perf["sharpe"],
            "cagr": perf["cagr"],
            "max_drawdown": perf["max_drawdown"],
            "beats_benchmark_riskadj": bool(beats),
        })
    any_survivor = any(r["after_tax_annual"] > 0 and r["beats_benchmark_riskadj"]
                       for r in rows)
    return {
        "tickers": list(tickers),
        "benchmark": benchmark,
        "benchmark_sharpe": bench_stats["pre_tax"]["sharpe"],
        "benchmark_cagr": bench_stats["pre_tax"]["cagr"],
        "cost_side": cost_side,
        "short_tax": short_tax,
        "long_tax": long_tax,
        "rows": rows,
        "any_survivor": any_survivor,
        "n_horizons_tested": len(horizons),
    }


def run_horizon_confirm(
    tickers: Sequence[str],
    *,
    horizon_days: int,
    benchmark: str = "QQQ",
    years: float = 8.0,
    cost_side: float = 0.0003,
    short_tax: float = 0.35,
    long_tax: float = 0.15,
    signal: str = "trend",
    fetch: Callable[[str, float], OHLC] = fetch_ohlc,
) -> Dict[str, Any]:
    """Stability confirmation for ONE pre-chosen horizon and signal.

    The sweep picked this horizon using the whole window — a multiple-testing
    risk. This check asks: does it beat the benchmark in BOTH halves of the
    period independently? A real edge should show up in each half; a lucky
    stretch shows up in one. Also reports the consistency profile the owner
    actually cares about: win rate together with the size of the losses.

    ``signal`` selects the trade generator: ``"trend"`` (the EMA/RSI+momentum
    strategy, annualized at the fixed 252/h cadence) or ``"reversal"`` (the
    oversold-bounce strategy, whose entries are irregular so it is annualized
    at the *actual* per-window trade frequency).
    """
    h = int(horizon_days)
    tax = tax_for_horizon(h, short_tax, long_tax)
    gen = TRADE_GENERATORS.get(signal, walk_forward_trades)
    per_ticker = {tk: gen(fetch(tk, years), horizon_days=h) for tk in tickers}
    bench = fetch(benchmark, years)
    all_trades = sorted((t for ts in per_ticker.values() for t in ts),
                        key=lambda t: t.entry_date)
    if len(all_trades) < 8:
        return {"error": f"only {len(all_trades)} trades — too few to split"}
    mid_date = all_trades[len(all_trades) // 2].entry_date

    def half_stats(trades: List[Trade], bench_ohlc: OHLC,
                   date_lo, date_hi) -> Dict[str, Any]:
        per = {tk: [t for t in ts if date_lo <= t.entry_date < date_hi]
               for tk, ts in per_ticker.items()}
        wk = portfolio_weekly_returns(per, cost_side=cost_side, tax_rate=tax)
        if signal == "trend":
            ppy = TRADING_DAYS_PER_YEAR / h        # fixed cadence
        else:
            # irregular entries → annualize at the actual entry frequency
            span_years = max((pd.Timestamp(date_hi) - pd.Timestamp(date_lo))
                             .days / 365.25, 1e-9)
            ppy = max(len(wk) / span_years, 1e-9)
        st = perf_stats(wk, ppy)
        b_mask = (bench_ohlc.dates >= date_lo) & (bench_ohlc.dates < date_hi)
        b_close = bench_ohlc.close[np.asarray(b_mask)]
        b_r = b_close[1:] / b_close[:-1] - 1.0 if b_close.size > 2 else np.array([])
        b_st = perf_stats(b_r, TRADING_DAYS_PER_YEAR)
        nets = [(1.0 + t.r_executable) * (1.0 - cost_side) ** 2 - 1.0
                for t in trades]
        nets_arr = np.array(nets, dtype=float)
        losses = np.sort(nets_arr[nets_arr < 0])
        return {
            "n_trades": len(trades),
            "strategy": st,
            "benchmark": b_st,
            "beats": bool(math.isfinite(st["sharpe"])
                          and math.isfinite(b_st["sharpe"])
                          and st["sharpe"] > b_st["sharpe"]),
            "win_rate": float(np.mean(nets_arr > 0)) if nets_arr.size else None,
            "avg_win": float(np.mean(nets_arr[nets_arr > 0]))
            if (nets_arr > 0).any() else None,
            "avg_loss": float(np.mean(losses)) if losses.size else None,
            "worst_losses": [float(x) for x in losses[:5]],
        }

    lo = min(t.entry_date for t in all_trades)
    hi = max(t.entry_date for t in all_trades) + pd.Timedelta(days=1)
    first = half_stats([t for t in all_trades if t.entry_date < mid_date],
                       bench, lo, mid_date)
    second = half_stats([t for t in all_trades if t.entry_date >= mid_date],
                        bench, mid_date, hi)
    overall = half_stats(all_trades, bench, lo, hi)
    # Trade-sequence Monte Carlo on the full portfolio path: the honest
    # drawdown you should size against, not the single lucky ordering.
    overall_wk = portfolio_weekly_returns(per_ticker, cost_side=cost_side,
                                          tax_rate=tax)
    dd_mc = resample_drawdown(overall_wk)
    return {
        "tickers": list(tickers),
        "benchmark": benchmark,
        "horizon_days": h,
        "signal": signal,
        "cost_side": cost_side,
        "tax_rate": tax,
        "split_date": str(pd.Timestamp(mid_date).date()),
        "first_half": first,
        "second_half": second,
        "overall": overall,
        "drawdown_mc": dd_mc,
        "stable": bool(first["beats"] and second["beats"]),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "NVDA", "AAPL"])
    ap.add_argument("--benchmark", default="QQQ")
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--tax-rate", type=float, default=0.35)
    ap.add_argument("--out", default="VALIDATION.md")
    ap.add_argument("--sweep", action="store_true",
                    help="Option B: sweep holding horizons (weekly→annual) to "
                         "test whether trading LESS often survives cost+tax.")
    ap.add_argument("--horizons", type=int, nargs="+",
                    default=[5, 21, 63, 126, 252],
                    help="holding horizons in trading days (with --sweep)")
    ap.add_argument("--cost-side", type=float, default=0.0020,
                    help="one-way trading cost as a fraction (default 0.0020 "
                         "= 0.20%%). Robinhood is commission-free; for liquid "
                         "ETFs/megacaps the realistic cost is the half-spread, "
                         "~0.0002-0.0005 (0.02-0.05%%).")
    ap.add_argument("--confirm", type=int, default=None, metavar="HORIZON",
                    help="stability check for ONE horizon (trading days): "
                         "must beat the benchmark in BOTH halves of history "
                         "independently, else the sweep pick was luck.")
    ap.add_argument("--signal", default="trend", choices=list(TRADE_GENERATORS),
                    help="which strategy to test: 'trend' (EMA/RSI+momentum) "
                         "or 'reversal' (oversold-bounce). Default trend.")
    args = ap.parse_args(argv)

    try:
        if args.confirm:
            res = run_horizon_confirm(
                args.tickers, horizon_days=args.confirm,
                benchmark=args.benchmark, years=args.years,
                cost_side=args.cost_side, short_tax=args.tax_rate,
                signal=args.signal)
            from validation_report import write_confirm_report
            path = write_confirm_report(res, Path(args.out))
            print(f"Wrote {path}")
            return 0
        if args.sweep:
            sweep = run_turnover_sweep(
                args.tickers, horizons=args.horizons, benchmark=args.benchmark,
                years=args.years, short_tax=args.tax_rate,
                cost_side=args.cost_side)
            from validation_report import write_turnover_report
            path = write_turnover_report(sweep, Path(args.out))
            print(f"Wrote {path}")
            return 0
        result = run_full(args.tickers, benchmark=args.benchmark,
                          years=args.years, tax_rate=args.tax_rate)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: validation could not fetch/compute: {exc}", file=sys.stderr)
        print("This needs real OHLC (yfinance). Run where market data is "
              "reachable (the VPS), not the sandbox.", file=sys.stderr)
        return 2
    from validation_report import write_report
    path = write_report(result, Path(args.out), tax_rate=args.tax_rate)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
