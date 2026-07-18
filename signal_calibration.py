"""
Signal-conditioned drift calibration for the TradingView pipeline.

Why this module exists (the root-cause fix)
-------------------------------------------
The TradingView signal (EMA 9/21 trend + RSI 14 momentum) used to be consumed
as ticker/price/side/vol-multiplier only — it was **never mapped to drift**, so
``annual_drift`` stayed at the preset default 0.0 and every simulated
P(profit) was a diffusion coin-flip minus costs. This module converts the
signal *state* into a calibrated, shrunken conditional drift estimated from
historical forward returns:

1. Replicate the Pine script indicators exactly on daily closes
   (EMA(9), EMA(21) trend; RSI(14) with Wilder smoothing, matching ``ta.rsi``).
2. Bucket each day into trend {bullish, bearish} x RSI terciles
   {low < 40, 40 <= mid <= 60, high > 60}.
3. For each bucket, measure the forward ``horizon_days``-day log return with
   **overlapping samples** and a **Newey-West standard error**
   (Bartlett kernel, lag = horizon_days - 1) so the autocorrelation created by
   overlapping windows does not overstate the sample size. The effective
   sample size is ``n_eff = n * naive_var / long_run_var``.
4. Shrink the raw bucket mean toward zero:
       shrunk_mu = raw_mu * n_eff / (n_eff + k)
   and hard-zero any bucket with |t| < 1.0. The prior strength ``k`` is
   adaptive (k = K_PRIOR + n_eff * T_PRIOR / t^2) so that a bucket needs
   |t| >~ 2 to retain most of its mean and thin buckets shrink to zero as
   n_eff -> 0 (see the constants' comment). When calibration is
   statistically weak, drift shrinks toward zero and the downstream
   pipeline outputs NO TRADE.
5. Persist the table to ``calibration/{TICKER}_{horizon}d.json`` with both
   raw and shrunk values, and load it back with staleness checks.

CLI
---
    python signal_calibration.py AAPL --years 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants (bucket edges and shrinkage are the contract with the pipeline)
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252

# Pine-script indicator lengths (must match tradingview_alert_template.pine)
EMA_FAST_LEN = 9
EMA_SLOW_LEN = 21
RSI_LEN = 14

# RSI tercile edges: low < 40, 40 <= mid <= 60, high > 60
RSI_LOW_MAX = 40.0
RSI_HIGH_MIN = 60.0

TREND_BULLISH = "bullish"
TREND_BEARISH = "bearish"
TRENDS = (TREND_BULLISH, TREND_BEARISH)
RSI_TERCILES = ("low", "mid", "high")
ALL_BUCKETS = tuple(f"{t}_{m}" for t in TRENDS for m in RSI_TERCILES)

# Shrinkage: shrunk_mu = raw_mu * n_eff / (n_eff + k) with an *adaptive*
# prior strength
#     k = K_PRIOR + n_eff * T_PRIOR / t**2
# so the retention factor is 1 / (1 + K_PRIOR/n_eff + T_PRIOR/t^2).
# * For well-populated buckets this approaches t^2 / (t^2 + T_PRIOR): with
#   T_PRIOR = 4 a bucket retains 50% of its mean at exactly |t| = 2 and ~70%
#   at |t| = 3 — i.e. it needs |t| >~ 2 to retain most of its mean, no matter
#   how many samples inflated a noise bucket past the hard-zero gate.
# * The K_PRIOR term crushes thin buckets regardless of their (unreliable)
#   t-stat: retention -> 0 as n_eff -> 0.
K_PRIOR = 60.0
T_PRIOR = 4.0

# Any bucket with |t| < 1 is statistically indistinguishable from zero at even
# the most permissive standard — hard-zero it so no noise leaks into drift.
HARD_ZERO_T = 1.0

DEFAULT_CALIBRATION_DIR = Path("calibration")
STALE_WARN_DAYS = 30
STALE_ERROR_DAYS = 120


class CalibrationError(Exception):
    """Base error for calibration problems."""


class CalibrationDataError(CalibrationError):
    """Could not obtain usable price history."""


class CalibrationStaleError(CalibrationError):
    """Stored calibration is too old to trust."""


# ---------------------------------------------------------------------------
# Pine-exact indicators (daily closes)
# ---------------------------------------------------------------------------


def ema(values: np.ndarray, length: int) -> np.ndarray:
    """Pine ``ta.ema``: alpha = 2/(length+1), seeded with the SMA of the
    first ``length`` values. Entries before the seed are NaN."""
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    out = np.full(n, np.nan)
    if n < length or length < 1:
        return out
    alpha = 2.0 / (length + 1.0)
    out[length - 1] = float(np.mean(x[:length]))
    for i in range(length, n):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def wilder_rma(values: np.ndarray, length: int) -> np.ndarray:
    """Pine ``ta.rma`` (Wilder smoothing): alpha = 1/length, seeded with the
    SMA of the first ``length`` values. Entries before the seed are NaN."""
    x = np.asarray(values, dtype=float).ravel()
    n = x.size
    out = np.full(n, np.nan)
    if n < length or length < 1:
        return out
    alpha = 1.0 / length
    out[length - 1] = float(np.mean(x[:length]))
    for i in range(length, n):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def rsi(prices: np.ndarray, length: int = RSI_LEN) -> np.ndarray:
    """Pine ``ta.rsi``: RSI on Wilder-smoothed average gains/losses.

    Returns an array aligned with ``prices``; the first ``length`` entries are
    NaN (there are only ``len-1`` diffs before the seed completes).
    """
    p = np.asarray(prices, dtype=float).ravel()
    n = p.size
    out = np.full(n, np.nan)
    if n < length + 1:
        return out
    diff = np.diff(p)
    up = np.where(diff > 0, diff, 0.0)
    down = np.where(diff < 0, -diff, 0.0)
    up_rma = wilder_rma(up, length)
    down_rma = wilder_rma(down, length)
    for i in range(length - 1, diff.size):
        u, d = up_rma[i], down_rma[i]
        if not (np.isfinite(u) and np.isfinite(d)):
            continue
        if d == 0.0:
            out[i + 1] = 100.0 if u > 0 else 50.0
        else:
            rs = u / d
            out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out


def compute_features(prices: np.ndarray) -> pd.DataFrame:
    """Replicate the Pine script on daily closes.

    Returns a DataFrame with columns:
        close, ema_fast, ema_slow, trend ("bullish"/"bearish"/None),
        rsi, bucket (one of ALL_BUCKETS or None during warm-up).
    """
    p = np.asarray(prices, dtype=float).ravel()
    ema_fast = ema(p, EMA_FAST_LEN)
    ema_slow = ema(p, EMA_SLOW_LEN)
    rsi_vals = rsi(p, RSI_LEN)

    trend: List[Optional[str]] = []
    buckets: List[Optional[str]] = []
    for f, s, r in zip(ema_fast, ema_slow, rsi_vals):
        if np.isfinite(f) and np.isfinite(s):
            t = TREND_BULLISH if f > s else TREND_BEARISH
        else:
            t = None
        trend.append(t)
        if t is not None and np.isfinite(r):
            buckets.append(bucket(t, float(r)))
        else:
            buckets.append(None)

    return pd.DataFrame(
        {
            "close": p,
            "ema_fast": ema_fast,
            "ema_slow": ema_slow,
            "trend": trend,
            "rsi": rsi_vals,
            "bucket": buckets,
        }
    )


# ---------------------------------------------------------------------------
# Buckets
# ---------------------------------------------------------------------------


def rsi_tercile(rsi_value: float) -> str:
    """Map an RSI value to its tercile: low < 40, 40 <= mid <= 60, high > 60."""
    r = float(rsi_value)
    if not math.isfinite(r):
        raise ValueError(f"RSI must be finite, got {rsi_value!r}")
    if r < RSI_LOW_MAX:
        return "low"
    if r > RSI_HIGH_MIN:
        return "high"
    return "mid"


def bucket(trend: str, rsi: float) -> str:
    """6-way bucket: trend {bullish, bearish} x RSI tercile {low, mid, high}.

    Raises ValueError for an unknown trend or non-finite RSI (callers treat
    that as "undefined bucket" -> zero drift).
    """
    t = str(trend).strip().lower()
    if t not in TRENDS:
        raise ValueError(f"trend must be one of {TRENDS}, got {trend!r}")
    return f"{t}_{rsi_tercile(rsi)}"


# ---------------------------------------------------------------------------
# Newey-West statistics
# ---------------------------------------------------------------------------


def newey_west_stats(x: np.ndarray, lag: int) -> Dict[str, float]:
    """Mean, naive SE, Newey-West SE (Bartlett kernel), n_eff and t-stat.

    Overlapping ``h``-day forward returns sampled daily share ``h - 1`` days,
    which induces strong positive autocorrelation up to lag ``h - 1``; the
    naive SE understates uncertainty by roughly sqrt(h). The Newey-West
    long-run variance corrects this, and

        n_eff = n * naive_var / long_run_var   (clipped to [1, n])

    is the equivalent number of independent observations.
    """
    v = np.asarray(x, dtype=float).ravel()
    v = v[np.isfinite(v)]
    n = v.size
    if n < 2:
        return {
            "n": float(n), "mean": float(v.mean()) if n else float("nan"),
            "se_naive": float("nan"), "se_nw": float("nan"),
            "n_eff": float(n), "t_stat": float("nan"),
        }
    mean = float(v.mean())
    vc = v - mean
    gamma0 = float(vc @ vc) / n
    lrv = gamma0
    max_lag = min(int(lag), n - 1)
    for l in range(1, max_lag + 1):
        w = 1.0 - l / (max_lag + 1.0)
        cov = float(vc[l:] @ vc[:-l]) / n
        lrv += 2.0 * w * cov
    # Bartlett weights guarantee lrv >= 0; guard against numerical zero.
    lrv = max(lrv, 1e-18)
    se_naive = math.sqrt(gamma0 / n) if gamma0 > 0 else 0.0
    se_nw = math.sqrt(lrv / n)
    n_eff = float(np.clip(n * gamma0 / lrv, 1.0, n)) if gamma0 > 0 else 1.0
    t_stat = mean / se_nw if se_nw > 0 else float("nan")
    return {
        "n": float(n), "mean": mean, "se_naive": se_naive, "se_nw": se_nw,
        "n_eff": n_eff, "t_stat": t_stat,
    }


def shrink_mu(raw_mu: float, n_eff: float, t_stat: float,
              k: float = K_PRIOR, t_prior: float = T_PRIOR,
              hard_zero_t: float = HARD_ZERO_T) -> float:
    """Shrunken bucket mean: ``raw_mu * n_eff / (n_eff + k_eff)`` with the
    adaptive prior strength ``k_eff = k + n_eff * t_prior / t^2`` (see the
    K_PRIOR/T_PRIOR comment), hard-zeroed when |t| < ``hard_zero_t`` or the
    t-stat is undefined. Monotone toward 0 as ``n_eff`` -> 0 and as |t|
    weakens; never amplifies."""
    if not (math.isfinite(raw_mu) and math.isfinite(n_eff) and n_eff > 0):
        return 0.0
    if not math.isfinite(t_stat) or abs(t_stat) < hard_zero_t:
        return 0.0
    k_eff = float(k) + float(n_eff) * float(t_prior) / float(t_stat) ** 2
    return float(raw_mu) * float(n_eff) / (float(n_eff) + k_eff)


# ---------------------------------------------------------------------------
# Calibration table
# ---------------------------------------------------------------------------


@dataclass
class BucketStats:
    """Per-bucket forward-return statistics (per-horizon, i.e. 'weekly' for
    the default 5-day horizon)."""

    bucket: str
    n: int = 0
    n_eff: float = 0.0
    raw_mu_weekly: float = 0.0
    shrunk_mu_weekly: float = 0.0
    se_weekly: float = float("nan")       # Newey-West SE of the mean
    se_naive_weekly: float = float("nan")
    t_stat: float = float("nan")
    raw_mu_annual: float = 0.0            # weekly * (252 / horizon)
    shrunk_mu_annual: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CalibrationTable:
    ticker: str
    horizon_days: int
    created_at: str
    data_start: str
    data_end: str
    n_days: int
    years: float
    k_prior: float = K_PRIOR
    t_prior: float = T_PRIOR
    hard_zero_t: float = HARD_ZERO_T
    method: str = ("overlapping forward log returns, Newey-West SE "
                   "(Bartlett, lag=horizon_days-1)")
    buckets: Dict[str, BucketStats] = field(default_factory=dict)

    @property
    def annualization(self) -> float:
        return TRADING_DAYS_PER_YEAR / float(self.horizon_days)

    def get(self, bucket_name: str) -> Optional[BucketStats]:
        return self.buckets.get(bucket_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "horizon_days": self.horizon_days,
            "created_at": self.created_at,
            "data_start": self.data_start,
            "data_end": self.data_end,
            "n_days": self.n_days,
            "years": self.years,
            "k_prior": self.k_prior,
            "t_prior": self.t_prior,
            "hard_zero_t": self.hard_zero_t,
            "method": self.method,
            "annualization": self.annualization,
            "buckets": {k: v.as_dict() for k, v in self.buckets.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CalibrationTable":
        buckets = {
            k: BucketStats(**{f: v[f] for f in BucketStats.__dataclass_fields__
                              if f in v})
            for k, v in (d.get("buckets") or {}).items()
        }
        return cls(
            ticker=d["ticker"],
            horizon_days=int(d["horizon_days"]),
            created_at=d["created_at"],
            data_start=d.get("data_start", ""),
            data_end=d.get("data_end", ""),
            n_days=int(d.get("n_days", 0)),
            years=float(d.get("years", 0.0)),
            k_prior=float(d.get("k_prior", K_PRIOR)),
            t_prior=float(d.get("t_prior", T_PRIOR)),
            hard_zero_t=float(d.get("hard_zero_t", HARD_ZERO_T)),
            method=d.get("method", ""),
            buckets=buckets,
        )

    # ---- persistence ------------------------------------------------------

    def path(self, calibration_dir: Path | str = DEFAULT_CALIBRATION_DIR) -> Path:
        return calibration_path(self.ticker, self.horizon_days, calibration_dir)

    def save(self, calibration_dir: Path | str = DEFAULT_CALIBRATION_DIR) -> Path:
        path = self.path(calibration_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n",
                        encoding="utf-8")
        return path

    # ---- display ----------------------------------------------------------

    def table_text(self) -> str:
        lines = [
            f"Calibration: {self.ticker}  horizon={self.horizon_days}d  "
            f"data {self.data_start} .. {self.data_end} ({self.n_days} days)",
            f"created_at={self.created_at}  k_prior={self.k_prior:.0f}  "
            f"t_prior={self.t_prior:.0f}  hard-zero |t|<{self.hard_zero_t:.1f}",
            f"{'bucket':<14}{'n':>6}{'n_eff':>8}{'raw_mu/wk':>11}"
            f"{'shrunk/wk':>11}{'NW se':>9}{'t':>7}{'shrunk/yr':>11}",
        ]
        for name in ALL_BUCKETS:
            b = self.buckets.get(name)
            if b is None:
                lines.append(f"{name:<14}{'—':>6}")
                continue
            lines.append(
                f"{name:<14}{b.n:>6d}{b.n_eff:>8.1f}"
                f"{b.raw_mu_weekly:>+11.4%}{b.shrunk_mu_weekly:>+11.4%}"
                f"{b.se_weekly:>9.4%}"
                f"{(b.t_stat if math.isfinite(b.t_stat) else float('nan')):>7.2f}"
                f"{b.shrunk_mu_annual:>+11.2%}"
            )
        return "\n".join(lines)


def calibration_path(ticker: str, horizon_days: int,
                     calibration_dir: Path | str = DEFAULT_CALIBRATION_DIR,
                     ) -> Path:
    return Path(calibration_dir) / f"{ticker.upper()}_{int(horizon_days)}d.json"


# ---------------------------------------------------------------------------
# History download
# ---------------------------------------------------------------------------


def download_history(ticker: str, years: float = 8.0,
                     ) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Download daily adjusted closes via yfinance (auto_adjust=True)."""
    import yfinance as yf

    period_years = max(1, int(math.ceil(years)))
    data = yf.download(
        ticker,
        period=f"{period_years}y",
        progress=False,
        auto_adjust=True,
    )
    if data is None or len(data) < 2:
        raise CalibrationDataError(f"no price history returned for {ticker!r}")
    close = data["Close"]
    if hasattr(close, "columns"):  # multi-ticker frame -> first column
        close = close.iloc[:, 0]
    close = close.dropna()
    prices = np.asarray(close, dtype=float).ravel()
    if prices.size < EMA_SLOW_LEN + 10:
        raise CalibrationDataError(
            f"insufficient history for {ticker!r} ({prices.size} rows)"
        )
    return prices, pd.DatetimeIndex(close.index)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate(
    ticker: str,
    years: float = 8.0,
    horizon_days: int = 5,
    *,
    prices: Optional[np.ndarray] = None,
    dates: Optional[pd.DatetimeIndex] = None,
    now: Optional[datetime] = None,
) -> CalibrationTable:
    """Build a signal-conditioned drift calibration table.

    For each day ``t`` with a defined bucket, records the forward
    ``horizon_days``-day log return ``r_fwd(t) = log(P[t+h] / P[t])``.
    Samples are **overlapping** (one per day) and the per-bucket standard
    error uses Newey-West with lag = ``horizon_days - 1`` (see
    :func:`newey_west_stats`); ``n_eff`` reflects the overlap.

    ``prices``/``dates`` may be supplied directly (tests, offline use);
    otherwise history is downloaded via yfinance (auto_adjust=True).
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    if prices is None:
        prices, dates = download_history(ticker, years)
    p = np.asarray(prices, dtype=float).ravel()
    if np.any(~np.isfinite(p)) or np.any(p <= 0):
        raise CalibrationDataError("prices must be finite and positive")
    if p.size < EMA_SLOW_LEN + horizon_days + 2:
        raise CalibrationDataError(
            f"need more than {EMA_SLOW_LEN + horizon_days + 2} prices, "
            f"got {p.size}"
        )

    feats = compute_features(p)
    log_p = np.log(p)
    h = int(horizon_days)
    fwd = np.full(p.size, np.nan)
    fwd[:-h] = log_p[h:] - log_p[:-h]
    feats["fwd_ret"] = fwd

    ann = TRADING_DAYS_PER_YEAR / float(h)
    buckets: Dict[str, BucketStats] = {}
    for name in ALL_BUCKETS:
        mask = (feats["bucket"] == name) & np.isfinite(feats["fwd_ret"])
        sample = feats.loc[mask, "fwd_ret"].to_numpy(dtype=float)
        st = newey_west_stats(sample, lag=h - 1)
        raw_mu = st["mean"] if math.isfinite(st["mean"]) else 0.0
        shrunk = shrink_mu(raw_mu, st["n_eff"], st["t_stat"])
        buckets[name] = BucketStats(
            bucket=name,
            n=int(st["n"]),
            n_eff=float(st["n_eff"]),
            raw_mu_weekly=float(raw_mu),
            shrunk_mu_weekly=float(shrunk),
            se_weekly=float(st["se_nw"]),
            se_naive_weekly=float(st["se_naive"]),
            t_stat=float(st["t_stat"]),
            raw_mu_annual=float(raw_mu) * ann,
            shrunk_mu_annual=float(shrunk) * ann,
        )

    ts_now = (now or datetime.now(timezone.utc)).replace(microsecond=0)
    if dates is not None and len(dates) == p.size:
        start_s = str(pd.Timestamp(dates[0]).date())
        end_s = str(pd.Timestamp(dates[-1]).date())
    else:
        start_s, end_s = "row_0", f"row_{p.size - 1}"
    return CalibrationTable(
        ticker=str(ticker).upper(),
        horizon_days=h,
        created_at=ts_now.isoformat(),
        data_start=start_s,
        data_end=end_s,
        n_days=int(p.size),
        years=float(years),
        buckets=buckets,
    )


def load_calibration(
    ticker: str,
    horizon_days: int = 5,
    calibration_dir: Path | str = DEFAULT_CALIBRATION_DIR,
    *,
    warn_stale_days: int = STALE_WARN_DAYS,
    error_stale_days: int = STALE_ERROR_DAYS,
    now: Optional[datetime] = None,
) -> CalibrationTable:
    """Load a stored calibration table.

    Warns (``UserWarning``) if the table is older than ``warn_stale_days``
    and raises :class:`CalibrationStaleError` if older than
    ``error_stale_days`` (drift edges decay; stale tables must not be
    trusted silently).
    """
    path = calibration_path(ticker, horizon_days, calibration_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"No calibration for {ticker.upper()} at {path}. "
            f"Run: python signal_calibration.py {ticker.upper()} --years 8"
        )
    table = CalibrationTable.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )
    try:
        created = datetime.fromisoformat(table.created_at)
    except ValueError as exc:
        raise CalibrationError(
            f"Unparseable created_at in {path}: {table.created_at!r}"
        ) from exc
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    ref = now or datetime.now(timezone.utc)
    age_days = (ref - created).total_seconds() / 86400.0
    if age_days > error_stale_days:
        raise CalibrationStaleError(
            f"Calibration for {table.ticker} is {age_days:.0f} days old "
            f"(> {error_stale_days}); re-run signal_calibration.py"
        )
    if age_days > warn_stale_days:
        warnings.warn(
            f"Calibration for {table.ticker} is {age_days:.0f} days old "
            f"(> {warn_stale_days}); consider re-running signal_calibration.py",
            UserWarning,
            stacklevel=2,
        )
    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate signal-conditioned drift from history "
                    "(EMA 9/21 trend x RSI 14 terciles).",
    )
    ap.add_argument("ticker")
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--horizon", type=int, default=5,
                    help="forward-return horizon in trading days (default 5)")
    ap.add_argument("--calibration-dir", default=str(DEFAULT_CALIBRATION_DIR))
    args = ap.parse_args(argv)

    try:
        table = calibrate(args.ticker, years=args.years,
                          horizon_days=args.horizon)
    except CalibrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    path = table.save(args.calibration_dir)
    print(table.table_text())
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
