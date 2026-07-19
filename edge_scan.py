#!/usr/bin/env python3
"""
edge_scan.py — "look at the chart for me": scan a watchlist and report the
best *verified* edge right now, at weekly and monthly horizons.

What this does (plain English)
------------------------------
For each ticker it downloads the real daily price history, computes today's
signal state exactly like the TradingView script does (EMA 9/21 trend +
RSI 14), then asks the calibration machinery: "historically, when this
stock was in *this exact state*, what happened over the next 5 trading
days (a week) and the next 21 trading days (a month) — and is that pattern
statistically real, or noise?"

Everything visible on a chart image is computed from this same data, so
this is the honest version of "analyze the chart": it reads the numbers
the chart is drawn from, not the pixels.

The output is ranked by verified edge. When a state has no statistically
real pattern the edge is exactly 0.00% and it says so — this tool will
frequently (correctly) report "no verified edge"; that is the design, not
a failure. Monthly-horizon estimates overlap heavily (21-day windows), so
their effective sample sizes are small and they shrink hard toward zero —
expect monthly edges to be verified even more rarely than weekly ones.

Usage
-----
    python edge_scan.py PLTR NVDA TSLA
    python edge_scan.py PLTR NVDA TSLA --years 8 --horizons 5 21
    # optionally persist the tables for run_weekly_from_tv.py:
    python edge_scan.py PLTR --save-calibration-dir calibration
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from signal_calibration import (
    CalibrationError,
    calibrate,
    compute_features,
    download_history,
)

TRADING_DAYS_PER_YEAR = 252
DEFAULT_HORIZONS = (5, 21)  # weekly, monthly (trading days)

HORIZON_LABEL = {5: "weekly (5d)", 21: "monthly (21d)"}


@dataclass
class EdgeRow:
    ticker: str
    price: float
    trend: str
    rsi: float
    bucket: Optional[str]
    horizon_days: int
    shrunk_mu_period: float    # expected log return over the horizon
    shrunk_mu_annual: float
    raw_mu_period: float
    t_stat: float
    n_eff: float
    n: int

    @property
    def verified(self) -> bool:
        return self.shrunk_mu_period != 0.0

    @property
    def direction(self) -> str:
        if not self.verified:
            return "none"
        return "long" if self.shrunk_mu_period > 0 else "short"

    def label(self) -> str:
        """Drift-source label — repo rule: no naked probabilities/edges."""
        h = HORIZON_LABEL.get(self.horizon_days, f"{self.horizon_days}d")
        if not self.verified:
            return (f"{h}: no verified edge in bucket {self.bucket} "
                    f"(t={self.t_stat:.2f}, n_eff={self.n_eff:.0f} — "
                    f"treated as noise)")
        side = "rises" if self.shrunk_mu_period > 0 else "falls"
        return (f"{h}: {self.shrunk_mu_period:+.2%}/period "
                f"({self.shrunk_mu_annual:+.1%}/yr) — historically {side} in "
                f"bucket {self.bucket} [t={self.t_stat:.2f}, "
                f"n_eff={self.n_eff:.0f}]")


def scan_ticker(
    ticker: str,
    *,
    years: float = 8.0,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    prices: Optional[np.ndarray] = None,
    dates=None,
    save_calibration_dir: Optional[str] = None,
) -> List[EdgeRow]:
    """Compute today's signal state and the calibrated edge per horizon.

    ``prices`` may be injected for offline use/tests; otherwise history is
    downloaded (same source as calibration and the weekly pipeline).
    """
    if prices is None:
        prices, dates = download_history(ticker, years)
    p = np.asarray(prices, dtype=float).ravel()
    feats = compute_features(p)
    last = feats.iloc[-1]
    bucket = last["bucket"]
    bucket = None if (bucket is None or (isinstance(bucket, float)
                                         and math.isnan(bucket))) else bucket
    trend = last["trend"] if isinstance(last["trend"], str) else "unknown"
    rsi = float(last["rsi"]) if np.isfinite(last["rsi"]) else float("nan")

    rows: List[EdgeRow] = []
    for h in horizons:
        table = calibrate(ticker, years=years, horizon_days=int(h),
                          prices=p, dates=dates)
        if save_calibration_dir:
            table.save(save_calibration_dir)
        stats = table.get(bucket) if bucket else None
        if stats is None or stats.n < 2:
            rows.append(EdgeRow(
                ticker=ticker.upper(), price=float(p[-1]), trend=trend,
                rsi=rsi, bucket=bucket, horizon_days=int(h),
                shrunk_mu_period=0.0, shrunk_mu_annual=0.0,
                raw_mu_period=0.0, t_stat=float("nan"), n_eff=0.0, n=0,
            ))
            continue
        rows.append(EdgeRow(
            ticker=ticker.upper(), price=float(p[-1]), trend=trend, rsi=rsi,
            bucket=bucket, horizon_days=int(h),
            shrunk_mu_period=float(stats.shrunk_mu_weekly),
            shrunk_mu_annual=float(stats.shrunk_mu_annual),
            raw_mu_period=float(stats.raw_mu_weekly),
            t_stat=float(stats.t_stat),
            n_eff=float(stats.n_eff),
            n=int(stats.n),
        ))
    return rows


def rank_rows(rows: List[EdgeRow]) -> List[EdgeRow]:
    """Verified edges first, largest absolute annualized edge on top."""
    return sorted(rows, key=lambda r: (not r.verified,
                                       -abs(r.shrunk_mu_annual)))


def report_text(all_rows: List[EdgeRow]) -> str:
    lines = [
        "=" * 72,
        "EDGE SCAN — current state vs calibrated history",
        "(edge = expected drift ONLY when statistically verified; 0.00% means",
        " the pattern is noise and the honest answer is 'no edge')",
        "=" * 72,
    ]
    by_ticker: Dict[str, List[EdgeRow]] = {}
    for r in all_rows:
        by_ticker.setdefault(r.ticker, []).append(r)
    for ticker, rows in by_ticker.items():
        head = rows[0]
        lines.append(f"\n{ticker}  price={head.price:.2f}  "
                     f"state: {head.trend}, RSI {head.rsi:.1f} "
                     f"→ bucket {head.bucket or 'undefined (warm-up)'}")
        for r in rows:
            lines.append(f"  {r.label()}")

    ranked = [r for r in rank_rows(all_rows) if r.verified]
    lines.append("\n" + "-" * 72)
    if ranked:
        best = ranked[0]
        lines.append(
            f"BEST VERIFIED EDGE: {best.ticker} "
            f"{HORIZON_LABEL.get(best.horizon_days, best.horizon_days)} — "
            f"{best.direction.upper()} side, {best.shrunk_mu_annual:+.1%}/yr "
            f"[bucket={best.bucket}, t={best.t_stat:.2f}, "
            f"n_eff={best.n_eff:.0f}]"
        )
        lines.append(
            "Reminder: an edge is a drift estimate, not a guarantee — the "
            "weekly pipeline still checks it against breakeven, costs and "
            "the kill-switch before any TRADE verdict."
        )
    else:
        lines.append(
            "BEST VERIFIED EDGE: none right now — every current bucket is "
            "statistically indistinguishable from noise. Not trading is the "
            "correct action on this information."
        )
    lines.append("=" * 72)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Scan tickers for the best verified edge at weekly and "
                    "monthly horizons (honest chart analysis: reads the data "
                    "the chart is drawn from).")
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--horizons", type=int, nargs="+",
                    default=list(DEFAULT_HORIZONS),
                    help="horizons in trading days (default: 5 21)")
    ap.add_argument("--save-calibration-dir", default=None,
                    help="optionally persist the tables (e.g. 'calibration') "
                         "so run_weekly_from_tv.py uses the fresh ones")
    args = ap.parse_args(argv)

    all_rows: List[EdgeRow] = []
    failures: List[Tuple[str, str]] = []
    for t in args.tickers:
        try:
            all_rows.extend(scan_ticker(
                t, years=args.years, horizons=args.horizons,
                save_calibration_dir=args.save_calibration_dir))
        except CalibrationError as exc:
            failures.append((t.upper(), str(exc)))
    if all_rows:
        print(report_text(all_rows))
    for t, msg in failures:
        print(f"WARNING: {t} skipped — {msg}", file=sys.stderr)
    if not all_rows:
        print("ERROR: no ticker could be scanned.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
