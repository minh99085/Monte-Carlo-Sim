#!/usr/bin/env python3
"""
paper_train.py — accumulate a paper-trading track record for training.

Why this exists
---------------
The production pipeline (``run_weekly_from_tv.py``) only produces a verdict
when a TradingView alert arrives — roughly one ticker per week. That is far
too slow to *train* on: the kill-switch needs 20 settled TRADE verdicts
before it even activates, and the calibration-vs-realized statistics
(``outcome_tracker.report_stats``) need a sample to be meaningful.

This driver runs the *exact same brain* — calibrated drift, Monte Carlo,
breakeven, 2x cost stress, momentum agreement filter, sizing, kill-switch —
across a whole watchlist on demand. For each ticker it reconstructs today's
signal state from real history (the same EMA/RSI the TradingView script
sends), runs the full decision, and logs every verdict to a **dedicated
paper log** (kept separate from the live ``outputs/trade_log.jsonl`` so
training never contaminates the real ledger). Then it settles whatever has
matured against real forward prices and prints a training report.

No orders are placed anywhere. A "paper trade" here is a logged verdict that
``outcome_tracker`` later settles against real closes — identical to how the
live pipeline is scored. This is broker-independent; the Robinhood bot is
the *live* executor for a later stage and is not involved in training.

Usage
-----
    # one training pass over a watchlist (decide + settle + report)
    python paper_train.py SPY QQQ XLK NVDA AAPL MSFT \
        --calibration-dir calibration

    # settle/report only (no new decisions)
    python paper_train.py --report-only --paper-log outputs/paper_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import outcome_tracker
import run_weekly_from_tv as rwt
from signal_calibration import CalibrationError, compute_features, download_history

DEFAULT_PAPER_LOG = Path("outputs") / "paper_log.jsonl"
DEFAULT_PAPER_VERDICTS = Path("outputs") / "paper_verdicts"
DEFAULT_PAPER_SIGNALS = Path("outputs") / "paper_signals"


def synthesize_signal(ticker: str, prices: np.ndarray) -> Optional[Dict[str, Any]]:
    """Reconstruct today's TradingView-equivalent signal from real history.

    Returns None during indicator warm-up (no defined trend/RSI yet) — the
    same "no signal" state the live bridge would produce.
    """
    feats = compute_features(np.asarray(prices, dtype=float).ravel())
    last = feats.iloc[-1]
    trend = last["trend"]
    rsi = last["rsi"]
    if not isinstance(trend, str) or not np.isfinite(rsi):
        return None
    return {
        "received_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0).isoformat(),
        "source": "paper_train",
        "ticker": ticker.upper(),
        "price": float(last["close"]),
        "trend": trend,
        "momentum": float(rsi),
        "timeframe": "D",
        "strategy": "paper_train",
    }


def _write_signal(signal: Dict[str, Any], signals_dir: Path) -> Path:
    signals_dir.mkdir(parents=True, exist_ok=True)
    path = signals_dir / f"{signal['ticker']}_latest.json"
    path.write_text(json.dumps(signal) + "\n", encoding="utf-8")
    return path


def decide_ticker(
    ticker: str,
    *,
    calibration_dir: Path,
    paper_log: Path,
    verdict_dir: Path,
    signals_dir: Path,
    horizon_days: int = rwt.WEEK_DAYS,
    paths: int = 40_000,
    years_history: float = 8.0,
    prices: Optional[np.ndarray] = None,
    write_files: bool = True,
    agreement_filter: bool = True,
    s0: Optional[float] = None,
    sigma: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Run the full decision for one ticker and log the verdict to the paper
    log. Returns the verdict dict, or None when the ticker has no usable
    signal state (warm-up) — never raises for a single bad ticker."""
    if prices is None:
        prices, _ = download_history(ticker, years_history)
    signal = synthesize_signal(ticker, prices)
    if signal is None:
        return None
    _write_signal(signal, signals_dir)
    settings = rwt.PipelineSettings(
        data_dir=signals_dir,
        signal_path=signals_dir / f"{ticker.upper()}_latest.json",
        calibration_dir=calibration_dir,
        horizon_days=horizon_days,
        paths=paths,
        trade_log=paper_log,
        verdict_dir=verdict_dir,
        years_history=years_history,
        write_files=write_files,
        noise_sweep=False,      # informational only; skip for bulk speed
        agreement_filter=agreement_filter,
        s0=s0,
        sigma=sigma,
    )
    return rwt.run_pipeline(settings)


def paper_train(
    tickers: Sequence[str],
    *,
    calibration_dir: Path,
    paper_log: Path = DEFAULT_PAPER_LOG,
    verdict_dir: Path = DEFAULT_PAPER_VERDICTS,
    signals_dir: Path = DEFAULT_PAPER_SIGNALS,
    paths: int = 40_000,
    years_history: float = 8.0,
    do_settle: bool = True,
    price_fetcher=None,
    prices_by_ticker: Optional[Dict[str, np.ndarray]] = None,
    agreement_filter: bool = True,
) -> Dict[str, Any]:
    """One training pass: decide every ticker, settle matured trades, report."""
    decided: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for t in tickers:
        prices = (prices_by_ticker or {}).get(t.upper())
        try:
            v = decide_ticker(
                t, calibration_dir=calibration_dir, paper_log=paper_log,
                verdict_dir=verdict_dir, signals_dir=signals_dir,
                paths=paths, years_history=years_history, prices=prices,
                agreement_filter=agreement_filter)
        except rwt.FallbackDataError as exc:
            skipped.append({"ticker": t.upper(), "reason": str(exc)})
            continue
        except (CalibrationError, Exception) as exc:  # noqa: BLE001
            skipped.append({"ticker": t.upper(), "reason": repr(exc)})
            continue
        if v is None:
            skipped.append({"ticker": t.upper(), "reason": "warm-up: no signal"})
        else:
            decided.append(v)

    settle_counts = (outcome_tracker.settle(paper_log, price_fetcher=price_fetcher)
                     if do_settle else {"settled": 0, "pending": 0, "skipped": 0})
    stats = outcome_tracker.report_stats(paper_log)
    ks_tripped, ks_reason = outcome_tracker.check_kill_switch(paper_log)
    return {
        "decided": decided,
        "n_trade": sum(1 for v in decided if v["verdict"] == "TRADE"),
        "n_no_trade": sum(1 for v in decided if v["verdict"] == "NO_TRADE"),
        "skipped": skipped,
        "settle_counts": settle_counts,
        "stats": stats,
        "kill_switch": {"tripped": ks_tripped, "reason": ks_reason},
    }


def report_text(result: Dict[str, Any]) -> str:
    s = result["stats"]
    lines = [
        "=" * 68,
        "PAPER TRAINING PASS",
        "=" * 68,
        f"Decided this pass: {len(result['decided'])} "
        f"(TRADE {result['n_trade']}, NO_TRADE {result['n_no_trade']})",
    ]
    for v in result["decided"]:
        if v["verdict"] == "TRADE":
            sh = v["sizing"]["shares"]
            lines.append(f"  TRADE   {v['ticker']:<6} {v['side']:<5} "
                         f"{sh} sh — {v['reason']}")
    if result["skipped"]:
        lines.append(f"Skipped: {len(result['skipped'])}")
        for sk in result["skipped"]:
            lines.append(f"  - {sk['ticker']}: {sk['reason']}")
    sc = result["settle_counts"]
    lines.append(f"\nSettlement: {sc['settled']} newly settled, "
                 f"{sc['pending']} still maturing, {sc['skipped']} skipped")
    lines.append("-" * 68)
    lines.append("Track record (settled TRADE verdicts):")
    lines.append(f"  logged={s.get('n_logged', 0)}  "
                 f"trade_verdicts={s.get('n_trade_verdicts', 0)}  "
                 f"settled_trades={s.get('n_settled_trades', 0)}")
    if s.get("n_settled_trades"):
        hr = s.get("hit_rate")
        mr = s.get("mean_realized_pnl_pct")
        mp = s.get("mean_predicted_expectancy_pct")
        cal = s.get("calibration_ratio")
        lines.append(f"  hit_rate={hr:.1%}" if hr is not None else "  hit_rate=n/a")
        lines.append(
            f"  mean realized P&L/trade={mr:+.3%}  "
            f"predicted={mp:+.3%}" if (mr is not None and mp is not None)
            else f"  mean realized P&L/trade="
                 f"{'n/a' if mr is None else format(mr, '+.3%')}")
        if cal is not None:
            lines.append(f"  calibration ratio (realized/predicted)={cal:.2f} "
                         "(1.0 = perfectly calibrated; <1 = optimistic)")
    else:
        lines.append("  (no settled TRADE verdicts yet — keep running passes; "
                     "the kill-switch needs 20 to activate)")
    lines.append("-" * 68)
    lines.append(f"Kill-switch: {result['kill_switch']['reason']}")
    lines.append("=" * 68)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Accumulate a paper-trading track record for training by "
                    "running the full decision across a watchlist and "
                    "settling matured trades against real prices.")
    ap.add_argument("tickers", nargs="*", help="watchlist to decide this pass")
    ap.add_argument("--calibration-dir", default="calibration")
    ap.add_argument("--paper-log", default=str(DEFAULT_PAPER_LOG))
    ap.add_argument("--verdict-dir", default=str(DEFAULT_PAPER_VERDICTS))
    ap.add_argument("--signals-dir", default=str(DEFAULT_PAPER_SIGNALS))
    ap.add_argument("--paths", type=int, default=40_000)
    ap.add_argument("--years", type=float, default=8.0)
    ap.add_argument("--report-only", action="store_true",
                    help="only settle + report; make no new decisions")
    args = ap.parse_args(argv)

    if args.report_only or not args.tickers:
        paper_log = Path(args.paper_log)
        settle_counts = outcome_tracker.settle(paper_log)
        result = {
            "decided": [], "n_trade": 0, "n_no_trade": 0, "skipped": [],
            "settle_counts": settle_counts,
            "stats": outcome_tracker.report_stats(paper_log),
            "kill_switch": dict(zip(
                ("tripped", "reason"),
                outcome_tracker.check_kill_switch(paper_log))),
        }
        print(report_text(result))
        return 0

    result = paper_train(
        args.tickers,
        calibration_dir=Path(args.calibration_dir),
        paper_log=Path(args.paper_log),
        verdict_dir=Path(args.verdict_dir),
        signals_dir=Path(args.signals_dir),
        paths=int(args.paths),
        years_history=float(args.years),
    )
    print(report_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
