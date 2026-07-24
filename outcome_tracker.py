"""
Outcome tracking and kill-switch for the TradingView drift pipeline (Phase C).

Every verdict produced by ``run_weekly_from_tv.py`` — TRADE or NO_TRADE — is
appended as one line to ``outputs/trade_log.jsonl``. Later, ``settle`` fetches
the realized price path over the holding window and replays the *recorded*
stop/TP rule on it (same engine as the Monte Carlo paths:
``tactical_simulator.apply_rule_to_one_path``), so predicted and realized
expectancy are directly comparable. ``report`` summarizes calibration quality,
and the kill-switch blocks new TRADE verdicts once the trailing 20 settled
TRADE outcomes have negative mean realized P&L.

Commands
--------
    python outcome_tracker.py settle [--log outputs/trade_log.jsonl]
    python outcome_tracker.py report [--log outputs/trade_log.jsonl]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_TRADE_LOG = Path("outputs") / "trade_log.jsonl"
KILL_SWITCH_WINDOW = 20

# price_fetcher(ticker, start_iso_date) -> (dates_iso list, closes ndarray)
PriceFetcher = Callable[[str, str], Tuple[List[str], np.ndarray]]


# ---------------------------------------------------------------------------
# Log I/O
# ---------------------------------------------------------------------------


def read_log(log_path: Path | str = DEFAULT_TRADE_LOG) -> List[Dict[str, Any]]:
    path = Path(log_path)
    if not path.is_file():
        return []
    entries: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            entries.append(obj)
    return entries


def write_log(entries: List[Dict[str, Any]],
              log_path: Path | str = DEFAULT_TRADE_LOG) -> Path:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    return path


def log_verdict(verdict: Dict[str, Any],
                log_path: Path | str = DEFAULT_TRADE_LOG) -> Path:
    """Append one verdict (traded or not) to the trade log."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(verdict)
    record.setdefault("settled", False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Settlement
# ---------------------------------------------------------------------------


def _default_price_fetcher(ticker: str, start_iso: str,
                           ) -> Tuple[List[str], np.ndarray]:
    """Fetch daily adjusted closes from ``start_iso`` to today (yfinance)."""
    import yfinance as yf

    data = yf.download(ticker, start=start_iso, progress=False,
                       auto_adjust=True)
    if data is None or len(data) == 0:
        return [], np.asarray([], dtype=float)
    close = data["Close"]
    if hasattr(close, "columns"):
        close = close.iloc[:, 0]
    close = close.dropna()
    dates = [str(d.date()) for d in close.index]
    return dates, np.asarray(close, dtype=float).ravel()


def _entry_date(entry: Dict[str, Any]) -> Optional[str]:
    ts = entry.get("signal_received_at_utc") or entry.get("timestamp_utc")
    if not ts:
        return None
    try:
        return str(datetime.fromisoformat(str(ts)).date())
    except ValueError:
        return None


def settle_entry(
    entry: Dict[str, Any],
    dates: List[str],
    closes: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Settle one log entry against a realized close series.

    Fill contract (unified with v2's next-bar labeling): entry is the first
    close STRICTLY AFTER the signal date — a decision made on day t can only
    be filled in the next session, never at the same bar it was decided on
    (that was lookahead: research could "enter" at a price execution never
    had). The path is that entry close plus the next ``horizon_days``
    closes. Returns the settlement dict or None when the window is not
    complete yet.
    """
    entry_date = _entry_date(entry)
    if entry_date is None:
        return None
    horizon = int(entry.get("horizon_days") or 5)
    idx = next((i for i, d in enumerate(dates) if d > entry_date), None)
    if idx is None:
        return None
    path = np.asarray(closes[idx: idx + horizon + 1], dtype=float)
    if path.size < horizon + 1:
        return None  # not enough trading days elapsed yet

    from tactical_config import TradingRule
    from tactical_simulator import apply_rule_to_one_path

    structure = entry.get("structure") or {}
    side = str(entry.get("side") or "long")
    stop_pct = float(structure.get("stop_pct") or 0.0)
    tp_raw = structure.get("tp_pct")
    tp_pct = float(tp_raw) if tp_raw else None
    cost = float(entry.get("cost_per_side") or 0.0)

    rule = TradingRule(
        name="settlement replay",
        entry_condition=f"Enter {side} at recorded signal close",
        exit_condition="Recorded stop / TP / time exit",
        stop_loss_pct=stop_pct,
        take_profit_pct=tp_pct,
        max_holding_days=horizon,
        side=side,
        entry_day=0,
        allow_reentry=False,
    )
    trades, meta = apply_rule_to_one_path(path, rule, cost=cost, side=side)
    entry_close = float(path[0])
    exit_close = float(path[-1])
    realized_ret = exit_close / entry_close - 1.0
    realized_log_ret = math.log(exit_close / entry_close)
    if trades:
        realized_pnl_pct = float(sum(t.pnl_pct for t in trades))
        realized_pnl = float(meta["total_pnl"])
        exit_reason = trades[-1].exit_reason
    else:
        realized_pnl_pct = 0.0
        realized_pnl = 0.0
        exit_reason = "no_trade"

    return {
        "settled_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0).isoformat(),
        "entry_rule": "next_session_close",
        "entry_date": dates[idx],
        "exit_date": dates[idx + horizon],
        "entry_close": entry_close,
        "exit_close": exit_close,
        "realized_ret": realized_ret,          # buy-and-hold close-to-close
        "realized_log_ret": realized_log_ret,
        "realized_pnl": realized_pnl,          # $ per share under recorded rule
        "realized_pnl_pct": realized_pnl_pct,  # fraction of entry price
        "hit": realized_pnl_pct > 0.0,
        "exit_reason": exit_reason,
    }


def settle(
    log_path: Path | str = DEFAULT_TRADE_LOG,
    *,
    price_fetcher: Optional[PriceFetcher] = None,
) -> Dict[str, int]:
    """Settle all unsettled entries whose holding window has completed."""
    fetch = price_fetcher or _default_price_fetcher
    entries = read_log(log_path)
    n_settled = n_pending = n_skipped = 0
    history_cache: Dict[Tuple[str, str], Tuple[List[str], np.ndarray]] = {}

    for entry in entries:
        if entry.get("settled"):
            continue
        ticker = str(entry.get("ticker") or "").upper()
        entry_date = _entry_date(entry)
        if not ticker or entry_date is None:
            n_skipped += 1
            continue
        key = (ticker, entry_date)
        if key not in history_cache:
            try:
                history_cache[key] = fetch(ticker, entry_date)
            except Exception:  # noqa: BLE001 - fetch failure -> stay pending
                history_cache[key] = ([], np.asarray([], dtype=float))
        dates, closes = history_cache[key]
        settlement = settle_entry(entry, dates, closes) if len(dates) else None
        if settlement is None:
            n_pending += 1
            continue
        entry["settled"] = True
        entry["settlement"] = settlement
        n_settled += 1

    if n_settled:
        write_log(entries, log_path)
    return {"settled": n_settled, "pending": n_pending, "skipped": n_skipped}


# ---------------------------------------------------------------------------
# Reporting + kill-switch
# ---------------------------------------------------------------------------


def _settled_trades(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        e for e in entries
        if e.get("settled") and e.get("verdict") == "TRADE"
        and isinstance(e.get("settlement"), dict)
    ]


def report_stats(log_path: Path | str = DEFAULT_TRADE_LOG) -> Dict[str, Any]:
    """Rolling stats over settled TRADE verdicts: hit rate vs predicted
    P(win), realized vs predicted expectancy, and the calibration ratio."""
    entries = read_log(log_path)
    trades = _settled_trades(entries)
    out: Dict[str, Any] = {
        "n_logged": len(entries),
        "n_trade_verdicts": sum(1 for e in entries if e.get("verdict") == "TRADE"),
        "n_settled_trades": len(trades),
    }
    if not trades:
        return out

    realized = np.asarray(
        [t["settlement"]["realized_pnl_pct"] for t in trades], dtype=float)
    hits = np.asarray([bool(t["settlement"]["hit"]) for t in trades])
    predicted_p = np.asarray(
        [float(t.get("p_win") or np.nan) for t in trades], dtype=float)
    predicted_mu = np.asarray(
        [float(t.get("expectancy_pct") or np.nan) for t in trades], dtype=float)

    mean_realized = float(np.mean(realized))
    mean_predicted = float(np.nanmean(predicted_mu))
    out.update({
        "hit_rate": float(np.mean(hits)),
        "predicted_p_win_mean": (
            float(np.nanmean(predicted_p))
            if np.any(np.isfinite(predicted_p)) else None
        ),
        "mean_realized_pnl_pct": mean_realized,
        "mean_predicted_expectancy_pct": (
            mean_predicted if math.isfinite(mean_predicted) else None
        ),
        "calibration_ratio": (
            mean_realized / mean_predicted
            if math.isfinite(mean_predicted) and mean_predicted != 0.0
            else None
        ),
        "trailing20_mean_realized_pnl_pct": float(
            np.mean(realized[-KILL_SWITCH_WINDOW:])
        ),
    })
    return out


def check_kill_switch(
    log_path: Path | str = DEFAULT_TRADE_LOG,
    *,
    window: int = KILL_SWITCH_WINDOW,
) -> Tuple[bool, str]:
    """Kill-switch: trip when the trailing ``window`` settled TRADE verdicts
    have negative mean realized P&L. Returns (tripped, reason)."""
    trades = _settled_trades(read_log(log_path))
    if len(trades) < window:
        return False, (
            f"kill-switch inactive: {len(trades)} settled TRADE verdicts "
            f"(< {window})"
        )
    tail = trades[-window:]
    mean_pnl = float(np.mean(
        [t["settlement"]["realized_pnl_pct"] for t in tail]))
    if mean_pnl < 0.0:
        return True, (
            f"KILL-SWITCH TRIPPED: trailing {window} settled TRADE verdicts "
            f"have mean realized P&L {mean_pnl:+.3%} < 0. The signal is not "
            f"delivering its calibrated edge; re-calibrate before trading "
            f"(override with --override-killswitch)."
        )
    return False, (
        f"kill-switch clear: trailing {window} mean realized P&L "
        f"{mean_pnl:+.3%}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("command", choices=["settle", "report"])
    ap.add_argument("--log", default=str(DEFAULT_TRADE_LOG))
    args = ap.parse_args(argv)

    if args.command == "settle":
        counts = settle(args.log)
        print(f"Settled {counts['settled']}, pending {counts['pending']}, "
              f"skipped {counts['skipped']} (log: {args.log})")
        tripped, reason = check_kill_switch(args.log)
        print(reason)
        return 0

    stats = report_stats(args.log)
    print(f"Trade log report ({args.log})")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k:36s} {v:+.4f}")
        else:
            print(f"  {k:36s} {v}")
    tripped, reason = check_kill_switch(args.log)
    print(f"  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
