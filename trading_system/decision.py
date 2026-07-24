"""Decision engine — one live TradingView signal → one verdict.

Pipeline: load bars → build the as-of-now feature vector → fit the final
calibrated model on ALL labeled history strictly before today → calibrated
P(profit) → gate at the threshold → size via the configured sizer → write a
verdict JSON in the SAME schema run_weekly_from_tv.py produced, so the
existing hand (mc_bridge → Robinhood bot safety gates) consumes it
unchanged. DRY_RUN discipline: this module never talks to Robinhood at all;
the bot's phase-1 bridge is paper-only and live trading additionally sits
behind the bot's own RH_LIVE_TRADING_ENABLED flag.

Cash-account note: short signals are recorded as NO_TRADE (no shorting).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from trading_system import load_config
from trading_system.barriers import triple_barrier_label
from trading_system.data import Bars, FetchFn, fetch_bars, realized_vol_daily
from trading_system.features import build_matrix, feature_vector
from trading_system.model import fit_final_model
from trading_system.primary import signal_history
from trading_system.sizing import shares_for

DEFAULT_GAUNTLET_REPORT = Path("outputs") / "gauntlet_report.json"


def gauntlet_ready(report_path: Path | str = DEFAULT_GAUNTLET_REPORT) -> bool:
    """True only while the current gauntlet report shows ALL gates passing.

    This is the source of the ``gauntlet_pass`` marker stamped on every
    verdict. The bot's bridge refuses to treat any unmarked verdict as
    execution-eligible, so a missing/failed/unreadable report safely means
    False — paper-only.
    """
    try:
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(report, dict) and report.get("ready") is True


def build_certificate(
    report_path: Path | str = DEFAULT_GAUNTLET_REPORT,
) -> Optional[Dict[str, Any]]:
    """Verifiable research identity for a verdict.

    Ties the decision to the exact gauntlet report (byte hash) and config
    (hash embedded in the report) it was made under. The bot's executor
    re-hashes the report on ITS side of the mount and refuses any verdict
    whose certificate does not match — a bare ready:true can no longer
    make anything execution-eligible.
    """
    import hashlib

    try:
        raw = Path(report_path).read_bytes()
        report = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(report, dict):
        return None
    return {
        "engine": "meta_label_v2",
        "report_hash": hashlib.sha256(raw).hexdigest(),
        "config_hash": report.get("config_hash"),
        "universe": report.get("tickers"),
        "report_ready": report.get("ready") is True,
    }


def decide(
    signal: Dict[str, Any],
    *,
    cfg: Optional[dict] = None,
    fetch: FetchFn = fetch_bars,
    now: Optional[datetime] = None,
    gauntlet_report: Path | str = DEFAULT_GAUNTLET_REPORT,
) -> Dict[str, Any]:
    """Return a verdict dict for one normalized TV signal."""
    cfg = cfg or load_config()
    now = now or datetime.now(timezone.utc)
    ticker = str(signal.get("ticker", "")).upper()
    trend = str(signal.get("trend", "")).lower()
    direction = "long" if trend == "bullish" else "short"
    verdict: Dict[str, Any] = {
        "timestamp_utc": now.replace(microsecond=0).isoformat(),
        "ticker": ticker,
        "engine": "meta_label_v2",
        "side": direction,
        "verdict": "NO_TRADE",
        "reason": "",
        "horizon_days": int(cfg["barriers"]["max_hold"]),
        # Fill contract: v2 labels every trade from the NEXT session's open;
        # a live fill must follow the same rule (s0 below is the reference
        # price at decision time, never the assumed fill).
        "entry_rule": "next_session_open",
        # Provenance marker: true ONLY while the full validation gauntlet
        # (all six gates incl. the one-shot holdout) currently passes. The
        # bot's bridge treats anything without this as paper-only, never
        # executable.
        "gauntlet_pass": gauntlet_ready(gauntlet_report),
        # Verifiable identity: report hash + config hash + universe. The
        # executor re-hashes the report on its side and refuses mismatches.
        "certificate": build_certificate(gauntlet_report),
    }
    if direction == "short":
        verdict["reason"] = "short signal — cash account cannot short; recorded only"
        return verdict

    bars = fetch(ticker, float(cfg["universe"]["years"]))
    t = len(bars) - 1

    # Train on everything labeled strictly before today.
    sigs = signal_history(bars, cfg["signals"].get("tv_history_dir"))
    b = cfg["barriers"]
    labeled = triple_barrier_label(
        bars, [s for s in sigs if s.t + int(b["max_hold"]) + 1 < t],
        k_pt=float(b["k_pt"]), k_sl=float(b["k_sl"]),
        max_hold=int(b["max_hold"]), vol_window=int(b["vol_window"]),
        cost_per_side=float(cfg["costs"]["per_side"]))
    if len(labeled) < int(cfg["signals"]["min_per_symbol"]):
        verdict["reason"] = (f"insufficient labeled history for {ticker} "
                            f"({len(labeled)}) — collect more signals in DRY_RUN")
        return verdict
    X, kept = build_matrix(bars, [l.signal.t for l in labeled],
                           [l.signal.direction for l in labeled])
    y = np.array([labeled[i].label for i in kept])
    if len(np.unique(y)) < 2:
        verdict["reason"] = "degenerate label history (all one class)"
        return verdict
    model = fit_final_model(X, y)

    feats = feature_vector(bars, t, direction)
    if feats is None:
        verdict["reason"] = "feature warm-up — not enough history at signal time"
        return verdict
    p = float(model.predict_proba(feats.reshape(1, -1))[:, 1][0])
    verdict["p_profit_calibrated"] = p

    m = cfg["model"]
    entry = float(signal.get("price") or bars.close[t])
    sigma = realized_vol_daily(bars.close, int(b["vol_window"]))[t]
    stop = entry * (1.0 - float(b["k_sl"]) * float(sigma)
                    * np.sqrt(int(b["max_hold"])))
    risk = cfg["risk"]
    plan = shares_for(
        p=p, sizer=str(m["sizer"]), threshold=float(m["threshold"]),
        equity=float(risk["account_equity"]), risk_pct=float(risk["risk_pct"]),
        entry=entry, stop=stop,
        max_position_pct=float(risk["max_position_pct"]),
        buying_power=float(risk["account_equity"]),
    )
    verdict.update({
        "s0": entry,
        "structure": {"stop_pct": float(max(0.0, 1.0 - stop / entry)),
                      "stop_price": float(stop)},
        "sizing": {"shares": int(plan.shares), "notional": plan.notional,
                   "multiplier": plan.multiplier, "capped_by": plan.capped_by},
    })
    if p > float(m["threshold"]) and plan.shares >= 1:
        verdict["verdict"] = "TRADE"
        verdict["reason"] = (f"calibrated P(profit)={p:.3f} > "
                            f"{float(m['threshold']):.2f}; sizer "
                            f"{m['sizer']} multiplier {plan.multiplier:.2f}")
    else:
        verdict["reason"] = (f"calibrated P(profit)={p:.3f} <= threshold "
                            f"{float(m['threshold']):.2f}"
                            if p <= float(m["threshold"])
                            else "sized to zero shares")
    return verdict


def write_verdict(verdict: Dict[str, Any],
                  verdict_dir: Path | str = "outputs/verdicts") -> Path:
    d = Path(verdict_dir)
    d.mkdir(parents=True, exist_ok=True)
    ts = verdict["timestamp_utc"].replace(":", "").replace("-", "")
    path = d / f"{ts}_{verdict['ticker']}.json"
    path.write_text(json.dumps(verdict, indent=2, default=str) + "\n",
                    encoding="utf-8")
    return path


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Meta-label decision for the latest TradingView signal.")
    ap.add_argument("--signal-file", default="tv_data/latest_signal.json")
    ap.add_argument("--verdict-dir", default="outputs/verdicts")
    args = ap.parse_args(argv)
    signal = json.loads(Path(args.signal_file).read_text(encoding="utf-8"))
    verdict = decide(signal)
    path = write_verdict(verdict, args.verdict_dir)
    print(json.dumps(verdict, indent=2, default=str))
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
