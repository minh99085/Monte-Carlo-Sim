"""Primary signal stream — the trades the brain will judge.

The primary model is the SAME mechanical rule the TradingView Pine script
fires on: an EMA(9)/EMA(21) cross on daily closes (bullish flip → long
candidate, bearish flip → short candidate). Because the rule is mechanical,
its full historical alert stream is exactly reconstructible from price
history — which is what makes the spec's ">=100 signals per symbol" gate
satisfiable today instead of after years of live collection.

Live TradingView alerts saved by the webhook bridge (tv_data/history/*.json)
are merged in by (symbol, date) so the archive keeps growing and the live
stream gradually replaces reconstruction going forward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from signal_calibration import EMA_FAST_LEN, EMA_SLOW_LEN, compute_features
from trading_system.data import Bars


@dataclass(frozen=True)
class PrimarySignal:
    symbol: str
    t: int                    # bar index into the Bars it was built from
    date: pd.Timestamp
    direction: str            # "long" | "short"
    price: float              # close at signal bar
    source: str               # "reconstructed" | "tradingview"


def reconstruct_signals(bars: Bars) -> List[PrimarySignal]:
    """Historical EMA-cross flip events, exactly as the Pine script alerts.

    A signal exists at bar t when the fast/slow EMA relationship flips
    between t-1 and t (both EMAs defined at both bars). Uses only data up to
    t — the flip is observable at the close of t, matching the script's
    ``barstate.isconfirmed`` alert.
    """
    feats = compute_features(bars.close)
    trend = feats["trend"].to_numpy(dtype=object)
    out: List[PrimarySignal] = []
    warm = EMA_SLOW_LEN + 1
    for t in range(max(warm, 1), len(bars)):
        prev, cur = trend[t - 1], trend[t]
        if not isinstance(prev, str) or not isinstance(cur, str) or prev == cur:
            continue
        out.append(PrimarySignal(
            symbol=bars.symbol, t=t, date=bars.dates[t],
            direction="long" if cur == "bullish" else "short",
            price=float(bars.close[t]), source="reconstructed",
        ))
    return out


def load_tv_archive(history_dir: Path | str,
                    symbol: Optional[str] = None) -> List[dict]:
    """Read the webhook bridge's saved alert files (JSON, one per alert)."""
    d = Path(history_dir)
    if not d.is_dir():
        return []
    rows: List[dict] = []
    for p in sorted(d.glob("*.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(row, dict):
            continue
        if symbol and str(row.get("ticker", "")).upper() != symbol.upper():
            continue
        rows.append(row)
    return rows


def merge_live_signals(reconstructed: List[PrimarySignal], bars: Bars,
                       archive: List[dict]) -> List[PrimarySignal]:
    """Overlay live TV alerts onto the reconstructed stream.

    A live alert on the same (symbol, date) as a reconstructed flip replaces
    it (source becomes "tradingview"); live alerts on other dates are added
    when the date maps to a known bar. Result stays sorted by bar index.
    """
    by_date: Dict[pd.Timestamp, int] = {
        pd.Timestamp(d).normalize(): i for i, d in enumerate(bars.dates)}
    merged: Dict[int, PrimarySignal] = {s.t: s for s in reconstructed}
    for row in archive:
        ts = row.get("received_at_utc") or row.get("bar_time")
        trend = str(row.get("trend", "")).lower()
        if not ts or trend not in ("bullish", "bearish"):
            continue
        try:
            day = pd.Timestamp(ts).normalize().tz_localize(None)
        except (ValueError, TypeError):
            continue
        idx = by_date.get(day)
        if idx is None:
            continue
        merged[idx] = PrimarySignal(
            symbol=bars.symbol, t=idx, date=bars.dates[idx],
            direction="long" if trend == "bullish" else "short",
            price=float(row.get("price") or bars.close[idx]),
            source="tradingview",
        )
    return [merged[k] for k in sorted(merged)]


def signal_history(bars: Bars, tv_history_dir: Path | str | None = None
                   ) -> List[PrimarySignal]:
    recon = reconstruct_signals(bars)
    if tv_history_dir:
        archive = load_tv_archive(tv_history_dir, bars.symbol)
        return merge_live_signals(recon, bars, archive)
    return recon
