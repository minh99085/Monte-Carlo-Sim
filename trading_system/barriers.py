"""Triple-barrier labeling of primary signals (López de Prado).

For each historical signal at bar t:
  upper barrier  = entry * (1 + k_pt * sigma_t * sqrt(max_hold))
  lower barrier  = entry * (1 - k_sl * sigma_t * sqrt(max_hold))
  vertical       = t + max_hold bars
Entry is the NEXT bar's open (executable, not the signal close — the same
fill discipline validate_edge.py enforces). The path is walked bar by bar
with highs/lows; the meta-label is 1 iff the trade PROFITS net of costs —
profit barrier first for the trade's direction, with the vertical barrier
settled at the exit close. For shorts the barrier roles invert.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from trading_system.data import Bars, realized_vol_daily
from trading_system.primary import PrimarySignal


@dataclass(frozen=True)
class LabeledSignal:
    signal: PrimarySignal
    entry_price: float          # next-bar open
    exit_index: int
    exit_price: float
    exit_reason: str            # "profit_barrier" | "stop_barrier" | "vertical"
    gross_return: float         # signed, in the trade's direction
    net_return: float           # after per-side costs (2 sides)
    label: int                  # 1 iff net_return > 0


def triple_barrier_label(
    bars: Bars,
    signals: List[PrimarySignal],
    *,
    k_pt: float = 1.5,
    k_sl: float = 1.5,
    max_hold: int = 10,
    vol_window: int = 60,
    cost_per_side: float = 0.0005,
) -> List[LabeledSignal]:
    sigma = realized_vol_daily(bars.close, vol_window)
    n = len(bars)
    out: List[LabeledSignal] = []
    horizon_scale = math.sqrt(max_hold)
    for s in signals:
        t = s.t
        if t + 1 >= n or not np.isfinite(sigma[t]) or sigma[t] <= 0:
            continue
        entry_i = t + 1
        entry = float(bars.open[entry_i])
        if entry <= 0:
            continue
        width_up = k_pt * sigma[t] * horizon_scale
        width_dn = k_sl * sigma[t] * horizon_scale
        if s.direction == "long":
            profit_lvl = entry * (1.0 + width_up)
            stop_lvl = entry * (1.0 - width_dn)
        else:
            profit_lvl = entry * (1.0 - width_up)
            stop_lvl = entry * (1.0 + width_dn)
        last_i = min(entry_i + max_hold, n - 1)
        exit_i, exit_px, reason = last_i, float(bars.close[last_i]), "vertical"
        for i in range(entry_i, last_i + 1):
            hi, lo = float(bars.high[i]), float(bars.low[i])
            if s.direction == "long":
                hit_stop = lo <= stop_lvl
                hit_profit = hi >= profit_lvl
            else:
                hit_stop = hi >= stop_lvl
                hit_profit = lo <= profit_lvl
            # Conservative same-bar tie-break: assume the STOP traded first.
            if hit_stop:
                exit_i, exit_px, reason = i, stop_lvl, "stop_barrier"
                break
            if hit_profit:
                exit_i, exit_px, reason = i, profit_lvl, "profit_barrier"
                break
        raw = exit_px / entry - 1.0
        gross = raw if s.direction == "long" else -raw
        net = (1.0 + gross) * (1.0 - cost_per_side) ** 2 - 1.0
        out.append(LabeledSignal(
            signal=s, entry_price=entry, exit_index=exit_i,
            exit_price=exit_px, exit_reason=reason,
            gross_return=float(gross), net_return=float(net),
            label=int(net > 0.0),
        ))
    return out
