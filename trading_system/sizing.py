"""Bet sizing — multipliers in [0, 1] on the fixed-% risk base size.

Implements the two baselines the spec orders first:

* **A4** — all-or-nothing: 1 above the probability threshold, else 0.
* **B1** — de Prado bet size: ``norm.cdf((p - 0.5) / sqrt(p * (1 - p)))``,
  hard-zeroed below 0.5 (never bet against the model).

The final share count applies the multiplier to the fixed-% risk formula
``shares = (equity * risk_pct) / (entry - stop)`` and caps by position
limit and buying power.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def size_A4(p: float, threshold: float = 0.60) -> float:
    """All-or-nothing at the calibrated-probability threshold."""
    return 1.0 if p > threshold else 0.0


def size_B1(p: float, threshold: float = 0.5) -> float:
    """de Prado bet size; zero at/below 0.5 (and below any higher gate)."""
    if p <= max(0.5, threshold - 1e-12) or p >= 1.0:
        return 0.0 if p < 1.0 else 1.0
    z = (p - 0.5) / math.sqrt(p * (1.0 - p))
    return float(2.0 * _norm_cdf(z) - 1.0)


SIZERS = {"A4": size_A4, "B1": size_B1}


@dataclass(frozen=True)
class PositionPlan:
    shares: float
    notional: float
    risk_dollars: float
    multiplier: float
    capped_by: str          # "none" | "position_pct" | "buying_power"


def shares_for(
    *,
    p: float,
    sizer: str,
    threshold: float,
    equity: float,
    risk_pct: float,
    entry: float,
    stop: float,
    max_position_pct: float,
    buying_power: float,
) -> PositionPlan:
    mult = SIZERS[sizer](p, threshold)
    per_share_risk = abs(entry - stop)
    if mult <= 0.0 or per_share_risk <= 0 or entry <= 0:
        return PositionPlan(0.0, 0.0, 0.0, mult, "none")
    base = (equity * risk_pct) / per_share_risk
    shares = base * mult
    capped_by = "none"
    max_shares_pos = (equity * max_position_pct) / entry
    if shares > max_shares_pos:
        shares, capped_by = max_shares_pos, "position_pct"
    max_shares_bp = buying_power / entry
    if shares > max_shares_bp:
        shares, capped_by = max_shares_bp, "buying_power"
    shares = float(max(shares, 0.0))
    return PositionPlan(
        shares=shares,
        notional=float(shares * entry),
        risk_dollars=float(shares * per_share_risk),
        multiplier=float(mult),
        capped_by=capped_by,
    )
