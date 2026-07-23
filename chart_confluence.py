"""Multi-chart confluence: combine per-chart vision reads into one stance.

The chart battery for ONE asset (weekly/monthly holding system):

    role "weekly"  1W bars, ~3y   regime   "is the tide rising?"
    role "daily"   1D bars, ~9mo  timing   "is now a sane entry?"
    role "ratio"   ASSET/SPY 1D   strength "is it beating the market?" (optional)

Rules are pre-specified and have NO tunable knobs — this module must never
become another overfitting surface. Long-only (cash account). A trade stance
requires weekly AND daily to agree bullish; the ratio chart and RSI can only
scale size DOWN, never up. Any missing or low-confidence read degrades toward
NO_TRADE. Output is advisory + paper only: it feeds the dashboard, never an
order path.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Same floor the bot's vision config uses (min_overall_confidence): below
# this the image read is too unreliable to act on at all.
MIN_CONFIDENCE = 0.45
# Daily RSI above this = chasing an extended move; halve size.
RSI_OVERBOUGHT = 75.0

REQUIRED_ROLES = ("weekly", "daily")
KNOWN_ROLES = ("weekly", "daily", "ratio")


def _norm_bias(value: Any) -> str:
    b = str(value or "").strip().lower()
    return b if b in ("bullish", "bearish", "neutral") else "neutral"


def _to_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # drop NaN


def combine(charts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine per-chart summaries into one advisory stance.

    Each chart dict: role, bias, confidence, and optionally rsi + ticker.
    Returns stance ALIGNED_LONG | MIXED | NO_TRADE | INCOMPLETE with a
    size_multiplier in (0, 1] when allowed (each caution flag halves it,
    and they stack) and plain-language reasons.
    """
    by_role: Dict[str, Dict[str, Any]] = {}
    for c in charts or []:
        role = str(c.get("role", "")).strip().lower()
        if role in KNOWN_ROLES and role not in by_role:
            by_role[role] = c

    reasons: List[str] = []
    warnings: List[str] = []

    missing = [r for r in REQUIRED_ROLES if r not in by_role]
    if missing:
        return {
            "stance": "INCOMPLETE",
            "allowed": False,
            "size_multiplier": 0.0,
            "reasons": [f"missing required chart(s): {', '.join(missing)}"],
            "warnings": [],
            "min_confidence": None,
        }

    confs = [
        _to_float(by_role[r].get("confidence")) or 0.0 for r in by_role
    ]
    min_conf = min(confs)
    if min_conf < MIN_CONFIDENCE:
        return {
            "stance": "NO_TRADE",
            "allowed": False,
            "size_multiplier": 0.0,
            "reasons": [
                f"lowest chart-read confidence {min_conf:.2f} is under the "
                f"{MIN_CONFIDENCE:.2f} floor — image too unreliable to act on"
            ],
            "warnings": warnings,
            "min_confidence": min_conf,
        }

    # Weekly vs daily must be the same symbol (the ratio chart is exempt —
    # its symbol is legitimately ASSET/SPY).
    t_w = str(by_role["weekly"].get("ticker") or "").strip().upper()
    t_d = str(by_role["daily"].get("ticker") or "").strip().upper()
    if t_w and t_d and t_w != t_d:
        return {
            "stance": "NO_TRADE",
            "allowed": False,
            "size_multiplier": 0.0,
            "reasons": [
                f"weekly chart reads as {t_w} but daily reads as {t_d} — "
                "not the same asset; re-upload matching charts"
            ],
            "warnings": warnings,
            "min_confidence": min_conf,
        }

    weekly_bias = _norm_bias(by_role["weekly"].get("bias"))
    daily_bias = _norm_bias(by_role["daily"].get("bias"))

    # Regime gate: long-only account, so a bearish tide ends the discussion.
    if weekly_bias == "bearish":
        return {
            "stance": "NO_TRADE",
            "allowed": False,
            "size_multiplier": 0.0,
            "reasons": ["weekly regime is bearish — long-only account stays "
                        "in cash when the tide is falling"],
            "warnings": warnings,
            "min_confidence": min_conf,
        }

    # Timing gate: only a bullish daily read on a non-bearish weekly trades.
    if daily_bias != "bullish":
        return {
            "stance": "MIXED",
            "allowed": False,
            "size_multiplier": 0.0,
            "reasons": [f"weekly regime is {weekly_bias} but daily timing "
                        f"reads {daily_bias} — wait for agreement"],
            "warnings": warnings,
            "min_confidence": min_conf,
        }

    mult = 1.0
    reasons.append("weekly regime and daily timing agree bullish")
    if weekly_bias == "neutral":
        mult *= 0.5
        reasons.append("weekly regime only neutral — half size")

    rsi_d = _to_float(by_role["daily"].get("rsi"))
    if rsi_d is not None and rsi_d > RSI_OVERBOUGHT:
        mult *= 0.5
        warnings.append(
            f"daily RSI {rsi_d:.0f} is overbought (>{RSI_OVERBOUGHT:.0f}) — "
            "chasing risk, size halved"
        )

    ratio = by_role.get("ratio")
    if ratio is not None:
        ratio_bias = _norm_bias(ratio.get("bias"))
        if ratio_bias == "bearish":
            mult *= 0.5
            warnings.append("asset is lagging SPY (ratio chart bearish) — "
                            "size halved")
        elif ratio_bias == "bullish":
            reasons.append("asset is leading SPY (ratio chart bullish)")

    return {
        "stance": "ALIGNED_LONG",
        "allowed": True,
        "size_multiplier": mult,
        "reasons": reasons,
        "warnings": warnings,
        "min_confidence": min_conf,
    }
