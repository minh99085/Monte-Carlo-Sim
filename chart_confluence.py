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


def combine(
    charts: List[Dict[str, Any]],
    position: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Combine per-chart summaries into one advisory stance + action verb.

    Each chart dict: role, bias, confidence, and optionally rsi + ticker.

    Flat (no ``position``): stance ALIGNED_LONG | MIXED | NO_TRADE |
    INCOMPLETE with action BUY | WAIT | AVOID | UPLOAD_MORE and a
    size_multiplier in (0, 1] when allowed (each caution flag halves it,
    and they stack).

    Holding (``position`` = the open paper position for this symbol): the
    question changes from "enter?" to "stay in?", so the same reads map to
    HOLDING/EXIT with action HOLD | SELL — exits on a bearish weekly regime
    (tide turned) or a bearish daily+ratio pair (deterioration). Long-only
    account: there is never a short action.
    """
    if position:
        return _combine_holding(charts, position)
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
            "action": "UPLOAD_MORE",
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
            "action": "AVOID",
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
            "action": "AVOID",
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
            "action": "AVOID",
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
            "action": "WAIT",
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
        "action": "BUY",
        "allowed": True,
        "size_multiplier": mult,
        "reasons": reasons,
        "warnings": warnings,
        "min_confidence": min_conf,
    }


def _combine_holding(
    charts: List[Dict[str, Any]],
    position: Dict[str, Any],
) -> Dict[str, Any]:
    """Already holding this symbol: HOLD or SELL, never a fresh BUY.

    Conservative by construction: unreliable or incomplete reads keep the
    position (HOLD with a warning) rather than acting on bad information;
    exits require a clear signal — the weekly tide turning bearish, or the
    daily AND ratio both bearish (deterioration while the tide is slack).
    """
    sym = str(position.get("symbol") or "").upper()
    by_role: Dict[str, Dict[str, Any]] = {}
    for c in charts or []:
        role = str(c.get("role", "")).strip().lower()
        if role in KNOWN_ROLES and role not in by_role:
            by_role[role] = c

    def hold(reasons: List[str], warnings: List[str],
             min_conf: Optional[float]) -> Dict[str, Any]:
        return {"stance": "HOLDING", "action": "HOLD", "allowed": False,
                "size_multiplier": 0.0, "reasons": reasons,
                "warnings": warnings, "min_confidence": min_conf}

    missing = [r for r in REQUIRED_ROLES if r not in by_role]
    if missing:
        return hold(
            [f"holding {sym}; missing chart(s) {', '.join(missing)} — "
             "keeping the position until a full read says otherwise"],
            ["upload weekly + daily charts to complete the review"], None)

    confs = [_to_float(by_role[r].get("confidence")) or 0.0 for r in by_role]
    min_conf = min(confs)
    if min_conf < MIN_CONFIDENCE:
        return hold(
            [f"holding {sym}; chart-read confidence {min_conf:.2f} is under "
             f"the {MIN_CONFIDENCE:.2f} floor — not exiting on unreliable "
             "reads"],
            ["re-upload clearer screenshots to complete the review"],
            min_conf)

    # Charts must be for the held symbol (ratio exempt).
    t_w = str(by_role["weekly"].get("ticker") or "").strip().upper()
    t_d = str(by_role["daily"].get("ticker") or "").strip().upper()
    charted = {t for t in (t_w, t_d) if t}
    if sym and charted and sym not in charted:
        return hold(
            [f"position is {sym} but charts read as "
             f"{'/'.join(sorted(charted))} — review not applied"],
            [f"upload {sym} charts to review this position"], min_conf)

    weekly_bias = _norm_bias(by_role["weekly"].get("bias"))
    daily_bias = _norm_bias(by_role["daily"].get("bias"))
    ratio = by_role.get("ratio")
    ratio_bias = _norm_bias(ratio.get("bias")) if ratio is not None else None

    def sell(reason: str, warnings: List[str]) -> Dict[str, Any]:
        return {"stance": "EXIT", "action": "SELL", "allowed": False,
                "size_multiplier": 0.0, "reasons": [reason],
                "warnings": warnings, "min_confidence": min_conf}

    if weekly_bias == "bearish":
        return sell(
            f"weekly regime turned bearish — the tide that justified "
            f"holding {sym} is gone; exit and hold cash", [])

    if daily_bias == "bearish" and ratio_bias == "bearish":
        return sell(
            f"{sym} is deteriorating: daily timing bearish AND it is "
            "lagging SPY — exit before the weekly turns", [])

    reasons = [f"weekly regime is {weekly_bias} and daily is {daily_bias} — "
               "thesis intact, keep holding"]
    warnings: List[str] = []
    if daily_bias == "bearish":
        warnings.append("daily timing is bearish but the ratio holds up — "
                        "watch closely; re-review in 2–3 trading days")
    return hold(reasons, warnings, min_conf)
