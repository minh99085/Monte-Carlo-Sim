"""
Map chart vision extraction + MCP market data → Monte Carlo tactical parameters.

Safety principles
-----------------
* Image-derived numbers are *soft* signals only.
* Starting price and realized volatility prefer MCP (or yfinance fallback).
* Stop/TP from chart levels are clamped to safe ranges and scaled by confidence.
* Hard risk limits (notional, daily loss, PDT) remain outside this module —
  they live in Robinhood ``SafeRobinhoodClient``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from chart_vision_models import (
    Action,
    Bias,
    ChartExtractionResult,
    ChartTradeDecision,
    LevelKind,
    MCPMarketSnapshot,
    PositionSizeSuggestion,
    RiskSummary,
    ValidationResult,
    ValidationStatus,
)
from tactical_config import TacticalConfig, TradingRule, preset_5_day
from tv_integration import apply_tradingview_to_tactical, trend_to_side


# Default clamps for image-suggested stops/targets (fractions of entry).
MIN_STOP_PCT = 0.005   # 0.5%
MAX_STOP_PCT = 0.08    # 8%
MIN_TP_PCT = 0.005
MAX_TP_PCT = 0.15
DEFAULT_STOP_PCT = 0.02
DEFAULT_TP_PCT = 0.03

# ADX(14) below this = no real trend → MC drift forced mean-neutral.
# 20-25 is the classic Wilder "trend is tradeable" threshold; 20 is the gate.
ADX_TREND_MIN = 20.0

# ATR(14)-based adaptive stop: stop distance = ATR_STOP_MULT × ATR. A 2×ATR
# stop sits outside normal daily noise so trades aren't shaken out early.
ATR_STOP_MULT = 2.0

# Soft drift magnitudes when no calibration table is available.
# Image path is secondary; keep conservative.
BIAS_DRIFT_ANNUAL = {
    Bias.BULLISH: 0.05,
    Bias.BEARISH: -0.05,
    Bias.NEUTRAL: 0.0,
    Bias.UNCLEAR: 0.0,
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def levels_to_stop_tp(
    extraction: ChartExtractionResult,
    entry_price: float,
    side: str,
    *,
    confidence: float = 0.5,
    atr_abs: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], List[str]]:
    """
    Suggest stop/TP percentages (and absolute prices) from chart levels.

    ``atr_abs`` is ATR(14) in price units (from real bars). When present it
    drives an adaptive stop: with no usable chart level the stop is
    ``ATR_STOP_MULT × ATR`` (volatility-scaled, replacing the fixed default),
    and any chart-derived stop is floored at 1×ATR so it never sits inside
    normal daily noise.

    Returns
    -------
    stop_pct, tp_pct, stop_price, tp_price, notes
    """
    notes: List[str] = []
    if entry_price <= 0:
        return None, None, None, None, ["entry_price invalid; no level mapping"]

    supports = [lv for lv in extraction.levels if lv.kind == LevelKind.SUPPORT]
    resistances = [lv for lv in extraction.levels if lv.kind == LevelKind.RESISTANCE]

    stop_price: Optional[float] = None
    tp_price: Optional[float] = None

    side = side.lower()
    if side == "long":
        below = [lv for lv in supports if lv.price < entry_price]
        above = [lv for lv in resistances if lv.price > entry_price]
        if below:
            # Nearest strong support below entry
            below.sort(key=lambda lv: (-lv.strength, entry_price - lv.price))
            stop_price = float(below[0].price)
            notes.append(f"Stop from nearest support {stop_price}")
        if above:
            above.sort(key=lambda lv: (-lv.strength, lv.price - entry_price))
            tp_price = float(above[0].price)
            notes.append(f"TP from nearest resistance {tp_price}")
    elif side == "short":
        above = [lv for lv in resistances if lv.price > entry_price]
        below = [lv for lv in supports if lv.price < entry_price]
        if above:
            above.sort(key=lambda lv: (-lv.strength, lv.price - entry_price))
            stop_price = float(above[0].price)
            notes.append(f"Stop from nearest resistance {stop_price}")
        if below:
            below.sort(key=lambda lv: (-lv.strength, entry_price - lv.price))
            tp_price = float(below[0].price)
            notes.append(f"TP from nearest support {tp_price}")

    stop_pct: Optional[float] = None
    tp_pct: Optional[float] = None

    atr_pct = (atr_abs / entry_price) if (atr_abs and atr_abs > 0) else None
    if stop_price is not None:
        stop_pct = abs(entry_price - stop_price) / entry_price
        # Widen stop slightly when confidence is low (avoid tight image noise)
        conf_scale = 1.0 + (1.0 - _clamp(confidence, 0.0, 1.0)) * 0.5
        stop_pct = stop_pct * conf_scale
        # Never let a chart-derived stop sit inside one ATR of normal noise.
        if atr_pct is not None and stop_pct < atr_pct:
            notes.append(
                f"chart stop {stop_pct:.4f} < 1×ATR {atr_pct:.4f} → widened to ATR"
            )
            stop_pct = atr_pct
        stop_pct = _clamp(stop_pct, MIN_STOP_PCT, MAX_STOP_PCT)
        notes.append(f"stop_pct={stop_pct:.4f} (confidence-scaled, ATR-floored)")
    elif atr_pct is not None:
        stop_pct = _clamp(ATR_STOP_MULT * atr_pct, MIN_STOP_PCT, MAX_STOP_PCT)
        notes.append(
            f"No usable stop level; adaptive stop = {ATR_STOP_MULT:g}×ATR "
            f"({atr_pct:.4f}) → stop_pct={stop_pct:.4f}"
        )
    else:
        stop_pct = DEFAULT_STOP_PCT
        notes.append(f"No usable stop level or ATR; default stop_pct={stop_pct}")

    if tp_price is not None:
        tp_pct = abs(tp_price - entry_price) / entry_price
        tp_pct = _clamp(tp_pct, MIN_TP_PCT, MAX_TP_PCT)
        notes.append(f"tp_pct={tp_pct:.4f}")
    else:
        # Asymmetric default: modest TP relative to stop
        tp_pct = max(DEFAULT_TP_PCT, stop_pct * 1.5 if stop_pct else DEFAULT_TP_PCT)
        tp_pct = _clamp(tp_pct, MIN_TP_PCT, MAX_TP_PCT)
        notes.append(f"No usable TP level; default tp_pct={tp_pct}")

    # Recompute absolute prices from clamped percentages for consistency
    if side == "long":
        stop_price = entry_price * (1.0 - stop_pct)
        tp_price = entry_price * (1.0 + tp_pct)
    elif side == "short":
        stop_price = entry_price * (1.0 + stop_pct)
        tp_price = entry_price * (1.0 - tp_pct)

    return stop_pct, tp_pct, stop_price, tp_price, notes


def bias_and_macd_to_side(extraction: ChartExtractionResult) -> Optional[str]:
    """Preferred trade side from vision bias + MACD cross (soft)."""
    side = trend_to_side(extraction.bias.value)
    macd_cross = (extraction.indicators.macd.cross or "").lower()
    if side is None:
        if macd_cross == "bullish_cross":
            return "long"
        if macd_cross == "bearish_cross":
            return "short"
        return None
    # Mild confirmation: if MACD strongly opposes, leave side but caller may flat
    return side


def soft_drift_from_vision(
    extraction: ChartExtractionResult,
    *,
    adjusted_confidence: float,
) -> Tuple[float, str]:
    """
    Conservative conditional drift when calibration tables are unavailable.

    Drift is scaled by adjusted confidence so low-confidence charts do not
    invent edge.
    """
    base = BIAS_DRIFT_ANNUAL.get(extraction.bias, 0.0)
    # RSI extremes gently tilt drift
    rsi = extraction.indicators.rsi.value
    tilt = 0.0
    if rsi is not None:
        if rsi >= 70:
            tilt = -0.01  # overbought → slight mean-reversion dampener for long bias
        elif rsi <= 30:
            tilt = 0.01
    mu = (base + tilt) * _clamp(adjusted_confidence, 0.0, 1.0)
    reason = (
        f"soft vision drift base={base:+.2%} tilt={tilt:+.2%} "
        f"× conf={adjusted_confidence:.2f} → μ={mu:+.2%}/yr"
    )
    return float(mu), reason


def vol_scale_from_indicators(extraction: ChartExtractionResult) -> float:
    """
    Soft volatility scale from RSI extremity / MACD energy.
    Centered at 1.0; clamped to [0.7, 1.4].
    """
    rsi = extraction.indicators.rsi.value
    scale = 1.0
    if rsi is not None:
        extremity = abs(float(rsi) - 50.0) / 50.0
        scale *= 1.0 + 0.25 * extremity
    hist = extraction.indicators.macd.histogram
    if hist is not None:
        # Large absolute histogram → slightly higher vol regime
        scale *= 1.0 + min(0.15, abs(float(hist)) * 0.01)
    return _clamp(scale, 0.7, 1.4)


def map_to_tactical(
    extraction: ChartExtractionResult,
    mcp: Optional[MCPMarketSnapshot] = None,
    validation: Optional[ValidationResult] = None,
    *,
    paths: int = 100_000,
    horizon_days: int = 5,
    seed: Optional[int] = 42,
    use_calibration_drift: bool = True,
    calibration_dir: str = "calibration",
    force_flat_on_reject: bool = True,
) -> Tuple[TacticalConfig, TradingRule, Dict[str, Any]]:
    """
    Convert vision + MCP into a ``TacticalConfig`` + ``TradingRule``.

    Returns ``(tactical, rule, meta)`` where ``meta`` holds mapping notes and
    the intermediate TV-compatible signal dict.
    """
    notes: List[str] = []
    adj_conf = (
        validation.adjusted_confidence
        if validation is not None
        else extraction.confidence.overall
    )
    status = validation.status if validation is not None else ValidationStatus.SKIPPED

    # --- Authoritative price / vol from MCP when present ---
    start_price: Optional[float] = None
    if mcp and mcp.last_price and mcp.last_price > 0:
        start_price = float(mcp.last_price)
        notes.append(f"Starting price from MCP: {start_price}")
    elif extraction.image_last_price and extraction.image_last_price > 0:
        start_price = float(extraction.image_last_price)
        notes.append(
            f"Starting price from IMAGE only (lower confidence): {start_price}"
        )

    realized_vol = mcp.realized_vol_annual if mcp else None
    vol_scale = vol_scale_from_indicators(extraction)
    annual_vol: Optional[float] = None
    if realized_vol is not None and realized_vol > 0:
        annual_vol = float(realized_vol) * vol_scale
        notes.append(
            f"Vol from MCP realized {realized_vol:.4f} × scale {vol_scale:.3f} "
            f"→ {annual_vol:.4f}"
        )
    else:
        notes.append(f"No MCP realized vol; vol scale={vol_scale:.3f} reserved for later")

    side = bias_and_macd_to_side(extraction)
    if status == ValidationStatus.REJECTED and force_flat_on_reject:
        side = None
        notes.append("Validation REJECTED → force flat (no side)")

    stop_pct = DEFAULT_STOP_PCT
    tp_pct = DEFAULT_TP_PCT
    stop_price = tp_price = None
    atr_abs = None
    if mcp and mcp.computed_indicators:
        atr_abs = mcp.computed_indicators.get("atr14")
        try:
            atr_abs = float(atr_abs) if atr_abs is not None else None
        except (TypeError, ValueError):
            atr_abs = None
    if side and start_price:
        stop_pct, tp_pct, stop_price, tp_price, lv_notes = levels_to_stop_tp(
            extraction, start_price, side, confidence=adj_conf, atr_abs=atr_abs
        )
        notes.extend(lv_notes)
        stop_pct = stop_pct or DEFAULT_STOP_PCT
        tp_pct = tp_pct or DEFAULT_TP_PCT

    rule = TradingRule(
        name=f"Chart vision {extraction.bias.value}",
        entry_condition=(
            f"Enter {side or 'flat'} from chart vision bias={extraction.bias.value}"
        ),
        exit_condition=f"Exit at max hold ({horizon_days}d) or stop/TP",
        stop_loss_pct=float(stop_pct),
        max_holding_days=int(horizon_days),
        side=side,
        entry_day=0,
        take_profit_pct=float(tp_pct) if tp_pct else None,
        notes="Derived from chart vision; image levels are soft only",
    ).validate()

    tactical = preset_5_day(ticker=extraction.ticker).with_rule(rule)
    tactical = replace(
        tactical,
        ticker=extraction.ticker.upper(),
        horizon_days=int(horizon_days),
        paths=int(paths),
        seed=seed,
        starting_price=start_price,
        annual_volatility=annual_vol,
        annual_drift=0.0,
        notes="chart_vision mapped config",
    )

    # Build TV-compatible signal; prefer MCP price in the signal for drift path
    signal = extraction.to_tv_signal_dict()
    if start_price is not None:
        signal["price"] = start_price
        signal["close"] = start_price

    # Apply shared TV wiring (side alignment, optional calibrated drift)
    if use_calibration_drift and status != ValidationStatus.REJECTED:
        tactical, rule, tv_ctx, jump_m = apply_tradingview_to_tactical(
            tactical,
            rule,
            signal,
            use_ticker=True,
            use_price=True,
            align_side_to_trend=side is not None,
            filter_against_trend=False,
            scale_vol_by_momentum=False,
            scale_jumps_by_momentum=True,
            signal_path="chart_vision",
            base_sigma=annual_vol,
            use_signal_drift=True,
            drift_horizon_days=horizon_days,
            calibration_dir=calibration_dir,
        )
        notes.extend(tv_ctx.notes)
        # If calibration produced no drift, fall back to soft vision drift
        if tv_ctx.drift_estimate is None or tv_ctx.drift_estimate.source != "calibration":
            mu, reason = soft_drift_from_vision(extraction, adjusted_confidence=adj_conf)
            tactical = replace(tactical, annual_drift=mu)
            notes.append(reason)
        meta_jump = jump_m
    else:
        mu, reason = soft_drift_from_vision(extraction, adjusted_confidence=adj_conf)
        if status == ValidationStatus.REJECTED:
            mu = 0.0
            notes.append("Rejected validation → drift forced to 0")
        tactical = replace(tactical, annual_drift=mu)
        notes.append(reason)
        meta_jump = 1.0

    # --- ADX(14) trend-strength gate on the drift ---------------------------
    # A directional Monte-Carlo drift is only trustworthy when a real trend
    # exists. ADX < ADX_TREND_MIN means the market is choppy/rangebound, so any
    # bias-derived drift is likely noise — force the MC mean-neutral rather
    # than betting on a trend that isn't there. ADX >= threshold leaves the
    # drift intact. Computed from real OHLC bars on the bot side.
    adx = (mcp.computed_indicators or {}).get("adx14") if mcp else None
    adx_gated = False
    if adx is not None:
        try:
            adx_f = float(adx)
        except (TypeError, ValueError):
            adx_f = None
        if adx_f is not None:
            if adx_f < ADX_TREND_MIN and tactical.annual_drift != 0.0:
                tactical = replace(tactical, annual_drift=0.0)
                adx_gated = True
                notes.append(
                    f"ADX {adx_f:.1f} < {ADX_TREND_MIN:.0f} (no trend) → drift "
                    f"forced to 0 (mean-neutral MC; don't bet on noise)"
                )
            else:
                notes.append(
                    f"ADX {adx_f:.1f} ≥ {ADX_TREND_MIN:.0f} → trend confirmed, "
                    f"drift retained ({tactical.annual_drift:+.2%}/yr)"
                )

    # If side is None (flat), attach a no-op style rule that still validates
    if side is None:
        rule = replace(
            rule,
            side="long",
            entry_condition="No entry (flat recommendation)",
            notes="flat — validation or unclear bias",
            stop_loss_pct=DEFAULT_STOP_PCT,
            take_profit_pct=DEFAULT_TP_PCT,
        )
        # Mark flat in meta; decision layer will set Action.FLAT
        notes.append("No trade side — recommendation will be flat")

    meta: Dict[str, Any] = {
        "notes": notes,
        "signal": signal,
        "side": side,
        "stop_pct": stop_pct,
        "tp_pct": tp_pct,
        "stop_price": stop_price,
        "tp_price": tp_price,
        "adjusted_confidence": adj_conf,
        "validation_status": status.value if hasattr(status, "value") else str(status),
        "jump_multiplier": meta_jump,
        "vol_scale": vol_scale,
        "adx": adx,
        "adx_gated": adx_gated,
    }
    return tactical.validate(), rule.validate(), meta


def suggest_position_size(
    *,
    entry_price: float,
    stop_loss_pct: float,
    equity: Optional[float],
    buying_power: Optional[float],
    max_order_notional_usd: float = 100.0,
    max_position_pct: float = 10.0,
    risk_per_trade_pct: float = 0.5,
    adjusted_confidence: float = 0.5,
    min_confidence_for_size: float = 0.35,
) -> PositionSizeSuggestion:
    """
    Risk-budget position sizing. Confidence scales size down; never exceeds
    configured notional / concentration caps.
    """
    caps: List[str] = []
    notes: List[str] = []

    if entry_price <= 0 or stop_loss_pct <= 0:
        return PositionSizeSuggestion(
            method="none",
            notes=["invalid entry/stop for sizing"],
        )

    if adjusted_confidence < min_confidence_for_size:
        return PositionSizeSuggestion(
            method="confidence_block",
            notes=[
                f"confidence {adjusted_confidence:.2f} < min {min_confidence_for_size}"
            ],
        )

    # Dollar risk budget from equity
    eq = float(equity) if equity and equity > 0 else None
    if eq is None:
        # Fall back to max notional only
        risk_budget = max_order_notional_usd * stop_loss_pct
        notes.append("No equity; sizing from max notional only")
        method = "max_notional_fallback"
    else:
        risk_budget = eq * (risk_per_trade_pct / 100.0)
        method = "fixed_fractional_risk"
        notes.append(
            f"Risk budget = equity ${eq:.2f} × {risk_per_trade_pct}% = ${risk_budget:.2f}"
        )

    # shares such that stop loss ≈ risk budget
    raw_notional = risk_budget / stop_loss_pct
    conf_scale = _clamp(adjusted_confidence, 0.0, 1.0)
    raw_notional *= conf_scale
    notes.append(f"Confidence scale ×{conf_scale:.2f} on notional")

    notional = raw_notional
    if notional > max_order_notional_usd:
        notional = max_order_notional_usd
        caps.append("max_order_notional")

    if eq is not None:
        conc_cap = eq * (max_position_pct / 100.0)
        if notional > conc_cap:
            notional = conc_cap
            caps.append("max_position_pct")

    if buying_power is not None and buying_power > 0 and notional > buying_power:
        notional = float(buying_power)
        caps.append("buying_power")

    shares = notional / entry_price if entry_price > 0 else 0.0
    return PositionSizeSuggestion(
        shares=float(max(0.0, shares)),
        notional_usd=float(max(0.0, notional)),
        risk_budget_usd=float(risk_budget),
        method=method,
        capped_by=caps,
        notes=notes,
    )


def risk_summary_from_tactical_stats(
    stats: Dict[str, Any],
    pnl_pct: Optional[Any] = None,
) -> RiskSummary:
    """Build RiskSummary from ``TacticalResult.stats`` (+ optional pnl_pct array).

    ``var_*_pct`` / ``es_*_pct`` are expressed in **return fraction** units
    (same as ``pnl_pct``), not absolute dollars.
    """
    import numpy as np

    var_95 = None
    var_99 = None
    es_95 = None
    if pnl_pct is not None:
        arr = np.asarray(pnl_pct, dtype=np.float64)
        if arr.size:
            var_95 = float(np.percentile(arr, 5))
            var_99 = float(np.percentile(arr, 1))
            # ES 95: mean of outcomes at or below 5th percentile
            tail = arr[arr <= var_95]
            es_95 = float(np.mean(tail)) if tail.size else var_95

    return RiskSummary(
        n_paths=int(stats.get("n_paths") or 0),
        prob_profit=stats.get("prob_profit"),
        prob_loss=stats.get("prob_loss"),
        prob_flat=stats.get("prob_flat"),
        avg_pnl_pct=stats.get("avg_pnl_pct"),
        median_pnl_pct=stats.get("median_pnl_pct"),
        worst_pnl_pct=stats.get("worst_pnl_pct"),
        best_pnl_pct=stats.get("best_pnl_pct"),
        var_95_pct=var_95,
        var_99_pct=var_99,
        es_95_pct=es_95,
        ruin_probability=stats.get("stop_hit_rate"),
        stop_hit_rate=stats.get("stop_hit_rate"),
        take_profit_rate=stats.get("take_profit_rate"),
        pnl_p05=stats.get("pnl_p05"),
        pnl_p95=stats.get("pnl_p95"),
        extra={
            k: stats[k]
            for k in (
                "avg_pnl",
                "median_pnl",
                "std_pnl",
                "avg_holding_days",
                "frac_paths_with_trade",
            )
            if k in stats
        },
    )


def decide_action(
    *,
    side: Optional[str],
    validation: Optional[ValidationResult],
    risk: RiskSummary,
    min_prob_profit: float = 0.48,
    max_stop_hit_rate: float = 0.65,
) -> Action:
    """Map side + risk filters into long/short/flat."""
    if validation is not None and validation.status == ValidationStatus.REJECTED:
        return Action.FLAT
    if side not in ("long", "short"):
        return Action.FLAT
    if risk.prob_profit is not None and risk.prob_profit < min_prob_profit:
        return Action.FLAT
    if risk.stop_hit_rate is not None and risk.stop_hit_rate > max_stop_hit_rate:
        return Action.FLAT
    return Action.LONG if side == "long" else Action.SHORT


def build_decision(
    *,
    extraction: ChartExtractionResult,
    tactical: TacticalConfig,
    rule: TradingRule,
    meta: Dict[str, Any],
    risk: RiskSummary,
    validation: Optional[ValidationResult] = None,
    mcp: Optional[MCPMarketSnapshot] = None,
    position_size: Optional[PositionSizeSuggestion] = None,
    execution_mode: str = "recommendation_only",
    min_prob_profit: float = 0.48,
) -> ChartTradeDecision:
    """Assemble the final structured decision object."""
    side = meta.get("side")
    action = decide_action(
        side=side,
        validation=validation,
        risk=risk,
        min_prob_profit=min_prob_profit,
    )
    adj = float(meta.get("adjusted_confidence") or extraction.confidence.overall)
    status = (
        validation.status if validation is not None else ValidationStatus.SKIPPED
    )

    executable = (
        execution_mode == "gated_execution"
        and action != Action.FLAT
        and status in (ValidationStatus.PASSED, ValidationStatus.DOWNWEIGHTED)
        and adj >= 0.5
    )

    rationale_parts = [
        f"Bias={extraction.bias.value} (vision conf={extraction.confidence.overall:.2f}, "
        f"adjusted={adj:.2f}).",
        f"Validation={status.value}.",
        f"MC paths={risk.n_paths:,}, P(profit)={risk.prob_profit}, "
        f"VaR95={risk.var_95_pct}, stop_hit={risk.stop_hit_rate}.",
        f"Action={action.value}.",
    ]
    if extraction.extraction_warnings:
        rationale_parts.append(
            "Vision warnings: " + "; ".join(extraction.extraction_warnings[:5])
        )

    return ChartTradeDecision(
        ticker=tactical.ticker,
        action=action,
        suggested_side=side if side in ("long", "short") else None,
        position_size=position_size or PositionSizeSuggestion(),
        stop_loss_pct=rule.stop_loss_pct,
        take_profit_pct=rule.take_profit_pct,
        stop_loss_price=meta.get("stop_price"),
        take_profit_price=meta.get("tp_price"),
        risk=risk,
        vision=extraction,
        validation=validation,
        mcp=mcp,
        mc_paths=int(tactical.paths),
        mc_horizon_days=int(tactical.horizon_days),
        mc_seed=tactical.seed,
        starting_price=tactical.starting_price,
        annual_volatility=tactical.annual_volatility,
        annual_drift=tactical.annual_drift,
        vision_confidence=float(extraction.confidence.overall),
        validation_status=status,
        execution_mode=execution_mode,  # type: ignore[arg-type]
        executable=executable,
        rationale=" ".join(rationale_parts),
        notes=list(meta.get("notes") or []),
    )
