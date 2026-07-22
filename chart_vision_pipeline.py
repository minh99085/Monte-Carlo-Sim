"""
Chart-vision → Monte Carlo decision pipeline (MC-side).

Does **not** call vision APIs or Robinhood MCP. Callers supply a validated
``ChartExtractionResult`` and optional ``MCPMarketSnapshot`` / ``ValidationResult``.

Typical flow (Robinhood plugin orchestrates the outer steps)::

    image → vision extract → MCP validate → map_to_tactical → run 100k MC → decision
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from chart_vision_mapper import (
    build_decision,
    map_to_tactical,
    risk_summary_from_tactical_stats,
    suggest_position_size,
)
from chart_vision_models import (
    ChartExtractionResult,
    ChartTradeDecision,
    MCPMarketSnapshot,
    ValidationResult,
    ValidationStatus,
)
from tactical_simulator import run_tactical_simulation


def run_chart_vision_mc(
    extraction: ChartExtractionResult,
    *,
    mcp: Optional[MCPMarketSnapshot] = None,
    validation: Optional[ValidationResult] = None,
    paths: int = 100_000,
    horizon_days: int = 5,
    seed: Optional[int] = 42,
    use_calibration_drift: bool = True,
    calibration_dir: str = "calibration",
    execution_mode: str = "recommendation_only",
    max_order_notional_usd: float = 100.0,
    max_position_pct: float = 10.0,
    risk_per_trade_pct: float = 0.5,
    min_prob_profit: float = 0.48,
    force_flat_on_reject: bool = True,
) -> ChartTradeDecision:
    """
    Map vision+MCP → tactical config, run Monte Carlo, return decision object.

    Parameters
    ----------
    paths :
        Default **100_000** (production tactical standard).
    execution_mode :
        ``log_only`` | ``recommendation_only`` | ``gated_execution``.
        This module never places orders.
    """
    if validation is not None and validation.status == ValidationStatus.REJECTED:
        # Still allow a flat decision with empty-ish MC if caller wants audit,
        # but force flat mapping.
        force_flat_on_reject = True

    tactical, rule, meta = map_to_tactical(
        extraction,
        mcp=mcp,
        validation=validation,
        paths=paths,
        horizon_days=horizon_days,
        seed=seed,
        use_calibration_drift=use_calibration_drift,
        calibration_dir=calibration_dir,
        force_flat_on_reject=force_flat_on_reject,
    )

    result = run_tactical_simulation(
        tactical,
        rule=rule,
        use_tradingview=False,
        # Ensure explicit market params when provided
        starting_price=tactical.starting_price,
        annual_drift=tactical.annual_drift,
        annual_volatility=tactical.annual_volatility,
        paths=tactical.paths,
        seed=tactical.seed,
        horizon_days=tactical.horizon_days,
    )

    risk = risk_summary_from_tactical_stats(
        result.stats,
        pnl_pct=getattr(result, "pnl_pct", None),
    )

    entry = float(tactical.starting_price or 0.0)
    adj = float(meta.get("adjusted_confidence") or extraction.confidence.overall)
    pos = suggest_position_size(
        entry_price=entry if entry > 0 else 1.0,
        stop_loss_pct=float(rule.stop_loss_pct or 0.02),
        equity=mcp.portfolio_equity if mcp else None,
        buying_power=mcp.buying_power if mcp else None,
        max_order_notional_usd=max_order_notional_usd,
        max_position_pct=max_position_pct,
        risk_per_trade_pct=risk_per_trade_pct,
        adjusted_confidence=adj,
    )

    decision = build_decision(
        extraction=extraction,
        tactical=tactical,
        rule=rule,
        meta=meta,
        risk=risk,
        validation=validation,
        mcp=mcp,
        position_size=pos,
        execution_mode=execution_mode,
        min_prob_profit=min_prob_profit,
    )
    decision.notes = list(decision.notes) + list(result.notes or [])
    return decision


def demo_extraction(
    ticker: str = "AAPL",
    *,
    bias: str = "bullish",
    rsi: float = 58.0,
    price: float = 190.0,
    confidence: float = 0.72,
) -> ChartExtractionResult:
    """Synthetic extraction for offline demos / tests (no vision API)."""
    from chart_vision_models import (
        Bias,
        FieldConfidence,
        IndicatorBundle,
        LevelKind,
        MACDState,
        PriceLevel,
        RSIState,
    )

    return ChartExtractionResult(
        ticker=ticker,
        timeframe="1H",
        indicators=IndicatorBundle(
            rsi=RSIState(value=rsi, zone="neutral", confidence=0.8),
            macd=MACDState(
                macd_line=0.4,
                signal_line=0.2,
                histogram=0.2,
                cross="bullish_cross" if bias == "bullish" else "none",
                confidence=0.7,
            ),
        ),
        levels=[
            PriceLevel(price=price * 0.98, kind=LevelKind.SUPPORT, strength=0.7),
            PriceLevel(price=price * 1.03, kind=LevelKind.RESISTANCE, strength=0.65),
        ],
        bias=Bias(bias) if bias in {b.value for b in Bias} else Bias.UNCLEAR,
        confidence=FieldConfidence(
            ticker=0.9,
            timeframe=0.8,
            indicators=0.75,
            levels=0.6,
            bias=0.7,
            price=0.55,
            overall=confidence,
        ),
        raw_model_description="demo synthetic chart extraction",
        extraction_warnings=["synthetic_demo"],
        image_last_price=price,
        provider="demo",
        model="demo",
    )


def run_demo(
    *,
    paths: int = 5_000,
    seed: int = 42,
    ticker: str = "AAPL",
    price: float = 190.0,
) -> ChartTradeDecision:
    """End-to-end offline demo: synthetic vision → MC → decision (no MCP)."""
    extraction = demo_extraction(ticker=ticker, price=price)
    mcp = MCPMarketSnapshot(
        ticker=ticker,
        last_price=price,
        realized_vol_annual=0.28,
        portfolio_equity=10_000.0,
        buying_power=5_000.0,
    )
    validation = ValidationResult(
        status=ValidationStatus.PASSED,
        overall_confidence=extraction.confidence.overall,
        adjusted_confidence=extraction.confidence.overall,
        ticker_confirmed=True,
        price_rel_error=0.0,
        notes=["demo validation (MCP mock)"],
    )
    return run_chart_vision_mc(
        extraction,
        mcp=mcp,
        validation=validation,
        paths=paths,
        seed=seed,
        execution_mode="recommendation_only",
    )


if __name__ == "__main__":
    import json

    decision = run_demo(paths=2000, seed=42)
    print(decision.summary_text())
    print("\n--- JSON ---")
    print(json.dumps(decision.model_dump(mode="json"), indent=2, default=str)[:4000])
