"""Unit tests for chart vision models, mapper, pipeline, and scoring."""

from __future__ import annotations

import json

import pytest

from chart_vision_mapper import (
    bias_and_macd_to_side,
    levels_to_stop_tp,
    map_to_tactical,
    suggest_position_size,
)
from chart_vision_models import (
    Bias,
    ChartExtractionResult,
    FieldConfidence,
    IndicatorBundle,
    LevelKind,
    MACDState,
    MCPMarketSnapshot,
    PriceLevel,
    RSIState,
    ValidationResult,
    ValidationStatus,
)
from chart_vision_pipeline import demo_extraction, run_chart_vision_mc, run_demo
from chart_vision_scoring import score_pair, run_scoring


def _extraction(**kwargs) -> ChartExtractionResult:
    base = dict(
        ticker="AAPL",
        timeframe="1H",
        indicators=IndicatorBundle(
            rsi=RSIState(value=55.0, zone="neutral", confidence=0.8),
            macd=MACDState(
                macd_line=0.2,
                signal_line=0.1,
                histogram=0.1,
                cross="bullish_cross",
                confidence=0.7,
            ),
        ),
        levels=[
            PriceLevel(price=185.0, kind=LevelKind.SUPPORT, strength=0.8),
            PriceLevel(price=195.0, kind=LevelKind.RESISTANCE, strength=0.7),
        ],
        bias=Bias.BULLISH,
        confidence=FieldConfidence(
            ticker=0.9,
            timeframe=0.8,
            indicators=0.75,
            levels=0.7,
            bias=0.8,
            price=0.6,
            overall=0.75,
        ),
        raw_model_description="test",
        image_last_price=190.0,
    )
    base.update(kwargs)
    return ChartExtractionResult(**base)


def test_ticker_normalization():
    e = _extraction(ticker="nasdaq:aapl")
    assert e.ticker == "AAPL"


def test_bias_aliases():
    e = ChartExtractionResult(
        ticker="MSFT",
        timeframe="D",
        bias="buy",
        confidence=FieldConfidence(overall=0.5),
    )
    assert e.bias == Bias.BULLISH


def test_tv_signal_dict_shape():
    e = _extraction()
    d = e.to_tv_signal_dict()
    assert d["ticker"] == "AAPL"
    assert d["trend"] == "bullish"
    assert d["momentum"] == 55.0
    assert d["source"] == "chart_vision"


def test_levels_to_stop_tp_long():
    e = _extraction()
    stop_pct, tp_pct, stop_px, tp_px, notes = levels_to_stop_tp(
        e, 190.0, "long", confidence=0.8
    )
    assert stop_pct is not None and stop_pct > 0
    assert tp_pct is not None and tp_pct > 0
    assert stop_px < 190.0 < tp_px
    assert notes


def test_map_prefers_mcp_price():
    e = _extraction(image_last_price=999.0)
    mcp = MCPMarketSnapshot(ticker="AAPL", last_price=190.5, realized_vol_annual=0.3)
    val = ValidationResult(
        status=ValidationStatus.PASSED,
        overall_confidence=0.75,
        adjusted_confidence=0.75,
        ticker_confirmed=True,
    )
    tactical, rule, meta = map_to_tactical(
        e, mcp=mcp, validation=val, paths=1000, seed=1, use_calibration_drift=False
    )
    assert tactical.starting_price == pytest.approx(190.5)
    assert meta["side"] == "long"
    assert rule.side == "long"


def test_reject_forces_flat():
    e = _extraction()
    val = ValidationResult(
        status=ValidationStatus.REJECTED,
        overall_confidence=0.2,
        adjusted_confidence=0.1,
        ticker_confirmed=False,
    )
    tactical, rule, meta = map_to_tactical(
        e,
        validation=val,
        paths=500,
        seed=1,
        use_calibration_drift=False,
        force_flat_on_reject=True,
    )
    assert meta["side"] is None


def test_position_size_caps():
    pos = suggest_position_size(
        entry_price=100.0,
        stop_loss_pct=0.02,
        equity=10_000.0,
        buying_power=50_000.0,
        max_order_notional_usd=100.0,
        max_position_pct=10.0,
        risk_per_trade_pct=1.0,
        adjusted_confidence=1.0,
    )
    assert pos.notional_usd <= 100.0
    assert "max_order_notional" in pos.capped_by or pos.notional_usd <= 100.0


def test_pipeline_demo_small_paths():
    decision = run_demo(paths=500, seed=7)
    assert decision.ticker == "AAPL"
    assert decision.mc_paths == 500
    assert decision.risk.n_paths == 500
    assert decision.validation_status == ValidationStatus.PASSED
    assert decision.risk.prob_profit is not None
    text = decision.summary_text()
    assert "Chart Vision Trade Decision" in text


def test_scoring_perfect_match(tmp_path):
    e = demo_extraction()
    case = tmp_path / "c1"
    case.mkdir()
    (case / "ground_truth.json").write_text(
        e.model_dump_json(indent=2), encoding="utf-8"
    )
    (case / "prediction.json").write_text(
        e.model_dump_json(indent=2), encoding="utf-8"
    )
    report = run_scoring(eval_dir=tmp_path)
    assert report["aggregate"]["n"] == 1
    assert report["aggregate"]["ticker_accuracy"] == 1.0
    assert report["aggregate"]["bias_accuracy"] == 1.0


def test_score_pair_rsi_error():
    t = demo_extraction(rsi=50.0)
    p = demo_extraction(rsi=55.0)
    s = score_pair(t, p)
    assert s["rsi_abs_error"] == pytest.approx(5.0)
    assert s["ticker_accuracy"] == 1.0


def test_bias_and_macd_side():
    e = _extraction(bias=Bias.BEARISH)
    e.indicators.macd.cross = "bearish_cross"
    assert bias_and_macd_to_side(e) == "short"
