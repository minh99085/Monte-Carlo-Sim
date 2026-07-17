"""Tests for TradingView → tactical integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from tactical_config import TradingRule, preset_5_day
from tactical_simulator import run_tactical_simulation
from tv_integration import (
    apply_tradingview_to_tactical,
    load_tradingview_signal,
    momentum_to_jump_multiplier,
    momentum_to_vol_multiplier,
    trend_to_side,
    write_demo_signal,
)


def test_trend_to_side():
    assert trend_to_side("bullish") == "long"
    assert trend_to_side("BEARISH") == "short"
    assert trend_to_side("sideways") is None


def test_momentum_multipliers():
    assert abs(momentum_to_vol_multiplier(50.0) - 1.0) < 1e-9
    assert momentum_to_vol_multiplier(90.0) > 1.0
    assert momentum_to_vol_multiplier(10.0) < 1.0
    assert momentum_to_jump_multiplier(50.0) == 1.0
    assert momentum_to_jump_multiplier(90.0) > 1.0


def test_write_and_load_demo(tmp_path: Path):
    path = write_demo_signal(
        tmp_path / "latest_signal.json",
        ticker="MSFT",
        price=420.0,
        trend="bearish",
        momentum=35.0,
    )
    data = load_tradingview_signal(tmp_path)
    assert data is not None
    assert data["ticker"] == "MSFT"
    assert data["trend"] == "bearish"
    assert path.is_file()


def test_apply_tv_aligns_side_and_vol(tmp_path: Path):
    write_demo_signal(
        tmp_path / "latest_signal.json",
        ticker="NVDA",
        price=100.0,
        trend="bullish",
        momentum=80.0,
    )
    signal = load_tradingview_signal(tmp_path)
    cfg = preset_5_day("AAPL", annual_volatility=0.20, starting_price=50.0)
    rule = TradingRule(
        name="r",
        entry_condition="Enter long",
        exit_condition="hold",
        stop_loss_pct=0.02,
        max_holding_days=5,
        side="short",
    )
    cfg2, rule2, ctx, jump_m = apply_tradingview_to_tactical(
        cfg, rule, signal, signal_path=str(tmp_path / "latest_signal.json"),
        base_sigma=0.20,
    )
    assert ctx.used is True
    assert cfg2.ticker == "NVDA"
    assert cfg2.starting_price == 100.0
    assert rule2.side == "long"
    assert cfg2.annual_volatility is not None
    assert cfg2.annual_volatility > 0.20  # high RSI → higher vol
    assert jump_m >= 1.0


def test_run_with_tradingview_demo(tmp_path: Path):
    write_demo_signal(
        tmp_path / "latest_signal.json",
        ticker="AAPL",
        price=150.0,
        trend="bearish",
        momentum=30.0,
    )
    cfg = preset_5_day(
        "AAPL",
        paths=800,
        seed=11,
        annual_volatility=0.25,
        annual_drift=0.0,
    )
    rule = TradingRule(
        name="tv",
        entry_condition="Enter long",
        exit_condition="hold",
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        max_holding_days=5,
        side="long",
    )
    res = run_tactical_simulation(
        cfg.with_rule(rule),
        use_tradingview=True,
        tv_data_dir=tmp_path,
        tv_require_signal=True,
    )
    assert res.stats["used_tradingview_data"] is True
    assert res.side == "short"  # aligned to bearish
    assert res.config.starting_price == 150.0
    assert "YES" in res.summary_text()
    assert res.tv_context is not None and res.tv_context.used
    d = res.to_stats_dict()
    assert d["used_tradingview_data"] is True
    assert "tradingview" in d


def test_run_without_tv_file_is_graceful(tmp_path: Path):
    cfg = preset_5_day(
        "AAPL", paths=500, seed=2, starting_price=100.0,
        annual_volatility=0.2, annual_drift=0.0,
    )
    res = run_tactical_simulation(
        cfg,
        use_tradingview=True,
        tv_data_dir=tmp_path,  # empty dir
        tv_require_signal=False,
    )
    assert res.stats["used_tradingview_data"] is False
    assert "NO" in res.summary_text()


def test_require_signal_raises(tmp_path: Path):
    cfg = preset_5_day(
        "AAPL", paths=200, seed=1, starting_price=100.0, annual_volatility=0.2,
    )
    with pytest.raises(FileNotFoundError):
        run_tactical_simulation(
            cfg,
            use_tradingview=True,
            tv_data_dir=tmp_path,
            tv_require_signal=True,
        )


def test_filter_against_trend_blocks_trades(tmp_path: Path):
    write_demo_signal(
        tmp_path / "latest_signal.json",
        ticker="AAPL", price=100.0, trend="bullish", momentum=55.0,
    )
    cfg = preset_5_day(
        "AAPL", paths=300, seed=5, annual_volatility=0.2, annual_drift=0.0,
        starting_price=100.0,
    )
    # Force short while TV is bullish; filter should block
    rule = TradingRule(
        name="fight",
        entry_condition="Enter short",
        exit_condition="hold",
        stop_loss_pct=0.02,
        max_holding_days=5,
        side="short",
    )
    res = run_tactical_simulation(
        cfg.with_rule(rule),
        use_tradingview=True,
        tv_data_dir=tmp_path,
        tv_align_side=False,  # keep short
        tv_filter_against_trend=True,
    )
    # Align is off, filter sees short vs bullish → block
    assert res.stats["avg_trades_per_path"] == 0.0 or res.stats["frac_paths_with_trade"] == 0.0


def test_example_script_imports():
    import run_tactical_with_tv as demo
    assert hasattr(demo, "main")
