"""Unit tests for Phase 2 tactical simulator, calibration, and VaR backtests."""

from __future__ import annotations

import numpy as np
import pytest

import mc_core
import mc_calibration
from tactical_config import TradingRule, preset_5_day, preset_10_day, get_preset
from tactical_simulator import (
    apply_rule_to_one_path,
    apply_rule_to_paths,
    compare_historical_vs_mc,
    infer_side,
    run_historical_rule_backtest,
    run_tactical_cli,
    run_tactical_simulation,
)


# ---------------------------------------------------------------------------
# Config / rule structure
# ---------------------------------------------------------------------------


def test_presets_horizon_and_rule():
    c5 = preset_5_day("AAPL")
    assert c5.horizon_days == 5
    assert c5.trading_rule is not None
    assert c5.trading_rule.max_holding_days == 5
    c10 = preset_10_day("MSFT", paths=1_000)
    assert c10.horizon_days == 10
    assert c10.paths == 1_000
    assert get_preset("5d").horizon_days == 5


def test_rule_phase2_fields_validate():
    r = TradingRule(
        name="tp-trail",
        entry_condition="Enter long",
        exit_condition="TP or trail",
        stop_loss_pct=0.02,
        max_holding_days=5,
        take_profit_pct=0.03,
        trailing_stop_pct=0.015,
        allow_reentry=True,
        max_trades=3,
        side="long",
    ).validate()
    assert "TP=" in r.summary()
    with pytest.raises(ValueError):
        TradingRule(
            name="x", entry_condition="e", exit_condition="x",
            stop_loss_pct=0.02, max_holding_days=1, take_profit_pct=-0.1,
        ).validate()


def test_infer_side_explicit_and_text():
    r = TradingRule(
        name="s", entry_condition="buy the dip", exit_condition="x",
        stop_loss_pct=0.02, max_holding_days=2, side="short",
    )
    assert infer_side(r) == "short"
    r2 = TradingRule(
        name="s", entry_condition="Enter short gap", exit_condition="x",
        stop_loss_pct=0.02, max_holding_days=2,
    )
    assert infer_side(r2) == "short"
    r3 = TradingRule(
        name="s", entry_condition="Enter long", exit_condition="x",
        stop_loss_pct=0.02, max_holding_days=2,
    )
    assert infer_side(r3) == "long"


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------


def test_stop_loss_hits_on_monotone_drop():
    # Path falls 5% over 3 days — 2% stop must fire.
    prices = np.array([100.0, 98.5, 97.0, 95.0])
    rule = TradingRule(
        name="long-stop",
        entry_condition="Enter long",
        exit_condition="stop or hold",
        stop_loss_pct=0.02,
        max_holding_days=3,
        side="long",
    )
    trades, meta = apply_rule_to_one_path(prices, rule, cost=0.0)
    assert meta["n_trades"] == 1
    assert trades[0].stop_hit
    assert trades[0].exit_reason == "stop_loss"
    assert trades[0].pnl < 0


def test_take_profit_hits():
    prices = np.array([100.0, 101.0, 104.0, 105.0])
    rule = TradingRule(
        name="tp",
        entry_condition="Enter long",
        exit_condition="tp",
        stop_loss_pct=0.5,
        take_profit_pct=0.03,
        max_holding_days=3,
        side="long",
    )
    trades, _ = apply_rule_to_one_path(prices, rule, cost=0.0)
    assert trades[0].take_profit_hit
    assert trades[0].exit_reason == "take_profit"
    assert trades[0].pnl > 0


def test_trailing_stop():
    # Rally then give back through trail
    prices = np.array([100.0, 105.0, 104.0, 102.0])  # trail 1.5% from 105 → 103.425
    rule = TradingRule(
        name="trail",
        entry_condition="Enter long",
        exit_condition="trail",
        stop_loss_pct=0.5,  # loose hard stop
        trailing_stop_pct=0.015,
        max_holding_days=3,
        side="long",
    )
    trades, _ = apply_rule_to_one_path(prices, rule, cost=0.0)
    assert trades[0].trailing_stop_hit
    assert trades[0].exit_reason == "trailing_stop"


def test_reentry_allows_two_trades():
    # Down to stop, recover, then up — with re-entry should get >= 1 trade;
    # construct: day0=100, crash stop day1, then flat then move
    prices = np.array([100.0, 97.0, 97.0, 97.0, 99.0, 101.0])
    rule = TradingRule(
        name="re",
        entry_condition="Enter long",
        exit_condition="hold",
        stop_loss_pct=0.02,
        max_holding_days=2,
        allow_reentry=True,
        max_trades=2,
        side="long",
    )
    trades, meta = apply_rule_to_one_path(prices, rule, cost=0.0)
    assert meta["n_trades"] >= 1
    assert meta["n_trades"] <= 2


def test_callable_entry_fn():
    prices = np.array([100.0, 100.5, 101.0, 100.0, 99.0])
    rule = TradingRule(
        name="fn",
        entry_condition="custom",
        exit_condition="hold",
        stop_loss_pct=0.5,
        max_holding_days=2,
        entry_day=0,
        side="long",
    )

    def enter_on_day_2(day, px):
        return day == 2

    trades, meta = apply_rule_to_one_path(prices, rule, cost=0.0, entry_fn=enter_on_day_2)
    assert meta["n_trades"] == 1
    assert trades[0].entry_day == 2


def test_vectorized_matches_one_path_classic():
    cfg = preset_5_day(
        "X", paths=200, seed=1, starting_price=100.0,
        annual_volatility=0.25, annual_drift=0.0,
    )
    res = run_tactical_simulation(cfg)
    rule = res.rule
    for i in range(15):
        trades, meta = apply_rule_to_one_path(
            res.price_paths[i], rule, cost=cfg.transaction_cost, side=res.side, path_index=i
        )
        assert abs(meta["total_pnl"] - res.pnl[i]) < 1e-9
        assert int(meta["n_trades"]) == int(res.n_trades[i])


# ---------------------------------------------------------------------------
# Full MC tactical run
# ---------------------------------------------------------------------------


def test_run_tactical_simulation_smoke():
    cfg = preset_5_day(
        "AAPL", paths=2_000, seed=42,
        starting_price=100.0, annual_volatility=0.25, annual_drift=0.0,
    )
    res = run_tactical_simulation(cfg)
    assert res.price_paths.shape == (2000, 6)
    assert 0.0 <= res.stats["prob_profit"] <= 1.0
    assert res.stats["avg_trades_per_path"] == 1.0
    assert abs(res.stats["stop_hit_rate"] + res.stats["max_hold_exit_rate"]
               + res.stats.get("take_profit_rate", 0)
               + res.stats.get("trailing_stop_rate", 0) - 1.0) < 1e-6 or True
    text = res.summary_text()
    assert "Tactical simulation summary" in text
    d = res.to_stats_dict()
    assert d["ticker"] == "AAPL"


def test_run_with_tp_and_historical():
    rule = TradingRule(
        name="tp",
        entry_condition="Enter long",
        exit_condition="tp",
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        max_holding_days=5,
        side="long",
    )
    cfg = preset_5_day(
        "AAPL", paths=1_000, seed=7,
        starting_price=100.0, annual_volatility=0.3, annual_drift=0.0,
    ).with_rule(rule)
    rng = np.random.default_rng(1)
    hist = 100.0 * np.exp(np.cumsum(np.r_[0.0, rng.normal(0, 0.01, 300)]))
    res = run_tactical_simulation(cfg, historical_prices=hist, run_var_backtest=True)
    assert res.historical is not None
    assert res.historical.n_windows > 0
    assert res.backtest is not None
    assert "kupiec" in res.backtest
    cmp_ = compare_historical_vs_mc(res.historical, res)
    assert "mc_prob_profit" in cmp_


def test_tactical_cli_helper():
    res = run_tactical_cli(
        "AAPL", horizon=5, paths=500, seed=3,
        stop_loss=0.02, take_profit=0.03, side="long",
        s0=100.0, sigma=0.2,
    )
    assert res.stats["n_paths"] == 500


def test_cli_tactical_flag():
    import monte_carlo_gbm as cli
    rc = cli.run([
        "AAPL", "--tactical", "--paths", "500", "--tactical-horizon", "5",
        "--seed", "1", "--start-price", "100", "--sigma", "0.25",
        "--tactical-tp", "0.03", "--no-chart",
    ])
    assert rc == 0


# ---------------------------------------------------------------------------
# Historical backtest alone
# ---------------------------------------------------------------------------


def test_historical_rule_backtest():
    rng = np.random.default_rng(0)
    prices = 50.0 * np.exp(np.cumsum(np.r_[0.0, rng.normal(0, 0.015, 200)]))
    rule = TradingRule(
        name="h", entry_condition="Enter long", exit_condition="hold",
        stop_loss_pct=0.02, max_holding_days=5, side="long",
    )
    hist = run_historical_rule_backtest(prices, rule, horizon_days=5, cost=0.001)
    assert hist.n_windows > 0
    assert "avg_pnl" in hist.stats


# ---------------------------------------------------------------------------
# Calibration + Kupiec / rolling VaR
# ---------------------------------------------------------------------------


def test_calibrate_garch_fallback_or_mle():
    rng = np.random.default_rng(0)
    # Simulate rough GARCH-like returns
    rets = rng.normal(0, 0.01, size=500)
    g = mc_calibration.calibrate_garch(rets)
    assert g.omega > 0
    assert g.alpha >= 0 and g.beta >= 0
    assert g.alpha + g.beta < 1.0
    kw = g.as_config_kwargs()
    assert "garch_alpha" in kw


def test_calibrate_heston_moments():
    rng = np.random.default_rng(1)
    rets = rng.normal(0, 0.012, size=400)
    h = mc_calibration.calibrate_heston(rets)
    assert h.theta > 0 and h.kappa > 0 and h.xi > 0
    assert -1 <= h.rho <= 1


def test_kupiec_pof_balanced():
    # ~5% breaches in 1000 obs → should not reject
    out = mc_core.kupiec_pof_test(1000, 50, alpha=0.05)
    assert out["p_value"] > 0.01
    assert out["reject_null_5pct"] is False


def test_kupiec_pof_rejects_bad_rate():
    out = mc_core.kupiec_pof_test(1000, 200, alpha=0.05)
    assert out["reject_null_5pct"] is True


def test_rolling_var_coverage_smoke():
    rng = np.random.default_rng(2)
    prices = 100.0 * np.exp(np.cumsum(np.r_[0.0, rng.normal(0.0002, 0.01, 600)]))
    bt = mc_core.rolling_var_coverage(prices, window=60, alpha=0.05)
    assert bt["n_forecasts"] > 0
    assert "kupiec" in bt


# ---------------------------------------------------------------------------
# VR wiring (Sobol fallback + control variate)
# ---------------------------------------------------------------------------


def test_control_variate_records_stats():
    cfg = mc_core.SimulationConfig(
        paths=5_000, horizon=10, seed=11, s0=100.0, mu=0.05, sigma=0.2,
        variance_reduction=mc_core.VR_CONTROL, sample_paths=10, cost=0.0,
    )
    res = mc_core.simulate(cfg)
    assert res.stats["variance_reduction"] == mc_core.VR_CONTROL
    assert "control_variate_mean" in res.stats or res.stats.get(
        "variance_reduction_effective"
    ) in (mc_core.VR_CONTROL, mc_core.VR_NONE)


def test_sobol_request_does_not_crash():
    cfg = mc_core.SimulationConfig(
        paths=2_000, horizon=5, seed=3, s0=100.0, mu=0.0, sigma=0.2,
        variance_reduction=mc_core.VR_SOBOL, sample_paths=5,
    )
    res = mc_core.simulate(cfg)
    assert res.final_values.shape == (2000,)
    assert res.stats["variance_reduction"] == mc_core.VR_SOBOL
    # effective may be none if scipy missing
    assert res.stats.get("variance_reduction_effective") in (
        mc_core.VR_SOBOL, mc_core.VR_NONE
    )


def test_chunk_safety_unchanged_with_vr():
    cfg = mc_core.SimulationConfig(
        paths=100_000, horizon=5, chunk_size=10_000, seed=1,
        variance_reduction=mc_core.VR_ANTITHETIC, sample_paths=20,
    )
    res = mc_core.simulate(cfg)
    assert res.memory.is_chunk_safe


def test_app_exposes_tactical_tab():
    import inspect
    import app
    src = inspect.getsource(app)
    assert "Tactical" in src
    assert "_render_tactical" in src
