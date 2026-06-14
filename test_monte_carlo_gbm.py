"""Tests for the Monte Carlo GBM simulator (core, CLI and GUI helpers).

Covered requirements:
    * app.py imports successfully
    * 10,000-path preview simulation works
    * 100,000-path standard simulation works
    * 1,000,000-path configuration uses chunked execution (no full-path matrix)
    * fixed random seed is reproducible
    * VaR and Expected Shortfall calculations are tested
    * CSV / JSON export functions are tested
    * serious mode does not require full-path matrix allocation
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import mc_core


# ---------------------------------------------------------------------------
# Import / smoke tests
# ---------------------------------------------------------------------------


def test_app_imports():
    import app  # noqa: F401

    assert hasattr(app, "main")
    assert callable(app.main)
    assert callable(app.build_config_from_inputs)


def test_cli_module_imports():
    import monte_carlo_gbm  # noqa: F401

    assert hasattr(monte_carlo_gbm, "run")
    assert callable(monte_carlo_gbm.run)


# ---------------------------------------------------------------------------
# Path-mode helpers
# ---------------------------------------------------------------------------


def test_path_mode_presets():
    assert mc_core.resolve_path_mode("Preview") == 10_000
    assert mc_core.resolve_path_mode("Standard") == 100_000
    assert mc_core.resolve_path_mode("Serious") == 1_000_000


def test_tail_risk_mode_bounds():
    assert mc_core.resolve_path_mode("Tail-risk (advanced)", 2_000_000) == 2_000_000
    assert mc_core.resolve_path_mode("Tail-risk (advanced)", 5_000_000) == 5_000_000
    with pytest.raises(ValueError):
        mc_core.resolve_path_mode("Tail-risk (advanced)", 1_000_000)
    with pytest.raises(ValueError):
        mc_core.resolve_path_mode("Tail-risk (advanced)", 6_000_000)
    with pytest.raises(ValueError):
        mc_core.resolve_path_mode("Tail-risk (advanced)", None)


def test_tail_risk_warning():
    assert mc_core.tail_risk_warning(100_000) is None
    assert mc_core.tail_risk_warning(1_000_000) is None
    assert mc_core.tail_risk_warning(2_000_000) is not None


# ---------------------------------------------------------------------------
# Simulations at the required scales
# ---------------------------------------------------------------------------


def test_preview_10k_simulation():
    cfg = mc_core.SimulationConfig(
        ticker="TEST", s0=100.0, paths=10_000, horizon=21,
        mu=0.08, sigma=0.2, chunk_size=5_000, seed=1,
    )
    result = mc_core.simulate(cfg)
    assert result.final_values.shape == (10_000,)
    assert np.all(result.final_values > 0)
    assert result.stats["paths"] == 10_000
    assert result.memory.is_chunk_safe


def test_standard_100k_simulation():
    cfg = mc_core.SimulationConfig(
        ticker="TEST", s0=100.0, paths=100_000, horizon=21,
        mu=0.08, sigma=0.2, chunk_size=25_000, seed=2,
    )
    result = mc_core.simulate(cfg)
    assert result.final_values.shape == (100_000,)
    assert result.stats["paths"] == 100_000
    # Expected value should be near the analytical GBM mean S0 * exp(mu * T).
    t = cfg.horizon * cfg.dt
    analytical_mean = cfg.s0 * np.exp(cfg.mu * t)
    assert result.stats["expected_value"] == pytest.approx(analytical_mean, rel=0.05)
    assert result.memory.is_chunk_safe


def test_serious_1m_is_chunk_safe():
    """A 1,000,000-path run must NOT allocate the full path x step matrix."""
    cfg = mc_core.SimulationConfig(
        ticker="TEST", s0=100.0, paths=1_000_000, horizon=10,
        mu=0.05, sigma=0.2, chunk_size=mc_core.DEFAULT_SERIOUS_CHUNK, seed=3,
        sample_paths=50,
    )
    result = mc_core.simulate(cfg)
    mem = result.memory

    assert result.final_values.shape == (1_000_000,)
    # The full (forbidden) matrix would be paths * (horizon + 1) elements.
    full = 1_000_000 * (10 + 1)
    assert mem.full_matrix_elements == full
    # Largest 2-D array actually allocated is just the sample block.
    assert mem.peak_matrix_elements == 50 * (10 + 1)
    assert mem.peak_matrix_elements < full
    # Working buffers are bounded by the chunk size, not the path count.
    assert mem.peak_vector_elements <= cfg.chunk_size
    assert mem.peak_vector_elements < cfg.paths
    assert mem.is_chunk_safe


def test_serious_chunk_default_in_recommended_range():
    assert 25_000 <= mc_core.DEFAULT_SERIOUS_CHUNK <= 50_000


def test_chunk_size_does_not_change_path_count():
    common = dict(s0=100.0, paths=40_000, horizon=15, mu=0.07, sigma=0.25, seed=9)
    r1 = mc_core.simulate(mc_core.SimulationConfig(chunk_size=10_000, **common))
    r2 = mc_core.simulate(mc_core.SimulationConfig(chunk_size=40_000, **common))
    assert r1.final_values.shape == r2.final_values.shape == (40_000,)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_fixed_seed_is_reproducible():
    cfg = mc_core.SimulationConfig(
        s0=100.0, paths=20_000, horizon=30, mu=0.06, sigma=0.22,
        chunk_size=7_000, seed=12345,
    )
    r1 = mc_core.simulate(cfg)
    r2 = mc_core.simulate(cfg)
    assert np.array_equal(r1.final_values, r2.final_values)
    assert r1.stats["expected_value"] == r2.stats["expected_value"]


def test_different_seed_changes_results():
    base = dict(s0=100.0, paths=20_000, horizon=30, mu=0.06, sigma=0.22, chunk_size=7_000)
    r1 = mc_core.simulate(mc_core.SimulationConfig(seed=1, **base))
    r2 = mc_core.simulate(mc_core.SimulationConfig(seed=2, **base))
    assert not np.array_equal(r1.final_values, r2.final_values)


# ---------------------------------------------------------------------------
# VaR and Expected Shortfall
# ---------------------------------------------------------------------------


def test_var_and_es_basic_properties():
    rng = np.random.default_rng(0)
    pnl = rng.normal(0.0, 10.0, size=200_000)

    var95 = mc_core.value_at_risk(pnl, 95.0)
    var99 = mc_core.value_at_risk(pnl, 99.0)
    var999 = mc_core.value_at_risk(pnl, 99.9)

    # Losses get larger as confidence increases.
    assert var95 < var99 < var999
    # All represent real losses for a zero-mean distribution.
    assert var95 > 0

    es95 = mc_core.expected_shortfall(pnl, 95.0)
    es99 = mc_core.expected_shortfall(pnl, 99.0)
    # Expected Shortfall is never smaller than VaR at the same level.
    assert es95 >= var95
    assert es99 >= var99


def test_var_matches_percentile_definition():
    pnl = np.linspace(-100.0, 100.0, 100_001)
    # VaR_95 == -(5th percentile of P&L).
    expected = -np.percentile(pnl, 5.0)
    assert mc_core.value_at_risk(pnl, 95.0) == pytest.approx(expected)


def test_expected_shortfall_handles_tiny_tail():
    # With very few samples the 99.9% tail can be empty -> fall back to worst loss.
    pnl = np.array([-5.0, -1.0, 0.0, 2.0, 3.0])
    es = mc_core.expected_shortfall(pnl, 99.9)
    assert es == pytest.approx(5.0)


def test_statistics_bundle_structure():
    cfg = mc_core.SimulationConfig(s0=100.0, paths=5_000, horizon=10, seed=7)
    result = mc_core.simulate(cfg)
    s = result.stats
    for key in ("expected_value", "median_value", "prob_profit", "prob_loss",
                "var", "expected_shortfall", "percentiles"):
        assert key in s
    for level_key in ("95", "99", "99.9"):
        assert level_key in s["var"]
        assert level_key in s["expected_shortfall"]
        assert "value" in s["var"][level_key]
        assert "pct" in s["var"][level_key]
    assert s["prob_profit"] + s["prob_loss"] <= 1.0 + 1e-9
    for p in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        assert str(p) in s["percentiles"]


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def test_costs_reduce_ending_value():
    gross = np.array([100.0, 120.0, 80.0])
    net = mc_core.apply_costs(gross, s0=100.0, cost=0.01)
    assert np.all(net < gross)
    # Zero cost is a no-op.
    assert np.array_equal(mc_core.apply_costs(gross, 100.0, 0.0), gross)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def _small_result():
    cfg = mc_core.SimulationConfig(
        ticker="EXP", s0=100.0, paths=5_000, horizon=10, seed=11
    )
    return mc_core.simulate(cfg)


def test_csv_export_string():
    result = _small_result()
    text = mc_core.report_to_csv(result)
    assert "metric,value" in text
    assert "expected_ending_value" in text
    assert "VaR_95_value" in text
    assert "ES_99.9_value" in text
    assert "percentile_50" in text
    # Header + at least the core metric rows.
    assert len(text.strip().splitlines()) > 20


def test_json_export_string():
    result = _small_result()
    text = mc_core.report_to_json(result)
    data = json.loads(text)
    assert data["config"]["ticker"] == "EXP"
    assert "statistics" in data
    assert "memory" in data
    assert data["memory"]["is_chunk_safe"] is True
    assert "var" in data["statistics"]


def test_write_csv_and_json_files(tmp_path):
    result = _small_result()
    csv_path = tmp_path / "summary.csv"
    json_path = tmp_path / "report.json"

    mc_core.write_csv(result, str(csv_path))
    mc_core.write_json(result, str(json_path))

    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert json_path.exists() and json_path.stat().st_size > 0
    parsed = json.loads(json_path.read_text())
    assert parsed["statistics"]["paths"] == 5_000


# ---------------------------------------------------------------------------
# GUI helper
# ---------------------------------------------------------------------------


def test_build_config_from_inputs():
    import app

    cfg = app.build_config_from_inputs(
        ticker="aapl", s0=150.0, paths=10_000, horizon=63, mu=0.1,
        sigma=0.3, chunk_size=5_000, seed=42, cost=0.001,
    )
    assert isinstance(cfg, mc_core.SimulationConfig)
    assert cfg.ticker == "aapl"
    assert cfg.paths == 10_000
    result = mc_core.simulate(cfg)
    assert result.final_values.shape == (10_000,)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs", [
    {"paths": 0},
    {"horizon": 0},
    {"s0": 0.0},
    {"cost": 1.5},
    {"chunk_size": 0},
])
def test_invalid_config_raises(kwargs):
    base = dict(s0=100.0, paths=1_000, horizon=10)
    base.update(kwargs)
    with pytest.raises(ValueError):
        mc_core.SimulationConfig(**base).validate()
