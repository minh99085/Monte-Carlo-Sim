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


def test_mc_core_defines_all_path_constants():
    """Regression guard: every path-mode constant app.py needs must exist."""
    assert mc_core.PREVIEW_PATHS == 10_000
    assert mc_core.STANDARD_PATHS == 100_000
    assert mc_core.SERIOUS_PATHS == 1_000_000
    assert mc_core.CUSTOM_MIN_PATHS == 1_000
    assert mc_core.CUSTOM_MAX_PATHS == 1_000_000
    assert mc_core.TAIL_RISK_MIN_PATHS == 2_000_000
    assert mc_core.TAIL_RISK_MAX_PATHS == 5_000_000
    assert mc_core.DEFAULT_SERIOUS_CHUNK_SIZE in (25_000, 50_000)
    # Backwards-compatible alias still resolves.
    assert mc_core.DEFAULT_SERIOUS_CHUNK == mc_core.DEFAULT_SERIOUS_CHUNK_SIZE
    # Presets are built from the constants, so they can never drift.
    assert mc_core.PATH_MODES["Preview"] == mc_core.PREVIEW_PATHS
    assert mc_core.PATH_MODES["Standard"] == mc_core.STANDARD_PATHS
    assert mc_core.PATH_MODES["Serious"] == mc_core.SERIOUS_PATHS


def test_app_path_mode_settings_all_modes_no_attribute_error():
    """app.path_mode_settings must work for every mode (the crash repro)."""
    import app

    expected_defaults = {
        "Preview": 10_000,
        "Standard": 100_000,
        "Serious": 1_000_000,
        "Custom": 250_000,
        "Tail-risk (advanced)": 2_000_000,
    }
    for mode, default in expected_defaults.items():
        lo, hi, got_default, step = app.path_mode_settings(mode)
        assert got_default == default
        assert lo <= got_default <= hi
        assert step > 0


def test_app_resolve_path_count_defaults():
    """The GUI helper resolves preset modes to their default path counts."""
    import app

    assert app.resolve_path_count("Preview") == 10_000
    assert app.resolve_path_count("Standard") == 100_000
    assert app.resolve_path_count("Serious") == 1_000_000


def test_app_resolve_path_count_accepts_explicit_paths():
    """The GUI helper accepts an explicitly edited path count."""
    import app

    # Preset modes remain editable across the safe range.
    assert app.resolve_path_count("Preview", 12_345) == 12_345
    assert app.resolve_path_count("Standard", 500_000) == 500_000
    # Custom mode can configure 250,000 paths.
    assert app.resolve_path_count("Custom", 250_000) == 250_000
    # Tail-risk advanced accepts 2,000,000-5,000,000.
    assert app.resolve_path_count("Tail-risk (advanced)", 2_000_000) == 2_000_000


def test_app_resolve_path_count_enforces_bounds():
    import app

    with pytest.raises(ValueError):
        app.resolve_path_count("Custom", 500)            # below custom min
    with pytest.raises(ValueError):
        app.resolve_path_count("Custom", 2_000_000)      # above custom max
    with pytest.raises(ValueError):
        app.resolve_path_count("Tail-risk (advanced)", 1_000_000)  # below tail min
    with pytest.raises(ValueError):
        app.resolve_path_count("Tail-risk (advanced)", 6_000_000)  # above tail max


def test_app_path_mode_settings():
    import app

    for mode in ("Preview", "Standard", "Custom"):
        lo, hi, default, step = app.path_mode_settings(mode)
        assert lo == mc_core.CUSTOM_MIN_PATHS
        assert hi == mc_core.CUSTOM_MAX_PATHS
    lo, hi, default, step = app.path_mode_settings("Tail-risk (advanced)")
    assert (lo, hi) == (mc_core.TAIL_RISK_MIN_PATHS, mc_core.TAIL_RISK_MAX_PATHS)
    assert app.path_mode_settings("Preview")[2] == 10_000
    assert app.path_mode_settings("Serious")[2] == 1_000_000


def test_custom_mode_bounds():
    # Custom mode lets the user type any safe count from 1,000 to 1,000,000.
    assert mc_core.resolve_path_mode("Custom", 250_000) == 250_000
    assert mc_core.resolve_path_mode("Custom", mc_core.CUSTOM_MIN_PATHS) == 1_000
    assert mc_core.resolve_path_mode("Custom", mc_core.CUSTOM_MAX_PATHS) == 1_000_000
    with pytest.raises(ValueError):
        mc_core.resolve_path_mode("Custom", 500)
    with pytest.raises(ValueError):
        mc_core.resolve_path_mode("Custom", 2_000_000)
    with pytest.raises(ValueError):
        mc_core.resolve_path_mode("Custom", None)


def test_custom_mode_250k_simulation_is_chunk_safe():
    paths = mc_core.resolve_path_mode("Custom", 250_000)
    cfg = mc_core.SimulationConfig(
        ticker="CUSTOM", s0=100.0, paths=paths, horizon=10,
        mu=0.05, sigma=0.2, chunk_size=mc_core.DEFAULT_SERIOUS_CHUNK, seed=4,
    )
    result = mc_core.simulate(cfg)
    assert result.final_values.shape == (250_000,)
    assert result.memory.is_chunk_safe
    assert result.memory.peak_vector_elements <= cfg.chunk_size


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


def test_tail_risk_2m_routes_through_chunking_without_running():
    """Tail-risk configures 2,000,000 paths and stays chunk-safe.

    Memory safety is asserted via ``predict_memory`` so the test stays fast and
    never actually simulates 2,000,000 paths.
    """
    import app

    paths = app.resolve_path_count("Tail-risk (advanced)", 2_000_000)
    assert paths == 2_000_000

    cfg = mc_core.SimulationConfig(
        ticker="TAIL", s0=100.0, paths=paths, horizon=252,
        mu=0.05, sigma=0.2, chunk_size=mc_core.DEFAULT_SERIOUS_CHUNK,
        sample_paths=50,
    )
    mem = mc_core.predict_memory(cfg)
    # A full 2,000,000 x (252 + 1) matrix must never be allocated.
    assert mem.full_matrix_elements == 2_000_000 * (252 + 1)
    assert mem.peak_matrix_elements == 50 * (252 + 1)
    assert mem.peak_matrix_elements < mem.full_matrix_elements
    assert mem.peak_vector_elements == cfg.chunk_size
    assert mem.peak_vector_elements < cfg.paths
    assert mem.is_chunk_safe


def test_predict_memory_matches_simulate():
    """predict_memory must agree with the accounting done inside simulate."""
    cfg = mc_core.SimulationConfig(
        s0=100.0, paths=30_000, horizon=12, chunk_size=8_000, seed=1, sample_paths=40
    )
    predicted = mc_core.predict_memory(cfg)
    actual = mc_core.simulate(cfg).memory
    assert predicted.peak_matrix_elements == actual.peak_matrix_elements
    assert predicted.peak_vector_elements == actual.peak_vector_elements
    assert predicted.is_chunk_safe == actual.is_chunk_safe


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


def test_build_config_accepts_explicit_paths():
    import app

    # An explicitly edited "Number of paths" value flows straight through.
    for explicit in (1_000, 250_000, 1_000_000):
        cfg = app.build_config_from_inputs(
            ticker="TST", s0=100.0, paths=explicit, horizon=10, mu=0.05,
            sigma=0.2, chunk_size=50_000, seed=1, cost=0.0,
        )
        assert cfg.paths == explicit


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


# ---------------------------------------------------------------------------
# Parameter estimation (annualized mu / sigma)
# ---------------------------------------------------------------------------


def test_annualized_parameters_constant_growth():
    # A series with constant daily growth g has zero volatility and a drift of
    # ln(1 + g) * 252 (the 0.5*sigma^2 correction vanishes when sigma == 0).
    g = 0.001
    n = 600
    prices = 100.0 * (1.0 + g) ** np.arange(n)
    mu, sigma = mc_core.annualized_parameters(prices)
    assert sigma == pytest.approx(0.0, abs=1e-9)
    assert mu == pytest.approx(np.log(1.0 + g) * 252, rel=1e-6)


def test_annualized_parameters_volatility_ordering():
    rng = np.random.default_rng(0)
    base = rng.normal(0.0, 0.01, size=2_000)
    calm = 100.0 * np.exp(np.cumsum(base))
    wild = 100.0 * np.exp(np.cumsum(base * 3.0))
    _, sigma_calm = mc_core.annualized_parameters(calm)
    _, sigma_wild = mc_core.annualized_parameters(wild)
    assert sigma_wild > sigma_calm > 0


def test_annualized_parameters_rejects_bad_input():
    with pytest.raises(ValueError):
        mc_core.annualized_parameters([100.0])
    with pytest.raises(ValueError):
        mc_core.annualized_parameters([100.0, -5.0, 110.0])


# ---------------------------------------------------------------------------
# Standalone million-path runner (run_gbm_million.py)
# ---------------------------------------------------------------------------


def test_run_gbm_million_imports():
    import run_gbm_million  # noqa: F401

    assert callable(run_gbm_million.run)
    assert callable(run_gbm_million.build_config)
    assert callable(run_gbm_million.summarize)


def test_million_config_is_chunk_safe_without_running():
    """A 1,000,000 x 252 run must never materialize the full matrix."""
    import run_gbm_million as r

    cfg = r.build_config(
        ticker="AAPL", s0=100.0, mu=0.1, sigma=0.25,
        paths=1_000_000, horizon=252, seed=42,
        chunk_size=mc_core.DEFAULT_SERIOUS_CHUNK_SIZE,
    )
    assert cfg.paths == 1_000_000
    assert cfg.sample_paths <= r.MAX_SAMPLE_PATHS_ON_CHART

    mem = mc_core.predict_memory(cfg)
    assert mem.full_matrix_elements == 1_000_000 * (252 + 1)
    assert mem.peak_matrix_elements == cfg.sample_paths * (252 + 1)
    assert mem.peak_matrix_elements < mem.full_matrix_elements
    assert mem.peak_vector_elements == cfg.chunk_size
    assert mem.peak_vector_elements < cfg.paths
    assert mem.is_chunk_safe


def test_probability_buckets():
    import run_gbm_million as r

    s0 = 100.0
    # 10 values: which exceed +20% (>120) and which fall below -10% (<90)?
    fv = np.array([80.0, 85.0, 89.0, 95.0, 100.0, 110.0, 121.0, 130.0, 150.0, 200.0])
    # > 120: 121, 130, 150, 200 -> 4/10
    assert r.probability_gain(fv, s0, 0.20) == pytest.approx(0.4)
    # < 90: 80, 85, 89 -> 3/10
    assert r.probability_drop(fv, s0, 0.10) == pytest.approx(0.3)
    # A gain bucket plus a loss bucket are mutually exclusive and <= 1.
    assert r.probability_gain(fv, s0, 0.20) + r.probability_drop(fv, s0, 0.10) <= 1.0


def test_summarize_has_required_fields():
    import run_gbm_million as r

    cfg = r.build_config(
        ticker="TST", s0=100.0, mu=0.08, sigma=0.2,
        paths=20_000, horizon=21, seed=7, chunk_size=10_000,
    )
    result = mc_core.simulate(cfg)
    summary = r.summarize(result)
    for field in (
        "average_ending_price", "median_ending_price",
        "percentile_1", "percentile_5", "percentile_95", "percentile_99",
        "prob_gain_more_than_20pct", "prob_drop_more_than_10pct",
        "chunk_safe",
    ):
        assert field in summary
    assert summary["chunk_safe"] is True
    assert 0.0 <= summary["prob_gain_more_than_20pct"] <= 1.0
    assert 0.0 <= summary["prob_drop_more_than_10pct"] <= 1.0


def test_write_outputs_creates_csv_json_png(tmp_path):
    import run_gbm_million as r

    cfg = r.build_config(
        ticker="OUT", s0=100.0, mu=0.05, sigma=0.2,
        paths=5_000, horizon=10, seed=3, chunk_size=2_500,
    )
    result = mc_core.simulate(cfg)
    summary = r.summarize(result)
    written = r.write_outputs(result, summary, outdir=str(tmp_path), make_chart=True)

    assert set(written) == {"csv", "json", "png"}
    for path in written.values():
        assert __import__("os").path.exists(path)
        assert __import__("os").path.getsize(path) > 0

    parsed = json.loads(open(written["json"]).read())
    assert parsed["ticker"] == "OUT"
    assert "prob_gain_more_than_20pct" in parsed
