"""Tests for the institutional Investment Report (mc_report)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import mc_core
import mc_report


# ---------------------------------------------------------------------------
# Helpers: synthetic offline market parameters (no network)
# ---------------------------------------------------------------------------


def _market(seed=0, n=800, mu=0.10, sigma=0.25, s0=100.0, source="yfinance"):
    rng = np.random.default_rng(seed)
    daily_sd = sigma / math.sqrt(mc_core.TRADING_DAYS_PER_YEAR)
    daily_mu = mu / mc_core.TRADING_DAYS_PER_YEAR - 0.5 * daily_sd ** 2
    r = rng.normal(daily_mu, daily_sd, n)
    return mc_core.MarketParameters(
        s0=s0, mu=mu, sigma=sigma, source=source, daily_log_returns=r)


def _build(**overrides):
    cfg_kwargs = dict(ticker="AAPL", horizon=63, paths=3_000, chunk_size=1_000,
                      seed=1, risk_tolerance="Moderate")
    cfg_kwargs.update(overrides)
    cfg = mc_report.InvestmentReportConfig(**cfg_kwargs)
    return mc_report.build_investment_report(
        cfg, market=_market(0), benchmark_market=_market(1, sigma=0.18),
        include_fundamentals=False,
    )


# ---------------------------------------------------------------------------
# Imports / config
# ---------------------------------------------------------------------------


def test_mc_report_imports():
    assert callable(mc_report.build_investment_report)
    assert mc_report.INSTITUTIONAL_MODELS == tuple(mc_core.MODELS)


def test_report_config_validation():
    with pytest.raises(ValueError):
        mc_report.InvestmentReportConfig(risk_tolerance="Bogus").validate()
    with pytest.raises(ValueError):
        mc_report.InvestmentReportConfig(max_acceptable_loss_pct=1.5).validate()


# ---------------------------------------------------------------------------
# Report generation (small path counts)
# ---------------------------------------------------------------------------


def test_build_investment_report_small():
    report = _build()
    assert report["schema"] == "investment_report_v1"
    assert report["inputs"]["ticker"] == "AAPL"
    assert report["all_chunk_safe"] is True
    # Full institutional stack ran.
    assert len(report["comparison_rows"]) == len(mc_core.MODELS)
    assert report["failed_models"] == []


def test_report_contains_all_required_sections():
    report = _build()
    for heading in mc_report.REPORT_SECTIONS:
        assert heading in report["plain_english"]
        assert report["plain_english"][heading].strip()
    md = mc_report.render_markdown(report)
    for heading in mc_report.REPORT_SECTIONS:
        assert f"## {heading}" in md


def test_investment_label_is_allowed():
    report = _build()
    assert report["investment_label"] in mc_report.INVESTMENT_LABELS


def test_model_confidence_is_valid():
    report = _build()
    assert report["model_confidence"] in mc_report.CONFIDENCE_LEVELS


def test_no_guaranteed_advice_language():
    md = mc_report.render_markdown(_build()).lower()
    assert "guaranteed buy" not in md
    assert "guaranteed sell" not in md
    assert "guaranteed" not in md


# ---------------------------------------------------------------------------
# Stress tests
# ---------------------------------------------------------------------------


def test_stress_tests_included():
    report = _build()
    stress = report["stress_tests"]
    assert len(stress["rows"]) == len(mc_report.stress_scenarios())
    for row in stress["rows"]:
        for key in ("scenario", "expected_value", "median_value",
                    "prob_loss", "prob_loss_20", "prob_ruin"):
            assert key in row
    assert stress["worst_scenario"] is not None
    names = {r["scenario"] for r in stress["rows"]}
    assert any("Combined" in n for n in names)
    assert "Zero drift" in names


def test_stress_suite_runner_direct():
    base = mc_core.SimulationConfig(s0=100.0, paths=2_000, horizon=20, mu=0.1,
                                    sigma=0.3, chunk_size=1_000, seed=2)
    stress = mc_report.run_stress_suite(base)
    assert len(stress["rows"]) == 7
    # The combined stress should be among the most damaging (lowest expected).
    assert stress["worst_scenario"] is not None


# ---------------------------------------------------------------------------
# Model-risk score / confidence
# ---------------------------------------------------------------------------


def test_model_risk_score_levels():
    rows = [{"prob_loss_20": 0.1}, {"prob_loss_20": 0.12}]
    good = mc_report.model_risk_score(
        hist_length=2000, effective_mu=0.1, sigma=0.2, model_rows=rows,
        median_prob_loss_20=0.1)
    assert good["confidence"] == mc_report.CONFIDENCE_HIGH

    bad = mc_report.model_risk_score(
        hist_length=50, effective_mu=2.0, sigma=0.9,
        model_rows=[{"prob_loss_20": 0.05}, {"prob_loss_20": 0.45}],
        median_prob_loss_20=0.4, any_model_failed=True)
    assert bad["confidence"] == mc_report.CONFIDENCE_LOW
    assert any("Do not rely" in w for w in bad["warnings"])


def test_low_confidence_forces_too_uncertain_label():
    label = mc_report.investment_label(
        prob_profit=0.9, prob_exceed_max_loss=0.01,
        risk_tolerance="Aggressive", confidence=mc_report.CONFIDENCE_LOW)
    assert label == mc_report.LABEL_TOO_UNCERTAIN


def test_investment_label_branches():
    fav = mc_report.investment_label(prob_profit=0.7, prob_exceed_max_loss=0.02,
                                     risk_tolerance="Moderate",
                                     confidence=mc_report.CONFIDENCE_HIGH)
    assert fav == mc_report.LABEL_FAVORABLE
    unf = mc_report.investment_label(prob_profit=0.3, prob_exceed_max_loss=0.5,
                                     risk_tolerance="Conservative",
                                     confidence=mc_report.CONFIDENCE_HIGH)
    assert unf == mc_report.LABEL_UNFAVORABLE


# ---------------------------------------------------------------------------
# Benchmark comparison
# ---------------------------------------------------------------------------


def test_benchmark_comparison_handles_spy():
    report = _build(benchmark="SPY")
    bm = report["benchmark_comparison"]
    assert bm["available"] is True
    assert bm["benchmark"] == "SPY"
    for key in ("beta", "correlation", "excess_return", "downside_vs_benchmark",
                "ticker_max_drawdown", "benchmark_max_drawdown"):
        assert key in bm
        assert np.isfinite(bm[key])


def test_benchmark_comparison_direct():
    rng = np.random.default_rng(0)
    base = rng.normal(0.0003, 0.02, 600)
    a = base + rng.normal(0, 0.005, 600)
    b = base
    bm = mc_report.benchmark_comparison(a, b, ticker="X", benchmark="SPY")
    assert bm["available"]
    assert bm["correlation"] > 0.5  # constructed to be correlated


def test_benchmark_handles_short_history():
    bm = mc_report.benchmark_comparison(np.array([0.01]), np.array([0.01]))
    assert bm["available"] is False


# ---------------------------------------------------------------------------
# Fundamentals (must never crash, even offline / bad ticker)
# ---------------------------------------------------------------------------


def test_fundamentals_never_crash():
    f = mc_report.fetch_fundamentals("DEFINITELYNOTAREALTICKERXYZ123")
    assert isinstance(f, dict)
    assert "available" in f
    if not f["available"]:
        assert "note" in f


def test_report_with_fundamentals_does_not_crash():
    cfg = mc_report.InvestmentReportConfig(ticker="AAPL", horizon=21, paths=2_000,
                                           chunk_size=1_000, seed=1)
    report = mc_report.build_investment_report(
        cfg, market=_market(0), benchmark_market=_market(1),
        include_fundamentals=True)
    assert "fundamentals" in report
    assert isinstance(report["fundamentals"], dict)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def test_report_json_export():
    report = _build()
    data = json.loads(mc_report.report_to_json(report))
    assert data["investment_label"] in mc_report.INVESTMENT_LABELS
    assert data["model_confidence"] in mc_report.CONFIDENCE_LEVELS
    assert "_comparison" not in data            # internal handle excluded
    assert "stress_tests" in data
    assert "plain_english" in data


def test_report_comparison_csv_export():
    report = _build()
    csv_text = mc_report.comparison_csv(report)
    assert "model" in csv_text
    assert mc_core.MODEL_GBM in csv_text
    assert len(csv_text.strip().splitlines()) > len(mc_core.MODELS)


def test_write_investment_report_files(tmp_path):
    report = _build()
    paths = mc_report.write_investment_report(report, outdir=str(tmp_path))
    assert set(paths) == {"markdown", "json", "csv"}
    import os
    for p in paths.values():
        assert os.path.exists(p) and os.path.getsize(p) > 0
    md = open(paths["markdown"]).read()
    assert "# Investment Risk Report" in md
    assert "Bottom Line" in md
    parsed = json.loads(open(paths["json"]).read())
    assert parsed["inputs"]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# Chunk-safety with a large configured path count (no real run)
# ---------------------------------------------------------------------------


def test_report_million_path_config_chunk_safe_without_running():
    for model in mc_report.INSTITUTIONAL_MODELS:
        cfg = mc_core.SimulationConfig(
            s0=100.0, paths=1_000_000, horizon=252,
            chunk_size=mc_core.DEFAULT_SERIOUS_CHUNK_SIZE, model=model,
            historical_returns=_market(0).daily_log_returns,
        )
        mem = mc_core.predict_memory(cfg)
        assert mem.is_chunk_safe


# ---------------------------------------------------------------------------
# GUI wiring guard
# ---------------------------------------------------------------------------


def test_app_exposes_investment_report_tab():
    import inspect
    import app
    assert callable(app._render_investment_report)
    src = inspect.getsource(app.main)
    assert "Investment Report" in src
    assert "_render_investment_report" in src
