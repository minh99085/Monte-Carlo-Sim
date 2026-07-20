"""Acceptance tests for the four decision-quality enhancements:

1. Momentum lens — 12-1 time-series momentum as a second, independent
   signal family with its own calibration tables.
2. Walk-forward validation — calibration must pass an exam on held-out
   years it never trained on, or the bucket is hard-zeroed.
3. 2x cost stress — a TRADE must also beat the breakeven computed at
   doubled trading costs.
4. Agreement filter — TRADE requires the momentum lens to independently
   verify an edge in the same direction (when a momentum table exists).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

import edge_scan as es
import run_weekly_from_tv as rwt
import signal_calibration as sc
from test_run_weekly_from_tv import (
    FAST_BE_PATHS,
    FAST_PATHS,
    _fast_settings,
    _write_signal,
    gen_gbm_prices,
    gen_regime_prices,
)

POWER_SIGMA = 0.10


# ---------------------------------------------------------------------------
# Synthetic worlds
# ---------------------------------------------------------------------------


def gen_momentum_regime_prices(n: int, seed: int, sigma_annual: float,
                               mu_up_annual: float) -> np.ndarray:
    """Daily closes whose log-drift is ``mu_up_annual`` only while the
    trailing (causal) 12-1 momentum is positive."""
    rng = np.random.default_rng(seed)
    sd = sigma_annual / math.sqrt(252)
    rets = [0.0005] * (sc.MOM_LOOKBACK_DAYS + 1)
    for i in range(len(rets), n):
        m = sum(rets[i - (sc.MOM_LOOKBACK_DAYS - 1):i - sc.MOM_SKIP_DAYS])
        mu_d = mu_up_annual / 252.0 if m > 0 else 0.0
        rets.append(rng.normal(mu_d, sd))
    return 100.0 * np.exp(np.cumsum(np.r_[0.0, rets]))


def make_momentum_table(ticker: str, tmpdir: Path, *,
                        shrunk_mu_weekly: float, t_stat: float = 3.0,
                        n: int = 800, n_eff: float = 300.0,
                        horizon_days: int = 5) -> Path:
    """Hand-built momentum calibration table (deterministic fixtures)."""
    ann = 252.0 / horizon_days
    buckets = {}
    for name in sc.MOM_BUCKETS:
        mu = shrunk_mu_weekly if name == sc.MOM_BUCKET_UP else 0.0
        buckets[name] = sc.BucketStats(
            bucket=name, n=n, n_eff=n_eff,
            raw_mu_weekly=mu * 1.5, shrunk_mu_weekly=mu,
            se_weekly=0.003, se_naive_weekly=0.002,
            t_stat=t_stat if mu != 0.0 else 0.2,
            raw_mu_annual=mu * 1.5 * ann, shrunk_mu_annual=mu * ann,
        )
    table = sc.CalibrationTable(
        ticker=ticker.upper(), horizon_days=horizon_days,
        created_at=datetime.now(timezone.utc).isoformat(),
        data_start="2018-01-01", data_end="2026-07-01",
        n_days=2000, years=8.0,
        feature_set=sc.FEATURE_SET_MOMENTUM,
        buckets=buckets,
    )
    return table.save(tmpdir)


def fake_history_market(mu_daily: float = 0.0004, n: int = 600,
                        seed: int = 9, sigma_daily: float = 0.012):
    """A patched estimate_parameters_from_history returning real-looking
    history (positive cumulative return → momentum bucket mom_up)."""
    from mc_core import MarketParameters

    rng = np.random.default_rng(seed)
    rets = rng.normal(mu_daily, sigma_daily, size=n)

    def _fake(ticker, years=3.0, **kw):
        return MarketParameters(
            s0=100.0, mu=0.05, sigma=0.2, source="history",
            note="test", daily_log_returns=rets,
        )

    return _fake, rets


# ---------------------------------------------------------------------------
# 1. Momentum lens
# ---------------------------------------------------------------------------


class TestMomentumLens:
    def test_momentum_12_1_exact_and_warmup(self):
        p = np.exp(np.linspace(0.0, 1.0, 300))
        mom = sc.momentum_12_1(p)
        assert np.all(np.isnan(mom[:sc.MOM_LOOKBACK_DAYS]))
        t = 260
        want = math.log(p[t - sc.MOM_SKIP_DAYS] / p[t - sc.MOM_LOOKBACK_DAYS])
        assert mom[t] == pytest.approx(want, abs=1e-12)

    def test_momentum_bucket_sign(self):
        assert sc.momentum_bucket(0.01) == "mom_up"
        assert sc.momentum_bucket(-0.01) == "mom_down"
        assert sc.momentum_bucket(0.0) == "mom_down"
        with pytest.raises(ValueError):
            sc.momentum_bucket(float("nan"))

    def test_calibrate_recovers_momentum_conditional_edge(self):
        prices = gen_momentum_regime_prices(2520, seed=1,
                                            sigma_annual=POWER_SIGMA,
                                            mu_up_annual=0.30)
        table = sc.calibrate("MOMT", years=10.0, horizon_days=5,
                             prices=prices,
                             feature_set=sc.FEATURE_SET_MOMENTUM)
        up = table.buckets["mom_up"]
        assert up.shrunk_mu_weekly > 0
        assert up.t_stat > 2.0
        assert up.wf_pass is True  # stationary edge survives the exam

    def test_momentum_table_roundtrip_and_filename(self, tmp_path: Path):
        prices = gen_gbm_prices(2520, seed=5, sigma_annual=0.15)
        table = sc.calibrate("RTM", years=10.0, horizon_days=5,
                             prices=prices,
                             feature_set=sc.FEATURE_SET_MOMENTUM)
        path = table.save(tmp_path)
        assert path.name == "RTM_5d_mom.json"
        # legacy ema_rsi filename is untouched by the momentum lens
        assert not (tmp_path / "RTM_5d.json").exists()
        loaded = sc.load_calibration("RTM", 5, tmp_path,
                                     feature_set=sc.FEATURE_SET_MOMENTUM)
        assert loaded.feature_set == sc.FEATURE_SET_MOMENTUM
        assert set(loaded.buckets) == set(sc.MOM_BUCKETS)

    def test_too_short_history_raises(self):
        prices = gen_gbm_prices(120, seed=6, sigma_annual=0.2)
        with pytest.raises(sc.CalibrationDataError):
            sc.calibrate("SHORT", years=0.5, horizon_days=5, prices=prices,
                         feature_set=sc.FEATURE_SET_MOMENTUM)


# ---------------------------------------------------------------------------
# 2. Walk-forward validation
# ---------------------------------------------------------------------------


class TestWalkForward:
    def test_sign_flip_pattern_is_zeroed(self):
        """A pattern that reverses in the held-out era must not survive."""
        rng = np.random.default_rng(2)
        sd = POWER_SIGMA / math.sqrt(252)
        r1 = rng.normal(0.50 / 252, sd, 1890)   # training era: strong bull
        r2 = rng.normal(-0.80 / 252, sd, 630)   # holdout era: strong bear
        prices = 100.0 * np.exp(np.cumsum(np.r_[0.0, r1, r2]))
        t_off = sc.calibrate("FLIP", years=10.0, horizon_days=5,
                             prices=prices, walk_forward=False)
        t_on = sc.calibrate("FLIP", years=10.0, horizon_days=5,
                            prices=prices, walk_forward=True)
        zeroed = [nm for nm in sc.ALL_BUCKETS
                  if t_off.buckets[nm].shrunk_mu_weekly != 0.0
                  and t_on.buckets[nm].shrunk_mu_weekly == 0.0]
        assert zeroed, "walk-forward failed to zero any flipped bucket"
        for nm in zeroed:
            b = t_on.buckets[nm]
            assert b.wf_pass is False
            assert "walk-forward" in b.wf_note

    def test_stationary_edge_survives_unchanged(self):
        prices = gen_regime_prices(2520, seed=42, sigma_annual=POWER_SIGMA,
                                   mu_bull_annual=0.25)
        t_off = sc.calibrate("PWR", years=10.0, horizon_days=5,
                             prices=prices, walk_forward=False)
        t_on = sc.calibrate("PWR", years=10.0, horizon_days=5,
                            prices=prices, walk_forward=True)
        survivors = [nm for nm in sc.ALL_BUCKETS
                     if t_on.buckets[nm].shrunk_mu_weekly != 0.0]
        assert survivors, "expected the stationary edge to survive"
        for nm in survivors:
            assert t_on.buckets[nm].wf_pass is True
            assert (t_on.buckets[nm].shrunk_mu_weekly
                    == pytest.approx(t_off.buckets[nm].shrunk_mu_weekly))

    def test_short_data_skips_walk_forward(self):
        prices = gen_gbm_prices(60, seed=3, sigma_annual=0.2)
        table = sc.calibrate("TINY", years=0.25, horizon_days=5,
                             prices=prices, walk_forward=True)
        assert table.walk_forward is not None
        assert table.walk_forward["applied"] is False
        assert "skipped" in table.walk_forward["reason"]

    def test_metadata_recorded_when_applied(self):
        prices = gen_gbm_prices(2520, seed=8, sigma_annual=0.15)
        table = sc.calibrate("META", years=10.0, horizon_days=5,
                             prices=prices, walk_forward=True)
        wf = table.walk_forward
        assert wf["applied"] is True
        assert wf["train_days"] + wf["test_days"] == len(prices)
        # roundtrips through JSON persistence
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            table.save(d)
            back = sc.load_calibration("META", 5, d)
            assert back.walk_forward["applied"] is True


# ---------------------------------------------------------------------------
# 3. 2x cost stress
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def power_world(tmp_path_factory):
    prices = gen_regime_prices(2520, seed=42, sigma_annual=POWER_SIGMA,
                               mu_bull_annual=0.25)
    table = sc.calibrate("PWR", years=10.0, horizon_days=5, prices=prices)
    d = tmp_path_factory.mktemp("power_enh")
    table.save(d)
    return d


class TestCostStress:
    def test_extreme_stress_blocks_trade(self, power_world, tmp_path: Path):
        _write_signal(tmp_path / "tv", "PWR", "bullish", 65.0)
        base = dict(s0=100.0, sigma=POWER_SIGMA, paths=8000, be_paths=6000)
        # stress disabled -> the strong edge trades
        v_off = rwt.run_pipeline(_fast_settings(
            tmp_path, power_world, cost_stress_mult=1.0, **base))
        assert v_off["verdict"] == "TRADE", v_off["reason"]
        assert v_off["breakeven_mu_weekly_stress"] is None
        # absurd 40x stress -> breakeven at stressed costs exceeds the edge
        v_on = rwt.run_pipeline(_fast_settings(
            tmp_path, power_world, cost_stress_mult=40.0, **base))
        assert v_on["verdict"] == "NO_TRADE"
        assert "cost stress" in v_on["reason"]
        assert (v_on["breakeven_mu_weekly_stress"]
                > v_on["breakeven_mu_weekly"])

    def test_default_2x_stress_keeps_strong_edge(self, power_world,
                                                 tmp_path: Path):
        _write_signal(tmp_path / "tv", "PWR", "bullish", 65.0)
        v = rwt.run_pipeline(_fast_settings(
            tmp_path, power_world, s0=100.0, sigma=POWER_SIGMA,
            paths=8000, be_paths=6000))
        assert v["cost_stress_mult"] == pytest.approx(2.0)
        assert v["verdict"] == "TRADE", v["reason"]
        assert "survives 2.0× cost stress" in v["reason"]
        assert (v["breakeven_mu_weekly_stress"]
                > v["breakeven_mu_weekly"])

    def test_no_stress_simulation_when_no_edge(self, power_world,
                                               tmp_path: Path):
        # bearish signal has no edge -> stress breakeven is never computed
        _write_signal(tmp_path / "tv", "PWR", "bearish", 50.0)
        v = rwt.run_pipeline(_fast_settings(
            tmp_path, power_world, s0=100.0, sigma=POWER_SIGMA))
        assert v["verdict"] == "NO_TRADE"
        assert v["breakeven_mu_weekly_stress"] is None


# ---------------------------------------------------------------------------
# 4. Agreement filter
# ---------------------------------------------------------------------------


class TestAgreementFilter:
    def _run(self, tmp_path, calib_dir, monkeypatch, **kw):
        fake, _ = fake_history_market()
        monkeypatch.setattr(rwt, "estimate_parameters_from_history", fake)
        _write_signal(tmp_path / "tv", "PWR", "bullish", 65.0)
        settings = _fast_settings(tmp_path, calib_dir,
                                  s0=None, sigma=POWER_SIGMA,
                                  paths=8000, be_paths=6000, **kw)
        return rwt.run_pipeline(settings)

    def test_unverified_momentum_blocks_trade(self, power_world, tmp_path,
                                              monkeypatch):
        """Second witness says noise -> no TRADE even with a primary edge."""
        make_momentum_table("PWR", power_world, shrunk_mu_weekly=0.0)
        v = self._run(tmp_path, power_world, monkeypatch)
        assert v["momentum_lens"]["active"] is True
        assert v["momentum_lens"]["bucket"] == "mom_up"
        assert v["momentum_lens"]["agrees"] is False
        assert v["verdict"] == "NO_TRADE"
        assert "agreement filter" in v["reason"]

    def test_agreeing_momentum_allows_trade(self, power_world, tmp_path,
                                            monkeypatch):
        make_momentum_table("PWR", power_world, shrunk_mu_weekly=0.004)
        v = self._run(tmp_path, power_world, monkeypatch)
        assert v["momentum_lens"]["agrees"] is True
        assert v["verdict"] == "TRADE", v["reason"]

    def test_opposite_momentum_blocks_trade(self, power_world, tmp_path,
                                            monkeypatch):
        make_momentum_table("PWR", power_world, shrunk_mu_weekly=-0.004)
        v = self._run(tmp_path, power_world, monkeypatch)
        assert v["momentum_lens"]["direction"] == "short"
        assert v["verdict"] == "NO_TRADE"
        assert "agreement filter" in v["reason"]

    def test_missing_table_is_inactive_not_blocking(self, tmp_path,
                                                    monkeypatch,
                                                    tmp_path_factory):
        # fresh calib dir with only the ema_rsi table -> lens inactive
        prices = gen_regime_prices(2520, seed=42, sigma_annual=POWER_SIGMA,
                                   mu_bull_annual=0.25)
        d = tmp_path_factory.mktemp("no_mom")
        sc.calibrate("PWR", years=10.0, horizon_days=5,
                     prices=prices).save(d)
        v = self._run(tmp_path, d, monkeypatch)
        assert v["momentum_lens"]["active"] is False
        assert "no momentum calibration table" in v["momentum_lens"]["note"]
        assert v["verdict"] == "TRADE", v["reason"]

    def test_manual_market_override_is_inactive(self, power_world, tmp_path):
        make_momentum_table("PWR", power_world, shrunk_mu_weekly=0.0)
        _write_signal(tmp_path / "tv", "PWR", "bullish", 65.0)
        v = rwt.run_pipeline(_fast_settings(
            tmp_path, power_world, s0=100.0, sigma=POWER_SIGMA,
            paths=8000, be_paths=6000))
        assert v["momentum_lens"]["active"] is False
        assert "manual market override" in v["momentum_lens"]["note"]
        assert v["verdict"] == "TRADE", v["reason"]

    def test_filter_can_be_disabled(self, power_world, tmp_path, monkeypatch):
        make_momentum_table("PWR", power_world, shrunk_mu_weekly=0.0)
        v = self._run(tmp_path, power_world, monkeypatch,
                      agreement_filter=False)
        assert v["momentum_lens"]["active"] is False
        assert v["verdict"] == "TRADE", v["reason"]

    def test_short_history_lens_inactive(self):
        lens = rwt.momentum_lens_check(
            "X", np.zeros(100), "long", 5, Path("nonexistent"))
        assert lens["active"] is False
        assert "too short" in lens["note"]


# ---------------------------------------------------------------------------
# Edge scan: dual lens + agreement in the report
# ---------------------------------------------------------------------------


class TestEdgeScanDualLens:
    def test_rows_cover_both_lenses_and_horizons(self):
        prices = gen_gbm_prices(2520, seed=7, sigma_annual=0.20)
        rows = es.scan_ticker("DUAL", prices=prices)
        lenses = {(r.lens, r.horizon_days) for r in rows}
        assert lenses == {
            (sc.FEATURE_SET_EMA_RSI, 5), (sc.FEATURE_SET_EMA_RSI, 21),
            (sc.FEATURE_SET_MOMENTUM, 5), (sc.FEATURE_SET_MOMENTUM, 21),
        }
        text = es.report_text(rows)
        assert "trend+RSI" in text
        assert "momentum" in text

    def test_agreement_detection_and_ranking(self):
        def row(lens, mu, horizon=5, ticker="T"):
            return es.EdgeRow(
                ticker=ticker, price=100.0, trend="bullish", rsi=65.0,
                lens=lens, bucket="b", horizon_days=horizon,
                shrunk_mu_period=mu, shrunk_mu_annual=mu * 252 / horizon,
                raw_mu_period=mu, t_stat=2.5, n_eff=100.0, n=300,
            )

        # T: both lenses agree long (small edge); U: single lens (huge edge)
        rows = [
            row(sc.FEATURE_SET_EMA_RSI, 0.001),
            row(sc.FEATURE_SET_MOMENTUM, 0.002),
            row(sc.FEATURE_SET_EMA_RSI, 0.02, ticker="U"),
            row(sc.FEATURE_SET_MOMENTUM, 0.0, ticker="U"),
        ]
        agree = es.agreement_map(rows)
        assert agree[("T", 5)] is True
        assert agree[("U", 5)] is False
        ranked = es.rank_rows(rows)
        # agreement outranks a bigger single-lens edge
        assert ranked[0].ticker == "T"
        text = es.report_text(rows)
        assert "AGREEMENT" in text
        assert "BOTH LENSES AGREE" in text

    def test_short_history_drops_momentum_lens_only(self):
        prices = gen_gbm_prices(200, seed=9, sigma_annual=0.2)
        rows = es.scan_ticker("SHRT", prices=prices, horizons=(5,))
        assert {r.lens for r in rows} == {sc.FEATURE_SET_EMA_RSI}
