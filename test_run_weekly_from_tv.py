"""Acceptance tests for the calibrated-drift decision pipeline
(run_weekly_from_tv.py + tv_integration drift wiring).

Includes the critical null test (zero-drift world must produce NO_TRADE and
near-zero calibrated drift) and the power test (a real conditional edge must
be recovered and traded).
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

import outcome_tracker as ot
import run_weekly_from_tv as rwt
import signal_calibration as sc
from tv_integration import (
    load_tradingview_signal,
    signal_to_drift,
    write_demo_signal,
)

# Reduced path counts keep the suite fast; seeds make results deterministic.
# be_paths stays high enough that the breakeven estimate is honest: at 20k
# paths the MC standard error of expectancy (~0.015% weekly) is well below
# the structure's true breakeven (~0.1% weekly), so noise cannot fake a
# near-zero breakeven and let spurious micro-edges through.
FAST_PATHS = 4000
FAST_BE_PATHS = 20_000


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def gen_gbm_prices(n: int, seed: int, sigma_annual: float,
                   mu_annual: float = 0.0, s0: float = 100.0) -> np.ndarray:
    """Plain GBM daily closes with constant log-drift mu_annual/252."""
    rng = np.random.default_rng(seed)
    sd = sigma_annual / math.sqrt(252)
    rets = rng.normal(mu_annual / 252.0, sd, size=n)
    return s0 * np.exp(np.cumsum(np.r_[0.0, rets]))


def gen_regime_prices(n: int, seed: int, sigma_annual: float,
                      mu_bull_annual: float, mu_bear_annual: float = 0.0,
                      s0: float = 100.0) -> np.ndarray:
    """Daily closes whose log-drift is ``mu_bull_annual`` only while
    EMA9 > EMA21 (state computed causally from past closes), else
    ``mu_bear_annual``."""
    rng = np.random.default_rng(seed)
    sd = sigma_annual / math.sqrt(252)
    alpha_f = 2.0 / (9 + 1)
    alpha_s = 2.0 / (21 + 1)
    prices = [s0]
    ema_f = ema_s = None
    fbuf: list = []
    sbuf: list = []
    for _ in range(n):
        p = prices[-1]
        if ema_f is None:
            fbuf.append(p)
            if len(fbuf) == 9:
                ema_f = sum(fbuf) / 9.0
        else:
            ema_f = alpha_f * p + (1 - alpha_f) * ema_f
        if ema_s is None:
            sbuf.append(p)
            if len(sbuf) == 21:
                ema_s = sum(sbuf) / 21.0
        else:
            ema_s = alpha_s * p + (1 - alpha_s) * ema_s
        bull = ema_f is not None and ema_s is not None and ema_f > ema_s
        mu_d = (mu_bull_annual if bull else mu_bear_annual) / 252.0
        prices.append(p * math.exp(rng.normal(mu_d, sd)))
    return np.asarray(prices)


def _fast_settings(tmp_path: Path, calib_dir: Path, *, s0: float,
                   sigma: float, **kw) -> rwt.PipelineSettings:
    defaults = dict(
        data_dir=tmp_path / "tv",
        calibration_dir=calib_dir,
        paths=FAST_PATHS,
        be_paths=FAST_BE_PATHS,
        seed=42,
        s0=s0,
        sigma=sigma,
        trade_log=tmp_path / "trade_log.jsonl",
        verdict_dir=tmp_path / "verdicts",
        write_files=False,
        noise_sweep=False,  # informational only; skip for speed
    )
    defaults.update(kw)
    return rwt.PipelineSettings(**defaults)


# ---------------------------------------------------------------------------
# Fixtures (module-scoped: calibration is reused across tests)
# ---------------------------------------------------------------------------

NULL_SIGMA = 0.15
POWER_SIGMA = 0.10


@pytest.fixture(scope="module")
def null_calib(tmp_path_factory) -> Path:
    """10y of zero-drift GBM, calibrated."""
    prices = gen_gbm_prices(2520, seed=303, sigma_annual=NULL_SIGMA)
    table = sc.calibrate("NULLT", years=10.0, horizon_days=5, prices=prices)
    d = tmp_path_factory.mktemp("null_calib")
    table.save(d)
    return d


@pytest.fixture(scope="module")
def power_calib(tmp_path_factory):
    """10y where drift is +25%/yr only while EMA9 > EMA21."""
    prices = gen_regime_prices(2520, seed=42, sigma_annual=POWER_SIGMA,
                               mu_bull_annual=0.25)
    table = sc.calibrate("PWR", years=10.0, horizon_days=5, prices=prices)
    d = tmp_path_factory.mktemp("power_calib")
    table.save(d)
    return d, table


def _write_signal(data_dir: Path, ticker: str, trend: str, momentum: float,
                  price: float = 100.0, age_hours: float = 0.0) -> Path:
    path = data_dir / "latest_signal.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    received = (datetime.now(timezone.utc) - timedelta(hours=age_hours))
    record = {
        "received_at_utc": received.replace(microsecond=0).isoformat(),
        "source": "test",
        "ticker": ticker,
        "price": price,
        "trend": trend,
        "momentum": momentum,
        "timeframe": "D",
        "strategy": "test",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 2. Null test (critical): zero-drift world must not trade
# ---------------------------------------------------------------------------


class TestNull:
    def test_shrunk_drift_near_zero(self, null_calib: Path):
        table = sc.load_calibration("NULLT", 5, null_calib)
        mus = [b.shrunk_mu_annual for b in table.buckets.values() if b.n >= 2]
        assert mus, "expected populated buckets"
        assert float(np.mean(np.abs(mus))) < 0.02  # < 2%/yr mean |shrunk|

    def test_50_alerts_mostly_no_trade(self, null_calib: Path,
                                       tmp_path: Path):
        rsi_by_tercile = {"low": 30.0, "mid": 50.0, "high": 70.0}
        combos = [(t, r) for t in ("bullish", "bearish")
                  for r in rsi_by_tercile.values()]
        verdicts = []
        for i in range(50):
            trend, rsi = combos[i % len(combos)]
            _write_signal(tmp_path / "tv", "NULLT", trend, rsi + (i % 3))
            settings = _fast_settings(tmp_path, null_calib,
                                      s0=100.0, sigma=NULL_SIGMA, seed=1000 + i)
            v = rwt.run_pipeline(settings)
            verdicts.append(v)
        n_no_trade = sum(1 for v in verdicts if v["verdict"] == "NO_TRADE")
        assert n_no_trade >= 48, (  # >= 95% of 50 (allow noise)
            f"only {n_no_trade}/50 NO_TRADE in a zero-drift world; "
            "the pipeline is fabricating edge"
        )
        # Every verdict must label its drift source — no naked probabilities.
        for v in verdicts:
            assert v["drift_label"]
            assert v["drift_estimate"]["source"] in ("calibration", "none")


# ---------------------------------------------------------------------------
# 3. Power test: a real conditional edge must be recovered and traded
# ---------------------------------------------------------------------------


class TestPower:
    def test_calibration_recovers_bullish_edge(self, power_calib):
        _, table = power_calib
        populated = [b for name, b in table.buckets.items()
                     if name.startswith("bullish") and b.n >= 30]
        assert populated, "expected populated bullish buckets"
        for b in populated:
            assert b.shrunk_mu_weekly > 0, b.bucket
            assert b.t_stat > 2.0, (b.bucket, b.t_stat)

    def test_bullish_alert_trades_with_sizing(self, power_calib,
                                              tmp_path: Path):
        calib_dir, _ = power_calib
        _write_signal(tmp_path / "tv", "PWR", "bullish", 65.0)
        settings = _fast_settings(tmp_path, calib_dir,
                                  s0=100.0, sigma=POWER_SIGMA,
                                  paths=8000, be_paths=6000)
        v = rwt.run_pipeline(settings)
        assert v["verdict"] == "TRADE", v["reason"]
        assert v["sizing"]["shares"] > 0
        assert v["sizing"]["notional"] > 0
        assert v["edge_weekly"] > v["breakeven_mu_weekly"]
        assert v["drift_estimate"]["source"] == "calibration"
        assert v["drift_estimate"]["t_stat"] > 2.0

    def test_bearish_alert_gets_no_calibrated_edge(self, power_calib,
                                                   tmp_path: Path):
        calib_dir, _ = power_calib
        _write_signal(tmp_path / "tv", "PWR", "bearish", 50.0)
        settings = _fast_settings(tmp_path, calib_dir,
                                  s0=100.0, sigma=POWER_SIGMA)
        v = rwt.run_pipeline(settings)
        # bearish buckets carry ~zero drift -> short side has no edge
        assert v["verdict"] == "NO_TRADE"


# ---------------------------------------------------------------------------
# 4. Freshness
# ---------------------------------------------------------------------------


class TestFreshness:
    def test_31h_old_signal_is_ignored(self, power_calib, tmp_path: Path):
        calib_dir, _ = power_calib
        _write_signal(tmp_path / "tv", "PWR", "bullish", 65.0, age_hours=31.0)
        settings = _fast_settings(tmp_path, calib_dir,
                                  s0=100.0, sigma=POWER_SIGMA)
        with pytest.raises(rwt.NoSignalError, match="stale"):
            rwt.run_pipeline(settings)

    def test_1h_old_signal_is_used(self, power_calib, tmp_path: Path):
        calib_dir, _ = power_calib
        _write_signal(tmp_path / "tv", "PWR", "bullish", 65.0, age_hours=1.0)
        settings = _fast_settings(tmp_path, calib_dir,
                                  s0=100.0, sigma=POWER_SIGMA)
        v = rwt.run_pipeline(settings)
        assert v["ticker"] == "PWR"
        assert 0.5 < v["signal_age_hours"] < 2.0

    def test_load_signal_freshness_helper(self, tmp_path: Path):
        _write_signal(tmp_path, "AAPL", "bullish", 55.0, age_hours=31.0)
        assert load_tradingview_signal(tmp_path, max_age_hours=30) is None
        assert load_tradingview_signal(tmp_path) is not None  # opt-in check
        _write_signal(tmp_path, "AAPL", "bullish", 55.0, age_hours=1.0)
        assert load_tradingview_signal(tmp_path, max_age_hours=30) is not None

    def test_undatable_signal_counts_as_stale(self, tmp_path: Path):
        path = tmp_path / "latest_signal.json"
        path.write_text(json.dumps({"ticker": "AAPL", "trend": "bullish",
                                    "momentum": 55.0}))
        assert load_tradingview_signal(tmp_path, max_age_hours=30) is None


# ---------------------------------------------------------------------------
# 5. Fallback market data must hard-fail
# ---------------------------------------------------------------------------


class TestFallback:
    def test_fallback_source_exits_nonzero_no_verdict(self, power_calib,
                                                      tmp_path: Path,
                                                      monkeypatch, capsys):
        calib_dir, _ = power_calib
        _write_signal(tmp_path / "tv", "PWR", "bullish", 65.0)

        from mc_core import MarketParameters

        def fake_market(ticker, years=3.0, **kw):
            return MarketParameters(
                s0=100.0, mu=0.05, sigma=0.2, source="fallback",
                note="synthetic",
                daily_log_returns=np.random.default_rng(0).normal(
                    0, 0.01, 504),
            )

        monkeypatch.setattr(rwt, "estimate_parameters_from_history",
                            fake_market)
        verdict_dir = tmp_path / "verdicts"
        rc = rwt.main([
            "--data-dir", str(tmp_path / "tv"),
            "--calibration-dir", str(calib_dir),
            "--paths", str(FAST_PATHS),
            "--trade-log", str(tmp_path / "log.jsonl"),
            "--verdict-dir", str(verdict_dir),
        ])
        assert rc == 2
        assert "fallback" in capsys.readouterr().err
        assert not list(verdict_dir.glob("*.json")) if verdict_dir.exists() else True
        assert not (tmp_path / "log.jsonl").exists()


# ---------------------------------------------------------------------------
# 6. Kill-switch
# ---------------------------------------------------------------------------


def _seed_losing_log(log: Path, n: int = 20):
    for _ in range(n):
        v = {
            "ticker": "PWR", "verdict": "TRADE", "side": "long",
            "horizon_days": 5, "settled": True,
            "settlement": {"realized_pnl_pct": -0.01, "hit": False},
        }
        ot.log_verdict(v, log)


class TestKillSwitch:
    def test_20_losers_block_trade_verdicts(self, power_calib,
                                            tmp_path: Path):
        calib_dir, _ = power_calib
        log = tmp_path / "trade_log.jsonl"
        _seed_losing_log(log)
        _write_signal(tmp_path / "tv", "PWR", "bullish", 65.0)
        settings = _fast_settings(tmp_path, calib_dir,
                                  s0=100.0, sigma=POWER_SIGMA,
                                  paths=8000, be_paths=6000,
                                  trade_log=log)
        v = rwt.run_pipeline(settings)
        assert v["kill_switch"]["tripped"] is True
        assert v["verdict"] == "NO_TRADE"
        assert "KILL-SWITCH" in v["reason"]

    def test_override_restores_trade(self, power_calib, tmp_path: Path):
        calib_dir, _ = power_calib
        log = tmp_path / "trade_log.jsonl"
        _seed_losing_log(log)
        _write_signal(tmp_path / "tv", "PWR", "bullish", 65.0)
        settings = _fast_settings(tmp_path, calib_dir,
                                  s0=100.0, sigma=POWER_SIGMA,
                                  paths=8000, be_paths=6000,
                                  trade_log=log, override_killswitch=True)
        v = rwt.run_pipeline(settings)
        assert v["kill_switch"]["tripped"] is True
        assert v["verdict"] == "TRADE"


# ---------------------------------------------------------------------------
# 7. End-to-end demo run through the CLI
# ---------------------------------------------------------------------------


REQUIRED_VERDICT_KEYS = {
    "timestamp_utc", "ticker", "verdict", "reason", "side", "horizon_days",
    "signal", "drift_estimate", "drift_label", "s0", "annual_sigma",
    "sigma_method", "market_source", "cost_per_side", "structure",
    "noise_stop_rate", "breakeven_mu_annual", "breakeven_mu_weekly",
    "expectancy_pct", "p_win", "sizing", "kill_switch",
}


class TestEndToEnd:
    def test_demo_cli_produces_wellformed_verdict(self, power_calib,
                                                  tmp_path: Path, capsys):
        calib_dir, _ = power_calib
        verdict_dir = tmp_path / "verdicts"
        rc = rwt.main([
            "--demo", "--ticker", "PWR", "--price", "100",
            "--trend", "bullish", "--momentum", "65",
            "--data-dir", str(tmp_path / "tv"),
            "--calibration-dir", str(calib_dir),
            "--s0", "100", "--sigma", str(POWER_SIGMA),
            "--paths", "8000",
            "--trade-log", str(tmp_path / "log.jsonl"),
            "--verdict-dir", str(verdict_dir),
        ])
        assert rc == 0
        files = sorted(verdict_dir.glob("*_PWR.json"))
        assert len(files) == 1
        verdict = json.loads(files[0].read_text())
        assert REQUIRED_VERDICT_KEYS <= set(verdict)
        assert verdict["verdict"] in ("TRADE", "NO_TRADE")
        assert verdict["drift_estimate"]["bucket"] == "bullish_high"
        # zero-drift noise-stop-out sweep across stop multiples
        sweep = verdict["noise_stop_sweep"]
        assert [r["stop_mult_weekly_sigma"] for r in sweep] == [0.5, 1.0, 1.5, 2.0]
        assert all(0.0 <= r["noise_stop_rate"] <= 1.0 for r in sweep)
        # tighter stops get hit by noise more often
        assert sweep[0]["noise_stop_rate"] > sweep[-1]["noise_stop_rate"]
        # verdict also appended to the trade log
        entries = ot.read_log(tmp_path / "log.jsonl")
        assert len(entries) == 1
        # printed block labels the drift source (no naked probabilities)
        out = capsys.readouterr().out
        assert "bucket=bullish_high" in out
        assert "t=" in out and "n_eff=" in out

    def test_cost_parameter_is_threaded(self, power_calib, tmp_path: Path):
        calib_dir, _ = power_calib
        _write_signal(tmp_path / "tv", "PWR", "bullish", 65.0)
        settings = _fast_settings(tmp_path, calib_dir, s0=100.0,
                                  sigma=POWER_SIGMA, cost=0.0012)
        v = rwt.run_pipeline(settings)
        assert v["cost_per_side"] == pytest.approx(0.0012)
        # default is 5 bps/side, not the old hardcoded 10
        assert rwt.DEFAULT_COST == pytest.approx(0.0005)


# ---------------------------------------------------------------------------
# Drift wiring details
# ---------------------------------------------------------------------------


class TestSignalToDrift:
    def test_missing_calibration_gives_zero_with_reason(self, tmp_path: Path):
        sig = {"trend": "bullish", "momentum": 65.0}
        de = signal_to_drift(sig, "NOCAL", calibration_dir=tmp_path)
        assert de.mu_annual == 0.0
        assert de.source == "none"
        assert "no calibration" in de.reason

    def test_undefined_bucket_gives_zero(self, power_calib):
        calib_dir, _ = power_calib
        de = signal_to_drift({"trend": "sideways", "momentum": 50.0}, "PWR",
                             calibration_dir=calib_dir)
        assert de.mu_annual == 0.0
        assert "undefined bucket" in de.reason

    def test_populated_bucket_returns_shrunk_value(self, power_calib):
        calib_dir, table = power_calib
        de = signal_to_drift({"trend": "bullish", "momentum": 65.0}, "PWR",
                             calibration_dir=calib_dir)
        assert de.source == "calibration"
        assert de.bucket == "bullish_high"
        assert de.mu_annual == pytest.approx(
            table.buckets["bullish_high"].shrunk_mu_annual)

    def test_apply_tv_sets_annual_drift(self, power_calib, tmp_path: Path):
        from tactical_config import TradingRule, preset_5_day
        from tv_integration import apply_tradingview_to_tactical

        calib_dir, table = power_calib
        write_demo_signal(tmp_path / "latest_signal.json", ticker="PWR",
                          price=100.0, trend="bullish", momentum=65.0)
        signal = load_tradingview_signal(tmp_path)
        cfg = preset_5_day("PWR", annual_volatility=0.2, starting_price=100.0)
        rule = TradingRule(name="r", entry_condition="enter long",
                           exit_condition="hold", stop_loss_pct=0.02,
                           max_holding_days=5, side="long")
        cfg2, rule2, ctx, _ = apply_tradingview_to_tactical(
            cfg, rule, signal, calibration_dir=calib_dir)
        assert ctx.drift_estimate is not None
        assert cfg2.annual_drift == pytest.approx(
            table.buckets["bullish_high"].shrunk_mu_annual)
        assert cfg2.annual_drift != 0.0
        assert any("annual_drift set from signal" in n for n in ctx.notes)
        # context serialization carries the estimate
        assert ctx.as_dict()["drift_estimate"]["bucket"] == "bullish_high"
