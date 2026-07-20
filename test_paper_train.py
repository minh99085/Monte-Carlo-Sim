"""Acceptance tests for paper_train.py — the paper-trading training driver.

No network: history is injected and settlement uses a synthetic price
fetcher. Verifies the driver decides across a watchlist, logs to a
dedicated paper log, settles matured trades via outcome_tracker, and
surfaces the training stats + kill-switch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import outcome_tracker as ot
import paper_train as pt
import run_weekly_from_tv as rwt
import signal_calibration as sc
from mc_core import MarketParameters
from test_run_weekly_from_tv import gen_gbm_prices, gen_regime_prices

POWER_SIGMA = 0.10


def _patch_market(monkeypatch, prices_by_ticker):
    """Serve market data from injected prices (no network), keeping the
    momentum lens active via real daily_log_returns."""
    def _fake(ticker, years=3.0, **kw):
        p = np.asarray(prices_by_ticker[ticker.upper()], dtype=float).ravel()
        rets = np.diff(np.log(p))
        return MarketParameters(
            s0=float(p[-1]), mu=0.0,
            sigma=float(np.std(rets) * np.sqrt(252)), source="history",
            note="test", daily_log_returns=rets,
        )
    monkeypatch.setattr(rwt, "estimate_parameters_from_history", _fake)


@pytest.fixture(scope="module")
def power_calib(tmp_path_factory):
    """A world with a real bullish conditional edge, both lenses calibrated."""
    prices = gen_regime_prices(2520, seed=42, sigma_annual=POWER_SIGMA,
                               mu_bull_annual=0.25)
    d = tmp_path_factory.mktemp("pt_calib")
    for fs in sc.FEATURE_SETS:
        sc.calibrate("PWR", years=10.0, horizon_days=5, prices=prices,
                     feature_set=fs).save(d)
    return d, prices


def _bullish_tail_prices(seed: int) -> np.ndarray:
    """Regime prices that end in a bullish (EMA9>EMA21) state."""
    return gen_regime_prices(2520, seed=seed, sigma_annual=POWER_SIGMA,
                             mu_bull_annual=0.25)


def test_synthesize_signal_from_history(power_calib):
    _, prices = power_calib
    sig = pt.synthesize_signal("PWR", prices)
    assert sig is not None
    assert sig["ticker"] == "PWR"
    assert sig["trend"] in ("bullish", "bearish")
    assert 0.0 <= sig["momentum"] <= 100.0
    assert sig["price"] == pytest.approx(float(prices[-1]))


def test_synthesize_signal_warmup_returns_none():
    assert pt.synthesize_signal("X", np.array([100.0, 101.0, 102.0])) is None


def test_decide_ticker_logs_to_paper_log(power_calib, tmp_path: Path,
                                         monkeypatch):
    calib_dir, prices = power_calib
    _patch_market(monkeypatch, {"PWR": prices})
    paper_log = tmp_path / "paper.jsonl"
    v = pt.decide_ticker(
        "PWR", calibration_dir=calib_dir, paper_log=paper_log,
        verdict_dir=tmp_path / "v", signals_dir=tmp_path / "sig",
        paths=8000, prices=prices)
    assert v is not None
    assert v["ticker"] == "PWR"
    assert v["verdict"] in ("TRADE", "NO_TRADE")
    # momentum lens ran (real returns injected)
    assert v["momentum_lens"]["active"] is True
    # verdict was appended to the dedicated paper log
    entries = ot.read_log(paper_log)
    assert len(entries) == 1
    assert entries[0]["ticker"] == "PWR"


def test_paper_train_pass_decides_and_reports(power_calib, tmp_path: Path,
                                              monkeypatch):
    calib_dir, prices = power_calib
    # three "tickers" all mapped to the same edge-bearing world
    prices_by = {"PWR": prices, "PWR2": _bullish_tail_prices(7),
                 "PWR3": _bullish_tail_prices(11)}
    # calibrate the aliases too
    for alias in ("PWR2", "PWR3"):
        for fs in sc.FEATURE_SETS:
            t = sc.calibrate(alias, years=10.0, horizon_days=5,
                             prices=prices_by[alias], feature_set=fs)
            t.save(calib_dir)
    _patch_market(monkeypatch, prices_by)

    paper_log = tmp_path / "paper.jsonl"
    result = pt.paper_train(
        ["PWR", "PWR2", "PWR3"], calibration_dir=calib_dir,
        paper_log=paper_log, verdict_dir=tmp_path / "v",
        signals_dir=tmp_path / "sig", paths=8000,
        do_settle=False, prices_by_ticker=prices_by)
    assert len(result["decided"]) == 3
    assert result["n_trade"] + result["n_no_trade"] == 3
    txt = pt.report_text(result)
    assert "PAPER TRAINING PASS" in txt
    assert "Kill-switch" in txt


def test_settlement_builds_track_record(power_calib, tmp_path: Path):
    """Log several TRADE verdicts, then settle them with a synthetic
    forward-price fetcher and confirm the stats populate. The verdicts are
    built directly (settlement only reads side/structure/cost), so the test
    is independent of whether today's decision happens to be a TRADE."""
    calib_dir, _ = power_calib
    paper_log = tmp_path / "paper.jsonl"

    base = {
        "ticker": "PWR", "verdict": "TRADE", "side": "long",
        "horizon_days": 5, "cost_per_side": 0.0005,
        "structure": {"stop_pct": 0.05, "tp_pct": None},
        "p_win": 0.55, "expectancy_pct": 0.004,
    }
    for i in range(20):
        e = dict(base)
        e["timestamp_utc"] = f"2020-01-{i + 1:02d}T00:00:00+00:00"
        e["signal_received_at_utc"] = e["timestamp_utc"]
        e["settled"] = False
        ot.log_verdict(e, paper_log)

    # Synthetic fetcher: a gently rising series so longs realize a small gain.
    def fetcher(ticker, start_iso):
        dates = [f"2020-01-{d:02d}" for d in range(1, 28)]
        closes = 100.0 * np.exp(np.linspace(0, 0.05, len(dates)))
        return dates, closes

    result = pt.paper_train(
        [], calibration_dir=calib_dir, paper_log=paper_log,
        verdict_dir=tmp_path / "v", signals_dir=tmp_path / "sig",
        do_settle=True, price_fetcher=fetcher)
    s = result["stats"]
    assert s["n_settled_trades"] >= 20
    assert s["hit_rate"] is not None
    assert result["settle_counts"]["settled"] >= 20
    # kill-switch is now active (>=20 settled) and, with positive drift, clear
    assert "kill-switch" in result["kill_switch"]["reason"].lower()


def test_bad_ticker_does_not_abort_pass(power_calib, tmp_path: Path,
                                        monkeypatch):
    calib_dir, prices = power_calib
    prices_by = {"PWR": prices}  # NOCAL_TICKER has no injected prices
    _patch_market(monkeypatch, prices_by)
    result = pt.paper_train(
        ["PWR", "NOCAL_TICKER"], calibration_dir=calib_dir,
        paper_log=tmp_path / "p.jsonl", verdict_dir=tmp_path / "v",
        signals_dir=tmp_path / "sig", paths=8000, do_settle=False,
        prices_by_ticker=prices_by)
    # PWR decided; the other is skipped (no injected prices → download fails
    # offline) but the pass does not crash
    assert any(v["ticker"] == "PWR" for v in result["decided"])
    assert any(sk["ticker"] == "NOCAL_TICKER" for sk in result["skipped"])


def test_report_only_mode(power_calib, tmp_path: Path):
    calib_dir, _ = power_calib
    paper_log = tmp_path / "empty.jsonl"
    rc = pt.main(["--report-only", "--paper-log", str(paper_log)])
    assert rc == 0
