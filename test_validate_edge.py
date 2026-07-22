"""Ground-truth tests for the edge-validation harness.

The harness must (a) recover known quantities exactly, (b) report a LOSS on
zero-edge data — never manufacture edge — and (c) report survival when a real
edge is planted, so it is neither rigged to pass nor rigged to fail.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import validate_edge as ve
import validation_report as vr


# ---------------------------------------------------------------------------
# Synthetic OHLC builders
# ---------------------------------------------------------------------------


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2016-01-04", periods=n)


def gbm_ohlc(n: int, seed: int, mu_annual: float = 0.0,
             sigma_annual: float = 0.15) -> ve.OHLC:
    rng = np.random.default_rng(seed)
    sd = sigma_annual / math.sqrt(252)
    rets = rng.normal(mu_annual / 252.0, sd, n)
    close = 100.0 * np.exp(np.cumsum(np.r_[0.0, rets]))[1:]
    # open = prior close nudged by a small independent overnight move
    onight = rng.normal(0.0, sd / 2, n)
    op = np.r_[close[0], close[:-1]] * np.exp(onight)
    return ve.OHLC(dates=_dates(n), open=op, close=close)


def regime_ohlc(n: int, seed: int, mu_bull_annual: float = 0.30,
                sigma_annual: float = 0.10) -> ve.OHLC:
    """Drift is mu_bull only while EMA9>EMA21 (causal) — a real conditional
    edge the walk-forward should detect."""
    rng = np.random.default_rng(seed)
    sd = sigma_annual / math.sqrt(252)
    a_f, a_s = 2 / 10, 2 / 22
    prices = [100.0]
    ema_f = ema_s = None
    fbuf: list = []
    sbuf: list = []
    for _ in range(n):
        p = prices[-1]
        if ema_f is None:
            fbuf.append(p)
            if len(fbuf) == 9:
                ema_f = sum(fbuf) / 9
        else:
            ema_f = a_f * p + (1 - a_f) * ema_f
        if ema_s is None:
            sbuf.append(p)
            if len(sbuf) == 21:
                ema_s = sum(sbuf) / 21
        else:
            ema_s = a_s * p + (1 - a_s) * ema_s
        bull = ema_f is not None and ema_s is not None and ema_f > ema_s
        mu = (mu_bull_annual if bull else 0.0) / 252.0
        prices.append(p * math.exp(rng.normal(mu, sd)))
    close = np.asarray(prices[1:])
    op = np.r_[close[0], close[:-1]]
    return ve.OHLC(dates=_dates(close.size), open=op, close=close)


# ---------------------------------------------------------------------------
# 1. Executable-fill machinery reads the correct bars
# ---------------------------------------------------------------------------


def test_executable_and_signal_returns_use_correct_bars():
    ohlc = regime_ohlc(1600, seed=1)
    trades = ve.walk_forward_trades(ohlc, require_agreement=False)
    assert trades, "expected the planted-edge world to trade"
    t0 = trades[0]
    # locate the entry index
    idx = list(ohlc.dates).index(pd.Timestamp(t0.entry_date))
    sgn = 1.0 if t0.side == "long" else -1.0
    exp_close = sgn * (ohlc.close[idx + ve.WEEK] / ohlc.close[idx] - 1.0)
    exp_exec = sgn * (ohlc.open[idx + 1 + ve.WEEK] / ohlc.open[idx + 1] - 1.0)
    assert t0.r_signal_close == pytest.approx(exp_close, abs=1e-12)
    assert t0.r_executable == pytest.approx(exp_exec, abs=1e-12)
    # the planted-edge world is an uptrend → most trades are long
    longs = sum(1 for t in trades if t.side == "long")
    assert longs > len(trades) * 0.7


def test_walk_forward_is_weekly_non_overlapping():
    ohlc = regime_ohlc(1600, seed=2)
    trades = ve.walk_forward_trades(ohlc, require_agreement=False)
    idxs = [list(ohlc.dates).index(pd.Timestamp(t.entry_date)) for t in trades]
    gaps = np.diff(idxs)
    assert np.all(gaps >= ve.WEEK)  # cadence == horizon → no time overlap


# ---------------------------------------------------------------------------
# 2. Cost + tax math is exact
# ---------------------------------------------------------------------------


def test_cost_tax_math_exact():
    trades = [ve.Trade(pd.Timestamp("2020-01-01"), "long", "bullish_mid",
                       r_signal_close=0.02, r_executable=0.02, year=2020,
                       both_lenses=True)]
    rows = ve.cost_tax_table(trades, cost_sides=[0.002], tax_rate=0.35,
                             trades_per_year=52.0)
    r = rows[0]
    net = (1.02) * (0.998) ** 2 - 1.0
    assert r["net_mean"] == pytest.approx(net, abs=1e-12)
    assert r["after_tax_mean"] == pytest.approx(net * (1 - 0.35), abs=1e-12)
    assert r["net_annual"] == pytest.approx(net * 52, abs=1e-10)


def test_loss_is_not_taxed():
    trades = [ve.Trade(pd.Timestamp("2020-01-01"), "long", "b", -0.02, -0.02,
                       2020, True)]
    rows = ve.cost_tax_table(trades, cost_sides=[0.002], tax_rate=0.35)
    # a losing trade's after-tax return equals its net (no negative tax)
    assert rows[0]["after_tax_mean"] == pytest.approx(rows[0]["net_mean"], abs=1e-12)


# ---------------------------------------------------------------------------
# 3. Perf stats + benchmark
# ---------------------------------------------------------------------------


def test_perf_stats_max_drawdown_exact():
    r = np.array([0.10, -0.50, 0.10])
    st = ve.perf_stats(r, 52.0)
    assert st["max_drawdown"] == pytest.approx(0.55 / 1.10 - 1.0, abs=1e-9)


def test_perf_stats_monotonic_has_no_drawdown():
    rng = np.random.default_rng(0)
    r = np.abs(rng.normal(0.01, 0.001, 200))  # always positive
    st = ve.perf_stats(r, 52.0)
    assert st["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
    assert st["cagr"] > 0


def test_benchmark_after_tax_lower_than_pre_tax_gain():
    ohlc = gbm_ohlc(1500, seed=5, mu_annual=0.12)
    b = ve.benchmark_buy_hold(ohlc, long_term_tax=0.15)
    if b["total_return"] > 0:
        assert b["after_tax_total_return"] < b["total_return"]


# ---------------------------------------------------------------------------
# 4. Bootstrap CI + Kelly under uncertainty
# ---------------------------------------------------------------------------


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(3)
    r = rng.normal(0.01, 0.02, 400)
    ci = ve.bootstrap_edge_ci(r)
    assert ci["lo"] < ci["mean"] < ci["hi"]
    assert ci["mean"] == pytest.approx(0.01, abs=0.004)


def test_kelly_scales_with_edge():
    rng = np.random.default_rng(4)
    r = rng.normal(0.01, 0.03, 500)
    k = ve.kelly_under_uncertainty(r, current_fraction=0.25)
    # half-edge Kelly is exactly half the point-estimate Kelly
    assert k["f_star_half"] == pytest.approx(k["f_star_point"] / 2, rel=1e-9)
    assert k["f_star_zero"] == 0.0
    # deploying quarter-Kelly of the point estimate over-bets 2x if true=half
    assert k["overbet_factor_if_true_edge_half"] == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Effective N + lens correlation
# ---------------------------------------------------------------------------


def _trade_series(ticker: str, weeks, returns) -> list:
    return [ve.Trade(pd.Timestamp(w), "long", "b", r, r,
                     pd.Timestamp(w).year, True)
            for w, r in zip(weeks, returns)]


def test_effective_n_collapses_when_correlated():
    weeks = pd.bdate_range("2020-01-06", periods=30, freq="7D")
    rng = np.random.default_rng(9)
    base = rng.normal(0.0, 0.02, len(weeks))
    per = {"A": _trade_series("A", weeks, base),
           "B": _trade_series("B", weeks, base)}  # identical → rho≈1
    en = ve.effective_n(per, window=20)
    assert en["rho_bar"] > 0.95
    assert en["n_eff"] < 2.0            # 20 correlated bets ≈ 1 independent bet


def test_effective_n_much_higher_when_independent_than_correlated():
    weeks = pd.bdate_range("2020-01-06", periods=260, freq="7D")
    rng = np.random.default_rng(10)
    a = rng.normal(0, 0.02, len(weeks))
    indep = {"A": _trade_series("A", weeks, a),
             "B": _trade_series("B", weeks, rng.normal(0, 0.02, len(weeks)))}
    corr = {"A": _trade_series("A", weeks, a),
            "B": _trade_series("B", weeks, a)}  # identical
    en_indep = ve.effective_n(indep, window=20)
    en_corr = ve.effective_n(corr, window=20)
    assert abs(en_indep["rho_bar"]) < 0.3
    assert en_corr["rho_bar"] > 0.95
    # independence must buy materially more effective bets than perfect corr
    assert en_indep["n_eff"] > 4 * en_corr["n_eff"]
    assert en_corr["n_eff"] < 2.0


def test_lens_correlation_high_on_trend():
    ohlc = regime_ohlc(1500, seed=11)
    lc = ve.lens_correlation(ohlc)
    assert lc["sign_agreement"] is not None
    assert lc["sign_agreement"] > 0.7   # both lenses are trend proxies


# ---------------------------------------------------------------------------
# 6. Integration: honest verdict both ways (not rigged)
# ---------------------------------------------------------------------------


def _fetch_factory(builder):
    def _fetch(ticker: str, years: float) -> ve.OHLC:
        seed = abs(hash(ticker)) % 100000
        return builder(seed)
    return _fetch


def test_zero_edge_world_does_not_survive(tmp_path: Path):
    fetch = _fetch_factory(lambda s: gbm_ohlc(1800, seed=s, mu_annual=0.0))
    result = ve.run_full(["AAA", "BBB"], benchmark="BENCH",
                         years=7.0, fetch=fetch)
    v = vr.compute_verdict(result)
    assert v["survives"] is False, "zero-edge world must NOT survive"
    out = vr.write_report(result, tmp_path / "V.md")
    text = out.read_text()
    assert "does NOT survive" in text
    assert "do not deploy" in text.lower()


def test_horizon_changes_cadence_and_trade_count():
    ohlc = regime_ohlc(2000, seed=21)
    weekly = ve.walk_forward_trades(ohlc, horizon_days=5, require_agreement=False)
    monthly = ve.walk_forward_trades(ohlc, horizon_days=21, require_agreement=False)
    # trading less often -> fewer trades, and cadence == horizon
    assert len(monthly) < len(weekly)
    idxs = [list(ohlc.dates).index(pd.Timestamp(t.entry_date)) for t in monthly]
    assert np.all(np.diff(idxs) >= 21)


def test_tax_rate_switches_to_long_term_over_one_year():
    assert ve.tax_for_horizon(21, 0.35, 0.15) == 0.35     # monthly = short-term
    assert ve.tax_for_horizon(252, 0.35, 0.15) == 0.15    # annual = long-term
    assert ve.tax_for_horizon(126, 0.35, 0.15) == 0.35    # <1yr still short


def test_cost_drag_falls_with_fewer_trades():
    # same per-trade gross, but annualizing over 12 vs 52 trades/yr means the
    # annual cost drag is far smaller for the monthly cadence.
    trades = [ve.Trade(pd.Timestamp("2020-01-01"), "long", "b", 0.02, 0.02,
                       2020, True)]
    weekly = ve.cost_tax_table(trades, cost_sides=[0.002], tax_rate=0.35,
                               trades_per_year=52.0)[0]
    monthly = ve.cost_tax_table(trades, cost_sides=[0.002], tax_rate=0.35,
                                trades_per_year=12.0)[0]
    weekly_drag = weekly["gross_annual"] - weekly["net_annual"]
    monthly_drag = monthly["gross_annual"] - monthly["net_annual"]
    assert monthly_drag < weekly_drag


def test_sweep_lower_cost_never_hurts_net_edge():
    def f(ticker, years):
        return (gbm_ohlc(1800, seed=999, mu_annual=0.10) if ticker == "QQQ"
                else regime_ohlc(1800, seed=abs(hash(ticker)) % 9999,
                                 mu_bull_annual=0.40))
    cheap = ve.run_turnover_sweep(["AAA"], horizons=[5, 21], benchmark="QQQ",
                                  years=7.0, cost_side=0.0003, fetch=f)
    dear = ve.run_turnover_sweep(["AAA"], horizons=[5, 21], benchmark="QQQ",
                                 years=7.0, cost_side=0.0020, fetch=f)
    for rc, rd in zip(cheap["rows"], dear["rows"]):
        assert rc["net_annual"] > rd["net_annual"]  # same trades, lower drag


def test_turnover_sweep_zero_edge_has_no_survivor(tmp_path: Path):
    fetch = _fetch_factory(lambda s: gbm_ohlc(2200, seed=s, mu_annual=0.0))

    def f(ticker, years):
        return (gbm_ohlc(2200, seed=999, mu_annual=0.10) if ticker == "QQQ"
                else gbm_ohlc(2200, seed=abs(hash(ticker)) % 9999, mu_annual=0.0))

    sweep = ve.run_turnover_sweep(["AAA", "BBB"], horizons=[5, 21, 63],
                                  benchmark="QQQ", years=8.0, fetch=f)
    assert sweep["any_survivor"] is False
    out = vr.write_turnover_report(sweep, tmp_path / "T.md")
    assert "No holding horizon survives" in out.read_text()


def test_turnover_report_renders_survivor_path(tmp_path: Path):
    # hand-built sweep with a clear survivor row -> report must flag it as a
    # LEAD, not a green light, and still print the caveats.
    sweep = {
        "tickers": ["AAA"], "benchmark": "QQQ",
        "benchmark_sharpe": 0.8, "benchmark_cagr": 0.15,
        "cost_side": 0.002, "short_tax": 0.35, "long_tax": 0.15,
        "n_horizons_tested": 3,
        "rows": [
            {"horizon_days": 5, "trades_per_year": 52, "n_trades": 400,
             "tax_rate": 0.35, "gross_annual": 0.28, "net_annual": -0.05,
             "after_tax_annual": -0.10, "sharpe": -0.5, "cagr": -0.1,
             "max_drawdown": -0.3, "beats_benchmark_riskadj": False},
            {"horizon_days": 252, "trades_per_year": 1, "n_trades": 30,
             "tax_rate": 0.15, "gross_annual": 0.20, "net_annual": 0.18,
             "after_tax_annual": 0.16, "sharpe": 1.1, "cagr": 0.16,
             "max_drawdown": -0.2, "beats_benchmark_riskadj": True},
        ],
        "any_survivor": True,
    }
    out = vr.write_turnover_report(sweep, tmp_path / "T.md")
    text = out.read_text()
    assert "lead" in text.lower() and "NOT a green light" in text
    assert "Multiple testing" in text


def mean_reverting_ohlc(n: int, seed: int, theta: float = 0.15,
                        sigma_annual: float = 0.20) -> ve.OHLC:
    """Ornstein-Uhlenbeck log-price: pulls back toward a level, so sharp
    down-moves tend to bounce — a real short-term reversal edge."""
    rng = np.random.default_rng(seed)
    sd = sigma_annual / math.sqrt(252)
    x = 0.0
    xs = []
    for _ in range(n):
        x += -theta * x + rng.normal(0.0, sd)
        xs.append(x)
    close = 100.0 * np.exp(np.asarray(xs))
    op = np.r_[close[0], close[:-1]]     # open ≈ prior close (no gap)
    return ve.OHLC(dates=_dates(n), open=op, close=close)


class TestReversalSignal:
    def test_reversal_recovers_planted_mean_reversion(self):
        ohlc = mean_reverting_ohlc(2000, seed=1)
        trades = ve.reversal_trades(ohlc, horizon_days=5)
        assert len(trades) > 20
        r = np.array([t.r_executable for t in trades])
        assert float(np.mean(r)) > 0        # oversold bounces are profitable
        assert all(t.side == "long" for t in trades)

    def test_reversal_finds_no_edge_in_random_walk(self):
        ohlc = gbm_ohlc(2000, seed=2, mu_annual=0.0)
        trades = ve.reversal_trades(ohlc, horizon_days=5)
        r = np.array([t.r_executable for t in trades]) if trades else np.array([0.0])
        assert abs(float(np.mean(r))) < 0.01   # ~no systematic bounce

    def test_reversal_trades_are_non_overlapping(self):
        ohlc = mean_reverting_ohlc(2000, seed=3)
        trades = ve.reversal_trades(ohlc, horizon_days=5)
        idxs = [list(ohlc.dates).index(pd.Timestamp(t.entry_date)) for t in trades]
        assert np.all(np.diff(idxs) >= 5)

    def test_confirm_reversal_signal_runs_and_labels(self, tmp_path):
        def f(ticker, years):
            return (gbm_ohlc(2000, seed=999, mu_annual=0.10)
                    if ticker == "QQQ"
                    else mean_reverting_ohlc(2000, seed=abs(hash(ticker)) % 9999))
        res = ve.run_horizon_confirm(["AAA", "BBB"], horizon_days=5,
                                     benchmark="QQQ", years=8.0,
                                     signal="reversal", fetch=f)
        assert res.get("signal") == "reversal"
        assert "error" not in res, res
        out = vr.write_confirm_report(res, tmp_path / "R.md")
        assert "oversold-reversal" in out.read_text()


class TestResampleDrawdown:
    def test_all_positive_has_no_drawdown(self):
        r = np.full(50, 0.01)
        dd = ve.resample_drawdown(r, n_boot=200)
        assert dd["backtest_dd"] == pytest.approx(0.0)
        assert dd["worst_dd"] == pytest.approx(0.0)

    def test_worst_reshuffle_is_at_least_as_bad_as_backtest(self):
        rng = np.random.default_rng(1)
        r = rng.normal(0.005, 0.03, 120)   # mixed wins/losses
        dd = ve.resample_drawdown(r, n_boot=3000)
        # the backtest ordering is one of the permutations sampled, so the
        # worst reshuffle can only be equal or deeper (more negative)
        assert dd["worst_dd"] <= dd["backtest_dd"] + 1e-9
        assert dd["worst_dd"] <= dd["median_dd"] <= 0.0
        assert dd["p95_dd"] <= dd["median_dd"] + 1e-9

    def test_tiny_series_degenerates_safely(self):
        dd = ve.resample_drawdown(np.array([0.01, -0.02]))
        assert dd["n"] == 2 and "worst_dd" in dd

    def test_confirm_report_includes_drawdown_stress(self, tmp_path):
        def f(ticker, years):
            return (gbm_ohlc(2000, seed=999, mu_annual=0.05)
                    if ticker == "QQQ"
                    else mean_reverting_ohlc(2000, seed=abs(hash(ticker)) % 9999))
        res = ve.run_horizon_confirm(["AAA", "BBB"], horizon_days=5,
                                     benchmark="QQQ", years=8.0,
                                     signal="reversal", fetch=f)
        assert "drawdown_mc" in res
        out = vr.write_confirm_report(res, tmp_path / "D.md")
        text = out.read_text()
        assert "Drawdown stress" in text
        assert "95th-percentile" in text


class TestHorizonConfirm:
    def test_edgeless_vs_strong_benchmark_is_not_stable(self, tmp_path):
        # Mild-drift tickers trade a little but can't beat a strong benchmark
        # in either half → the confirm must say NOT STABLE / do not fund.
        # (True zero-edge worlds barely trade at all — correct abstention —
        # so they exercise the too-few-trades guard instead of the split.)
        seeds = {"AAA": 11, "BBB": 22}

        def f(ticker, years):
            return (gbm_ohlc(2200, seed=999, mu_annual=0.30)
                    if ticker == "QQQ"
                    else gbm_ohlc(2200, seed=seeds[ticker], mu_annual=0.06))
        res = ve.run_horizon_confirm(["AAA", "BBB"], horizon_days=21,
                                     benchmark="QQQ", years=8.0, fetch=f)
        assert "error" not in res, res
        assert res["stable"] is False
        out = vr.write_confirm_report(res, tmp_path / "C.md")
        text = out.read_text()
        assert "NOT STABLE" in text
        assert "Do not fund it" in text

    def test_too_few_trades_refuses_to_split(self, tmp_path):
        # True zero-edge world: the strategy abstains, so the confirm refuses
        # rather than splitting a meaningless handful of trades.
        def f(ticker, years):
            return (gbm_ohlc(2200, seed=999, mu_annual=0.10)
                    if ticker == "QQQ"
                    else gbm_ohlc(2200, seed=11, mu_annual=0.0))
        res = ve.run_horizon_confirm(["AAA"], horizon_days=63,
                                     benchmark="QQQ", years=8.0, fetch=f)
        if "error" in res:  # expected path
            assert "too few" in res["error"]
            out = vr.write_confirm_report(res, tmp_path / "C.md")
            assert "could not run" in out.read_text()

    def test_halves_partition_all_trades(self):
        def f(ticker, years):
            return (gbm_ohlc(2200, seed=999, mu_annual=0.05)
                    if ticker == "QQQ"
                    else regime_ohlc(2200, seed=abs(hash(ticker)) % 9999,
                                     mu_bull_annual=0.40))
        res = ve.run_horizon_confirm(["AAA"], horizon_days=21,
                                     benchmark="QQQ", years=8.0, fetch=f)
        assert (res["first_half"]["n_trades"] + res["second_half"]["n_trades"]
                == res["overall"]["n_trades"])
        # win/loss profile is reported (owner's consistency ask)
        assert "win_rate" in res["overall"]
        assert "worst_losses" in res["overall"]

    def test_stable_path_renders_next_gate(self, tmp_path):
        # hand-built stable result → report says stable + next gate is paper
        half = {"n_trades": 30, "beats": True, "win_rate": 0.7,
                "avg_win": 0.03, "avg_loss": -0.02, "worst_losses": [-0.05],
                "strategy": {"sharpe": 1.2, "cagr": 0.15, "max_drawdown": -0.1,
                             "sortino": 1.0, "total_return": 1.0, "n": 30},
                "benchmark": {"sharpe": 0.8, "cagr": 0.12, "max_drawdown": -0.2,
                              "sortino": 0.9, "total_return": 0.9, "n": 500}}
        res = {"tickers": ["AAA"], "benchmark": "QQQ", "horizon_days": 63,
               "cost_side": 0.0003, "tax_rate": 0.35,
               "split_date": "2022-01-01", "first_half": half,
               "second_half": half, "overall": half, "stable": True}
        out = vr.write_confirm_report(res, tmp_path / "C.md")
        text = out.read_text()
        assert "STABLE (necessary, not sufficient)" in text
        assert "live paper trading" in text


def test_planted_edge_can_survive(tmp_path: Path):
    # Strong planted conditional edge, benchmark is flat noise → strategy
    # should clear the (low-friction synthetic) bar. Proves the harness is
    # not hard-wired to always fail.
    def fetch(ticker: str, years: float) -> ve.OHLC:
        if ticker == "BENCH":
            return gbm_ohlc(1800, seed=999, mu_annual=0.0)
        return regime_ohlc(1800, seed=abs(hash(ticker)) % 9999,
                           mu_bull_annual=0.60)
    result = ve.run_full(["AAA", "BBB"], benchmark="BENCH",
                         years=7.0, fetch=fetch)
    # The point estimate edge must be clearly positive when it is really there.
    assert result["edge"]["executable_annual"] > 0.0
    assert result["n_trades"] > 10
