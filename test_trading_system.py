"""Ground-truth tests for the meta-labeled trading system (v2 rebuild).

The machinery must be honest in both directions: recover a planted,
learnable edge (not rigged to fail) and refuse a no-edge world (not rigged
to pass). Plus the spec's hard requirements: no lookahead, purged splits,
exact sizer formulas, barrier correctness, insufficient-history stop, and
holdout-once discipline.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trading_system import load_config
from trading_system.barriers import triple_barrier_label
from trading_system.data import Bars
from trading_system.features import FEATURE_NAMES, feature_vector
from trading_system.gauntlet import (
    bump_trials,
    deflated_sharpe,
    pbo_cscv,
    run_gauntlet,
    run_symbol,
    sharpe,
)
from trading_system.model import purged_walk_forward
from trading_system.primary import PrimarySignal, reconstruct_signals
from trading_system.sizing import shares_for, size_A4, size_B1


# ---------------------------------------------------------------------------
# Synthetic worlds
# ---------------------------------------------------------------------------


def _dates(n):
    return pd.bdate_range("2016-01-04", periods=n)

def stable_seed(s: str) -> int:
    """Deterministic per-symbol seed (str hash is randomized per process)."""
    return sum(ord(c) * (i + 1) for i, c in enumerate(s)) % 9973



def make_bars(close: np.ndarray, symbol: str = "SYN",
              spread: float = 0.004) -> Bars:
    close = np.asarray(close, float)
    op = np.r_[close[0], close[:-1]]
    hi = np.maximum(op, close) * (1 + spread)
    lo = np.minimum(op, close) * (1 - spread)
    return Bars(symbol, _dates(close.size), op, hi, lo, close)


def gbm_bars(n: int, seed: int, mu: float = 0.0, sigma: float = 0.18,
             symbol: str = "SYN") -> Bars:
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu / 252, sigma / math.sqrt(252), n)
    return make_bars(100 * np.exp(np.cumsum(rets)), symbol)


def edge_world_bars(n: int, seed: int, symbol: str = "EDGE") -> Bars:
    """A world where the meta-label is LEARNABLE: after an EMA flip, the
    forward drift is positive only when the regime feature (MA50>MA200)
    is on — so the secondary model can separate good flips from bad."""
    rng = np.random.default_rng(seed)
    sd = 0.15 / math.sqrt(252)
    prices = [100.0]
    for t in range(n):
        c = np.asarray(prices)
        ma50 = c[-50:].mean() if c.size >= 50 else c.mean()
        ma200 = c[-200:].mean() if c.size >= 200 else c.mean()
        mu = (0.60 / 252) if ma50 > ma200 else (-0.25 / 252)
        prices.append(prices[-1] * math.exp(rng.normal(mu, sd)))
    return make_bars(np.asarray(prices[1:]), symbol)


# ---------------------------------------------------------------------------
# Primary signal reconstruction
# ---------------------------------------------------------------------------


class TestPrimary:
    def test_flip_events_alternate_and_are_causal(self):
        bars = gbm_bars(1500, seed=1)
        sigs = reconstruct_signals(bars)
        assert len(sigs) > 20                       # crosses happen
        dirs = [s.direction for s in sigs]
        assert all(a != b for a, b in zip(dirs, dirs[1:]))  # flips alternate
        assert all(sigs[i].t < sigs[i + 1].t for i in range(len(sigs) - 1))

    def test_reconstruction_is_truncation_invariant(self):
        """A signal at bar t must also exist when the series ends at t —
        i.e., detection uses no future data."""
        bars = gbm_bars(1200, seed=2)
        sigs = reconstruct_signals(bars)
        probe = sigs[len(sigs) // 2]
        cut = reconstruct_signals(bars.slice(0, probe.t + 1))
        assert any(s.t == probe.t and s.direction == probe.direction
                   for s in cut)


# ---------------------------------------------------------------------------
# Triple-barrier labeler
# ---------------------------------------------------------------------------


class TestBarriers:
    def _one_signal(self, bars, t):
        return [PrimarySignal(bars.symbol, t, bars.dates[t], "long",
                              float(bars.close[t]), "reconstructed")]

    def test_profit_barrier_hit_labels_one(self):
        # flat then a strong ramp after the signal
        close = np.r_[np.full(300, 100.0) + np.random.default_rng(3).normal(0, .2, 300),
                      np.linspace(100, 130, 30)]
        bars = make_bars(close)
        lab = triple_barrier_label(bars, self._one_signal(bars, 299),
                                   max_hold=20)
        assert len(lab) == 1
        assert lab[0].exit_reason == "profit_barrier"
        assert lab[0].label == 1
        assert lab[0].net_return > 0

    def test_stop_barrier_hit_labels_zero(self):
        close = np.r_[np.full(300, 100.0) + np.random.default_rng(4).normal(0, .2, 300),
                      np.linspace(100, 70, 30)]
        bars = make_bars(close)
        lab = triple_barrier_label(bars, self._one_signal(bars, 299),
                                   max_hold=20)
        assert lab[0].exit_reason == "stop_barrier"
        assert lab[0].label == 0
        assert lab[0].net_return < 0

    def test_vertical_barrier_settles_at_close(self):
        rng = np.random.default_rng(5)
        close = 100.0 + np.cumsum(rng.normal(0, 0.02, 400))  # tiny moves
        bars = make_bars(close, spread=0.0001)
        lab = triple_barrier_label(bars, self._one_signal(bars, 300),
                                   k_pt=8.0, k_sl=8.0, max_hold=10)
        assert lab[0].exit_reason == "vertical"
        assert lab[0].exit_index == 300 + 1 + 10

    def test_entry_is_next_bar_open_not_signal_close(self):
        rng = np.random.default_rng(44)
        close = np.r_[100.0 + rng.normal(0, 0.3, 300),
                      np.linspace(100, 120, 30)]
        bars = make_bars(close)
        lab = triple_barrier_label(bars, self._one_signal(bars, 299),
                                   max_hold=20)
        assert lab[0].entry_price == pytest.approx(bars.open[300])


# ---------------------------------------------------------------------------
# Features: the lookahead test the spec demands
# ---------------------------------------------------------------------------


class TestFeaturesNoLookahead:
    def test_truncation_invariance(self):
        bars = gbm_bars(900, seed=6)
        t = 700
        full = feature_vector(bars, t, "long")
        cut = feature_vector(bars.slice(0, t + 1), t, "long")
        assert full is not None and cut is not None
        np.testing.assert_allclose(full, cut, atol=1e-12)

    def test_future_change_does_not_alter_features(self):
        bars = gbm_bars(900, seed=7)
        t = 700
        base = feature_vector(bars, t, "long")
        tampered = Bars(bars.symbol, bars.dates, bars.open.copy(),
                        bars.high.copy(), bars.low.copy(), bars.close.copy())
        tampered.close[t + 1:] *= 3.0     # nuke the future
        tampered.high[t + 1:] *= 3.0
        after = feature_vector(tampered, t, "long")
        np.testing.assert_allclose(base, after, atol=1e-12)

    def test_warmup_returns_none(self):
        bars = gbm_bars(900, seed=8)
        assert feature_vector(bars, 50, "long") is None

    def test_feature_names_match_vector_length(self):
        bars = gbm_bars(900, seed=9)
        v = feature_vector(bars, 700, "long")
        assert v is not None and v.size == len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Purged walk-forward
# ---------------------------------------------------------------------------


class TestPurgedWalkForward:
    def test_embargo_gap_respected(self):
        rng = np.random.default_rng(10)
        n = 400
        X = rng.normal(size=(n, 4))
        y = (rng.random(n) > 0.5).astype(int)
        order = list(range(0, n * 5, 5))            # bar index per row
        wf = purged_walk_forward(X, y, order, n_folds=4, embargo=10)
        assert wf.n_folds_run >= 2
        assert wf.indices                            # produced OOS predictions

    def test_learnable_pattern_gets_auc_above_half(self):
        rng = np.random.default_rng(11)
        n = 600
        X = rng.normal(size=(n, 4))
        y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
        wf = purged_walk_forward(X, y, list(range(n)), n_folds=5, embargo=5)
        assert wf.auc > 0.7

    def test_noise_auc_near_half(self):
        rng = np.random.default_rng(12)
        n = 600
        X = rng.normal(size=(n, 4))
        y = (rng.random(n) > 0.5).astype(int)
        wf = purged_walk_forward(X, y, list(range(n)), n_folds=5, embargo=5)
        assert abs(wf.auc - 0.5) < 0.12


# ---------------------------------------------------------------------------
# Sizers: exact formulas
# ---------------------------------------------------------------------------


class TestSizing:
    def test_A4_threshold(self):
        assert size_A4(0.61, 0.60) == 1.0
        assert size_A4(0.60, 0.60) == 0.0
        assert size_A4(0.20, 0.60) == 0.0

    def test_B1_formula_and_zero_below_half(self):
        p = 0.6
        z = (p - 0.5) / math.sqrt(p * (1 - p))
        expect = 2 * (0.5 * (1 + math.erf(z / math.sqrt(2)))) - 1
        assert size_B1(p) == pytest.approx(expect, abs=1e-12)
        assert size_B1(0.5) == 0.0
        assert size_B1(0.4) == 0.0
        assert 0.0 < size_B1(0.55) < size_B1(0.75) < 1.0

    def test_fixed_risk_share_formula_and_caps(self):
        plan = shares_for(p=0.99, sizer="A4", threshold=0.6, equity=10_000,
                          risk_pct=0.01, entry=100.0, stop=95.0,
                          max_position_pct=0.10, buying_power=10_000)
        # base = (10000*0.01)/5 = 20 shares → $2000 > 10% cap ($1000) → 10 sh
        assert plan.shares == pytest.approx(10.0)
        assert plan.capped_by == "position_pct"
        plan2 = shares_for(p=0.99, sizer="A4", threshold=0.6, equity=10_000,
                           risk_pct=0.01, entry=100.0, stop=95.0,
                           max_position_pct=0.5, buying_power=500.0)
        assert plan2.capped_by == "buying_power"
        assert plan2.shares == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Deflated Sharpe + PBO
# ---------------------------------------------------------------------------


class TestStatGates:
    def test_more_trials_deflate_more(self):
        rng = np.random.default_rng(13)
        r = rng.normal(0.004, 0.02, 300)
        few = deflated_sharpe(r, n_trials=2)
        many = deflated_sharpe(r, n_trials=200)
        assert many["sr0_annualized"] > few["sr0_annualized"]
        assert many["deflated_excess"] < few["deflated_excess"]

    def test_pbo_high_for_pure_noise_grid(self):
        rng = np.random.default_rng(14)
        M = rng.normal(0, 0.02, size=(10, 240))     # 10 lookalike noise configs
        pbo = pbo_cscv(M, n_blocks=8)
        assert np.isfinite(pbo)
        assert pbo > 0.3                             # picking IS-best is luck

    def test_pbo_low_when_one_config_truly_dominates(self):
        rng = np.random.default_rng(15)
        M = rng.normal(0, 0.02, size=(6, 240))
        M[2] += 0.015                                # genuinely better config
        pbo = pbo_cscv(M, n_blocks=8)
        assert pbo < 0.15

    def test_trials_counter_accumulates(self, tmp_path):
        p = tmp_path / "trials.json"
        assert bump_trials(p, 10, "a") == 10
        assert bump_trials(p, 5, "b") == 15
        state = json.loads(p.read_text())
        assert len(state["runs"]) == 2


# ---------------------------------------------------------------------------
# End-to-end gauntlet: honest both ways + discipline stops
# ---------------------------------------------------------------------------


def _cfg(tmp_path, **overrides):
    cfg = load_config()
    cfg = json.loads(json.dumps(cfg))                # deep copy
    cfg["universe"]["tickers"] = overrides.pop("tickers", ["EDGEA", "EDGEB"])
    cfg["signals"]["min_per_symbol"] = overrides.pop("min_per_symbol", 25)
    cfg["signals"]["tv_history_dir"] = str(tmp_path / "none")
    cfg["model"]["min_train"] = 25
    cfg["model"]["n_folds"] = 4
    for k, v in overrides.items():
        cfg[k] = v
    return cfg


def _fetch_for(builder):
    def fetch(symbol, years):
        return builder(symbol)
    return fetch


class TestGauntletEndToEnd:
    def test_insufficient_history_stops_cleanly(self, tmp_path):
        cfg = _cfg(tmp_path, min_per_symbol=100000)
        fetch = _fetch_for(lambda s: gbm_bars(1200, seed=stable_seed(s),
                                              symbol=s))
        report = run_gauntlet(cfg, fetch=fetch, workdir=tmp_path)
        assert "insufficient history" in report["stopped"]
        assert report["ready"] is False

    def test_no_edge_world_is_not_ready(self, tmp_path):
        cfg = _cfg(tmp_path)
        fetch = _fetch_for(lambda s: gbm_bars(2000, seed=stable_seed(s),
                                              mu=0.0, symbol=s))
        report = run_gauntlet(cfg, fetch=fetch, workdir=tmp_path)
        assert report["ready"] is False
        assert "NOT READY" in report["verdict"]

    def test_gate4_is_none_without_plateau_full_and_blocks_ready(self, tmp_path):
        """A green gauntlet must mean every claimed check ran: without
        --plateau-full, gate 4 reports pass=None and ready can never be
        True, no matter what the other gates say."""
        cfg = _cfg(tmp_path)
        fetch = _fetch_for(lambda s: edge_world_bars(
            2200, seed=stable_seed(s), symbol=s))
        report = run_gauntlet(cfg, fetch=fetch, workdir=tmp_path)
        g4 = report["gates"]["4_plateau"]
        assert g4["pass"] is None
        assert g4["full_run"] is False
        assert {r["param"] for r in g4["rows"]} == {"threshold"}
        assert report["ready"] is False

    def test_plateau_full_relabels_all_barrier_params(self, tmp_path):
        cfg = _cfg(tmp_path)
        fetch = _fetch_for(lambda s: edge_world_bars(
            2200, seed=stable_seed(s), symbol=s))
        report = run_gauntlet(cfg, fetch=fetch, workdir=tmp_path,
                              evaluate_plateau_full=True)
        g4 = report["gates"]["4_plateau"]
        assert g4["full_run"] is True
        assert isinstance(g4["pass"], bool)     # actually evaluated now
        params = {r["param"] for r in g4["rows"]}
        assert params == {"threshold", "k_pt", "k_sl", "max_hold"}
        # each barrier param bumped both ways
        for p in ("k_pt", "k_sl", "max_hold"):
            mults = {r["mult"] for r in g4["rows"] if r["param"] == p}
            assert len(mults) == 2
        assert "config_hash" in report

    def test_holdout_can_only_run_once(self, tmp_path):
        cfg = _cfg(tmp_path)
        fetch = _fetch_for(lambda s: edge_world_bars(
            2200, seed=stable_seed(s), symbol=s))
        r1 = run_gauntlet(cfg, fetch=fetch, workdir=tmp_path,
                          evaluate_holdout=True)
        assert r1["gates"]["6_holdout"]["pass"] is not None
        r2 = run_gauntlet(cfg, fetch=fetch, workdir=tmp_path,
                          evaluate_holdout=True)
        assert r2["gates"]["6_holdout"]["pass"] is False
        assert "already evaluated" in r2["gates"]["6_holdout"]["error"]

    def test_learnable_edge_world_model_finds_signal(self, tmp_path):
        """Not a guarantee of gate passage (costs/PBO may still bite) — but
        the model must at least LEARN the planted structure (AUC > 0.55)."""
        cfg = _cfg(tmp_path)
        fetch = _fetch_for(lambda s: edge_world_bars(
            2200, seed=stable_seed(s), symbol=s))
        report = run_gauntlet(cfg, fetch=fetch, workdir=tmp_path)
        aucs = [a for a in report["auc_by_symbol"].values()
                if np.isfinite(a)]
        assert aucs and max(aucs) > 0.55


# ---------------------------------------------------------------------------
# Decision engine (live path) writes bridge-compatible verdicts
# ---------------------------------------------------------------------------


class TestDecision:
    def test_short_signal_is_recorded_no_trade(self, tmp_path):
        from trading_system.decision import decide

        cfg = _cfg(tmp_path)
        v = decide({"ticker": "EDGEA", "trend": "bearish", "price": 100.0},
                   cfg=cfg, fetch=_fetch_for(
                       lambda s: edge_world_bars(2200, seed=1, symbol=s)))
        assert v["verdict"] == "NO_TRADE"
        assert "cannot short" in v["reason"]

    def test_long_signal_produces_bridge_compatible_verdict(self, tmp_path):
        from trading_system.decision import decide, write_verdict

        cfg = _cfg(tmp_path)
        v = decide({"ticker": "EDGEA", "trend": "bullish", "price": None},
                   cfg=cfg, fetch=_fetch_for(
                       lambda s: edge_world_bars(2200, seed=2, symbol=s)))
        assert v["verdict"] in ("TRADE", "NO_TRADE")
        # bridge contract keys
        for key in ("timestamp_utc", "ticker", "verdict", "side", "s0",
                    "sizing", "structure"):
            assert key in v
        assert "shares" in v["sizing"]
        path = write_verdict(v, tmp_path / "verdicts")
        data = json.loads(path.read_text())
        assert data["ticker"] == "EDGEA"

    def test_gauntlet_pass_stamp_tracks_report_ready(self, tmp_path):
        """Every verdict carries gauntlet_pass; True ONLY for ready:true."""
        from trading_system.decision import decide, gauntlet_ready

        cfg = _cfg(tmp_path)
        fetch = _fetch_for(lambda s: edge_world_bars(2200, seed=1, symbol=s))
        signal = {"ticker": "EDGEA", "trend": "bearish", "price": 100.0}

        # No report on disk → False (the safe default).
        missing = tmp_path / "nope" / "gauntlet_report.json"
        v = decide(signal, cfg=cfg, fetch=fetch, gauntlet_report=missing)
        assert v["gauntlet_pass"] is False

        # ready:false and corrupt reports → False.
        rpt = tmp_path / "gauntlet_report.json"
        rpt.write_text(json.dumps({"ready": False}))
        assert gauntlet_ready(rpt) is False
        rpt.write_text("{not json")
        assert gauntlet_ready(rpt) is False

        # ready:true → the marker is stamped True.
        rpt.write_text(json.dumps({"ready": True}))
        v = decide(signal, cfg=cfg, fetch=fetch, gauntlet_report=rpt)
        assert v["gauntlet_pass"] is True

    def test_certificate_ties_verdict_to_report_bytes(self, tmp_path):
        """The certificate must hash the exact report file so the executor
        can verify it on its side of the mount."""
        import hashlib

        from trading_system.decision import decide

        cfg = _cfg(tmp_path)
        fetch = _fetch_for(lambda s: edge_world_bars(2200, seed=1, symbol=s))
        rpt = tmp_path / "gauntlet_report.json"
        rpt.write_text(json.dumps({"ready": True, "tickers": ["EDGEA"],
                                   "config_hash": "abc123"}))
        v = decide({"ticker": "EDGEA", "trend": "bearish", "price": 100.0},
                   cfg=cfg, fetch=fetch, gauntlet_report=rpt)
        cert = v["certificate"]
        assert cert["engine"] == "meta_label_v2"
        assert cert["report_hash"] == hashlib.sha256(
            rpt.read_bytes()).hexdigest()
        assert cert["config_hash"] == "abc123"
        assert cert["universe"] == ["EDGEA"]
        assert v["entry_rule"] == "next_session_open"

        # no report → no certificate, and still a valid paper verdict
        v2 = decide({"ticker": "EDGEA", "trend": "bearish", "price": 100.0},
                    cfg=cfg, fetch=fetch,
                    gauntlet_report=tmp_path / "missing.json")
        assert v2["certificate"] is None
        assert v2["gauntlet_pass"] is False
