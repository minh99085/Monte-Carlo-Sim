"""Phase 5 acceptance tests: v2 default flip + observer extraction.

The golden hashes below were captured from the engine BEFORE Phase 5's
changes (pre-flip, pre-extraction). They are permanent regression anchors:
now that both engine paths share code, a bug could change legacy and v2
identically and slip past the pairwise equivalence tests — these absolute
anchors would still catch it.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import mc_core
from mc_core import (
    DrawdownObserver,
    SampleRecorder,
    SimulationConfig,
    simulate,
)

_HIST = np.random.default_rng(1).normal(0.0004, 0.012, size=750)


def _cfg(model, **kw) -> SimulationConfig:
    base = dict(ticker="G", s0=100.0, paths=2500, horizon=21, mu=0.08,
                sigma=0.25, seed=99, chunk_size=700, sample_paths=4,
                model=model)
    if model in mc_core.BOOTSTRAP_MODELS:
        base["historical_returns"] = _HIST
    base.update(kw)
    return SimulationConfig(**base)


def _hashes(result):
    return (hashlib.sha256(result.final_values.tobytes()).hexdigest()[:16],
            hashlib.sha256(result.sample_trajectories.tobytes()).hexdigest()[:16])


# (final_values sha, sample_trajectories sha, prob_ruin, mean_max_drawdown,
#  mean_drawdown_duration) — captured pre-Phase-5.
GOLDEN_PLAIN = {
    "GBM Normal": ("03812b6a95a9e91a", "90f2884480a3914a",
                   0.0, 0.06810303009827672, 12.8164),
    "Student-t GBM": ("26670c31a9a0ef19", "835de550bea95474",
                      0.0, 0.06470769684223716, 12.74),
    "Historical Bootstrap": ("2ef640b998e42a15", "decd9ce1e74c4baf",
                             0.0, 0.05387514323564889, 13.3124),
    "Block Bootstrap": ("3bcbcafe71b203fb", "bc11ae346e708911",
                        0.0, 0.055651108481404615, 13.6908),
    "Merton Jump-Diffusion": ("ada00f0f97e3eb61", "d9c5b302be63d13d",
                              0.0, 0.06990357508212594, 12.8244),
    "Regime Switching": ("d6ed0e285e9ff587", "deb146ad5216bb34",
                         0.0, 0.08236019931968278, 12.6876),
    "Heston Stochastic Volatility": ("d1542d45d4f7fed0", "fa993ad1e217bb69",
                                     0.0, 0.06777819997359429, 12.5828),
    "GARCH(1,1)": ("a4d8b38669b28c34", "2fec191b41fa9d66",
                   0.0, 0.06652270422032976, 12.8644),
    "Kou Jump-Diffusion": ("73eb862e660f17b7", "7a2395ec7c696e85",
                           0.0, 0.07056790106847384, 12.7308),
}

GOLDEN_VARIANTS = {
    "antithetic": (dict(model="GBM Normal", variance_reduction="antithetic"),
                   "cb92bc2be69704e5", "ca7a9f18bb6a27b7"),
    "sobol-heston": (dict(model="Heston Stochastic Volatility",
                          variance_reduction="sobol"),
                     "c81a4312a9b7c1c3", "bac521f78f838e84"),
    "stress": (dict(model="GBM Normal", stress_enabled=True,
                    stress_crash_pct=0.05, stress_vol_multiplier=1.2,
                    stress_drift_haircut=0.4),
               "bc1c882e0d5382a0", "a2cd9708c3f9b638"),
    # Control variate adjusts only the reported mean; raw samples must stay
    # identical to the plain run.
    "control-variate": (dict(model="GBM Normal",
                             variance_reduction="control_variate"),
                        "03812b6a95a9e91a", "90f2884480a3914a"),
    "cost-halfdrift": (dict(model="GBM Normal", cost=0.002,
                            drift_mode=mc_core.DRIFT_HALF),
                       "48305063cf9dd800", "341b29dc6d916c23"),
}


class TestGoldenAnchors:
    @pytest.mark.parametrize("model", sorted(GOLDEN_PLAIN))
    @pytest.mark.parametrize("engine", ["v2", "legacy"])
    def test_pre_phase5_outputs_reproduced(self, model, engine):
        hf, hs, ruin, mdd, mddur = GOLDEN_PLAIN[model]
        r = simulate(_cfg(model, engine=engine))
        assert _hashes(r) == (hf, hs), (model, engine)
        assert r.stats["prob_ruin"] == ruin
        assert r.stats["mean_max_drawdown"] == mdd
        assert r.stats["mean_drawdown_duration"] == mddur

    @pytest.mark.parametrize("tag", sorted(GOLDEN_VARIANTS))
    def test_variant_goldens(self, tag):
        kw, hf, hs = GOLDEN_VARIANTS[tag]
        if "sobol" in str(kw.get("variance_reduction")) \
                and not mc_core.sobol_available():
            pytest.skip("SciPy Sobol not available")
        r = simulate(_cfg(**kw))
        assert _hashes(r) == (hf, hs), tag


class TestDefaultFlip:
    def test_default_engine_is_v2(self):
        assert SimulationConfig().engine == "v2"
        r = simulate(_cfg("GBM Normal"))
        assert r.stats["engine"] == "v2"

    def test_legacy_escape_hatch_field(self):
        r = simulate(_cfg("GBM Normal", engine="legacy"))
        assert r.stats["engine"] == "legacy"
        # bit-identical either way
        assert _hashes(r) == GOLDEN_PLAIN["GBM Normal"][:2]

    def test_legacy_escape_hatch_env(self, monkeypatch):
        monkeypatch.setenv(mc_core.ENGINE_ENV_VAR, "legacy")
        r = simulate(_cfg("GBM Normal"))       # config default v2, env wins
        assert r.stats["engine"] == "legacy"
        assert _hashes(r) == GOLDEN_PLAIN["GBM Normal"][:2]

    def test_engine_key_always_present(self):
        for eng in ("v2", "legacy"):
            assert "engine" in simulate(_cfg("GBM Normal", engine=eng)).stats

    def test_tactical_pipeline_unaffected_numerically(self):
        # tactical_simulator builds SimulationConfig without an engine field;
        # the flipped default must not change its numbers (v2 is
        # bit-identical). Compare against a forced-legacy run.
        from tactical_config import preset_5_day
        from tactical_simulator import run_tactical_simulation
        cfg = preset_5_day("T", paths=1_500, seed=3, starting_price=100.0,
                           annual_volatility=0.2, annual_drift=0.0)
        default = run_tactical_simulation(cfg)
        import os
        os.environ[mc_core.ENGINE_ENV_VAR] = "legacy"
        try:
            legacy = run_tactical_simulation(cfg)
        finally:
            del os.environ[mc_core.ENGINE_ENV_VAR]
        assert np.array_equal(default.pnl, legacy.pnl)


class TestObserverUnits:
    def test_drawdown_observer_matches_hand_computation(self):
        obs = DrawdownObserver(ruin_level=50.0, drawdown_threshold=0.30)
        p0 = np.array([100.0, 100.0, 100.0])
        obs.begin_chunk(p0)
        # path A: drifts up (no drawdown); B: -40% dip then recovery;
        # C: crashes through ruin.
        steps = [np.array([110.0, 90.0, 60.0]),
                 np.array([120.0, 60.0, 45.0]),
                 np.array([130.0, 80.0, 55.0])]
        for i, prices in enumerate(steps, start=1):
            obs.observe(i, prices)
        obs.end_chunk(steps[-1])
        vals = obs.values()
        assert vals[0] == pytest.approx(0.0)
        assert vals[1] == pytest.approx(0.40)       # 100 -> 60
        assert vals[2] == pytest.approx(0.55)       # 100 -> 45
        assert obs.drawdown_hits == 2               # >= 30% threshold
        assert obs.ruin_hits == 1                   # path C hit <= 50
        assert obs.dd_duration_sum == 0 + 3 + 3     # underwater streaks

    def test_drawdown_observer_multi_chunk_accumulates(self):
        obs = DrawdownObserver(ruin_level=10.0, drawdown_threshold=0.5)
        for _ in range(3):
            p = np.array([100.0, 100.0])
            obs.begin_chunk(p)
            obs.observe(1, np.array([40.0, 120.0]))   # one 60% dd per chunk
            obs.end_chunk(np.array([40.0, 120.0]))
        assert obs.drawdown_hits == 3
        assert obs.values().size == 6

    def test_sample_recorder_spans_chunks(self):
        rec = SampleRecorder(n_sample=5, steps=2, s0=100.0)
        # chunk of 3 paths then chunk of 4: rows 0-2 then 3-4 recorded
        for chunk_vals in ([11.0, 12.0, 13.0], [21.0, 22.0, 23.0, 24.0]):
            arr = np.asarray(chunk_vals)
            rec.begin_chunk(arr)
            rec.observe(1, arr)
            rec.observe(2, arr * 10)
            rec.end_chunk(arr)
        assert rec.matrix.shape == (5, 3)
        np.testing.assert_allclose(rec.matrix[:, 0], 100.0)
        np.testing.assert_allclose(rec.matrix[:, 1], [11, 12, 13, 21, 22])
        np.testing.assert_allclose(rec.matrix[:, 2], [110, 120, 130, 210, 220])

    def test_observers_run_on_v2_path_generator(self):
        # The whole point of extraction: drawdown metrics on option runs.
        from mc_engine import GBMProcess, PathGenerator
        from mc_payoffs import EuropeanPricer
        gen = PathGenerator(GBMProcess(0.05, 0.4, 1 / 252), s0=100.0,
                            paths=4_000, steps=126, chunk_size=1_000, seed=7)
        pricer = EuropeanPricer(strike=100.0, maturity=0.5, r=0.05)
        dd = DrawdownObserver(ruin_level=50.0, drawdown_threshold=0.20)
        gen.run([pricer, dd])
        assert pricer.values().size == 4_000
        assert dd.values().size == 4_000
        assert 0.0 < dd.drawdown_hits / 4_000 < 1.0   # 40% vol: some 20% dds
