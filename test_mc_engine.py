"""Phase 1 tests for mc_engine (see REVIEW.md).

The core guarantees under test:
1. With the feature flag on, GBM and Heston runs are **bit-identical** to the
   legacy engine (same RNG draw order, same arithmetic).
2. The standalone PathGenerator + TerminalValuePricer pipeline reproduces the
   legacy engine's terminal values exactly for the same seed, and agrees
   within Monte Carlo tolerance across seeds.
3. Chunk-safety is preserved: no ``paths × steps`` allocation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import mc_core
import mc_engine
from mc_core import SimulationConfig, simulate
from mc_engine import (
    GBMProcess,
    HestonProcess,
    PathGenerator,
    StreamingMoments,
    TerminalValuePricer,
    generator_from_config,
    process_from_config,
)


def _cfg(**kw) -> SimulationConfig:
    base = dict(
        ticker="TEST",
        s0=100.0,
        paths=4_000,
        horizon=30,
        mu=0.08,
        sigma=0.25,
        seed=42,
        chunk_size=1_000,
        sample_paths=5,
    )
    base.update(kw)
    return SimulationConfig(**base)


def _final_values(cfg: SimulationConfig) -> np.ndarray:
    return simulate(cfg).final_values


# ---------------------------------------------------------------------------
# 1. Flagged engine is bit-identical to legacy
# ---------------------------------------------------------------------------


class TestFlaggedEquivalence:
    @pytest.mark.parametrize("model", [mc_core.MODEL_GBM, mc_core.MODEL_HESTON])
    def test_exact_match_plain(self, model):
        legacy = _final_values(_cfg(model=model))
        v2 = _final_values(_cfg(model=model, engine="v2"))
        assert np.array_equal(legacy, v2)

    @pytest.mark.parametrize("model", [mc_core.MODEL_GBM, mc_core.MODEL_HESTON])
    def test_exact_match_antithetic(self, model):
        legacy = _final_values(_cfg(model=model, variance_reduction="antithetic"))
        v2 = _final_values(
            _cfg(model=model, variance_reduction="antithetic", engine="v2"))
        assert np.array_equal(legacy, v2)

    @pytest.mark.skipif(not mc_core.sobol_available(),
                        reason="SciPy Sobol not available")
    @pytest.mark.parametrize("model", [mc_core.MODEL_GBM, mc_core.MODEL_HESTON])
    def test_exact_match_sobol(self, model):
        legacy = _final_values(_cfg(model=model, variance_reduction="sobol"))
        v2 = _final_values(
            _cfg(model=model, variance_reduction="sobol", engine="v2"))
        assert np.array_equal(legacy, v2)

    def test_exact_match_with_stress_crash(self):
        kw = dict(model=mc_core.MODEL_GBM, stress_enabled=True,
                  stress_crash_pct=0.10, stress_vol_multiplier=1.5,
                  stress_drift_haircut=0.5)
        legacy = _final_values(_cfg(**kw))
        v2 = _final_values(_cfg(engine="v2", **kw))
        assert np.array_equal(legacy, v2)

    def test_stats_equal_and_engine_recorded(self):
        r_legacy = simulate(_cfg(model=mc_core.MODEL_HESTON))
        r_v2 = simulate(_cfg(model=mc_core.MODEL_HESTON, engine="v2"))
        assert r_v2.stats["engine"] == "v2"
        assert "engine" not in r_legacy.stats  # default schema unchanged
        for key in ("expected_value", "median_value", "prob_profit",
                    "std_value", "prob_ruin", "mean_max_drawdown"):
            assert r_legacy.stats[key] == r_v2.stats[key], key

    def test_env_var_flag(self, monkeypatch):
        legacy = _final_values(_cfg())
        monkeypatch.setenv(mc_core.ENGINE_ENV_VAR, "v2")
        r = simulate(_cfg())
        assert r.stats["engine"] == "v2"
        assert np.array_equal(legacy, r.final_values)

    def test_unported_model_falls_back_to_legacy(self):
        kw = dict(model=mc_core.MODEL_STUDENT_T, t_df=5.0)
        legacy = _final_values(_cfg(**kw))
        r = simulate(_cfg(engine="v2", **kw))
        assert np.array_equal(legacy, r.final_values)
        assert "not ported" in r.stats["engine"]

    def test_invalid_engine_rejected(self):
        with pytest.raises(ValueError, match="engine"):
            _cfg(engine="v3").validate()


# ---------------------------------------------------------------------------
# 2. Standalone PathGenerator pipeline
# ---------------------------------------------------------------------------


class TestStandalonePipeline:
    @pytest.mark.parametrize("model", [mc_core.MODEL_GBM, mc_core.MODEL_HESTON])
    def test_reproduces_legacy_terminal_values_same_seed(self, model):
        cfg = _cfg(model=model, cost=0.001)
        legacy = _final_values(cfg)
        gen = generator_from_config(cfg)
        pricer = TerminalValuePricer(cfg.s0, cfg.cost)
        gen.run([pricer])
        assert np.array_equal(legacy, pricer.values())

    def test_statistical_agreement_across_seeds(self):
        # Different seed -> different samples, but the pipeline must agree
        # with the legacy engine within Monte Carlo error.
        cfg_a = _cfg(paths=20_000, seed=7)
        cfg_b = _cfg(paths=20_000, seed=1234)
        legacy = simulate(cfg_a)
        gen = generator_from_config(cfg_b)
        pricer = TerminalValuePricer(cfg_b.s0, cfg_b.cost)
        gen.run([pricer])
        v = pricer.values()
        se = legacy.stats["std_value"] / math.sqrt(cfg_a.paths)
        assert abs(float(np.mean(v)) - legacy.stats["expected_value"]) < 6 * se
        assert np.std(v) == pytest.approx(legacy.stats["std_value"], rel=0.05)

    def test_unported_model_raises(self):
        with pytest.raises(ValueError, match="not ported"):
            generator_from_config(_cfg(model=mc_core.MODEL_GARCH))

    def test_chunk_safety(self):
        cfg = _cfg(paths=10_000, chunk_size=500, horizon=50)
        gen = generator_from_config(cfg)
        pricer = TerminalValuePricer(cfg.s0, cfg.cost)
        mem = gen.run([pricer])
        assert pricer.values().size == cfg.paths
        assert mem.is_chunk_safe
        assert mem.peak_vector_elements == 500
        # No paths x steps allocation: nothing 2-D beyond the (optional)
        # per-chunk Sobol block, which this run does not use.
        assert mem.peak_matrix_elements == 0
        assert mem.full_matrix_elements == cfg.paths * (cfg.horizon + 1)

    def test_multiple_pricers_stream_together(self):
        # A tiny streaming observer alongside the terminal pricer: running
        # maximum per path (the lookback building block from REVIEW.md).
        class RunningMax(mc_engine.PathPricer):
            def __init__(self):
                self._chunks = []

            def begin_chunk(self, prices0):
                self._cur = prices0.copy()

            def observe(self, step_i, prices):
                np.maximum(self._cur, prices, out=self._cur)

            def end_chunk(self, prices):
                self._chunks.append(self._cur)

            def values(self):
                return np.concatenate(self._chunks)

        cfg = _cfg(paths=2_000, chunk_size=700)
        gen = generator_from_config(cfg)
        terminal = TerminalValuePricer(cfg.s0, 0.0)
        peak = RunningMax()
        gen.run([terminal, peak])
        t, m = terminal.values(), peak.values()
        assert t.size == m.size == cfg.paths
        assert np.all(m >= t - 1e-12)          # running max dominates terminal
        assert np.all(m >= cfg.s0 - 1e-12)     # includes the start price


# ---------------------------------------------------------------------------
# 3. Process contracts + streaming moments
# ---------------------------------------------------------------------------


class TestProcessContracts:
    def test_gbm_drift_diffusion(self):
        p = GBMProcess(mu=0.08, sigma=0.25, dt=1 / 252)
        assert p.drift({}, 10) == pytest.approx(0.08 - 0.5 * 0.25 ** 2)
        assert p.diffusion({}, 10) == pytest.approx(0.25)
        assert p.factors == 1
        # default Euler evolve == explicit precomputation
        z = np.array([0.0, 1.0, -2.0])
        expected = (0.08 - 0.5 * 0.25 ** 2) / 252 + 0.25 * math.sqrt(1 / 252) * z
        np.testing.assert_allclose(p.evolve(None, {}, z, 3), expected)

    def test_heston_state_and_truncation(self):
        p = HestonProcess(0.05, 1 / 252, kappa=1.5, theta=0.04, xi=0.3,
                          rho=-0.7, v0=0.04)
        st = p.init_state(4)
        assert st["v"].shape == (4,)
        np.testing.assert_allclose(st["v"], 0.04)
        assert p.factors == 2
        # negative variance is truncated in both drift and diffusion
        st_neg = {"v": np.array([-0.01, 0.04])}
        np.testing.assert_allclose(p.diffusion(st_neg, 2),
                                   [0.0, math.sqrt(0.04)])
        drift = p.drift(st_neg, 2)
        assert drift[0] == pytest.approx(0.05)  # -0.5 * max(v,0) = 0

    def test_process_from_config_defaults(self):
        cfg = _cfg(model=mc_core.MODEL_HESTON, sigma=0.2).validate()
        p = process_from_config(cfg)
        assert isinstance(p, HestonProcess)
        # theta/v0 default to effective sigma^2, as in the legacy closure
        assert p.theta == pytest.approx(0.04)
        assert p.v0 == pytest.approx(0.04)
        assert process_from_config(_cfg(model=mc_core.MODEL_KOU)) is None

    def test_generator_rejects_bad_args(self):
        p = GBMProcess(0.05, 0.2, 1 / 252)
        with pytest.raises(ValueError):
            PathGenerator(p, s0=0.0, paths=10, steps=5)
        with pytest.raises(ValueError):
            PathGenerator(p, s0=100.0, paths=0, steps=5)


class TestStreamingMoments:
    def test_matches_numpy_over_chunks(self):
        rng = np.random.default_rng(3)
        data = rng.normal(5.0, 2.0, size=10_000)
        sm = StreamingMoments()
        for chunk in np.array_split(data, 7):
            sm.add(chunk)
        assert sm.n == data.size
        assert sm.mean == pytest.approx(float(np.mean(data)), rel=1e-12)
        assert sm.variance == pytest.approx(float(np.var(data, ddof=1)), rel=1e-10)
        assert sm.min == pytest.approx(float(data.min()))
        assert sm.max == pytest.approx(float(data.max()))

    def test_empty_and_single(self):
        sm = StreamingMoments()
        sm.add(np.array([]))
        assert sm.n == 0
        sm.add(np.array([3.0]))
        assert sm.n == 1 and sm.mean == 3.0 and sm.variance == 0.0


# ---------------------------------------------------------------------------
# Phase 3 Part B: Sobol + Brownian bridge
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not mc_core.sobol_available(),
                    reason="SciPy Sobol not available")
class TestBrownianBridge:
    S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.25, 1.0

    def _price(self, pricer_cls, n, steps, seed, mode, **pkw):
        from mc_payoffs import EuropeanPricer  # noqa: F401 (import guard)
        gen = mc_engine.PathGenerator(
            GBMProcess(self.R, self.SIG, self.T / steps),
            s0=self.S0, paths=n, steps=steps, chunk_size=n, seed=seed,
            sobol=(mode in ("sobol", "bridge")), bridge=(mode == "bridge"))
        p = pricer_cls(strike=self.K, maturity=self.T, r=self.R, **pkw)
        gen.run([p])
        return p.price()

    def test_bridge_increments_are_standard_normal(self):
        bb = mc_engine.BrownianBridge(16, 1.0 / 16)
        rng = np.random.default_rng(0)
        inc = bb.transform(rng.standard_normal((100_000, 16)))
        assert np.abs(inc.mean(axis=0)).max() < 0.02
        assert np.abs(inc.std(axis=0) - 1.0).max() < 0.02
        corr = np.corrcoef(inc[:, :6].T)
        assert np.abs(corr - np.eye(6)).max() < 0.02

    def test_bridge_terminal_level_uses_first_dimension(self):
        # With only dimension 0 nonzero, the whole path is the linear
        # interpolation of the terminal level: W(t_i) = (t_i/T) * sqrt(T) z0.
        steps = 8
        bb = mc_engine.BrownianBridge(steps, 1.0 / steps)
        z = np.zeros((1, steps))
        z[0, 0] = 1.0
        inc = bb.transform(z)[0]
        w = np.cumsum(inc) * math.sqrt(1.0 / steps)
        expect = (np.arange(1, steps + 1) / steps) * 1.0  # sqrt(T)=1
        np.testing.assert_allclose(w, expect, atol=1e-12)

    def test_bridge_single_step_is_identity(self):
        bb = mc_engine.BrownianBridge(1, 0.25)
        z = np.array([[1.7], [-0.3]])
        np.testing.assert_allclose(bb.transform(z), z, atol=1e-12)

    def test_european_price_matches_bs(self):
        from mc_payoffs import EuropeanPricer, black_scholes_price
        bs = black_scholes_price(self.S0, self.K, self.R, self.SIG, self.T)
        price = self._price(EuropeanPricer, 8_192, 32, 3, "bridge")
        assert abs(price - bs) < 0.05  # QMC at 8k paths is very tight

    def test_convergence_slope_steeper_than_prng(self):
        from mc_payoffs import EuropeanPricer, black_scholes_price
        bs = black_scholes_price(self.S0, self.K, self.R, self.SIG, self.T)
        sizes = [256, 1024, 4096]
        reps = 8

        def rmse(mode):
            out = []
            for n in sizes:
                errs = [self._price(EuropeanPricer, n, 32, 1000 + i, mode) - bs
                        for i in range(reps)]
                out.append(float(np.sqrt(np.mean(np.square(errs)))))
            return out

        slope = lambda r: float(np.polyfit(np.log(sizes), np.log(r), 1)[0])
        s_prng, s_bridge = slope(rmse("prng")), slope(rmse("bridge"))
        assert -0.80 < s_prng < -0.30, s_prng          # ~ N^-0.5
        assert s_bridge < -0.80, s_bridge              # ~ N^-1
        assert s_bridge < s_prng - 0.25, (s_prng, s_bridge)

    def test_asian_bridge_beats_plain_sobol(self):
        # The headline: bridging front-loads variance into low Sobol
        # dimensions, which is what path-dependent payoffs need. Exact
        # discrete-geometric closed form as the error reference.
        from mc_payoffs import AsianGeometricPricer, geometric_asian_call_discrete
        ref = geometric_asian_call_discrete(self.S0, self.K, self.R,
                                            self.SIG, self.T, 64)

        def rmse(mode):
            errs = [self._price(AsianGeometricPricer, 4_096, 64, 2000 + i,
                                mode) - ref for i in range(8)]
            return float(np.sqrt(np.mean(np.square(errs))))

        e_sobol, e_bridge = rmse("sobol"), rmse("bridge")
        assert e_bridge < 0.5 * e_sobol, (e_sobol, e_bridge)

    def test_bridge_requires_sobol(self):
        with pytest.raises(ValueError, match="requires sobol"):
            mc_engine.PathGenerator(GBMProcess(0.05, 0.2, 1 / 32), s0=100,
                                    paths=10, steps=32, bridge=True)

    def test_bridge_rejects_multifactor(self):
        with pytest.raises(ValueError, match="single-factor"):
            mc_engine.PathGenerator(
                HestonProcess(0.05, 1 / 32, kappa=1.5, theta=0.04, xi=0.3,
                              rho=-0.7, v0=0.04),
                s0=100, paths=10, steps=32, sobol=True, bridge=True)

    def test_plain_sobol_unchanged_by_bridge_feature(self):
        # bridge=False must reproduce the pre-Phase-3 plain-Sobol stream.
        from mc_payoffs import EuropeanPricer
        a = self._price(EuropeanPricer, 2_048, 16, 5, "sobol")
        cfg = _cfg(paths=2_048, horizon=16, seed=5, chunk_size=2_048,
                   variance_reduction="sobol", s0=100.0, mu=0.05, sigma=0.25)
        # same shocks as the flagged v2 engine consumes -> same terminal set
        import mc_core as _mc
        legacy = simulate(cfg).final_values
        gen = mc_engine.generator_from_config(cfg)
        p = TerminalValuePricer(cfg.s0, cfg.cost)
        gen.run([p])
        assert np.array_equal(np.sort(legacy), np.sort(p.values()))
