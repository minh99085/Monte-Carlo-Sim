"""Phase 4 acceptance tests: all seven remaining models on the v2 engine.

The hard bar (same as Phase 1 set for GBM/Heston): flagged v2 runs must be
**byte-identical** to the legacy closures — np.array_equal on final-value
arrays — across plain / antithetic / Sobol / stress-crash variants, because
every process replicates the legacy RNG call order exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import mc_core
import mc_engine
import mc_options
from mc_core import SimulationConfig, simulate
from mc_engine import PathGenerator, TerminalValuePricer, generator_from_config
from mc_lsm import longstaff_schwartz_price
from mc_payoffs import BarrierPricer, EuropeanPricer
from mc_processes import (
    BlockBootstrapProcess,
    GARCHProcess,
    HistoricalBootstrapProcess,
    KouJumpProcess,
    MertonJumpProcess,
    RegimeSwitchingProcess,
    StudentTProcess,
    extended_process_from_config,
)

PORTED = [
    mc_core.MODEL_STUDENT_T,
    mc_core.MODEL_HIST_BOOTSTRAP,
    mc_core.MODEL_BLOCK_BOOTSTRAP,
    mc_core.MODEL_MERTON,
    mc_core.MODEL_KOU,
    mc_core.MODEL_GARCH,
    mc_core.MODEL_REGIME,
]

_HIST = np.random.default_rng(1).normal(0.0004, 0.012, size=750)


def _cfg(model, **kw) -> SimulationConfig:
    base = dict(
        ticker="TEST", s0=100.0, paths=3_000, horizon=25, mu=0.08,
        sigma=0.25, seed=42, chunk_size=800, sample_paths=3, model=model,
    )
    if model in mc_core.BOOTSTRAP_MODELS:
        base["historical_returns"] = _HIST
    base.update(kw)
    return SimulationConfig(**base)


def _final_values(cfg) -> np.ndarray:
    return simulate(cfg).final_values


# ---------------------------------------------------------------------------
# 1. Bit-identical equivalence, all models x all variants
# ---------------------------------------------------------------------------


class TestPortedModelEquivalence:
    @pytest.mark.parametrize("model", PORTED)
    def test_exact_match_plain(self, model):
        assert np.array_equal(_final_values(_cfg(model, engine="legacy")),
                              _final_values(_cfg(model, engine="v2")))

    @pytest.mark.parametrize("model", PORTED)
    def test_exact_match_antithetic(self, model):
        kw = dict(variance_reduction="antithetic")
        assert np.array_equal(_final_values(_cfg(model, engine="legacy", **kw)),
                              _final_values(_cfg(model, engine="v2", **kw)))

    @pytest.mark.skipif(not mc_core.sobol_available(),
                        reason="SciPy Sobol not available")
    @pytest.mark.parametrize("model", PORTED)
    def test_exact_match_sobol(self, model):
        kw = dict(variance_reduction="sobol")
        assert np.array_equal(_final_values(_cfg(model, engine="legacy", **kw)),
                              _final_values(_cfg(model, engine="v2", **kw)))

    @pytest.mark.parametrize("model", PORTED)
    def test_exact_match_stress_crash(self, model):
        kw = dict(stress_enabled=True, stress_crash_pct=0.08,
                  stress_vol_multiplier=1.4, stress_drift_haircut=0.3)
        assert np.array_equal(_final_values(_cfg(model, engine="legacy", **kw)),
                              _final_values(_cfg(model, engine="v2", **kw)))

    @pytest.mark.parametrize("model", PORTED)
    def test_stats_equal_and_engine_key_semantics(self, model):
        r_legacy = simulate(_cfg(model, engine="legacy"))
        r_v2 = simulate(_cfg(model, engine="v2"))
        assert r_legacy.stats["engine"] == "legacy"   # always recorded (Phase 5)
        assert r_v2.stats["engine"] == "v2"
        for key in ("expected_value", "median_value", "prob_profit",
                    "std_value", "prob_ruin", "mean_max_drawdown",
                    "mean_drawdown_duration"):
            assert r_legacy.stats[key] == r_v2.stats[key], (model, key)
        assert (r_legacy.stats["var"]["99"]["value"]
                == r_v2.stats["var"]["99"]["value"]), model

    @pytest.mark.parametrize("model", PORTED)
    def test_all_in_v2_supported(self, model):
        assert model in mc_engine.V2_SUPPORTED_MODELS

    @pytest.mark.parametrize("model", PORTED)
    def test_standalone_generator_reproduces_legacy(self, model):
        cfg = _cfg(model, cost=0.001)
        legacy = _final_values(cfg)
        gen = generator_from_config(cfg)
        pricer = TerminalValuePricer(cfg.s0, cfg.cost)
        mem = gen.run([pricer])
        assert np.array_equal(legacy, pricer.values())
        assert mem.is_chunk_safe


# ---------------------------------------------------------------------------
# 2. Process contracts
# ---------------------------------------------------------------------------


class TestProcessContracts:
    def test_gauss_modes_match_legacy_shock_channels(self):
        assert StudentTProcess(0.05, 0.2, 1 / 252, 5.0).gauss_mode == "none"
        assert MertonJumpProcess(0.05, 0.2, 1 / 252, intensity=1.0,
                                 jump_mean=-0.02,
                                 jump_vol=0.05).gauss_mode == "none"
        assert KouJumpProcess(0.05, 0.2, 1 / 252, intensity=1.0, p_up=0.4,
                              eta_up=25.0,
                              eta_down=15.0).gauss_mode == "plain"
        assert GARCHProcess(0.05, 0.2, 1 / 252, alpha=0.08,
                            beta=0.90).gauss_mode == "full"
        proc = extended_process_from_config(_cfg(mc_core.MODEL_REGIME).validate())
        assert proc.gauss_mode == "none"

    def test_bootstrap_has_no_gaussian_decomposition(self):
        p = HistoricalBootstrapProcess(_HIST, 0.0002)
        with pytest.raises(NotImplementedError):
            p.drift({}, 4)
        with pytest.raises(NotImplementedError):
            p.diffusion({}, 4)

    def test_block_bootstrap_state_is_chunk_local(self):
        p = BlockBootstrapProcess(_HIST, 0.0002, block_length=5)
        st = p.init_state(7)
        assert st["cur"].shape == st["rem"].shape == (7,)
        rng = np.random.default_rng(0)
        p.evolve(rng, st, None, 7)
        assert np.all(st["rem"] == 4)  # first step opened fresh blocks

    def test_bridge_rejected_for_non_gaussian_models(self):
        p = StudentTProcess(0.05, 0.2, 1 / 32, 5.0)
        with pytest.raises(ValueError, match="Gaussian-shock"):
            PathGenerator(p, s0=100, paths=10, steps=32, sobol=True,
                          bridge=True)

    def test_unknown_model_returns_none(self):
        cfg = _cfg(mc_core.MODEL_GBM).validate()
        assert extended_process_from_config(cfg) is None  # handled upstream


# ---------------------------------------------------------------------------
# 3. Option-pricing smoke under jumpy/regime dynamics
# ---------------------------------------------------------------------------


class TestOptionsUnderPortedModels:
    S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.20, 1.0

    def _euro(self, process, paths=30_000, seed=42, steps=64):
        gen = PathGenerator(process, s0=self.S0, paths=paths, steps=steps,
                            chunk_size=paths, seed=seed)
        p = EuropeanPricer(strike=self.K, maturity=self.T, r=self.R)
        gen.run([p])
        return p

    def test_jumpier_model_raises_option_value_at_equal_diffusion_vol(self):
        dt = self.T / 64
        gbm = self._euro(mc_engine.GBMProcess(self.R, self.SIG, dt))
        merton = self._euro(MertonJumpProcess(
            self.R, self.SIG, dt, intensity=2.0, jump_mean=-0.05,
            jump_vol=0.10))
        # Jumps add variance on top of the same diffusion vol -> pricier.
        assert merton.price() > gbm.price() + 2 * math.hypot(
            gbm.std_error(), merton.std_error())

    @pytest.mark.parametrize("model_flag", ["merton", "regime"])
    def test_cli_european_barrier_american(self, model_flag, capsys):
        common = ["TEST", "--strike", "100", "--maturity", "0.5",
                  "--r", "0.05", "--s0", "100", "--sigma", "0.2",
                  "--paths", "8000", "--seed", "3", "--model", model_flag]
        for extra in (["--option", "european"],
                      ["--option", "barrier", "--barrier", "80",
                       "--barrier-dir", "down", "--barrier-knock", "out"],
                      ["--option", "american", "--put"]):
            rc = mc_options.main(common + extra)
            out = capsys.readouterr().out
            assert rc == 0
            if "american" in extra:
                token = [l for l in out.splitlines() if "LSM price" in l][0]
                price = float(token.split()[2])
            else:
                token = [l for l in out.splitlines() if "PRICE =" in l][0]
                price = float(token.split("=")[1].split()[0])
            assert math.isfinite(price) and price >= 0.0, (model_flag, extra)

    def test_american_put_under_merton_at_least_gbm_european(self):
        res = longstaff_schwartz_price(
            lambda dt: MertonJumpProcess(self.R, self.SIG, dt, intensity=1.0,
                                         jump_mean=-0.05, jump_vol=0.10),
            s0=36.0, strike=40.0, maturity=1.0, r=0.06, call=False,
            paths=10_000, exercise_dates=50, degree=3, seed=11)
        assert math.isfinite(res.price)
        # American with extra jump risk > deep-ITM intrinsic-ish floor
        assert res.price > 4.0

    def test_barrier_in_out_parity_under_regime(self):
        preset = mc_core.REGIME_PRESETS["stock"]
        proc = RegimeSwitchingProcess(
            self.R, self.SIG, self.T / 64,
            mu_factors=preset["mu_factors"],
            sigma_factors=preset["sigma_factors"],
            transition=preset["transition"])
        gen = PathGenerator(proc, s0=self.S0, paths=8_000, steps=64,
                            chunk_size=8_000, seed=5)
        out = BarrierPricer(strike=self.K, barrier=85.0, maturity=self.T,
                            r=self.R, direction="down", knock="out")
        into = BarrierPricer(strike=self.K, barrier=85.0, maturity=self.T,
                             r=self.R, direction="down", knock="in")
        vanilla = EuropeanPricer(strike=self.K, maturity=self.T, r=self.R)
        gen.run([out, into, vanilla])
        np.testing.assert_allclose(out.values() + into.values(),
                                   vanilla.values(), atol=1e-12)
