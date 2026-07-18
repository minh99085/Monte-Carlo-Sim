"""Phase 2 acceptance tests: streaming option pricers vs closed forms.

All prices are computed under risk-neutral GBM (mu = r) so the closed-form
oracles apply. Seeds are fixed; tolerances are stated in MC standard
errors of the estimate itself.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import mc_core
from mc_engine import GBMProcess, HestonProcess, PathGenerator
from mc_payoffs import (
    AsianArithmeticPricer,
    AsianGeometricPricer,
    BarrierPricer,
    EuropeanPricer,
    LookbackPricer,
    bgk_adjusted_barrier,
    black_scholes_price,
    check_risk_neutral_drift,
    down_and_in_call,
    down_and_out_call,
    geometric_asian_call_discrete,
    norm_cdf,
)

S0, K, R, SIGMA, T = 100.0, 100.0, 0.05, 0.25, 1.0
STEPS = 252
DT = T / STEPS


def run_pricers(pricers, *, paths=60_000, seed=42, sigma=SIGMA, steps=STEPS,
                chunk_size=20_000, antithetic=False, sobol=False, s0=S0):
    process = GBMProcess(R, sigma, T / steps)  # risk-neutral: mu = r
    gen = PathGenerator(process, s0=s0, paths=paths, steps=steps,
                        chunk_size=chunk_size, seed=seed,
                        antithetic=antithetic, sobol=sobol)
    gen.run(pricers)
    return pricers


class TestEuropeanVsBlackScholes:
    @pytest.mark.parametrize("call", [True, False])
    def test_matches_bs_within_3_se(self, call):
        p, = run_pricers([EuropeanPricer(strike=K, maturity=T, r=R, call=call)])
        bs = black_scholes_price(S0, K, R, SIGMA, T, call=call)
        se = p.std_error()
        assert se < 0.15
        assert abs(p.price() - bs) < 3 * se, (p.price(), bs, se)

    def test_put_call_parity(self):
        c, p = run_pricers([
            EuropeanPricer(strike=K, maturity=T, r=R, call=True),
            EuropeanPricer(strike=K, maturity=T, r=R, call=False),
        ])
        # Same paths -> parity holds almost exactly (only payoff kinks differ)
        parity = c.price() - p.price() - (S0 - K * math.exp(-R * T))
        assert abs(parity) < 3 * c.std_error()

    def test_heston_reduces_to_bs_when_vol_of_vol_zero(self):
        # xi=0, v0=theta=sigma^2 -> variance pinned at sigma^2 -> GBM prices
        process = HestonProcess(R, DT, kappa=1.5, theta=SIGMA ** 2, xi=0.0,
                                rho=0.0, v0=SIGMA ** 2)
        gen = PathGenerator(process, s0=S0, paths=60_000, steps=STEPS,
                            chunk_size=20_000, seed=9)
        pricer = EuropeanPricer(strike=K, maturity=T, r=R, call=True)
        gen.run([pricer])
        bs = black_scholes_price(S0, K, R, SIGMA, T, call=True)
        assert abs(pricer.price() - bs) < 3 * pricer.std_error()


class TestBarrier:
    BARRIER = 85.0

    def _prices(self, **kw):
        out = BarrierPricer(strike=K, barrier=self.BARRIER, maturity=T, r=R,
                            call=True, direction="down", knock="out")
        into = BarrierPricer(strike=K, barrier=self.BARRIER, maturity=T, r=R,
                             call=True, direction="down", knock="in")
        vanilla = EuropeanPricer(strike=K, maturity=T, r=R, call=True)
        run_pricers([out, into, vanilla], **kw)
        return out, into, vanilla

    def test_in_out_parity_exact_per_path(self):
        out, into, vanilla = self._prices()
        np.testing.assert_allclose(out.values() + into.values(),
                                   vanilla.values(), rtol=0, atol=1e-12)

    def test_down_and_out_matches_bgk_adjusted_closed_form(self):
        out, _, _ = self._prices(paths=100_000)
        h_adj = bgk_adjusted_barrier(self.BARRIER, S0, SIGMA, DT)
        cf_discrete = down_and_out_call(S0, K, h_adj, R, SIGMA, T)
        se = out.std_error()
        assert abs(out.price() - cf_discrete) < 4 * se, \
            (out.price(), cf_discrete, se)

    def test_discrete_monitoring_bias_direction(self):
        # Discrete monitoring misses intra-step crossings: knock-out prices
        # ABOVE the continuous formula, knock-in BELOW.
        out, into, _ = self._prices(paths=100_000)
        cf_out_cont = down_and_out_call(S0, K, self.BARRIER, R, SIGMA, T)
        cf_in_cont = down_and_in_call(S0, K, self.BARRIER, R, SIGMA, T)
        assert out.price() + 2 * out.std_error() > cf_out_cont
        assert into.price() - 2 * into.std_error() < cf_in_cont

    def test_knocked_at_inception(self):
        # Start below a down barrier -> out option worthless, in == vanilla
        out = BarrierPricer(strike=K, barrier=S0 + 5, maturity=T, r=R,
                            call=True, direction="down", knock="out")
        run_pricers([out], paths=2_000, steps=10)
        assert out.price() == 0.0


class TestAsian:
    def test_geometric_matches_discrete_closed_form(self):
        p, = run_pricers(
            [AsianGeometricPricer(strike=K, maturity=T, r=R, call=True)],
            paths=100_000)
        cf = geometric_asian_call_discrete(S0, K, R, SIGMA, T, STEPS, call=True)
        se = p.std_error()
        assert abs(p.price() - cf) < 3 * se, (p.price(), cf, se)

    def test_arithmetic_above_geometric_control(self):
        # AM-GM: arithmetic mean >= geometric mean per path, so the
        # arithmetic Asian call dominates the geometric one path by path.
        arith = AsianArithmeticPricer(strike=K, maturity=T, r=R, call=True)
        geo = AsianGeometricPricer(strike=K, maturity=T, r=R, call=True)
        run_pricers([arith, geo])
        assert np.all(arith.values() >= geo.values() - 1e-12)
        assert arith.price() > geo.price()
        # ... but not by much (sanity band vs the exact geometric price)
        cf = geometric_asian_call_discrete(S0, K, R, SIGMA, T, STEPS)
        assert arith.price() - cf < 0.5

    def test_asian_below_european(self):
        # Averaging reduces effective volatility.
        asian = AsianArithmeticPricer(strike=K, maturity=T, r=R, call=True)
        euro = EuropeanPricer(strike=K, maturity=T, r=R, call=True)
        run_pricers([asian, euro])
        assert asian.price() < euro.price()


class TestLookback:
    def test_floating_dominates_atm_european(self):
        # Floating lookback call pays S_T - min >= S_T - S0 ~ ATM payoff.
        look = LookbackPricer(maturity=T, r=R, call=True, kind="floating")
        euro = EuropeanPricer(strike=S0, maturity=T, r=R, call=True)
        run_pricers([look, euro])
        assert np.all(look.values() >= euro.values() - 1e-12)
        assert look.price() > euro.price()

    def test_fixed_call_uses_running_max(self):
        look = LookbackPricer(maturity=T, r=R, call=True, kind="fixed",
                              strike=K)
        euro = EuropeanPricer(strike=K, maturity=T, r=R, call=True)
        run_pricers([look, euro])
        assert np.all(look.values() >= euro.values() - 1e-12)

    def test_payoffs_nonnegative(self):
        for kind in ("fixed", "floating"):
            for call in (True, False):
                p = LookbackPricer(maturity=T, r=R, call=call, kind=kind,
                                   strike=K)
                run_pricers([p], paths=5_000, steps=50)
                assert np.all(p.values() >= 0.0)


class TestVarianceReduction:
    def test_antithetic_reduces_std_error(self):
        paths = 40_000
        plain = EuropeanPricer(strike=K, maturity=T, r=R, call=True)
        run_pricers([plain], paths=paths, chunk_size=paths, seed=5)
        anti = EuropeanPricer(strike=K, maturity=T, r=R, call=True)
        run_pricers([anti], paths=paths, chunk_size=paths, seed=5,
                    antithetic=True)
        # Antithetic pairs are (i, i + paths/2) within the single chunk;
        # the proper se averages each pair first.
        v = anti.values()
        half = paths // 2
        pair_means = 0.5 * (v[:half] + v[half:])
        se_anti = float(np.std(pair_means, ddof=1) / math.sqrt(half))
        se_plain = plain.std_error()
        assert se_anti < 0.8 * se_plain, (se_anti, se_plain)
        # and the price is still right
        bs = black_scholes_price(S0, K, R, SIGMA, T, call=True)
        assert abs(anti.price() - bs) < 4 * se_plain

    @pytest.mark.skipif(not mc_core.sobol_available(),
                        reason="SciPy Sobol not available")
    def test_sobol_beats_plain_mc_on_average(self):
        bs = black_scholes_price(S0, K, R, SIGMA, T, call=True)

        def err(seed, sobol):
            p = EuropeanPricer(strike=K, maturity=T, r=R, call=True)
            run_pricers([p], paths=8_192, chunk_size=8_192, seed=seed,
                        sobol=sobol, steps=32)
            # closed form at the coarser grid is the same BS price
            return abs(p.price() - bs)

        seeds = [1, 2, 3, 4, 5]
        mean_plain = np.mean([err(s, False) for s in seeds])
        mean_sobol = np.mean([err(s, True) for s in seeds])
        assert mean_sobol < mean_plain, (mean_sobol, mean_plain)


class TestChunkInvariance:
    """Chunking must not change the computed answer.

    Bit-exact invariance across chunk sizes is impossible with a shared
    sequential PRNG stream (the path <-> draw mapping changes — the legacy
    engine behaves the same way), so this is asserted three ways:
    1. exactly, for the pricer state machines on identical paths fed with
       different chunk splits;
    2. exactly, for a deterministic (zero-vol) process;
    3. statistically, for the full random pipeline.
    """

    def _feed(self, pricer, matrix, chunk):
        for lo in range(0, matrix.shape[0], chunk):
            block = matrix[lo:lo + chunk]
            pricer.begin_chunk(block[:, 0].copy())
            for i in range(1, block.shape[1]):
                pricer.observe(i, block[:, i])
            pricer.end_chunk(block[:, -1])
        return pricer.values()

    def test_pricer_state_machines_chunk_invariant_exactly(self):
        rng = np.random.default_rng(0)
        n, steps = 4_000, 60
        rets = rng.normal(0.0, 0.02, size=(n, steps))
        matrix = S0 * np.exp(np.cumsum(np.c_[np.zeros(n), rets], axis=1))
        mk = lambda: [
            EuropeanPricer(strike=K, maturity=T, r=R),
            AsianArithmeticPricer(strike=K, maturity=T, r=R),
            BarrierPricer(strike=K, barrier=85.0, maturity=T, r=R),
            LookbackPricer(maturity=T, r=R, kind="floating"),
        ]
        small = [self._feed(p, matrix, 1_000) for p in mk()]
        big = [self._feed(p, matrix, 4_000) for p in mk()]
        for a, b in zip(small, big):
            np.testing.assert_array_equal(a, b)

    def test_zero_vol_process_chunk_invariant_exactly(self):
        for chunk in (1_000, 50_000):
            p, = run_pricers(
                [AsianArithmeticPricer(strike=90.0, maturity=T, r=R)],
                paths=5_000, sigma=0.0, chunk_size=chunk, steps=50)
            # deterministic forward path -> closed-form average
            grid = S0 * np.exp(R * T * np.arange(1, 51) / 50)
            expect = math.exp(-R * T) * max(float(grid.mean()) - 90.0, 0.0)
            assert p.price() == pytest.approx(expect, rel=1e-12)

    def test_random_pipeline_statistically_chunk_invariant(self):
        p_small, = run_pricers([EuropeanPricer(strike=K, maturity=T, r=R)],
                               paths=50_000, chunk_size=1_000)
        p_big, = run_pricers([EuropeanPricer(strike=K, maturity=T, r=R)],
                             paths=50_000, chunk_size=50_000)
        se = math.hypot(p_small.std_error(), p_big.std_error())
        assert abs(p_small.price() - p_big.price()) < 4 * se


class TestRiskNeutralGuard:
    def test_warns_on_historical_drift(self):
        with pytest.warns(UserWarning, match="NOT an arbitrage-free"):
            assert check_risk_neutral_drift(0.08, 0.05) is False
        assert check_risk_neutral_drift(0.05, 0.05) is True

    def test_norm_cdf_sanity(self):
        assert norm_cdf(0.0) == pytest.approx(0.5)
        assert norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)


class TestOptionsCLI:
    def test_european_cli_offline(self, capsys):
        import mc_options
        rc = mc_options.main([
            "TEST", "--option", "european", "--strike", "100",
            "--maturity", "1.0", "--r", "0.05", "--s0", "100",
            "--sigma", "0.25", "--paths", "20000", "--seed", "3",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "PRICE =" in out
        assert "risk-neutral" in out

    def test_american_cli_offline(self, capsys):
        import mc_options
        rc = mc_options.main([
            "TEST", "--option", "american", "--put", "--strike", "40",
            "--maturity", "1.0", "--r", "0.06", "--s0", "36",
            "--sigma", "0.2", "--paths", "10000", "--seed", "3",
        ])
        assert rc == 0
        out = capsys.readouterr().out
        assert "LSM price" in out

    def test_fallback_market_data_refused(self, monkeypatch, capsys):
        import mc_options
        from mc_core import MarketParameters

        monkeypatch.setattr(
            mc_options, "estimate_parameters_from_history",
            lambda t, **k: MarketParameters(s0=100, mu=0.08, sigma=0.2,
                                            source="fallback"))
        rc = mc_options.main([
            "TEST", "--option", "european", "--strike", "100",
            "--maturity", "1.0", "--r", "0.05",
        ])
        assert rc == 2
        assert "fallback" in capsys.readouterr().err
