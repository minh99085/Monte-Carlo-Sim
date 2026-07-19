"""Phase 6 acceptance tests: streaming statistics, fused kernels,
multi-asset process.

Hard bars:
* multi-asset generator reproduces ``simulate_portfolio`` **bit-identically**
  (same inputs, same seed, same chunking);
* the numpy fused kernel is **bit-identical** to the streaming engine;
* streaming estimators match exact statistics within their stated errors.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import mc_kernels
from mc_core import simulate_portfolio
from mc_engine import GBMProcess, HestonProcess, PathGenerator, TerminalValuePricer
from mc_multiasset import (
    AssetTerminalPricer,
    BasketTerminalPricer,
    MultiAssetGBMProcess,
    MultiAssetPathGenerator,
    multiasset_from_returns,
)
from mc_stats import (
    P2Quantile,
    ReservoirSample,
    StreamingRiskStats,
    StreamingStatsPricer,
)


# ---------------------------------------------------------------------------
# A. Streaming statistics
# ---------------------------------------------------------------------------


class TestP2Quantile:
    @pytest.mark.parametrize("p", [0.05, 0.5, 0.95])
    def test_matches_numpy_percentile(self, p):
        data = np.random.default_rng(7).normal(110.0, 20.0, 100_000)
        est = P2Quantile(p)
        est.add(data)
        exact = float(np.percentile(data, 100 * p))
        # P² error scales with density; ~0.1% of sigma at these sizes.
        assert abs(est.value - exact) < 0.2, (p, est.value, exact)

    def test_small_stream_exact(self):
        est = P2Quantile(0.5)
        est.add([3.0, 1.0, 2.0])
        assert est.value == 2.0

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            P2Quantile(1.5)


class TestReservoir:
    def test_keeps_everything_when_under_capacity(self):
        r = ReservoirSample(1_000, seed=0)
        data = np.arange(500.0)
        r.add(data)
        np.testing.assert_array_equal(np.sort(r.sample()), data)

    def test_unbiased_quantiles_over_capacity(self):
        rng = np.random.default_rng(3)
        data = rng.normal(0.0, 1.0, 400_000)
        r = ReservoirSample(40_000, seed=1)
        for chunk in np.array_split(data, 17):
            r.add(chunk)
        assert r.count == data.size
        for q in (1, 50, 99):
            exact = np.percentile(data, q)
            # sampling error ~ sqrt(p(1-p)/k)/f(q); generous 5-sigma band
            assert abs(r.percentile(q) - exact) < 0.12, q

    def test_deterministic_for_seed(self):
        a, b = ReservoirSample(100, seed=5), ReservoirSample(100, seed=5)
        data = np.random.default_rng(0).normal(size=10_000)
        a.add(data)
        b.add(data)
        np.testing.assert_array_equal(a.sample(), b.sample())


class TestStreamingRiskStats:
    def test_matches_exact_statistics(self):
        data = np.random.default_rng(7).normal(110.0, 20.0, 500_000)
        srs = StreamingRiskStats(100.0, reservoir=50_000, seed=1)
        for chunk in np.array_split(data, 23):
            srs.add(chunk)
        s = srs.summary()
        assert s["paths"] == data.size
        assert s["expected_value"] == pytest.approx(float(data.mean()),
                                                    rel=1e-12)
        assert s["std_value"] == pytest.approx(float(data.std(ddof=1)),
                                               rel=1e-10)
        assert s["min_value"] == data.min() and s["max_value"] == data.max()
        assert abs(s["median_value_p2"] - np.median(data)) < 0.2
        assert abs(s["percentiles"]["5"] - np.percentile(data, 5)) < 0.6
        exact_var99 = -float(np.percentile(data - 100.0, 1))
        assert abs(s["var"]["99"]["value"] - exact_var99) < 1.0
        # ES must exceed VaR at the same level
        assert s["expected_shortfall"]["99"]["value"] > s["var"]["99"]["value"]

    def test_pricer_streams_without_storing(self):
        gen = PathGenerator(GBMProcess(0.05, 0.25, 1 / 252), s0=100.0,
                            paths=30_000, steps=64, chunk_size=5_000, seed=9)
        exact = TerminalValuePricer(100.0, 0.001)
        stream = StreamingStatsPricer(100.0, 0.001, reservoir=20_000, seed=2)
        gen.run([exact, stream])
        v = exact.values()
        s = stream.stats.summary()
        assert s["paths"] == v.size
        assert s["expected_value"] == pytest.approx(float(v.mean()), rel=1e-12)
        assert abs(s["percentiles"]["50"] - np.median(v)) < 1.0
        with pytest.raises(NotImplementedError):
            stream.values()


# ---------------------------------------------------------------------------
# B. Fused kernels
# ---------------------------------------------------------------------------


class TestFusedKernel:
    def _run(self, kernel, **kw):
        args = dict(paths=20_000, steps=64, chunk_size=5_000, seed=42)
        args.update(kw)
        gen = PathGenerator(GBMProcess(0.08, 0.25, 1 / 252), s0=100.0,
                            kernel=kernel, **args)
        p = TerminalValuePricer(100.0, 0.001)
        gen.run([p])
        return p.values(), gen

    def test_numpy_backend_bit_identical(self, monkeypatch):
        monkeypatch.setattr(mc_kernels, "FUSED_BACKEND", "numpy")
        base, _ = self._run(False)
        fast, gen = self._run(True)
        assert gen.kernel_used
        assert np.array_equal(base, fast)

    def test_kernel_ineligible_falls_back(self):
        # Sobol path: kernel must not engage.
        import mc_core
        if not mc_core.sobol_available():
            pytest.skip("SciPy Sobol not available")
        _, gen = self._run(True, sobol=True)
        assert not gen.kernel_used
        # Non-GBM process: ineligible.
        g2 = PathGenerator(HestonProcess(0.05, 1 / 252, kappa=1.5, theta=0.04,
                                         xi=0.3, rho=-0.7, v0=0.04),
                           s0=100.0, paths=2_000, steps=16, kernel=True,
                           chunk_size=2_000, seed=1)
        g2.run([TerminalValuePricer(100.0, 0.0)])
        assert not g2.kernel_used
        # Observer present (needs per-step hooks): ineligible.
        from mc_core import DrawdownObserver
        g3 = PathGenerator(GBMProcess(0.08, 0.25, 1 / 252), s0=100.0,
                           paths=2_000, steps=16, kernel=True,
                           chunk_size=2_000, seed=1)
        g3.run([TerminalValuePricer(100.0, 0.0),
                DrawdownObserver(50.0, 0.5)])
        assert not g3.kernel_used

    def test_antithetic_supported_and_identical(self, monkeypatch):
        monkeypatch.setattr(mc_kernels, "FUSED_BACKEND", "numpy")
        base, _ = self._run(False, antithetic=True)
        fast, gen = self._run(True, antithetic=True)
        assert gen.kernel_used
        assert np.array_equal(base, fast)

    @pytest.mark.skipif(not mc_kernels.NUMBA_KERNEL_AVAILABLE,
                        reason="numba not installed")
    def test_numba_backend_close_to_streaming(self):
        base, _ = self._run(False)
        prices = np.full(20_000, 100.0)
        # direct kernel call on the same shocks, numba backend
        rng = np.random.default_rng(42)
        from mc_core import _draw_gauss
        z = np.empty((64, 5_000))
        vals = []
        for _ in range(4):  # four chunks of 5k, same chunking as _run
            for i in range(64):
                z[i] = _draw_gauss(rng, 5_000, False)
            p = np.full(5_000, 100.0)
            mc_kernels.fused_gbm_chunk(
                p, (0.08 - 0.5 * 0.25 ** 2) / 252,
                0.25 * math.sqrt(1 / 252), z, backend="numba")
            vals.append(p * (1.0 - 0.001) - 100.0 * 0.001)
        got = np.concatenate(vals)
        assert np.allclose(base, got, rtol=1e-12)

    def test_bad_backend_rejected(self):
        with pytest.raises(ValueError, match="backend"):
            mc_kernels.fused_gbm_chunk(np.ones(3), 0.0, 0.1,
                                       np.zeros((2, 3)), backend="gpu")


# ---------------------------------------------------------------------------
# C. Multi-asset process
# ---------------------------------------------------------------------------


def _returns(seed=5, k=3, n=600):
    rng = np.random.default_rng(seed)
    vols = [0.012, 0.020, 0.016][:k]
    return {f"T{j}": rng.normal(3e-4 / (j + 1), vols[j], n) for j in range(k)}


class TestMultiAsset:
    def test_bit_identical_to_simulate_portfolio(self):
        rets = _returns()
        legacy = simulate_portfolio(rets, paths=8_000, horizon=60, seed=11,
                                    chunk_size=3_000)
        proc, tickers, meta = multiasset_from_returns(rets)
        s0_vec = np.full(len(tickers), 100.0)
        gen = MultiAssetPathGenerator(proc, s0_vec=s0_vec, paths=8_000,
                                      steps=60, chunk_size=3_000, seed=11)
        basket = BasketTerminalPricer(
            np.full(len(tickers), 1.0 / len(tickers)), s0_vec)
        assets = AssetTerminalPricer(s0_vec)
        gen.run([basket, assets])
        assert np.array_equal(legacy["portfolio_values"], basket.values())
        for j, t in enumerate(tickers):
            assert assets.mean_gross_return()[j] == pytest.approx(
                legacy["per_asset"][t]["mean_gross_return"], rel=1e-12)
        assert gen.is_chunk_safe

    def test_drift_modes_match_legacy(self):
        rets = _returns(seed=8)
        for mode in ("Half historical drift", "Zero drift"):
            legacy = simulate_portfolio(rets, paths=2_000, horizon=20,
                                        seed=3, drift_mode=mode)
            proc, tickers, _ = multiasset_from_returns(rets, drift_mode=mode)
            s0_vec = np.full(len(tickers), 100.0)
            gen = MultiAssetPathGenerator(proc, s0_vec=s0_vec, paths=2_000,
                                          steps=20, chunk_size=50_000, seed=3)
            basket = BasketTerminalPricer(
                np.full(len(tickers), 1 / len(tickers)), s0_vec)
            gen.run([basket])
            assert np.array_equal(legacy["portfolio_values"], basket.values())

    def test_single_asset_reduces_to_gbm_semantics(self):
        # k=1: correlated shock == plain shock; expected terminal mean is
        # s0*exp(mu*T) within MC error.
        mus, sigmas = np.array([0.08]), np.array([0.25])
        proc = MultiAssetGBMProcess(mus, sigmas, np.eye(1), 1 / 252)
        s0_vec = np.array([100.0])
        gen = MultiAssetPathGenerator(proc, s0_vec=s0_vec, paths=40_000,
                                      steps=126, chunk_size=10_000, seed=2)
        basket = BasketTerminalPricer(np.array([1.0]), s0_vec)
        gen.run([basket])
        v = basket.values() * 100.0  # relative -> price
        expect = 100.0 * math.exp(0.08 * 0.5)
        se = v.std(ddof=1) / math.sqrt(v.size)
        assert abs(v.mean() - expect) < 4 * se

    def test_correlation_is_realized(self):
        # Two assets, rho=0.9: log-return correlation must come out high.
        corr = np.array([[1.0, 0.9], [0.9, 1.0]])
        chol = np.linalg.cholesky(corr)
        proc = MultiAssetGBMProcess([0.0, 0.0], [0.2, 0.2], chol, 1 / 252)
        s0_vec = np.array([100.0, 100.0])

        class LogRetGrab(BasketTerminalPricer):
            def end_chunk(self, prices):
                self._chunks.append(np.log(prices / self.s0_vec))

        grab = LogRetGrab(np.array([1.0, 0.0]), s0_vec)
        gen = MultiAssetPathGenerator(proc, s0_vec=s0_vec, paths=20_000,
                                      steps=64, chunk_size=20_000, seed=4)
        gen.run([grab])
        lr = np.vstack(grab._chunks)
        got = np.corrcoef(lr.T)[0, 1]
        assert abs(got - 0.9) < 0.02

    def test_dimension_validation(self):
        with pytest.raises(ValueError):
            MultiAssetGBMProcess([0.1, 0.1], [0.2], np.eye(2), 1 / 252)
        proc = MultiAssetGBMProcess([0.1], [0.2], np.eye(1), 1 / 252)
        with pytest.raises(ValueError):
            MultiAssetPathGenerator(proc, s0_vec=np.array([1.0, 2.0]),
                                    paths=10, steps=5)
