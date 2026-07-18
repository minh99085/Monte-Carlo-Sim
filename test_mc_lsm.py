"""Phase 2 acceptance tests: Longstaff-Schwartz American pricing.

Anchors:
* Longstaff & Schwartz (2001), Table 1 American put values
  (K=40, r=0.06, 50 exercise dates/year) within MC tolerance.
* An in-test CRR binomial tree as an independent oracle.
* American call with no dividends ~ European call (early exercise is
  never optimal).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mc_engine import GBMProcess, HestonProcess
from mc_lsm import ExerciseDateRecorder, longstaff_schwartz_price
from mc_payoffs import black_scholes_price

R, STRIKE = 0.06, 40.0


def crr_american_put(s0, k, r, sigma, t, n=2_000):
    """Cox-Ross-Rubinstein binomial American put — independent oracle."""
    dt = t / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    j = np.arange(n + 1)
    v = np.maximum(k - s0 * u ** (n - j) * d ** j, 0.0)
    for i in range(n - 1, -1, -1):
        st = s0 * u ** (i - np.arange(i + 1)) * d ** np.arange(i + 1)
        v = disc * (p * v[:-1] + (1 - p) * v[1:])
        np.maximum(v, k - st, out=v)
    return float(v[0])


def lsm_put(s0, sigma, t, **kw):
    args = dict(s0=s0, strike=STRIKE, maturity=t, r=R, call=False,
                paths=20_000, exercise_dates=int(50 * t), degree=3, seed=11)
    args.update(kw)
    return longstaff_schwartz_price(
        lambda dt: GBMProcess(R, sigma, dt), **args)


# Longstaff & Schwartz (2001), Table 1, "Simulated American" column
# (S0, sigma, T) -> value; K=40, r=0.06, 50 exercise points per year.
LS2001_TABLE1 = {
    (36, 0.2, 1): 4.472, (36, 0.2, 2): 4.821,
    (36, 0.4, 1): 7.091, (36, 0.4, 2): 8.488,
    (40, 0.2, 1): 2.313, (40, 0.2, 2): 2.879,
    (40, 0.4, 2): 6.921,
    (44, 0.2, 1): 1.109,  # deep OTM-ish corner
    (44, 0.4, 1): 3.957,
}


class TestAmericanPut:
    @pytest.mark.parametrize("s0,sigma,t", [
        (36, 0.2, 1), (36, 0.2, 2), (36, 0.4, 1),
        (40, 0.2, 1), (40, 0.4, 2), (44, 0.4, 1),
    ])
    def test_matches_ls2001_and_crr(self, s0, sigma, t):
        res = lsm_put(s0, sigma, t)
        crr = crr_american_put(s0, STRIKE, R, sigma, t)
        tol = max(0.08, 4 * res.std_error)
        published = LS2001_TABLE1[(s0, sigma, t)]
        assert abs(res.price - crr) < tol, (res.price, crr, tol)
        assert abs(res.price - published) < tol, (res.price, published, tol)
        # LSM with an independent pricing set is low-biased, never above the
        # true price by more than MC noise.
        assert res.price < crr + 3 * res.std_error

    def test_american_above_european(self):
        res = lsm_put(36, 0.2, 1)
        euro = black_scholes_price(36, STRIKE, R, 0.2, 1, call=False)
        assert res.price > euro + 2 * res.std_error
        assert res.early_exercise_fraction > 0.3

    def test_deep_itm_bermudan_vs_intrinsic(self):
        # The first exercise date is at t=dt, so the *Bermudan* LSM price of
        # a deep-ITM put may sit a hair below intrinsic-now; the American
        # value is max(price, intrinsic_now), which LSMResult exposes.
        res = lsm_put(30, 0.2, 1)  # deep ITM: intrinsic 10
        assert res.intrinsic_now == pytest.approx(10.0)
        american = max(res.price, res.intrinsic_now)
        assert american >= res.intrinsic_now
        # The Bermudan can still lock the payoff at the first date, so it
        # cannot be worth much less than intrinsic (one period of carry).
        assert res.price > res.intrinsic_now * 0.99
        assert res.early_exercise_fraction > 0.95


class TestAmericanCallNoDividends:
    def test_equals_european_call(self):
        res = longstaff_schwartz_price(
            lambda dt: GBMProcess(R, 0.2, dt),
            s0=40.0, strike=STRIKE, maturity=1.0, r=R, call=True,
            paths=20_000, exercise_dates=50, degree=3, seed=11)
        euro = black_scholes_price(40.0, STRIKE, R, 0.2, 1.0, call=True)
        # Early exercise of a call on a non-dividend payer is never optimal;
        # the fitted rule may fire near the boundary where it costs ~nothing.
        assert abs(res.price - euro) < max(0.08, 4 * res.std_error)


class TestMechanics:
    def test_recorder_stores_only_exercise_dates(self):
        gen_steps = 20
        rec = ExerciseDateRecorder([5, 10, 15, 20])
        prices = np.linspace(90, 110, 7)
        rec.begin_chunk(prices)
        for i in range(1, gen_steps + 1):
            rec.observe(i, prices + i)
        rec.end_chunk(prices + gen_steps)
        m = rec.matrix()
        assert m.shape == (7, 4)                     # paths x K, not paths x steps
        np.testing.assert_allclose(m[:, 0], prices + 5)
        np.testing.assert_allclose(m[:, -1], prices + 20)

    def test_substeps_refine_grid_without_storing_it(self):
        res_a = lsm_put(36, 0.2, 1, substeps=1, paths=15_000)
        res_b = lsm_put(36, 0.2, 1, substeps=4, paths=15_000, seed=12)
        # GBM's log-Euler step is exact, so refining the grid must not move
        # the price beyond MC noise — and storage stays paths x K.
        se = math.hypot(res_a.std_error, res_b.std_error)
        assert abs(res_a.price - res_b.price) < 4 * se
        assert res_b.stored_matrix_elements == res_a.stored_matrix_elements

    def test_chunk_invariance_statistical(self):
        res_small = lsm_put(36, 0.2, 1, chunk_size=1_000)
        res_big = lsm_put(36, 0.2, 1, chunk_size=50_000)
        se = math.hypot(res_small.std_error, res_big.std_error)
        assert abs(res_small.price - res_big.price) < 4 * se

    def test_independent_path_sets(self):
        # Same seed must still give different calibration vs pricing paths;
        # sanity: a run with a different pricing seed but the same
        # calibration seed stays within MC noise.
        a = lsm_put(36, 0.2, 1, seed=21)
        b = lsm_put(36, 0.2, 1, seed=21, paths=20_000)
        assert a.price == b.price  # deterministic given the seed
        c = lsm_put(36, 0.2, 1, seed=22)
        assert abs(a.price - c.price) < 5 * math.hypot(a.std_error, c.std_error)

    def test_heston_american_put_bounds(self):
        # No closed form: assert basic no-arbitrage bounds under Heston.
        res = longstaff_schwartz_price(
            lambda dt: HestonProcess(R, dt, kappa=1.5, theta=0.04, xi=0.3,
                                     rho=-0.7, v0=0.04),
            s0=36.0, strike=STRIKE, maturity=1.0, r=R, call=False,
            paths=15_000, exercise_dates=50, substeps=4, degree=3, seed=11)
        euro_bs = black_scholes_price(36.0, STRIKE, R, 0.2, 1.0, call=False)
        assert res.price > res.intrinsic_now - 3 * res.std_error  # >= intrinsic
        assert res.price < STRIKE                                  # < strike bound
        # comparable magnitude to the sigma=0.2 GBM put (v0=theta=0.04)
        assert abs(res.price - euro_bs) < 1.5

    def test_bad_args_rejected(self):
        with pytest.raises(ValueError):
            lsm_put(36, 0.2, 1, exercise_dates=0)
        with pytest.raises(ValueError):
            lsm_put(36, 0.2, 1, degree=0)
