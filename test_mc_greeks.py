"""Phase 3 Part A acceptance tests: Greeks with method-validity enforcement.

Hard targets (all under risk-neutral GBM, closed forms as oracles):
- European delta/gamma/vega within 3 SE of N(d1), n(d1)/(S0σ√T), S0 n(d1)√T.
- PW delta and LRM delta agree within combined SE.
- LRM gamma matches closed form where PW is (correctly) refused.
- Digital: PW delta rejected; LRM delta matches the closed-form digital delta.
- CRN finite differences collapse the variance vs independent reseeding.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mc_greeks import (
    GreekResult,
    LikelihoodRatioEstimator,
    MethodValidityError,
    PathwiseEstimator,
    bs_delta,
    bs_gamma,
    bs_vega,
    compute_greek,
    digital_delta,
    digital_price,
    finite_difference_greek,
    norm_pdf,
    price_and_greeks,
)
from mc_payoffs import black_scholes_price

S0, K, R, SIG, T = 100.0, 100.0, 0.05, 0.25, 1.0
KW = dict(s0=S0, strike=K, r=R, sigma=SIG, maturity=T,
          paths=100_000, steps=16, seed=42)


def _within(res: GreekResult, ref: float, n_se: float = 3.0):
    assert res.std_error > 0
    assert abs(res.value - ref) < n_se * res.std_error, (str(res), ref)


class TestEuropeanClosedForms:
    def test_pw_delta(self):
        _within(compute_greek("delta", "european", "pathwise", **KW),
                bs_delta(S0, K, R, SIG, T))

    def test_pw_vega(self):
        _within(compute_greek("vega", "european", "pathwise", **KW),
                bs_vega(S0, K, R, SIG, T))

    def test_lrm_delta_vega_gamma(self):
        for greek, ref in (("delta", bs_delta(S0, K, R, SIG, T)),
                           ("vega", bs_vega(S0, K, R, SIG, T)),
                           ("gamma", bs_gamma(S0, K, R, SIG, T))):
            _within(compute_greek(greek, "european", "lrm", **KW), ref)

    def test_put_deltas(self):
        ref = bs_delta(S0, K, R, SIG, T, call=False)
        _within(compute_greek("delta", "european", "pathwise",
                              call=False, **KW), ref)
        _within(compute_greek("delta", "european", "lrm",
                              call=False, **KW), ref)

    def test_pw_and_lrm_agree_two_methods_one_truth(self):
        # Independent seeds so the agreement is between genuine estimates.
        pw = compute_greek("delta", "european", "pathwise",
                           **{**KW, "seed": 7})
        lrm = compute_greek("delta", "european", "lrm", **{**KW, "seed": 8})
        combined = math.hypot(pw.std_error, lrm.std_error)
        assert abs(pw.value - lrm.value) < 3 * combined
        # LRM is the documented high-variance method
        assert lrm.std_error > pw.std_error

    def test_fd_crn_matches_closed_form(self):
        _within(finite_difference_greek("delta", "european", crn=True, **KW),
                bs_delta(S0, K, R, SIG, T))
        _within(finite_difference_greek("gamma", "european", crn=True,
                                        bump_rel=0.05, **KW),
                bs_gamma(S0, K, R, SIG, T), n_se=4.0)


class TestMethodValidity:
    def test_pw_gamma_rejected(self):
        with pytest.raises(MethodValidityError, match="lrm|fd"):
            compute_greek("gamma", "european", "pathwise", **KW)
        with pytest.raises(MethodValidityError):
            PathwiseEstimator("european", s0=S0, strike=K, r=R, sigma=SIG,
                              maturity=T, n_steps=16).result("gamma")

    def test_pw_digital_rejected(self):
        with pytest.raises(MethodValidityError, match="lrm|fd"):
            compute_greek("delta", "digital", "pathwise", **KW)
        with pytest.raises(MethodValidityError):
            PathwiseEstimator("digital", s0=S0, strike=K, r=R, sigma=SIG,
                              maturity=T, n_steps=16)

    def test_lrm_asian_rejected(self):
        with pytest.raises(MethodValidityError, match="pathwise|fd"):
            compute_greek("delta", "asian", "lrm", **KW)

    def test_auto_dispatch(self):
        assert compute_greek("delta", "european", **KW).method == "pathwise"
        assert compute_greek("gamma", "european", **KW).method == "lrm"
        assert compute_greek("delta", "digital", **KW).method == "lrm"
        assert compute_greek("delta", "asian", **KW).method == "pathwise"


class TestDigital:
    def test_lrm_delta_matches_closed_form(self):
        res = compute_greek("delta", "digital", "lrm", **KW)
        _within(res, digital_delta(S0, K, R, SIG, T))

    def test_price_matches_closed_form(self):
        out = price_and_greeks("digital", ("delta",), **KW)
        cf = digital_price(S0, K, R, SIG, T)
        se = out["price_se"]
        assert abs(out["price"] - cf) < 3 * se

    def test_fd_crn_cross_checks_lrm(self):
        lrm = compute_greek("delta", "digital", "lrm", **KW)
        # Digital FD needs a bump big enough to move paths across the
        # discontinuity smoothly; CRN still helps.
        fd = finite_difference_greek("delta", "digital", crn=True,
                                     bump_rel=0.02, **KW)
        assert abs(fd.value - lrm.value) < 4 * math.hypot(fd.std_error,
                                                          lrm.std_error)


class TestAsianPathwise:
    def test_pw_matches_crn_fd(self):
        kw = {**KW, "steps": 64}
        out = price_and_greeks("asian", ("delta", "vega"), **kw)
        fd_d = finite_difference_greek("delta", "asian", crn=True, **kw)
        fd_v = finite_difference_greek("vega", "asian", crn=True, **kw)
        for pw, fd in ((out["delta"], fd_d), (out["vega"], fd_v)):
            assert abs(pw.value - fd.value) < 3 * math.hypot(pw.std_error,
                                                             fd.std_error)

    def test_single_sweep_yields_price_and_greeks(self):
        out = price_and_greeks("asian", ("delta", "vega"),
                               **{**KW, "steps": 32})
        assert out["price"] > 0
        assert isinstance(out["delta"], GreekResult)
        assert isinstance(out["vega"], GreekResult)
        assert 0.0 < out["delta"].value < 1.0


class TestCRNVarianceCollapse:
    def test_crn_variance_much_smaller_than_independent(self):
        crn = finite_difference_greek("delta", "european", crn=True, **KW)
        ind = finite_difference_greek("delta", "european", crn=False, **KW)
        # variance ratio: CRN collapses O(Var/bump^2) to pathwise-like size
        assert crn.std_error < 0.2 * ind.std_error, \
            (crn.std_error, ind.std_error)
        # both remain unbiased within their own noise
        ref = bs_delta(S0, K, R, SIG, T)
        assert abs(crn.value - ref) < 3 * crn.std_error
        assert abs(ind.value - ref) < 3 * ind.std_error


class TestOracles:
    def test_bs_greeks_internal_consistency(self):
        # ATM-ish sanity + parity relations
        assert 0.5 < bs_delta(S0, K, R, SIG, T) < 0.7
        assert bs_delta(S0, K, R, SIG, T) - bs_delta(S0, K, R, SIG, T,
                                                     call=False) == pytest.approx(1.0)
        assert bs_gamma(S0, K, R, SIG, T) > 0
        assert bs_vega(S0, K, R, SIG, T) > 0
        # FD of the closed-form price reproduces the closed-form greeks
        h = 1e-4
        fd_delta = (black_scholes_price(S0 + h, K, R, SIG, T)
                    - black_scholes_price(S0 - h, K, R, SIG, T)) / (2 * h)
        assert fd_delta == pytest.approx(bs_delta(S0, K, R, SIG, T), abs=1e-6)
        assert norm_pdf(0.0) == pytest.approx(1 / math.sqrt(2 * math.pi))

    def test_chunking_does_not_change_estimators(self):
        a = compute_greek("delta", "european", "pathwise",
                          **{**KW, "paths": 20_000, "chunk_size": 1_000})
        b = compute_greek("delta", "european", "pathwise",
                          **{**KW, "paths": 20_000, "chunk_size": 20_000})
        # different chunking -> different draw mapping; statistical equality
        assert abs(a.value - b.value) < 4 * math.hypot(a.std_error,
                                                       b.std_error)
