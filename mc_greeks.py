"""
mc_greeks.py — Greeks on the v2 pipeline: pathwise, likelihood-ratio, and
common-random-number finite differences (Phase 3, Part A; REVIEW.md §c.3).

The point of this module is **method validity**, not just numbers. Each
estimator states exactly when it is unbiased, and the dispatcher
(:func:`compute_greek`) refuses invalid (greek, payoff, method) combinations
with an error that steers to a valid method:

Pathwise (PW)
    Differentiates the *payoff* along each path:
    ``d payoff/d theta = payoff'(S) · dS/d theta``. Unbiased only when the
    payoff is continuous and (almost everywhere) differentiable — European
    and arithmetic Asian calls/puts qualify. It is **invalid for gamma**
    (differentiating the call's indicator a second time drops a Dirac mass;
    PW would silently underestimate) and invalid for **discontinuous
    payoffs** (digitals; barriers at the barrier). This module raises for
    those instead of returning a biased number. Lowest variance where valid.

Likelihood-ratio (LRM)
    Differentiates the transition *density*, not the payoff, so it handles
    any payoff of the terminal price — including discontinuous ones — and
    it can do **gamma**. Implemented against the exact GBM lognormal
    terminal density (the v2 GBM log-Euler step is exact, so multi-step
    paths keep the exact terminal law). The score weights scale like
    ``1/(σ√T)`` (delta) and ``1/(S0²σ²T)`` (gamma): variance grows with
    maturity and with small σ√T — LRM is the high-variance fallback, and
    the reported standard errors say so honestly. Terminal payoffs only in
    this phase (European, digital); path-dependent per-step scores are
    future work.

Finite difference with common random numbers (CRN FD)
    Bump-and-reprice with the *same seed* on every leg, so all legs see the
    same shocks: the O(1/bump²) variance of independent-seed differences
    collapses to the variance of a smooth per-path difference. Central
    differences (bias O(bump²)); works for any pricer and any process
    (Heston included) — the universal cross-check.

All estimators stream through the standard PathPricer sweep (running
per-path accumulators, never ``paths × steps``), and one sweep yields
price + Greeks together (:func:`price_and_greeks`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from mc_engine import GBMProcess, PathGenerator, PathPricer, StochasticProcess
from mc_payoffs import (
    AsianArithmeticPricer,
    DigitalPricer,
    EuropeanPricer,
    norm_cdf,
)

__all__ = [
    "GreekResult",
    "MethodValidityError",
    "PathwiseEstimator",
    "LikelihoodRatioEstimator",
    "finite_difference_greek",
    "compute_greek",
    "price_and_greeks",
    "norm_pdf",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "digital_delta",
    "digital_price",
]

GREEKS = ("delta", "vega", "gamma")
METHODS = ("pathwise", "lrm", "fd")
PAYOFF_KINDS = ("european", "digital", "asian")

_PW_VALID = {("delta", "european"), ("vega", "european"),
             ("delta", "asian"), ("vega", "asian")}
_LRM_VALID = {(g, p) for g in GREEKS for p in ("european", "digital")}


class MethodValidityError(ValueError):
    """Raised for (greek, payoff, method) combos that would be biased."""


@dataclass
class GreekResult:
    greek: str
    method: str
    payoff: str
    value: float
    std_error: float
    n_paths: int
    detail: str = ""

    def __str__(self) -> str:
        extra = f"; {self.detail}" if self.detail else ""
        return (f"{self.greek} [{self.method}, {self.payoff}] = "
                f"{self.value:+.6f} ± {self.std_error:.6f} "
                f"(n={self.n_paths:,}{extra})")


def _mean_se(contrib: np.ndarray):
    n = int(contrib.size)
    mean = float(np.mean(contrib)) if n else float("nan")
    se = float(np.std(contrib, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    return mean, se, n


# ---------------------------------------------------------------------------
# Pathwise estimator (streaming; GBM; smooth payoffs only)
# ---------------------------------------------------------------------------


class PathwiseEstimator(PathPricer):
    """Streaming pathwise delta and vega under GBM.

    Everything is recovered from observed prices: under GBM the Brownian
    level is ``W_t = (ln(S_t/S0) − (r − σ²/2)t)/σ``, giving
    ``dS_t/dS0 = S_t/S0`` and ``dS_t/dσ = S_t (W_t − σ t)``. For the Asian,
    the average's sensitivities are the running means of those, and the
    chain rule multiplies by the payoff indicator — all per-path running
    sums, chunk-safe.

    Valid: European / arithmetic-Asian call & put, delta & vega. Anything
    else (gamma, digitals, barriers) raises :class:`MethodValidityError`.
    """

    def __init__(self, payoff: str, *, s0: float, strike: float, r: float,
                 sigma: float, maturity: float, n_steps: int,
                 call: bool = True):
        if payoff not in ("european", "asian"):
            raise MethodValidityError(
                f"pathwise Greeks are invalid for payoff {payoff!r} "
                "(discontinuous payoff) — use method='lrm' or 'fd'"
            )
        if sigma <= 0:
            raise ValueError("pathwise estimator needs sigma > 0")
        self.payoff = payoff
        self.s0 = float(s0)
        self.strike = float(strike)
        self.r = float(r)
        self.sigma = float(sigma)
        self.maturity = float(maturity)
        self.n_steps = int(n_steps)
        self.call = bool(call)
        self.discount = math.exp(-self.r * self.maturity)
        self._delta: List[np.ndarray] = []
        self._vega: List[np.ndarray] = []

    def _w(self, prices: np.ndarray, t: float) -> np.ndarray:
        return (np.log(prices / self.s0)
                - (self.r - 0.5 * self.sigma ** 2) * t) / self.sigma

    def begin_chunk(self, prices0: np.ndarray) -> None:
        if self.payoff == "asian":
            self._sum = np.zeros_like(prices0)
            self._dvega = np.zeros_like(prices0)

    def observe(self, step_i: int, prices: np.ndarray) -> None:
        if self.payoff != "asian":
            return
        t_i = self.maturity * step_i / self.n_steps
        self._sum += prices
        self._dvega += prices * (self._w(prices, t_i) - self.sigma * t_i)

    def end_chunk(self, prices: np.ndarray) -> None:
        sign = 1.0 if self.call else -1.0
        if self.payoff == "european":
            itm = (prices > self.strike) if self.call else (prices < self.strike)
            ds_ds0 = prices / self.s0
            ds_dsig = prices * (self._w(prices, self.maturity)
                                - self.sigma * self.maturity)
        else:  # asian
            avg = self._sum / self.n_steps
            itm = (avg > self.strike) if self.call else (avg < self.strike)
            ds_ds0 = avg / self.s0
            ds_dsig = self._dvega / self.n_steps
        self._delta.append(self.discount * sign * itm * ds_ds0)
        self._vega.append(self.discount * sign * itm * ds_dsig)

    def values(self) -> np.ndarray:
        raise NotImplementedError("use result('delta') / result('vega')")

    def result(self, greek: str) -> GreekResult:
        if greek == "delta":
            arr = np.concatenate(self._delta)
        elif greek == "vega":
            arr = np.concatenate(self._vega)
        else:
            raise MethodValidityError(
                "pathwise gamma is biased (the payoff derivative is "
                "discontinuous at the strike) — use method='lrm' or 'fd'"
            )
        value, se, n = _mean_se(arr)
        return GreekResult(greek, "pathwise", self.payoff, value, se, n)


# ---------------------------------------------------------------------------
# Likelihood-ratio estimator (terminal payoffs; exact GBM density)
# ---------------------------------------------------------------------------


class LikelihoodRatioEstimator(PathPricer):
    """LRM delta / vega / gamma for terminal payoffs under GBM.

    With ``Z = (ln(S_T/S0) − (r − σ²/2)T)/(σ√T)`` the score weights of the
    lognormal terminal density are::

        delta:  Z / (S0 σ √T)
        vega:   (Z² − 1)/σ − Z √T
        gamma:  (Z² − 1)/(S0² σ² T) − Z/(S0² σ √T)

    The estimator is ``E[e^{-rT} · payoff(S_T) · weight]`` — the payoff is
    never differentiated, so discontinuous payoffs (digitals) are fine and
    gamma exists. Variance note: the weights blow up as σ√T shrinks and the
    payoff·weight product widens with maturity — expect larger standard
    errors than PW where both apply.
    """

    def __init__(self, payoff: str, *, s0: float, strike: float, r: float,
                 sigma: float, maturity: float, call: bool = True,
                 cash: float = 1.0):
        if payoff not in ("european", "digital"):
            raise MethodValidityError(
                f"LRM is implemented for terminal payoffs only, not "
                f"{payoff!r} — use method='pathwise' (delta/vega) or 'fd'"
            )
        if sigma <= 0 or maturity <= 0:
            raise ValueError("LRM needs sigma > 0 and maturity > 0")
        self.payoff = payoff
        self.s0 = float(s0)
        self.strike = float(strike)
        self.r = float(r)
        self.sigma = float(sigma)
        self.maturity = float(maturity)
        self.call = bool(call)
        self.cash = float(cash)
        self.discount = math.exp(-self.r * self.maturity)
        self._contrib: Dict[str, List[np.ndarray]] = {g: [] for g in GREEKS}

    def _payoff(self, s_t: np.ndarray) -> np.ndarray:
        if self.payoff == "european":
            if self.call:
                return np.maximum(s_t - self.strike, 0.0)
            return np.maximum(self.strike - s_t, 0.0)
        itm = (s_t > self.strike) if self.call else (s_t < self.strike)
        return self.cash * itm.astype(np.float64)

    def end_chunk(self, prices: np.ndarray) -> None:
        s0, sig, t = self.s0, self.sigma, self.maturity
        sq = sig * math.sqrt(t)
        z = (np.log(prices / s0) - (self.r - 0.5 * sig ** 2) * t) / sq
        pay = self.discount * self._payoff(prices)
        self._contrib["delta"].append(pay * z / (s0 * sq))
        self._contrib["vega"].append(pay * ((z ** 2 - 1.0) / sig
                                            - z * math.sqrt(t)))
        self._contrib["gamma"].append(
            pay * ((z ** 2 - 1.0) / (s0 ** 2 * sig ** 2 * t)
                   - z / (s0 ** 2 * sq)))

    def values(self) -> np.ndarray:
        raise NotImplementedError("use result(greek)")

    def result(self, greek: str) -> GreekResult:
        if greek not in GREEKS:
            raise ValueError(f"unknown greek {greek!r}")
        value, se, n = _mean_se(np.concatenate(self._contrib[greek]))
        return GreekResult(greek, "lrm", self.payoff, value, se, n,
                           detail="high-variance method; see docstring")


# ---------------------------------------------------------------------------
# Finite differences with common random numbers
# ---------------------------------------------------------------------------


def _make_pricer(payoff: str, *, strike: float, maturity: float, r: float,
                 call: bool, cash: float) -> PathPricer:
    if payoff == "european":
        return EuropeanPricer(strike=strike, maturity=maturity, r=r, call=call)
    if payoff == "digital":
        return DigitalPricer(strike=strike, maturity=maturity, r=r, call=call,
                             cash=cash)
    if payoff == "asian":
        return AsianArithmeticPricer(strike=strike, maturity=maturity, r=r,
                                     call=call)
    raise ValueError(f"unknown payoff kind {payoff!r}")


def finite_difference_greek(
    greek: str,
    payoff: str,
    *,
    s0: float,
    strike: float,
    r: float,
    sigma: float,
    maturity: float,
    call: bool = True,
    cash: float = 1.0,
    paths: int = 100_000,
    steps: int = 64,
    seed: Optional[int] = 42,
    chunk_size: int = 50_000,
    bump_rel: float = 0.01,
    crn: bool = True,
    process_factory: Optional[Callable[[float, float, float], StochasticProcess]] = None,
) -> GreekResult:
    """Central-difference greek by bump-and-reprice.

    ``crn=True`` (default) reruns every leg with the **same seed** — common
    random numbers — so the per-path difference is smooth and the variance
    is comparable to a pathwise estimator. ``crn=False`` reseeds each leg
    independently, exploding the variance by O(Var(payoff)/bump²); it exists
    to demonstrate exactly that (see the validation table). Bias is
    O(bump²) from the central difference.

    ``process_factory(mu, sigma, dt)`` lets the same machinery difference a
    non-GBM process (e.g. Heston); default is risk-neutral GBM.
    """
    if greek not in GREEKS:
        raise ValueError(f"unknown greek {greek!r}")
    factory = process_factory or (lambda mu, sig, dt: GBMProcess(mu, sig, dt))
    dt = maturity / steps

    def leg(s0_leg: float, sigma_leg: float, leg_seed) -> np.ndarray:
        gen = PathGenerator(factory(r, sigma_leg, dt), s0=s0_leg, paths=paths,
                            steps=steps, chunk_size=chunk_size, seed=leg_seed)
        pricer = _make_pricer(payoff, strike=strike, maturity=maturity, r=r,
                              call=call, cash=cash)
        gen.run([pricer])
        return pricer.values()

    seeds = ((seed, seed, seed) if crn or seed is None
             else (seed, seed + 101, seed + 202))
    if greek == "delta":
        h = bump_rel * s0
        up, dn = leg(s0 + h, sigma, seeds[0]), leg(s0 - h, sigma, seeds[1])
        per_path = (up - dn) / (2.0 * h)
    elif greek == "vega":
        h = bump_rel * sigma
        up, dn = leg(s0, sigma + h, seeds[0]), leg(s0, sigma - h, seeds[1])
        per_path = (up - dn) / (2.0 * h)
    else:  # gamma
        h = bump_rel * s0
        up = leg(s0 + h, sigma, seeds[0])
        mid = leg(s0, sigma, seeds[2])
        dn = leg(s0 - h, sigma, seeds[1])
        per_path = (up - 2.0 * mid + dn) / (h * h)

    if crn:
        value, se, n = _mean_se(per_path)
    else:
        # Independent legs: per-path pairing is meaningless; the estimator
        # variance is the sum of the legs' variances over the bump scale.
        value = float(np.mean(per_path))
        n = per_path.size
        if greek == "gamma":
            var = (np.var(up, ddof=1) + 4.0 * np.var(mid, ddof=1)
                   + np.var(dn, ddof=1)) / (h * h) ** 2
        else:
            var = (np.var(up, ddof=1) + np.var(dn, ddof=1)) / (2.0 * h) ** 2
        se = float(math.sqrt(var / n))
    return GreekResult(greek, "fd", payoff, value, se, n,
                       detail=f"central, bump={bump_rel:.3%}, "
                              f"{'CRN' if crn else 'independent seeds'}")


# ---------------------------------------------------------------------------
# Dispatcher + one-sweep driver
# ---------------------------------------------------------------------------


def _auto_method(greek: str, payoff: str) -> str:
    if (greek, payoff) in _PW_VALID:
        return "pathwise"
    if (greek, payoff) in _LRM_VALID:
        return "lrm"
    return "fd"


def _check_validity(greek: str, payoff: str, method: str) -> None:
    if method == "pathwise" and (greek, payoff) not in _PW_VALID:
        if greek == "gamma":
            raise MethodValidityError(
                "pathwise gamma is biased (discontinuous payoff derivative "
                "at the strike) — use method='lrm' (terminal payoffs) or 'fd'"
            )
        raise MethodValidityError(
            f"pathwise {greek} is invalid for payoff {payoff!r} "
            "(payoff is discontinuous) — use method='lrm' or 'fd'"
        )
    if method == "lrm" and (greek, payoff) not in _LRM_VALID:
        raise MethodValidityError(
            f"LRM is implemented for terminal payoffs only (european, "
            f"digital), not {payoff!r} — use method='pathwise' or 'fd'"
        )


def price_and_greeks(
    payoff: str,
    greeks=("delta", "vega"),
    *,
    s0: float,
    strike: float,
    r: float,
    sigma: float,
    maturity: float,
    call: bool = True,
    cash: float = 1.0,
    paths: int = 100_000,
    steps: int = 64,
    seed: Optional[int] = 42,
    chunk_size: int = 50_000,
    method: str = "auto",
) -> Dict[str, object]:
    """One path sweep → price plus the requested Greeks.

    The price pricer and the greek estimators are run as siblings in a
    single ``PathGenerator.run`` sweep (auto method choice per greek unless
    ``method`` forces one; forcing an invalid method raises).
    """
    if payoff not in PAYOFF_KINDS:
        raise ValueError(f"payoff must be one of {PAYOFF_KINDS}")
    chosen = {}
    for g in greeks:
        m = _auto_method(g, payoff) if method == "auto" else method
        _check_validity(g, payoff, m)
        if m == "fd":
            raise MethodValidityError(
                f"{g} for {payoff!r} needs bump-and-reprice — call "
                "finite_difference_greek() (multiple sweeps by nature)"
            )
        chosen[g] = m

    pricer = _make_pricer(payoff, strike=strike, maturity=maturity, r=r,
                          call=call, cash=cash)
    sweep: List[PathPricer] = [pricer]
    pw = lrm = None
    if any(m == "pathwise" for m in chosen.values()):
        pw = PathwiseEstimator(payoff, s0=s0, strike=strike, r=r, sigma=sigma,
                               maturity=maturity, n_steps=steps, call=call)
        sweep.append(pw)
    if any(m == "lrm" for m in chosen.values()):
        lrm = LikelihoodRatioEstimator(payoff, s0=s0, strike=strike, r=r,
                                       sigma=sigma, maturity=maturity,
                                       call=call, cash=cash)
        sweep.append(lrm)

    gen = PathGenerator(GBMProcess(r, sigma, maturity / steps), s0=s0,
                        paths=paths, steps=steps, chunk_size=chunk_size,
                        seed=seed)
    gen.run(sweep)

    out: Dict[str, object] = {
        "price": pricer.price(),
        "price_se": pricer.std_error(),
        "n_paths": paths,
    }
    for g, m in chosen.items():
        out[g] = (pw if m == "pathwise" else lrm).result(g)
    return out


def compute_greek(
    greek: str,
    payoff: str,
    method: str = "auto",
    **kwargs,
) -> GreekResult:
    """Dispatch a single greek to a *valid* estimator.

    ``method='auto'`` picks pathwise where valid, then LRM, then finite
    differences. Forcing an invalid method raises
    :class:`MethodValidityError` with a pointer to a valid one.
    """
    if greek not in GREEKS:
        raise ValueError(f"unknown greek {greek!r}")
    if payoff not in PAYOFF_KINDS:
        raise ValueError(f"payoff must be one of {PAYOFF_KINDS}")
    m = _auto_method(greek, payoff) if method == "auto" else method
    if m not in METHODS:
        raise ValueError(f"unknown method {m!r}")
    _check_validity(greek, payoff, m)
    if m == "fd":
        return finite_difference_greek(greek, payoff, **kwargs)
    fd_only = {"bump_rel", "crn", "process_factory"}
    clean = {k: v for k, v in kwargs.items() if k not in fd_only}
    res = price_and_greeks(payoff, greeks=(greek,), method=m, **clean)
    return res[greek]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Closed-form Greek oracles (validation; math.erf/exp only)
# ---------------------------------------------------------------------------


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(s0, strike, r, sigma, t):
    sq = sigma * math.sqrt(t)
    d1 = (math.log(s0 / strike) + (r + 0.5 * sigma ** 2) * t) / sq
    return d1, d1 - sq


def bs_delta(s0, strike, r, sigma, t, *, call=True) -> float:
    d1, _ = _d1_d2(s0, strike, r, sigma, t)
    return norm_cdf(d1) if call else norm_cdf(d1) - 1.0


def bs_gamma(s0, strike, r, sigma, t) -> float:
    d1, _ = _d1_d2(s0, strike, r, sigma, t)
    return norm_pdf(d1) / (s0 * sigma * math.sqrt(t))


def bs_vega(s0, strike, r, sigma, t) -> float:
    d1, _ = _d1_d2(s0, strike, r, sigma, t)
    return s0 * norm_pdf(d1) * math.sqrt(t)


def digital_price(s0, strike, r, sigma, t, *, call=True, cash=1.0) -> float:
    _, d2 = _d1_d2(s0, strike, r, sigma, t)
    p = norm_cdf(d2) if call else norm_cdf(-d2)
    return cash * math.exp(-r * t) * p


def digital_delta(s0, strike, r, sigma, t, *, call=True, cash=1.0) -> float:
    _, d2 = _d1_d2(s0, strike, r, sigma, t)
    sign = 1.0 if call else -1.0
    return sign * cash * math.exp(-r * t) * norm_pdf(d2) / (s0 * sigma * math.sqrt(t))
