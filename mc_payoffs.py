"""
mc_payoffs.py — streaming option PathPricers for the v2 engine (Phase 2).

Every pricer here consumes the evolving chunk **streamingly** — a running
sum, extreme, or flag per path — so the chunk-safe memory guarantee of
``mc_engine.PathGenerator`` survives: nothing ever allocates ``paths ×
steps``.

Pricing convention (read this before trusting a number)
-------------------------------------------------------
All pricers discount the terminal payoff at a user-supplied continuously
compounded risk-free rate ``r`` over ``maturity`` years. A Monte Carlo
price is an *option price* only when the paths are simulated under the
**risk-neutral measure** — for the models in this repo that means building
the process with drift ``mu = r`` (e.g. ``GBMProcess(r, sigma, dt)``).
Simulating with historical drift gives an actuarial expected payoff, not a
market price. Use :func:`check_risk_neutral_drift` (the ``mc_options.py``
entry point always sets ``mu = r`` and says so).

Monitoring convention: pricers observe the path at every generator step
``1..N`` (plus the start price where noted). Asian averages run over the
``N`` step observations; barriers/lookbacks also include the start price.

Closed-form oracles for validation live at the bottom of this module
(Black-Scholes, Reiner-Rubinstein down-barrier, discrete-geometric Asian à
la Kemna-Vorst, Broadie-Glasserman-Kou barrier continuity correction).
They use ``math.erf`` only — no SciPy dependency.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional

import numpy as np

from mc_engine import PathPricer

__all__ = [
    "EuropeanPricer",
    "AsianArithmeticPricer",
    "AsianGeometricPricer",
    "BarrierPricer",
    "DigitalPricer",
    "LookbackPricer",
    "check_risk_neutral_drift",
    "norm_cdf",
    "black_scholes_price",
    "down_and_out_call",
    "down_and_in_call",
    "geometric_asian_call_discrete",
    "bgk_adjusted_barrier",
]


def check_risk_neutral_drift(mu: float, r: float, *, tol: float = 1e-10) -> bool:
    """Warn (and return False) when the simulated drift is not the risk-free
    rate — the resulting number is an expected payoff, not an option price."""
    if abs(float(mu) - float(r)) <= tol:
        return True
    warnings.warn(
        f"Simulated drift mu={mu:.6g} != risk-free rate r={r:.6g}: the "
        "discounted expectation is NOT an arbitrage-free option price. "
        "Build the process with mu = r for pricing.",
        UserWarning,
        stacklevel=2,
    )
    return False


class _OptionPricer(PathPricer):
    """Shared plumbing: strike/discount bookkeeping and chunk collection."""

    def __init__(self, *, strike: float, maturity: float, r: float,
                 call: bool = True):
        if strike <= 0 or maturity <= 0:
            raise ValueError("strike and maturity must be > 0")
        self.strike = float(strike)
        self.maturity = float(maturity)
        self.r = float(r)
        self.call = bool(call)
        self.discount = math.exp(-self.r * self.maturity)
        self._chunks: List[np.ndarray] = []

    def _intrinsic(self, s: np.ndarray) -> np.ndarray:
        if self.call:
            return np.maximum(s - self.strike, 0.0)
        return np.maximum(self.strike - s, 0.0)

    def _emit(self, discounted: np.ndarray) -> None:
        self._chunks.append(np.asarray(discounted, dtype=np.float64))

    def values(self) -> np.ndarray:
        if not self._chunks:
            return np.asarray([], dtype=np.float64)
        return np.concatenate(self._chunks)

    # Convenience for reporting -------------------------------------------
    def price(self) -> float:
        v = self.values()
        return float(np.mean(v)) if v.size else float("nan")

    def std_error(self) -> float:
        """Naive iid standard error of the price estimate. Under antithetic
        sampling the true error is smaller (samples are paired); under Sobol
        this is not a valid error estimate at all — use replications."""
        v = self.values()
        if v.size < 2:
            return float("nan")
        return float(np.std(v, ddof=1) / math.sqrt(v.size))


class EuropeanPricer(_OptionPricer):
    """Vanilla European call/put on the terminal price (validation anchor:
    must match Black-Scholes under risk-neutral GBM)."""

    def end_chunk(self, prices: np.ndarray) -> None:
        self._emit(self.discount * self._intrinsic(prices))


class AsianArithmeticPricer(_OptionPricer):
    """Arithmetic-average Asian call/put, averaging the ``N`` step
    observations (start price excluded). Streaming state: one running sum
    per path."""

    def begin_chunk(self, prices0: np.ndarray) -> None:
        self._sum = np.zeros_like(prices0)
        self._count = 0

    def observe(self, step_i: int, prices: np.ndarray) -> None:
        self._sum += prices
        self._count += 1

    def end_chunk(self, prices: np.ndarray) -> None:
        if self._count == 0:
            raise RuntimeError("Asian pricer saw no observations")
        mean = self._sum / self._count
        self._emit(self.discount * self._intrinsic(mean))


class AsianGeometricPricer(_OptionPricer):
    """Geometric-average Asian call/put (same discrete monitoring as the
    arithmetic pricer). Exists mainly as a validation control: its discrete
    closed form (:func:`geometric_asian_call_discrete`) is exact, and by
    AM-GM the arithmetic Asian call must price above it."""

    def begin_chunk(self, prices0: np.ndarray) -> None:
        self._logsum = np.zeros_like(prices0)
        self._count = 0

    def observe(self, step_i: int, prices: np.ndarray) -> None:
        self._logsum += np.log(prices)
        self._count += 1

    def end_chunk(self, prices: np.ndarray) -> None:
        if self._count == 0:
            raise RuntimeError("Asian pricer saw no observations")
        gmean = np.exp(self._logsum / self._count)
        self._emit(self.discount * self._intrinsic(gmean))


class DigitalPricer(_OptionPricer):
    """Cash-or-nothing digital: pays ``cash`` when the terminal price
    finishes in the money (``S_T > K`` for a call, ``S_T < K`` for a put).

    The payoff is **discontinuous** at the strike — pathwise Greeks are
    invalid for it (the derivative of an indicator is a delta function);
    use the likelihood-ratio method (mc_greeks.py) instead.
    """

    def __init__(self, *, strike: float, maturity: float, r: float,
                 call: bool = True, cash: float = 1.0):
        super().__init__(strike=strike, maturity=maturity, r=r, call=call)
        self.cash = float(cash)

    def end_chunk(self, prices: np.ndarray) -> None:
        itm = (prices > self.strike) if self.call else (prices < self.strike)
        self._emit(self.discount * self.cash * itm.astype(np.float64))


class BarrierPricer(_OptionPricer):
    """Knock-in / knock-out barrier call/put with **discrete monitoring**.

    The barrier is checked at the start price and at every generator step.

    Discrete-vs-continuous monitoring bias
    --------------------------------------
    A path can cross the barrier *between* monitoring dates and come back;
    discrete monitoring misses those crossings, so the knock probability is
    understated relative to the continuous-barrier closed forms. Hence a
    discretely monitored **knock-out** prices *above* the continuous formula
    and a **knock-in** prices *below* it. Broadie–Glasserman–Kou's
    correction shifts the barrier by ``exp(±0.5826 σ √Δt)``
    (:func:`bgk_adjusted_barrier`) to approximate the discrete price with
    the continuous formula; the validation tests use it.

    Streaming state: one boolean knock flag per path.
    """

    def __init__(self, *, strike: float, barrier: float, maturity: float,
                 r: float, call: bool = True, direction: str = "down",
                 knock: str = "out"):
        super().__init__(strike=strike, maturity=maturity, r=r, call=call)
        if direction not in ("up", "down"):
            raise ValueError("direction must be 'up' or 'down'")
        if knock not in ("in", "out"):
            raise ValueError("knock must be 'in' or 'out'")
        if barrier <= 0:
            raise ValueError("barrier must be > 0")
        self.barrier = float(barrier)
        self.direction = direction
        self.knock = knock

    def _crossed(self, prices: np.ndarray) -> np.ndarray:
        if self.direction == "up":
            return prices >= self.barrier
        return prices <= self.barrier

    def begin_chunk(self, prices0: np.ndarray) -> None:
        # A start price already beyond the barrier knocks at inception.
        self._hit = self._crossed(prices0).copy()

    def observe(self, step_i: int, prices: np.ndarray) -> None:
        self._hit |= self._crossed(prices)

    def end_chunk(self, prices: np.ndarray) -> None:
        payoff = self._intrinsic(prices)
        if self.knock == "out":
            payoff = np.where(self._hit, 0.0, payoff)
        else:
            payoff = np.where(self._hit, payoff, 0.0)
        self._emit(self.discount * payoff)


class LookbackPricer(_OptionPricer):
    """Fixed- or floating-strike lookback call/put over the discretely
    monitored path extreme (start price included).

    * ``kind="fixed"``: call pays ``max(M_max − K, 0)``, put pays
      ``max(K − M_min, 0)``.
    * ``kind="floating"``: call pays ``S_T − M_min``, put pays
      ``M_max − S_T`` (``strike`` is ignored; pass any positive number).

    Discrete monitoring understates the true extreme, so both variants
    price below their continuous-lookback closed forms. Streaming state:
    one running min and/or max per path.
    """

    def __init__(self, *, maturity: float, r: float, call: bool = True,
                 kind: str = "floating", strike: float = 1.0):
        super().__init__(strike=strike, maturity=maturity, r=r, call=call)
        if kind not in ("fixed", "floating"):
            raise ValueError("kind must be 'fixed' or 'floating'")
        self.kind = kind

    def begin_chunk(self, prices0: np.ndarray) -> None:
        self._min = prices0.copy()
        self._max = prices0.copy()

    def observe(self, step_i: int, prices: np.ndarray) -> None:
        np.minimum(self._min, prices, out=self._min)
        np.maximum(self._max, prices, out=self._max)

    def end_chunk(self, prices: np.ndarray) -> None:
        if self.kind == "floating":
            payoff = (prices - self._min) if self.call else (self._max - prices)
            payoff = np.maximum(payoff, 0.0)  # non-negative by construction
        else:
            payoff = self._intrinsic(self._max if self.call else self._min)
        self._emit(self.discount * payoff)


# ---------------------------------------------------------------------------
# Closed-form validation oracles (math.erf only — no SciPy)
# ---------------------------------------------------------------------------


def norm_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def black_scholes_price(s0: float, strike: float, r: float, sigma: float,
                        maturity: float, *, call: bool = True) -> float:
    """Black–Scholes European price (no dividends)."""
    if sigma <= 0 or maturity <= 0:
        intr = max(s0 - strike, 0.0) if call else max(strike - s0, 0.0)
        return intr
    sq = sigma * math.sqrt(maturity)
    d1 = (math.log(s0 / strike) + (r + 0.5 * sigma ** 2) * maturity) / sq
    d2 = d1 - sq
    if call:
        return s0 * norm_cdf(d1) - strike * math.exp(-r * maturity) * norm_cdf(d2)
    return strike * math.exp(-r * maturity) * norm_cdf(-d2) - s0 * norm_cdf(-d1)


def down_and_in_call(s0: float, strike: float, barrier: float, r: float,
                     sigma: float, maturity: float) -> float:
    """Reiner–Rubinstein down-and-in call with **continuous** monitoring,
    for barrier <= strike and barrier < s0."""
    if not (barrier < s0 and barrier <= strike):
        raise ValueError("formula requires barrier < s0 and barrier <= strike")
    lam = (r + 0.5 * sigma ** 2) / sigma ** 2
    sq = sigma * math.sqrt(maturity)
    y = math.log(barrier ** 2 / (s0 * strike)) / sq + lam * sq
    return (s0 * (barrier / s0) ** (2.0 * lam) * norm_cdf(y)
            - strike * math.exp(-r * maturity)
            * (barrier / s0) ** (2.0 * lam - 2.0) * norm_cdf(y - sq))


def down_and_out_call(s0: float, strike: float, barrier: float, r: float,
                      sigma: float, maturity: float) -> float:
    """Continuous-monitoring down-and-out call via in–out parity."""
    return (black_scholes_price(s0, strike, r, sigma, maturity, call=True)
            - down_and_in_call(s0, strike, barrier, r, sigma, maturity))


def bgk_adjusted_barrier(barrier: float, s0: float, sigma: float,
                         dt: float) -> float:
    """Broadie–Glasserman–Kou continuity correction: the continuous formula
    evaluated at the shifted barrier approximates the *discretely* monitored
    price. The barrier moves **away** from the start price by
    ``exp(0.5826 σ √Δt)``."""
    beta = 0.5826
    shift = math.exp(beta * sigma * math.sqrt(dt))
    return barrier / shift if barrier < s0 else barrier * shift


def geometric_asian_call_discrete(s0: float, strike: float, r: float,
                                  sigma: float, maturity: float,
                                  n_obs: int, *, call: bool = True) -> float:
    """Exact price of the discretely monitored geometric-average Asian
    option (equally spaced observations at ``t_i = i·T/N``, ``i = 1..N`` —
    the same convention as :class:`AsianGeometricPricer`).

    The log geometric mean is normal with
    ``m = ln S0 + (r − σ²/2)·T·(N+1)/(2N)`` and
    ``v = σ²·T·(N+1)(2N+1)/(6N²)`` — the Kemna–Vorst result specialized to
    discrete averaging.
    """
    n = int(n_obs)
    t = float(maturity)
    m = math.log(s0) + (r - 0.5 * sigma ** 2) * t * (n + 1) / (2.0 * n)
    v = sigma ** 2 * t * (n + 1) * (2.0 * n + 1) / (6.0 * n ** 2)
    sq = math.sqrt(v)
    disc = math.exp(-r * t)
    d1 = (m - math.log(strike) + v) / sq
    d2 = d1 - sq
    fwd = math.exp(m + 0.5 * v)
    if call:
        return disc * (fwd * norm_cdf(d1) - strike * norm_cdf(d2))
    return disc * (strike * norm_cdf(-d2) - fwd * norm_cdf(-d1))
