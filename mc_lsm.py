"""
mc_lsm.py — Longstaff-Schwartz American/Bermudan pricer on the v2 engine
(Phase 2; the design follows QuantLib's LongstaffSchwartzPathPricer split
into a calibration pass and an independent pricing pass — see REVIEW.md).

Memory model (the one documented exception to full streaming)
-------------------------------------------------------------
The backward regression needs a cross-section of the path values at every
**exercise date**, so this module stores a ``paths × K`` matrix where ``K``
is the number of exercise dates — never ``paths × steps``. The matrix is
assembled chunk by chunk from a streaming :class:`ExerciseDateRecorder`
(each chunk contributes a ``chunk × K`` block), the calibration matrix is
released before the pricing pass, and with ``substeps > 1`` the process is
evolved on a finer grid between exercise dates without storing it. For the
defaults (100k paths x 50 dates x 8 bytes ≈ 40 MB) this is deliberate and
bounded; it is reported in the result so nothing is silent.

Algorithm (Longstaff & Schwartz, 2001)
--------------------------------------
1. **Calibration pass** (its own seed): simulate paths, record prices at the
   K exercise dates. Walk backward from maturity; at each date regress the
   discounted realized continuation cashflow on a polynomial basis of the
   *in-the-money* paths only, and update the cashflow where immediate
   exercise beats the fitted continuation value. Store the per-date
   regression coefficients.
2. **Pricing pass** (independent seed): simulate fresh paths and apply the
   *fitted* exercise rule forward — exercise at the first date where
   intrinsic value exceeds the fitted continuation estimate. Using
   independent paths removes the in-sample (foresight) bias of single-pass
   LSM; the remaining bias from an imperfect exercise rule is *low*, i.e.
   the estimator is conservative.

Pricing requires risk-neutral dynamics: build the process with drift
``mu = r`` (see mc_payoffs docstring).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from mc_engine import PathGenerator, PathPricer, StochasticProcess

__all__ = ["ExerciseDateRecorder", "LSMResult", "longstaff_schwartz_price"]


class ExerciseDateRecorder(PathPricer):
    """Streaming recorder that keeps path values at selected steps only.

    ``record_steps`` are 1-based generator step indices (e.g. every
    ``substeps``-th step). Memory: one ``chunk × K`` block per chunk,
    concatenated to ``paths × K`` after the run.
    """

    def __init__(self, record_steps: Sequence[int]):
        self.record_steps = sorted(int(s) for s in record_steps)
        if not self.record_steps or self.record_steps[0] < 1:
            raise ValueError("record_steps must be 1-based step indices")
        self._want = set(self.record_steps)
        self._blocks: List[np.ndarray] = []
        self._cur: List[np.ndarray] = []

    def begin_chunk(self, prices0: np.ndarray) -> None:
        self._cur = []

    def observe(self, step_i: int, prices: np.ndarray) -> None:
        if step_i in self._want:
            self._cur.append(prices.copy())

    def end_chunk(self, prices: np.ndarray) -> None:
        if len(self._cur) != len(self.record_steps):
            raise RuntimeError(
                f"recorded {len(self._cur)} of {len(self.record_steps)} "
                "exercise dates — generator steps do not cover them"
            )
        self._blocks.append(np.column_stack(self._cur))
        self._cur = []

    def matrix(self) -> np.ndarray:
        """(paths, K) matrix of prices at the exercise dates."""
        if not self._blocks:
            return np.empty((0, len(self.record_steps)))
        return np.vstack(self._blocks)

    def values(self) -> np.ndarray:  # PathPricer contract; not used here
        return self.matrix()[:, -1] if self._blocks else np.asarray([])


@dataclass
class LSMResult:
    price: float
    std_error: float
    paths: int
    calibration_paths: int
    exercise_dates: int
    degree: int
    early_exercise_fraction: float   # pricing-pass paths exercised before T
    intrinsic_now: float             # immediate-exercise value at t=0
    stored_matrix_elements: int      # paths × K actually allocated (documented)
    coefficients: List[Optional[np.ndarray]] = field(repr=False, default=None)  # type: ignore[assignment]

    def summary(self) -> str:
        return (
            f"LSM price {self.price:.4f} ± {self.std_error:.4f} "
            f"(paths={self.paths:,}+{self.calibration_paths:,} calib, "
            f"K={self.exercise_dates} exercise dates, degree={self.degree}, "
            f"early-exercise {self.early_exercise_fraction:.1%})"
        )


def _poly_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """Vandermonde basis [1, x, ..., x^degree] on the normalized state."""
    return np.vander(x, N=degree + 1, increasing=True)


def longstaff_schwartz_price(
    process_factory: Callable[[float], StochasticProcess],
    *,
    s0: float,
    strike: float,
    maturity: float,
    r: float,
    call: bool = False,
    paths: int = 50_000,
    calibration_paths: Optional[int] = None,
    exercise_dates: int = 50,
    substeps: int = 1,
    degree: int = 3,
    seed: Optional[int] = 42,
    chunk_size: int = 50_000,
    antithetic: bool = False,
) -> LSMResult:
    """Price an American/Bermudan option by Longstaff-Schwartz.

    ``process_factory(dt)`` must build the risk-neutral process for a step
    of length ``dt`` (e.g. ``lambda dt: GBMProcess(r, sigma, dt)``) —
    Bermudan exercise at ``exercise_dates`` equally spaced dates in
    ``(0, maturity]``; with 50+ dates per year this approximates the
    American price (Longstaff-Schwartz 2001 use 50/year). ``substeps``
    refines the simulation grid between exercise dates (useful for Heston's
    Euler scheme) without storing the intermediate values.
    """
    if exercise_dates < 1 or substeps < 1 or degree < 1:
        raise ValueError("exercise_dates, substeps and degree must be >= 1")
    k_dates = int(exercise_dates)
    steps = k_dates * int(substeps)
    dt = float(maturity) / steps
    dt_ex = float(maturity) / k_dates
    disc_step = math.exp(-r * dt_ex)
    record_steps = [substeps * (j + 1) for j in range(k_dates)]
    n_price = int(paths)
    n_calib = int(calibration_paths) if calibration_paths is not None else n_price
    n_basis = degree + 1

    def intrinsic(s: np.ndarray) -> np.ndarray:
        return np.maximum(s - strike, 0.0) if call else np.maximum(strike - s, 0.0)

    def simulate_matrix(n: int, run_seed: Optional[int]) -> np.ndarray:
        gen = PathGenerator(
            process_factory(dt),
            s0=s0, paths=n, steps=steps, chunk_size=chunk_size,
            seed=run_seed, antithetic=antithetic,
        )
        rec = ExerciseDateRecorder(record_steps)
        gen.run([rec])
        return rec.matrix()

    # ---- 1. calibration pass: fit the exercise rule backward -------------
    calib_seed = None if seed is None else int(seed)
    s_cal = simulate_matrix(n_calib, calib_seed)
    cashflow = intrinsic(s_cal[:, -1])
    coefs: List[Optional[np.ndarray]] = [None] * k_dates
    for t in range(k_dates - 2, -1, -1):
        cashflow *= disc_step                      # value discounted to date t
        intr = intrinsic(s_cal[:, t])
        itm = intr > 0.0
        if int(itm.sum()) > n_basis:
            x = _poly_basis(s_cal[itm, t] / strike, degree)
            beta, *_ = np.linalg.lstsq(x, cashflow[itm], rcond=None)
            continuation = x @ beta
            exercise = intr[itm] > continuation
            idx = np.flatnonzero(itm)[exercise]
            cashflow[idx] = intr[idx]
            coefs[t] = beta
    del s_cal  # release the calibration matrix before the pricing pass

    # ---- 2. pricing pass: apply the fitted rule on fresh paths -----------
    price_seed = None if seed is None else int(seed) + 977_231
    s_pri = simulate_matrix(n_price, price_seed)
    value = np.zeros(n_price, dtype=np.float64)
    alive = np.ones(n_price, dtype=bool)
    for t in range(k_dates - 1):
        beta = coefs[t]
        if beta is None:
            continue                               # never exercise at this date
        intr = intrinsic(s_pri[:, t])
        cand = alive & (intr > 0.0)
        if not cand.any():
            continue
        x = _poly_basis(s_pri[cand, t] / strike, degree)
        continuation = x @ beta
        exercise = intr[cand] > continuation
        idx = np.flatnonzero(cand)[exercise]
        value[idx] = intr[idx] * math.exp(-r * dt_ex * (t + 1))
        alive[idx] = False
    if alive.any():
        value[alive] = intrinsic(s_pri[alive, -1]) * math.exp(-r * maturity)

    price = float(np.mean(value))
    se = float(np.std(value, ddof=1) / math.sqrt(n_price)) if n_price > 1 else float("nan")
    return LSMResult(
        price=price,
        std_error=se,
        paths=n_price,
        calibration_paths=n_calib,
        exercise_dates=k_dates,
        degree=degree,
        early_exercise_fraction=float(1.0 - np.mean(alive)),
        intrinsic_now=float(max(s0 - strike, 0.0) if call else max(strike - s0, 0.0)),
        stored_matrix_elements=int(max(n_calib, n_price) * k_dates),
        coefficients=coefs,
    )
