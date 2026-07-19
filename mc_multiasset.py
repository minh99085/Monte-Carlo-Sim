"""
mc_multiasset.py — correlated multi-asset GBM on the v2 architecture
(Phase 6; REVIEW.md §c.5 "simulate_portfolio re-expressed as a
multi-dimensional process on the same generator").

``mc_core.simulate_portfolio`` is a second, standalone GBM implementation.
This module re-expresses it as the same three layers the single-asset
engine uses — a process (``MultiAssetGBMProcess``), a chunk-safe driver
(``MultiAssetPathGenerator``), and streaming pricers over ``(n, k)`` price
blocks — and reproduces ``simulate_portfolio``'s portfolio values
**bit-identically** for the same inputs and seed (the same hard bar every
port in this refactor has met): per step, one ``standard_normal((n, k))``
draw, correlated through the Cholesky factor, applied as
``prices *= exp(drift_step + vol_step * (z @ chol.T))``.

Chunk safety: the largest allocation is one ``(chunk, k)`` price block plus
one ``(chunk, k)`` shock block — never ``paths × steps`` (or
``paths × steps × k``).

``multiasset_from_returns`` builds the process from historical returns
using the *same* helpers as ``simulate_portfolio`` (``align_returns``,
``shrink_covariance``, ``cholesky_safe``), so parameterization is shared,
not re-derived.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mc_core import (
    DRIFT_HALF,
    DRIFT_HISTORICAL,
    DRIFT_MANUAL,
    DRIFT_ZERO,
    TRADING_DAYS_PER_YEAR,
    align_returns,
    cholesky_safe,
    shrink_covariance,
)

__all__ = [
    "MultiAssetGBMProcess",
    "MultiAssetPathGenerator",
    "MultiPathPricer",
    "BasketTerminalPricer",
    "AssetTerminalPricer",
    "multiasset_from_returns",
]


class MultiAssetGBMProcess:
    """k-asset correlated GBM in log space.

    ``evolve_block(rng, prices)`` advances an ``(n, k)`` price block one
    step, drawing the same ``standard_normal((n, k))`` the legacy portfolio
    loop draws and applying the identical arithmetic.
    """

    def __init__(self, mus: np.ndarray, sigmas: np.ndarray,
                 chol: np.ndarray, dt: float):
        self.mus = np.asarray(mus, dtype=np.float64).ravel()
        self.sigmas = np.asarray(sigmas, dtype=np.float64).ravel()
        self.chol = np.asarray(chol, dtype=np.float64)
        self.k = self.mus.size
        if self.sigmas.size != self.k or self.chol.shape != (self.k, self.k):
            raise ValueError("mus, sigmas and chol dimensions must agree")
        self.dt = float(dt)
        self.sqrt_dt = math.sqrt(self.dt)
        self.drift_step = (self.mus - 0.5 * self.sigmas ** 2) * self.dt
        self.vol_step = self.sigmas * self.sqrt_dt
        self.factors = self.k

    def evolve_block(self, rng: np.random.Generator,
                     prices: np.ndarray) -> None:
        z = rng.standard_normal(prices.shape)
        corr_shock = z @ self.chol.T
        prices *= np.exp(self.drift_step + self.vol_step * corr_shock)


class MultiPathPricer:
    """Streaming consumer of ``(n, k)`` price blocks (the multi-asset
    PathPricer protocol — same begin/observe/end shape as the scalar one)."""

    def begin_chunk(self, prices0: np.ndarray) -> None:  # noqa: B027
        pass

    def observe(self, step_i: int, prices: np.ndarray) -> None:  # noqa: B027
        pass

    def end_chunk(self, prices: np.ndarray) -> None:  # noqa: B027
        pass

    def values(self) -> np.ndarray:
        raise NotImplementedError


class BasketTerminalPricer(MultiPathPricer):
    """Weighted sum of per-asset gross returns at the horizon — exactly the
    quantity ``simulate_portfolio`` reports (relative to 1.0)."""

    def __init__(self, weights: np.ndarray, s0_vec: np.ndarray):
        self.weights = np.asarray(weights, dtype=np.float64).ravel()
        self.s0_vec = np.asarray(s0_vec, dtype=np.float64).ravel()
        self._chunks: List[np.ndarray] = []

    def end_chunk(self, prices: np.ndarray) -> None:
        rel = prices / self.s0_vec
        self._chunks.append((rel * self.weights).sum(axis=1))

    def values(self) -> np.ndarray:
        if not self._chunks:
            return np.asarray([], dtype=np.float64)
        return np.concatenate(self._chunks)


class AssetTerminalPricer(MultiPathPricer):
    """Running per-asset mean gross return (streaming; no per-path storage),
    matching ``simulate_portfolio``'s per-asset summary."""

    def __init__(self, s0_vec: np.ndarray):
        self.s0_vec = np.asarray(s0_vec, dtype=np.float64).ravel()
        self._sum = np.zeros_like(self.s0_vec)
        self.count = 0

    def end_chunk(self, prices: np.ndarray) -> None:
        rel = prices / self.s0_vec
        self._sum += rel.sum(axis=0)
        self.count += prices.shape[0]

    def mean_gross_return(self) -> np.ndarray:
        return self._sum / max(self.count, 1)

    def values(self) -> np.ndarray:
        return self.mean_gross_return()


class MultiAssetPathGenerator:
    """Chunk-safe multi-asset driver, mirroring ``PathGenerator`` with
    ``(n, k)`` blocks. Same chunk slicing and seeding semantics as
    ``simulate_portfolio``'s loop, so runs are reproducible against it."""

    def __init__(self, process: MultiAssetGBMProcess, *, s0_vec: np.ndarray,
                 paths: int, steps: int, chunk_size: int = 50_000,
                 seed: Optional[int] = None):
        if paths < 1 or steps < 1 or chunk_size < 1:
            raise ValueError("paths, steps and chunk_size must be >= 1")
        self.process = process
        self.s0_vec = np.asarray(s0_vec, dtype=np.float64).ravel()
        if self.s0_vec.size != process.k:
            raise ValueError("s0_vec length must equal the asset count")
        self.paths = int(paths)
        self.steps = int(steps)
        self.chunk_size = int(chunk_size)
        self.seed = seed
        self.peak_block_elements = 0

    def run(self, pricers: Sequence[MultiPathPricer]) -> None:
        rng = np.random.default_rng(self.seed)
        produced = 0
        while produced < self.paths:
            n = min(self.chunk_size, self.paths - produced)
            prices = np.tile(self.s0_vec, (n, 1)).astype(np.float64)
            self.peak_block_elements = max(self.peak_block_elements,
                                           prices.size)
            for p in pricers:
                p.begin_chunk(prices)
            for i in range(1, self.steps + 1):
                self.process.evolve_block(rng, prices)
                for p in pricers:
                    p.observe(i, prices)
            for p in pricers:
                p.end_chunk(prices)
            produced += n

    @property
    def is_chunk_safe(self) -> bool:
        return self.peak_block_elements < self.paths * self.process.k \
            or self.paths <= self.chunk_size


def multiasset_from_returns(
    returns_by_ticker: Dict[str, np.ndarray],
    *,
    dt: float = 1.0 / TRADING_DAYS_PER_YEAR,
    drift_mode: str = DRIFT_HISTORICAL,
    manual_drift: Optional[float] = None,
) -> Tuple[MultiAssetGBMProcess, List[str], Dict[str, Any]]:
    """Build a MultiAssetGBMProcess from historical returns with the exact
    parameterization ``simulate_portfolio`` uses (same helpers, same drift
    modes, same shrunk covariance and jittered Cholesky). Returns
    (process, tickers, meta)."""
    tickers, big_r = align_returns(returns_by_ticker)
    k = len(tickers)
    t_hist = big_r.shape[0]
    mus = np.empty(k)
    sigmas = np.empty(k)
    for j in range(k):
        daily_mu = float(np.mean(big_r[:, j]))
        daily_sd = float(np.std(big_r[:, j], ddof=1)) if t_hist > 1 else 0.0
        sigmas[j] = daily_sd * math.sqrt(TRADING_DAYS_PER_YEAR)
        mus[j] = daily_mu * TRADING_DAYS_PER_YEAR + 0.5 * sigmas[j] ** 2
    if drift_mode == DRIFT_HALF:
        mus = 0.5 * mus
    elif drift_mode == DRIFT_ZERO:
        mus = np.zeros_like(mus)
    elif drift_mode == DRIFT_MANUAL and manual_drift is not None:
        mus = np.full_like(mus, float(manual_drift))

    cov, shrink_method = shrink_covariance(big_r)
    sd = np.sqrt(np.clip(np.diag(cov), 1e-16, None))
    corr = cov / np.outer(sd, sd)
    np.clip(corr, -1.0, 1.0, out=corr)
    chol, jittered = cholesky_safe(corr)

    meta = {
        "correlation_matrix": corr,
        "covariance_method": shrink_method,
        "cholesky_jittered": bool(jittered),
        "annual_mus": mus,
        "annual_sigmas": sigmas,
    }
    return MultiAssetGBMProcess(mus, sigmas, chol, dt), tickers, meta
