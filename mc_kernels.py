"""
mc_kernels.py — optional fused inner-loop kernels (Phase 6; REVIEW.md §c.5).

The streaming engine pays one Python-level step iteration per time step
(cheap, ∝ steps not paths) plus per-step observer callbacks. When a run
needs **terminal values only** (no per-step observers), the whole chunk
evolution can be fused into one call:

* pure-NumPy fused loop (always available) — performs *exactly* the same
  per-step operations in the same order as the streaming path
  (``prices *= exp(drift_dt + vol_sqdt * Z[i])``), on a shock matrix drawn
  with the identical RNG sequence, so results are **bit-identical**;
* a Numba ``@njit`` version of the same loop, compiled at first use when
  ``numba`` is importable. Numba's libm ``exp`` may differ from NumPy's
  vectorized ``exp`` in the last ulp, so the JIT path is validated to
  ``rtol=1e-12`` rather than bit equality — this is why kernels are
  **opt-in** (``PathGenerator(kernel=True)``) and the default engine path
  never uses them.

Memory note: the fused path pre-draws a ``(steps, chunk)`` shock matrix —
the same footprint class as the per-chunk Sobol block, tracked in
``MemoryInfo``. Chunk safety is preserved (never ``paths × steps``).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

__all__ = ["numba_available", "fused_gbm_chunk", "FUSED_BACKEND",
           "NUMBA_KERNEL_AVAILABLE"]


def numba_available() -> bool:
    try:  # pragma: no cover - environment dependent
        import numba  # noqa: F401
        return True
    except Exception:
        return False


def _fused_gbm_numpy(prices: np.ndarray, drift_dt: float, vol_sqdt: float,
                     z: np.ndarray) -> np.ndarray:
    # Same ops, same order as the streaming per-step path -> bit-identical.
    for i in range(z.shape[0]):
        prices *= np.exp(drift_dt + vol_sqdt * z[i])
    return prices


def _fused_gbm_kernel(prices, drift_dt, vol_sqdt, zt):  # pragma: no cover
    # zt is (n, steps): per-path rows are contiguous, so each parallel
    # worker walks its paths' shocks sequentially in cache order.
    n, steps = zt.shape
    for j in range(n):  # numba replaces range with prange when parallel
        acc = 0.0
        for i in range(steps):
            acc += drift_dt + vol_sqdt * zt[j, i]
        prices[j] *= math.exp(acc)
    return prices


_fused_gbm_jit = None
if numba_available():  # pragma: no cover - depends on optional dep
    try:
        import numba
        from numba import njit, prange

        def _kernel_parallel(prices, drift_dt, vol_sqdt, zt):
            n, steps = zt.shape
            for j in prange(n):
                acc = 0.0
                for i in range(steps):
                    acc += drift_dt + vol_sqdt * zt[j, i]
                prices[j] *= math.exp(acc)
            return prices

        _fused_gbm_jit = njit(cache=True, fastmath=False,
                              parallel=True)(_kernel_parallel)
    except Exception:
        _fused_gbm_jit = None

# Default backend is "numpy": bit-identical to the streaming engine and, as
# the Phase 6 benchmarks in REVIEW.md show, the run is dominated by shock
# generation (which the bit-compatibility constraint pins to the same
# per-step RNG calls), so the JIT kernel cannot outperform it. "numba" stays
# available for explicit experimentation via the backend argument.
FUSED_BACKEND = "numpy"
NUMBA_KERNEL_AVAILABLE = _fused_gbm_jit is not None


def fused_gbm_chunk(prices: np.ndarray, drift_dt: float, vol_sqdt: float,
                    z: np.ndarray, *, backend: Optional[str] = None,
                    ) -> np.ndarray:
    """Evolve a GBM chunk to its terminal prices in one fused call.

    ``z`` has shape ``(steps, n)`` with row ``i`` holding step ``i+1``'s
    standard-normal shocks — the exact per-step draw order of the streaming
    engine. ``backend`` forces ``"numpy"`` or ``"numba"`` (default: numba
    when importable, else numpy).
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != prices.size:
        raise ValueError("z must have shape (steps, len(prices))")
    use = backend or FUSED_BACKEND
    if use == "numba":
        if _fused_gbm_jit is None:
            raise ValueError("numba backend requested but numba is not available")
        # The JIT kernel accumulates each path's log increments then applies
        # one exp (mathematically identical for GBM; differs from the
        # per-step product only at ~1e-14 relative). Transposed copy gives
        # each parallel worker contiguous per-path shocks.
        zt = np.ascontiguousarray(z.T)
        return _fused_gbm_jit(prices, float(drift_dt), float(vol_sqdt), zt)
    if use == "numpy":
        return _fused_gbm_numpy(prices, float(drift_dt), float(vol_sqdt), z)
    raise ValueError(f"unknown backend {use!r}")
