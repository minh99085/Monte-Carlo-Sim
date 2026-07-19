"""
mc_stats.py — streaming statistics accumulators (Phase 6; REVIEW.md layer 4).

Completes the QuantLib-style Statistics layer: headline risk numbers
without holding every sample. Three tools with explicit trade-offs:

* :class:`P2Quantile` — the classic Jain & Chlamtac (1985) P² estimator:
  O(1) memory per tracked quantile, sequential updates, small deterministic
  approximation error that shrinks with the stream. The textbook streaming
  quantile; best when you need one or two quantiles of an endless stream.
* :class:`ReservoirSample` — uniform reservoir sampling (Vitter's Algorithm
  R, vectorized per chunk): keeps a bounded random subset, from which any
  quantile / VaR / expected-shortfall estimate is an *unbiased* sample
  statistic with MC error ~1/sqrt(k). Preferred here for the full
  percentile table and tail averages, which P² cannot provide.
* :class:`StreamingRiskStats` — the composite used by the pipeline: exact
  streaming moments/min/max (Welford, via ``mc_engine.StreamingMoments``),
  a P² median, and a reservoir for percentiles/VaR/ES. Memory is O(k)
  regardless of path count.

:class:`StreamingStatsPricer` adapts the composite to the PathPricer
protocol so a v2 ``PathGenerator`` sweep can produce risk statistics for
runs too large to keep ``final_values`` in memory (>10^7 paths). The
default ``simulate()`` path is unchanged — its exact array-based statistics
(and the Phase 5 golden anchors) stay as they are.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import numpy as np

from mc_engine import PathPricer, StreamingMoments

__all__ = [
    "P2Quantile",
    "ReservoirSample",
    "StreamingRiskStats",
    "StreamingStatsPricer",
]


class P2Quantile:
    """Jain & Chlamtac's P² single-quantile estimator (5 markers, O(1) mem).

    Updates are inherently sequential (one value at a time); ``add`` accepts
    arrays for convenience but loops internally — use it for one or two
    quantiles, not a full table (that's the reservoir's job).
    """

    def __init__(self, p: float):
        if not 0.0 < p < 1.0:
            raise ValueError("p must be in (0, 1)")
        self.p = float(p)
        self._init: List[float] = []
        self._q: Optional[np.ndarray] = None    # marker heights
        self._n: Optional[np.ndarray] = None    # marker positions (1-based)
        self._np: Optional[np.ndarray] = None   # desired positions
        self.count = 0

    def add(self, values) -> None:
        for v in np.asarray(values, dtype=float).ravel():
            self._add_one(float(v))

    def _add_one(self, x: float) -> None:
        self.count += 1
        if self._q is None:
            self._init.append(x)
            if len(self._init) == 5:
                self._init.sort()
                p = self.p
                self._q = np.asarray(self._init, dtype=float)
                self._n = np.arange(1.0, 6.0)
                self._np = np.array(
                    [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0])
            return
        q, n = self._q, self._n
        # locate cell and update extremes
        if x < q[0]:
            q[0] = x
            k = 0
        elif x >= q[4]:
            q[4] = x
            k = 3
        else:
            k = int(np.searchsorted(q, x, side="right")) - 1
            k = min(max(k, 0), 3)
        n[k + 1:] += 1.0
        p = self.p
        self._np += np.array([0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0])
        # adjust interior markers with parabolic (fallback linear) moves
        for i in (1, 2, 3):
            d = self._np[i] - n[i]
            if (d >= 1.0 and n[i + 1] - n[i] > 1.0) or \
               (d <= -1.0 and n[i - 1] - n[i] < -1.0):
                s = 1.0 if d >= 1.0 else -1.0
                qp = q[i] + s / (n[i + 1] - n[i - 1]) * (
                    (n[i] - n[i - 1] + s) * (q[i + 1] - q[i]) / (n[i + 1] - n[i])
                    + (n[i + 1] - n[i] - s) * (q[i] - q[i - 1]) / (n[i] - n[i - 1])
                )
                if not q[i - 1] < qp < q[i + 1]:
                    qp = q[i] + s * (q[i + int(s)] - q[i]) / (n[i + int(s)] - n[i])
                q[i] = qp
                n[i] += s

    @property
    def value(self) -> float:
        if self._q is not None:
            return float(self._q[2])
        if not self._init:
            return float("nan")
        return float(np.percentile(np.asarray(self._init), 100.0 * self.p))


class ReservoirSample:
    """Uniform random reservoir of at most ``k`` values (Algorithm R,
    vectorized per chunk). Any statistic of the reservoir is an unbiased
    estimate of the same statistic of the full stream, with sampling error
    ~1/sqrt(k). Deterministic for a fixed seed."""

    def __init__(self, k: int = 50_000, seed: Optional[int] = 0):
        if k < 2:
            raise ValueError("reservoir size must be >= 2")
        self.k = int(k)
        self._rng = np.random.default_rng(seed)
        self._buf = np.empty(self.k, dtype=np.float64)
        self.count = 0

    def add(self, values) -> None:
        v = np.asarray(values, dtype=np.float64).ravel()
        if v.size == 0:
            return
        i = 0
        # fill phase
        if self.count < self.k:
            take = min(self.k - self.count, v.size)
            self._buf[self.count:self.count + take] = v[:take]
            self.count += take
            i = take
        if i >= v.size:
            return
        rest = v[i:]
        # replacement phase: item with global index m survives w.p. k/(m+1)
        idx = self.count + np.arange(rest.size)  # 0-based global indices
        accept = self._rng.random(rest.size) < (self.k / (idx + 1.0))
        slots = self._rng.integers(0, self.k, size=int(accept.sum()))
        self._buf[slots] = rest[accept]
        self.count += rest.size

    def sample(self) -> np.ndarray:
        return self._buf[: min(self.count, self.k)].copy()

    def percentile(self, q) -> np.ndarray:
        return np.percentile(self.sample(), q)


class StreamingRiskStats:
    """Composite streaming risk statistics: exact moments + P² median +
    reservoir percentiles/VaR/ES. Memory O(reservoir size)."""

    def __init__(self, s0: float, *, reservoir: int = 50_000,
                 seed: Optional[int] = 0):
        self.s0 = float(s0)
        self.moments = StreamingMoments()
        self.median_p2 = P2Quantile(0.5)
        self.res = ReservoirSample(reservoir, seed)
        self._p2_budget = 200_000  # P² is sequential; cap its workload

    def add(self, final_values: np.ndarray) -> None:
        v = np.asarray(final_values, dtype=np.float64).ravel()
        self.moments.add(v)
        self.res.add(v)
        if self.median_p2.count < self._p2_budget:
            self.median_p2.add(v[: self._p2_budget - self.median_p2.count])

    def summary(self) -> Dict[str, object]:
        m = self.moments
        samp = self.res.sample()
        pnl = samp - self.s0
        out: Dict[str, object] = {
            "paths": m.n,
            "s0": self.s0,
            "expected_value": m.mean,
            "std_value": m.std,
            "min_value": m.min,
            "max_value": m.max,
            "expected_return": m.mean / self.s0 - 1.0 if self.s0 else float("nan"),
            "median_value_p2": self.median_p2.value,
            "reservoir_size": samp.size,
            "estimator": "streaming (exact moments; reservoir quantiles)",
        }
        if samp.size:
            out["percentiles"] = {
                str(p): float(np.percentile(samp, p))
                for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)
            }
            var = {}
            es = {}
            for level in (95.0, 99.0, 99.9):
                q = np.percentile(pnl, 100.0 - level)
                tail = pnl[pnl <= q]
                key = ("%g" % level)
                var[key] = {"value": float(-q), "pct": float(-q) / self.s0}
                es_val = float(-np.mean(tail)) if tail.size else float(-q)
                es[key] = {"value": es_val, "pct": es_val / self.s0}
            out["var"] = var
            out["expected_shortfall"] = es
        return out


class StreamingStatsPricer(PathPricer):
    """PathPricer adapter: stream terminal values into StreamingRiskStats
    without ever materializing the full final-value array. Optionally
    applies the same net-of-cost transform as ``TerminalValuePricer``."""

    def __init__(self, s0: float, cost: float = 0.0, *,
                 reservoir: int = 50_000, seed: Optional[int] = 0):
        from mc_core import apply_costs
        self._apply_costs = apply_costs
        self.s0 = float(s0)
        self.cost = float(cost)
        self.stats = StreamingRiskStats(s0, reservoir=reservoir, seed=seed)

    def end_chunk(self, prices: np.ndarray) -> None:
        self.stats.add(self._apply_costs(prices, self.s0, self.cost))

    def values(self) -> np.ndarray:
        raise NotImplementedError(
            "StreamingStatsPricer deliberately stores no per-path values — "
            "use .stats.summary()"
        )
