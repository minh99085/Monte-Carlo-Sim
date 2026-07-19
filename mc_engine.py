"""
mc_engine.py — Phase 1 of the QuantLib-inspired engine refactor (see REVIEW.md).

Decoupled layers:

    StochasticProcess  →  PathGenerator  →  PathPricer(s)  →  Statistics
     (drift+diffusion)     (chunk loop,       (streaming        (streaming
                            shocks, VR)        payoffs)          moments)

Design contract (the parts borrowed from QuantLib's architecture):

* A **StochasticProcess** defines *only* its dynamics: ``drift`` and
  ``diffusion`` per step, chunk-local ``init_state``, and an ``evolve``
  discretization that defaults to Euler in log space. Models with bespoke
  schemes (Heston full truncation) override ``evolve`` only.
* A **PathGenerator** owns everything about randomness and iteration:
  chunk slicing, seeding, antithetic mirroring, per-chunk Sobol shocks and
  the stress-crash overlay. It never allocates a ``paths × steps`` matrix —
  the unit of work is a chunk-length price vector evolved step by step
  (the memory guarantee of ``mc_core.simulate`` is preserved structurally).
* A **PathPricer** consumes the evolving chunk *streamingly*
  (``begin_chunk`` / ``observe`` / ``end_chunk``) and emits one value per
  path. Terminal payoffs simply ignore ``observe``; path-dependent payoffs
  (Asian, barrier, lookback — Phase 3) reduce to running statistics.

Phase 1 scope: GBM and Heston only, behind a feature flag
(``SimulationConfig.engine = "v2"`` or the ``MC_ENGINE=v2`` environment
variable). The legacy engine remains the default and byte-identical; the
flagged path routes the same chunk loop through process objects with
identical arithmetic and RNG draw order, which the tests assert exactly.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from mc_core import (
    MODEL_BLOCK_BOOTSTRAP,
    MODEL_GARCH,
    MODEL_GBM,
    MODEL_HESTON,
    MODEL_HIST_BOOTSTRAP,
    MODEL_KOU,
    MODEL_MERTON,
    MODEL_REGIME,
    MODEL_STUDENT_T,
    DrawdownObserver,   # re-exported: duck-typed streaming observers shared
    SampleRecorder,     # between simulate() and this pipeline (Phase 5)
    MemoryInfo,
    SimulationConfig,
    _draw_gauss,
    _sobol_standard_normals,
    apply_costs,
)

# Models the v2 engine can run. Phase 1 ported GBM/Heston; Phase 4 ported the
# remaining seven (mc_processes.py). Anything not listed falls back to the
# legacy closures (recorded by mc_core in stats["engine"]).
V2_SUPPORTED_MODELS = (
    MODEL_GBM,
    MODEL_HESTON,
    MODEL_STUDENT_T,
    MODEL_HIST_BOOTSTRAP,
    MODEL_BLOCK_BOOTSTRAP,
    MODEL_MERTON,
    MODEL_KOU,
    MODEL_GARCH,
    MODEL_REGIME,
)

# Per-chunk Sobol seed stride — must match mc_core.simulate so the flagged
# path reproduces the legacy shock stream bit-for-bit.
SOBOL_CHUNK_SEED_STRIDE = 1_000_003


# ---------------------------------------------------------------------------
# Layer 1: StochasticProcess — dynamics only
# ---------------------------------------------------------------------------


class StochasticProcess:
    """One asset evolving in log-price space, one step at a time.

    ``drift`` / ``diffusion`` are the *definition* of the process (per-year
    log-drift rate and volatility given the current chunk state); ``evolve``
    is the *discretization* and defaults to Euler in log space. ``z`` is the
    primary Gaussian shock supplied by the generator (so antithetic / Sobol
    policy lives in one place); processes needing extra randomness draw it
    from ``rng`` inside ``evolve`` and declare it via ``factors``.
    """

    #: independent random factors consumed per step (QuantLib: factors())
    factors: int = 1

    #: how this process consumes the generator's shared Gaussian shock —
    #: this is the contract that keeps v2 bit-identical to the legacy
    #: closures, whose models draw randomness through three different
    #: channels:
    #:   "full"  – z comes from the loop's gauss(rng, n, state, i):
    #:             antithetic-aware AND Sobol-aware (GBM, Heston z1, GARCH)
    #:   "plain" – z comes from gauss(rng, n): antithetic-aware but NEVER
    #:             Sobol, matching legacy Kou exactly
    #:   "none"  – the process draws all of its own randomness from ``rng``
    #:             inside evolve() in legacy order; the generator must not
    #:             touch the RNG for it (Student-t, Merton, regime,
    #:             bootstraps)
    gauss_mode: str = "full"

    def __init__(self, dt: float):
        self.dt = float(dt)
        self.sqrt_dt = math.sqrt(self.dt)

    def init_state(self, n: int) -> Dict[str, np.ndarray]:
        """Chunk-local state arrays (each of length ``n``); empty for
        memoryless processes."""
        return {}

    def drift(self, state: Dict[str, np.ndarray], n: int):
        """Per-year log-drift rate (scalar or length-``n`` array)."""
        raise NotImplementedError

    def diffusion(self, state: Dict[str, np.ndarray], n: int):
        """Per-year volatility (scalar or length-``n`` array)."""
        raise NotImplementedError

    def evolve(self, rng: np.random.Generator, state: Dict[str, np.ndarray],
               z: np.ndarray, n: int) -> np.ndarray:
        """One-step log return. Default: Euler in log space."""
        return self.drift(state, n) * self.dt + \
            self.diffusion(state, n) * self.sqrt_dt * z


class GBMProcess(StochasticProcess):
    """Geometric Brownian motion. Arithmetic matches the legacy GBM closure
    exactly: ``(mu - sigma^2/2) dt + sigma sqrt(dt) z``."""

    def __init__(self, mu: float, sigma: float, dt: float):
        super().__init__(dt)
        self.mu = float(mu)
        self.sigma = float(sigma)
        # Precomputed with the same expressions as mc_core._build_step_engine
        # so the flagged path is bit-identical to the legacy engine.
        self._drift_dt = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        self._vol_sqdt = self.sigma * self.sqrt_dt

    def drift(self, state, n):
        return self.mu - 0.5 * self.sigma ** 2

    def diffusion(self, state, n):
        return self.sigma

    def evolve(self, rng, state, z, n):
        return self._drift_dt + self._vol_sqdt * z


class HestonProcess(StochasticProcess):
    """Heston stochastic volatility with full-truncation Euler.

    ``evolve`` is overridden (the variance SDE needs its own correlated
    shock and truncation scheme); the arithmetic and RNG draw order match
    the legacy Heston closure exactly: the generator supplies ``z`` (the
    price shock, Sobol-eligible), then one extra ``standard_normal`` is
    drawn for the variance factor.
    """

    factors = 2

    def __init__(self, mu: float, dt: float, *, kappa: float, theta: float,
                 xi: float, rho: float, v0: float):
        super().__init__(dt)
        self.mu = float(mu)
        self.kappa = float(kappa)
        self.theta = float(theta)
        self.xi = float(xi)
        self.rho = float(rho)
        self.v0 = float(v0)
        self.rho_c = math.sqrt(max(0.0, 1.0 - self.rho ** 2))

    def init_state(self, n: int):
        return {"v": np.full(n, self.v0, dtype=np.float64)}

    def drift(self, state, n):
        v_pos = np.maximum(state["v"], 0.0)
        return self.mu - 0.5 * v_pos

    def diffusion(self, state, n):
        return np.sqrt(np.maximum(state["v"], 0.0))

    def evolve(self, rng, state, z, n):
        v = state["v"]
        # Full-truncation: use max(v, 0) wherever variance enters.
        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)
        z2 = rng.standard_normal(n)
        zv = self.rho * z + self.rho_c * z2
        v_new = (v + self.kappa * (self.theta - v_pos) * self.dt
                 + self.xi * sqrt_v * self.sqrt_dt * zv)
        state["v"] = np.maximum(v_new, 0.0)
        return (self.mu - 0.5 * v_pos) * self.dt + sqrt_v * self.sqrt_dt * z


def process_from_config(cfg: SimulationConfig) -> Optional[StochasticProcess]:
    """Build a StochasticProcess for the configured model, or None when the
    model is not yet ported (caller falls back to the legacy engine)."""
    mu = cfg.effective_mu()
    sigma = cfg.effective_sigma()
    if cfg.model == MODEL_GBM:
        return GBMProcess(mu, sigma, cfg.dt)
    if cfg.model == MODEL_HESTON:
        theta = float(cfg.heston_theta) if cfg.heston_theta is not None else sigma ** 2
        v0 = float(cfg.heston_v0) if cfg.heston_v0 is not None else sigma ** 2
        return HestonProcess(
            mu, cfg.dt,
            kappa=float(cfg.heston_kappa),
            theta=theta,
            xi=float(cfg.heston_xi),
            rho=float(cfg.heston_rho),
            v0=v0,
        )
    # Phase 4 ports live in mc_processes.py (REVIEW.md §c file layout).
    from mc_processes import extended_process_from_config
    return extended_process_from_config(cfg)


def legacy_engine_from_config(
    cfg: SimulationConfig,
    gauss: Callable[..., np.ndarray],
) -> Optional[tuple]:
    """Adapter used by ``mc_core._build_step_engine`` when the v2 flag is on.

    Returns the ``(init_chunk, step)`` pair the legacy chunk loop expects,
    implemented on a StochasticProcess. ``gauss`` is the loop's shock
    closure (antithetic / Sobol aware), so draw order — and therefore every
    output — is bit-identical to the legacy engine.
    """
    process = process_from_config(cfg)
    if process is None:
        return None
    mode = getattr(process, "gauss_mode", "full")

    def init_chunk(rng, n):
        state = process.init_state(n)
        return state if state else None  # legacy loop expects None or dict

    if mode == "full":
        def step(rng, n, state, i):
            z = gauss(rng, n, state, i)
            return process.evolve(rng, state, z, n)
    elif mode == "plain":
        # Legacy Kou calls gauss(rng, n) without state/step: antithetic
        # applies but Sobol never does — reproduce that call exactly.
        def step(rng, n, state, i):
            z = gauss(rng, n)
            return process.evolve(rng, state, z, n)
    else:  # "none": the process draws everything itself, in legacy order
        def step(rng, n, state, i):
            return process.evolve(rng, state, None, n)

    return init_chunk, step


# ---------------------------------------------------------------------------
# Layer 3: PathPricer — streaming payoffs
# ---------------------------------------------------------------------------


class PathPricer:
    """Streaming payoff/observer over evolving chunks.

    The generator calls ``begin_chunk(prices0)`` once per chunk,
    ``observe(step_i, prices)`` after every step, and ``end_chunk(prices)``
    with the terminal chunk prices. ``values()`` returns one number per
    path after the run. Terminal payoffs can ignore ``observe`` entirely;
    path-dependent payoffs reduce to running statistics, so the chunk-safe
    memory guarantee survives.
    """

    def begin_chunk(self, prices0: np.ndarray) -> None:  # noqa: B027
        pass

    def observe(self, step_i: int, prices: np.ndarray) -> None:  # noqa: B027
        pass

    def end_chunk(self, prices: np.ndarray) -> None:  # noqa: B027
        pass

    def values(self) -> np.ndarray:
        raise NotImplementedError


class TerminalValuePricer(PathPricer):
    """Net-of-cost terminal value — the payoff today's ``simulate`` prices.

    Uses the same ``apply_costs`` as the legacy engine so values line up
    exactly.
    """

    def __init__(self, s0: float, cost: float = 0.0):
        self.s0 = float(s0)
        self.cost = float(cost)
        self._chunks: List[np.ndarray] = []

    def end_chunk(self, prices: np.ndarray) -> None:
        self._chunks.append(apply_costs(prices, self.s0, self.cost))

    def values(self) -> np.ndarray:
        if not self._chunks:
            return np.asarray([], dtype=np.float64)
        return np.concatenate(self._chunks)


# ---------------------------------------------------------------------------
# Brownian bridge (QMC dimension ordering; Phase 3, REVIEW.md §c.4)
# ---------------------------------------------------------------------------


class BrownianBridge:
    """Coarse-to-fine Brownian path construction (Jäckel's algorithm,
    *Monte Carlo Methods in Finance*, ch. 10 — reimplemented in NumPy).

    Why: a Sobol sequence's low dimensions are its best-distributed ones.
    Filling the path increment-by-increment spends dimension 1 on the first
    tiny step. The bridge instead uses dimension 1 for the **terminal**
    Brownian level (which carries the largest share of path variance), then
    successive dimensions for midpoints, recursively halving. Low Sobol
    dimensions therefore control the largest-variance features of the path,
    which is what makes QMC converge near O(1/N) for path-dependent payoffs.

    Works on an equally spaced grid of ``steps`` points ``t_i = (i+1)·dt``.
    ``transform(z)`` maps a ``(n_paths, steps)`` matrix of independent
    standard normals — ordered by Sobol dimension — into per-step standard
    normal *increments* (same shape), which the generator consumes exactly
    like plain per-step shocks. The transform allocates one extra
    ``chunk × steps`` work array — the same footprint class as the Sobol
    shock matrix itself, tracked by MemoryInfo.
    """

    def __init__(self, steps: int, dt: float = 1.0):
        if steps < 1:
            raise ValueError("steps must be >= 1")
        self.steps = int(steps)
        self.dt = float(dt)
        n = self.steps
        t = dt * (np.arange(n, dtype=np.float64) + 1.0)
        self._t = t
        self.bridge_index = np.zeros(n, dtype=np.int64)
        self.left_index = np.zeros(n, dtype=np.int64)
        self.right_index = np.zeros(n, dtype=np.int64)
        self.left_weight = np.zeros(n, dtype=np.float64)
        self.right_weight = np.zeros(n, dtype=np.float64)
        self.std_dev = np.zeros(n, dtype=np.float64)

        taken = np.zeros(n, dtype=bool)
        # Dimension 0 -> the terminal point.
        self.bridge_index[0] = n - 1
        self.std_dev[0] = math.sqrt(t[n - 1])
        taken[n - 1] = True
        j = 0
        for i in range(1, n):
            # j: first index not yet constructed; k: next constructed index.
            while taken[j]:
                j += 1
            k = j
            while not taken[k]:
                k += 1
            # Bisect the unset run [j, k): l is its midpoint.
            l = j + ((k - 1 - j) >> 1)
            self.bridge_index[i] = l
            self.left_index[i] = j
            self.right_index[i] = k
            t_l, t_k = t[l], t[k]
            t_j = t[j - 1] if j > 0 else 0.0
            span = t_k - t_j
            self.left_weight[i] = (t_k - t_l) / span
            self.right_weight[i] = (t_l - t_j) / span
            self.std_dev[i] = math.sqrt((t_l - t_j) * (t_k - t_l) / span)
            taken[l] = True
            j = k + 1
            if j >= n:
                j = 0

    def transform(self, z: np.ndarray) -> np.ndarray:
        """(n_paths, steps) dimension-ordered normals → per-step standard
        normal increments (n_paths, steps)."""
        z = np.asarray(z, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != self.steps:
            raise ValueError(f"z must have shape (n_paths, {self.steps})")
        n = self.steps
        w = np.empty_like(z)  # Brownian levels W(t_i)
        w[:, n - 1] = self.std_dev[0] * z[:, 0]
        for i in range(1, n):
            l = self.bridge_index[i]
            j = self.left_index[i]
            k = self.right_index[i]
            mid = self.right_weight[i] * w[:, k] + self.std_dev[i] * z[:, i]
            if j > 0:
                mid += self.left_weight[i] * w[:, j - 1]
            w[:, l] = mid
        # Levels -> increments, renormalized to per-step standard normals.
        inv_sq = 1.0 / math.sqrt(self.dt)
        out = np.empty_like(w)
        out[:, 0] = w[:, 0] * inv_sq
        out[:, 1:] = np.diff(w, axis=1) * inv_sq
        return out


# ---------------------------------------------------------------------------
# Layer 2: PathGenerator — the chunk-safe driver
# ---------------------------------------------------------------------------


class PathGenerator:
    """Chunk-safe path driver: evolves chunk-length price vectors step by
    step through a StochasticProcess and feeds streaming PathPricers.

    Mirrors the semantics of ``mc_core.simulate``'s loop — same seeding,
    same antithetic mirroring (via ``mc_core._draw_gauss``), same per-chunk
    Sobol seeds and stress-crash overlay — so a run with identical
    parameters reproduces the legacy engine's terminal values exactly.
    The largest arrays ever held are chunk-length vectors (plus, under
    Sobol, one ``chunk × steps`` shock matrix per chunk, as in the legacy
    engine); the full ``paths × steps`` matrix is never allocated.
    """

    def __init__(
        self,
        process: StochasticProcess,
        *,
        s0: float,
        paths: int,
        steps: int,
        chunk_size: int = 50_000,
        seed: Optional[int] = None,
        antithetic: bool = False,
        sobol: bool = False,
        bridge: bool = False,
        crash_log: float = 0.0,
    ):
        if paths < 1 or steps < 1 or chunk_size < 1 or s0 <= 0:
            raise ValueError("paths, steps, chunk_size must be >= 1 and s0 > 0")
        if bridge and not sobol:
            raise ValueError(
                "bridge=True requires sobol=True — the Brownian bridge is a "
                "QMC dimension-ordering device (pseudorandom shocks gain "
                "nothing from reordering)"
            )
        if bridge and (getattr(process, "factors", 1) > 1
                       or getattr(process, "gauss_mode", "full") != "full"):
            raise ValueError(
                "Brownian bridge is implemented for single-factor, "
                "Gaussian-shock processes only (GBM, GARCH). Multi-factor "
                "bridging (Heston) and non-Gaussian-driven models are "
                "documented as out of scope in REVIEW.md — use plain sobol "
                "or pseudorandom instead."
            )
        self.process = process
        self.s0 = float(s0)
        self.paths = int(paths)
        self.steps = int(steps)
        self.chunk_size = int(chunk_size)
        self.seed = seed
        self.antithetic = bool(antithetic)
        self.sobol = bool(sobol)
        self.bridge = bool(bridge)
        self._bridge = (
            BrownianBridge(self.steps, getattr(process, "dt", 1.0))
            if bridge else None
        )
        self.crash_log = float(crash_log)
        self.memory = MemoryInfo(
            paths=self.paths, horizon=self.steps, chunk_size=self.chunk_size
        )

    def run(self, pricers: Sequence[PathPricer]) -> MemoryInfo:
        rng = np.random.default_rng(self.seed)
        produced = 0
        chunk_idx = 0
        while produced < self.paths:
            n = min(self.chunk_size, self.paths - produced)
            prices = np.full(n, self.s0, dtype=np.float64)
            self.memory.peak_vector_elements = max(
                self.memory.peak_vector_elements, prices.size
            )
            state: Dict[str, Any] = self.process.init_state(n) or {}
            gauss_mode = getattr(self.process, "gauss_mode", "full")
            use_sobol = self.sobol and gauss_mode == "full"
            if use_sobol:
                sobol_seed = (
                    None if self.seed is None
                    else int(self.seed) + chunk_idx * SOBOL_CHUNK_SEED_STRIDE
                )
                shocks = _sobol_standard_normals(n, self.steps, sobol_seed)
                if self._bridge is not None:
                    # Reorder Sobol dimensions coarse-to-fine over the time
                    # grid; output is per-step standard normal increments.
                    shocks = self._bridge.transform(shocks)
                state["sobol_z"] = shocks
                self.memory.peak_matrix_elements = max(
                    self.memory.peak_matrix_elements,
                    n * self.steps * (2 if self._bridge is not None else 1),
                )

            for p in pricers:
                p.begin_chunk(prices)
            for i in range(1, self.steps + 1):
                # Shock policy mirrors the legacy closures exactly:
                # "full" models take the (Sobol-eligible) shared shock,
                # "plain" models take antithetic-only draws (legacy Kou),
                # "none" models draw everything inside evolve().
                if gauss_mode == "none":
                    z = None
                elif use_sobol:
                    z = state["sobol_z"][:, i - 1]
                else:
                    z = _draw_gauss(rng, n, self.antithetic)
                log_ret = self.process.evolve(rng, state, z, n)
                if i == 1 and self.crash_log:
                    log_ret = log_ret + self.crash_log
                prices *= np.exp(log_ret)
                for p in pricers:
                    p.observe(i, prices)
            for p in pricers:
                p.end_chunk(prices)

            produced += n
            chunk_idx += 1
        return self.memory


def generator_from_config(cfg: SimulationConfig) -> PathGenerator:
    """Standalone v2 pipeline entry: build a PathGenerator mirroring the
    settings of a legacy ``SimulationConfig`` (GBM/Heston only)."""
    cfg = cfg.validate()
    process = process_from_config(cfg)
    if process is None:
        raise ValueError(
            f"model {cfg.model!r} is not ported to the v2 engine yet "
            f"(supported: {V2_SUPPORTED_MODELS})"
        )
    crash_log = (
        math.log(1.0 - cfg.stress_crash_pct)
        if cfg.stress_enabled and cfg.stress_crash_pct > 0
        else 0.0
    )
    return PathGenerator(
        process,
        s0=cfg.s0,
        paths=cfg.paths,
        steps=cfg.horizon,
        chunk_size=cfg.chunk_size,
        seed=cfg.seed,
        antithetic=(cfg.variance_reduction == "antithetic"),
        sobol=(cfg.variance_reduction == "sobol"),
        crash_log=crash_log,
    )


# ---------------------------------------------------------------------------
# Layer 4 (seed): streaming statistics
# ---------------------------------------------------------------------------


class StreamingMoments:
    """Welford accumulator for count / mean / variance over chunked adds.

    The seed of the QuantLib-style Statistics layer: headline moments no
    longer require holding every sample. Quantile accumulators follow in a
    later phase (see REVIEW.md §b, layer 4).
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self._m2 = 0.0
        self._min = math.inf
        self._max = -math.inf

    def add(self, values: np.ndarray) -> None:
        v = np.asarray(values, dtype=np.float64).ravel()
        if v.size == 0:
            return
        n_b = v.size
        mean_b = float(v.mean())
        m2_b = float(((v - mean_b) ** 2).sum())
        if self.n == 0:
            self.n, self.mean, self._m2 = n_b, mean_b, m2_b
        else:
            delta = mean_b - self.mean
            total = self.n + n_b
            self._m2 += m2_b + delta * delta * self.n * n_b / total
            self.mean += delta * n_b / total
            self.n = total
        self._min = min(self._min, float(v.min()))
        self._max = max(self._max, float(v.max()))

    @property
    def variance(self) -> float:
        return self._m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max
