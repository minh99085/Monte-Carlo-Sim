"""Shared Monte Carlo simulation core for Geometric Brownian Motion (GBM).

This module is the single source of truth for the simulation logic used by both
the command line tool (``monte_carlo_gbm.py``) and the Streamlit GUI (``app.py``).

Design goals (CPU-first, Windows-laptop friendly):

* **Memory safety** -- a 1,000,000+ path simulation must NEVER allocate a full
  ``paths x steps`` matrix.  Paths are produced in chunks, and within a chunk we
  evolve the price vector step by step.  Only the following are retained:

    1. the final value of every path (a single 1-D array of length ``paths``),
    2. aggregate statistics, and
    3. a small, bounded number of *sample* full trajectories for plotting.

* **Reproducibility** -- a fixed ``seed`` produces identical results for a given
  configuration.

* **No heavy dependencies** -- only ``numpy`` is required for the maths.  Market
  data (``yfinance``) and plotting (``matplotlib``) are imported lazily so the
  core stays importable on a minimal install / offline machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Optional, Dict, List, Any
import csv
import io
import json
import math
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Path-mode constants -- the single source of truth shared by the CLI and GUI.
# app.py must only reference constants defined here.
# ---------------------------------------------------------------------------

# Preset path counts.
PREVIEW_PATHS = 10_000
STANDARD_PATHS = 100_000
SERIOUS_PATHS = 1_000_000

# Custom mode bounds (inclusive): any safe path count up to the serious preset.
CUSTOM_MIN_PATHS = 1_000
CUSTOM_MAX_PATHS = 1_000_000

# Advanced tail-risk mode bounds (inclusive).
TAIL_RISK_MIN_PATHS = 2_000_000
TAIL_RISK_MAX_PATHS = 5_000_000

# Default chunk size for the heavy "Serious" mode.  Keeping this in the
# 25,000-50,000 range bounds peak memory regardless of total path count.
DEFAULT_SERIOUS_CHUNK_SIZE = 50_000
# Backwards-compatible alias (older callers used this name).
DEFAULT_SERIOUS_CHUNK = DEFAULT_SERIOUS_CHUNK_SIZE

# Path-mode presets exposed by the GUI/CLI.  ``None`` means "user supplied"
# (``Custom`` picks 1,000-1,000,000; ``Tail-risk (advanced)`` picks
# 2,000,000-5,000,000).  Built from the constants above so the two never drift.
PATH_MODES: Dict[str, Optional[int]] = {
    "Preview": PREVIEW_PATHS,
    "Standard": STANDARD_PATHS,
    "Serious": SERIOUS_PATHS,
    "Custom": None,
    "Tail-risk (advanced)": None,
}

# Confidence levels (percent) used for VaR / Expected Shortfall.
RISK_LEVELS = (95.0, 99.0, 99.9)

# Percentiles reported in the percentile table.
REPORT_PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)

# Offline fallback parameters used when market data cannot be fetched.
FALLBACK_S0 = 100.0
FALLBACK_MU = 0.08      # annualized drift
FALLBACK_SIGMA = 0.20   # annualized volatility


# ---------------------------------------------------------------------------
# Model catalogue
# ---------------------------------------------------------------------------

MODEL_GBM = "GBM Normal"
MODEL_STUDENT_T = "Student-t GBM"
MODEL_HIST_BOOTSTRAP = "Historical Bootstrap"
MODEL_BLOCK_BOOTSTRAP = "Block Bootstrap"
MODEL_MERTON = "Merton Jump-Diffusion"
MODEL_REGIME = "Regime Switching"
MODEL_HESTON = "Heston Stochastic Volatility"
MODEL_GARCH = "GARCH(1,1)"
MODEL_KOU = "Kou Jump-Diffusion"

MODELS = (
    MODEL_GBM,
    MODEL_STUDENT_T,
    MODEL_HIST_BOOTSTRAP,
    MODEL_BLOCK_BOOTSTRAP,
    MODEL_MERTON,
    MODEL_REGIME,
    MODEL_HESTON,
    MODEL_GARCH,
    MODEL_KOU,
)

# Models that need a historical daily-return series to run.
BOOTSTRAP_MODELS = (MODEL_HIST_BOOTSTRAP, MODEL_BLOCK_BOOTSTRAP)

# Version stamp for the maths layer (bumped when models/metrics change shape).
MATH_MODEL_VERSION = "2.0"

# ---------------------------------------------------------------------------
# Variance-reduction methods
# ---------------------------------------------------------------------------

VR_NONE = "none"
VR_ANTITHETIC = "antithetic"
VR_SOBOL = "sobol"
VR_CONTROL = "control_variate"
VARIANCE_REDUCTION_METHODS = (VR_NONE, VR_ANTITHETIC, VR_SOBOL, VR_CONTROL)

# ---------------------------------------------------------------------------
# Engine selection (Phase 1 of the mc_engine refactor — see REVIEW.md)
# ---------------------------------------------------------------------------

ENGINE_LEGACY = "legacy"
ENGINE_V2 = "v2"
ENGINES = (ENGINE_LEGACY, ENGINE_V2)
# Environment override so the CLI/GUI can opt in without new flags.
ENGINE_ENV_VAR = "MC_ENGINE"


def _effective_engine(cfg: "SimulationConfig") -> str:
    env = os.environ.get(ENGINE_ENV_VAR, "").strip().lower()
    if env in ENGINES:
        return env
    return cfg.engine

# ---------------------------------------------------------------------------
# Conservative drift modes
# ---------------------------------------------------------------------------

DRIFT_HISTORICAL = "Historical drift"
DRIFT_HALF = "Half historical drift"
DRIFT_ZERO = "Zero drift"
DRIFT_MANUAL = "Manual drift"

DRIFT_MODES = (DRIFT_HISTORICAL, DRIFT_HALF, DRIFT_ZERO, DRIFT_MANUAL)

# ---------------------------------------------------------------------------
# Merton jump-diffusion presets (per-year jump intensity, mean, vol)
# ---------------------------------------------------------------------------

JUMP_PRESETS = {
    "stock": {"intensity": 1.0, "mean": -0.02, "vol": 0.05},
    "crypto": {"intensity": 6.0, "mean": -0.03, "vol": 0.12},
}

# ---------------------------------------------------------------------------
# Regime-switching presets.  Each regime carries a drift multiplier and a
# volatility multiplier applied to the base (mu, sigma); ``transition`` is a
# row-stochastic matrix P[i, j] = P(next regime j | current regime i).
# Regime order: 0 = normal, 1 = high-volatility, 2 = crash/bear.
# ---------------------------------------------------------------------------

REGIME_PRESETS = {
    "stock": {
        "names": ("normal", "high-vol", "crash"),
        "mu_factors": (1.0, 0.3, -4.0),
        "sigma_factors": (1.0, 1.8, 3.0),
        "transition": (
            (0.970, 0.027, 0.003),
            (0.050, 0.930, 0.020),
            (0.100, 0.200, 0.700),
        ),
    },
    "crypto": {
        "names": ("normal", "high-vol", "crash"),
        "mu_factors": (1.0, 0.2, -5.0),
        "sigma_factors": (1.3, 2.5, 4.0),
        "transition": (
            (0.940, 0.050, 0.010),
            (0.060, 0.900, 0.040),
            (0.080, 0.250, 0.670),
        ),
    },
}

# Default threshold for the "probability of a large drawdown" metric.
DRAWDOWN_THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Configuration / result containers
# ---------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """Immutable description of a single Monte Carlo run."""

    ticker: str = "ASSET"
    s0: float = FALLBACK_S0
    paths: int = 100_000
    horizon: int = 252                 # number of forward steps (trading days)
    mu: float = FALLBACK_MU            # annualized drift
    sigma: float = FALLBACK_SIGMA      # annualized volatility
    dt: float = 1.0 / TRADING_DAYS_PER_YEAR
    chunk_size: int = DEFAULT_SERIOUS_CHUNK
    seed: Optional[int] = None
    cost: float = 0.0                  # proportional transaction cost / slippage
    sample_paths: int = 50             # full trajectories retained for plotting
    convergence_points: int = 200      # points sampled for the convergence curve

    # ---- Model selection & realism knobs -------------------------------
    model: str = MODEL_GBM

    # Conservative drift mode.
    drift_mode: str = DRIFT_HISTORICAL
    manual_drift: Optional[float] = None      # used when drift_mode == DRIFT_MANUAL

    # Student-t GBM.
    t_df: float = 5.0                         # degrees of freedom (> 2)

    # Bootstrap models (daily log returns sampled from history).
    historical_returns: Optional[np.ndarray] = None
    block_length: int = 20                    # block bootstrap block size (days)

    # Merton jump-diffusion (annualized intensity; per-jump mean/vol in log space).
    jump_intensity: float = 1.0
    jump_mean: float = -0.02
    jump_vol: float = 0.05

    # Regime switching.
    regime_preset: str = "stock"

    # Heston stochastic volatility (theta/v0 default to sigma^2 when None).
    heston_kappa: float = 1.5
    heston_theta: Optional[float] = None
    heston_xi: float = 0.3
    heston_rho: float = -0.7
    heston_v0: Optional[float] = None

    # GARCH(1,1) (omega defaults so long-run variance == sigma^2 when None).
    garch_omega: Optional[float] = None
    garch_alpha: float = 0.08
    garch_beta: float = 0.90

    # Kou double-exponential jump diffusion.
    kou_intensity: float = 1.0      # jumps per year
    kou_p_up: float = 0.4           # probability a jump is positive
    kou_eta_up: float = 25.0        # up-jump rate (mean up-jump = 1/eta_up)
    kou_eta_down: float = 15.0      # down-jump rate (mean down-jump = 1/eta_down)

    # Variance reduction.
    variance_reduction: str = VR_NONE

    # Risk-of-ruin threshold (ruin if price ever falls below this * s0).
    ruin_threshold: float = 0.50

    # Deterministic stress overlay (applied on top of any model).
    stress_enabled: bool = False
    stress_crash_pct: float = 0.0             # one-day crash, e.g. 0.20 == -20%
    stress_vol_multiplier: float = 1.0
    stress_drift_haircut: float = 0.0         # fraction of drift removed [0, 1]

    # Threshold (fraction) for the large-drawdown probability metric.
    drawdown_threshold: float = DRAWDOWN_THRESHOLD

    # Engine selection. As of Phase 5 the default is "v2" (the mc_engine
    # StochasticProcess pipeline — all nine models, bit-identical to the
    # legacy closures). "legacy" remains available as a one-release escape
    # hatch, and the MC_ENGINE environment variable overrides either way
    # without CLI changes (MC_ENGINE=legacy to fall back globally).
    engine: str = ENGINE_V2

    def validate(self) -> "SimulationConfig":
        if self.paths < 1:
            raise ValueError("paths must be >= 1")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.s0 <= 0:
            raise ValueError("s0 must be > 0")
        if self.sigma < 0:
            raise ValueError("sigma must be >= 0")
        if self.dt <= 0:
            raise ValueError("dt must be > 0")
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if not (0.0 <= self.cost < 1.0):
            raise ValueError("cost must be in [0, 1)")
        if self.sample_paths < 0:
            raise ValueError("sample_paths must be >= 0")
        if self.model not in MODELS:
            raise ValueError(f"Unknown model: {self.model!r}")
        if self.drift_mode not in DRIFT_MODES:
            raise ValueError(f"Unknown drift_mode: {self.drift_mode!r}")
        if self.model == MODEL_STUDENT_T and self.t_df <= 2.0:
            raise ValueError("Student-t degrees of freedom must be > 2")
        if self.model in BOOTSTRAP_MODELS:
            if self.historical_returns is None or np.asarray(self.historical_returns).size < 2:
                raise ValueError(
                    f"{self.model} requires historical_returns (>= 2 daily returns)"
                )
        if self.model == MODEL_BLOCK_BOOTSTRAP and self.block_length < 1:
            raise ValueError("block_length must be >= 1")
        if self.model == MODEL_MERTON and self.jump_intensity < 0:
            raise ValueError("jump_intensity must be >= 0")
        if self.model == MODEL_REGIME and self.regime_preset not in REGIME_PRESETS:
            raise ValueError(f"Unknown regime_preset: {self.regime_preset!r}")
        if self.model == MODEL_HESTON:
            if self.heston_kappa < 0 or self.heston_xi < 0:
                raise ValueError("Heston kappa and xi must be >= 0")
            if not (-1.0 <= self.heston_rho <= 1.0):
                raise ValueError("Heston rho must be in [-1, 1]")
        if self.model == MODEL_GARCH:
            if self.garch_alpha < 0 or self.garch_beta < 0:
                raise ValueError("GARCH alpha and beta must be >= 0")
            if self.garch_alpha + self.garch_beta >= 1.0:
                raise ValueError("GARCH requires alpha + beta < 1 for stationarity")
        if self.model == MODEL_KOU:
            if self.kou_intensity < 0:
                raise ValueError("kou_intensity must be >= 0")
            if not (0.0 <= self.kou_p_up <= 1.0):
                raise ValueError("kou_p_up must be in [0, 1]")
            if self.kou_eta_up <= 1.0 or self.kou_eta_down <= 0.0:
                # eta_up > 1 keeps E[e^{up jump}] finite.
                raise ValueError("kou_eta_up must be > 1 and kou_eta_down > 0")
        if self.variance_reduction not in VARIANCE_REDUCTION_METHODS:
            raise ValueError(f"Unknown variance_reduction: {self.variance_reduction!r}")
        if self.engine not in ENGINES:
            raise ValueError(f"Unknown engine: {self.engine!r} (use one of {ENGINES})")
        if not (0.0 < self.ruin_threshold < 1.0):
            raise ValueError("ruin_threshold must be in (0, 1)")
        if not (0.0 <= self.stress_crash_pct < 1.0):
            raise ValueError("stress_crash_pct must be in [0, 1)")
        if self.stress_vol_multiplier <= 0:
            raise ValueError("stress_vol_multiplier must be > 0")
        if not (0.0 <= self.stress_drift_haircut <= 1.0):
            raise ValueError("stress_drift_haircut must be in [0, 1]")
        return self

    # ------------------------------------------------------------------
    # Derived drift / volatility (drift mode + stress overlay applied)
    # ------------------------------------------------------------------
    def effective_mu(self) -> float:
        if self.drift_mode == DRIFT_HISTORICAL:
            base = self.mu
        elif self.drift_mode == DRIFT_HALF:
            base = 0.5 * self.mu
        elif self.drift_mode == DRIFT_ZERO:
            base = 0.0
        else:  # DRIFT_MANUAL
            base = self.mu if self.manual_drift is None else float(self.manual_drift)
        if self.stress_enabled:
            base *= (1.0 - self.stress_drift_haircut)
        return base

    def effective_sigma(self) -> float:
        sigma = self.sigma
        if self.stress_enabled:
            sigma *= self.stress_vol_multiplier
        return sigma


@dataclass
class MemoryInfo:
    """Memory accounting that proves chunked execution is in effect."""

    paths: int
    horizon: int
    chunk_size: int
    # Number of elements in the largest 2-D array actually allocated during the
    # run.  For a chunk-safe simulation this is bounded by the sample-path block
    # and is *far* smaller than ``paths * (horizon + 1)``.
    peak_matrix_elements: int = 0
    # Largest 1-D working buffer (== effective chunk size).
    peak_vector_elements: int = 0

    @property
    def full_matrix_elements(self) -> int:
        """Elements that a naive (forbidden) full-matrix run would allocate."""
        return self.paths * (self.horizon + 1)

    @property
    def full_matrix_bytes(self) -> int:
        return self.full_matrix_elements * 8  # float64

    @property
    def peak_bytes(self) -> int:
        return (self.peak_matrix_elements + self.peak_vector_elements + self.paths) * 8

    @property
    def is_chunk_safe(self) -> bool:
        """True when we avoided allocating the full path x step matrix."""
        return self.peak_matrix_elements < self.full_matrix_elements

    def status(self, available_bytes: Optional[int] = None) -> str:
        gb = 1024 ** 3
        full_gb = self.full_matrix_bytes / gb
        peak_gb = self.peak_bytes / gb
        msg = (
            f"Chunked execution {'ACTIVE' if self.is_chunk_safe else 'NOT active'}. "
            f"Peak working memory ~{peak_gb:.3f} GB "
            f"(a full {self.paths:,}x{self.horizon + 1} matrix would need "
            f"~{full_gb:.3f} GB)."
        )
        if available_bytes is not None:
            avail_gb = available_bytes / gb
            if self.peak_bytes > available_bytes:
                msg += (
                    f" WARNING: estimated peak ~{peak_gb:.3f} GB exceeds available "
                    f"~{avail_gb:.3f} GB."
                )
            else:
                msg += f" Fits within available ~{avail_gb:.3f} GB."
        return msg


@dataclass
class SimulationResult:
    config: SimulationConfig
    final_values: np.ndarray            # net ending values, length == paths
    sample_trajectories: np.ndarray     # shape (n_sample, horizon + 1)
    convergence_paths: np.ndarray       # x-axis: cumulative path counts
    convergence_means: np.ndarray       # y-axis: running mean of net ending value
    stats: Dict[str, Any]
    memory: MemoryInfo
    runtime_seconds: float

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def expected_value(self) -> float:
        return self.stats["expected_value"]

    @property
    def median_value(self) -> float:
        return self.stats["median_value"]


# ---------------------------------------------------------------------------
# Market data (optional, lazy, offline-safe)
# ---------------------------------------------------------------------------


@dataclass
class MarketParameters:
    s0: float
    mu: float
    sigma: float
    source: str          # "yfinance" or "fallback"
    note: str = ""
    daily_log_returns: Optional[np.ndarray] = None  # for bootstrap models


def annualized_parameters(prices) -> "tuple[float, float]":
    """Estimate annualized ``(mu, sigma)`` from a 1-D price series.

    Uses daily log returns:

    * daily drift   = mean(log returns)
    * daily vol     = std(log returns, ddof=1)
    * annual sigma  = daily vol * sqrt(252)
    * annual mu     = daily drift * 252 + 0.5 * (daily vol ** 2) * 252

    The ``0.5 * sigma^2`` term converts the estimated *log* drift back into the
    simple annualized drift expected by the GBM formula
    ``exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)``, so the simulated
    expected log-growth matches the history.
    """

    prices = np.asarray(prices, dtype=float).ravel()
    if prices.size < 2:
        raise ValueError("need at least two prices to estimate parameters")
    if np.any(prices <= 0):
        raise ValueError("prices must be strictly positive")

    log_ret = np.diff(np.log(prices))
    daily_mu = float(np.mean(log_ret))
    daily_sigma = float(np.std(log_ret, ddof=1)) if log_ret.size > 1 else 0.0
    mu = daily_mu * TRADING_DAYS_PER_YEAR + 0.5 * (daily_sigma ** 2) * TRADING_DAYS_PER_YEAR
    sigma = daily_sigma * math.sqrt(TRADING_DAYS_PER_YEAR)
    return mu, sigma


def estimate_parameters_from_history(
    ticker: str,
    years: float = 3.0,
    *,
    s0_override: Optional[float] = None,
) -> MarketParameters:
    """Estimate ``s0``, annualized ``mu`` and ``sigma`` from price history.

    Uses ``yfinance`` when available and reachable; otherwise falls back to safe
    defaults so the tool keeps working offline (e.g. in CI).  ``s0_override``
    always wins for the starting price when provided.
    """

    note = ""
    try:  # pragma: no cover - network path is environment dependent
        import yfinance as yf  # imported lazily

        period = f"{max(1, int(round(years)))}y"
        data = yf.download(
            ticker,
            period=period,
            progress=False,
            auto_adjust=True,
        )
        if data is None or len(data) < 2:
            raise ValueError("no data returned")

        close = data["Close"]
        # ``Close`` can be a DataFrame (multi-ticker) -- collapse to a Series.
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        prices = np.asarray(close.dropna(), dtype=float).ravel()
        if prices.size < 2:
            raise ValueError("insufficient price history")

        mu, sigma = annualized_parameters(prices)
        log_ret = np.diff(np.log(prices))
        s0 = float(prices[-1]) if s0_override is None else float(s0_override)
        return MarketParameters(
            s0=s0, mu=mu, sigma=sigma, source="yfinance", note=note,
            daily_log_returns=log_ret,
        )
    except Exception as exc:  # noqa: BLE001 - any failure -> safe fallback
        note = f"Falling back to default parameters ({type(exc).__name__}: {exc})."
        s0 = FALLBACK_S0 if s0_override is None else float(s0_override)
        # Synthesize a small pseudo-history so bootstrap models still function.
        synth_rng = np.random.default_rng(0)
        daily_sigma = FALLBACK_SIGMA / math.sqrt(TRADING_DAYS_PER_YEAR)
        daily_mu = FALLBACK_MU / TRADING_DAYS_PER_YEAR - 0.5 * daily_sigma ** 2
        synth_returns = synth_rng.normal(daily_mu, daily_sigma, size=504)
        return MarketParameters(
            s0=s0, mu=FALLBACK_MU, sigma=FALLBACK_SIGMA, source="fallback", note=note,
            daily_log_returns=synth_returns,
        )


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def apply_costs(gross_ending: np.ndarray, s0: float, cost: float) -> np.ndarray:
    """Apply proportional transaction cost / slippage to gross ending prices.

    Models a round-trip: a proportional ``cost`` is charged on the entry
    notional (``s0``) and on the exit notional (the gross ending price).
    """

    if cost <= 0:
        return gross_ending
    return gross_ending * (1.0 - cost) - s0 * cost


# ---------------------------------------------------------------------------
# Core simulation (chunked, memory-safe)
# ---------------------------------------------------------------------------


def predict_memory(config: "SimulationConfig") -> MemoryInfo:
    """Predict the memory footprint of :func:`simulate` *without* running it.

    Mirrors the accounting performed inside :func:`simulate`: the only 2-D array
    kept is the bounded sample-trajectory block, and the largest 1-D working
    buffer is bounded by the chunk size.  This lets callers (and tests) confirm
    chunk-safety for very large path counts without paying for the simulation.
    """

    cfg = config
    n_sample = min(cfg.sample_paths, cfg.paths)
    mem = MemoryInfo(
        paths=cfg.paths, horizon=cfg.horizon, chunk_size=cfg.chunk_size
    )
    mem.peak_matrix_elements = n_sample * (cfg.horizon + 1)
    mem.peak_vector_elements = min(cfg.chunk_size, cfg.paths)
    return mem


def _bootstrap_daily_drift(cfg: SimulationConfig, emp_mean: float) -> float:
    """Target daily log-drift for bootstrap models given the drift mode."""
    if cfg.drift_mode == DRIFT_HISTORICAL:
        d = emp_mean
    elif cfg.drift_mode == DRIFT_HALF:
        d = 0.5 * emp_mean
    elif cfg.drift_mode == DRIFT_ZERO:
        d = 0.0
    else:  # DRIFT_MANUAL
        manual = cfg.mu if cfg.manual_drift is None else float(cfg.manual_drift)
        d = manual / TRADING_DAYS_PER_YEAR
    if cfg.stress_enabled:
        d *= (1.0 - cfg.stress_drift_haircut)
    return d


def _draw_gauss(rng, n: int, antithetic: bool) -> np.ndarray:
    """Standard normal draws, optionally antithetic across the two chunk halves."""
    if not antithetic:
        return rng.standard_normal(n)
    half = (n + 1) // 2
    z = rng.standard_normal(half)
    out = np.empty(n, dtype=np.float64)
    out[:half] = z
    out[half:] = -z[: n - half]
    return out


def _sobol_standard_normals(n_paths: int, n_dims: int, seed: Optional[int]) -> np.ndarray:
    """Generate an ``(n_paths, n_dims)`` matrix of standard normals via Sobol QMC.

    Requires SciPy.  Caller must guard with :func:`sobol_available`.  Values are
    clipped away from {0,1} before the inverse-normal map so quantiles stay finite.
    Memory is bounded by the *chunk* size the caller requests, not total paths.
    """
    from scipy.stats import qmc  # type: ignore

    # SciPy Sobol is most balanced at power-of-two sizes; generate then slice.
    n_gen = 1 << max(1, int(math.ceil(math.log2(max(n_paths, 2)))))
    eng = qmc.Sobol(d=int(n_dims), scramble=True, seed=seed)
    try:
        u = eng.random(n_gen)
    except Exception:  # pragma: no cover
        u = eng.random_base2(m=int(math.log2(n_gen)))
    u = np.clip(np.asarray(u[:n_paths], dtype=np.float64), 1e-12, 1.0 - 1e-12)
    # Phi^{-1}(u) = sqrt(2) * erfinv(2u - 1)
    if hasattr(np, "erfinv"):
        z = math.sqrt(2.0) * np.erfinv(2.0 * u - 1.0)
    else:  # pragma: no cover - ancient numpy
        z = np.empty_like(u)
        for idx, ui in np.ndenumerate(u):
            z[idx] = _norm_ppf(float(ui))
    return np.asarray(z, dtype=np.float64)


def _attach_sobol_shocks(state, n: int, steps: int, seed: Optional[int]):
    """Ensure ``state`` is a dict carrying a chunk-local Sobol shock matrix."""
    z = _sobol_standard_normals(n, steps, seed)
    if state is None:
        return {"sobol_z": z}
    if isinstance(state, dict):
        state = dict(state)
        state["sobol_z"] = z
        return state
    return {"sobol_z": z, "_wrapped": state}


def _build_step_engine(cfg: SimulationConfig):
    """Return ``(init_chunk, step)`` closures implementing the selected model.

    ``init_chunk(rng, n)`` builds any per-chunk per-path state (bounded by the
    chunk size -- never by total paths).  ``step(rng, n, state, i)`` returns the
    log-return for every path on step ``i`` (1-based).  All state and working
    arrays are length ``n`` (the chunk), preserving chunk-safe memory use.

    Variance reduction
    ------------------
    * **antithetic** -- mirrored Gaussian draws inside each chunk.
    * **sobol** -- when SciPy is available, per-chunk Sobol normals are stored on
      ``state['sobol_z']`` (shape ``n x steps``) and consumed step-by-step for
      models that draw Gaussian shocks (GBM, Heston z1, GARCH).  Callers must
      attach the matrix via :func:`_attach_sobol_shocks` before the step loop.
    * **control_variate** -- applied *after* the path loop in :func:`simulate`
      (GBM terminal-price control); does not change the step engine.
    """

    dt = cfg.dt
    mu = cfg.effective_mu()
    sigma = cfg.effective_sigma()
    sqrt_dt = math.sqrt(dt)
    diff_drift = (mu - 0.5 * sigma ** 2) * dt
    diff_vol = sigma * sqrt_dt
    model = cfg.model
    antithetic = cfg.variance_reduction == VR_ANTITHETIC

    def gauss(rng, n, state=None, step_i=None):
        # Prefer precomputed Sobol normals when the simulate loop attached them.
        if (
            state is not None
            and isinstance(state, dict)
            and "sobol_z" in state
            and step_i is not None
        ):
            return state["sobol_z"][:, int(step_i) - 1]
        return _draw_gauss(rng, n, antithetic)

    # ---- v2 engine (opt-in): route supported models through mc_engine's ----
    # StochasticProcess objects. The adapter reuses this loop's ``gauss``
    # closure, so RNG draw order — and every output — is bit-identical to the
    # legacy closures below. Unsupported models fall through to legacy.
    if _effective_engine(cfg) == ENGINE_V2:
        try:
            from mc_engine import legacy_engine_from_config
        except ImportError:  # pragma: no cover - mc_engine ships with the repo
            pass
        else:
            adapted = legacy_engine_from_config(cfg, gauss)
            if adapted is not None:
                return adapted

    if model == MODEL_GBM:
        def init_chunk(rng, n):
            return None

        def step(rng, n, state, i):
            return diff_drift + diff_vol * gauss(rng, n, state, i)
        return init_chunk, step

    if model == MODEL_STUDENT_T:
        df = float(cfg.t_df)
        scale = math.sqrt((df - 2.0) / df)  # standardize t to unit variance

        def init_chunk(rng, n):
            return None

        def step(rng, n, state, i):
            # Student-t keeps its own heavy-tail draws; Sobol is not applied here.
            shock = rng.standard_t(df, size=n) * scale
            return diff_drift + diff_vol * shock
        return init_chunk, step

    if model in BOOTSTRAP_MODELS:
        r = np.asarray(cfg.historical_returns, dtype=np.float64).ravel()
        emp_mean = float(np.mean(r))
        centered = r - emp_mean
        n_hist = centered.size
        vol_mult = cfg.stress_vol_multiplier if cfg.stress_enabled else 1.0
        target_daily = _bootstrap_daily_drift(cfg, emp_mean)

        if model == MODEL_HIST_BOOTSTRAP:
            def init_chunk(rng, n):
                return None

            def step(rng, n, state, i):
                idx = rng.integers(0, n_hist, size=n)
                return centered[idx] * vol_mult + target_daily
            return init_chunk, step

        block_len = int(cfg.block_length)

        def init_chunk(rng, n):
            return {
                "cur": np.zeros(n, dtype=np.int64),
                "rem": np.zeros(n, dtype=np.int64),
            }

        def step(rng, n, state, i):
            cur = state["cur"]
            rem = state["rem"]
            new_mask = rem <= 0
            if new_mask.any():
                starts = rng.integers(0, n_hist, size=n)
                cur[new_mask] = starts[new_mask]
                rem[new_mask] = block_len
            read = centered[cur % n_hist]
            cur += 1
            rem -= 1
            return read * vol_mult + target_daily
        return init_chunk, step

    if model == MODEL_MERTON:
        lam_dt = cfg.jump_intensity * dt
        jm = float(cfg.jump_mean)
        jv = float(cfg.jump_vol)
        k = math.exp(jm + 0.5 * jv ** 2) - 1.0  # expected proportional jump
        comp_drift = (mu - 0.5 * sigma ** 2 - cfg.jump_intensity * k) * dt

        def init_chunk(rng, n):
            return None

        def step(rng, n, state, i):
            z = rng.standard_normal(n)
            n_jumps = rng.poisson(lam_dt, size=n)
            # Sum of n_jumps iid N(jm, jv^2) ~ N(n_jumps*jm, n_jumps*jv^2).
            jump = n_jumps * jm + jv * np.sqrt(n_jumps) * rng.standard_normal(n)
            return comp_drift + diff_vol * z + jump
        return init_chunk, step

    if model == MODEL_REGIME:
        preset = REGIME_PRESETS[cfg.regime_preset]
        mu_f = np.asarray(preset["mu_factors"], dtype=np.float64)
        sig_f = np.asarray(preset["sigma_factors"], dtype=np.float64)
        cum_p = np.cumsum(np.asarray(preset["transition"], dtype=np.float64), axis=1)
        n_regimes = cum_p.shape[0]
        regime_mu = mu * mu_f
        regime_sig = sigma * sig_f
        regime_drift = (regime_mu - 0.5 * regime_sig ** 2) * dt
        regime_vol = regime_sig * sqrt_dt

        def init_chunk(rng, n):
            return {"regime": np.zeros(n, dtype=np.int64)}  # start in "normal"

        def step(rng, n, state, i):
            reg = state["regime"]
            u = rng.random(n)
            new_reg = (u[:, None] >= cum_p[reg]).sum(axis=1)
            np.clip(new_reg, 0, n_regimes - 1, out=new_reg)
            state["regime"] = new_reg
            z = rng.standard_normal(n)
            return regime_drift[new_reg] + regime_vol[new_reg] * z
        return init_chunk, step

    if model == MODEL_HESTON:
        kappa = float(cfg.heston_kappa)
        theta = float(cfg.heston_theta) if cfg.heston_theta is not None else sigma ** 2
        xi = float(cfg.heston_xi)
        rho = float(cfg.heston_rho)
        v0 = float(cfg.heston_v0) if cfg.heston_v0 is not None else sigma ** 2
        rho_c = math.sqrt(max(0.0, 1.0 - rho ** 2))

        def init_chunk(rng, n):
            return {"v": np.full(n, v0, dtype=np.float64)}

        def step(rng, n, state, i):
            v = state["v"]
            # Full-truncation: use max(v, 0) wherever variance enters.
            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos)
            z1 = gauss(rng, n, state, i)
            z2 = rng.standard_normal(n)
            zv = rho * z1 + rho_c * z2  # correlated shock for the variance SDE
            # Variance update (full-truncation Euler) -- never stays negative.
            v_new = (v + kappa * (theta - v_pos) * dt
                     + xi * sqrt_v * sqrt_dt * zv)
            state["v"] = np.maximum(v_new, 0.0)
            # Log-price increment uses the (truncated) current variance.
            return (mu - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z1
        return init_chunk, step

    if model == MODEL_GARCH:
        alpha = float(cfg.garch_alpha)
        beta = float(cfg.garch_beta)
        sigma_daily2 = (sigma ** 2) * dt  # long-run per-step variance target
        if cfg.garch_omega is not None:
            omega = float(cfg.garch_omega)
        else:
            omega = sigma_daily2 * (1.0 - alpha - beta)
        mu_step = mu * dt

        def init_chunk(rng, n):
            return {"var": np.full(n, sigma_daily2, dtype=np.float64)}

        def step(rng, n, state, i):
            var = state["var"]                       # current per-step variance
            var = np.maximum(var, 1e-300)            # keep strictly positive
            vol = np.sqrt(var)
            z = gauss(rng, n, state, i)
            shock = vol * z                          # mean-zero return shock
            log_ret = mu_step - 0.5 * var + shock
            # GARCH(1,1) recursion on the realized shock.
            state["var"] = omega + alpha * shock ** 2 + beta * var
            return log_ret
        return init_chunk, step

    if model == MODEL_KOU:
        lam_dt = cfg.kou_intensity * dt
        p_up = float(cfg.kou_p_up)
        eta_up = float(cfg.kou_eta_up)
        eta_down = float(cfg.kou_eta_down)
        # Compensator E[e^J - 1] for the double-exponential jump.
        k = (p_up * eta_up / (eta_up - 1.0)
             + (1.0 - p_up) * eta_down / (eta_down + 1.0)) - 1.0
        comp_drift = (mu - 0.5 * sigma ** 2 - cfg.kou_intensity * k) * dt

        def init_chunk(rng, n):
            return None

        def step(rng, n, state, i):
            z = gauss(rng, n)
            n_jumps = rng.poisson(lam_dt, size=n)
            total = int(n_jumps.sum())
            jump = np.zeros(n, dtype=np.float64)
            if total > 0:
                # One double-exponential draw per individual jump, summed per path.
                up_mask = rng.random(total) < p_up
                mags = np.empty(total, dtype=np.float64)
                mags[up_mask] = rng.exponential(1.0 / eta_up, size=int(up_mask.sum()))
                down = ~up_mask
                mags[down] = -rng.exponential(1.0 / eta_down, size=int(down.sum()))
                # Scatter-add each jump back to its path.
                path_idx = np.repeat(np.arange(n), n_jumps)
                np.add.at(jump, path_idx, mags)
            return comp_drift + diff_vol * z + jump
        return init_chunk, step

    raise ValueError(f"Unsupported model: {model!r}")


class DrawdownObserver:
    """Streaming drawdown / ruin / underwater-duration tracker (Phase 5).

    Extracted from the previously inlined accumulators in :func:`simulate`;
    the per-step arithmetic and its order are preserved exactly, so results
    are bit-identical. Duck-types the ``mc_engine.PathPricer`` streaming
    protocol (``begin_chunk`` / ``observe`` / ``end_chunk``) so the same
    observer runs in the v2 ``PathGenerator`` — option runs can now report
    drawdown metrics for free. All state is chunk-local.
    """

    def __init__(self, ruin_level: float, drawdown_threshold: float):
        self.ruin_level = float(ruin_level)
        self.drawdown_threshold = float(drawdown_threshold)
        self.drawdown_hits = 0
        self.ruin_hits = 0
        self.dd_duration_sum = 0.0
        self.dd_depth_sum = 0.0
        self._max_dd_chunks: List[np.ndarray] = []

    def begin_chunk(self, prices0: np.ndarray) -> None:
        self._running_max = prices0.copy()
        self._max_dd = np.zeros(prices0.size, dtype=np.float64)
        self._cur_uw = np.zeros(prices0.size, dtype=np.int64)
        self._max_uw = np.zeros(prices0.size, dtype=np.int64)
        self._ruin_hit = prices0 <= self.ruin_level

    def observe(self, step_i: int, prices: np.ndarray) -> None:
        np.maximum(self._running_max, prices, out=self._running_max)
        dd = 1.0 - prices / self._running_max
        np.maximum(self._max_dd, dd, out=self._max_dd)
        self._ruin_hit |= prices <= self.ruin_level
        underwater = prices < self._running_max
        self._cur_uw = np.where(underwater, self._cur_uw + 1, 0)
        np.maximum(self._max_uw, self._cur_uw, out=self._max_uw)

    def end_chunk(self, prices: np.ndarray) -> None:
        self.drawdown_hits += int(
            np.count_nonzero(self._max_dd >= self.drawdown_threshold))
        self.ruin_hits += int(np.count_nonzero(self._ruin_hit))
        self.dd_duration_sum += float(self._max_uw.sum())
        self.dd_depth_sum += float(self._max_dd.sum())
        self._max_dd_chunks.append(self._max_dd)

    def values(self) -> np.ndarray:
        """Per-path maximum drawdown fraction (PathPricer contract)."""
        if not self._max_dd_chunks:
            return np.asarray([], dtype=np.float64)
        return np.concatenate(self._max_dd_chunks)


class SampleRecorder:
    """Streaming recorder for the bounded sample-trajectory block (Phase 5).

    Fills the first ``n_sample`` paths' full trajectories, chunk by chunk —
    the only deliberately retained 2-D array, exactly as ``simulate`` always
    did. Duck-types the PathPricer protocol; reusable on the v2 pipeline.
    """

    def __init__(self, n_sample: int, steps: int, s0: float):
        self.n_sample = int(n_sample)
        self.steps = int(steps)
        self.matrix = np.empty((self.n_sample, self.steps + 1),
                               dtype=np.float64)
        if self.n_sample:
            self.matrix[:, 0] = float(s0)
        self._produced = 0
        self._take = 0

    def begin_chunk(self, prices0: np.ndarray) -> None:
        self._take = max(0, min(self.n_sample - self._produced, prices0.size))

    def observe(self, step_i: int, prices: np.ndarray) -> None:
        if self._take:
            self.matrix[self._produced:self._produced + self._take, step_i] = (
                prices[:self._take]
            )

    def end_chunk(self, prices: np.ndarray) -> None:
        self._produced += prices.size

    def values(self) -> np.ndarray:
        """Terminal sample values (PathPricer contract)."""
        return self.matrix[:, -1] if self.n_sample else np.asarray([])


def simulate(config: SimulationConfig) -> SimulationResult:
    """Run a chunked Monte Carlo simulation for the configured model.

    The simulation evolves each chunk one step at a time, so the largest 2-D
    array ever allocated is the bounded sample-trajectory block -- never the full
    ``paths x steps`` matrix.  This holds for every model (GBM, Student-t,
    bootstrap, block bootstrap, Merton jumps, regime switching) because each
    model's per-step state is bounded by the chunk size.

    Variance reduction (wired into this loop)
    -----------------------------------------
    * ``antithetic`` -- mirrored Gaussian shocks per chunk (step engine).
    * ``sobol`` -- when SciPy is available, each chunk draws a Sobol normal
      matrix of shape ``(chunk, steps)`` (still O(chunk × steps), never
      O(paths × steps) for the full run's working set beyond the sample block).
      Falls back to plain MC with a stats warning if SciPy is missing.
    * ``control_variate`` -- after the path loop, for GBM-family terminal
      values, adjusts the *reported mean* using the known analytic E[S_T]
      (raw samples kept for VaR/percentiles so risk metrics stay distributional).
    """

    cfg = config.validate()
    rng = np.random.default_rng(cfg.seed)

    steps = cfg.horizon
    init_chunk, step_fn = _build_step_engine(cfg)

    # Resolve effective VR (Sobol may fall back).
    vr_requested = cfg.variance_reduction
    vr_effective = vr_requested
    vr_notes: List[str] = []
    use_sobol = False
    if vr_requested == VR_SOBOL:
        if sobol_available():
            use_sobol = True
        else:
            vr_effective = VR_NONE
            vr_notes.append(
                "Sobol QMC requested but SciPy is unavailable; fell back to plain MC."
            )

    # Deterministic one-day crash applied on the first step (any model).
    crash_log = (
        math.log(1.0 - cfg.stress_crash_pct)
        if cfg.stress_enabled and cfg.stress_crash_pct > 0
        else 0.0
    )
    dd_threshold = cfg.drawdown_threshold
    ruin_level = cfg.ruin_threshold * cfg.s0
    # Streaming observers (extracted from the loop in Phase 5; identical
    # arithmetic, and reusable on the v2 PathGenerator pipeline).
    dd_observer = DrawdownObserver(ruin_level, dd_threshold)

    final_values = np.empty(cfg.paths, dtype=np.float64)
    # Gross (pre-cost) terminal prices — used for control-variate post-process.
    gross_terminals = np.empty(cfg.paths, dtype=np.float64)

    n_sample = min(cfg.sample_paths, cfg.paths)
    sampler = SampleRecorder(n_sample, steps, cfg.s0)
    sample_trajectories = sampler.matrix

    memory = MemoryInfo(
        paths=cfg.paths, horizon=cfg.horizon, chunk_size=cfg.chunk_size
    )
    # The sample block is the only 2-D array we deliberately keep long-term.
    # Sobol attaches a temporary (chunk x steps) shock matrix per chunk only.
    memory.peak_matrix_elements = max(memory.peak_matrix_elements, n_sample * (steps + 1))
    if use_sobol:
        memory.peak_matrix_elements = max(
            memory.peak_matrix_elements,
            min(cfg.chunk_size, cfg.paths) * steps,
        )

    start = time.perf_counter()
    produced = 0
    chunk_idx = 0
    while produced < cfg.paths:
        this_chunk = min(cfg.chunk_size, cfg.paths - produced)

        # Current price vector for this chunk (1-D, bounded by chunk size).
        prices = np.full(this_chunk, cfg.s0, dtype=np.float64)
        memory.peak_vector_elements = max(memory.peak_vector_elements, prices.size)

        # Per-chunk model state (bounded by chunk size).
        state = init_chunk(rng, this_chunk)
        if use_sobol:
            # Distinct seed per chunk keeps streams independent while remaining
            # reproducible for a fixed global seed.
            sobol_seed = None if cfg.seed is None else int(cfg.seed) + chunk_idx * 1_000_003
            state = _attach_sobol_shocks(state, this_chunk, steps, sobol_seed)

        dd_observer.begin_chunk(prices)
        sampler.begin_chunk(prices)

        for step in range(1, steps + 1):
            log_ret = step_fn(rng, this_chunk, state, step)
            if step == 1 and crash_log:
                log_ret = log_ret + crash_log
            prices *= np.exp(log_ret)
            dd_observer.observe(step, prices)
            sampler.observe(step, prices)

        dd_observer.end_chunk(prices)
        sampler.end_chunk(prices)
        gross = prices
        net = apply_costs(gross, cfg.s0, cfg.cost)
        final_values[produced:produced + this_chunk] = net
        gross_terminals[produced:produced + this_chunk] = gross
        produced += this_chunk
        chunk_idx += 1

    runtime = time.perf_counter() - start

    # ---- Control variate post-process (GBM terminal mean) ---------------
    # Keep raw final_values for distributional risk metrics (VaR, percentiles).
    # Report a CV-adjusted expected value when requested and model is GBM.
    cv_beta = None
    cv_mean = None
    if vr_requested == VR_CONTROL:
        if cfg.model == MODEL_GBM and not cfg.stress_enabled:
            T = steps * cfg.dt
            mu_eff = cfg.effective_mu()
            analytic_ST = cfg.s0 * math.exp(mu_eff * T)
            # Map analytic gross E[S_T] through the same cost model.
            if cfg.cost > 0.0:
                analytic_net = analytic_ST * (1.0 - cfg.cost) - cfg.s0 * cfg.cost
            else:
                analytic_net = analytic_ST
            g = gross_terminals
            y = final_values
            if cfg.paths > 1 and float(np.var(g)) > 0.0:
                cov_yg = float(np.cov(y, g, ddof=1)[0, 1])
                var_g = float(np.var(g, ddof=1))
                cv_beta = cov_yg / var_g
                # Unbiased control: E[g] = analytic_ST under GBM.
                adjusted = y - cv_beta * (g - analytic_ST)
                cv_mean = float(np.mean(adjusted))
            else:
                cv_beta = 0.0
                cv_mean = float(np.mean(y))
            vr_effective = VR_CONTROL
            vr_notes.append(
                f"Control variate applied to expected value (beta={cv_beta:.4f}); "
                "VaR/percentiles use raw samples."
            )
        else:
            vr_effective = VR_NONE
            vr_notes.append(
                "Control variate is implemented for unstressed GBM only; "
                "fell back to plain MC mean."
            )

    convergence_paths, convergence_means = _convergence_curve(
        final_values, cfg.convergence_points
    )
    stats = compute_statistics(
        final_values, cfg.s0, runtime=runtime,
        drawdown_prob=dd_observer.drawdown_hits / cfg.paths,
        drawdown_threshold=dd_threshold,
    )
    if cv_mean is not None:
        stats["expected_value_raw"] = stats["expected_value"]
        stats["expected_value"] = cv_mean
        stats["expected_return"] = (cv_mean / cfg.s0) - 1.0
        stats["control_variate_beta"] = cv_beta
        stats["control_variate_mean"] = cv_mean
    stats["model"] = cfg.model
    stats["drift_mode"] = cfg.drift_mode
    # Phase 5: the engine is always recorded (additive schema change shipped
    # together with the v2 default flip).
    if _effective_engine(cfg) == ENGINE_V2:
        try:
            from mc_engine import V2_SUPPORTED_MODELS
            stats["engine"] = (
                ENGINE_V2 if cfg.model in V2_SUPPORTED_MODELS
                else f"{ENGINE_LEGACY} (v2 requested; model not ported)"
            )
        except ImportError:  # pragma: no cover
            stats["engine"] = ENGINE_LEGACY
    else:
        stats["engine"] = ENGINE_LEGACY
    stats["variance_reduction"] = vr_requested
    stats["variance_reduction_effective"] = vr_effective
    if vr_notes:
        stats["variance_reduction_notes"] = vr_notes
    # Path-based risk metrics (averaged over paths).
    stats["mean_max_drawdown"] = dd_observer.dd_depth_sum / cfg.paths
    stats["mean_drawdown_duration"] = dd_observer.dd_duration_sum / cfg.paths
    stats["prob_ruin"] = dd_observer.ruin_hits / cfg.paths
    stats["ruin_threshold"] = cfg.ruin_threshold
    _augment_risk_metrics(stats, cfg)

    return SimulationResult(
        config=cfg,
        final_values=final_values,
        sample_trajectories=sample_trajectories,
        convergence_paths=convergence_paths,
        convergence_means=convergence_means,
        stats=stats,
        memory=memory,
        runtime_seconds=runtime,
    )


def _convergence_curve(final_values: np.ndarray, points: int):
    """Running mean of the net ending value as paths accumulate.

    Computed from the already-stored final values, so it adds no per-step memory
    cost.  Uses a cumulative sum (1-D, length ``paths``) sampled at log-spaced
    indices to show Monte Carlo convergence cheaply.
    """

    n = final_values.size
    if n == 0:
        return np.array([]), np.array([])
    points = max(2, min(points, n))
    idx = np.unique(
        np.geomspace(1, n, num=points).astype(np.int64)
    )
    idx = idx[idx >= 1]
    cumsum = np.cumsum(final_values)
    means = cumsum[idx - 1] / idx
    return idx, means


# ---------------------------------------------------------------------------
# Statistics: expected/median, P(profit/loss), VaR, Expected Shortfall
# ---------------------------------------------------------------------------


def value_at_risk(pnl: np.ndarray, confidence: float) -> float:
    """VaR as a positive loss in value terms at the given confidence (percent).

    ``pnl`` is profit-and-loss per unit (net ending value minus ``s0``).
    VaR_95 = -(5th percentile of P&L); a positive number means an expected loss.
    """

    alpha = 100.0 - confidence
    threshold = np.percentile(pnl, alpha)
    return float(-threshold)


def expected_shortfall(pnl: np.ndarray, confidence: float) -> float:
    """Expected Shortfall (a.k.a. CVaR): mean loss in the worst tail.

    Returns a positive number representing the average loss conditional on being
    at or beyond the VaR threshold.
    """

    alpha = 100.0 - confidence
    threshold = np.percentile(pnl, alpha)
    tail = pnl[pnl <= threshold]
    if tail.size == 0:
        # Not enough samples to populate the tail -> use the worst observation.
        return float(-np.min(pnl))
    return float(-np.mean(tail))


KELLY_WARNING = (
    "Kelly fraction is a theoretical estimate only; it assumes the simulated "
    "return distribution is exactly correct and ignores estimation error. Do "
    "NOT use it directly for position sizing."
)


def _augment_risk_metrics(stats: Dict[str, Any], cfg: "SimulationConfig") -> None:
    """Annualize ratios and add Calmar using the horizon length from ``cfg``."""
    years = cfg.horizon * cfg.dt
    ann = math.sqrt(1.0 / years) if years > 0 else 0.0
    stats["sharpe_annual"] = stats.get("sharpe", 0.0) * ann
    stats["sortino_annual"] = stats.get("sortino", 0.0) * ann

    exp_ret = stats.get("expected_return", 0.0)
    if years > 0 and (1.0 + exp_ret) > 0:
        annualized_return = (1.0 + exp_ret) ** (1.0 / years) - 1.0
    else:
        annualized_return = exp_ret
    stats["annualized_return"] = annualized_return

    mdd = stats.get("mean_max_drawdown", 0.0)
    stats["calmar"] = (annualized_return / mdd) if mdd > 0 else 0.0
    stats["kelly_warning"] = KELLY_WARNING


def compute_statistics(
    final_values: np.ndarray,
    s0: float,
    *,
    runtime: Optional[float] = None,
    drawdown_prob: Optional[float] = None,
    drawdown_threshold: float = DRAWDOWN_THRESHOLD,
) -> Dict[str, Any]:
    """Compute the full statistics bundle from net ending values."""

    fv = np.asarray(final_values, dtype=np.float64)
    n = fv.size
    pnl = fv - s0
    returns = fv / s0 - 1.0

    expected_value = float(np.mean(fv))
    median_value = float(np.median(fv))
    prob_profit = float(np.mean(fv > s0))
    prob_loss = float(np.mean(fv < s0))

    # Probability buckets relative to the starting price.
    prob_gain_20 = float(np.mean(fv > s0 * 1.20))
    prob_loss_10 = float(np.mean(fv < s0 * 0.90))
    prob_loss_20 = float(np.mean(fv < s0 * 0.80))

    # Worst 1% average ending value (mean of the lowest 1% of outcomes).
    p1 = np.percentile(fv, 1)
    worst_tail = fv[fv <= p1]
    worst_1pct_avg = float(np.mean(worst_tail)) if worst_tail.size else float(np.min(fv))

    var = {}
    es = {}
    for level in RISK_LEVELS:
        key = _level_key(level)
        v = value_at_risk(pnl, level)
        e = expected_shortfall(pnl, level)
        var[key] = {"value": v, "pct": v / s0}
        es[key] = {"value": e, "pct": e / s0}

    percentiles = {
        str(p): float(np.percentile(fv, p)) for p in REPORT_PERCENTILES
    }

    # Return-distribution risk ratios (over the whole horizon period).
    ret_mean = float(np.mean(returns))
    ret_std = float(np.std(returns, ddof=1)) if n > 1 else 0.0
    downside = returns[returns < 0.0]
    downside_std = float(np.sqrt(np.mean(downside ** 2))) if downside.size else 0.0
    sharpe = ret_mean / ret_std if ret_std > 0 else 0.0
    sortino = ret_mean / downside_std if downside_std > 0 else 0.0
    kelly = ret_mean / (ret_std ** 2) if ret_std > 0 else 0.0

    stats: Dict[str, Any] = {
        "paths": int(n),
        "s0": float(s0),
        "expected_value": expected_value,
        "median_value": median_value,
        "expected_return": expected_value / s0 - 1.0,
        "std_value": float(np.std(fv, ddof=1)) if n > 1 else 0.0,
        "min_value": float(np.min(fv)),
        "max_value": float(np.max(fv)),
        "prob_profit": prob_profit,
        "prob_loss": prob_loss,
        "prob_gain_20": prob_gain_20,
        "prob_loss_10": prob_loss_10,
        "prob_loss_20": prob_loss_20,
        "worst_1pct_avg_value": worst_1pct_avg,
        "drawdown_threshold": float(drawdown_threshold),
        "mean_return": ret_mean,
        "return_std": ret_std,
        "downside_std": downside_std,
        "sharpe": sharpe,
        "sortino": sortino,
        "kelly_fraction": kelly,
        "var": var,
        "expected_shortfall": es,
        "percentiles": percentiles,
    }
    if drawdown_prob is not None:
        stats["prob_drawdown"] = float(drawdown_prob)
    if runtime is not None:
        stats["runtime_seconds"] = float(runtime)
        stats["paths_per_second"] = float(n / runtime) if runtime > 0 else float("inf")
    return stats


def _level_key(level: float) -> str:
    """Stable string key for a confidence level (e.g. 99.9 -> '99.9')."""
    if float(level).is_integer():
        return str(int(level))
    return ("%g" % level)


# ---------------------------------------------------------------------------
# Reporting / export
# ---------------------------------------------------------------------------


def _volatility_source(cfg: SimulationConfig) -> str:
    if cfg.model in BOOTSTRAP_MODELS:
        src = "empirical daily returns (bootstrap)"
    else:
        src = "annualized historical estimate"
    if cfg.stress_enabled and cfg.stress_vol_multiplier != 1.0:
        src += f" x{cfg.stress_vol_multiplier:g} stress"
    return src


def model_assumptions(cfg: SimulationConfig,
                      market: Optional[MarketParameters] = None) -> Dict[str, Any]:
    """Human/machine readable description of the selected model and its inputs."""

    a: Dict[str, Any] = {
        "model": cfg.model,
        "drift_mode": cfg.drift_mode,
        "effective_mu_annual": cfg.effective_mu(),
        "effective_sigma_annual": cfg.effective_sigma(),
        "volatility_source": _volatility_source(cfg),
        "stress_enabled": cfg.stress_enabled,
    }
    if cfg.drift_mode == DRIFT_MANUAL:
        a["manual_drift"] = cfg.manual_drift
    if cfg.model == MODEL_STUDENT_T:
        a["t_df"] = cfg.t_df
    if cfg.model == MODEL_BLOCK_BOOTSTRAP:
        a["block_length"] = cfg.block_length
    if cfg.model in BOOTSTRAP_MODELS:
        n = 0 if cfg.historical_returns is None else int(np.asarray(cfg.historical_returns).size)
        a["historical_returns_count"] = n
    if cfg.model == MODEL_MERTON:
        a["jump_intensity"] = cfg.jump_intensity
        a["jump_mean"] = cfg.jump_mean
        a["jump_vol"] = cfg.jump_vol
    if cfg.model == MODEL_REGIME:
        a["regime_preset"] = cfg.regime_preset
    if cfg.model == MODEL_HESTON:
        a["heston"] = {
            "kappa": cfg.heston_kappa,
            "theta": cfg.heston_theta if cfg.heston_theta is not None else cfg.effective_sigma() ** 2,
            "xi": cfg.heston_xi,
            "rho": cfg.heston_rho,
            "v0": cfg.heston_v0 if cfg.heston_v0 is not None else cfg.effective_sigma() ** 2,
        }
    if cfg.model == MODEL_GARCH:
        a["garch"] = {
            "omega": cfg.garch_omega,
            "alpha": cfg.garch_alpha,
            "beta": cfg.garch_beta,
        }
    if cfg.model == MODEL_KOU:
        a["kou"] = {
            "intensity": cfg.kou_intensity,
            "p_up": cfg.kou_p_up,
            "eta_up": cfg.kou_eta_up,
            "eta_down": cfg.kou_eta_down,
        }
    a["variance_reduction"] = cfg.variance_reduction
    a["math_model_version"] = MATH_MODEL_VERSION
    a["stress"] = {
        "enabled": cfg.stress_enabled,
        "one_day_crash_pct": cfg.stress_crash_pct,
        "vol_multiplier": cfg.stress_vol_multiplier,
        "drift_haircut": cfg.stress_drift_haircut,
    }
    if market is not None:
        a["data_source"] = market.source
    return a


def build_report(
    result: SimulationResult,
    market: Optional[MarketParameters] = None,
    *,
    evt: Optional[Dict[str, Any]] = None,
    portfolio: Optional[Dict[str, Any]] = None,
    backtest: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Assemble a serialisable report dict from a simulation result.

    Optional ``evt``, ``portfolio``, ``backtest`` and ``warnings`` sections are
    included when supplied so exports carry the full risk-lab metadata.
    """

    cfg = result.config
    if warnings is None:
        warnings = collect_warnings(cfg, market, evt=evt, backtest=backtest)
    report: Dict[str, Any] = {
        "math_model_version": MATH_MODEL_VERSION,
        "config": {
            "ticker": cfg.ticker,
            "s0": cfg.s0,
            "paths": cfg.paths,
            "horizon": cfg.horizon,
            "mu": cfg.mu,
            "sigma": cfg.sigma,
            "dt": cfg.dt,
            "chunk_size": cfg.chunk_size,
            "seed": cfg.seed,
            "cost": cfg.cost,
            "model": cfg.model,
            "drift_mode": cfg.drift_mode,
            "variance_reduction": cfg.variance_reduction,
            "ruin_threshold": cfg.ruin_threshold,
        },
        "model": model_assumptions(cfg, market),
        "variance_reduction_method": cfg.variance_reduction,
        "statistics": result.stats,
        "warnings": warnings,
        "memory": {
            "paths": result.memory.paths,
            "horizon": result.memory.horizon,
            "chunk_size": result.memory.chunk_size,
            "peak_matrix_elements": result.memory.peak_matrix_elements,
            "peak_vector_elements": result.memory.peak_vector_elements,
            "full_matrix_elements": result.memory.full_matrix_elements,
            "full_matrix_bytes": result.memory.full_matrix_bytes,
            "peak_bytes": result.memory.peak_bytes,
            "is_chunk_safe": result.memory.is_chunk_safe,
            "status": result.memory.status(),
        },
        "runtime_seconds": result.runtime_seconds,
    }
    if market is not None:
        report["market"] = {"source": market.source, "note": market.note}
    if evt is not None:
        report["evt"] = evt
    if portfolio is not None:
        report["portfolio"] = {k: v for k, v in portfolio.items()
                               if k != "portfolio_values"}
    if backtest is not None:
        report["backtest"] = backtest
    return report


def report_to_json(result: SimulationResult, market: Optional[MarketParameters] = None,
                   *, indent: int = 2, **kwargs) -> str:
    """Serialise the full report to a JSON string."""
    return json.dumps(build_report(result, market, **kwargs),
                      indent=indent, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def summary_rows(result: SimulationResult) -> List[List[Any]]:
    """Flat ``[metric, value]`` rows used for the CSV summary."""

    s = result.stats
    cfg = result.config
    assumptions = model_assumptions(cfg)
    rows: List[List[Any]] = [
        ["ticker", cfg.ticker],
        ["starting_price", cfg.s0],
        ["paths", cfg.paths],
        ["horizon_steps", cfg.horizon],
        ["model", cfg.model],
        ["math_model_version", MATH_MODEL_VERSION],
        ["drift_mode", cfg.drift_mode],
        ["variance_reduction_method", cfg.variance_reduction],
        ["volatility_source", assumptions["volatility_source"]],
        ["effective_mu_annual", assumptions["effective_mu_annual"]],
        ["effective_sigma_annual", assumptions["effective_sigma_annual"]],
        ["mu_annual", cfg.mu],
        ["sigma_annual", cfg.sigma],
        ["chunk_size", cfg.chunk_size],
        ["seed", cfg.seed],
        ["transaction_cost", cfg.cost],
        # Model-specific parameters.
        ["t_df", cfg.t_df],
        ["block_length", cfg.block_length],
        ["jump_intensity", cfg.jump_intensity],
        ["jump_mean", cfg.jump_mean],
        ["jump_vol", cfg.jump_vol],
        ["regime_preset", cfg.regime_preset],
        ["heston_kappa", cfg.heston_kappa],
        ["heston_theta", cfg.heston_theta],
        ["heston_xi", cfg.heston_xi],
        ["heston_rho", cfg.heston_rho],
        ["heston_v0", cfg.heston_v0],
        ["garch_omega", cfg.garch_omega],
        ["garch_alpha", cfg.garch_alpha],
        ["garch_beta", cfg.garch_beta],
        ["kou_intensity", cfg.kou_intensity],
        ["kou_p_up", cfg.kou_p_up],
        ["kou_eta_up", cfg.kou_eta_up],
        ["kou_eta_down", cfg.kou_eta_down],
        ["ruin_threshold", cfg.ruin_threshold],
        # Stress overlay.
        ["stress_enabled", cfg.stress_enabled],
        ["stress_one_day_crash_pct", cfg.stress_crash_pct],
        ["stress_vol_multiplier", cfg.stress_vol_multiplier],
        ["stress_drift_haircut", cfg.stress_drift_haircut],
        # Core outcome metrics.
        ["expected_ending_value", s["expected_value"]],
        ["median_ending_value", s["median_value"]],
        ["expected_return", s["expected_return"]],
        ["std_ending_value", s["std_value"]],
        ["min_ending_value", s["min_value"]],
        ["max_ending_value", s["max_value"]],
        ["prob_profit", s["prob_profit"]],
        ["prob_loss", s["prob_loss"]],
        # Probability buckets.
        ["prob_gain_more_than_20pct", s["prob_gain_20"]],
        ["prob_loss_more_than_10pct", s["prob_loss_10"]],
        ["prob_loss_more_than_20pct", s["prob_loss_20"]],
        ["prob_drawdown_50pct", s.get("prob_drawdown", "")],
        ["worst_1pct_avg_ending_value", s["worst_1pct_avg_value"]],
        # Advanced risk metrics.
        ["mean_max_drawdown", s.get("mean_max_drawdown", "")],
        ["mean_drawdown_duration", s.get("mean_drawdown_duration", "")],
        ["prob_ruin", s.get("prob_ruin", "")],
        ["sharpe_annual", s.get("sharpe_annual", "")],
        ["sortino_annual", s.get("sortino_annual", "")],
        ["calmar", s.get("calmar", "")],
        ["annualized_return", s.get("annualized_return", "")],
        ["kelly_fraction", s.get("kelly_fraction", "")],
    ]
    for level in RISK_LEVELS:
        key = _level_key(level)
        rows.append([f"VaR_{key}_value", s["var"][key]["value"]])
        rows.append([f"VaR_{key}_pct", s["var"][key]["pct"]])
    for level in RISK_LEVELS:
        key = _level_key(level)
        rows.append([f"ES_{key}_value", s["expected_shortfall"][key]["value"]])
        rows.append([f"ES_{key}_pct", s["expected_shortfall"][key]["pct"]])
    for p, v in s["percentiles"].items():
        rows.append([f"percentile_{p}", v])
    rows.append(["runtime_seconds", s.get("runtime_seconds", result.runtime_seconds)])
    rows.append(["paths_per_second", s.get("paths_per_second", "")])
    rows.append(["peak_matrix_elements", result.memory.peak_matrix_elements])
    rows.append(["full_matrix_elements", result.memory.full_matrix_elements])
    rows.append(["chunk_safe", result.memory.is_chunk_safe])
    return rows


def report_to_csv(result: SimulationResult) -> str:
    """Serialise the summary to a CSV string (``metric,value`` rows)."""

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["metric", "value"])
    for row in summary_rows(result):
        writer.writerow(row)
    return buffer.getvalue()


def write_csv(result: SimulationResult, path: str) -> str:
    text = report_to_csv(result)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(text)
    return path


def write_json(result: SimulationResult, path: str,
               market: Optional[MarketParameters] = None, **kwargs) -> str:
    text = report_to_json(result, market, **kwargs)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# ---------------------------------------------------------------------------
# Path-mode helpers (shared between CLI and GUI)
# ---------------------------------------------------------------------------


def resolve_path_mode(mode: str, explicit_paths: Optional[int] = None) -> int:
    """Translate a path-mode label into a concrete path count.

    Preset modes (Preview/Standard/Serious) ignore ``explicit_paths`` unless
    none of the presets apply.  ``Custom`` and ``Tail-risk (advanced)`` require
    ``explicit_paths`` and validate it against their respective safe ranges.
    """

    if mode not in PATH_MODES:
        raise ValueError(f"Unknown path mode: {mode!r}")
    preset = PATH_MODES[mode]
    if preset is not None:
        return preset
    if explicit_paths is None:
        raise ValueError(f"{mode} mode requires an explicit path count")
    explicit_paths = int(explicit_paths)
    if mode == "Custom":
        if not (CUSTOM_MIN_PATHS <= explicit_paths <= CUSTOM_MAX_PATHS):
            raise ValueError(
                f"Custom paths must be between {CUSTOM_MIN_PATHS:,} and "
                f"{CUSTOM_MAX_PATHS:,}"
            )
        return explicit_paths
    # Tail-risk (advanced)
    if not (TAIL_RISK_MIN_PATHS <= explicit_paths <= TAIL_RISK_MAX_PATHS):
        raise ValueError(
            f"Tail-risk paths must be between {TAIL_RISK_MIN_PATHS:,} and "
            f"{TAIL_RISK_MAX_PATHS:,}"
        )
    return explicit_paths


def tail_risk_warning(paths: int) -> Optional[str]:
    """Return a warning string when the path count is in tail-risk territory."""

    if paths > PATH_MODES["Serious"]:
        return (
            f"Tail-risk advanced mode: {paths:,} paths. This is CPU/RAM intensive. "
            "Keep chunk size at 25,000-50,000 and expect a longer runtime."
        )
    return None


# ---------------------------------------------------------------------------
# Model comparison (run several models on identical settings, side-by-side)
# ---------------------------------------------------------------------------

# Columns reported in the comparison table, in display order.
COMPARISON_COLUMNS = (
    "model",
    "expected_ending_value",
    "median_ending_value",
    "prob_profit",
    "prob_loss",
    "prob_gain_20",
    "prob_loss_10",
    "prob_loss_20",
    "prob_drawdown_50",
    "percentile_5",
    "percentile_95",
    "var_99",
    "es_99",
    "evt_var_99",
    "evt_es_99",
    "max_drawdown_prob",
    "prob_ruin",
    "sharpe",
    "sortino",
    "disagreement_rank",
    "runtime_seconds",
    "chunk_safe",
)


def comparison_row(result: SimulationResult,
                   evt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Flatten a :class:`SimulationResult` into one comparison-table row.

    ``disagreement_rank`` is filled in later by :func:`compare_models` once all
    models are known.  ``evt`` (optional) supplies EVT 99% tail metrics.
    """

    s = result.stats
    evt_var_99 = float("nan")
    evt_es_99 = float("nan")
    if evt is not None and not evt.get("error"):
        evt_var_99 = evt["var"].get("99", float("nan"))
        evt_es_99 = evt["es"].get("99", float("nan"))
    return {
        "model": result.config.model,
        "expected_ending_value": s["expected_value"],
        "median_ending_value": s["median_value"],
        "prob_profit": s["prob_profit"],
        "prob_loss": s["prob_loss"],
        "prob_gain_20": s["prob_gain_20"],
        "prob_loss_10": s["prob_loss_10"],
        "prob_loss_20": s["prob_loss_20"],
        "prob_drawdown_50": s.get("prob_drawdown", float("nan")),
        "percentile_5": s["percentiles"]["5"],
        "percentile_95": s["percentiles"]["95"],
        "var_99": s["var"]["99"]["value"],
        "es_99": s["expected_shortfall"]["99"]["value"],
        "evt_var_99": evt_var_99,
        "evt_es_99": evt_es_99,
        "max_drawdown_prob": s.get("prob_drawdown", float("nan")),
        "prob_ruin": s.get("prob_ruin", float("nan")),
        "sharpe": s.get("sharpe_annual", s.get("sharpe", float("nan"))),
        "sortino": s.get("sortino_annual", s.get("sortino", float("nan"))),
        "disagreement_rank": 0,
        "runtime_seconds": s.get("runtime_seconds", result.runtime_seconds),
        "chunk_safe": result.memory.is_chunk_safe,
    }


@dataclass
class ComparisonReport:
    """Result of comparing several models on identical settings."""

    base_config: SimulationConfig
    rows: List[Dict[str, Any]]
    per_model: Dict[str, Dict[str, Any]]   # model -> {stats, memory, assumptions}
    most_conservative: Optional[str]
    market_source: Optional[str] = None

    @property
    def models(self) -> List[str]:
        return [r["model"] for r in self.rows]

    @property
    def all_chunk_safe(self) -> bool:
        return all(r["chunk_safe"] for r in self.rows)


def most_conservative_model(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Pick the most conservative model.

    Ranking favours the model with the highest probability of a >20% loss and,
    as a tie-breaker, the highest 99% Expected Shortfall.
    """

    if not rows:
        return None
    best = max(rows, key=lambda r: (r["prob_loss_20"], r["es_99"]))
    return best["model"]


def _assign_disagreement_ranks(rows: List[Dict[str, Any]]) -> None:
    """Rank models by how far they diverge from the cross-model consensus.

    For a set of key metrics, each model's robust z-distance from the median is
    averaged into a disagreement score; rank 1 == most divergent model.
    """

    if not rows:
        return
    metrics = ("expected_ending_value", "prob_loss_20", "es_99", "prob_drawdown_50")
    scores = np.zeros(len(rows), dtype=float)
    for metric in metrics:
        vals = np.array([float(r.get(metric, np.nan)) for r in rows], dtype=float)
        if not np.all(np.isfinite(vals)):
            continue
        med = np.median(vals)
        mad = np.median(np.abs(vals - med))
        scale = mad if mad > 1e-12 else (np.std(vals) if np.std(vals) > 1e-12 else 1.0)
        scores += np.abs(vals - med) / scale
    order = np.argsort(-scores)  # most divergent first
    for rank, idx in enumerate(order, start=1):
        rows[idx]["disagreement_rank"] = int(rank)
        rows[idx]["disagreement_score"] = float(scores[idx])


def compare_models(
    base_config: SimulationConfig,
    models: "Optional[List[str]]" = None,
    *,
    market: Optional[MarketParameters] = None,
    evt: bool = False,
) -> ComparisonReport:
    """Run several models on identical settings and assemble a comparison.

    Each model reuses the chunk-safe :func:`simulate` engine, so no run ever
    allocates a full ``paths x steps`` matrix.  ``base_config`` supplies the
    shared ticker/paths/horizon/chunk/seed/drift settings; only ``model`` (and
    its model-specific knobs already present on the config) varies per run.
    When ``evt`` is true an EVT tail-risk fit is added per model.
    """

    if models is None:
        models = list(MODELS)
    if not models:
        raise ValueError("compare_models requires at least one model")

    base = replace(base_config)  # shallow copy so we never mutate the caller's config
    rows: List[Dict[str, Any]] = []
    per_model: Dict[str, Dict[str, Any]] = {}

    for model in models:
        if model not in MODELS:
            raise ValueError(f"Unknown model: {model!r}")
        cfg = replace(base, model=model)
        result = simulate(cfg)
        evt_result = evt_from_result(result) if evt else None
        rows.append(comparison_row(result, evt=evt_result))
        per_model[model] = {
            "assumptions": model_assumptions(cfg, market),
            "statistics": result.stats,
            "evt": evt_result,
            "memory": {
                "is_chunk_safe": result.memory.is_chunk_safe,
                "peak_matrix_elements": result.memory.peak_matrix_elements,
                "full_matrix_elements": result.memory.full_matrix_elements,
                "peak_vector_elements": result.memory.peak_vector_elements,
                "status": result.memory.status(),
            },
        }

    _assign_disagreement_ranks(rows)

    return ComparisonReport(
        base_config=base,
        rows=rows,
        per_model=per_model,
        most_conservative=most_conservative_model(rows),
        market_source=(market.source if market is not None else None),
    )


def comparison_to_csv(report: ComparisonReport) -> str:
    """Serialise the comparison table to CSV (one row per model)."""

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(list(COMPARISON_COLUMNS))
    for row in report.rows:
        writer.writerow([row[col] for col in COMPARISON_COLUMNS])
    # Trailing annotation rows for shared context / headline takeaway.
    writer.writerow([])
    writer.writerow(["most_conservative_model", report.most_conservative])
    writer.writerow(["ticker", report.base_config.ticker])
    writer.writerow(["paths", report.base_config.paths])
    writer.writerow(["horizon", report.base_config.horizon])
    writer.writerow(["chunk_size", report.base_config.chunk_size])
    writer.writerow(["seed", report.base_config.seed])
    writer.writerow(["drift_mode", report.base_config.drift_mode])
    writer.writerow(["all_chunk_safe", report.all_chunk_safe])
    return buffer.getvalue()


def comparison_to_json(report: ComparisonReport, *, indent: int = 2) -> str:
    """Serialise the full comparison (table + per-model metadata) to JSON."""

    cfg = report.base_config
    payload = {
        "comparison": {
            "ticker": cfg.ticker,
            "paths": cfg.paths,
            "horizon": cfg.horizon,
            "chunk_size": cfg.chunk_size,
            "seed": cfg.seed,
            "drift_mode": cfg.drift_mode,
            "s0": cfg.s0,
            "models": report.models,
            "most_conservative_model": report.most_conservative,
            "all_chunk_safe": report.all_chunk_safe,
            "market_source": report.market_source,
        },
        "table": report.rows,
        "per_model": report.per_model,
    }
    return json.dumps(payload, indent=indent, default=_json_default)


def write_comparison_csv(report: ComparisonReport, path: str) -> str:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        fh.write(comparison_to_csv(report))
    return path


def write_comparison_json(report: ComparisonReport, path: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(comparison_to_json(report))
    return path


# ---------------------------------------------------------------------------
# EVT tail-risk module (Generalized Pareto over threshold exceedances)
# ---------------------------------------------------------------------------

EVT_LEVELS = (95.0, 99.0, 99.5, 99.9)
EVT_MIN_EXCEEDANCES = 50


def gpd_fit_mom(exceedances: np.ndarray) -> "tuple[float, float]":
    """Fit a Generalized Pareto Distribution by the method of moments.

    Returns ``(shape_xi, scale_beta)``.  MoM is closed-form and dependency-free
    (no scipy needed); valid for shape < 0.5.
    """
    y = np.asarray(exceedances, dtype=float)
    m = float(np.mean(y))
    v = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    if v <= 0 or m <= 0:
        return 0.0, max(m, 1e-12)
    xi = 0.5 * (1.0 - (m * m) / v)
    beta = m * (1.0 - xi)
    if beta <= 0:
        beta = max(m, 1e-12)
    return float(xi), float(beta)


def evt_tail_risk(
    losses: np.ndarray,
    *,
    levels=EVT_LEVELS,
    threshold_pct: float = 95.0,
    min_exceedances: int = EVT_MIN_EXCEEDANCES,
) -> Dict[str, Any]:
    """Peaks-over-threshold EVT tail-risk estimate.

    ``losses`` should be positive-is-bad (e.g. negative returns).  Fits a GPD to
    exceedances over the ``threshold_pct`` percentile and returns POT VaR/ES at
    each confidence level, the threshold, exceedance count, and a warning when
    the sample is too small.
    """
    x = np.asarray(losses, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    result: Dict[str, Any] = {
        "levels": list(levels),
        "threshold_pct": float(threshold_pct),
        "var": {},
        "es": {},
        "warning": None,
        "error": None,
    }
    if n < 10:
        result["error"] = "too few samples for EVT"
        result["warning"] = f"EVT skipped: only {n} samples."
        return result

    u = float(np.percentile(x, threshold_pct))
    exceed = x[x > u] - u
    nu = exceed.size
    result["threshold"] = u
    result["n_exceedances"] = int(nu)

    if nu < 2:
        result["error"] = "no exceedances above threshold"
        result["warning"] = "EVT: not enough exceedances above threshold."
        return result

    xi, beta = gpd_fit_mom(exceed)
    result["shape_xi"] = xi
    result["scale_beta"] = beta

    rate = nu / n  # exceedance rate
    for level in levels:
        p = level / 100.0
        tail = 1.0 - p
        if abs(xi) < 1e-8:
            var_l = u + beta * math.log(rate / tail) if tail > 0 else float("inf")
        else:
            var_l = u + (beta / xi) * ((tail / rate) ** (-xi) - 1.0)
        result["var"][_level_key(level)] = float(var_l)
        if xi < 1.0:
            es_l = (var_l + beta - xi * u) / (1.0 - xi)
        else:
            es_l = float("inf")
        result["es"][_level_key(level)] = float(es_l)

    if nu < min_exceedances:
        result["warning"] = (
            f"EVT warning: only {nu} exceedances (< {min_exceedances}); "
            "tail estimates are unreliable."
        )
    return result


def evt_from_result(result: SimulationResult, **kwargs) -> Dict[str, Any]:
    """EVT tail-risk from a simulation result's ending-value loss distribution."""
    s0 = result.config.s0
    returns = result.final_values / s0 - 1.0
    losses = -returns  # positive == loss
    return evt_tail_risk(losses, **kwargs)


# ---------------------------------------------------------------------------
# Variance-reduction convergence diagnostics (GBM terminal expectation)
# ---------------------------------------------------------------------------


def variance_reduction_diagnostics(
    config: SimulationConfig,
    path_counts=(1_000, 2_000, 5_000, 10_000),
) -> Dict[str, Any]:
    """Compare plain MC vs antithetic vs control-variate for GBM E[S_T].

    Uses the GBM terminal-price law directly (one normal per path), so it is
    fast and never allocates a path x step matrix.  Reports, per path count, the
    estimate and standard error for each method plus the analytic target.
    """
    cfg = config
    mu = cfg.effective_mu()
    sigma = cfg.effective_sigma()
    T = cfg.horizon * cfg.dt
    a = (mu - 0.5 * sigma ** 2) * T
    b = sigma * math.sqrt(T)
    analytic = cfg.s0 * math.exp(mu * T)
    rng = np.random.default_rng(cfg.seed)

    rows = []
    for n in path_counts:
        # Plain MC.
        z = rng.standard_normal(n)
        st = cfg.s0 * np.exp(a + b * z)
        plain_mean = float(np.mean(st))
        plain_se = float(np.std(st, ddof=1) / math.sqrt(n)) if n > 1 else 0.0

        # Antithetic.
        half = (n + 1) // 2
        za = rng.standard_normal(half)
        z_anti = np.concatenate([za, -za])[:n]
        st_a = cfg.s0 * np.exp(a + b * z_anti)
        anti_mean = float(np.mean(st_a))
        anti_se = float(np.std(st_a, ddof=1) / math.sqrt(n)) if n > 1 else 0.0

        # Control variate (control = z, known mean 0).
        zc = rng.standard_normal(n)
        st_c = cfg.s0 * np.exp(a + b * zc)
        if n > 1 and np.var(zc) > 0:
            beta = float(np.cov(st_c, zc, ddof=1)[0, 1] / np.var(zc, ddof=1))
        else:
            beta = 0.0
        cv = st_c - beta * zc
        cv_mean = float(np.mean(cv))
        cv_se = float(np.std(cv, ddof=1) / math.sqrt(n)) if n > 1 else 0.0

        rows.append({
            "paths": int(n),
            "analytic": analytic,
            "plain_mean": plain_mean, "plain_se": plain_se,
            "antithetic_mean": anti_mean, "antithetic_se": anti_se,
            "control_variate_mean": cv_mean, "control_variate_se": cv_se,
            "antithetic_se_ratio": (anti_se / plain_se) if plain_se > 0 else float("nan"),
            "control_variate_se_ratio": (cv_se / plain_se) if plain_se > 0 else float("nan"),
        })
    return {"analytic_expected_value": analytic, "rows": rows}


def sobol_available() -> bool:
    """True when scipy's Sobol QMC engine is importable (optional dependency)."""
    try:  # pragma: no cover - depends on environment
        from scipy.stats import qmc  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Portfolio mode (laptop-safe correlated multi-asset simulation)
# ---------------------------------------------------------------------------


def align_returns(returns_by_ticker: "Dict[str, np.ndarray]") -> "tuple[List[str], np.ndarray]":
    """Tail-align per-ticker daily returns into a (T, k) matrix."""
    tickers = list(returns_by_ticker.keys())
    series = [np.asarray(returns_by_ticker[t], dtype=float).ravel() for t in tickers]
    min_len = min(s.size for s in series)
    if min_len < 2:
        raise ValueError("need at least two aligned returns per ticker")
    matrix = np.column_stack([s[-min_len:] for s in series])
    return tickers, matrix


def shrink_covariance(returns_matrix: np.ndarray) -> "tuple[np.ndarray, str]":
    """Estimate a covariance matrix with shrinkage; returns (cov, method).

    Uses scikit-learn's Ledoit-Wolf when available, otherwise a dependency-free
    linear shrinkage toward the diagonal of sample variances.
    """
    X = np.asarray(returns_matrix, dtype=float)
    try:  # pragma: no cover - optional dependency
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(X).covariance_
        return np.asarray(cov, dtype=float), "ledoit_wolf"
    except Exception:
        sample = np.cov(X, rowvar=False, ddof=1)
        sample = np.atleast_2d(sample)
        target = np.diag(np.diag(sample))
        k = sample.shape[0]
        delta = min(1.0, max(0.0, k / max(X.shape[0], 1)))  # shrink more when T is small
        delta = max(delta, 0.10)
        cov = (1.0 - delta) * sample + delta * target
        return cov, "diagonal_shrinkage"


def cholesky_safe(cov: np.ndarray) -> "tuple[np.ndarray, bool]":
    """Cholesky factor with PD repair; returns (L, jittered)."""
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    try:
        return np.linalg.cholesky(cov), False
    except np.linalg.LinAlgError:
        # Repair: floor eigenvalues to a small positive value.
        vals, vecs = np.linalg.eigh((cov + cov.T) / 2.0)
        vals = np.clip(vals, 1e-10, None)
        repaired = (vecs * vals) @ vecs.T
        return np.linalg.cholesky(repaired), True


def simulate_portfolio(
    returns_by_ticker: "Dict[str, np.ndarray]",
    *,
    weights: "Optional[np.ndarray]" = None,
    s0_by_ticker: "Optional[Dict[str, float]]" = None,
    paths: int = 10_000,
    horizon: int = 252,
    dt: float = 1.0 / TRADING_DAYS_PER_YEAR,
    chunk_size: int = DEFAULT_SERIOUS_CHUNK_SIZE,
    seed: Optional[int] = None,
    drift_mode: str = DRIFT_HISTORICAL,
    manual_drift: Optional[float] = None,
) -> Dict[str, Any]:
    """Chunk-safe correlated multi-asset GBM portfolio simulation.

    Per chunk we hold an (n x k) price block (k assets, n = chunk paths) and
    evolve it step by step.  We never allocate a paths x steps matrix.
    """
    tickers, R = align_returns(returns_by_ticker)
    k = len(tickers)
    T_hist = R.shape[0]

    # Per-asset annualized parameters from each column's daily log returns.
    mus = np.empty(k)
    sigmas = np.empty(k)
    for j in range(k):
        daily_mu = float(np.mean(R[:, j]))
        daily_sd = float(np.std(R[:, j], ddof=1)) if T_hist > 1 else 0.0
        sigmas[j] = daily_sd * math.sqrt(TRADING_DAYS_PER_YEAR)
        mus[j] = daily_mu * TRADING_DAYS_PER_YEAR + 0.5 * sigmas[j] ** 2

    # Apply conservative drift mode per asset.
    if drift_mode == DRIFT_HALF:
        mus = 0.5 * mus
    elif drift_mode == DRIFT_ZERO:
        mus = np.zeros_like(mus)
    elif drift_mode == DRIFT_MANUAL and manual_drift is not None:
        mus = np.full_like(mus, float(manual_drift))

    cov, shrink_method = shrink_covariance(R)
    sd = np.sqrt(np.clip(np.diag(cov), 1e-16, None))
    corr = cov / np.outer(sd, sd)
    np.clip(corr, -1.0, 1.0, out=corr)
    chol, jittered = cholesky_safe(corr)

    if weights is None:
        weights = np.full(k, 1.0 / k)
    weights = np.asarray(weights, dtype=float).ravel()
    if weights.size != k:
        raise ValueError("weights length must match number of tickers")
    weights = weights / weights.sum()

    if s0_by_ticker is None:
        s0_vec = np.full(k, 100.0)
    else:
        s0_vec = np.array([float(s0_by_ticker.get(t, 100.0)) for t in tickers])

    sqrt_dt = math.sqrt(dt)
    drift_step = (mus - 0.5 * sigmas ** 2) * dt
    vol_step = sigmas * sqrt_dt
    rng = np.random.default_rng(seed)

    portfolio_values = np.empty(paths, dtype=np.float64)
    per_asset_final_sum = np.zeros(k)
    peak_block = 0
    produced = 0
    while produced < paths:
        n = min(chunk_size, paths - produced)
        prices = np.tile(s0_vec, (n, 1)).astype(np.float64)  # (n, k) block
        peak_block = max(peak_block, prices.size)
        for _ in range(horizon):
            z = rng.standard_normal((n, k))
            corr_shock = z @ chol.T
            prices *= np.exp(drift_step + vol_step * corr_shock)
        # Portfolio ending value: weighted sum of per-asset gross returns.
        rel = prices / s0_vec
        portfolio_values[produced:produced + n] = (rel * weights).sum(axis=1)
        per_asset_final_sum += rel.sum(axis=0)
        produced += n

    port_stats = compute_statistics(portfolio_values, 1.0)
    per_asset = {
        tickers[j]: {
            "mean_gross_return": float(per_asset_final_sum[j] / paths),
            "annual_mu": float(mus[j]),
            "annual_sigma": float(sigmas[j]),
            "weight": float(weights[j]),
        }
        for j in range(k)
    }
    full_block = paths * k
    return {
        "tickers": tickers,
        "weights": weights.tolist(),
        "paths": int(paths),
        "horizon": int(horizon),
        "portfolio_values": portfolio_values,   # length == paths (relative to 1.0)
        "statistics": port_stats,
        "per_asset": per_asset,
        "correlation_matrix": corr.tolist(),
        "covariance_method": shrink_method,
        "cholesky_jittered": bool(jittered),
        "chunk_safe": peak_block < full_block,
        "peak_block_elements": int(peak_block),
        "full_block_elements": int(full_block),
    }


def portfolio_correlation_csv(portfolio: Dict[str, Any]) -> str:
    """Correlation matrix as CSV (first column = ticker labels)."""
    tickers = portfolio["tickers"]
    corr = portfolio["correlation_matrix"]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([""] + list(tickers))
    for i, t in enumerate(tickers):
        writer.writerow([t] + [f"{c:.6f}" for c in corr[i]])
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Model validation / historical backtest
# ---------------------------------------------------------------------------

HIGH_DRIFT_WARNING_LEVEL = 0.50    # annualized effective drift above this -> warn
SHORT_HISTORY_MIN = TRADING_DAYS_PER_YEAR  # < 1y of data -> warn


def backtest_percentile_bands(
    prices: np.ndarray,
    horizon: int,
    *,
    window: int = TRADING_DAYS_PER_YEAR,
    dt: float = 1.0 / TRADING_DAYS_PER_YEAR,
    lower: float = 5.0,
    upper: float = 95.0,
) -> Dict[str, Any]:
    """Rolling GBM backtest: do realized forward returns fall in the model band?

    For each rolling estimation window we fit GBM (mu, sigma) and form analytic
    5th/50th/95th percentile bands for the ``horizon``-step forward return, then
    check whether the realized forward return lands inside the band.  Reports the
    empirical coverage (ideally ~0.90 for a 5-95 band).
    """
    p = np.asarray(prices, dtype=float).ravel()
    n = p.size
    result: Dict[str, Any] = {
        "window": int(window), "horizon": int(horizon),
        "n_windows": 0, "coverage": float("nan"),
        "band": [lower, upper], "warning": None,
    }
    if n < window + horizon + 2:
        result["warning"] = "Not enough history for a rolling backtest."
        return result

    from math import sqrt as _sqrt
    inside = 0
    total = 0
    z_lo = _norm_ppf(lower / 100.0)
    z_hi = _norm_ppf(upper / 100.0)
    starts = range(0, n - window - horizon, max(1, horizon))
    for t in starts:
        win = p[t:t + window]
        mu, sigma = annualized_parameters(win)
        T = horizon * dt
        center = (mu - 0.5 * sigma ** 2) * T
        spread = sigma * _sqrt(T)
        lo = math.exp(center + z_lo * spread) - 1.0
        hi = math.exp(center + z_hi * spread) - 1.0
        realized = p[t + window + horizon - 1] / p[t + window - 1] - 1.0
        inside += int(lo <= realized <= hi)
        total += 1
    result["n_windows"] = total
    result["coverage"] = (inside / total) if total else float("nan")
    if total < 10:
        result["warning"] = f"Only {total} backtest windows; coverage is noisy."
    return result


def kupiec_pof_test(
    n_obs: int,
    n_breaches: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Kupiec proportion-of-failures test for VaR exception rate.

    Under a correct VaR at level ``1 - alpha``, breach indicators are i.i.d.
    Bernoulli(alpha).  This likelihood-ratio test checks whether the observed
    breach rate equals ``alpha``.

    Parameters
    ----------
    n_obs :
        Number of forecast days (or windows).
    n_breaches :
        Count of VaR breaches (loss > VaR).
    alpha :
        Tail probability of the VaR (e.g. 0.05 for 95% VaR).

    Returns
    -------
    dict with LR statistic, p-value (chi-square 1 d.f.), observed rate, and a
    simple ``reject_null_5pct`` flag (reject = model coverage looks wrong).
    """
    n = int(n_obs)
    x = int(n_breaches)
    a = float(alpha)
    out: Dict[str, Any] = {
        "n_obs": n,
        "n_breaches": x,
        "alpha": a,
        "breach_rate": float("nan"),
        "lr_stat": float("nan"),
        "p_value": float("nan"),
        "reject_null_5pct": False,
        "warning": None,
    }
    if n <= 0 or not (0.0 < a < 1.0) or x < 0 or x > n:
        out["warning"] = "Invalid inputs for Kupiec test."
        return out
    pi_hat = x / n
    out["breach_rate"] = pi_hat
    # LR = -2 ln( L(alpha) / L(pi_hat) )
    # Guard edges where log(0) would appear.
    def _safe_log(p: float) -> float:
        return math.log(max(p, 1e-300))

    if x == 0:
        lr = -2.0 * (n * _safe_log(1.0 - a) - n * _safe_log(1.0 - pi_hat if pi_hat < 1 else 1e-300))
        # When x=0, pi_hat=0: L(pi)= (1-0)^n = 1, so ln L(pi)=0
        lr = -2.0 * (n * _safe_log(1.0 - a))
    elif x == n:
        lr = -2.0 * (n * _safe_log(a))
    else:
        lr = -2.0 * (
            x * _safe_log(a) + (n - x) * _safe_log(1.0 - a)
            - x * _safe_log(pi_hat) - (n - x) * _safe_log(1.0 - pi_hat)
        )
    out["lr_stat"] = float(lr)
    # Survival function of chi-square(1): P(X > lr) = erfc(sqrt(lr/2))
    # chi2(1) CDF related to erf; p = erfc(sqrt(lr/2))
    out["p_value"] = float(math.erfc(math.sqrt(max(lr, 0.0) / 2.0)))
    out["reject_null_5pct"] = bool(out["p_value"] < 0.05)
    return out


def rolling_var_coverage(
    prices: np.ndarray,
    *,
    window: int = 252,
    alpha: float = 0.05,
    method: str = "historical",
) -> Dict[str, Any]:
    """Rolling one-day VaR coverage test on a price series.

    For each day ``t`` after ``window`` history:
      1. Estimate a one-day loss VaR from the past ``window`` daily returns
         (historical percentile, or Gaussian using sample mean/vol).
      2. Compare to the *realized* next-day loss ``-(r_{t})``.
      3. Count a breach when realized loss > VaR.

    Returns breach stats plus a Kupiec POF test.

    Parameters
    ----------
    prices :
        Positive price levels (oldest → newest).
    window :
        Estimation window length in trading days.
    alpha :
        VaR tail level (0.05 → 95% VaR).
    method :
        ``\"historical\"`` (empirical quantile) or ``\"gaussian\"``.
    """
    p = np.asarray(prices, dtype=float).ravel()
    out: Dict[str, Any] = {
        "window": int(window),
        "alpha": float(alpha),
        "method": method,
        "n_forecasts": 0,
        "n_breaches": 0,
        "breach_rate": float("nan"),
        "kupiec": {},
        "warning": None,
    }
    if p.size < window + 2:
        out["warning"] = "Not enough prices for rolling VaR coverage."
        return out
    if np.any(p <= 0):
        out["warning"] = "Prices must be strictly positive."
        return out

    log_r = np.diff(np.log(p))
    breaches = 0
    total = 0
    # forecast day index i uses returns log_r[i-window : i] to VaR-check log_r[i]
    for i in range(window, log_r.size):
        hist = log_r[i - window:i]
        realized_loss = -float(log_r[i])  # positive when price falls
        if method == "gaussian":
            m = float(np.mean(hist))
            s = float(np.std(hist, ddof=1)) if hist.size > 1 else 0.0
            # VaR on loss scale: - (mean + z_alpha * vol) for left-tail return
            z = _norm_ppf(alpha)
            var_loss = - (m + z * s)
        else:
            # Historical VaR: - quantile_alpha(returns) as loss threshold
            q = float(np.quantile(hist, alpha))
            var_loss = -q
        if realized_loss > var_loss + 1e-15:
            breaches += 1
        total += 1

    out["n_forecasts"] = total
    out["n_breaches"] = breaches
    out["breach_rate"] = (breaches / total) if total else float("nan")
    out["kupiec"] = kupiec_pof_test(total, breaches, alpha=alpha)
    if total < 50:
        out["warning"] = f"Only {total} VaR forecasts; Kupiec test is low-power."
    return out


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation, no scipy)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def collect_warnings(
    config: SimulationConfig,
    market: Optional[MarketParameters] = None,
    *,
    evt: Optional[Dict[str, Any]] = None,
    backtest: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Assemble model-validation warnings for the report."""
    warnings: List[str] = []
    eff_mu = config.effective_mu()
    if eff_mu > HIGH_DRIFT_WARNING_LEVEL:
        warnings.append(
            f"Calibration warning: effective annual drift {eff_mu:.1%} is very high; "
            "results may be over-optimistic. Consider a conservative drift mode."
        )
    if market is not None and market.daily_log_returns is not None:
        n_hist = int(np.asarray(market.daily_log_returns).size)
        if n_hist < SHORT_HISTORY_MIN:
            warnings.append(
                f"Short-history warning: only {n_hist} daily returns "
                f"(< {SHORT_HISTORY_MIN}); parameter estimates are unreliable."
            )
    if market is not None and market.source == "fallback":
        warnings.append("Data warning: using offline fallback parameters (no live data).")
    if evt is not None:
        if evt.get("error"):
            warnings.append(f"EVT warning: {evt['error']}.")
        elif evt.get("warning"):
            warnings.append(evt["warning"])
    if backtest is not None and backtest.get("warning"):
        warnings.append(f"Backtest warning: {backtest['warning']}")
    return warnings
