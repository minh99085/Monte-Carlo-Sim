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

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any
import csv
import io
import json
import math
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
        return self


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
        s0 = float(prices[-1]) if s0_override is None else float(s0_override)
        return MarketParameters(s0=s0, mu=mu, sigma=sigma, source="yfinance", note=note)
    except Exception as exc:  # noqa: BLE001 - any failure -> safe fallback
        note = f"Falling back to default parameters ({type(exc).__name__}: {exc})."
        s0 = FALLBACK_S0 if s0_override is None else float(s0_override)
        return MarketParameters(
            s0=s0, mu=FALLBACK_MU, sigma=FALLBACK_SIGMA, source="fallback", note=note
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


def simulate(config: SimulationConfig) -> SimulationResult:
    """Run a chunked GBM Monte Carlo simulation.

    The simulation evolves each chunk one step at a time, so the largest 2-D
    array ever allocated is the bounded sample-trajectory block -- never the full
    ``paths x steps`` matrix.
    """

    cfg = config.validate()
    rng = np.random.default_rng(cfg.seed)

    steps = cfg.horizon
    drift = (cfg.mu - 0.5 * cfg.sigma ** 2) * cfg.dt
    vol = cfg.sigma * math.sqrt(cfg.dt)

    final_values = np.empty(cfg.paths, dtype=np.float64)

    n_sample = min(cfg.sample_paths, cfg.paths)
    sample_trajectories = np.empty((n_sample, steps + 1), dtype=np.float64)
    if n_sample:
        sample_trajectories[:, 0] = cfg.s0

    memory = MemoryInfo(
        paths=cfg.paths, horizon=cfg.horizon, chunk_size=cfg.chunk_size
    )
    # The sample block is the only 2-D array we deliberately keep.
    memory.peak_matrix_elements = max(memory.peak_matrix_elements, n_sample * (steps + 1))

    start = time.perf_counter()
    produced = 0
    while produced < cfg.paths:
        this_chunk = min(cfg.chunk_size, cfg.paths - produced)

        # Current price vector for this chunk (1-D, bounded by chunk size).
        prices = np.full(this_chunk, cfg.s0, dtype=np.float64)
        memory.peak_vector_elements = max(memory.peak_vector_elements, prices.size)

        # How many of this chunk's paths feed the global sample block.
        sample_in_chunk = max(0, min(n_sample - produced, this_chunk))

        for step in range(1, steps + 1):
            z = rng.standard_normal(this_chunk)
            prices *= np.exp(drift + vol * z)
            if sample_in_chunk:
                sample_trajectories[produced:produced + sample_in_chunk, step] = (
                    prices[:sample_in_chunk]
                )

        gross = prices
        net = apply_costs(gross, cfg.s0, cfg.cost)
        final_values[produced:produced + this_chunk] = net
        produced += this_chunk

    runtime = time.perf_counter() - start

    convergence_paths, convergence_means = _convergence_curve(
        final_values, cfg.convergence_points
    )
    stats = compute_statistics(final_values, cfg.s0, runtime=runtime)

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


def compute_statistics(
    final_values: np.ndarray,
    s0: float,
    *,
    runtime: Optional[float] = None,
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
        "mean_return": float(np.mean(returns)),
        "var": var,
        "expected_shortfall": es,
        "percentiles": percentiles,
    }
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


def build_report(result: SimulationResult, market: Optional[MarketParameters] = None) -> Dict[str, Any]:
    """Assemble a serialisable report dict from a simulation result."""

    cfg = result.config
    report: Dict[str, Any] = {
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
        },
        "statistics": result.stats,
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
        report["market"] = {
            "source": market.source,
            "note": market.note,
        }
    return report


def report_to_json(result: SimulationResult, market: Optional[MarketParameters] = None,
                   *, indent: int = 2) -> str:
    """Serialise the full report to a JSON string."""
    return json.dumps(build_report(result, market), indent=indent, default=_json_default)


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
    rows: List[List[Any]] = [
        ["ticker", cfg.ticker],
        ["starting_price", cfg.s0],
        ["paths", cfg.paths],
        ["horizon_steps", cfg.horizon],
        ["mu_annual", cfg.mu],
        ["sigma_annual", cfg.sigma],
        ["chunk_size", cfg.chunk_size],
        ["seed", cfg.seed],
        ["transaction_cost", cfg.cost],
        ["expected_ending_value", s["expected_value"]],
        ["median_ending_value", s["median_value"]],
        ["expected_return", s["expected_return"]],
        ["std_ending_value", s["std_value"]],
        ["min_ending_value", s["min_value"]],
        ["max_ending_value", s["max_value"]],
        ["prob_profit", s["prob_profit"]],
        ["prob_loss", s["prob_loss"]],
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
               market: Optional[MarketParameters] = None) -> str:
    text = report_to_json(result, market)
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
