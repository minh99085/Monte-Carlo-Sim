"""
Phase 2 — Tactical trading-rule simulator.

What this module does (plain English)
-------------------------------------
1. Ask the existing Monte Carlo engine (``mc_core.simulate``) to invent many
   short future price stories for a stock (typically 5–10 trading days).
2. On **each** story, walk day by day and apply a Phase 1 ``TradingRule``:
      - enter the trade (long or short),
      - watch for a stop-loss hit,
      - force an exit at the maximum holding period if still open.
3. Record profit/loss, whether the stop was hit, and how many trades happened.
4. Summarize the distribution across all stories (chance of profit, average P&L,
   worst loss, stop-out rate, typical trade count, …).

Phase 1 only *described* rules. Phase 2 actually *tests* them on simulated paths.

This file does **not** modify ``mc_core.py``. It reuses the engine as a library.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mc_core import (
    DRIFT_MANUAL,
    DRIFT_ZERO,
    FALLBACK_S0,
    FALLBACK_SIGMA,
    SimulationConfig,
    estimate_parameters_from_history,
    simulate,
)
from tactical_config import TacticalConfig, TradingRule, preset_5_day


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How we label why a trade closed on a given path.
EXIT_STOP_LOSS = "stop_loss"
EXIT_MAX_HOLDING = "max_holding"
EXIT_NO_TRADE = "no_trade"

SIDE_LONG = "long"
SIDE_SHORT = "short"

# Soft guard: storing every full path needs paths × (horizon+1) floats.
# Short horizons stay cheap; this only warns / raises for extreme requests.
_MAX_PATH_MATRIX_ELEMENTS = 20_000_000  # ~160 MB of float64


# ---------------------------------------------------------------------------
# Per-path and run-level result containers
# ---------------------------------------------------------------------------


@dataclass
class PathTradeResult:
    """
    What happened on a *single* simulated price story.

    All money figures are in the same units as the simulated price
    (e.g. dollars per share), unless the name ends in ``_pct``
    (those are fractions: 0.01 == +1%).
    """

    path_index: int
    side: str                       # "long" or "short"
    entered: bool                   # did we open a position?
    n_trades: int                   # completed round-trips on this path (0 or 1)
    entry_day: int                  # 0 = start of horizon (price index 0)
    exit_day: int                   # day index of exit close (1..horizon)
    entry_price: float
    exit_price: float
    pnl: float                      # cash P&L after transaction costs (per share)
    pnl_pct: float                  # P&L as a fraction of entry price
    stop_hit: bool
    exit_reason: str                # stop_loss | max_holding | no_trade
    holding_days: int               # how many trading days the position was open


@dataclass
class TacticalResult:
    """
    Full result of testing a trading rule on many Monte Carlo paths.

    * ``pnl`` / ``stop_hit`` / … arrays are one value **per path**.
    * ``stats`` is a plain dict of summary numbers for humans and reports.
    """

    config: TacticalConfig
    rule: TradingRule
    side: str

    # Per-path outcomes (length == number of simulated paths)
    pnl: np.ndarray
    pnl_pct: np.ndarray
    n_trades: np.ndarray
    stop_hit: np.ndarray            # bool
    exit_reason: np.ndarray         # dtype=object, strings
    entry_day: np.ndarray
    exit_day: np.ndarray
    holding_days: np.ndarray
    entry_price: np.ndarray
    exit_price: np.ndarray

    # Price matrix used for the test: shape (paths, horizon + 1)
    # Column 0 is the starting price; column d is the close of trading day d.
    price_paths: np.ndarray

    # Engine metadata
    simulation_config: SimulationConfig
    market_source: str              # "yfinance", "fallback", or "manual"
    runtime_seconds: float
    stats: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def summary_text(self) -> str:
        """Multi-line plain-English summary of the run."""
        s = self.stats
        lines = [
            "=== Tactical simulation summary ===",
            f"Ticker:           {self.config.ticker}",
            f"Horizon:          {self.config.horizon_days} trading day(s)",
            f"Paths:            {s.get('n_paths', len(self.pnl)):,}",
            f"Side:             {self.side}",
            f"Rule:             {self.rule.name}",
            f"Stop loss:        {self.rule.stop_loss_pct * 100:.2f}%",
            f"Max hold:         {self.rule.max_holding_days} day(s)",
            f"Market source:    {self.market_source}",
            f"Runtime:          {self.runtime_seconds:.3f}s",
            "",
            "--- Outcome distribution ---",
            f"Chance of profit:     {s['prob_profit'] * 100:.2f}%",
            f"Chance of loss:       {s['prob_loss'] * 100:.2f}%",
            f"Chance of flat:       {s['prob_flat'] * 100:.2f}%",
            f"Average P&L:          {s['avg_pnl']:+.4f}  ({s['avg_pnl_pct'] * 100:+.3f}%)",
            f"Median P&L:           {s['median_pnl']:+.4f}  ({s['median_pnl_pct'] * 100:+.3f}%)",
            f"Best P&L:             {s['best_pnl']:+.4f}  ({s['best_pnl_pct'] * 100:+.3f}%)",
            f"Worst P&L:            {s['worst_pnl']:+.4f}  ({s['worst_pnl_pct'] * 100:+.3f}%)",
            f"P&L std. dev.:        {s['std_pnl']:.4f}",
            "",
            "--- Trade / stop stats ---",
            f"Avg trades / path:    {s['avg_trades_per_path']:.3f}",
            f"Paths with a trade:   {s['frac_paths_with_trade'] * 100:.2f}%",
            f"Stop-loss hit rate:   {s['stop_hit_rate'] * 100:.2f}%",
            f"Max-hold exit rate:   {s['max_hold_exit_rate'] * 100:.2f}%",
            f"Avg holding days:     {s['avg_holding_days']:.2f}",
            f"5th pct P&L:          {s['pnl_p05']:+.4f}",
            f"95th pct P&L:         {s['pnl_p95']:+.4f}",
        ]
        return "\n".join(lines)

    def to_stats_dict(self) -> Dict[str, Any]:
        """Copy of the summary stats plus a few identifiers (JSON-friendly)."""
        out = dict(self.stats)
        out["ticker"] = self.config.ticker
        out["horizon_days"] = self.config.horizon_days
        out["side"] = self.side
        out["rule_name"] = self.rule.name
        out["stop_loss_pct"] = self.rule.stop_loss_pct
        out["max_holding_days"] = self.rule.max_holding_days
        out["market_source"] = self.market_source
        out["runtime_seconds"] = self.runtime_seconds
        return out


# ---------------------------------------------------------------------------
# Building / resolving configuration for the core engine
# ---------------------------------------------------------------------------


def resolve_market_parameters(
    tactical: TacticalConfig,
) -> Tuple[float, float, float, str]:
    """
    Decide starting price (s0), annual drift (mu), and annual volatility (sigma).

    Order of preference:
      1. Values already set on the TacticalConfig.
      2. Live / historical estimate via mc_core (yfinance, with offline fallback).
      3. Hard-coded safe fallbacks from mc_core.

    Returns ``(s0, mu, sigma, source_label)``.
    """
    need_s0 = tactical.starting_price is None
    need_vol = tactical.annual_volatility is None
    # Drift: TacticalConfig defaults to 0.0; only fetch history if truly unset.
    need_mu = tactical.annual_drift is None

    if not (need_s0 or need_vol or need_mu):
        return (
            float(tactical.starting_price),
            float(tactical.annual_drift),
            float(tactical.annual_volatility),
            "manual",
        )

    market = estimate_parameters_from_history(
        tactical.ticker,
        s0_override=tactical.starting_price,
    )
    s0 = float(tactical.starting_price) if tactical.starting_price is not None else market.s0
    mu = float(tactical.annual_drift) if tactical.annual_drift is not None else market.mu
    sigma = (
        float(tactical.annual_volatility)
        if tactical.annual_volatility is not None
        else market.sigma
    )
    return s0, mu, sigma, market.source


def build_simulation_config(
    tactical: TacticalConfig,
    *,
    s0: float,
    mu: float,
    sigma: float,
) -> SimulationConfig:
    """
    Turn a Phase 1 TacticalConfig into an mc_core SimulationConfig.

    Important Phase 2 choice
    ------------------------
    We set ``sample_paths = paths`` so the engine keeps the **full price
    trajectory** for every path. That is safe for short horizons (5–10 days)
    because the matrix is only ``paths × (horizon+1)`` — e.g. 100,000 × 6
    floats ≈ 5 MB — and it lets us walk each story day by day.
    """
    tactical.validate()
    n_paths = int(tactical.paths)
    horizon = int(tactical.horizon_days)

    matrix_elements = n_paths * (horizon + 1)
    if matrix_elements > _MAX_PATH_MATRIX_ELEMENTS:
        raise ValueError(
            f"Refusing to store {n_paths:,} full paths of length {horizon + 1} "
            f"({matrix_elements:,} elements). Reduce paths or horizon for "
            f"tactical rule testing (short horizons of 5–10 days are intended)."
        )

    # Map drift: zero-drift is common for short-horizon tactics.
    if mu == 0.0:
        drift_mode = DRIFT_ZERO
        manual_drift = None
    else:
        drift_mode = DRIFT_MANUAL
        manual_drift = mu

    kwargs = tactical.to_simulation_kwargs()
    kwargs.update(
        {
            "s0": float(s0),
            "mu": float(mu),
            "sigma": float(sigma),
            "sample_paths": n_paths,  # keep EVERY path for rule testing
            "drift_mode": drift_mode,
            "manual_drift": manual_drift,
            # Buy-and-hold cost inside mc_core is separate; we apply trade costs
            # ourselves when the rule exits. Keep engine cost at 0 so final_values
            # stay "gross" if anyone inspects them.
            "cost": 0.0,
        }
    )
    return SimulationConfig(**kwargs).validate()


def generate_price_paths(
    tactical: TacticalConfig,
) -> Tuple[np.ndarray, SimulationConfig, str, float]:
    """
    Generate price paths with the existing Monte Carlo engine.

    Returns
    -------
    price_paths : ndarray, shape (paths, horizon + 1)
        Full trajectories. Column 0 is the start price.
    sim_cfg : SimulationConfig
        The exact engine config that was run.
    market_source : str
        Where s0 / sigma came from.
    engine_runtime : float
        Seconds spent inside ``mc_core.simulate``.
    """
    s0, mu, sigma, source = resolve_market_parameters(tactical)
    sim_cfg = build_simulation_config(tactical, s0=s0, mu=mu, sigma=sigma)
    t0 = time.perf_counter()
    result = simulate(sim_cfg)
    engine_runtime = time.perf_counter() - t0

    paths = np.asarray(result.sample_trajectories, dtype=np.float64)
    if paths.shape != (sim_cfg.paths, sim_cfg.horizon + 1):
        raise RuntimeError(
            f"Engine returned sample_trajectories shape {paths.shape}, "
            f"expected {(sim_cfg.paths, sim_cfg.horizon + 1)}. "
            "Full-path capture failed."
        )
    return paths, sim_cfg, source, engine_runtime


# ---------------------------------------------------------------------------
# How the trading rule is applied (the core of Phase 2)
# ---------------------------------------------------------------------------


def infer_side(rule: TradingRule) -> str:
    """
    Decide long vs short from the free-text entry condition.

    Phase 1 stores entry/exit as plain English. Phase 2 uses a simple rule:
      - if the word \"short\" appears in the entry text → short
      - otherwise → long

    Examples that become short:
      \"Enter short if open gaps up more than 1%\"
    Examples that become long:
      \"Enter long at the start of day 1\"
      \"Buy the open of day 1\"
    """
    text = (rule.entry_condition or "").lower()
    if "short" in text:
        return SIDE_SHORT
    return SIDE_LONG


def _trade_pnl(
    side: str,
    entry_price: float,
    exit_price: float,
    cost: float,
) -> Tuple[float, float]:
    """
    Cash P&L and percent P&L for one round-trip, including proportional costs.

    Cost model (same spirit as mc_core.apply_costs):
      - pay ``cost`` on the entry notional, and
      - pay ``cost`` on the exit notional.

    Long:  buy at entry, sell at exit
           pnl = exit*(1-cost) - entry*(1+cost)

    Short: sell at entry, buy back at exit
           pnl = entry*(1-cost) - exit*(1+cost)
    """
    entry = float(entry_price)
    exit_ = float(exit_price)
    c = float(cost)
    if side == SIDE_SHORT:
        pnl = entry * (1.0 - c) - exit_ * (1.0 + c)
    else:
        pnl = exit_ * (1.0 - c) - entry * (1.0 + c)
    pnl_pct = pnl / entry if entry != 0.0 else 0.0
    return pnl, pnl_pct


def apply_rule_to_one_path(
    prices: np.ndarray,
    rule: TradingRule,
    *,
    cost: float = 0.0,
    side: Optional[str] = None,
    path_index: int = 0,
) -> PathTradeResult:
    """
    Walk **one** price path day by day and apply the trading rule.

    Path layout
    -----------
    ``prices[0]``  = price at the start of the window (\"now\")
    ``prices[d]``  = simulated close after ``d`` trading days
    length         = horizon + 1

    How the rule is applied (step by step)
    --------------------------------------
    1. **Enter** at the start of the window (day 0 / ``prices[0]``).
       Phase 2 uses a single tactical entry per path — matching the Phase 1
       default language \"Enter long at the start of day 1\".
    2. For each later day ``d = 1, 2, …, max_holding_days`` (capped by the
       path length):
         a. Read that day's close ``prices[d]``.
         b. **Stop loss** (if configured):
              - long:  stop if close <= entry × (1 − stop_loss_pct)
              - short: stop if close >= entry × (1 + stop_loss_pct)
            If hit → exit immediately at that close.
         c. If ``d`` reaches ``max_holding_days`` without a stop →
            **time stop**: exit at that close.
    3. Record P&L (with transaction costs), stop-hit flag, and trade count.

    The free-text ``exit_condition`` is treated as documentation in Phase 2;
    the *executable* exits are stop-loss and maximum holding period (the two
    quantitative fields on ``TradingRule``).
    """
    rule.validate()
    prices = np.asarray(prices, dtype=np.float64).ravel()
    if prices.size < 2:
        raise ValueError("price path must have at least a start and one day")

    horizon = prices.size - 1
    side = side or infer_side(rule)
    if side not in (SIDE_LONG, SIDE_SHORT):
        raise ValueError(f"side must be 'long' or 'short', got {side!r}")

    # Cap the hold to what the path actually contains.
    max_hold = min(int(rule.max_holding_days), horizon)
    if max_hold < 1:
        raise ValueError("max holding period must allow at least 1 day on this path")

    entry_day = 0
    entry_price = float(prices[entry_day])
    stop_pct = float(rule.stop_loss_pct)

    stop_hit = False
    exit_day = max_hold
    exit_reason = EXIT_MAX_HOLDING

    # --- Day-by-day walk ------------------------------------------------
    # We look at the close of each trading day after entry.
    for d in range(1, max_hold + 1):
        close = float(prices[d])

        # 1) Stop-loss check (skip if stop_loss_pct == 0 → "no stop").
        if stop_pct > 0.0:
            if side == SIDE_LONG:
                # Long loses when price falls.
                if close <= entry_price * (1.0 - stop_pct):
                    stop_hit = True
                    exit_day = d
                    exit_reason = EXIT_STOP_LOSS
                    break
            else:
                # Short loses when price rises.
                if close >= entry_price * (1.0 + stop_pct):
                    stop_hit = True
                    exit_day = d
                    exit_reason = EXIT_STOP_LOSS
                    break

        # 2) Time stop: last allowed day of the hold window.
        if d == max_hold:
            exit_day = d
            exit_reason = EXIT_MAX_HOLDING
            break

    exit_price = float(prices[exit_day])
    pnl, pnl_pct = _trade_pnl(side, entry_price, exit_price, cost)
    holding_days = int(exit_day - entry_day)

    return PathTradeResult(
        path_index=path_index,
        side=side,
        entered=True,
        n_trades=1,
        entry_day=entry_day,
        exit_day=exit_day,
        entry_price=entry_price,
        exit_price=exit_price,
        pnl=pnl,
        pnl_pct=pnl_pct,
        stop_hit=stop_hit,
        exit_reason=exit_reason,
        holding_days=holding_days,
    )


def apply_rule_to_paths(
    price_paths: np.ndarray,
    rule: TradingRule,
    *,
    cost: float = 0.0,
    side: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Apply the trading rule to **every** path (vectorized where it helps).

    This is the bulk version of :func:`apply_rule_to_one_path`. The logic is
    the same; the implementation avoids a pure-Python loop over hundreds of
    thousands of paths for the stop/time-exit search.

    Parameters
    ----------
    price_paths :
        Array of shape ``(n_paths, horizon + 1)``.
    rule :
        Phase 1 ``TradingRule``.
    cost :
        Proportional round-trip friction (e.g. 0.001 = 0.1%).
    side :
        Optional override; otherwise inferred from the rule text.

    Returns
    -------
    dict of numpy arrays, one entry per path for each field
    (pnl, pnl_pct, n_trades, stop_hit, exit_reason, …).
    """
    rule.validate()
    paths = np.asarray(price_paths, dtype=np.float64)
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("price_paths must have shape (n_paths, horizon+1)")

    n_paths, n_cols = paths.shape
    horizon = n_cols - 1
    side = side or infer_side(rule)
    max_hold = min(int(rule.max_holding_days), horizon)
    if max_hold < 1:
        raise ValueError("max holding period must allow at least 1 day")

    entry_day = 0
    entry = paths[:, entry_day].copy()
    stop_pct = float(rule.stop_loss_pct)

    # Default: exit at the max-hold close for everyone; stop may fire earlier.
    exit_day = np.full(n_paths, max_hold, dtype=np.int64)
    stop_hit = np.zeros(n_paths, dtype=bool)

    # --- Day-by-day stop scan (still only max_hold ≤ 10 steps) -----------
    # For each day d = 1..max_hold, mark paths that hit the stop *for the
    # first time* on that day and lock their exit_day.
    if stop_pct > 0.0:
        still_open = np.ones(n_paths, dtype=bool)
        for d in range(1, max_hold + 1):
            close = paths[:, d]
            if side == SIDE_LONG:
                hit_today = close <= entry * (1.0 - stop_pct)
            else:
                hit_today = close >= entry * (1.0 + stop_pct)
            newly_stopped = hit_today & still_open
            if newly_stopped.any():
                exit_day[newly_stopped] = d
                stop_hit[newly_stopped] = True
                still_open[newly_stopped] = False
            if not still_open.any():
                break

    # Exit prices via advanced indexing.
    exit_price = paths[np.arange(n_paths), exit_day]
    entry_price = entry

    # P&L with costs (vectorized form of _trade_pnl).
    c = float(cost)
    if side == SIDE_SHORT:
        pnl = entry_price * (1.0 - c) - exit_price * (1.0 + c)
    else:
        pnl = exit_price * (1.0 - c) - entry_price * (1.0 + c)
    pnl_pct = np.divide(pnl, entry_price, out=np.zeros_like(pnl), where=entry_price != 0)

    holding_days = exit_day - entry_day  # entry_day is 0
    n_trades = np.ones(n_paths, dtype=np.int64)
    exit_reason = np.where(stop_hit, EXIT_STOP_LOSS, EXIT_MAX_HOLDING).astype(object)

    return {
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "n_trades": n_trades,
        "stop_hit": stop_hit,
        "exit_reason": exit_reason,
        "entry_day": np.zeros(n_paths, dtype=np.int64),
        "exit_day": exit_day,
        "holding_days": holding_days.astype(np.int64),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "side": side,
    }


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------


def compute_tactical_stats(
    outcomes: Dict[str, np.ndarray],
    *,
    n_paths: int,
) -> Dict[str, Any]:
    """
    Turn per-path arrays into a clear distribution summary.

    Includes:
      - chance of profit / loss / flat
      - average / median / best / worst P&L (cash and %)
      - stop-loss hit rate, max-hold exit rate
      - average trades per path and average holding days
      - a few percentiles for the left/right tails
    """
    pnl = np.asarray(outcomes["pnl"], dtype=np.float64)
    pnl_pct = np.asarray(outcomes["pnl_pct"], dtype=np.float64)
    n_trades = np.asarray(outcomes["n_trades"], dtype=np.float64)
    stop_hit = np.asarray(outcomes["stop_hit"], dtype=bool)
    exit_reason = outcomes["exit_reason"]
    holding = np.asarray(outcomes["holding_days"], dtype=np.float64)

    # Treat tiny residuals as flat so float noise does not inflate win rate.
    eps = 1e-12
    profit = pnl > eps
    loss = pnl < -eps
    flat = ~(profit | loss)

    max_hold_exits = np.array(
        [r == EXIT_MAX_HOLDING for r in exit_reason], dtype=bool
    )

    stats: Dict[str, Any] = {
        "n_paths": int(n_paths),
        "prob_profit": float(np.mean(profit)),
        "prob_loss": float(np.mean(loss)),
        "prob_flat": float(np.mean(flat)),
        "avg_pnl": float(np.mean(pnl)),
        "median_pnl": float(np.median(pnl)),
        "std_pnl": float(np.std(pnl, ddof=1)) if n_paths > 1 else 0.0,
        "best_pnl": float(np.max(pnl)),
        "worst_pnl": float(np.min(pnl)),
        "avg_pnl_pct": float(np.mean(pnl_pct)),
        "median_pnl_pct": float(np.median(pnl_pct)),
        "best_pnl_pct": float(np.max(pnl_pct)),
        "worst_pnl_pct": float(np.min(pnl_pct)),
        "pnl_p05": float(np.percentile(pnl, 5)),
        "pnl_p25": float(np.percentile(pnl, 25)),
        "pnl_p75": float(np.percentile(pnl, 75)),
        "pnl_p95": float(np.percentile(pnl, 95)),
        "avg_trades_per_path": float(np.mean(n_trades)),
        "total_trades": int(np.sum(n_trades)),
        "frac_paths_with_trade": float(np.mean(n_trades > 0)),
        "stop_hit_rate": float(np.mean(stop_hit)),
        "max_hold_exit_rate": float(np.mean(max_hold_exits)),
        "avg_holding_days": float(np.mean(holding)),
        "median_holding_days": float(np.median(holding)),
    }
    return stats


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_tactical_simulation(
    tactical: Optional[TacticalConfig] = None,
    *,
    rule: Optional[TradingRule] = None,
    **overrides: Any,
) -> TacticalResult:
    """
    Run a full short-horizon tactical simulation.

    Parameters
    ----------
    tactical :
        A Phase 1 ``TacticalConfig``. If omitted, uses the 5-day preset.
    rule :
        Optional rule override. If omitted, uses ``tactical.trading_rule``
        (presets always attach a default rule).
    **overrides :
        Optional field overrides applied to a copy of the config before the
        run (e.g. ``paths=20_000``, ``seed=1``).

    Example
    -------
    >>> from tactical_config import preset_5_day
    >>> from tactical_simulator import run_tactical_simulation
    >>> result = run_tactical_simulation(preset_5_day("AAPL", paths=10_000))
    >>> print(result.summary_text())
    """
    if tactical is None:
        tactical = preset_5_day()
    if overrides:
        from dataclasses import replace as dc_replace

        tactical = dc_replace(tactical, **overrides)

    tactical = tactical.validate()

    active_rule = rule if rule is not None else tactical.trading_rule
    if active_rule is None:
        raise ValueError(
            "No trading rule provided. Attach one with "
            "TacticalConfig.with_rule(...) or pass rule=..."
        )
    active_rule = active_rule.validate()

    # Soft consistency: hold period must fit inside the simulated window.
    if active_rule.max_holding_days > tactical.horizon_days:
        raise ValueError(
            f"rule.max_holding_days ({active_rule.max_holding_days}) exceeds "
            f"horizon_days ({tactical.horizon_days})"
        )

    t0 = time.perf_counter()

    # 1) Invent many short price stories with the existing engine.
    price_paths, sim_cfg, market_source, _engine_rt = generate_price_paths(tactical)

    # 2) Walk each story and apply the rule.
    side = infer_side(active_rule)
    outcomes = apply_rule_to_paths(
        price_paths,
        active_rule,
        cost=float(tactical.transaction_cost),
        side=side,
    )

    # 3) Summarize the distribution of outcomes.
    stats = compute_tactical_stats(outcomes, n_paths=price_paths.shape[0])
    runtime = time.perf_counter() - t0
    stats["runtime_seconds"] = runtime
    stats["engine_s0"] = float(sim_cfg.s0)
    stats["engine_mu"] = float(sim_cfg.mu)
    stats["engine_sigma"] = float(sim_cfg.sigma)
    stats["engine_model"] = sim_cfg.model

    return TacticalResult(
        config=tactical,
        rule=active_rule,
        side=side,
        pnl=outcomes["pnl"],
        pnl_pct=outcomes["pnl_pct"],
        n_trades=outcomes["n_trades"],
        stop_hit=outcomes["stop_hit"],
        exit_reason=outcomes["exit_reason"],
        entry_day=outcomes["entry_day"],
        exit_day=outcomes["exit_day"],
        holding_days=outcomes["holding_days"],
        entry_price=outcomes["entry_price"],
        exit_price=outcomes["exit_price"],
        price_paths=price_paths,
        simulation_config=sim_cfg,
        market_source=market_source,
        runtime_seconds=runtime,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Tiny demo when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tactical_config import preset_5_day, preset_10_day, TradingRule

    print("Running Phase 2 demo (5-day preset, 20,000 paths)...\n")
    cfg = preset_5_day(
        "AAPL",
        paths=20_000,
        seed=42,
        starting_price=100.0,
        annual_volatility=0.25,  # fixed vol so the demo works offline
        annual_drift=0.0,
    )
    result = run_tactical_simulation(cfg)
    print(result.summary_text())
    print()

    # Custom short rule on a 10-day window.
    short_rule = TradingRule(
        name="Illustrative short, 5-day hold",
        entry_condition="Enter short at the start of day 1",
        exit_condition="Cover at max hold or stop",
        stop_loss_pct=0.02,
        max_holding_days=5,
    )
    cfg10 = preset_10_day(
        "MSFT",
        paths=10_000,
        seed=7,
        starting_price=100.0,
        annual_volatility=0.30,
        annual_drift=0.0,
    ).with_rule(short_rule)
    result10 = run_tactical_simulation(cfg10)
    print("\n--- Custom short rule on 10-day horizon ---\n")
    print(result10.summary_text())
