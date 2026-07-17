"""
Production-grade tactical trading-rule simulator (Phases 1–3).

What this module does (plain English)
-------------------------------------
1. Ask the existing Monte Carlo engine (``mc_core.simulate``) to invent many
   short future price stories for a stock (typically 5–10 trading days).
2. On **each** story, walk day by day and apply a ``TradingRule``:
      - flexible entry (fixed day, text side, or callable signal),
      - stop-loss, take-profit, trailing stop,
      - max holding period,
      - optional re-entry within the horizon.
3. Optionally run the **same rule on historical prices** (rolling windows) and
   compare to the Monte Carlo distribution.
4. Optionally read the latest **TradingView webhook** JSON (Phase 3 bridge)
   to set ticker/price, align side to trend, and scale volatility/jumps
   from momentum.
5. Summarize P&L, trade counts, stop/TP hit rates, and whether TV data was used.

This file reuses ``mc_core``; it does not break the long-horizon buy-and-hold path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mc_core import (
    DRIFT_MANUAL,
    DRIFT_ZERO,
    SimulationConfig,
    estimate_parameters_from_history,
    kupiec_pof_test,
    rolling_var_coverage,
    simulate,
)
from tactical_config import TacticalConfig, TradingRule, preset_5_day
from tv_integration import (
    TVSignalContext,
    apply_tradingview_to_tactical,
    load_tradingview_signal,
    resolve_signal_path,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXIT_STOP_LOSS = "stop_loss"
EXIT_TAKE_PROFIT = "take_profit"
EXIT_TRAILING_STOP = "trailing_stop"
EXIT_MAX_HOLDING = "max_holding"
EXIT_NO_TRADE = "no_trade"
EXIT_SIGNAL = "signal_exit"

SIDE_LONG = "long"
SIDE_SHORT = "short"

_MAX_PATH_MATRIX_ELEMENTS = 20_000_000  # ~160 MB float64


# ---------------------------------------------------------------------------
# Optional Numba (graceful fallback)
# ---------------------------------------------------------------------------

def _numba_available() -> bool:
    try:  # pragma: no cover - env dependent
        import numba  # noqa: F401
        return True
    except Exception:
        return False


def _classic_stop_hold_kernel(
    paths: np.ndarray,
    max_hold: int,
    stop_pct: float,
    is_long: bool,
    cost: float,
):
    """
    Classic enter-at-t0 / hard-stop / max-hold engine over all paths.

    Pure Python + NumPy types so the same body can run under Numba ``njit``
    or as a plain function.  Returns ``(exit_day, stop_hit, pnl, pnl_pct)``.
    """
    n_paths, n_cols = paths.shape
    horizon = n_cols - 1
    mh = max_hold if max_hold < horizon else horizon
    if mh < 1:
        mh = 1

    exit_day = np.empty(n_paths, dtype=np.int64)
    stop_hit = np.zeros(n_paths, dtype=np.bool_)
    pnl = np.empty(n_paths, dtype=np.float64)
    pnl_pct = np.empty(n_paths, dtype=np.float64)

    for i in range(n_paths):
        entry = paths[i, 0]
        ed = mh
        hit = False
        if stop_pct > 0.0:
            for d in range(1, mh + 1):
                close = paths[i, d]
                if is_long:
                    if close <= entry * (1.0 - stop_pct):
                        ed = d
                        hit = True
                        break
                else:
                    if close >= entry * (1.0 + stop_pct):
                        ed = d
                        hit = True
                        break
        exit_px = paths[i, ed]
        if is_long:
            p = exit_px * (1.0 - cost) - entry * (1.0 + cost)
        else:
            p = entry * (1.0 - cost) - exit_px * (1.0 + cost)
        exit_day[i] = ed
        stop_hit[i] = hit
        pnl[i] = p
        pnl_pct[i] = p / entry if entry != 0.0 else 0.0

    return exit_day, stop_hit, pnl, pnl_pct


# Compile with Numba when present; otherwise keep the pure-Python kernel.
_classic_stop_hold_kernel_jit = None
if _numba_available():  # pragma: no cover - depends on optional dep
    try:
        from numba import njit

        _classic_stop_hold_kernel_jit = njit(cache=True)(_classic_stop_hold_kernel)
    except Exception:
        _classic_stop_hold_kernel_jit = None


def _run_classic_stop_hold(
    paths: np.ndarray,
    max_hold: int,
    stop_pct: float,
    is_long: bool,
    cost: float,
):
    """Dispatch to Numba JIT kernel when available, else pure Python."""
    fn = _classic_stop_hold_kernel_jit or _classic_stop_hold_kernel
    return fn(paths, int(max_hold), float(stop_pct), bool(is_long), float(cost))


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class PathTradeResult:
    """Outcome for one completed trade (a path may hold several)."""

    path_index: int
    trade_index: int
    side: str
    entered: bool
    n_trades: int                 # total completed on this path after this trade
    entry_day: int
    exit_day: int
    entry_price: float
    exit_price: float
    pnl: float
    pnl_pct: float
    stop_hit: bool
    take_profit_hit: bool
    trailing_stop_hit: bool
    exit_reason: str
    holding_days: int


@dataclass
class TacticalResult:
    """Full Monte Carlo tactical run (one value per path unless noted)."""

    config: TacticalConfig
    rule: TradingRule
    side: str

    pnl: np.ndarray                 # sum of trade PnLs per path
    pnl_pct: np.ndarray             # pnl / first entry (or 0 if no trade)
    n_trades: np.ndarray
    stop_hit: np.ndarray            # True if *any* trade stopped out
    take_profit_hit: np.ndarray
    trailing_stop_hit: np.ndarray
    exit_reason: np.ndarray         # last exit reason per path
    entry_day: np.ndarray           # first entry day (-1 if none)
    exit_day: np.ndarray            # last exit day (-1 if none)
    holding_days: np.ndarray        # total days in market
    entry_price: np.ndarray         # first entry price (nan if none)
    exit_price: np.ndarray          # last exit price (nan if none)

    price_paths: np.ndarray
    simulation_config: SimulationConfig
    market_source: str
    runtime_seconds: float
    stats: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    historical: Optional["HistoricalTacticalResult"] = None
    backtest: Optional[Dict[str, Any]] = None
    # Phase 3 integration: optional TradingView signal context
    tv_context: Optional[TVSignalContext] = None

    def summary_text(self) -> str:
        s = self.stats
        lines = [
            "=== Tactical simulation summary ===",
            f"Ticker:           {self.config.ticker}",
            f"Horizon:          {self.config.horizon_days} trading day(s)",
            f"Paths:            {s.get('n_paths', len(self.pnl)):,}",
            f"Side:             {self.side}",
            f"Rule:             {self.rule.name}",
            f"Stop loss:        {self.rule.stop_loss_pct * 100:.2f}%",
            f"Take profit:      "
            + (
                f"{self.rule.take_profit_pct * 100:.2f}%"
                if self.rule.take_profit_pct
                else "off"
            ),
            f"Trailing stop:    "
            + (
                f"{self.rule.trailing_stop_pct * 100:.2f}%"
                if self.rule.trailing_stop_pct
                else "off"
            ),
            f"Max hold:         {self.rule.max_holding_days} day(s)",
            f"Re-entry:         {self.rule.allow_reentry} (max_trades={self.rule.max_trades})",
            f"Market source:    {self.market_source}",
            f"Runtime:          {self.runtime_seconds:.3f}s",
            f"Used TradingView: {'YES' if s.get('used_tradingview_data') else 'NO'}",
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
            f"Take-profit rate:     {s.get('take_profit_rate', 0.0) * 100:.2f}%",
            f"Trailing-stop rate:   {s.get('trailing_stop_rate', 0.0) * 100:.2f}%",
            f"Max-hold exit rate:   {s['max_hold_exit_rate'] * 100:.2f}%",
            f"Avg holding days:     {s['avg_holding_days']:.2f}",
            f"5th pct P&L:          {s['pnl_p05']:+.4f}",
            f"95th pct P&L:         {s['pnl_p95']:+.4f}",
        ]
        if self.tv_context is not None:
            lines.append("")
            lines.append("--- TradingView context ---")
            lines.extend(self.tv_context.summary_lines())
        if self.notes:
            lines.append("")
            lines.append("--- Notes ---")
            lines.extend(f"  * {n}" for n in self.notes)
        if self.historical is not None:
            hs = self.historical.stats
            lines += [
                "",
                "--- Historical window backtest ---",
                f"Windows:              {hs.get('n_windows', 0)}",
                f"Hist. profit chance:  {hs.get('prob_profit', float('nan')) * 100:.2f}%",
                f"Hist. avg P&L:        {hs.get('avg_pnl', float('nan')):+.4f}",
                f"Hist. stop rate:      {hs.get('stop_hit_rate', float('nan')) * 100:.2f}%",
            ]
        if self.backtest is not None:
            lines += [
                "",
                "--- Rolling VaR coverage (reference) ---",
                f"Breach rate:          {self.backtest.get('breach_rate', float('nan'))}",
                f"Kupiec p-value:       "
                f"{self.backtest.get('kupiec', {}).get('p_value', float('nan'))}",
            ]
        return "\n".join(lines)

    def to_stats_dict(self) -> Dict[str, Any]:
        out = dict(self.stats)
        out["ticker"] = self.config.ticker
        out["horizon_days"] = self.config.horizon_days
        out["side"] = self.side
        out["rule_name"] = self.rule.name
        out["stop_loss_pct"] = self.rule.stop_loss_pct
        out["take_profit_pct"] = self.rule.take_profit_pct
        out["trailing_stop_pct"] = self.rule.trailing_stop_pct
        out["max_holding_days"] = self.rule.max_holding_days
        out["allow_reentry"] = self.rule.allow_reentry
        out["max_trades"] = self.rule.max_trades
        out["market_source"] = self.market_source
        out["runtime_seconds"] = self.runtime_seconds
        out["notes"] = list(self.notes)
        out["used_tradingview_data"] = bool(
            self.tv_context.used if self.tv_context is not None else False
        )
        if self.tv_context is not None:
            out["tradingview"] = self.tv_context.as_dict()
        return out


@dataclass
class HistoricalTacticalResult:
    """Rule applied to rolling windows of a real price series."""

    prices_used: int
    horizon_days: int
    n_windows: int
    pnl: np.ndarray
    n_trades: np.ndarray
    stop_hit: np.ndarray
    stats: Dict[str, Any]
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Market / engine helpers
# ---------------------------------------------------------------------------


def resolve_market_parameters(
    tactical: TacticalConfig,
) -> Tuple[float, float, float, str]:
    need_s0 = tactical.starting_price is None
    need_vol = tactical.annual_volatility is None
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
    variance_reduction: str = "none",
    jump_intensity_multiplier: float = 1.0,
) -> SimulationConfig:
    tactical.validate()
    n_paths = int(tactical.paths)
    horizon = int(tactical.horizon_days)

    matrix_elements = n_paths * (horizon + 1)
    if matrix_elements > _MAX_PATH_MATRIX_ELEMENTS:
        raise ValueError(
            f"Refusing to store {n_paths:,} full paths of length {horizon + 1} "
            f"({matrix_elements:,} elements). Reduce paths or horizon."
        )

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
            "sample_paths": n_paths,
            "drift_mode": drift_mode,
            "manual_drift": manual_drift,
            "cost": 0.0,  # trade costs applied in the rule engine
            "variance_reduction": variance_reduction,
        }
    )
    # Momentum-scaled jump intensity (only affects jump models; safe default for GBM).
    if jump_intensity_multiplier and jump_intensity_multiplier != 1.0:
        base_jump = float(kwargs.get("jump_intensity", 1.0) or 1.0)
        kwargs["jump_intensity"] = base_jump * float(jump_intensity_multiplier)
        base_kou = float(kwargs.get("kou_intensity", 1.0) or 1.0)
        kwargs["kou_intensity"] = base_kou * float(jump_intensity_multiplier)
    return SimulationConfig(**kwargs).validate()


def generate_price_paths(
    tactical: TacticalConfig,
    *,
    variance_reduction: str = "none",
    jump_intensity_multiplier: float = 1.0,
) -> Tuple[np.ndarray, SimulationConfig, str, float]:
    s0, mu, sigma, source = resolve_market_parameters(tactical)
    sim_cfg = build_simulation_config(
        tactical,
        s0=s0,
        mu=mu,
        sigma=sigma,
        variance_reduction=variance_reduction,
        jump_intensity_multiplier=jump_intensity_multiplier,
    )
    t0 = time.perf_counter()
    result = simulate(sim_cfg)
    engine_runtime = time.perf_counter() - t0

    paths = np.asarray(result.sample_trajectories, dtype=np.float64)
    if paths.shape != (sim_cfg.paths, sim_cfg.horizon + 1):
        raise RuntimeError(
            f"Engine returned sample_trajectories shape {paths.shape}, "
            f"expected {(sim_cfg.paths, sim_cfg.horizon + 1)}."
        )
    return paths, sim_cfg, source, engine_runtime


# ---------------------------------------------------------------------------
# Rule application (flexible engine)
# ---------------------------------------------------------------------------


def infer_side(rule: TradingRule) -> str:
    """Explicit side wins; else detect 'short' in entry text; else long."""
    if rule.side is not None:
        return str(rule.side).lower()
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
    entry = float(entry_price)
    exit_ = float(exit_price)
    c = float(cost)
    if side == SIDE_SHORT:
        pnl = entry * (1.0 - c) - exit_ * (1.0 + c)
    else:
        pnl = exit_ * (1.0 - c) - entry * (1.0 + c)
    pnl_pct = pnl / entry if entry != 0.0 else 0.0
    return pnl, pnl_pct


def _default_entry_signal(day: int, prices: np.ndarray, rule: TradingRule) -> bool:
    """Built-in entry: enter on ``rule.entry_day`` (and re-entry days if allowed)."""
    return day >= int(rule.entry_day)


def apply_rule_to_one_path(
    prices: np.ndarray,
    rule: TradingRule,
    *,
    cost: float = 0.0,
    side: Optional[str] = None,
    path_index: int = 0,
    entry_fn: Optional[Callable[[int, np.ndarray], bool]] = None,
) -> Tuple[List[PathTradeResult], Dict[str, Any]]:
    """
    Walk one price path day by day with full Phase 2 rule support.

    How the rule is applied
    -----------------------
    1. Start flat. From ``entry_day`` onward, check the entry signal each day.
       Default signal = "day >= entry_day". Override with ``rule.entry_fn`` or
       the ``entry_fn`` argument.
    2. When entry fires, open long/short at that day's close (day 0 = start price).
    3. While in a trade, each later day check (in order):
         a. hard stop-loss
         b. take-profit
         c. trailing stop (from running favorable extreme)
         d. max holding period
    4. On exit, record P&L. If ``allow_reentry`` and trades < ``max_trades``,
       go flat and look for another entry after the exit day.
    5. Edge cases: empty path, zero prices, max_hold beyond horizon — logged in
       ``notes`` and handled safely (no crash).
    """
    rule.validate()
    prices = np.asarray(prices, dtype=np.float64).ravel()
    notes: List[str] = []
    if prices.size < 2:
        notes.append("Path too short; no trade.")
        return [], {"notes": notes, "n_trades": 0, "total_pnl": 0.0}

    if np.any(~np.isfinite(prices)) or np.any(prices <= 0):
        notes.append("Non-finite or non-positive prices detected; skipping path.")
        logger.warning("Path %s has invalid prices; skipped.", path_index)
        return [], {"notes": notes, "n_trades": 0, "total_pnl": 0.0}

    horizon = prices.size - 1
    side = side or infer_side(rule)
    max_hold = min(int(rule.max_holding_days), horizon)
    if max_hold < 1:
        notes.append("max_holding_days does not fit path; no trade.")
        return [], {"notes": notes, "n_trades": 0, "total_pnl": 0.0}

    signal = entry_fn or rule.entry_fn or (
        lambda d, p, _r=rule: _default_entry_signal(d, p, _r)
    )

    stop_pct = float(rule.stop_loss_pct)
    tp_pct = float(rule.take_profit_pct) if rule.take_profit_pct else None
    trail_pct = float(rule.trailing_stop_pct) if rule.trailing_stop_pct else None
    max_trades = int(rule.max_trades) if rule.allow_reentry else 1
    max_trades = max(1, max_trades)

    trades: List[PathTradeResult] = []
    day = 0
    total_pnl = 0.0

    while len(trades) < max_trades and day <= horizon:
        # --- seek entry ---
        entry_day = None
        while day <= horizon:
            try:
                if signal(day, prices[: day + 1]):
                    entry_day = day
                    break
            except Exception as exc:  # noqa: BLE001
                notes.append(f"entry_fn error on day {day}: {type(exc).__name__}")
                logger.exception("entry_fn failed path=%s day=%s", path_index, day)
                day = horizon + 1
                break
            day += 1

        if entry_day is None:
            break

        entry_price = float(prices[entry_day])
        # Favorable extreme for trailing stop
        run_ext = entry_price  # high for long, low for short

        exit_day = min(entry_day + max_hold, horizon)
        exit_reason = EXIT_MAX_HOLDING
        stop_hit = take_profit_hit = trailing_stop_hit = False

        # Need at least one day after entry for a mark-to-market exit when possible.
        # If entry is on the last index, force flat no-op.
        if entry_day >= horizon:
            notes.append("Entry signal on final bar; cannot hold — skipped.")
            break

        for d in range(entry_day + 1, min(entry_day + max_hold, horizon) + 1):
            close = float(prices[d])

            # Update trailing extreme
            if side == SIDE_LONG:
                if close > run_ext:
                    run_ext = close
            else:
                if close < run_ext:
                    run_ext = close

            # 1) Hard stop
            if stop_pct > 0.0:
                if side == SIDE_LONG and close <= entry_price * (1.0 - stop_pct):
                    stop_hit = True
                    exit_day, exit_reason = d, EXIT_STOP_LOSS
                    break
                if side == SIDE_SHORT and close >= entry_price * (1.0 + stop_pct):
                    stop_hit = True
                    exit_day, exit_reason = d, EXIT_STOP_LOSS
                    break

            # 2) Take profit
            if tp_pct is not None:
                if side == SIDE_LONG and close >= entry_price * (1.0 + tp_pct):
                    take_profit_hit = True
                    exit_day, exit_reason = d, EXIT_TAKE_PROFIT
                    break
                if side == SIDE_SHORT and close <= entry_price * (1.0 - tp_pct):
                    take_profit_hit = True
                    exit_day, exit_reason = d, EXIT_TAKE_PROFIT
                    break

            # 3) Trailing stop
            if trail_pct is not None:
                if side == SIDE_LONG and close <= run_ext * (1.0 - trail_pct):
                    trailing_stop_hit = True
                    exit_day, exit_reason = d, EXIT_TRAILING_STOP
                    break
                if side == SIDE_SHORT and close >= run_ext * (1.0 + trail_pct):
                    trailing_stop_hit = True
                    exit_day, exit_reason = d, EXIT_TRAILING_STOP
                    break

            # 4) Time stop
            if d == min(entry_day + max_hold, horizon):
                exit_day, exit_reason = d, EXIT_MAX_HOLDING
                break

        exit_price = float(prices[exit_day])
        pnl, pnl_pct = _trade_pnl(side, entry_price, exit_price, cost)
        total_pnl += pnl
        holding_days = int(exit_day - entry_day)

        trades.append(
            PathTradeResult(
                path_index=path_index,
                trade_index=len(trades),
                side=side,
                entered=True,
                n_trades=len(trades) + 1,
                entry_day=entry_day,
                exit_day=exit_day,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                pnl_pct=pnl_pct,
                stop_hit=stop_hit,
                take_profit_hit=take_profit_hit,
                trailing_stop_hit=trailing_stop_hit,
                exit_reason=exit_reason,
                holding_days=holding_days,
            )
        )

        # Next search starts the day after exit
        day = exit_day + 1
        if not rule.allow_reentry:
            break

    meta = {
        "notes": notes,
        "n_trades": len(trades),
        "total_pnl": total_pnl,
        "side": side,
    }
    return trades, meta


def apply_rule_to_paths(
    price_paths: np.ndarray,
    rule: TradingRule,
    *,
    cost: float = 0.0,
    side: Optional[str] = None,
    entry_fn: Optional[Callable[[int, np.ndarray], bool]] = None,
) -> Dict[str, np.ndarray]:
    """
    Apply the rule to every path.

    Uses the full flexible engine path-by-path (supports re-entry, TP, trail,
    callable entries). For the classic single-trade stop+hold case without
    extras, a faster vectorized path is used automatically.
    """
    rule.validate()
    paths = np.asarray(price_paths, dtype=np.float64)
    if paths.ndim != 2 or paths.shape[1] < 2:
        raise ValueError("price_paths must have shape (n_paths, horizon+1)")

    n_paths = paths.shape[0]
    side = side or infer_side(rule)

    use_fast = (
        entry_fn is None
        and rule.entry_fn is None
        and not rule.allow_reentry
        and rule.max_trades <= 1
        and rule.take_profit_pct is None
        and rule.trailing_stop_pct is None
        and int(rule.entry_day) == 0
    )

    if use_fast:
        return _apply_rule_vectorized_simple(paths, rule, cost=cost, side=side)

    # Flexible path-by-path
    pnl = np.zeros(n_paths, dtype=np.float64)
    pnl_pct = np.zeros(n_paths, dtype=np.float64)
    n_trades = np.zeros(n_paths, dtype=np.int64)
    stop_hit = np.zeros(n_paths, dtype=bool)
    tp_hit = np.zeros(n_paths, dtype=bool)
    trail_hit = np.zeros(n_paths, dtype=bool)
    exit_reason = np.full(n_paths, EXIT_NO_TRADE, dtype=object)
    entry_day = np.full(n_paths, -1, dtype=np.int64)
    exit_day = np.full(n_paths, -1, dtype=np.int64)
    holding_days = np.zeros(n_paths, dtype=np.int64)
    entry_price = np.full(n_paths, np.nan, dtype=np.float64)
    exit_price = np.full(n_paths, np.nan, dtype=np.float64)

    for i in range(n_paths):
        trades, meta = apply_rule_to_one_path(
            paths[i], rule, cost=cost, side=side, path_index=i, entry_fn=entry_fn
        )
        if not trades:
            continue
        n_trades[i] = len(trades)
        pnl[i] = sum(t.pnl for t in trades)
        entry_price[i] = trades[0].entry_price
        exit_price[i] = trades[-1].exit_price
        entry_day[i] = trades[0].entry_day
        exit_day[i] = trades[-1].exit_day
        holding_days[i] = sum(t.holding_days for t in trades)
        stop_hit[i] = any(t.stop_hit for t in trades)
        tp_hit[i] = any(t.take_profit_hit for t in trades)
        trail_hit[i] = any(t.trailing_stop_hit for t in trades)
        exit_reason[i] = trades[-1].exit_reason
        if entry_price[i] and entry_price[i] == entry_price[i]:
            pnl_pct[i] = pnl[i] / entry_price[i]

    return {
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "n_trades": n_trades,
        "stop_hit": stop_hit,
        "take_profit_hit": tp_hit,
        "trailing_stop_hit": trail_hit,
        "exit_reason": exit_reason,
        "entry_day": entry_day,
        "exit_day": exit_day,
        "holding_days": holding_days,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "side": side,
    }


def _apply_rule_vectorized_simple(
    paths: np.ndarray,
    rule: TradingRule,
    *,
    cost: float,
    side: str,
) -> Dict[str, np.ndarray]:
    """
    Fast path for classic enter-at-0 / stop / max-hold rules.

    Uses an optional Numba JIT kernel when ``numba`` is installed; otherwise a
    pure-Python kernel with the same math (still fine for short 5–10 day horizons).
    """
    n_paths, n_cols = paths.shape
    horizon = n_cols - 1
    max_hold = min(int(rule.max_holding_days), horizon)
    if max_hold < 1:
        max_hold = 1
    entry = np.asarray(paths[:, 0], dtype=np.float64).copy()
    stop_pct = float(rule.stop_loss_pct)
    is_long = side == SIDE_LONG

    # Prefer NumPy vectorized scan (very fast); fall back to kernel for parity /
    # when we want the Numba path exercised.  We use the kernel always when
    # Numba is available so the optional dependency actually accelerates work;
    # without Numba we keep the proven vectorized day-scan.
    if _classic_stop_hold_kernel_jit is not None:
        exit_day, stop_hit, pnl, pnl_pct = _run_classic_stop_hold(
            np.ascontiguousarray(paths, dtype=np.float64),
            max_hold,
            stop_pct,
            is_long,
            float(cost),
        )
        stop_hit = np.asarray(stop_hit, dtype=bool)
        exit_day = np.asarray(exit_day, dtype=np.int64)
        pnl = np.asarray(pnl, dtype=np.float64)
        pnl_pct = np.asarray(pnl_pct, dtype=np.float64)
    else:
        exit_day = np.full(n_paths, max_hold, dtype=np.int64)
        stop_hit = np.zeros(n_paths, dtype=bool)
        if stop_pct > 0.0 and max_hold >= 1:
            still_open = np.ones(n_paths, dtype=bool)
            for d in range(1, max_hold + 1):
                close = paths[:, d]
                if is_long:
                    hit_today = close <= entry * (1.0 - stop_pct)
                else:
                    hit_today = close >= entry * (1.0 + stop_pct)
                newly = hit_today & still_open
                if newly.any():
                    exit_day[newly] = d
                    stop_hit[newly] = True
                    still_open[newly] = False
                if not still_open.any():
                    break
        rows = np.arange(n_paths)
        exit_price_tmp = paths[rows, exit_day]
        c = float(cost)
        if is_long:
            pnl = exit_price_tmp * (1.0 - c) - entry * (1.0 + c)
        else:
            pnl = entry * (1.0 - c) - exit_price_tmp * (1.0 + c)
        pnl_pct = np.divide(pnl, entry, out=np.zeros_like(pnl), where=entry != 0)

    rows = np.arange(n_paths)
    exit_price = paths[rows, exit_day]

    return {
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "n_trades": np.ones(n_paths, dtype=np.int64),
        "stop_hit": stop_hit,
        "take_profit_hit": np.zeros(n_paths, dtype=bool),
        "trailing_stop_hit": np.zeros(n_paths, dtype=bool),
        "exit_reason": np.where(stop_hit, EXIT_STOP_LOSS, EXIT_MAX_HOLDING).astype(object),
        "entry_day": np.zeros(n_paths, dtype=np.int64),
        "exit_day": exit_day,
        "holding_days": exit_day.astype(np.int64),
        "entry_price": entry,
        "exit_price": exit_price,
        "side": side,
        "numba_used": _classic_stop_hold_kernel_jit is not None,
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_tactical_stats(
    outcomes: Dict[str, np.ndarray],
    *,
    n_paths: int,
) -> Dict[str, Any]:
    pnl = np.asarray(outcomes["pnl"], dtype=np.float64)
    pnl_pct = np.asarray(outcomes["pnl_pct"], dtype=np.float64)
    n_trades = np.asarray(outcomes["n_trades"], dtype=np.float64)
    stop_hit = np.asarray(outcomes["stop_hit"], dtype=bool)
    tp_hit = np.asarray(outcomes.get("take_profit_hit", np.zeros(n_paths, dtype=bool)), dtype=bool)
    trail_hit = np.asarray(
        outcomes.get("trailing_stop_hit", np.zeros(n_paths, dtype=bool)), dtype=bool
    )
    exit_reason = outcomes["exit_reason"]
    holding = np.asarray(outcomes["holding_days"], dtype=np.float64)

    eps = 1e-12
    profit = pnl > eps
    loss = pnl < -eps
    flat = ~(profit | loss)
    max_hold_exits = np.array([r == EXIT_MAX_HOLDING for r in exit_reason], dtype=bool)

    if n_paths == 0:
        return {"n_paths": 0}

    return {
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
        "take_profit_rate": float(np.mean(tp_hit)),
        "trailing_stop_rate": float(np.mean(trail_hit)),
        "max_hold_exit_rate": float(np.mean(max_hold_exits)),
        "avg_holding_days": float(np.mean(holding)),
        "median_holding_days": float(np.median(holding)),
        "numba_available": _numba_available(),
        "numba_used": bool(outcomes.get("numba_used", False)),
    }


# ---------------------------------------------------------------------------
# Historical mode
# ---------------------------------------------------------------------------


def run_historical_rule_backtest(
    prices: np.ndarray,
    rule: TradingRule,
    *,
    horizon_days: int,
    cost: float = 0.0,
    step: int = 1,
    entry_fn: Optional[Callable[[int, np.ndarray], bool]] = None,
) -> HistoricalTacticalResult:
    """
    Slide a window of length ``horizon_days + 1`` across a real price series
    and apply the trading rule to each window (same engine as MC paths).
    """
    rule.validate()
    p = np.asarray(prices, dtype=np.float64).ravel()
    notes: List[str] = []
    win = int(horizon_days) + 1
    if p.size < win:
        notes.append("Not enough historical prices for even one window.")
        empty = np.asarray([], dtype=np.float64)
        return HistoricalTacticalResult(
            prices_used=int(p.size),
            horizon_days=int(horizon_days),
            n_windows=0,
            pnl=empty,
            n_trades=empty.astype(np.int64),
            stop_hit=empty.astype(bool),
            stats={"n_windows": 0, "prob_profit": float("nan"), "avg_pnl": float("nan"),
                   "stop_hit_rate": float("nan")},
            notes=notes,
        )

    pnls: List[float] = []
    n_tr: List[int] = []
    stops: List[bool] = []
    for start in range(0, p.size - win + 1, max(1, int(step))):
        window = p[start: start + win]
        trades, meta = apply_rule_to_one_path(
            window, rule, cost=cost, path_index=start, entry_fn=entry_fn
        )
        pnls.append(float(meta["total_pnl"]))
        n_tr.append(int(meta["n_trades"]))
        stops.append(any(t.stop_hit for t in trades) if trades else False)

    pnl_a = np.asarray(pnls, dtype=np.float64)
    n_a = np.asarray(n_tr, dtype=np.int64)
    s_a = np.asarray(stops, dtype=bool)
    eps = 1e-12
    stats = {
        "n_windows": int(pnl_a.size),
        "prob_profit": float(np.mean(pnl_a > eps)) if pnl_a.size else float("nan"),
        "prob_loss": float(np.mean(pnl_a < -eps)) if pnl_a.size else float("nan"),
        "avg_pnl": float(np.mean(pnl_a)) if pnl_a.size else float("nan"),
        "median_pnl": float(np.median(pnl_a)) if pnl_a.size else float("nan"),
        "worst_pnl": float(np.min(pnl_a)) if pnl_a.size else float("nan"),
        "best_pnl": float(np.max(pnl_a)) if pnl_a.size else float("nan"),
        "stop_hit_rate": float(np.mean(s_a)) if s_a.size else float("nan"),
        "avg_trades_per_window": float(np.mean(n_a)) if n_a.size else float("nan"),
    }
    return HistoricalTacticalResult(
        prices_used=int(p.size),
        horizon_days=int(horizon_days),
        n_windows=int(pnl_a.size),
        pnl=pnl_a,
        n_trades=n_a,
        stop_hit=s_a,
        stats=stats,
        notes=notes,
    )


def compare_historical_vs_mc(
    historical: HistoricalTacticalResult,
    mc: TacticalResult,
) -> Dict[str, Any]:
    """Side-by-side comparison of historical window stats vs MC distribution."""
    hs, ms = historical.stats, mc.stats
    return {
        "hist_n_windows": hs.get("n_windows"),
        "mc_n_paths": ms.get("n_paths"),
        "hist_prob_profit": hs.get("prob_profit"),
        "mc_prob_profit": ms.get("prob_profit"),
        "hist_avg_pnl": hs.get("avg_pnl"),
        "mc_avg_pnl": ms.get("avg_pnl"),
        "hist_stop_hit_rate": hs.get("stop_hit_rate"),
        "mc_stop_hit_rate": ms.get("stop_hit_rate"),
        "prob_profit_gap": (
            None
            if hs.get("prob_profit") is None or ms.get("prob_profit") is None
            else float(hs["prob_profit"]) - float(ms["prob_profit"])
        ),
    }


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def run_tactical_simulation(
    tactical: Optional[TacticalConfig] = None,
    *,
    rule: Optional[TradingRule] = None,
    entry_fn: Optional[Callable[[int, np.ndarray], bool]] = None,
    historical_prices: Optional[np.ndarray] = None,
    run_var_backtest: bool = False,
    variance_reduction: str = "none",
    # --- TradingView integration (Phase 3 bridge file) ---
    use_tradingview: bool = False,
    tv_data_dir: str | Path = "tv_data",
    tv_signal_path: Optional[str | Path] = None,
    tv_use_ticker: bool = True,
    tv_use_price: bool = True,
    tv_align_side: bool = True,
    tv_filter_against_trend: bool = False,
    tv_scale_vol: bool = True,
    tv_scale_jumps: bool = True,
    tv_require_signal: bool = False,
    **overrides: Any,
) -> TacticalResult:
    """
    Run a full short-horizon tactical Monte Carlo (+ optional historical / TV).

    Parameters
    ----------
    tactical :
        Phase 1/2 ``TacticalConfig``. Defaults to 5-day preset.
    rule :
        Optional rule override.
    entry_fn :
        Optional callable entry signal ``(day, prices_so_far) -> bool``.
    historical_prices :
        If provided, also run rolling historical rule backtest and attach it.
    run_var_backtest :
        If True and historical prices are available (or can be fetched), run
        rolling VaR coverage + Kupiec test as a calibration sanity check.
    variance_reduction :
        Forwarded to ``mc_core.simulate`` (``none`` / ``antithetic`` / ``sobol`` /
        ``control_variate``).
    use_tradingview :
        If True, load ``tv_data/latest_signal.json`` (or ``tv_signal_path``) and
        apply trend/momentum/price to the run.
    tv_require_signal :
        If True with ``use_tradingview``, raise when no signal file is found.
    **overrides :
        Field overrides on the config (e.g. ``paths=20_000``).
    """
    if tactical is None:
        tactical = preset_5_day()
    if overrides:
        from dataclasses import replace as dc_replace
        tactical = dc_replace(tactical, **overrides)

    tactical = tactical.validate()
    notes: List[str] = []
    tv_ctx: Optional[TVSignalContext] = None
    jump_mult = 1.0

    active_rule = rule if rule is not None else tactical.trading_rule
    if active_rule is None:
        raise ValueError(
            "No trading rule provided. Attach one with "
            "TacticalConfig.with_rule(...) or pass rule=..."
        )
    active_rule = active_rule.validate()

    # ---- Optional TradingView context ---------------------------------
    if use_tradingview:
        sig_path = resolve_signal_path(tv_data_dir, tv_signal_path)
        signal = load_tradingview_signal(tv_data_dir, tv_signal_path)
        if signal is None and tv_require_signal:
            raise FileNotFoundError(
                f"TradingView signal required but not found at {sig_path}. "
                "Run tv_webhook_bridge.py and fire an alert, or use "
                "tv_integration.write_demo_signal() for an offline demo."
            )
        tactical, active_rule, tv_ctx, jump_mult = apply_tradingview_to_tactical(
            tactical,
            active_rule,
            signal,
            use_ticker=tv_use_ticker,
            use_price=tv_use_price,
            align_side_to_trend=tv_align_side,
            filter_against_trend=tv_filter_against_trend,
            scale_vol_by_momentum=tv_scale_vol,
            scale_jumps_by_momentum=tv_scale_jumps,
            signal_path=str(sig_path),
            base_sigma=tactical.annual_volatility,
        )
        notes.extend(tv_ctx.notes)
        if tv_ctx.used:
            notes.append("Simulation used live TradingView bridge data.")
        else:
            notes.append("TradingView mode on, but no signal file — ran without TV data.")

        # If sigma was still None, resolve market then apply vol multiplier.
        if (
            tv_ctx.used
            and tv_scale_vol
            and tactical.annual_volatility is None
            and tv_ctx.vol_multiplier != 1.0
        ):
            _s0, _mu, base_sig, _src = resolve_market_parameters(tactical)
            from dataclasses import replace as dc_replace
            tactical = dc_replace(
                tactical,
                starting_price=tactical.starting_price or _s0,
                annual_drift=tactical.annual_drift,
                annual_volatility=float(base_sig) * float(tv_ctx.vol_multiplier),
            )
            notes.append(
                f"Post-market vol scale: {base_sig:.4f} × {tv_ctx.vol_multiplier:.3f} "
                f"→ {tactical.annual_volatility:.4f}"
            )

        # Block all entries when trend filter says no
        if tv_ctx is not None and not tv_ctx.trades_allowed:
            def _block(_day, _px):
                return False
            entry_fn = _block
            notes.append("All path entries blocked by TV trend filter.")

    active_rule = active_rule.validate()
    tactical = tactical.validate()

    if active_rule.max_holding_days > tactical.horizon_days:
        raise ValueError(
            f"rule.max_holding_days ({active_rule.max_holding_days}) exceeds "
            f"horizon_days ({tactical.horizon_days})"
        )

    t0 = time.perf_counter()
    price_paths, sim_cfg, market_source, _eng_rt = generate_price_paths(
        tactical,
        variance_reduction=variance_reduction,
        jump_intensity_multiplier=jump_mult,
    )
    if variance_reduction != "none":
        notes.append(f"MC variance_reduction requested: {variance_reduction}")

    side = infer_side(active_rule)
    outcomes = apply_rule_to_paths(
        price_paths,
        active_rule,
        cost=float(tactical.transaction_cost),
        side=side,
        entry_fn=entry_fn,
    )
    stats = compute_tactical_stats(outcomes, n_paths=price_paths.shape[0])
    runtime = time.perf_counter() - t0
    stats["runtime_seconds"] = runtime
    stats["engine_s0"] = float(sim_cfg.s0)
    stats["engine_mu"] = float(sim_cfg.mu)
    stats["engine_sigma"] = float(sim_cfg.sigma)
    stats["engine_model"] = sim_cfg.model
    stats["variance_reduction"] = variance_reduction
    stats["used_tradingview_data"] = bool(tv_ctx.used) if tv_ctx is not None else False
    stats["jump_intensity_multiplier"] = float(jump_mult)
    if tv_ctx is not None and tv_ctx.used:
        stats["tv_trend"] = tv_ctx.trend
        stats["tv_momentum"] = tv_ctx.momentum
        stats["tv_vol_multiplier"] = tv_ctx.vol_multiplier

    hist_result: Optional[HistoricalTacticalResult] = None
    bt: Optional[Dict[str, Any]] = None

    prices_for_hist = historical_prices
    if prices_for_hist is None and (run_var_backtest):
        # Best-effort fetch for VaR coverage only
        try:
            mkt = estimate_parameters_from_history(tactical.ticker)
            if mkt.daily_log_returns is not None and mkt.daily_log_returns.size > 2:
                # reconstruct a pseudo price path from log returns
                lr = np.asarray(mkt.daily_log_returns, dtype=float)
                px = float(mkt.s0) * np.exp(np.cumsum(np.r_[0.0, lr]))
                # re-scale so last price is s0
                px = px * (float(mkt.s0) / px[-1])
                prices_for_hist = px
                notes.append("Synthesized price path from historical log returns for backtest.")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Could not build historical series: {type(exc).__name__}")

    if historical_prices is not None:
        hist_result = run_historical_rule_backtest(
            historical_prices,
            active_rule,
            horizon_days=tactical.horizon_days,
            cost=float(tactical.transaction_cost),
            entry_fn=entry_fn,
        )
        stats["historical"] = hist_result.stats
        notes.extend(hist_result.notes)

    if run_var_backtest and prices_for_hist is not None:
        bt = rolling_var_coverage(prices_for_hist, window=min(252, max(20, len(prices_for_hist) // 4)))
        stats["var_backtest"] = {
            "breach_rate": bt.get("breach_rate"),
            "kupiec_p_value": bt.get("kupiec", {}).get("p_value"),
            "n_breaches": bt.get("n_breaches"),
            "n_forecasts": bt.get("n_forecasts"),
        }
        if bt.get("warning"):
            notes.append(str(bt["warning"]))

    return TacticalResult(
        config=tactical,
        rule=active_rule,
        side=side,
        pnl=outcomes["pnl"],
        pnl_pct=outcomes["pnl_pct"],
        n_trades=outcomes["n_trades"],
        stop_hit=outcomes["stop_hit"],
        take_profit_hit=outcomes["take_profit_hit"],
        trailing_stop_hit=outcomes["trailing_stop_hit"],
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
        notes=notes,
        historical=hist_result,
        backtest=bt,
        tv_context=tv_ctx,
    )


# ---------------------------------------------------------------------------
# CLI helper used by monte_carlo_gbm.py
# ---------------------------------------------------------------------------


def run_tactical_cli(
    ticker: str = "AAPL",
    *,
    horizon: int = 5,
    paths: int = 20_000,
    seed: int = 42,
    stop_loss: float = 0.02,
    take_profit: Optional[float] = None,
    trailing_stop: Optional[float] = None,
    max_holding: Optional[int] = None,
    side: str = "long",
    cost: float = 0.001,
    s0: Optional[float] = None,
    sigma: Optional[float] = None,
    allow_reentry: bool = False,
    max_trades: int = 1,
    historical: bool = False,
    var_backtest: bool = False,
    variance_reduction: str = "none",
    use_tradingview: bool = False,
    tv_data_dir: str = "tv_data",
    tv_signal_path: Optional[str] = None,
    tv_align_side: bool = True,
    tv_filter_against_trend: bool = False,
    tv_require_signal: bool = False,
) -> TacticalResult:
    """Build a config/rule from CLI-like kwargs and run the tactical simulator."""
    from tactical_config import preset_5_day, preset_10_day

    hold = int(max_holding) if max_holding is not None else int(horizon)
    if horizon <= 5:
        cfg = preset_5_day(ticker, paths=paths, seed=seed, transaction_cost=cost)
        # force horizon if user asked for something other than 5
        from dataclasses import replace as dc_replace
        cfg = dc_replace(cfg, horizon_days=int(horizon))
    else:
        cfg = preset_10_day(ticker, paths=paths, seed=seed, transaction_cost=cost)
        from dataclasses import replace as dc_replace
        cfg = dc_replace(cfg, horizon_days=int(horizon))

    if s0 is not None:
        from dataclasses import replace as dc_replace
        cfg = dc_replace(cfg, starting_price=float(s0))
    if sigma is not None:
        from dataclasses import replace as dc_replace
        cfg = dc_replace(cfg, annual_volatility=float(sigma))

    rule = TradingRule(
        name=f"CLI {side} {horizon}d",
        entry_condition=f"Enter {side} at start",
        exit_condition="Exit on stop/TP/trail/max hold",
        stop_loss_pct=float(stop_loss),
        max_holding_days=hold,
        side=side,
        take_profit_pct=take_profit,
        trailing_stop_pct=trailing_stop,
        allow_reentry=allow_reentry,
        max_trades=max_trades,
    )
    cfg = cfg.with_rule(rule)

    hist_px = None
    if historical or var_backtest:
        mkt = estimate_parameters_from_history(ticker, s0_override=s0)
        if mkt.daily_log_returns is not None and mkt.daily_log_returns.size > 2:
            lr = np.asarray(mkt.daily_log_returns, dtype=float)
            px = float(mkt.s0) * np.exp(np.cumsum(np.r_[0.0, lr]))
            hist_px = px * (float(mkt.s0) / px[-1])

    return run_tactical_simulation(
        cfg,
        historical_prices=hist_px if historical else None,
        run_var_backtest=var_backtest,
        variance_reduction=variance_reduction,
        use_tradingview=use_tradingview,
        tv_data_dir=tv_data_dir,
        tv_signal_path=tv_signal_path,
        tv_align_side=tv_align_side,
        tv_filter_against_trend=tv_filter_against_trend,
        tv_require_signal=tv_require_signal,
    )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from tactical_config import preset_5_day, preset_10_day, TradingRule

    print("Running Phase 2 demo (5-day, TP+stop, 20,000 paths)...\n")
    rule = TradingRule(
        name="Long 5d, 2% stop, 3% TP",
        entry_condition="Enter long at the start of day 1",
        exit_condition="TP, stop, or max hold",
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        max_holding_days=5,
        side="long",
    )
    cfg = preset_5_day(
        "AAPL",
        paths=20_000,
        seed=42,
        starting_price=100.0,
        annual_volatility=0.25,
        annual_drift=0.0,
    ).with_rule(rule)
    result = run_tactical_simulation(cfg)
    print(result.summary_text())
    print()

    # Synthetic historical series for offline demo
    rng = np.random.default_rng(0)
    steps = 400
    rets = rng.normal(0, 0.01, size=steps)
    hist = 100.0 * np.exp(np.cumsum(np.r_[0.0, rets]))
    result_h = run_tactical_simulation(
        cfg,
        historical_prices=hist,
        run_var_backtest=True,
    )
    print("\n--- With synthetic historical + VaR backtest ---\n")
    print(result_h.summary_text())
