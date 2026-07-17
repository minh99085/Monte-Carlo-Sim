"""
Tactical (short-horizon) configuration for Monte Carlo simulations.

This module is Phase 1 of turning the existing multi-month Monte Carlo tool
into a practical 5–10 trading-day tactical trading simulator.

What lives here
---------------
1. **TacticalConfig** – easy knobs for short-horizon price path simulations
   (how many days ahead, how many stories to simulate, seed, costs, etc.).
2. **TradingRule** – a simple structure for describing a trade idea
   (when to enter, when to exit, stop loss, max hold time).
3. **Presets** – ready-made configs for "5 trading days" and "10 trading days".

Phase 1 only *defines* these settings and rules. It does not change the
original simulation engine yet. Later phases will wire them into mc_core.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Any, Callable, Dict, Optional

# Optional callable: entry_fn(day_index, prices_up_to_day) -> bool
EntryFn = Callable[[int, Any], bool]


# ---------------------------------------------------------------------------
# Shared constants (kept local so this file is easy to read on its own)
# ---------------------------------------------------------------------------

# U.S. equities typically have about 252 trading days in a calendar year.
TRADING_DAYS_PER_YEAR = 252

# Friendly names for the two main short-horizon presets.
PRESET_5_DAY = "5 trading days"
PRESET_10_DAY = "10 trading days"


# ---------------------------------------------------------------------------
# Trading rule structure
# ---------------------------------------------------------------------------


@dataclass
class TradingRule:
    """
    A simple description of one tactical trade idea.

    Think of this as a checklist for a short-term trade:

    * **Entry** – when do we open the position?
    * **Exit**  – when do we take profit / close for a normal reason?
    * **Stop**  – where do we cut the loss if the trade goes wrong?
    * **Max hold** – how many trading days will we stay in at most?

    In Phase 1 these are *descriptions* you can store and pass around.
    The simulation engine is not forced to execute them yet; that comes later.

    All percentage fields use **fractions**, not whole percents:
        0.02 means 2%,  0.05 means 5%.
    """

    # Human-readable name so you can tell rules apart in reports.
    # Example: "Momentum long, 5-day hold"
    name: str = "unnamed rule"

    # Plain-English (or short formula) description of when to enter.
    # Examples:
    #   "Enter long at the open of day 1"
    #   "Enter long if simulated day-1 return > +0.5%"
    #   "Enter short if price gaps down more than 1%"
    entry_condition: str = "Enter long at the start of the simulation"

    # Plain-English description of a normal (non-stop) exit.
    # Examples:
    #   "Exit at the close of the last holding day"
    #   "Exit if unrealized gain reaches +3%"
    #   "Exit if price closes above the prior day's high"
    exit_condition: str = "Exit at the end of the maximum holding period"

    # Stop-loss level as a fraction of entry price.
    # Example: 0.02 means "exit if price falls 2% from entry".
    # Use 0.0 only if you truly want no stop (not recommended for tactics).
    stop_loss_pct: float = 0.02

    # Hard cap on how long the trade can stay open (trading days).
    # Even if entry/exit conditions never fire, the position is closed
    # after this many days. Must be at least 1.
    max_holding_days: int = 5

    # Optional notes for humans reading the rule later (strategy thesis, etc.).
    notes: str = ""

    # ---- Phase 2 structured controls (all optional / backward-compatible) ----

    # Explicit side: "long", "short", or None (infer from entry_condition text).
    side: Optional[str] = None

    # First day index (0 = start of window) when an entry may be considered.
    entry_day: int = 0

    # Take-profit as a fraction of entry (e.g. 0.03 = +3%). None = disabled.
    take_profit_pct: Optional[float] = None

    # Trailing stop as a fraction from the favorable extreme since entry.
    # Long: trail from running high; short: trail from running low.
    # None = disabled. Example: 0.015 = 1.5% trail.
    trailing_stop_pct: Optional[float] = None

    # Allow opening another trade after an exit within the same path/window.
    allow_reentry: bool = False

    # Maximum completed round-trips per path/window (1 = classic single trade).
    max_trades: int = 1

    # Optional Python callable for flexible entry signals.
    # Signature: entry_fn(day: int, prices_so_far: ndarray) -> bool
    # Not serialized to JSON; use for research notebooks / unit tests.
    entry_fn: Optional[EntryFn] = None

    def validate(self) -> "TradingRule":
        """Check that the rule values make sense. Raises ValueError if not."""
        if not self.name or not str(self.name).strip():
            raise ValueError("TradingRule.name must be a non-empty string")
        if not self.entry_condition or not str(self.entry_condition).strip():
            raise ValueError("TradingRule.entry_condition must be non-empty")
        if not self.exit_condition or not str(self.exit_condition).strip():
            raise ValueError("TradingRule.exit_condition must be non-empty")
        if self.stop_loss_pct < 0.0 or self.stop_loss_pct >= 1.0:
            raise ValueError(
                "TradingRule.stop_loss_pct must be in [0, 1) "
                "(e.g. 0.02 for a 2% stop)"
            )
        if self.max_holding_days < 1:
            raise ValueError("TradingRule.max_holding_days must be >= 1")
        if self.side is not None and str(self.side).lower() not in ("long", "short"):
            raise ValueError("TradingRule.side must be 'long', 'short', or None")
        if self.entry_day < 0:
            raise ValueError("TradingRule.entry_day must be >= 0")
        if self.take_profit_pct is not None and self.take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be > 0 when set")
        if self.trailing_stop_pct is not None and not (0.0 < self.trailing_stop_pct < 1.0):
            raise ValueError("trailing_stop_pct must be in (0, 1) when set")
        if self.max_trades < 1:
            raise ValueError("max_trades must be >= 1")
        return self

    def summary(self) -> str:
        """One-line plain-English summary of the rule."""
        stop_txt = (
            f"{self.stop_loss_pct * 100:.2f}% stop"
            if self.stop_loss_pct > 0
            else "no stop"
        )
        bits = [
            f"{self.name}: enter=[{self.entry_condition}]; "
            f"exit=[{self.exit_condition}]; {stop_txt}; "
            f"max hold={self.max_holding_days} day(s)"
        ]
        if self.take_profit_pct:
            bits.append(f"TP={self.take_profit_pct * 100:.2f}%")
        if self.trailing_stop_pct:
            bits.append(f"trail={self.trailing_stop_pct * 100:.2f}%")
        if self.allow_reentry:
            bits.append(f"reentry(max_trades={self.max_trades})")
        if self.side:
            bits.append(f"side={self.side}")
        return "; ".join(bits) if len(bits) > 1 else bits[0]


# ---------------------------------------------------------------------------
# Tactical simulation configuration
# ---------------------------------------------------------------------------


@dataclass
class TacticalConfig:
    """
    Settings for a short-horizon (tactical) Monte Carlo run.

    The original tool defaults to a 252-day (about one year) horizon.
    This class defaults to **5 trading days** and other values that fit
    short-term tactical use.

    Field guide (plain English)
    ---------------------------
    ticker
        Stock or asset symbol, e.g. "AAPL" or "MSFT".
    horizon_days
        How many trading days into the future to simulate.
        For tactics this is usually 5 or 10 (not 252).
    paths
        How many possible "future price stories" to generate.
        More paths = smoother risk numbers, but slower.
        100,000 is a solid default for short horizons.
    seed
        Random seed for reproducibility. Same seed + same settings
        => same simulated results. Use None for a fresh random run.
    starting_price
        Optional fixed starting price. If None, later phases can pull
        the latest market close (via the existing market-data helpers).
    annual_drift
        Expected long-run annualized return used by GBM-style models.
        For short-horizon tactics, many people prefer 0.0 (zero drift)
        so results are driven more by volatility than by optimistic growth.
    annual_volatility
        Annualized volatility (e.g. 0.25 = 25% per year). If None,
        later phases can estimate it from history.
    transaction_cost
        Proportional cost / slippage on the trade, as a fraction.
        Example: 0.001 = 0.1% round-trip friction. Short-horizon trades
        feel costs more, so a small non-zero default is useful.
    sample_paths
        How many full price paths to keep for charts (not used for stats).
        Keep this small to save memory.
    trading_rule
        Optional TradingRule attached to this config (entry/exit/stop/hold).
    notes
        Free-text notes about why you are running this scenario.
    """

    # --- Identity ---
    ticker: str = "AAPL"

    # --- Horizon (the main tactical knob) ---
    # Number of forward *trading* days to simulate (not calendar days).
    horizon_days: int = 5

    # --- Simulation size ---
    # Number of Monte Carlo paths (future price stories).
    paths: int = 100_000

    # Optional seed so two runs with the same settings match exactly.
    seed: Optional[int] = 42

    # --- Market parameters (optional overrides) ---
    # If left as None, later integration can fill these from history.
    starting_price: Optional[float] = None
    annual_drift: Optional[float] = 0.0          # conservative short-horizon default
    annual_volatility: Optional[float] = None    # e.g. 0.25 for 25% annual vol

    # --- Friction ---
    # Proportional transaction cost / slippage, e.g. 0.001 = 0.10%.
    transaction_cost: float = 0.001

    # --- Plotting / diagnostics ---
    # Full trajectories retained only for charts (stats use all paths).
    sample_paths: int = 40

    # --- Attached trade idea (optional) ---
    trading_rule: Optional[TradingRule] = None

    # --- Human notes ---
    notes: str = ""

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> "TacticalConfig":
        """Raise ValueError if any setting is out of range."""
        if not self.ticker or not str(self.ticker).strip():
            raise ValueError("ticker must be a non-empty string")
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be >= 1 (use 5 or 10 for tactics)")
        if self.paths < 1:
            raise ValueError("paths must be >= 1")
        if self.starting_price is not None and self.starting_price <= 0:
            raise ValueError("starting_price must be > 0 when set")
        if self.annual_volatility is not None and self.annual_volatility < 0:
            raise ValueError("annual_volatility must be >= 0 when set")
        if not (0.0 <= self.transaction_cost < 1.0):
            raise ValueError("transaction_cost must be in [0, 1)")
        if self.sample_paths < 0:
            raise ValueError("sample_paths must be >= 0")
        if self.trading_rule is not None:
            self.trading_rule.validate()
            # Soft consistency check: rule hold period should not exceed horizon.
            if self.trading_rule.max_holding_days > self.horizon_days:
                raise ValueError(
                    "trading_rule.max_holding_days "
                    f"({self.trading_rule.max_holding_days}) cannot exceed "
                    f"horizon_days ({self.horizon_days})"
                )
        return self

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def with_rule(self, rule: TradingRule) -> "TacticalConfig":
        """Return a copy of this config with a trading rule attached."""
        return replace(self, trading_rule=rule.validate())

    def summary(self) -> str:
        """Short plain-English summary of the configuration."""
        rule_txt = (
            self.trading_rule.summary()
            if self.trading_rule is not None
            else "no trading rule attached"
        )
        drift_txt = (
            f"{self.annual_drift * 100:.2f}% ann."
            if self.annual_drift is not None
            else "from market history"
        )
        vol_txt = (
            f"{self.annual_volatility * 100:.2f}% ann."
            if self.annual_volatility is not None
            else "from market history"
        )
        return (
            f"{self.ticker}: {self.horizon_days} trading day(s), "
            f"{self.paths:,} paths, seed={self.seed}, "
            f"drift={drift_txt}, vol={vol_txt}, "
            f"cost={self.transaction_cost * 100:.3f}% | {rule_txt}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict (useful for JSON exports later)."""
        data = asdict(self)
        # Callables are not JSON-serializable — drop entry_fn if present.
        rule = data.get("trading_rule")
        if isinstance(rule, dict) and "entry_fn" in rule:
            rule["entry_fn"] = None if rule["entry_fn"] is None else "<callable>"
        return data

    def to_simulation_kwargs(self) -> Dict[str, Any]:
        """
        Map tactical settings onto the field names used by SimulationConfig
        in mc_core.py.

        Phase 1 only *prepares* this mapping. Callers in later phases can do:

            from mc_core import SimulationConfig
            cfg = SimulationConfig(**tactical.to_simulation_kwargs())

        Only fields that have a clear one-to-one match are included.
        Fields left as None here are omitted so SimulationConfig defaults
        (or market-data estimates) can fill them.
        """
        kwargs: Dict[str, Any] = {
            "ticker": self.ticker.strip().upper(),
            "paths": int(self.paths),
            "horizon": int(self.horizon_days),
            "dt": 1.0 / TRADING_DAYS_PER_YEAR,
            "cost": float(self.transaction_cost),
            "sample_paths": int(self.sample_paths),
            "seed": self.seed,
        }
        if self.starting_price is not None:
            kwargs["s0"] = float(self.starting_price)
        if self.annual_drift is not None:
            kwargs["mu"] = float(self.annual_drift)
        if self.annual_volatility is not None:
            kwargs["sigma"] = float(self.annual_volatility)
        return kwargs


# ---------------------------------------------------------------------------
# Ready-made presets
# ---------------------------------------------------------------------------


def _default_rule_for_horizon(horizon_days: int) -> TradingRule:
    """
    A sensible starter trading rule matched to the simulation horizon.

    - Enter at the start of the window.
    - Exit at the end of the max hold (same as the horizon).
    - Use a 2% stop loss (typical short-term risk budget).
    """
    return TradingRule(
        name=f"Default long, {horizon_days}-day hold",
        entry_condition="Enter long at the start of day 1",
        exit_condition=(
            f"Exit at the close of day {horizon_days} "
            "(end of maximum holding period)"
        ),
        stop_loss_pct=0.02,
        max_holding_days=horizon_days,
        notes=(
            "Starter rule for short-horizon Monte Carlo. "
            "Replace entry/exit text with your own strategy language."
        ),
    ).validate()


def preset_5_day(
    ticker: str = "AAPL",
    **overrides: Any,
) -> TacticalConfig:
    """
    Ready-made config for a **5 trading day** tactical simulation.

    Defaults (why they matter for short-term use)
    ---------------------------------------------
    * horizon_days = 5     → about one trading week
    * paths = 100,000      → stable risk stats without a huge runtime
    * seed = 42            → reproducible by default
    * annual_drift = 0.0   → do not assume optimistic growth over 5 days
    * transaction_cost = 0.1% → short trades care about friction
    * includes a default long TradingRule with a 2% stop and 5-day max hold

    You can override any field by keyword, e.g.:

        cfg = preset_5_day("MSFT", paths=50_000, seed=7)
    """
    base = TacticalConfig(
        ticker=ticker,
        horizon_days=5,
        paths=100_000,
        seed=42,
        annual_drift=0.0,
        transaction_cost=0.001,
        sample_paths=40,
        trading_rule=_default_rule_for_horizon(5),
        notes=f"Preset: {PRESET_5_DAY}",
    )
    if overrides:
        base = replace(base, **overrides)
    return base.validate()


def preset_10_day(
    ticker: str = "AAPL",
    **overrides: Any,
) -> TacticalConfig:
    """
    Ready-made config for a **10 trading day** tactical simulation.

    Same philosophy as the 5-day preset, but for about two trading weeks.
    Max holding period on the default rule is also 10 days.
    """
    base = TacticalConfig(
        ticker=ticker,
        horizon_days=10,
        paths=100_000,
        seed=42,
        annual_drift=0.0,
        transaction_cost=0.001,
        sample_paths=40,
        trading_rule=_default_rule_for_horizon(10),
        notes=f"Preset: {PRESET_10_DAY}",
    )
    if overrides:
        base = replace(base, **overrides)
    return base.validate()


# Friendly lookup so callers can pick a preset by name if they prefer.
TACTICAL_PRESETS = {
    "5d": preset_5_day,
    "5": preset_5_day,
    PRESET_5_DAY: preset_5_day,
    "10d": preset_10_day,
    "10": preset_10_day,
    PRESET_10_DAY: preset_10_day,
}


def get_preset(name: str, ticker: str = "AAPL", **overrides: Any) -> TacticalConfig:
    """
    Look up a preset by a short name.

    Accepted names (case-insensitive for the short keys):
        "5", "5d", "5 trading days"
        "10", "10d", "10 trading days"
    """
    key = name.strip().lower()
    # Normalize a few common variants.
    aliases = {
        "5": "5d",
        "5d": "5d",
        "5 day": "5d",
        "5 days": "5d",
        "5 trading day": "5d",
        "5 trading days": "5d",
        "10": "10d",
        "10d": "10d",
        "10 day": "10d",
        "10 days": "10d",
        "10 trading day": "10d",
        "10 trading days": "10d",
    }
    normalized = aliases.get(key)
    if normalized is None:
        # Also allow the original display strings if passed as-is.
        if name in TACTICAL_PRESETS:
            return TACTICAL_PRESETS[name](ticker=ticker, **overrides)
        raise KeyError(
            f"Unknown tactical preset {name!r}. "
            f"Use one of: 5, 5d, 10, 10d (or '{PRESET_5_DAY}' / '{PRESET_10_DAY}')."
        )
    return TACTICAL_PRESETS[normalized](ticker=ticker, **overrides)


# ---------------------------------------------------------------------------
# Tiny self-check when run directly: python tactical_config.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    five = preset_5_day("AAPL")
    ten = preset_10_day("MSFT", paths=50_000)
    custom_rule = TradingRule(
        name="Gap-fade short",
        entry_condition="Enter short if open gaps up more than 1%",
        exit_condition="Cover if price reverts to prior close, or at day 3",
        stop_loss_pct=0.015,
        max_holding_days=3,
        notes="Example custom rule for illustration only.",
    )
    custom = preset_5_day("TSLA").with_rule(custom_rule)

    print("=== 5-day preset ===")
    print(five.summary())
    print("simulation kwargs:", five.to_simulation_kwargs())
    print()
    print("=== 10-day preset ===")
    print(ten.summary())
    print()
    print("=== Custom rule on 5-day config ===")
    print(custom.summary())
    print(custom.trading_rule.summary() if custom.trading_rule else "")
