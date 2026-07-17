"""
TradingView → tactical Monte Carlo integration (Phases 1–3 connected).

Plain English
-------------
Phase 3 saves the latest TradingView alert into ``tv_data/latest_signal.json``.
This module **reads that file** and turns trend / momentum / price into
concrete knobs for a short-horizon tactical run:

  * **Ticker / price**  → starting point of the simulation
  * **Trend**           → preferred trade side (bullish→long, bearish→short)
  * **Momentum (RSI)**  → scale annual volatility (and optional jump intensity)

Nothing here talks to TradingView over the network. It only reads the JSON
file that ``tv_webhook_bridge.py`` already wrote.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tactical_config import TacticalConfig, TradingRule

# Default folder written by tv_webhook_bridge.py
DEFAULT_TV_DATA_DIR = Path("tv_data")
DEFAULT_TV_LATEST = "latest_signal.json"


@dataclass
class TVSignalContext:
    """
    What we learned from the latest TradingView file (or a mock dict).

    ``used`` is True only when a real signal was loaded and applied.
    """

    used: bool = False
    path: Optional[str] = None
    ticker: Optional[str] = None
    price: Optional[float] = None
    trend: Optional[str] = None
    momentum: Optional[float] = None
    timeframe: Optional[str] = None
    strategy: Optional[str] = None
    received_at_utc: Optional[str] = None
    # How we changed the tactical run
    side_aligned: Optional[str] = None
    vol_multiplier: float = 1.0
    jump_multiplier: float = 1.0
    trades_allowed: bool = True
    notes: List[str] = None  # type: ignore[assignment]
    raw: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []

    def as_dict(self) -> Dict[str, Any]:
        return {
            "used_tradingview_data": self.used,
            "signal_path": self.path,
            "ticker": self.ticker,
            "price": self.price,
            "trend": self.trend,
            "momentum": self.momentum,
            "timeframe": self.timeframe,
            "strategy": self.strategy,
            "received_at_utc": self.received_at_utc,
            "side_aligned": self.side_aligned,
            "vol_multiplier": self.vol_multiplier,
            "jump_multiplier": self.jump_multiplier,
            "trades_allowed": self.trades_allowed,
            "notes": list(self.notes),
        }

    def summary_lines(self) -> List[str]:
        if not self.used:
            return [
                "TradingView data:    NOT used",
                *(f"  * {n}" for n in self.notes),
            ]
        lines = [
            "TradingView data:    YES (live bridge file)",
            f"  Signal file:      {self.path}",
            f"  Received (UTC):   {self.received_at_utc}",
            f"  TV ticker:        {self.ticker}",
            f"  TV price:         {self.price}",
            f"  TV trend:         {self.trend}",
            f"  TV momentum:      {self.momentum}",
            f"  TV timeframe:     {self.timeframe}",
            f"  Side from trend:  {self.side_aligned}",
            f"  Vol multiplier:   {self.vol_multiplier:.3f}",
            f"  Jump multiplier:  {self.jump_multiplier:.3f}",
            f"  Trades allowed:   {self.trades_allowed}",
        ]
        lines.extend(f"  * {n}" for n in self.notes)
        return lines


def resolve_signal_path(
    data_dir: Path | str = DEFAULT_TV_DATA_DIR,
    signal_path: Optional[Path | str] = None,
) -> Path:
    """Return the path we will try to load."""
    if signal_path is not None:
        return Path(signal_path)
    return Path(data_dir) / DEFAULT_TV_LATEST


def load_tradingview_signal(
    data_dir: Path | str = DEFAULT_TV_DATA_DIR,
    signal_path: Optional[Path | str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Load the latest TradingView signal JSON, or return None if missing/invalid.

    Prefers :func:`tv_webhook_bridge.load_latest` when the package layout allows
    it; otherwise reads the file directly (same schema).
    """
    path = resolve_signal_path(data_dir, signal_path)
    # Try bridge helper first (keeps one parser for free-form / normalized files)
    try:
        from tv_webhook_bridge import load_latest, DEFAULT_DATA_DIR

        if signal_path is None and Path(data_dir) == Path(DEFAULT_DATA_DIR):
            data = load_latest(Path(data_dir))
            if data is not None:
                return data
        # Fall through to direct read for custom paths
    except Exception:
        pass

    if not path.is_file():
        return None
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def trend_to_side(trend: Optional[str]) -> Optional[str]:
    """Map TV trend text to long/short. Unknown → None."""
    if not trend:
        return None
    t = str(trend).strip().lower()
    if t in ("bullish", "bull", "long", "up", "buy"):
        return "long"
    if t in ("bearish", "bear", "short", "down", "sell"):
        return "short"
    return None


def momentum_to_vol_multiplier(
    momentum: Optional[float],
    *,
    strength: float = 0.40,
) -> float:
    """
    Turn an RSI-like momentum (0–100) into a volatility scale factor.

    * RSI 50 → 1.0 (unchanged)
    * RSI 90 → up to ``1 + strength`` (more lively market)
    * RSI 10 → down toward ``1 - strength`` (quieter), floored at 0.5

    ``strength`` is how strong the effect is (default ±40% at the extremes).
    """
    if momentum is None:
        return 1.0
    try:
        m = float(momentum)
    except (TypeError, ValueError):
        return 1.0
    # Center at 50, normalize to roughly [-1, 1]
    x = (m - 50.0) / 50.0
    x = max(-1.0, min(1.0, x))
    mult = 1.0 + strength * x
    return float(max(0.50, min(2.0, mult)))


def momentum_to_jump_multiplier(
    momentum: Optional[float],
    *,
    strength: float = 0.50,
) -> float:
    """
    Higher jump intensity when momentum is *extreme* (far from 50).

    RSI 50 → 1.0; RSI 20 or 80 → up to ``1 + strength``.
    Useful if the engine model is jump-diffusion; harmless for plain GBM.
    """
    if momentum is None:
        return 1.0
    try:
        m = float(momentum)
    except (TypeError, ValueError):
        return 1.0
    extremity = abs(m - 50.0) / 50.0  # 0 at center, 1 at 0 or 100
    extremity = max(0.0, min(1.0, extremity))
    mult = 1.0 + strength * extremity
    return float(max(1.0, min(2.5, mult)))


def apply_tradingview_to_tactical(
    tactical: TacticalConfig,
    rule: TradingRule,
    signal: Optional[Dict[str, Any]],
    *,
    use_ticker: bool = True,
    use_price: bool = True,
    align_side_to_trend: bool = True,
    filter_against_trend: bool = False,
    scale_vol_by_momentum: bool = True,
    scale_jumps_by_momentum: bool = True,
    momentum_vol_strength: float = 0.40,
    signal_path: Optional[str] = None,
    base_sigma: Optional[float] = None,
) -> Tuple[TacticalConfig, TradingRule, TVSignalContext, float]:
    """
    Apply a TradingView signal dict to config + rule.

    Returns
    -------
    tactical, rule, context, jump_intensity_multiplier

    The jump multiplier is returned separately so ``build_simulation_config``
    / the runner can push it into ``SimulationConfig`` when useful.
    """
    ctx = TVSignalContext(used=False, path=signal_path, notes=[])

    if not signal:
        ctx.notes.append("No TradingView signal file found (or empty).")
        return tactical, rule, ctx, 1.0

    ctx.used = True
    ctx.raw = dict(signal)
    ctx.ticker = _as_str(signal.get("ticker") or signal.get("symbol"))
    ctx.price = _as_float(signal.get("price") or signal.get("close"))
    ctx.trend = _as_str(signal.get("trend") or signal.get("direction"))
    if ctx.trend:
        ctx.trend = ctx.trend.lower()
    ctx.momentum = _as_float(signal.get("momentum") or signal.get("rsi"))
    ctx.timeframe = _as_str(signal.get("timeframe") or signal.get("tf"))
    ctx.strategy = _as_str(signal.get("strategy"))
    ctx.received_at_utc = _as_str(signal.get("received_at_utc"))
    ctx.path = signal_path

    # --- ticker ---
    if use_ticker and ctx.ticker and ctx.ticker.upper() not in ("UNKNOWN", ""):
        tactical = replace(tactical, ticker=ctx.ticker.upper())
        ctx.notes.append(f"Ticker set from TV: {tactical.ticker}")

    # --- starting price ---
    if use_price and ctx.price is not None and ctx.price > 0:
        tactical = replace(tactical, starting_price=float(ctx.price))
        ctx.notes.append(f"Starting price set from TV: {ctx.price}")

    # --- trend → side ---
    aligned = trend_to_side(ctx.trend)
    ctx.side_aligned = aligned
    if align_side_to_trend and aligned is not None:
        rule = replace(
            rule,
            side=aligned,
            entry_condition=f"Enter {aligned} (aligned to TV trend={ctx.trend})",
        )
        ctx.notes.append(f"Trade side aligned to TV trend → {aligned}")

    # Optional: block trades if user-fixed side fights the TV trend
    if filter_against_trend and aligned is not None:
        current_side = (rule.side or "long").lower()
        if current_side != aligned:
            ctx.trades_allowed = False
            ctx.notes.append(
                f"Trades blocked: rule side={current_side} conflicts with "
                f"TV trend={ctx.trend} ({aligned})."
            )

    # --- momentum → vol / jumps ---
    vol_m = 1.0
    jump_m = 1.0
    if scale_vol_by_momentum:
        vol_m = momentum_to_vol_multiplier(
            ctx.momentum, strength=momentum_vol_strength
        )
        ctx.vol_multiplier = vol_m
        # Apply to annual_volatility if we have a base
        base = base_sigma
        if base is None:
            base = tactical.annual_volatility
        if base is None:
            # Will be filled from market later; store multiplier in notes only.
            # Runner will re-apply after market resolve if needed.
            ctx.notes.append(
                f"Vol multiplier from momentum={ctx.momentum}: ×{vol_m:.3f} "
                "(applied once sigma is known)"
            )
        else:
            new_sigma = float(base) * vol_m
            tactical = replace(tactical, annual_volatility=new_sigma)
            ctx.notes.append(
                f"Volatility scaled by momentum RSI={ctx.momentum}: "
                f"{base:.4f} × {vol_m:.3f} → {new_sigma:.4f}"
            )
    if scale_jumps_by_momentum:
        jump_m = momentum_to_jump_multiplier(ctx.momentum)
        ctx.jump_multiplier = jump_m
        if jump_m != 1.0:
            ctx.notes.append(
                f"Jump-intensity multiplier from momentum extremity: ×{jump_m:.3f} "
                "(used if the MC model includes jumps)"
            )

    return tactical, rule, ctx, jump_m


def _as_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _as_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def write_demo_signal(
    path: Path | str = DEFAULT_TV_DATA_DIR / DEFAULT_TV_LATEST,
    *,
    ticker: str = "AAPL",
    price: float = 190.0,
    trend: str = "bullish",
    momentum: float = 62.0,
    timeframe: str = "5",
) -> Path:
    """
    Write a fake TradingView signal file for offline demos (no webhook needed).
    """
    import json
    from datetime import datetime, timezone

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "received_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": "demo",
        "ticker": ticker.upper(),
        "price": float(price),
        "trend": trend.lower(),
        "momentum": float(momentum),
        "timeframe": timeframe,
        "strategy": "demo_signal",
        "parse_status": "demo",
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path
