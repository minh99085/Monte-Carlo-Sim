"""Dashboard aggregator — brain + hands + signal, one JSON blob.

Read-only view over the trading system's on-disk state. Consumed by the
``/dashboard/api`` route in ``tv_webhook_bridge.py`` and (indirectly) by the
static HTML at ``/dashboard/``.

Sources:
    * Monte-Carlo-Sim (files under ``APP_DIR``):
        - tv_data/latest_signal.json           TradingView signal
        - outputs/paper_log.jsonl              paper training ledger
        - outputs/verdicts/*.json              live verdicts
        - outputs/paper_verdicts/*.json        paper training verdicts
        - calibration/*.json                   fitted drift tables

    * Robinhood bot (via its localhost HTTP API on 127.0.0.1:8810):
        - /api/health                          live-flag + MCP status
        - /api/robinhood/mc-bridge             bridge ledger + counts
      Server-side fetch keeps the bot's Docker volume out of the host's
      filesystem trust boundary (the bot API is bound to 127.0.0.1 only).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import outcome_tracker

DEFAULT_APP_DIR = Path("/opt/monte-carlo-sim")
DEFAULT_BOT_API = "http://127.0.0.1:8810"


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _stat_age_seconds(path: Path) -> Optional[float]:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def _iso_utc(ts: float) -> str:
    return (datetime.fromtimestamp(ts, tz=timezone.utc)
            .replace(microsecond=0).isoformat())


def _tail_jsonl(path: Path, n: int) -> List[Dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: List[Dict[str, Any]] = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                out.append(row)
        except ValueError:
            continue
    return out


def _http_json(url: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data if isinstance(data, dict) else None
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _signal_section(app_dir: Path) -> Dict[str, Any]:
    """Latest TradingView alert + freshness."""
    path = app_dir / "tv_data" / "latest_signal.json"
    signal = _read_json(path) if path.is_file() else None
    if not signal:
        return {
            "state": "waiting",
            "detail": "No TradingView alert received yet",
            "ticker": None,
            "trend": None,
            "rsi": None,
            "received_at_utc": None,
            "age_hours": None,
        }
    received = signal.get("received_at_utc")
    age_h: Optional[float] = None
    if received:
        try:
            then = datetime.fromisoformat(str(received))
            if then.tzinfo is None:
                then = then.replace(tzinfo=timezone.utc)
            age_h = (datetime.now(timezone.utc) - then).total_seconds() / 3600.0
        except ValueError:
            age_h = None
    if age_h is None:
        state, detail = "warn", "Signal has no timestamp"
    elif age_h > 30:
        state, detail = "warn", f"Last alert was {age_h:.1f}h ago — considered stale"
    else:
        state, detail = "good", f"Last alert {age_h:.1f}h ago"
    return {
        "state": state,
        "detail": detail,
        "ticker": signal.get("ticker"),
        "trend": signal.get("trend"),
        "rsi": signal.get("momentum") or signal.get("rsi"),
        "price": signal.get("price"),
        "received_at_utc": received,
        "age_hours": age_h,
    }


def _brain_section(app_dir: Path) -> Dict[str, Any]:
    """MC brain: paper track record + kill switch + calibration freshness."""
    paper_log = app_dir / "outputs" / "paper_log.jsonl"
    live_log = app_dir / "outputs" / "trade_log.jsonl"
    stats: Dict[str, Any] = {}
    ks_tripped = False
    ks_reason = ""
    if paper_log.is_file():
        try:
            stats = outcome_tracker.report_stats(paper_log)
            ks_tripped, ks_reason = outcome_tracker.check_kill_switch(paper_log)
        except Exception as exc:  # noqa: BLE001
            stats = {"error": str(exc)}
    calib_dir = app_dir / "calibration"
    calib_ages: List[float] = []
    if calib_dir.is_dir():
        for p in calib_dir.glob("*.json"):
            age = _stat_age_seconds(p)
            if age is not None:
                calib_ages.append(age)
    calib_age_days = (
        min(calib_ages) / 86400.0 if calib_ages else None
    )
    if not paper_log.is_file():
        state, detail = "warn", "No paper training yet"
    elif ks_tripped:
        state, detail = "critical", ks_reason
    elif not stats.get("n_settled_trades"):
        state, detail = "info", (
            f"{stats.get('n_trade_verdicts', 0)} TRADE verdicts logged; "
            "0 settled yet"
        )
    else:
        state, detail = "good", (
            f"{stats['n_settled_trades']} settled — "
            f"hit rate {stats.get('hit_rate', 0) * 100:.0f}%"
        )
    return {
        "state": state,
        "detail": detail,
        "kill_switch": {"tripped": ks_tripped, "reason": ks_reason},
        "stats": stats,
        "calibration_age_days": calib_age_days,
        "calibration_files": len(calib_ages),
        "live_verdicts_logged": (
            _line_count(live_log) if live_log.is_file() else 0
        ),
    }


def _line_count(path: Path) -> int:
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _bot_section(bot_api: str) -> Dict[str, Any]:
    """Robinhood bot: fetched via localhost HTTP so the tv-bridge does not
    have to reach into the bot's Docker volume."""
    health = _http_json(f"{bot_api}/api/health") or {}
    bridge = _http_json(f"{bot_api}/api/robinhood/mc-bridge") or {}
    reachable = bool(health)
    live = bool(health.get("live_trading_enabled"))
    if not reachable:
        state, detail = "warn", "Bot API not reachable on 127.0.0.1:8810"
    elif live:
        state, detail = "critical", "Live trading is ON — real orders can be placed"
    elif not health.get("mcp_connected"):
        state, detail = "info", "Paper mode; Robinhood MCP not connected (no OAuth)"
    else:
        state, detail = "good", "Paper mode, MCP connected"
    return {
        "state": state,
        "detail": detail,
        "reachable": reachable,
        "live_trading_enabled": live,
        "mcp_connected": bool(health.get("mcp_connected")),
        "tool_count": int(health.get("tool_count") or 0),
        "bridge": bridge,
    }


def _bridge_section(bot: Dict[str, Any]) -> Dict[str, Any]:
    """MC → bot bridge summary (folded out of the bot section for clarity)."""
    bridge = bot.get("bridge") or {}
    counts = bridge.get("counts") or {}
    ledger = bridge.get("recent") or []
    seen = int(counts.get("total") or 0)
    planned = int(counts.get("planned") or 0)
    skipped = int(counts.get("skipped") or 0)
    gate_blocked = int(counts.get("gate_blocked") or 0)
    if not bot.get("reachable"):
        state, detail = "warn", "Bridge status unknown (bot API unreachable)"
    elif not counts:
        state, detail = "info", "Bridge running — no verdicts processed yet"
    elif planned == 0 and (skipped + gate_blocked) > 0:
        state, detail = "info", (
            f"{seen} verdicts read — none passable "
            f"(skipped {skipped}, gated {gate_blocked})"
        )
    else:
        state, detail = "good", (
            f"{planned} paper plans / {seen} verdicts read"
        )
    return {
        "state": state,
        "detail": detail,
        "counts": counts,
        "recent": ledger[-8:],
    }


def _verdicts_section(app_dir: Path, limit: int = 12) -> List[Dict[str, Any]]:
    """Latest verdicts from paper + live dirs, newest first."""
    dirs = [app_dir / "outputs" / "verdicts",
            app_dir / "outputs" / "paper_verdicts"]
    files: List[Path] = []
    for d in dirs:
        if d.is_dir():
            files.extend(d.glob("*.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out: List[Dict[str, Any]] = []
    for p in files[:limit]:
        v = _read_json(p) or {}
        sizing = v.get("sizing") or {}
        out.append({
            "timestamp_utc": v.get("timestamp_utc"),
            "ticker": v.get("ticker"),
            "verdict": v.get("verdict"),
            "side": v.get("side"),
            "reason": v.get("reason"),
            "shares": sizing.get("shares"),
            "edge_weekly": v.get("edge_weekly"),
            "source": ("live" if p.parent.name == "verdicts" else "paper"),
        })
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_snapshot(
    app_dir: Path = DEFAULT_APP_DIR,
    bot_api: str = DEFAULT_BOT_API,
) -> Dict[str, Any]:
    """Build the full dashboard payload. Never raises; every section returns
    a well-formed dict even when the underlying data is missing."""
    signal = _signal_section(app_dir)
    brain = _brain_section(app_dir)
    bot = _bot_section(bot_api)
    bridge = _bridge_section(bot)
    verdicts = _verdicts_section(app_dir)
    overall = _worst({s["state"] for s in (signal, brain, bot, bridge)})
    return {
        "generated_at_utc": _iso_utc(time.time()),
        "app_dir": str(app_dir),
        "overall_state": overall,
        "signal": signal,
        "brain": brain,
        "bridge": bridge,
        "bot": bot,
        "verdicts": verdicts,
    }


_STATE_RANK = {"good": 0, "info": 1, "warn": 2, "critical": 3}


def _worst(states) -> str:
    return max(states, key=lambda s: _STATE_RANK.get(s, 1), default="info")
