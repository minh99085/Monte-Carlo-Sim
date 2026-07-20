"""Options watchlist scan loop — manual bias, review-before-place."""

from __future__ import annotations

import logging
import time
from collections import Counter
from typing import Any

from engine.robinhood.config import RobinhoodConfig
from engine.robinhood.constants import OPTIONS_TOOLS
from engine.robinhood.options_ledger import append_event, write_status
from engine.robinhood.options_market import SymbolMarketSnapshot, load_symbol_snapshot
from engine.robinhood.options_positions import (
    count_open_positions,
    fetch_option_positions,
    open_underlyings,
)
from engine.robinhood.options_readiness import evaluate_readiness, write_readiness_report
from engine.robinhood.options_state import (
    record_symbol_action,
    save_chain_snapshot,
    symbol_in_cooldown,
)
from engine.robinhood.options_strategy import OrderIntent, decide_order
from engine.robinhood.safety_gates import _extract_warnings

logger = logging.getLogger("hermes.robinhood.options_loop")


def _intent_dict(intent: OrderIntent) -> dict[str, Any]:
    return {
        "symbol": intent.symbol,
        "option_type": intent.option_type,
        "instrument_id": intent.instrument_id,
        "quantity": intent.quantity,
        "limit_price": intent.limit_price,
        "premium_usd": intent.premium_usd,
        "expiration_date": intent.expiration_date,
        "strike": intent.strike,
        "bias": intent.bias,
        "reason": intent.reason,
    }


def _cache_snapshot(snapshot: SymbolMarketSnapshot) -> dict[str, Any]:
    contracts = [
        {
            "instrument_id": c.instrument_id,
            "type": c.option_type,
            "strike": c.strike,
            "expiration_date": c.expiration_date,
            "dte": c.dte,
        }
        for c in snapshot.contracts[:12]
    ]
    quotes = {
        iid: {
            "bid": q.bid,
            "ask": q.ask,
            "mid": q.mid,
            "spread_pct": q.spread_pct,
        }
        for iid, q in snapshot.quotes.items()
    }
    return {
        "symbol": snapshot.symbol,
        "spot": snapshot.spot.last_price,
        "contracts": contracts,
        "quotes": quotes,
    }


def _active_symbols(config: RobinhoodConfig) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for sym in config.options_watchlist:
        bias = config.bias_for(sym)
        if bias != "none":
            out.append((sym.upper(), bias))
    return out


async def _run_review(client: Any, intent: OrderIntent, config: RobinhoodConfig) -> dict[str, Any]:
    args = intent.to_mcp_args()
    try:
        if hasattr(client, "review_option_order"):
            result = await client.review_option_order(args)
        else:
            result = await client.adapter.call_tool("review_option_order", {**args, "dry_run": True})
        warnings = _extract_warnings(result)
        return {"ok": True, "warnings": warnings, "raw_type": type(result).__name__}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


async def run_options_tick(
    client: Any,
    config: RobinhoodConfig,
    *,
    agent_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One scan pass across biased watchlist symbols. Returns status summary."""
    started = time.time()
    funnel: Counter[str] = Counter()
    results: list[dict[str, Any]] = []
    tools = set(await client.list_tools())
    missing = sorted(OPTIONS_TOOLS - tools)
    if missing:
        payload = {
            "loop_enabled": config.options_loop_enabled,
            "available": False,
            "reason": "options_tools_missing",
            "missing_tools": missing,
            "tool_count": len(tools),
            "active_symbols": [],
            "results": [],
            "funnel": dict(funnel),
            "duration_s": round(time.time() - started, 2),
        }
        write_status(config.data_dir, payload)
        append_event(
            config.data_dir,
            {"type": "scan_blocked", "reason": "options_tools_missing", "missing": missing},
        )
        return payload

    active = _active_symbols(config)
    if not active:
        payload = {
            "loop_enabled": config.options_loop_enabled,
            "available": True,
            "reason": "no_active_bias",
            "hint": "Set RH_OPTIONS_BIAS=call|put or RH_OPTIONS_BIAS_SPY=call",
            "watchlist_size": len(config.options_watchlist),
            "active_symbols": [],
            "results": [],
            "funnel": {"no_active_bias": 1},
            "duration_s": round(time.time() - started, 2),
        }
        write_status(config.data_dir, payload)
        return payload

    positions = await fetch_option_positions(client)
    if positions is None:
        # Positions UNKNOWN (every fetch attempt failed) — trading blind
        # would disable the already-open / max-open guards, so skip the scan.
        funnel["positions_unavailable"] += 1
        payload = {
            "loop_enabled": config.options_loop_enabled,
            "available": True,
            "live_trading_enabled": config.live_trading_enabled,
            "reason": "positions_unavailable",
            "hint": "get_option_positions failed on every arg shape; "
                    "not trading blind",
            "active_symbols": [{"symbol": s, "bias": b} for s, b in active],
            "results": [],
            "funnel": dict(funnel),
            "duration_s": round(time.time() - started, 2),
        }
        write_status(config.data_dir, payload)
        append_event(
            config.data_dir,
            {"type": "scan_skipped", "reason": "positions_unavailable"},
        )
        return payload
    open_syms = open_underlyings(positions)
    open_count = count_open_positions(positions)
    placed_this_tick = 0
    positions_summary = {
        "open_count": open_count,
        "max_open_positions": config.options_max_open_positions,
        "underlyings": sorted(open_syms),
    }

    if open_count >= config.options_max_open_positions:
        funnel["max_open_positions"] += 1
        payload = {
            "loop_enabled": config.options_loop_enabled,
            "available": True,
            "live_trading_enabled": config.live_trading_enabled,
            "reason": "max_open_positions",
            "positions": positions_summary,
            "watchlist_size": len(config.options_watchlist),
            "active_symbols": [{"symbol": s, "bias": b} for s, b in active],
            "results": [],
            "funnel": dict(funnel),
            "duration_s": round(time.time() - started, 2),
        }
        write_status(config.data_dir, payload)
        append_event(
            config.data_dir,
            {"type": "scan_skipped", "reason": "max_open_positions", **positions_summary},
        )
        report = evaluate_readiness(
            config,
            status=agent_status,
            options_status=payload,
            min_paper_scans=config.options_min_paper_scans,
        )
        write_readiness_report(config.data_dir, report)
        payload["readiness"] = report.to_dict()
        write_status(config.data_dir, payload)
        return payload

    for symbol, bias in active:
        row: dict[str, Any] = {"symbol": symbol, "bias": bias}
        try:
            if symbol in open_syms:
                funnel["already_open"] += 1
                row["stage"] = "already_open"
                results.append(row)
                continue

            if symbol_in_cooldown(config.data_dir, symbol, config.options_symbol_cooldown_s):
                funnel["symbol_cooldown"] += 1
                row["stage"] = "symbol_cooldown"
                results.append(row)
                continue

            snapshot = await load_symbol_snapshot(client, symbol, config, bias)
            if not snapshot:
                funnel["no_market_data"] += 1
                row["stage"] = "no_market_data"
                results.append(row)
                continue

            save_chain_snapshot(config.data_dir, symbol, _cache_snapshot(snapshot))

            intent, stage = decide_order(snapshot, config, bias)
            funnel[stage] += 1
            row["stage"] = stage
            row["spot"] = snapshot.spot.last_price
            if not intent:
                results.append(row)
                continue

            row["intent"] = _intent_dict(intent)
            args = intent.to_mcp_args()

            if config.options_paper_review or not config.live_trading_enabled:
                review = await _run_review(client, intent, config)
                row["review"] = review
                if not review.get("ok"):
                    funnel["review_failed"] += 1
                    results.append(row)
                    continue

            if not config.live_trading_enabled:
                row["action"] = "paper_logged"
                record_symbol_action(
                    config.data_dir, symbol, "paper_intent", instrument_id=intent.instrument_id
                )
                append_event(
                    config.data_dir,
                    {"type": "paper_intent", "review": row.get("review"), **_intent_dict(intent)},
                )
                results.append(row)
                continue

            if (open_count + placed_this_tick
                    >= config.options_max_open_positions):
                # The pre-scan guard only checks the count at tick start;
                # without this, one tick could place an order per symbol
                # and blow through the cap.
                funnel["max_open_positions_intratick"] += 1
                row["action"] = "max_open_positions_reached"
                results.append(row)
                continue

            try:
                await client.call_tool("place_option_order", args)
                row["action"] = "placed"
                placed_this_tick += 1
                record_symbol_action(
                    config.data_dir, symbol, "placed", instrument_id=intent.instrument_id
                )
                append_event(
                    config.data_dir,
                    {"type": "order_placed", "review": row.get("review"), **_intent_dict(intent)},
                )
            except PermissionError as exc:
                row["action"] = "safety_blocked"
                row["error"] = str(exc)
                funnel["safety_blocked"] += 1
                append_event(
                    config.data_dir,
                    {
                        "type": "safety_blocked",
                        "error": str(exc),
                        **_intent_dict(intent),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                row["action"] = "place_failed"
                row["error"] = str(exc)
                funnel["place_failed"] += 1
                append_event(
                    config.data_dir,
                    {"type": "place_failed", "error": str(exc), **_intent_dict(intent)},
                )
        except Exception as exc:  # noqa: BLE001
            funnel["symbol_error"] += 1
            row["stage"] = "symbol_error"
            row["error"] = str(exc)
            logger.warning("options scan %s failed: %s", symbol, exc)
        results.append(row)

    payload = {
        "loop_enabled": config.options_loop_enabled,
        "available": True,
        "live_trading_enabled": config.live_trading_enabled,
        "watchlist_size": len(config.options_watchlist),
        "active_symbols": [{"symbol": s, "bias": b} for s, b in active],
        "positions": positions_summary,
        "results": results,
        "funnel": dict(funnel),
        "duration_s": round(time.time() - started, 2),
    }
    write_status(config.data_dir, payload)
    append_event(
        config.data_dir,
        {
            "type": "scan_complete",
            "active_count": len(active),
            "funnel": dict(funnel),
            "duration_s": payload["duration_s"],
            "open_positions": open_count,
        },
    )

    report = evaluate_readiness(
        config,
        status=agent_status,
        options_status=payload,
        min_paper_scans=config.options_min_paper_scans,
    )
    write_readiness_report(config.data_dir, report)
    payload["readiness"] = report.to_dict()
    write_status(config.data_dir, payload)
    return payload
