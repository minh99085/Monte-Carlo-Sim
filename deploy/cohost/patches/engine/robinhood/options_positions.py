"""Parse Robinhood MCP option position snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OptionPosition:
    symbol: str
    option_type: str
    quantity: float
    instrument_id: str | None = None
    strike: float | None = None
    expiration_date: str | None = None
    market_value: float | None = None


def _as_list(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "data", "items", "positions", "option_positions"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
        return [payload]
    return []


def _first_str(d: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        val = d.get(key)
        if val is not None and str(val).strip():
            return str(val)
    return None


def _first_float(d: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in d and d[key] is not None:
            try:
                return float(d[key])
            except (TypeError, ValueError):
                continue
    return None


def parse_option_positions(payload: Any) -> list[OptionPosition]:
    out: list[OptionPosition] = []
    for row in _as_list(payload):
        if not isinstance(row, dict):
            continue
        sym = (_first_str(row, "chain_symbol", "underlying_symbol", "symbol", "ticker") or "").upper()
        if not sym:
            continue
        qty = _first_float(row, "quantity", "qty", "shares", "contracts")
        if qty is None or abs(qty) < 1e-9:
            continue
        typ = (_first_str(row, "type", "option_type", "put_call", "side") or "").lower()
        if typ in ("c", "call"):
            typ = "call"
        elif typ in ("p", "put"):
            typ = "put"
        else:
            typ = typ or "unknown"
        out.append(
            OptionPosition(
                symbol=sym,
                option_type=typ,
                quantity=qty,
                instrument_id=_first_str(row, "instrument_id", "option_instrument_id", "id"),
                strike=_first_float(row, "strike_price", "strike"),
                expiration_date=_first_str(row, "expiration_date", "expiration"),
                market_value=_first_float(row, "market_value", "value"),
            )
        )
    return out


def open_underlyings(positions: list[OptionPosition]) -> set[str]:
    return {p.symbol.upper() for p in positions if abs(p.quantity) > 0}


def count_open_positions(positions: list[OptionPosition]) -> int:
    return sum(1 for p in positions if abs(p.quantity) > 0)


async def fetch_option_positions(client: Any) -> list[OptionPosition] | None:
    """Fetch open option positions.

    Returns a list on a successful call (possibly empty — a confirmed
    "no open positions"), or **None when every attempt failed**. Callers
    must treat None as "positions unknown" and skip trading rather than
    assume a flat book — an API hiccup must never disable the
    already-open / max-open-positions guards.
    """
    for args in ({}, {"state": "open"}, {"status": "open"}):
        try:
            raw = await client.call_tool("get_option_positions", args)
        except Exception:  # noqa: BLE001
            continue
        # A successful call is authoritative, even when it parses to [].
        return parse_option_positions(raw)
    return None
