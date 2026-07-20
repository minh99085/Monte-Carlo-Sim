"""Robinhood MCP market-data helpers for options scanning."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from engine.robinhood.config import RobinhoodConfig

logger = logging.getLogger("hermes.robinhood.options_market")


@dataclass(frozen=True)
class EquityQuote:
    symbol: str
    last_price: float
    prior_close: float | None = None


@dataclass(frozen=True)
class OptionContract:
    symbol: str
    option_type: str  # call | put
    strike: float
    expiration_date: str
    instrument_id: str
    chain_id: str | None = None
    dte: int | None = None


@dataclass(frozen=True)
class OptionQuote:
    instrument_id: str
    bid: float | None
    ask: float | None
    mid: float | None
    spread_pct: float | None


@dataclass(frozen=True)
class SymbolMarketSnapshot:
    symbol: str
    spot: EquityQuote
    contracts: list[OptionContract]
    quotes: dict[str, OptionQuote]


def _as_list(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("results", "data", "items", "instruments", "contracts", "quotes"):
            val = payload.get(key)
            if isinstance(val, list):
                return val
        return [payload]
    return []


def _first_float(d: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in d and d[key] is not None:
            try:
                return float(d[key])
            except (TypeError, ValueError):
                continue
    return None


def _first_str(d: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        val = d.get(key)
        if val is not None and str(val).strip():
            return str(val)
    return None


def parse_equity_quotes(payload: Any, symbols: list[str]) -> dict[str, EquityQuote]:
    out: dict[str, EquityQuote] = {}
    for row in _as_list(payload):
        if not isinstance(row, dict):
            continue
        sym = (_first_str(row, "symbol", "ticker") or "").upper()
        if not sym:
            continue
        last = _first_float(row, "last_trade_price", "last_price", "price", "mark_price")
        if last is None:
            last = _first_float(row, "adjusted_mark_price", "quote", "last")
        prior = _first_float(row, "previous_close", "prior_close", "prev_close")
        if last is not None:
            out[sym] = EquityQuote(symbol=sym, last_price=last, prior_close=prior)
    # Single-symbol wrapper shapes
    if not out and isinstance(payload, dict):
        sym = (_first_str(payload, "symbol", "ticker") or (symbols[0] if symbols else "")).upper()
        last = _first_float(payload, "last_trade_price", "last_price", "price")
        if sym and last is not None:
            out[sym] = EquityQuote(
                symbol=sym,
                last_price=last,
                prior_close=_first_float(payload, "previous_close", "prior_close"),
            )
    return out


def _parse_date(val: str | None) -> date | None:
    if not val:
        return None
    text = val.strip()[:10]
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _dte(expiration: str | None) -> int | None:
    exp = _parse_date(expiration)
    if not exp:
        return None
    return (exp - date.today()).days


def parse_option_instruments(
    payload: Any,
    *,
    underlying: str,
    min_dte: int,
    max_dte: int,
    strike_lo: float,
    strike_hi: float,
    option_type: str,
) -> list[OptionContract]:
    want = option_type.lower()
    out: list[OptionContract] = []
    for row in _as_list(payload):
        if not isinstance(row, dict):
            continue
        typ = (_first_str(row, "type", "option_type", "put_call", "side") or "").lower()
        if typ in ("c", "call"):
            typ = "call"
        elif typ in ("p", "put"):
            typ = "put"
        if typ != want:
            continue
        strike = _first_float(row, "strike_price", "strike")
        if strike is None or strike < strike_lo or strike > strike_hi:
            continue
        exp = _first_str(row, "expiration_date", "expiration", "expiry", "expires_at")
        dte = _dte(exp)
        # An unparseable expiration means we cannot verify the contract is
        # inside the DTE window — treat it as untradable rather than letting
        # it bypass the filter.
        if dte is None or dte < min_dte or dte > max_dte:
            continue
        iid = _first_str(
            row,
            "id",
            "instrument_id",
            "option_instrument_id",
            "option_id",
            "url",
        )
        if not iid or strike is None or not exp:
            continue
        out.append(
            OptionContract(
                symbol=underlying.upper(),
                option_type=typ,
                strike=strike,
                expiration_date=exp[:10] if len(exp) >= 10 else exp,
                instrument_id=iid,
                chain_id=_first_str(row, "chain_id", "option_chain_id"),
                dte=dte,
            )
        )
    out.sort(key=lambda c: (c.dte or 9999, abs(c.strike)))
    return out


def parse_option_quotes(payload: Any) -> dict[str, OptionQuote]:
    out: dict[str, OptionQuote] = {}
    for row in _as_list(payload):
        if not isinstance(row, dict):
            continue
        iid = _first_str(row, "instrument_id", "id", "option_instrument_id", "option_id")
        if not iid:
            continue
        bid = _first_float(row, "bid_price", "bid")
        ask = _first_float(row, "ask_price", "ask")
        mark = _first_float(row, "mark_price", "mid_price", "adjusted_mark_price", "mark")
        mid = mark
        if mid is None and bid is not None and ask is not None:
            mid = (bid + ask) / 2.0
        spread_pct = None
        if mid and mid > 0 and bid is not None and ask is not None:
            spread_pct = ((ask - bid) / mid) * 100.0
        out[iid] = OptionQuote(
            instrument_id=iid,
            bid=bid,
            ask=ask,
            mid=mid,
            spread_pct=spread_pct,
        )
    return out


def strike_band(spot: float, band_pct: float) -> tuple[float, float]:
    margin = spot * (band_pct / 100.0)
    return spot - margin, spot + margin


async def fetch_equity_quotes(client: Any, symbols: list[str]) -> dict[str, EquityQuote]:
    if not symbols:
        return {}
    # Robinhood MCP accepts symbol lists; try common argument shapes.
    for args in (
        {"symbols": symbols},
        {"symbol": symbols},
        {"tickers": symbols},
    ):
        try:
            raw = await client.call_tool("get_equity_quotes", args)
            parsed = parse_equity_quotes(raw, symbols)
            if parsed:
                return parsed
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_equity_quotes %s failed: %s", args, exc)
    return {}


async def fetch_option_chain_id(client: Any, symbol: str) -> str | None:
    for args in (
        {"symbol": symbol},
        {"chain_symbol": symbol},
        {"underlying_symbol": symbol},
    ):
        try:
            raw = await client.call_tool("get_option_chains", args)
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_option_chains %s failed: %s", args, exc)
            continue
        for row in _as_list(raw):
            if not isinstance(row, dict):
                continue
            cid = _first_str(row, "id", "chain_id", "option_chain_id", "url")
            if cid:
                return cid
        if isinstance(raw, dict):
            cid = _first_str(raw, "id", "chain_id", "option_chain_id")
            if cid:
                return cid
    return None


async def fetch_option_instruments(
    client: Any,
    *,
    symbol: str,
    chain_id: str | None,
    config: RobinhoodConfig,
    option_type: str,
    spot: float,
) -> list[OptionContract]:
    lo, hi = strike_band(spot, config.options_strike_band_pct)
    base_args: dict[str, Any] = {
        "symbol": symbol,
        "chain_symbol": symbol,
        "option_type": option_type,
        "type": option_type,
    }
    if chain_id:
        base_args["chain_id"] = chain_id
        base_args["option_chain_id"] = chain_id
    arg_variants = [
        base_args,
        {**base_args, "expiration_date_gte": config.options_min_dte},
        {"chain_id": chain_id, "symbol": symbol} if chain_id else base_args,
    ]
    for args in arg_variants:
        try:
            raw = await client.call_tool("get_option_instruments", args)
            contracts = parse_option_instruments(
                raw,
                underlying=symbol,
                min_dte=config.options_min_dte,
                max_dte=config.options_max_dte,
                strike_lo=lo,
                strike_hi=hi,
                option_type=option_type,
            )
            if contracts:
                return contracts
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_option_instruments %s failed: %s", args, exc)
    return []


async def fetch_option_quotes(client: Any, instrument_ids: list[str]) -> dict[str, OptionQuote]:
    if not instrument_ids:
        return {}
    for args in (
        {"instrument_ids": instrument_ids},
        {"ids": instrument_ids},
        {"option_instrument_ids": instrument_ids},
        {"instruments": instrument_ids},
    ):
        try:
            raw = await client.call_tool("get_option_quotes", args)
            parsed = parse_option_quotes(raw)
            if parsed:
                return parsed
        except Exception as exc:  # noqa: BLE001
            logger.debug("get_option_quotes %s failed: %s", args, exc)
    return {}


async def load_symbol_snapshot(
    client: Any,
    symbol: str,
    config: RobinhoodConfig,
    option_type: str,
) -> SymbolMarketSnapshot | None:
    quotes = await fetch_equity_quotes(client, [symbol])
    spot = quotes.get(symbol.upper())
    if not spot:
        return None
    chain_id = await fetch_option_chain_id(client, symbol)
    contracts = await fetch_option_instruments(
        client,
        symbol=symbol,
        chain_id=chain_id,
        config=config,
        option_type=option_type,
        spot=spot.last_price,
    )
    if not contracts:
        return None
    # Quote the nearest few candidates (by DTE/strike sort).
    sample_ids = [c.instrument_id for c in contracts[:8]]
    oquotes = await fetch_option_quotes(client, sample_ids)
    return SymbolMarketSnapshot(
        symbol=symbol.upper(),
        spot=spot,
        contracts=contracts,
        quotes=oquotes,
    )
