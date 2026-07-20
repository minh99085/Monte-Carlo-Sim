"""Tests for the safety/robustness hardening fixes.

Covers: persistent PDT + daily-loss state, the option contract multiplier in
the notional gate, fail-safe position fetching, the intra-tick max-open cap,
MCP text-block JSON unwrapping, DTE filtering of unparseable expirations,
and OAuth token file permissions.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from engine.robinhood.audit_log import AuditLog
from engine.robinhood.config import RobinhoodConfig
from engine.robinhood.options_market import parse_option_instruments
from engine.robinhood.options_positions import fetch_option_positions
from engine.robinhood.robinhood_mcp_adapter import _unwrap_block
from engine.robinhood.safety_gates import (
    OPTION_CONTRACT_MULTIPLIER,
    SAFETY_STATE_FILENAME,
    RobinhoodSafetyGates,
)


def make_config(tmp_path: Path, **overrides) -> RobinhoodConfig:
    cfg = RobinhoodConfig.from_env()
    return replace(
        cfg,
        data_dir=str(tmp_path / "data"),
        live_trading_enabled=overrides.pop("live_trading_enabled", True),
        **overrides,
    )


# ---------------------------------------------------------------------------
# 1. Persistent safety state (PDT counter + daily loss survive restarts)
# ---------------------------------------------------------------------------


class TestPersistentSafetyState:
    def test_day_trades_survive_restart(self, tmp_path):
        cfg = make_config(tmp_path)
        audit = AuditLog(cfg.data_dir)
        g1 = RobinhoodSafetyGates(cfg, audit)
        g1.day_trades.record()
        g1.day_trades.record()
        # "restart": brand-new gates instance over the same data dir
        g2 = RobinhoodSafetyGates(cfg, audit)
        assert g2.day_trades.count_last_5_days() == 2
        state = json.loads(
            (Path(cfg.data_dir) / SAFETY_STATE_FILENAME).read_text())
        assert len(state["day_trades"]) == 2

    def test_daily_loss_survives_restart(self, tmp_path):
        cfg = make_config(tmp_path, daily_loss_limit_usd=200.0)
        audit = AuditLog(cfg.data_dir)
        g1 = RobinhoodSafetyGates(cfg, audit)
        g1.record_realized_pnl(-500.0)
        g2 = RobinhoodSafetyGates(cfg, audit)  # restart
        v = g2.evaluate("place_equity_order",
                        {"symbol": "AAPL", "notional": 10})
        assert v.allowed is False
        assert "daily_loss" in v.reason

    def test_pdt_limit_enforced_after_restart(self, tmp_path):
        cfg = make_config(tmp_path, max_day_trades_5d=2)
        audit = AuditLog(cfg.data_dir)
        g1 = RobinhoodSafetyGates(cfg, audit)
        g1.day_trades.record()
        g1.day_trades.record()
        g2 = RobinhoodSafetyGates(cfg, audit)  # restart must not reset PDT
        v = g2.evaluate("place_equity_order",
                        {"symbol": "AAPL", "notional": 10})
        assert v.allowed is False
        assert "pdt" in v.reason.lower()


# ---------------------------------------------------------------------------
# 2. Option notional uses the 100x contract multiplier
# ---------------------------------------------------------------------------


class TestOptionNotionalMultiplier:
    def test_option_order_true_exposure_blocks(self, tmp_path):
        # 1 contract @ $2.50 = $250 true exposure; cap $100 must block it.
        cfg = make_config(tmp_path, max_order_notional_usd=100.0)
        gates = RobinhoodSafetyGates(cfg, AuditLog(cfg.data_dir))
        v = gates.evaluate("place_option_order",
                           {"symbol": "SPY", "quantity": 1,
                            "limit_price": 2.50})
        assert v.allowed is False
        assert "exceeds max" in v.reason

    def test_equity_order_unchanged(self, tmp_path):
        cfg = make_config(tmp_path, max_order_notional_usd=100.0)
        gates = RobinhoodSafetyGates(cfg, AuditLog(cfg.data_dir))
        v = gates.evaluate("review_equity_order",
                           {"symbol": "F", "quantity": 5, "limit_price": 10.0})
        assert v.allowed is True  # $50 < $100, no multiplier for shares

    def test_multiplier_constant(self):
        assert OPTION_CONTRACT_MULTIPLIER == 100.0


# ---------------------------------------------------------------------------
# 3. Fail-safe position fetch: None (unknown) vs [] (confirmed flat)
# ---------------------------------------------------------------------------


class _FailingClient:
    async def call_tool(self, name, arguments=None):
        raise RuntimeError("boom")


class _EmptyClient:
    async def call_tool(self, name, arguments=None):
        return {"results": []}


@pytest.mark.asyncio
async def test_positions_none_when_all_attempts_fail():
    assert await fetch_option_positions(_FailingClient()) is None


@pytest.mark.asyncio
async def test_positions_empty_list_is_authoritative():
    assert await fetch_option_positions(_EmptyClient()) == []


# ---------------------------------------------------------------------------
# 4. Loop: positions unknown -> scan skipped (never trade blind)
# ---------------------------------------------------------------------------


class _LoopClient:
    """Tools exist; positions always fail; nothing else is reachable."""

    def __init__(self):
        from engine.robinhood.constants import OPTIONS_TOOLS

        self._tools = set(OPTIONS_TOOLS) | {"get_equity_quotes", "get_portfolio"}

    async def list_tools(self):
        return sorted(self._tools)

    async def call_tool(self, name, arguments=None):
        raise RuntimeError("api down")


@pytest.mark.asyncio
async def test_tick_skips_when_positions_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("RH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RH_OPTIONS_BIAS", "call")
    from engine.robinhood.options_loop import run_options_tick

    cfg = RobinhoodConfig.from_env()
    out = await run_options_tick(_LoopClient(), cfg)
    assert out["reason"] == "positions_unavailable"
    assert out["results"] == []


# ---------------------------------------------------------------------------
# 5. MCP text-block unwrapping
# ---------------------------------------------------------------------------


class TestUnwrapBlock:
    def test_json_text_block_is_decoded(self):
        block = {"type": "text", "text": '{"results": [{"symbol": "SPY"}]}'}
        assert _unwrap_block(block) == {"results": [{"symbol": "SPY"}]}

    def test_plain_text_block_returns_text(self):
        assert _unwrap_block({"type": "text", "text": "hello"}) == "hello"

    def test_malformed_json_returns_text(self):
        assert _unwrap_block({"type": "text", "text": "{oops"}) == "{oops"

    def test_non_text_block_passthrough(self):
        block = {"type": "image", "data": "xyz"}
        assert _unwrap_block(block) is block


# ---------------------------------------------------------------------------
# 6. Unparseable expirations cannot bypass the DTE window
# ---------------------------------------------------------------------------


def test_unparseable_expiration_is_filtered():
    payload = [{
        "id": "opt-x",
        "type": "call",
        "strike_price": 500.0,
        "expiration_date": "someday",
    }]
    out = parse_option_instruments(
        payload, underlying="SPY", min_dte=0, max_dte=60,
        strike_lo=450.0, strike_hi=550.0, option_type="call")
    assert out == []


# ---------------------------------------------------------------------------
# 7. OAuth token file is owner-only
# ---------------------------------------------------------------------------


def test_token_file_permissions(tmp_path):
    from engine.robinhood.oauth_storage import FileTokenStorage

    storage = FileTokenStorage(tmp_path)
    storage._save({"tokens": {"access_token": "secret"}})
    mode = storage.path.stat().st_mode & 0o777
    assert mode == 0o600
