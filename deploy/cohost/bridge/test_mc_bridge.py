"""Offline tests for the Monte-Carlo-Sim → Robinhood paper bridge.

No network, no MCP: everything runs against a synthetic RobinhoodConfig and
verdict JSON files on disk.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from engine.robinhood.audit_log import AuditLog
from engine.robinhood.config import RobinhoodConfig
from engine.robinhood.mc_bridge import (
    LEDGER_FILENAME,
    STATE_FILENAME,
    BridgeState,
    map_verdict,
    process_once,
)
from engine.robinhood.safety_gates import RobinhoodSafetyGates


def make_config(tmp_path: Path, **overrides) -> RobinhoodConfig:
    from dataclasses import replace

    cfg = RobinhoodConfig.from_env()
    return replace(
        cfg,
        data_dir=str(tmp_path / "data"),
        live_trading_enabled=False,
        max_order_notional_usd=overrides.pop("max_order_notional_usd", 500.0),
        **overrides,
    )


def make_verdict(**overrides) -> dict:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    v = {
        "timestamp_utc": now.isoformat(),
        "ticker": "AAPL",
        "verdict": "TRADE",
        "side": "long",
        "horizon_days": 5,
        "s0": 100.0,
        "sizing": {"shares": 3},
    }
    v.update(overrides)
    return v


def write_verdict(dirpath: Path, name: str, verdict: dict) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    p.write_text(json.dumps(verdict), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# map_verdict
# ---------------------------------------------------------------------------


class TestMapVerdict:
    def test_happy_path_long_trade(self, tmp_path):
        cfg = make_config(tmp_path)
        plan, reason = map_verdict(make_verdict(), cfg)
        assert plan is not None
        assert reason == "mapped"
        args = plan.as_args()
        assert args["symbol"] == "AAPL"
        assert args["side"] == "buy"
        assert args["quantity"] == 3
        assert args["limit_price"] == pytest.approx(100.0)
        assert plan.notional == pytest.approx(300.0)
        assert plan.clamped_from is None

    def test_no_trade_is_skipped(self, tmp_path):
        cfg = make_config(tmp_path)
        plan, reason = map_verdict(make_verdict(verdict="NO_TRADE"), cfg)
        assert plan is None
        assert "not a TRADE" in reason

    def test_short_is_skipped_phase1(self, tmp_path):
        cfg = make_config(tmp_path)
        plan, reason = map_verdict(make_verdict(side="short"), cfg)
        assert plan is None
        assert "long-only" in reason

    def test_stale_verdict_is_skipped(self, tmp_path):
        cfg = make_config(tmp_path)
        old = (datetime.now(timezone.utc) - timedelta(hours=72)).isoformat()
        plan, reason = map_verdict(make_verdict(timestamp_utc=old), cfg,
                                   max_age_hours=48.0)
        assert plan is None
        assert "stale" in reason

    def test_undatable_verdict_is_skipped(self, tmp_path):
        cfg = make_config(tmp_path)
        plan, reason = map_verdict(make_verdict(timestamp_utc=None), cfg)
        assert plan is None
        assert "timestamp" in reason

    def test_zero_shares_skipped(self, tmp_path):
        cfg = make_config(tmp_path)
        plan, reason = map_verdict(
            make_verdict(sizing={"shares": 0}), cfg)
        assert plan is None
        assert "zero shares" in reason

    def test_quantity_clamped_by_notional_cap(self, tmp_path):
        cfg = make_config(tmp_path, max_order_notional_usd=250.0)
        plan, reason = map_verdict(make_verdict(), cfg)  # 3 sh × $100
        assert plan is not None
        assert plan.quantity == 2                        # floor(250/100)
        assert plan.clamped_from == 3
        assert "clamped" in reason

    def test_cap_below_one_share_skipped_with_guidance(self, tmp_path):
        cfg = make_config(tmp_path, max_order_notional_usd=100.0)
        plan, reason = map_verdict(make_verdict(s0=743.29), cfg)
        assert plan is None
        assert "RH_MAX_ORDER_NOTIONAL_USD" in reason  # tells the owner the fix


# ---------------------------------------------------------------------------
# process_once: gates + ledger + idempotency
# ---------------------------------------------------------------------------


class TestProcessOnce:
    def _run(self, tmp_path, cfg, dirs):
        audit = AuditLog(cfg.data_dir)
        gates = RobinhoodSafetyGates(cfg, audit)
        return process_once(dirs, cfg, gates=gates, audit=audit)

    def test_pass_plans_and_ledgers(self, tmp_path):
        cfg = make_config(tmp_path)
        vdir = tmp_path / "verdicts"
        write_verdict(vdir, "a_AAPL.json", make_verdict())
        write_verdict(vdir, "b_SPY.json", make_verdict(verdict="NO_TRADE",
                                                       ticker="SPY"))
        summary = self._run(tmp_path, cfg, [vdir])
        assert summary["new"] == 2
        assert summary["planned"] == 1
        assert summary["skipped"] == 1
        rows = [json.loads(l) for l in
                (Path(cfg.data_dir) / LEDGER_FILENAME).read_text().splitlines()]
        assert len(rows) == 2
        planned = next(r for r in rows if r["verdict_id"] == "a_AAPL.json")
        assert planned["gate_allowed"] is True
        assert "no order placed" in planned["outcome"]
        assert planned["order_plan"]["quantity"] == 3

    def test_idempotent_across_passes_and_restarts(self, tmp_path):
        cfg = make_config(tmp_path)
        vdir = tmp_path / "verdicts"
        write_verdict(vdir, "a_AAPL.json", make_verdict())
        s1 = self._run(tmp_path, cfg, [vdir])
        s2 = self._run(tmp_path, cfg, [vdir])          # same files again
        assert s1["new"] == 1 and s2["new"] == 0
        # simulate restart: state reloaded from disk
        state = BridgeState.load(cfg.data_dir)
        assert state.is_processed("a_AAPL.json")
        # ledger has exactly one row
        rows = (Path(cfg.data_dir) / LEDGER_FILENAME).read_text().splitlines()
        assert len(rows) == 1

    def test_gate_blocks_oversized_notional(self, tmp_path):
        # cap far below the order → local gate must block and record why
        cfg = make_config(tmp_path, max_order_notional_usd=250.0)
        vdir = tmp_path / "verdicts"
        # 3 shares × $100 clamps to 2 ($200) — fine. Force a gate block via
        # daily loss halt instead: record a big realized loss first.
        audit = AuditLog(cfg.data_dir)
        gates = RobinhoodSafetyGates(cfg, audit)
        gates.record_realized_pnl(-10_000.0)
        write_verdict(vdir, "a_AAPL.json", make_verdict())
        summary = process_once([vdir], cfg, gates=gates, audit=audit)
        assert summary["gate_blocked"] == 1
        row = json.loads(
            (Path(cfg.data_dir) / LEDGER_FILENAME).read_text().splitlines()[0])
        assert row["gate_allowed"] is False
        assert "daily_loss" in row["gate_reason"]

    def test_unreadable_file_is_recorded_not_fatal(self, tmp_path):
        cfg = make_config(tmp_path)
        vdir = tmp_path / "verdicts"
        vdir.mkdir(parents=True)
        (vdir / "broken.json").write_text("{not json", encoding="utf-8")
        summary = self._run(tmp_path, cfg, [vdir])
        assert summary["skipped"] == 1
        state = BridgeState.load(cfg.data_dir)
        assert state.is_processed("broken.json")

    def test_missing_dir_is_tolerated(self, tmp_path):
        cfg = make_config(tmp_path)
        summary = self._run(tmp_path, cfg, [tmp_path / "nope"])
        assert summary == {"seen": 0, "new": 0, "planned": 0,
                           "gate_blocked": 0, "skipped": 0}

    def test_no_place_tool_is_ever_evaluated(self, tmp_path, monkeypatch):
        """Phase-1 invariant: the bridge must never even *evaluate* a
        place_* tool, let alone call one."""
        cfg = make_config(tmp_path)
        audit = AuditLog(cfg.data_dir)
        gates = RobinhoodSafetyGates(cfg, audit)
        seen_tools = []
        orig = gates.evaluate

        def spy(tool, arguments=None, **kw):
            seen_tools.append(tool)
            return orig(tool, arguments, **kw)

        monkeypatch.setattr(gates, "evaluate", spy)
        vdir = tmp_path / "verdicts"
        write_verdict(vdir, "a_AAPL.json", make_verdict())
        process_once([vdir], cfg, gates=gates, audit=audit)
        assert seen_tools == ["review_equity_order"]
