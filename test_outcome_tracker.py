"""Tests for outcome_tracker.py (Phase C: logging, settlement, kill-switch)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import outcome_tracker as ot


def _mk_verdict(ticker="AAPL", verdict="TRADE", side="long",
                received="2026-01-05T15:00:00+00:00", stop=0.05, tp=None,
                cost=0.0, p_win=0.55, expectancy=0.002):
    return {
        "timestamp_utc": received,
        "ticker": ticker,
        "verdict": verdict,
        "side": side,
        "horizon_days": 5,
        "signal_received_at_utc": received,
        "structure": {"stop_pct": stop, "tp_pct": tp},
        "cost_per_side": cost,
        "p_win": p_win,
        "expectancy_pct": expectancy,
    }


def _fetcher_from(dates, closes):
    def fetch(ticker, start_iso):
        return list(dates), np.asarray(closes, dtype=float)
    return fetch


BDAYS = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
         "2026-01-09", "2026-01-12", "2026-01-13", "2026-01-14"]


class TestLog:
    def test_log_verdict_appends_all_verdicts(self, tmp_path: Path):
        log = tmp_path / "trade_log.jsonl"
        ot.log_verdict(_mk_verdict(verdict="TRADE"), log)
        ot.log_verdict(_mk_verdict(verdict="NO_TRADE"), log)
        entries = ot.read_log(log)
        assert len(entries) == 2
        assert entries[0]["settled"] is False
        assert {e["verdict"] for e in entries} == {"TRADE", "NO_TRADE"}


class TestSettle:
    def test_settles_completed_window(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        ot.log_verdict(_mk_verdict(), log)
        closes = [100, 101, 102, 103, 104, 105, 106, 107]
        counts = ot.settle(log, price_fetcher=_fetcher_from(BDAYS, closes))
        assert counts["settled"] == 1
        entry = ot.read_log(log)[0]
        assert entry["settled"] is True
        st = entry["settlement"]
        assert st["entry_close"] == 100.0
        assert st["exit_close"] == 105.0  # 5 trading days later
        assert st["realized_ret"] == pytest.approx(0.05)
        assert st["realized_pnl_pct"] == pytest.approx(0.05)
        assert st["hit"] is True

    def test_stop_replay_on_realized_path(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        ot.log_verdict(_mk_verdict(stop=0.02), log)
        # Drops through the 2% stop on day 2, then recovers
        closes = [100, 99.5, 97.5, 101, 103, 104, 105, 106]
        ot.settle(log, price_fetcher=_fetcher_from(BDAYS, closes))
        st = ot.read_log(log)[0]["settlement"]
        assert st["exit_reason"] == "stop_loss"
        assert st["realized_pnl_pct"] == pytest.approx(-0.025)
        assert st["hit"] is False
        # buy-and-hold return recorded separately
        assert st["realized_ret"] == pytest.approx(0.04)

    def test_incomplete_window_stays_pending(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        ot.log_verdict(_mk_verdict(), log)
        counts = ot.settle(
            log,
            price_fetcher=_fetcher_from(BDAYS[:4], [100, 101, 102, 103]),
        )
        assert counts["settled"] == 0
        assert counts["pending"] == 1
        assert ot.read_log(log)[0]["settled"] is False

    def test_fetch_failure_keeps_pending(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        ot.log_verdict(_mk_verdict(), log)

        def boom(ticker, start):
            raise RuntimeError("network down")

        counts = ot.settle(log, price_fetcher=boom)
        assert counts["settled"] == 0
        assert counts["pending"] == 1


class TestReportAndKillSwitch:
    def _seed_settled(self, log: Path, pnls, verdict="TRADE"):
        closes_up = [100, 101, 102, 103, 104, 105, 106, 107]
        for pnl in pnls:
            v = _mk_verdict(verdict=verdict)
            v["settled"] = True
            v["settlement"] = {
                "realized_pnl_pct": float(pnl),
                "realized_pnl": float(pnl) * 100.0,
                "realized_ret": float(pnl),
                "hit": pnl > 0,
                "exit_reason": "max_holding",
            }
            ot.log_verdict(v, log)

    def test_report_stats(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        self._seed_settled(log, [0.01, -0.005, 0.02, -0.01])
        stats = ot.report_stats(log)
        assert stats["n_settled_trades"] == 4
        assert stats["hit_rate"] == pytest.approx(0.5)
        assert stats["mean_realized_pnl_pct"] == pytest.approx(0.00375)
        assert stats["calibration_ratio"] == pytest.approx(0.00375 / 0.002)

    def test_kill_switch_needs_full_window(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        self._seed_settled(log, [-0.01] * 19)
        tripped, reason = ot.check_kill_switch(log)
        assert tripped is False
        assert "inactive" in reason

    def test_kill_switch_trips_on_20_losers(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        self._seed_settled(log, [-0.01] * 20)
        tripped, reason = ot.check_kill_switch(log)
        assert tripped is True
        assert "KILL-SWITCH" in reason

    def test_kill_switch_clear_on_winners(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        self._seed_settled(log, [0.01] * 20)
        tripped, _ = ot.check_kill_switch(log)
        assert tripped is False

    def test_no_trade_verdicts_do_not_count(self, tmp_path: Path):
        log = tmp_path / "log.jsonl"
        self._seed_settled(log, [-0.01] * 20, verdict="NO_TRADE")
        tripped, _ = ot.check_kill_switch(log)
        assert tripped is False

    def test_cli_report_runs(self, tmp_path: Path, capsys):
        log = tmp_path / "log.jsonl"
        self._seed_settled(log, [0.01, -0.01])
        assert ot.main(["report", "--log", str(log)]) == 0
        out = capsys.readouterr().out
        assert "hit_rate" in out
