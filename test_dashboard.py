"""Tests for the dashboard aggregator and the bridge's HTTP routes."""

from __future__ import annotations

import json
import threading
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

import dashboard as dash
import outcome_tracker
import tv_webhook_bridge as bridge


# ---------------------------------------------------------------------------
# Fake bot API (localhost mock the aggregator can fetch)
# ---------------------------------------------------------------------------


def _start_fake_bot(routes: dict[str, dict[str, Any]]) -> tuple[str, ThreadingHTTPServer]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # noqa: N802
            payload = routes.get(self.path)
            if payload is None:
                self.send_response(404)
                self.end_headers()
                return
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address
    return f"http://{host}:{port}", srv


@pytest.fixture
def fake_bot():
    servers: list[ThreadingHTTPServer] = []

    def _make(routes: dict[str, dict[str, Any]]) -> str:
        url, srv = _start_fake_bot(routes)
        servers.append(srv)
        return url

    yield _make
    for s in servers:
        s.shutdown()
        s.server_close()


# ---------------------------------------------------------------------------
# Fixture builder for a plausible on-disk MC state
# ---------------------------------------------------------------------------


def _make_mc_layout(tmp_path: Path) -> Path:
    (tmp_path / "tv_data").mkdir()
    (tmp_path / "outputs" / "verdicts").mkdir(parents=True)
    (tmp_path / "outputs" / "paper_verdicts").mkdir()
    (tmp_path / "calibration").mkdir()
    return tmp_path


def _write_signal(tmp_path: Path, *, age_hours: float, ticker: str = "SPY"):
    ts = (datetime.now(timezone.utc) - timedelta(hours=age_hours)) \
        .replace(microsecond=0).isoformat()
    (tmp_path / "tv_data" / "latest_signal.json").write_text(json.dumps({
        "received_at_utc": ts,
        "ticker": ticker,
        "trend": "bullish",
        "momentum": 55.0,
        "price": 743.29,
    }))


def _write_verdict(dirpath: Path, ticker: str, verdict: str, *,
                   shares: int = 33, ts: str | None = None):
    ts = ts or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    stem = ts.replace(":", "").replace("-", "") + "_" + ticker
    (dirpath / f"{stem}.json").write_text(json.dumps({
        "timestamp_utc": ts,
        "ticker": ticker,
        "verdict": verdict,
        "side": "long",
        "reason": "edge 0.42% weekly > breakeven 0.15%",
        "s0": 100.0,
        "sizing": {"shares": shares},
        "edge_weekly": 0.0042,
    }))


# ---------------------------------------------------------------------------
# 1. Section builders return well-formed dicts even with no data
# ---------------------------------------------------------------------------


class TestEmptyDeployment:
    def test_signal_waiting_when_missing(self, tmp_path):
        app = _make_mc_layout(tmp_path)
        s = dash._signal_section(app)
        assert s["state"] == "waiting"
        assert s["ticker"] is None

    def test_brain_warn_when_no_paper_log(self, tmp_path):
        app = _make_mc_layout(tmp_path)
        b = dash._brain_section(app)
        assert b["state"] == "warn"
        assert "No paper training" in b["detail"]

    def test_bot_warn_when_api_unreachable(self):
        bot = dash._bot_section("http://127.0.0.1:1")  # nothing listens
        assert bot["state"] == "warn"
        assert bot["reachable"] is False

    def test_snapshot_never_raises(self, tmp_path):
        snap = dash.build_snapshot(app_dir=tmp_path,
                                   bot_api="http://127.0.0.1:1")
        for key in ("generated_at_utc", "overall_state",
                    "signal", "brain", "bridge", "bot", "verdicts"):
            assert key in snap


# ---------------------------------------------------------------------------
# 2. Signal freshness gate
# ---------------------------------------------------------------------------


class TestSignalFreshness:
    def test_fresh_signal_good(self, tmp_path):
        app = _make_mc_layout(tmp_path)
        _write_signal(app, age_hours=1.5)
        s = dash._signal_section(app)
        assert s["state"] == "good"
        assert s["ticker"] == "SPY"

    def test_stale_signal_warn(self, tmp_path):
        app = _make_mc_layout(tmp_path)
        _write_signal(app, age_hours=48.0)
        s = dash._signal_section(app)
        assert s["state"] == "warn"
        assert "stale" in s["detail"]


# ---------------------------------------------------------------------------
# 3. Brain: kill switch surfaces as critical
# ---------------------------------------------------------------------------


class TestBrain:
    def test_kill_switch_makes_brain_critical(self, tmp_path):
        app = _make_mc_layout(tmp_path)
        log = app / "outputs" / "paper_log.jsonl"
        for i in range(20):
            entry = {
                "ticker": "PWR", "verdict": "TRADE", "side": "long",
                "horizon_days": 5, "settled": True,
                "settlement": {"realized_pnl_pct": -0.02, "hit": False},
            }
            outcome_tracker.log_verdict(entry, log)
        b = dash._brain_section(app)
        assert b["state"] == "critical"
        assert "KILL-SWITCH" in b["kill_switch"]["reason"]


# ---------------------------------------------------------------------------
# 4. Bot + bridge sections consume fake bot correctly
# ---------------------------------------------------------------------------


class TestBotAndBridge:
    def test_paper_bot_reports_good(self, fake_bot):
        url = fake_bot({
            "/api/health": {
                "status": "ok", "live_trading_enabled": False,
                "mcp_connected": True, "tool_count": 12,
            },
            "/api/robinhood/mc-bridge": {
                "counts": {"total": 5, "planned": 2, "skipped": 3, "gate_blocked": 0},
                "recent": [{"verdict_id": "x", "ticker": "SPY",
                            "mc_verdict": "TRADE",
                            "outcome": "paper_planned"}],
            },
        })
        bot = dash._bot_section(url)
        assert bot["state"] == "good"
        assert bot["reachable"] is True
        assert bot["live_trading_enabled"] is False
        bridge_ = dash._bridge_section(bot)
        assert bridge_["state"] == "good"
        assert bridge_["counts"]["planned"] == 2

    def test_live_trading_flags_critical(self, fake_bot):
        url = fake_bot({
            "/api/health": {"status": "ok", "live_trading_enabled": True,
                            "mcp_connected": True, "tool_count": 1},
            "/api/robinhood/mc-bridge": {"counts": {}, "recent": []},
        })
        bot = dash._bot_section(url)
        assert bot["state"] == "critical"


# ---------------------------------------------------------------------------
# 5. Verdicts section: newest first, capped, both dirs
# ---------------------------------------------------------------------------


class TestVerdicts:
    def test_verdicts_newest_first_across_dirs(self, tmp_path):
        app = _make_mc_layout(tmp_path)
        _write_verdict(app / "outputs" / "paper_verdicts", "AAPL", "TRADE",
                       ts="2026-07-20T10:00:00+00:00")
        _write_verdict(app / "outputs" / "verdicts", "SPY", "NO_TRADE",
                       ts="2026-07-20T11:00:00+00:00")
        rows = dash._verdicts_section(app)
        assert rows[0]["ticker"] == "SPY"       # newer
        assert rows[0]["source"] == "live"
        assert rows[1]["ticker"] == "AAPL"
        assert rows[1]["source"] == "paper"


# ---------------------------------------------------------------------------
# 6. Overall state = worst of all sections
# ---------------------------------------------------------------------------


def test_overall_state_takes_worst(tmp_path, fake_bot):
    app = _make_mc_layout(tmp_path)
    _write_signal(app, age_hours=1.0)          # good
    url = fake_bot({
        "/api/health": {"status": "ok", "live_trading_enabled": True,
                        "mcp_connected": True, "tool_count": 1},
        "/api/robinhood/mc-bridge": {"counts": {"total": 1, "planned": 1},
                                     "recent": []},
    })
    snap = dash.build_snapshot(app_dir=app, bot_api=url)
    assert snap["overall_state"] == "critical"


# ---------------------------------------------------------------------------
# 7. Bridge HTTP routes: /dashboard and /dashboard/api
# ---------------------------------------------------------------------------


@pytest.fixture
def running_bridge(tmp_path):
    (tmp_path / "outputs" / "paper_verdicts").mkdir(parents=True)
    html = tmp_path / "dashboard.html"
    html.write_text("<html>DASH</html>")
    state = bridge.BridgeState(
        secret="s", data_dir=tmp_path / "tv_data",
        dashboard_html_path=html,
        dashboard_app_dir=tmp_path,
        dashboard_bot_api="http://127.0.0.1:1",
    )
    bridge.ensure_data_dir(state.data_dir)
    handler = bridge.make_handler(state)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address
    yield f"http://{host}:{port}"
    srv.shutdown()
    srv.server_close()


class TestBridgeRoutes:
    def test_dashboard_html_served(self, running_bridge):
        with urllib.request.urlopen(f"{running_bridge}/dashboard") as r:
            assert r.status == 200
            body = r.read().decode()
        assert "DASH" in body

    def test_dashboard_api_returns_snapshot(self, running_bridge):
        with urllib.request.urlopen(f"{running_bridge}/dashboard/api") as r:
            assert r.status == 200
            data = json.loads(r.read())
        assert data["overall_state"] in ("good", "info", "warn", "critical")
        assert "signal" in data and "brain" in data and "bot" in data

    def test_dashboard_api_reports_no_secrets(self, running_bridge):
        with urllib.request.urlopen(f"{running_bridge}/dashboard/api") as r:
            data = json.loads(r.read())
        # Whole payload serialized: no secret/hmac keys of any kind leaked.
        blob = json.dumps(data).lower()
        for key in ("secret", "hmac", "token", "password", "api_key"):
            assert key not in blob
