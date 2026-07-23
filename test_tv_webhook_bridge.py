"""Tests for the Phase 3 TradingView webhook bridge (stdlib server)."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import tv_webhook_bridge as bridge


def test_parse_payload_json_and_embedded():
    raw = b'{"ticker":"AAPL","price":10,"trend":"bullish","momentum":55}'
    obj = bridge.parse_payload(raw)
    assert obj["ticker"] == "AAPL"
    assert obj["parse_status"] == "json"

    raw2 = b'Alert fired: {"ticker":"MSFT","trend":"bearish"} end'
    obj2 = bridge.parse_payload(raw2)
    assert obj2["ticker"] == "MSFT"
    assert obj2["parse_status"] == "embedded_json"

    raw3 = b"not json at all"
    obj3 = bridge.parse_payload(raw3)
    assert obj3["parse_status"] == "text"
    assert "raw_message" in obj3


def test_normalize_signal_aliases():
    payload = {
        "symbol": "aapl",
        "close": "100.5",
        "direction": "Bullish",
        "rsi": "62.1",
        "tf": "5",
        "parse_status": "json",
    }
    rec = bridge.normalize_signal(payload, source_ip="127.0.0.1")
    assert rec["ticker"] == "AAPL"
    assert rec["price"] == 100.5
    assert rec["trend"] == "bullish"
    assert rec["momentum"] == 62.1
    assert rec["timeframe"] == "5"
    assert rec["source"] == "tradingview"


def test_save_and_load_latest(tmp_path: Path):
    rec = bridge.normalize_signal(
        {"ticker": "TSLA", "price": 200, "trend": "bearish", "momentum": 40},
        source_ip="1.2.3.4",
    )
    latest, history = bridge.save_signal(rec, tmp_path)
    assert latest.is_file()
    assert history.is_file()
    loaded = bridge.load_latest(tmp_path)
    assert loaded is not None
    assert loaded["ticker"] == "TSLA"
    # Second save appends history
    bridge.save_signal(rec, tmp_path)
    lines = history.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_check_secret_query_header_body():
    expected = "s3cret"
    assert bridge.check_secret({}, {"secret": ["s3cret"]}, None, expected)
    assert bridge.check_secret({"X-Webhook-Secret": "s3cret"}, {}, None, expected)
    assert bridge.check_secret({"Authorization": "Bearer s3cret"}, {}, None, expected)
    assert bridge.check_secret({}, {}, {"secret": "s3cret"}, expected)
    assert not bridge.check_secret({}, {"secret": ["wrong"]}, None, expected)
    assert not bridge.check_secret({}, {}, None, expected)


def _http_json(method: str, url: str, data: bytes | None = None, headers: dict | None = None):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"raw": body}
        return exc.code, payload


def test_server_webhook_and_auth(tmp_path: Path):
    secret = "test-secret-xyz"
    host, port = "127.0.0.1", 8765
    state = bridge.BridgeState(secret=secret, data_dir=tmp_path)
    handler = bridge.make_handler(state)
    httpd = bridge.ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.15)
    try:
        code, health = _http_json("GET", f"http://{host}:{port}/health")
        assert code == 200
        assert health["status"] == "ok"

        # Unauthorized
        code, err = _http_json(
            "POST",
            f"http://{host}:{port}/webhook?secret=wrong",
            data=b'{"ticker":"X"}',
            headers={"Content-Type": "application/json"},
        )
        assert code == 401

        # Authorized
        payload = b'{"ticker":"AAPL","price":101.5,"trend":"bullish","momentum":55.5,"timeframe":"5"}'
        code, ok = _http_json(
            "POST",
            f"http://{host}:{port}/webhook?secret={secret}",
            data=payload,
            headers={"Content-Type": "text/plain"},
        )
        assert code == 200
        assert ok["status"] == "saved"
        assert ok["ticker"] == "AAPL"

        code, latest = _http_json("GET", f"http://{host}:{port}/latest?secret={secret}")
        assert code == 200
        assert latest["ticker"] == "AAPL"
        assert latest["trend"] == "bullish"
        assert latest["momentum"] == 55.5
        assert (tmp_path / "latest_signal.json").is_file()
        assert (tmp_path / "signal_history.jsonl").is_file()
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_pine_template_exists_and_has_json_fields():
    root = Path(__file__).resolve().parent
    pine = root / "tradingview_alert_template.pine"
    assert pine.is_file()
    text = pine.read_text(encoding="utf-8")
    assert "ticker" in text
    assert "momentum" in text
    assert "trend" in text
    assert "@version=5" in text


def test_cli_requires_secret():
    assert bridge.main([]) == 2


# ---------------------------------------------------------------------------
# Phase D: HMAC body signatures + consecutive-duplicate dedupe
# ---------------------------------------------------------------------------

import hashlib
import hmac as hmac_mod
from datetime import datetime, timedelta, timezone


def _sig(key: str, body: bytes) -> str:
    return hmac_mod.new(key.encode(), body, hashlib.sha256).hexdigest()


def test_check_hmac_signature_unit():
    key, body = "hmac-key", b'{"ticker":"AAPL"}'
    good = _sig(key, body)
    assert bridge.check_hmac_signature({"X-Signature-SHA256": good}, body, key)
    assert bridge.check_hmac_signature({"x-signature": "sha256=" + good}, body, key)
    assert not bridge.check_hmac_signature({"X-Signature-SHA256": "00" * 32}, body, key)
    assert not bridge.check_hmac_signature({}, body, key)  # missing → reject
    # No key configured → HMAC layer is a no-op
    assert bridge.check_hmac_signature({}, body, "")


def _rec(ticker="AAPL", trend="bullish", momentum=55.0, minutes_ago=0.0):
    ts = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago))
    return {
        "received_at_utc": ts.replace(microsecond=0).isoformat(),
        "ticker": ticker,
        "trend": trend,
        "momentum": momentum,
    }


def test_is_duplicate_signal_rules():
    prev = _rec(minutes_ago=5)
    assert bridge.is_duplicate_signal(_rec(), prev)
    # RSI moves within the same tercile → still a duplicate
    assert bridge.is_duplicate_signal(_rec(momentum=58.0), prev)
    # Different tercile, trend, or ticker → not a duplicate
    assert not bridge.is_duplicate_signal(_rec(momentum=75.0), prev)
    assert not bridge.is_duplicate_signal(_rec(trend="bearish"), prev)
    assert not bridge.is_duplicate_signal(_rec(ticker="MSFT"), prev)
    # Outside the 10-minute window → not a duplicate
    old = _rec(minutes_ago=11)
    assert not bridge.is_duplicate_signal(_rec(), old)
    # No previous signal → never a duplicate
    assert not bridge.is_duplicate_signal(_rec(), None)


# ---------------------------------------------------------------------------
# Chart-vision upload proxy: dashboard (public) → bot /api/chart/analyze (local)
# ---------------------------------------------------------------------------

from http.server import BaseHTTPRequestHandler


def _make_fake_bot_handler(rec: dict):
    """Stand-in for the Robinhood bot's localhost chart API."""

    class FakeBotHandler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/chart/config":
                self._json(200, {"enabled": True, "provider": "xai",
                                 "model": "grok-vision", "mc_paths": 100000,
                                 "has_api_key": True})
            else:
                self._json(404, {"ok": False, "error": "not_found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            rec["last"] = json.loads(self.rfile.read(n).decode())
            rec["calls"].append(self.path)
            self._json(200, {"ok": True,
                             "extraction": {"ticker": "NVDA", "bias": "bullish",
                                            "image_last_price": 120.5},
                             "decision": {"ticker": "NVDA", "action": "long",
                                          "executable": False,
                                          "vision_confidence": 0.7,
                                          "mc_paths": 100000, "mc_horizon_days": 5,
                                          "risk": {"prob_profit": 0.55,
                                                   "var_95_pct": -0.08}},
                             "warnings": []})

    return FakeBotHandler


def _serve(port: int, handler):
    httpd = bridge.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def test_chart_proxy_forwards_and_strips_secret(tmp_path: Path):
    secret = "dash-secret"
    rec = {"last": None, "calls": []}
    bot = _serve(8781, _make_fake_bot_handler(rec))
    state = bridge.BridgeState(secret=secret, data_dir=tmp_path,
                               dashboard_bot_api="http://127.0.0.1:8781")
    front = _serve(8782, bridge.make_handler(state))
    time.sleep(0.15)
    base = "http://127.0.0.1:8782"
    try:
        # config is proxied from the bot (has_api_key stays a bool)
        code, cfg = _http_json("GET", f"{base}/dashboard/chart/config")
        assert code == 200
        assert cfg["provider"] == "xai" and cfg["has_api_key"] is True

        # missing secret → 401, and the bot is never called
        body = json.dumps({"image_base64": "QUJD", "mime_type": "image/png"}).encode()
        code, _ = _http_json("POST", f"{base}/dashboard/chart/analyze",
                             data=body, headers={"Content-Type": "application/json"})
        assert code == 401
        assert rec["calls"] == []

        # correct secret → forwarded; secret + junk stripped before the bot
        body = json.dumps({"image_base64": "QUJD", "mime_type": "image/png",
                           "ticker_hint": "NVDA", "secret": secret,
                           "junk": "x"}).encode()
        code, res = _http_json("POST", f"{base}/dashboard/chart/analyze",
                               data=body, headers={"Content-Type": "application/json"})
        assert code == 200 and res["ok"] is True
        assert res["decision"]["action"] == "long"
        fwd = rec["last"]
        assert fwd["image_base64"] == "QUJD"
        assert fwd["ticker_hint"] == "NVDA"
        assert "secret" not in fwd and "junk" not in fwd
        assert rec["calls"] == ["/api/chart/analyze"]

        # no image provided → 400 (bot still not called a second time)
        body = json.dumps({"secret": secret}).encode()
        code, _ = _http_json("POST", f"{base}/dashboard/chart/analyze",
                             data=body, headers={"Content-Type": "application/json"})
        assert code == 400
        assert rec["calls"] == ["/api/chart/analyze"]
    finally:
        for s in (front, bot):
            s.shutdown()
            s.server_close()


def test_chart_proxy_bot_unreachable(tmp_path: Path):
    # Point the proxy at a dead port → clean 502, never a 500 crash.
    state = bridge.BridgeState(secret="s", data_dir=tmp_path,
                               dashboard_bot_api="http://127.0.0.1:9")
    front = _serve(8783, bridge.make_handler(state))
    time.sleep(0.15)
    try:
        body = json.dumps({"image_base64": "QUJD", "secret": "s"}).encode()
        code, res = _http_json("POST", "http://127.0.0.1:8783/dashboard/chart/analyze",
                               data=body, headers={"Content-Type": "application/json"})
        assert code == 502
        assert res["ok"] is False and "unreachable" in res["error"]
    finally:
        front.shutdown()
        front.server_close()


def test_chart_upload_size_cap(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(bridge, "MAX_CHART_UPLOAD_BYTES", 100)
    state = bridge.BridgeState(secret="s", data_dir=tmp_path,
                               dashboard_bot_api="http://127.0.0.1:9")
    front = _serve(8784, bridge.make_handler(state))
    time.sleep(0.15)
    try:
        big = json.dumps({"image_base64": "A" * 500, "secret": "s"}).encode()
        code, res = _http_json("POST", "http://127.0.0.1:8784/dashboard/chart/analyze",
                               data=big, headers={"Content-Type": "application/json"})
        assert code == 413
    finally:
        front.shutdown()
        front.server_close()


def test_bot_request_unreachable_unit():
    code, data = bridge._bot_request("http://127.0.0.1:9", "/api/chart/config",
                                     timeout=1.0)
    assert code == 502
    assert data["ok"] is False and "unreachable" in data["error"]


def test_server_hmac_and_dedupe(tmp_path: Path):
    secret, key = "s3cret", "hmac-key-123"
    host, port = "127.0.0.1", 8766
    state = bridge.BridgeState(secret=secret, data_dir=tmp_path,
                               hmac_key=key, dedupe_window=600)
    httpd = bridge.ThreadingHTTPServer((host, port), bridge.make_handler(state))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.15)
    url = f"http://{host}:{port}/webhook?secret={secret}"
    body = b'{"ticker":"AAPL","price":101.5,"trend":"bullish","momentum":55.5}'
    try:
        # Missing signature → 401
        code, err = _http_json("POST", url, data=body,
                               headers={"Content-Type": "application/json"})
        assert code == 401
        assert err["error"] == "bad_signature"

        # Valid signature → saved
        headers = {"Content-Type": "application/json",
                   "X-Signature-SHA256": _sig(key, body)}
        code, ok = _http_json("POST", url, data=body, headers=headers)
        assert code == 200
        assert ok["status"] == "saved"
        first = bridge.load_latest(tmp_path)
        first_ts = first["received_at_utc"]

        # Identical consecutive signal → history only, latest untouched
        code, dup = _http_json("POST", url, data=body, headers=headers)
        assert code == 200
        assert dup["status"] == "duplicate_ignored"
        latest = bridge.load_latest(tmp_path)
        assert latest["received_at_utc"] == first_ts
        history = (tmp_path / "signal_history.jsonl").read_text().strip()
        assert len(history.splitlines()) == 2
        assert "duplicate_of_received_at_utc" in history.splitlines()[1]

        # Different tercile → new latest
        body2 = b'{"ticker":"AAPL","price":102,"trend":"bullish","momentum":75}'
        headers2 = {"Content-Type": "application/json",
                    "X-Signature-SHA256": _sig(key, body2)}
        code, ok2 = _http_json("POST", url, data=body2, headers=headers2)
        assert code == 200
        assert ok2["status"] == "saved"
        assert bridge.load_latest(tmp_path)["momentum"] == 75.0
    finally:
        httpd.shutdown()
        httpd.server_close()
