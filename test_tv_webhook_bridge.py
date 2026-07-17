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
