"""
TradingView → Monte Carlo tactical bridge (Phase 3).

A small local HTTP server that:
  1. Receives webhook POSTs from TradingView alerts
  2. Checks a shared secret key
  3. Parses JSON (or JSON-in-text) payloads
  4. Saves the latest signal + an append-only history file

No extra packages required (stdlib only).

Quick start
-----------
    python tv_webhook_bridge.py --secret my-secret --port 5001

Then point a TradingView alert webhook at:

    http://YOUR_PUBLIC_URL/webhook?secret=my-secret

For local testing without TradingView, see PHASE3_README.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5001
DEFAULT_DATA_DIR = Path("tv_data")
LATEST_FILENAME = "latest_signal.json"
HISTORY_FILENAME = "signal_history.jsonl"

logger = logging.getLogger("tv_webhook_bridge")


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_data_dir(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def latest_path(data_dir: Path) -> Path:
    return data_dir / LATEST_FILENAME


def history_path(data_dir: Path) -> Path:
    return data_dir / HISTORY_FILENAME


def save_signal(record: Dict[str, Any], data_dir: Path) -> Tuple[Path, Path]:
    """
    Write ``record`` as the latest signal and append one line to history.

    Uses a lock so concurrent webhooks do not corrupt the files.
    """
    ensure_data_dir(data_dir)
    latest = latest_path(data_dir)
    history = history_path(data_dir)
    text = json.dumps(record, ensure_ascii=False, indent=2)
    line = json.dumps(record, ensure_ascii=False)

    with _SAVE_LOCK:
        latest.write_text(text + "\n", encoding="utf-8")
        with history.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return latest, history


_SAVE_LOCK = threading.Lock()


def load_latest(data_dir: Path) -> Optional[Dict[str, Any]]:
    path = latest_path(data_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------


def parse_payload(raw_body: bytes, content_type: str = "") -> Dict[str, Any]:
    """
    Turn a TradingView webhook body into a Python dict.

    TradingView usually sends the *alert message* as plain text. Our Pine
    template puts a single JSON object in that message, so we:
      1. Try strict JSON parse of the whole body
      2. Else try to find the first {...} block and parse that
      3. Else store the raw text under ``raw_message``
    """
    text = raw_body.decode("utf-8", errors="replace").strip()
    if not text:
        return {"raw_message": "", "parse_status": "empty"}

    # 1) Whole body is JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            obj = dict(obj)
            obj.setdefault("parse_status", "json")
            return obj
        return {"data": obj, "parse_status": "json_non_object"}
    except json.JSONDecodeError:
        pass

    # 2) JSON object embedded in text
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                obj = dict(obj)
                obj["parse_status"] = "embedded_json"
                obj["raw_message"] = text
                return obj
        except json.JSONDecodeError:
            pass

    # 3) Plain text fallback
    return {"raw_message": text, "parse_status": "text"}


def normalize_signal(payload: Dict[str, Any], *, source_ip: str = "") -> Dict[str, Any]:
    """
    Build a clean, consistent record for the tactical simulator to read later.

    Accepts flexible field names from Pine (ticker / symbol, trend / trend_direction,
    momentum / mom / rsi, price / close, etc.).
    """
    def _first(*keys, default=None):
        for k in keys:
            if k in payload and payload[k] not in (None, ""):
                return payload[k]
        return default

    ticker = _first("ticker", "symbol", "sym", default="UNKNOWN")
    price = _first("price", "close", "last", default=None)
    trend = _first("trend", "trend_direction", "direction", "signal", default="unknown")
    momentum = _first("momentum", "mom", "strength", "rsi", "mom_value", default=None)
    timeframe = _first("timeframe", "tf", "interval", default=None)
    strategy = _first("strategy", "script", "name", default="tv_webhook")

    # Coerce numbers when possible
    def _num(v):
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return v

    record = {
        "received_at_utc": utc_now_iso(),
        "source": "tradingview",
        "source_ip": source_ip,
        "ticker": str(ticker).upper() if ticker is not None else "UNKNOWN",
        "price": _num(price),
        "trend": str(trend).lower() if trend is not None else "unknown",
        "momentum": _num(momentum),
        "timeframe": timeframe,
        "strategy": strategy,
        "parse_status": payload.get("parse_status", "ok"),
        # Keep the full original payload for debugging / future fields
        "raw": payload,
    }
    return record


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def check_secret(
    headers: Dict[str, str],
    query: Dict[str, list],
    body_obj: Optional[Dict[str, Any]],
    expected: str,
) -> bool:
    """
    Accept the secret from (first match wins):
      - Query:   ?secret=...
      - Header:  X-Webhook-Secret: ...
      - Header:  Authorization: Bearer ...
      - JSON body field: "secret"
    """
    if not expected:
        return False

    q = query.get("secret") or query.get("token")
    if q and q[0] == expected:
        return True

    # Headers are case-insensitive in BaseHTTPRequestHandler but we normalize
    h_secret = headers.get("x-webhook-secret") or headers.get("X-Webhook-Secret")
    if h_secret and h_secret.strip() == expected:
        return True

    auth = headers.get("authorization") or headers.get("Authorization") or ""
    if auth.lower().startswith("bearer ") and auth[7:].strip() == expected:
        return True

    if isinstance(body_obj, dict):
        for key in ("secret", "token", "webhook_secret"):
            if body_obj.get(key) == expected:
                return True

    return False


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class BridgeState:
    """Shared config for request handlers."""

    def __init__(self, secret: str, data_dir: Path):
        self.secret = secret
        self.data_dir = data_dir
        self.received_count = 0
        self.lock = threading.Lock()


def make_handler(state: BridgeState):
    class WebhookHandler(BaseHTTPRequestHandler):
        # Quieter logs; we use the module logger
        def log_message(self, fmt: str, *args) -> None:
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send_json(self, code: int, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _headers_dict(self) -> Dict[str, str]:
            return {k: v for k, v in self.headers.items()}

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"

            if path in ("/", "/health"):
                latest = load_latest(state.data_dir)
                self._send_json(200, {
                    "status": "ok",
                    "service": "tv_webhook_bridge",
                    "received_count": state.received_count,
                    "data_dir": str(state.data_dir.resolve()),
                    "latest_ticker": (latest or {}).get("ticker"),
                    "latest_received_at_utc": (latest or {}).get("received_at_utc"),
                })
                return

            if path == "/latest":
                # Optional: require secret for reading latest too
                query = parse_qs(parsed.query)
                if not check_secret(self._headers_dict(), query, None, state.secret):
                    self._send_json(401, {"error": "unauthorized", "hint": "pass ?secret=..."})
                    return
                latest = load_latest(state.data_dir)
                if latest is None:
                    self._send_json(404, {"error": "no_signal_yet"})
                else:
                    self._send_json(200, latest)
                return

            self._send_json(404, {"error": "not_found", "paths": ["/", "/health", "/webhook", "/latest"]})

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path != "/webhook":
                self._send_json(404, {"error": "not_found", "use": "POST /webhook"})
                return

            length = int(self.headers.get("Content-Length") or 0)
            # Cap body size (1 MB) to avoid abuse
            if length > 1_000_000:
                self._send_json(413, {"error": "payload_too_large"})
                return
            raw = self.rfile.read(length) if length > 0 else b""
            content_type = self.headers.get("Content-Type", "")
            query = parse_qs(parsed.query)
            headers = self._headers_dict()

            # Peek JSON for secret-in-body without failing plain text
            body_for_auth: Optional[Dict[str, Any]] = None
            try:
                tmp = json.loads(raw.decode("utf-8", errors="replace") or "null")
                if isinstance(tmp, dict):
                    body_for_auth = tmp
            except json.JSONDecodeError:
                body_for_auth = None

            if not check_secret(headers, query, body_for_auth, state.secret):
                logger.warning("Rejected webhook from %s (bad secret)", self.client_address[0])
                self._send_json(401, {"error": "unauthorized"})
                return

            payload = parse_payload(raw, content_type)
            # Never persist the shared secret
            if isinstance(payload, dict):
                for k in ("secret", "token", "webhook_secret"):
                    payload.pop(k, None)

            record = normalize_signal(payload, source_ip=self.client_address[0])
            latest, history = save_signal(record, state.data_dir)

            with state.lock:
                state.received_count += 1
                count = state.received_count

            logger.info(
                "Saved signal #%s %s trend=%s momentum=%s price=%s → %s",
                count,
                record.get("ticker"),
                record.get("trend"),
                record.get("momentum"),
                record.get("price"),
                latest,
            )
            self._send_json(200, {
                "status": "saved",
                "received_count": count,
                "ticker": record.get("ticker"),
                "trend": record.get("trend"),
                "latest_file": str(latest),
                "history_file": str(history),
            })

    return WebhookHandler


def run_server(host: str, port: int, secret: str, data_dir: Path) -> None:
    ensure_data_dir(data_dir)
    state = BridgeState(secret=secret, data_dir=data_dir)
    handler = make_handler(state)
    httpd = ThreadingHTTPServer((host, port), handler)
    logger.info("TradingView webhook bridge listening on http://%s:%s", host, port)
    logger.info("  POST webhook : http://%s:%s/webhook?secret=<your-secret>", host, port)
    logger.info("  GET  health  : http://%s:%s/health", host, port)
    logger.info("  GET  latest  : http://%s:%s/latest?secret=<your-secret>", host, port)
    logger.info("  Data folder  : %s", data_dir.resolve())
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
    finally:
        httpd.server_close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Local TradingView webhook bridge for Monte-Carlo-Sim (Phase 3).",
    )
    p.add_argument(
        "--host",
        default=os.environ.get("TV_BRIDGE_HOST", DEFAULT_HOST),
        help=f"Bind address (default {DEFAULT_HOST}).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("TV_BRIDGE_PORT", DEFAULT_PORT)),
        help=f"Port (default {DEFAULT_PORT}).",
    )
    p.add_argument(
        "--secret",
        default=os.environ.get("TV_BRIDGE_SECRET", ""),
        help="Shared secret key required on every webhook (or set TV_BRIDGE_SECRET).",
    )
    p.add_argument(
        "--data-dir",
        default=os.environ.get("TV_BRIDGE_DATA_DIR", str(DEFAULT_DATA_DIR)),
        help=f"Folder for latest_signal.json + history (default {DEFAULT_DATA_DIR}).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    secret = (args.secret or "").strip()
    if not secret:
        logger.error(
            "A secret is required. Pass --secret YOUR_KEY or set TV_BRIDGE_SECRET."
        )
        return 2
    if secret in ("changeme", "secret", "password", "1234"):
        logger.warning("Weak secret detected — use a long random string in production.")

    data_dir = Path(args.data_dir)
    run_server(args.host, int(args.port), secret, data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
