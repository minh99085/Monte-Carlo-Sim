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
import hashlib
import hmac
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
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5001
DEFAULT_DATA_DIR = Path("tv_data")
LATEST_FILENAME = "latest_signal.json"
HISTORY_FILENAME = "signal_history.jsonl"

# Chart-image uploads are base64 JSON, larger than a webhook alert. A
# TradingView screenshot (base64-inflated ~33%) comfortably fits in 12 MB.
MAX_CHART_UPLOAD_BYTES = 12_000_000
# Fields the dashboard proxy is allowed to forward to the bot's chart API.
# Everything else (including the dashboard secret) is dropped.
_CHART_FORWARD_FIELDS = (
    "image_base64", "image_url", "image_path", "mime_type",
    "ticker_hint", "run_validation", "run_monte_carlo",
    "mc_paths", "execution_mode",
)

# Consecutive identical signals (same ticker / trend / RSI bucket) within this
# window are logged to history but do NOT rewrite latest_signal.json, so a
# repeated alert cannot refresh a signal's timestamp and defeat the
# downstream freshness check.
DEDUPE_WINDOW_SECONDS = 600

# RSI tercile edges — keep in sync with signal_calibration.RSI_LOW_MAX /
# RSI_HIGH_MIN (duplicated here so the bridge stays stdlib-only).
_RSI_LOW_MAX = 40.0
_RSI_HIGH_MIN = 60.0

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


def save_signal(
    record: Dict[str, Any],
    data_dir: Path,
    *,
    update_latest: bool = True,
) -> Tuple[Path, Path]:
    """
    Write ``record`` as the latest signal and append one line to history.

    With ``update_latest=False`` only the history line is appended (used for
    deduplicated repeats so latest_signal.json keeps its original timestamp).
    Uses a lock so concurrent webhooks do not corrupt the files.
    """
    ensure_data_dir(data_dir)
    latest = latest_path(data_dir)
    history = history_path(data_dir)
    text = json.dumps(record, ensure_ascii=False, indent=2)
    line = json.dumps(record, ensure_ascii=False)

    with _SAVE_LOCK:
        if update_latest:
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


def check_hmac_signature(
    headers: Dict[str, str],
    raw_body: bytes,
    hmac_key: str,
) -> bool:
    """
    Verify an HMAC-SHA256 signature of the raw request body.

    Accepted headers (first found wins): ``X-Signature-SHA256``,
    ``X-Signature``, ``X-Hub-Signature-256``. The value is the hex digest,
    optionally prefixed with ``sha256=``. Comparison is constant-time.
    """
    if not hmac_key:
        return True
    lowered = {str(k).lower(): v for k, v in headers.items()}
    provided = ""
    for name in ("x-signature-sha256", "x-signature", "x-hub-signature-256"):
        if lowered.get(name):
            provided = str(lowered[name]).strip()
            break
    if not provided:
        return False
    if provided.lower().startswith("sha256="):
        provided = provided[7:].strip()
    expected = hmac.new(hmac_key.encode("utf-8"), raw_body,
                        hashlib.sha256).hexdigest()
    return hmac.compare_digest(provided.lower(), expected)


def _rsi_bucket(momentum: Any) -> Optional[str]:
    """Stdlib tercile bucket for dedupe (mirror of signal_calibration edges)."""
    try:
        m = float(momentum)
    except (TypeError, ValueError):
        return None
    if m < _RSI_LOW_MAX:
        return "low"
    if m > _RSI_HIGH_MIN:
        return "high"
    return "mid"


def _parse_utc(ts: Any) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def is_duplicate_signal(
    record: Dict[str, Any],
    previous: Optional[Dict[str, Any]],
    window_seconds: float = DEDUPE_WINDOW_SECONDS,
) -> bool:
    """
    True when ``record`` repeats ``previous`` — same ticker, trend and
    bucketed RSI — within ``window_seconds``.
    """
    if not previous or window_seconds <= 0:
        return False
    if str(record.get("ticker")) != str(previous.get("ticker")):
        return False
    if str(record.get("trend")) != str(previous.get("trend")):
        return False
    if _rsi_bucket(record.get("momentum")) != _rsi_bucket(previous.get("momentum")):
        return False
    t_new = _parse_utc(record.get("received_at_utc"))
    t_old = _parse_utc(previous.get("received_at_utc"))
    if t_new is None or t_old is None:
        return False
    return abs((t_new - t_old).total_seconds()) <= float(window_seconds)


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class BridgeState:
    """Shared config for request handlers."""

    def __init__(
        self,
        secret: str,
        data_dir: Path,
        hmac_key: str = "",
        dedupe_window: float = DEDUPE_WINDOW_SECONDS,
        dashboard_html_path: Optional[Path] = None,
        dashboard_app_dir: Optional[Path] = None,
        dashboard_bot_api: str = "http://127.0.0.1:8810",
    ):
        self.secret = secret
        self.data_dir = data_dir
        self.hmac_key = hmac_key or ""
        self.dedupe_window = float(dedupe_window)
        self.received_count = 0
        self.lock = threading.Lock()
        # Dashboard file paths default to the module dir so a checkout run
        # with default flags finds the HTML alongside this file.
        module_dir = Path(__file__).resolve().parent
        self.dashboard_html_path = dashboard_html_path or (module_dir / "dashboard.html")
        self.dashboard_app_dir = dashboard_app_dir or module_dir
        self.dashboard_bot_api = dashboard_bot_api


def _bot_request(
    bot_api: str,
    path: str,
    *,
    method: str = "GET",
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 60.0,
) -> Tuple[int, Dict[str, Any]]:
    """Call the Robinhood bot's localhost HTTP API and return (status, json).

    The bot API is bound to 127.0.0.1, so this server-side hop is the only
    way a browser on the public dashboard can reach it — keeping the bot's
    trust boundary intact. Any transport error is surfaced as a structured
    dict rather than raised, so the dashboard degrades gracefully when the
    bot container is down.
    """
    url = bot_api.rstrip("/") + path
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            code = resp.getcode()
    except urllib.error.HTTPError as exc:
        # The bot answered with a non-2xx (e.g. 400/500) — pass its JSON body
        # through so the operator sees the real reason.
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            return exc.code, json.loads(raw)
        except (ValueError, TypeError):
            return exc.code, {"ok": False, "error": raw or f"HTTP {exc.code}"}
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return 502, {"ok": False,
                     "error": f"bot API unreachable at {url}: {exc}"}
    try:
        return code, json.loads(body)
    except (ValueError, TypeError):
        return code, {"ok": False, "error": "bot returned non-JSON response"}


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

        def _send_html(self, code: int, body: str) -> None:
            raw = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"

            # Dashboard: HTML page and its JSON API. Both are unauthenticated
            # because they carry only *derived* status (no signal contents,
            # no secrets); the raw /latest endpoint remains secret-gated.
            if path == "/dashboard":
                try:
                    html = state.dashboard_html_path.read_text(encoding="utf-8")
                except OSError:
                    self._send_html(500, "<h1>Dashboard HTML not installed</h1>")
                    return
                self._send_html(200, html)
                return
            if path == "/dashboard/api":
                try:
                    from dashboard import build_snapshot
                    snap = build_snapshot(
                        app_dir=state.dashboard_app_dir,
                        bot_api=state.dashboard_bot_api,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._send_json(500, {"error": "aggregate_failed",
                                          "detail": str(exc)})
                    return
                self._send_json(200, snap)
                return

            # Chart-vision provider config (proxied from the bot). Read-only,
            # no secrets: the bot returns has_api_key as a bool, never the key.
            if path == "/dashboard/chart/config":
                code, data = _bot_request(
                    state.dashboard_bot_api, "/api/chart/config", timeout=10.0
                )
                self._send_json(code, data)
                return

            # Paper book + scout (secret-gated: portfolio state + heavy
            # data pulls should not be world-readable on port 80).
            if path in ("/dashboard/paper/book", "/dashboard/scout"):
                query = parse_qs(parsed.query)
                if not check_secret(self._headers_dict(), query, None,
                                    state.secret):
                    self._send_json(401, {"ok": False,
                                          "error": "unauthorized"})
                    return
                target = ("/api/paper/book" if path.endswith("book")
                          else "/api/scout/run")
                code, data = _bot_request(
                    state.dashboard_bot_api, target, timeout=180.0)
                self._send_json(code, data)
                return

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

            self._send_json(404, {"error": "not_found",
                "paths": ["/", "/health", "/webhook", "/latest",
                          "/dashboard", "/dashboard/api",
                          "/dashboard/chart/config",
                          "/dashboard/chart/analyze"]})

        def _handle_chart_analyze(self) -> None:
            """Proxy an operator's chart-image upload to the bot's vision API.

            Secret-gated (the dashboard is public on port 80, and each call
            spends paid vision-API credits) and size-capped. Only whitelisted
            fields are forwarded; the secret is never passed to the bot.
            """
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_CHART_UPLOAD_BYTES:
                self._send_json(413, {"ok": False, "error": "image_too_large",
                                      "max_bytes": MAX_CHART_UPLOAD_BYTES})
                return
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8", errors="replace") or "null")
            except json.JSONDecodeError:
                body = None
            if not isinstance(body, dict):
                self._send_json(400, {"ok": False,
                                      "error": "expected JSON object body"})
                return

            headers = self._headers_dict()
            query = parse_qs(urlparse(self.path).query)
            if not check_secret(headers, query, body, state.secret):
                self._send_json(401, {"ok": False, "error": "unauthorized",
                    "hint": "pass the dashboard secret"})
                return

            if not any(body.get(k) for k in
                       ("image_base64", "image_url", "image_path")):
                self._send_json(400, {"ok": False,
                    "error": "provide image_base64, image_url, or image_path"})
                return

            forward = {k: body[k] for k in _CHART_FORWARD_FIELDS if k in body}
            code, data = _bot_request(
                state.dashboard_bot_api, "/api/chart/analyze",
                method="POST", payload=forward, timeout=120.0,
            )
            logger.info("Chart analyze proxied → bot returned %s (ok=%s)",
                        code, isinstance(data, dict) and data.get("ok"))
            self._send_json(code, data)

        def _handle_chart_confluence(self) -> None:
            """Combine per-chart vision reads into one advisory stance.

            Pure local computation (no bot call, no vision spend) but still
            secret-gated so the public dashboard exposes nothing interactive
            without the operator's secret.
            """
            length = int(self.headers.get("Content-Length") or 0)
            if length > 1_000_000:
                self._send_json(413, {"ok": False, "error": "payload_too_large"})
                return
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8", errors="replace") or "null")
            except json.JSONDecodeError:
                body = None
            if not isinstance(body, dict):
                self._send_json(400, {"ok": False,
                                      "error": "expected JSON object body"})
                return
            headers = self._headers_dict()
            query = parse_qs(urlparse(self.path).query)
            if not check_secret(headers, query, body, state.secret):
                self._send_json(401, {"ok": False, "error": "unauthorized"})
                return
            charts = body.get("charts")
            if not isinstance(charts, list) or not charts:
                self._send_json(400, {"ok": False,
                                      "error": "provide charts: [...]"})
                return
            position = body.get("position")
            if not isinstance(position, dict):
                position = None
            try:
                from chart_confluence import combine
                verdict = combine(charts, position=position)
            except Exception as exc:  # noqa: BLE001
                self._send_json(500, {"ok": False,
                                      "error": f"confluence_failed: {exc}"})
                return
            self._send_json(200, {"ok": True, "confluence": verdict})

        def _handle_paper_trade(self, path: str) -> None:
            """Proxy paper open/close to the bot (secret-gated, whitelisted)."""
            length = int(self.headers.get("Content-Length") or 0)
            if length > 100_000:
                self._send_json(413, {"ok": False, "error": "payload_too_large"})
                return
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8", errors="replace") or "null")
            except json.JSONDecodeError:
                body = None
            if not isinstance(body, dict):
                self._send_json(400, {"ok": False,
                                      "error": "expected JSON object body"})
                return
            query = parse_qs(urlparse(self.path).query)
            if not check_secret(self._headers_dict(), query, body,
                                state.secret):
                self._send_json(401, {"ok": False, "error": "unauthorized"})
                return
            if path.endswith("open"):
                target = "/api/paper/open"
                fields = ("symbol", "stop_pct", "horizon_days", "thesis")
            else:
                target = "/api/paper/close"
                fields = ("symbol", "reason")
            forward = {k: body[k] for k in fields if k in body}
            if not forward.get("symbol"):
                self._send_json(400, {"ok": False, "error": "symbol required"})
                return
            code, data = _bot_request(state.dashboard_bot_api, target,
                                      method="POST", payload=forward,
                                      timeout=60.0)
            logger.info("Paper %s %s → %s", target, forward.get("symbol"), code)
            self._send_json(code, data)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path.rstrip("/") or "/"
            if path == "/dashboard/chart/analyze":
                self._handle_chart_analyze()
                return
            if path == "/dashboard/chart/confluence":
                self._handle_chart_confluence()
                return
            if path in ("/dashboard/paper/open", "/dashboard/paper/close"):
                self._handle_paper_trade(path)
                return
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

            # Optional HMAC-SHA256 body signature (defense in depth)
            if state.hmac_key and not check_hmac_signature(headers, raw, state.hmac_key):
                logger.warning(
                    "Rejected webhook from %s (bad or missing HMAC signature)",
                    self.client_address[0],
                )
                self._send_json(401, {"error": "bad_signature",
                                      "hint": "X-Signature-SHA256: hex(hmac_sha256(body))"})
                return

            payload = parse_payload(raw, content_type)
            # Never persist the shared secret
            if isinstance(payload, dict):
                for k in ("secret", "token", "webhook_secret"):
                    payload.pop(k, None)

            record = normalize_signal(payload, source_ip=self.client_address[0])

            # Dedupe: identical consecutive signal within the window → keep it
            # in history but do NOT rewrite latest_signal.json (its timestamp
            # must not be refreshed by repeats).
            previous = load_latest(state.data_dir)
            duplicate = is_duplicate_signal(record, previous, state.dedupe_window)
            if duplicate:
                record["duplicate_of_received_at_utc"] = previous.get("received_at_utc")
                latest, history = save_signal(record, state.data_dir,
                                              update_latest=False)
                logger.info(
                    "Duplicate signal %s trend=%s momentum=%s within %ss — "
                    "history only, latest untouched",
                    record.get("ticker"), record.get("trend"),
                    record.get("momentum"), int(state.dedupe_window),
                )
                self._send_json(200, {
                    "status": "duplicate_ignored",
                    "ticker": record.get("ticker"),
                    "trend": record.get("trend"),
                    "history_file": str(history),
                })
                return

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


def run_server(
    host: str,
    port: int,
    secret: str,
    data_dir: Path,
    hmac_key: str = "",
    dedupe_window: float = DEDUPE_WINDOW_SECONDS,
    dashboard_html_path: Optional[Path] = None,
    dashboard_app_dir: Optional[Path] = None,
    dashboard_bot_api: str = "http://127.0.0.1:8810",
) -> None:
    ensure_data_dir(data_dir)
    state = BridgeState(
        secret=secret, data_dir=data_dir,
        hmac_key=hmac_key, dedupe_window=dedupe_window,
        dashboard_html_path=dashboard_html_path,
        dashboard_app_dir=dashboard_app_dir,
        dashboard_bot_api=dashboard_bot_api,
    )
    handler = make_handler(state)
    httpd = ThreadingHTTPServer((host, port), handler)
    logger.info("TradingView webhook bridge listening on http://%s:%s", host, port)
    logger.info("  POST webhook  : http://%s:%s/webhook?secret=<your-secret>", host, port)
    logger.info("  GET  health   : http://%s:%s/health", host, port)
    logger.info("  GET  latest   : http://%s:%s/latest?secret=<your-secret>", host, port)
    logger.info("  GET  dashboard: http://%s:%s/dashboard", host, port)
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
        "--hmac-key",
        default=os.environ.get("TV_BRIDGE_HMAC_KEY", ""),
        help="Optional HMAC-SHA256 key: when set, every webhook body must "
             "carry a matching X-Signature-SHA256 header "
             "(or set TV_BRIDGE_HMAC_KEY).",
    )
    p.add_argument(
        "--dedupe-window",
        type=float,
        default=float(os.environ.get("TV_BRIDGE_DEDUPE_WINDOW",
                                     DEDUPE_WINDOW_SECONDS)),
        help="Seconds within which an identical consecutive signal (same "
             "ticker/trend/RSI bucket) only goes to history, leaving "
             f"latest_signal.json untouched (default {DEDUPE_WINDOW_SECONDS}; "
             "0 disables).",
    )
    p.add_argument(
        "--dashboard-html",
        default=os.environ.get("TV_BRIDGE_DASHBOARD_HTML", ""),
        help="Path to dashboard.html (default: alongside this script).",
    )
    p.add_argument(
        "--dashboard-app-dir",
        default=os.environ.get("TV_BRIDGE_DASHBOARD_APP_DIR", ""),
        help="Monte-Carlo-Sim app dir the dashboard reads outputs from "
             "(default: this script's parent dir).",
    )
    p.add_argument(
        "--dashboard-bot-api",
        default=os.environ.get("TV_BRIDGE_DASHBOARD_BOT_API",
                               "http://127.0.0.1:8810"),
        help="Robinhood bot API URL the dashboard aggregates from "
             "(default http://127.0.0.1:8810).",
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
    run_server(
        args.host,
        int(args.port),
        secret,
        data_dir,
        hmac_key=(args.hmac_key or "").strip(),
        dedupe_window=float(args.dedupe_window),
        dashboard_html_path=Path(args.dashboard_html) if args.dashboard_html else None,
        dashboard_app_dir=Path(args.dashboard_app_dir) if args.dashboard_app_dir else None,
        dashboard_bot_api=(args.dashboard_bot_api or "").strip()
                          or "http://127.0.0.1:8810",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
