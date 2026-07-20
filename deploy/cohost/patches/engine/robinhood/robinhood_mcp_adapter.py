"""Robinhood Trading MCP adapter — OAuth, streamable HTTP, reconnect, health."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
from mcp import ClientSession
from mcp.client.auth import OAuthClientProvider
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl

from engine.robinhood.audit_log import AuditLog
from engine.robinhood.config import RobinhoodConfig
from engine.robinhood.constants import PLACE_TOOLS
from engine.robinhood.mcp_catalog import save_catalog
from engine.robinhood.oauth_storage import FileTokenStorage

logger = logging.getLogger("hermes.robinhood.mcp")


def _unwrap_block(block: Any) -> Any:
    """Turn an MCP text content block carrying JSON into its payload."""
    if isinstance(block, dict) and block.get("type") == "text":
        text = block.get("text")
        if isinstance(text, str):
            stripped = text.strip()
            if stripped.startswith(("{", "[")):
                try:
                    import json as _json

                    return _json.loads(stripped)
                except ValueError:
                    return text
            return text
    return block


@dataclass
class MCPHealth:
    connected: bool = False
    authenticated: bool = False
    last_ok_ts: float | None = None
    last_error: str | None = None
    tool_count: int = 0
    tools: list[str] = field(default_factory=list)
    reconnect_attempts: int = 0


class RobinhoodMCPAdapter:
    """Manages a long-lived MCP session to Robinhood's Trading MCP server."""

    def __init__(
        self,
        config: RobinhoodConfig,
        audit: AuditLog | None = None,
        *,
        redirect_handler: Callable[[str], Awaitable[None]] | None = None,
        callback_handler: Callable[[], Awaitable[tuple[str, str | None]]] | None = None,
    ) -> None:
        self.config = config
        self.audit = audit or AuditLog(config.data_dir)
        self.storage = FileTokenStorage(config.data_dir)
        self.health = MCPHealth()
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._http: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._redirect_handler = redirect_handler or self._default_redirect_handler
        self._callback_handler = callback_handler or self._default_callback_handler

    def _oauth_provider(self) -> OAuthClientProvider:
        return OAuthClientProvider(
            server_url=self.config.mcp_server_base,
            client_metadata=OAuthClientMetadata(
                client_name=self.config.oauth_client_name,
                redirect_uris=[AnyUrl(self.config.oauth_redirect_uri)],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
            ),
            storage=self.storage,
            redirect_handler=self._redirect_handler,
            callback_handler=self._callback_handler,
        )

    @staticmethod
    async def _default_redirect_handler(auth_url: str) -> None:
        logger.info("OAuth required — open this URL in a desktop browser: %s", auth_url)
        print(f"\n=== Robinhood OAuth ===\nOpen: {auth_url}\n")

    @staticmethod
    async def _default_callback_handler() -> tuple[str, str | None]:
        # VPS / headless: operator pastes callback URL after browser auth.
        callback_url = await asyncio.to_thread(
            input, "Paste the full callback URL from your browser: "
        )
        parsed = urlparse(callback_url.strip())
        params = parse_qs(parsed.query)
        if "code" not in params:
            raise ValueError("callback URL missing authorization code")
        return params["code"][0], (params.get("state") or [None])[0]

    async def connect(self, *, interactive_oauth: bool = False) -> None:
        """Establish MCP session. Raises if OAuth tokens missing and not interactive."""
        async with self._lock:
            await self._disconnect_unlocked()
            if not self.storage.has_tokens() and not interactive_oauth:
                self.health.connected = False
                self.health.authenticated = False
                self.health.last_error = (
                    "no OAuth tokens — run scripts/robinhood_oauth_login.py first"
                )
                raise RuntimeError(self.health.last_error)

            oauth = self._oauth_provider()
            self._http = httpx.AsyncClient(auth=oauth, follow_redirects=True, timeout=60.0)
            self._stack = AsyncExitStack()
            read, write, _ = await self._stack.enter_async_context(
                streamable_http_client(self.config.mcp_url, http_client=self._http)
            )
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._session = session
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            catalog_entries: list[dict[str, Any]] = []
            for t in tools.tools:
                schema = getattr(t, "inputSchema", None) or getattr(t, "input_schema", None)
                if hasattr(schema, "model_dump"):
                    schema = schema.model_dump(mode="json")
                catalog_entries.append(
                    {
                        "name": t.name,
                        "description": getattr(t, "description", "") or "",
                        "input_schema": schema,
                    }
                )
            save_catalog(self.config.data_dir, tools=catalog_entries, mcp_url=self.config.mcp_url)
            self.health = MCPHealth(
                connected=True,
                authenticated=self.storage.has_tokens(),
                last_ok_ts=time.time(),
                tool_count=len(names),
                tools=names,
            )
            self.audit.record("mcp_connected", details={"tools": names})
            logger.info("MCP connected — %d tools available", len(names))

    async def _disconnect_unlocked(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        if self._http is not None:
            await self._http.aclose()
        self._stack = None
        self._session = None
        self._http = None
        self.health.connected = False

    async def disconnect(self) -> None:
        async with self._lock:
            await self._disconnect_unlocked()

    async def health_check(self) -> MCPHealth:
        """Ping MCP via list_tools; updates health snapshot."""
        try:
            async with self._lock:
                if self._session is None:
                    raise RuntimeError("not connected")
                tools = await self._session.list_tools()
                names = sorted(t.name for t in tools.tools)
                self.health.connected = True
                self.health.last_ok_ts = time.time()
                self.health.last_error = None
                self.health.tool_count = len(names)
                self.health.tools = names
        except Exception as exc:  # noqa: BLE001
            self.health.connected = False
            self.health.last_error = str(exc)
            self.audit.record("mcp_health_fail", reason=str(exc))
        return self.health

    async def list_tools(self) -> list[str]:
        async with self._lock:
            if self._session is None:
                raise RuntimeError("not connected")
            tools = await self._session.list_tools()
            return sorted(t.name for t in tools.tools)

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """Raw MCP tool invocation (use SafeRobinhoodClient for gated calls)."""
        args = arguments or {}
        self.audit.record("mcp_tool_call", tool=name, details={"arguments": args})
        async with self._lock:
            if self._session is None:
                raise RuntimeError("not connected")
            result = await self._session.call_tool(name, args)
        # Normalize to plain dict/list for callers. Prefer the structured
        # result when the server provides one; otherwise unwrap content
        # blocks, JSON-decoding text blocks so downstream parsers see real
        # payloads instead of {"type": "text", "text": "..."} wrappers.
        structured = getattr(result, "structuredContent", None)
        if structured is None:
            structured = getattr(result, "structured_content", None)
        if isinstance(structured, (dict, list)) and structured:
            self.audit.record("mcp_tool_result", tool=name,
                              details={"structured": True})
            return structured
        payload: list[Any] = []
        for block in result.content:
            if hasattr(block, "model_dump"):
                payload.append(_unwrap_block(block.model_dump(mode="json")))
            else:
                payload.append(str(block))
        self.audit.record("mcp_tool_result", tool=name, details={"blocks": len(payload)})
        if len(payload) == 1:
            return payload[0]
        return payload

    async def run_reconnect_loop(self) -> None:
        """Background task: maintain connection with exponential backoff."""
        delay = self.config.reconnect_base_s
        while not self._stop.is_set():
            try:
                if not self.health.connected:
                    await self.connect(interactive_oauth=False)
                    delay = self.config.reconnect_base_s
                    self.health.reconnect_attempts = 0
                else:
                    await self.health_check()
                    if not self.health.connected:
                        await self.disconnect()
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=self.config.health_interval_s
                    )
                except asyncio.TimeoutError:
                    pass
            except Exception as exc:  # noqa: BLE001
                self.health.reconnect_attempts += 1
                self.health.last_error = str(exc)
                self.health.connected = False
                await self.disconnect()
                self.audit.record(
                    "mcp_reconnect_backoff",
                    reason=str(exc),
                    details={"delay_s": delay, "attempt": self.health.reconnect_attempts},
                )
                logger.warning("MCP reconnect in %.1fs: %s", delay, exc)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                delay = min(delay * 2, self.config.reconnect_max_s)

    def stop(self) -> None:
        self._stop.set()

    def status_dict(self) -> dict[str, Any]:
        return {
            "mcp_url": self.config.mcp_url,
            "connected": self.health.connected,
            "authenticated": self.health.authenticated,
            "last_ok_ts": self.health.last_ok_ts,
            "last_error": self.health.last_error,
            "tool_count": self.health.tool_count,
            "tools": self.health.tools,
            "reconnect_attempts": self.health.reconnect_attempts,
            "live_trading_enabled": self.config.live_trading_enabled,
            "has_oauth_tokens": self.storage.has_tokens(),
        }