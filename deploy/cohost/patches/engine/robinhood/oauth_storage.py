"""Persistent OAuth token storage for VPS — survives container restarts."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from mcp.client.auth import TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

logger = logging.getLogger("hermes.robinhood.oauth")


class FileTokenStorage(TokenStorage):
    """File-backed MCP OAuth storage at ``<data_dir>/robinhood_oauth_tokens.json``."""

    def __init__(self, data_dir: str | Path) -> None:
        self.path = Path(data_dir) / "robinhood_oauth_tokens.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict | None = None

    def _load(self) -> dict:
        if self._cache is not None:
            return self._cache
        if not self.path.exists():
            self._cache = {}
            return self._cache
        try:
            self._cache = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("oauth storage corrupt, starting fresh: %s", exc)
            self._cache = {}
        return self._cache

    def _save(self, payload: dict) -> None:
        self._cache = payload
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.chmod(0o600)  # OAuth tokens: owner-only before it becomes visible
        tmp.replace(self.path)

    async def get_tokens(self) -> OAuthToken | None:
        raw = self._load().get("tokens")
        if not raw:
            return None
        return OAuthToken.model_validate(raw)

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._load()
        data["tokens"] = tokens.model_dump(mode="json")
        self._save(data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        raw = self._load().get("client_info")
        if not raw:
            return None
        return OAuthClientInformationFull.model_validate(raw)

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._load()
        data["client_info"] = client_info.model_dump(mode="json")
        self._save(data)

    def has_tokens(self) -> bool:
        return bool(self._load().get("tokens"))