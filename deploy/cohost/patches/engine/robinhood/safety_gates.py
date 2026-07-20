"""Pre-trade safety gates for Robinhood Agentic live orders."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.robinhood.audit_log import AuditLog
from engine.robinhood.config import RobinhoodConfig
from engine.robinhood.constants import ORDER_TOOLS, PLACE_TOOLS, REVIEW_TOOLS

logger = logging.getLogger("hermes.robinhood.safety")

# Standard US equity option contract: 1 contract controls 100 shares, so the
# true dollar exposure of an option order is qty x price x 100.
OPTION_CONTRACT_MULTIPLIER = 100.0

SAFETY_STATE_FILENAME = "safety_state.json"


@dataclass
class SafetyVerdict:
    allowed: bool
    reason: str
    review_required: bool = False
    review_tool: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class DayTradeTracker:
    """Rolling PDT-style day-trade counter (operator-configured limit).

    Persists to ``safety_state.json`` when ``state_path`` is set, so a
    container restart cannot grant a fresh day-trade allowance.
    """

    trades: list[float] = field(default_factory=list)
    state_path: Path | None = None

    def record(self) -> None:
        self.trades.append(time.time())
        self._persist()

    def count_last_5_days(self) -> int:
        cutoff = time.time() - 5 * 86400
        pruned = [t for t in self.trades if t >= cutoff]
        if len(pruned) != len(self.trades):
            self.trades = pruned
            self._persist()
        return len(self.trades)

    def _persist(self) -> None:
        if self.state_path is not None:
            _update_safety_state(self.state_path, day_trades=list(self.trades))


def _load_safety_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _update_safety_state(path: Path, **fields: Any) -> None:
    state = _load_safety_state(path)
    state.update(fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


class RobinhoodSafetyGates:
    """Enforces sizing, loss, concentration, PDT, and review-before-place rules.

    Day-trade history and the daily realized-P&L accumulator persist to
    ``<data_dir>/safety_state.json`` — a restart must never reset the PDT
    counter or forget today's losses.
    """

    def __init__(
        self,
        config: RobinhoodConfig,
        audit: AuditLog | None = None,
    ) -> None:
        self.config = config
        self.audit = audit or AuditLog(config.data_dir)
        self._state_path = Path(config.data_dir) / SAFETY_STATE_FILENAME
        state = _load_safety_state(self._state_path)
        self.day_trades = DayTradeTracker(
            trades=[float(t) for t in (state.get("day_trades") or [])],
            state_path=self._state_path,
        )
        pnl = state.get("daily_pnl") or {}
        self._daily_pnl_day: str | None = pnl.get("day")
        self._daily_realized_pnl_usd = float(pnl.get("usd") or 0.0)

    def _today_key(self) -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def record_realized_pnl(self, pnl_usd: float) -> None:
        today = self._today_key()
        if self._daily_pnl_day != today:
            self._daily_pnl_day = today
            self._daily_realized_pnl_usd = 0.0
        self._daily_realized_pnl_usd += pnl_usd
        _update_safety_state(
            self._state_path,
            daily_pnl={"day": self._daily_pnl_day,
                       "usd": self._daily_realized_pnl_usd},
        )

    @staticmethod
    def _order_notional(args: dict[str, Any], *, contract_multiplier: float = 1.0) -> float:
        """Dollar exposure of an order. Explicit notional keys are taken
        as-is; qty x price is scaled by ``contract_multiplier`` (100 for
        option contracts — without it an option order's true exposure is
        under-counted 100x and the per-order cap is meaningless)."""
        for key in ("notional", "amount", "dollar_amount", "order_amount"):
            if key in args:
                try:
                    return float(args[key])
                except (TypeError, ValueError):
                    pass
        qty = args.get("quantity") or args.get("qty")
        price = args.get("price") or args.get("limit_price")
        if qty is not None and price is not None:
            try:
                return abs(float(qty) * float(price)) * float(contract_multiplier)
            except (TypeError, ValueError):
                pass
        return 0.0

    @staticmethod
    def _symbol(args: dict[str, Any]) -> str:
        for key in ("symbol", "instrument", "ticker"):
            val = args.get(key)
            if val:
                return str(val).upper()
        return ""

    def evaluate(
        self,
        tool: str,
        arguments: dict[str, Any] | None = None,
        *,
        portfolio: dict[str, Any] | None = None,
    ) -> SafetyVerdict:
        """Check whether a tool call may proceed (does not call Robinhood review APIs)."""
        args = dict(arguments or {})
        tool = tool.strip()

        if tool not in ORDER_TOOLS and tool not in PLACE_TOOLS:
            return SafetyVerdict(True, "non_order_tool")

        if not self.config.live_trading_enabled and tool in PLACE_TOOLS:
            verdict = SafetyVerdict(False, "live_trading_disabled")
            self.audit.record("safety_block", tool=tool, allowed=False, reason=verdict.reason)
            return verdict

        if self.config.approval_mode == "disabled" and tool in PLACE_TOOLS:
            verdict = SafetyVerdict(False, "approval_mode_disabled")
            self.audit.record("safety_block", tool=tool, allowed=False, reason=verdict.reason)
            return verdict

        notional = self._order_notional(
            args,
            contract_multiplier=(OPTION_CONTRACT_MULTIPLIER
                                 if "option" in tool else 1.0),
        )
        if notional > self.config.max_order_notional_usd:
            verdict = SafetyVerdict(
                False,
                f"notional ${notional:.2f} exceeds max ${self.config.max_order_notional_usd:.2f}",
            )
            self.audit.record("safety_block", tool=tool, allowed=False, reason=verdict.reason)
            return verdict

        if self._daily_realized_pnl_usd <= -abs(self.config.daily_loss_limit_usd):
            verdict = SafetyVerdict(False, "daily_loss_limit_reached")
            self.audit.record("safety_block", tool=tool, allowed=False, reason=verdict.reason)
            return verdict

        if tool in PLACE_TOOLS:
            dt_count = self.day_trades.count_last_5_days()
            if dt_count >= self.config.max_day_trades_5d:
                verdict = SafetyVerdict(False, f"pdt_limit_reached ({dt_count} day trades / 5d)")
                self.audit.record("safety_block", tool=tool, allowed=False, reason=verdict.reason)
                return verdict

        if tool in PLACE_TOOLS and "option" in tool:
            verdict = self._check_option_order(args, notional)
            if not verdict.allowed:
                self.audit.record("safety_block", tool=tool, allowed=False, reason=verdict.reason)
                return verdict

        if portfolio and tool in PLACE_TOOLS:
            verdict = self._check_concentration(args, portfolio, notional)
            if not verdict.allowed:
                self.audit.record("safety_block", tool=tool, allowed=False, reason=verdict.reason)
                return verdict

        if tool in PLACE_TOOLS and notional >= self.config.review_threshold_notional_usd:
            review_tool = (
                "review_option_order" if "option" in tool else "review_equity_order"
            )
            verdict = SafetyVerdict(
                True,
                "review_required_before_place",
                review_required=True,
                review_tool=review_tool,
            )
            self.audit.record("safety_review_required", tool=tool, allowed=True, reason=verdict.reason)
            return verdict

        verdict = SafetyVerdict(True, "ok")
        self.audit.record("safety_pass", tool=tool, allowed=True, reason=verdict.reason)
        return verdict

    def _check_concentration(
        self,
        args: dict[str, Any],
        portfolio: dict[str, Any],
        notional: float,
    ) -> SafetyVerdict:
        total = float(portfolio.get("total_value") or portfolio.get("equity") or 0)
        buying_power = float(portfolio.get("buying_power") or 0)
        if buying_power < self.config.min_buying_power_buffer_usd:
            return SafetyVerdict(False, "insufficient_buying_power_buffer")

        symbol = self._symbol(args)
        if total > 0 and notional > 0:
            new_pct = (notional / total) * 100
            if new_pct > self.config.max_position_pct:
                return SafetyVerdict(
                    False,
                    f"order {new_pct:.1f}% of portfolio exceeds max {self.config.max_position_pct:.1f}%",
                )
            if symbol:
                positions = portfolio.get("positions") or portfolio.get("equity_positions") or []
                sym_value = 0.0
                for pos in positions:
                    psym = str(pos.get("symbol") or pos.get("ticker") or "").upper()
                    if psym == symbol:
                        sym_value += float(pos.get("market_value") or pos.get("value") or 0)
                conc = ((sym_value + notional) / total) * 100
                if conc > self.config.max_symbol_concentration_pct:
                    return SafetyVerdict(
                        False,
                        f"{symbol} concentration {conc:.1f}% exceeds max "
                        f"{self.config.max_symbol_concentration_pct:.1f}%",
                    )
        return SafetyVerdict(True, "concentration_ok")

    def _check_option_order(self, args: dict[str, Any], notional: float) -> SafetyVerdict:
        symbol = self._symbol(args)
        if symbol and self.config.options_watchlist and symbol not in self.config.options_watchlist:
            return SafetyVerdict(False, f"{symbol} not in options watchlist")

        side = str(args.get("side") or "buy").lower()
        if self.config.options_long_only and side not in ("buy", "debit", "long"):
            return SafetyVerdict(False, f"long_only: side {side!r} blocked")

        qty_raw = args.get("quantity") or args.get("qty") or 1
        try:
            qty = int(qty_raw)
        except (TypeError, ValueError):
            qty = 1
        if qty > self.config.options_max_contracts:
            return SafetyVerdict(
                False,
                f"quantity {qty} exceeds max contracts {self.config.options_max_contracts}",
            )

        price = args.get("limit_price") or args.get("price")
        premium = 0.0
        if price is not None:
            try:
                premium = float(price) * qty * 100.0
            except (TypeError, ValueError):
                premium = 0.0
        elif notional > 0:
            premium = notional
        if premium > self.config.options_max_premium_usd:
            return SafetyVerdict(
                False,
                f"premium ${premium:.2f} exceeds max ${self.config.options_max_premium_usd:.2f}",
            )

        return SafetyVerdict(True, "option_checks_ok")

    async def enforce_review(
        self,
        adapter: Any,
        tool: str,
        arguments: dict[str, Any],
        verdict: SafetyVerdict,
    ) -> SafetyVerdict:
        """Call Robinhood review_* tool before place; block on hard warnings."""
        if not verdict.review_required or not verdict.review_tool:
            return verdict
        review_args = dict(arguments)
        review_args.setdefault("dry_run", True)
        try:
            review_result = await adapter.call_tool(verdict.review_tool, review_args)
        except Exception as exc:  # noqa: BLE001
            blocked = SafetyVerdict(False, f"review_call_failed: {exc}")
            self.audit.record("safety_block", tool=tool, allowed=False, reason=blocked.reason)
            return blocked

        warnings = _extract_warnings(review_result)
        verdict.warnings = warnings
        if _has_blocking_warnings(warnings):
            blocked = SafetyVerdict(False, "review_blocked", warnings=warnings)
            self.audit.record(
                "safety_block",
                tool=tool,
                allowed=False,
                reason=blocked.reason,
                details={"warnings": warnings},
            )
            return blocked

        self.audit.record(
            "safety_review_pass",
            tool=tool,
            allowed=True,
            details={"warnings": warnings},
        )
        return verdict


def _extract_warnings(review_result: Any) -> list[str]:
    if isinstance(review_result, dict):
        for key in ("warnings", "alerts", "issues", "messages"):
            val = review_result.get(key)
            if isinstance(val, list):
                return [str(x) for x in val]
        if review_result.get("blocked") or review_result.get("can_place") is False:
            return [str(review_result)]
    if isinstance(review_result, list):
        return [str(x) for x in review_result]
    return []


def _has_blocking_warnings(warnings: list[str]) -> bool:
    if not warnings:
        return False
    block_tokens = ("error", "reject", "blocked", "not tradable", "insufficient", "pattern day")
    joined = " ".join(warnings).lower()
    return any(tok in joined for tok in block_tokens)