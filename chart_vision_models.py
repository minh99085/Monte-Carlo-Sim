"""
Pydantic models for TradingView chart vision extraction and trade decisions.

Image-derived fields are intentionally treated as *lower confidence* than
structured TradingView webhooks or Robinhood MCP market data. See
``DESIGN_CHART_VISION.md``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Bias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    UNCLEAR = "unclear"


class LevelKind(str, Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    PIVOT = "pivot"
    OTHER = "other"


class Action(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class ValidationStatus(str, Enum):
    PASSED = "passed"
    DOWNWEIGHTED = "downweighted"
    REJECTED = "rejected"
    SKIPPED = "skipped"  # e.g. MCP unavailable, log-only path


class PriceLevel(BaseModel):
    """A support/resistance (or related) price level detected on the chart."""

    price: float = Field(..., gt=0, description="Price level in quote currency")
    kind: LevelKind = LevelKind.OTHER
    strength: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Relative strength of the level (0–1), image-derived",
    )
    label: Optional[str] = None

    @field_validator("price", mode="before")
    @classmethod
    def _coerce_price(cls, v: Any) -> float:
        return float(v)


class RSIState(BaseModel):
    value: Optional[float] = Field(None, ge=0.0, le=100.0)
    zone: Optional[str] = Field(
        None,
        description="Qualitative zone: oversold | neutral | overbought | unclear",
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class MACDState(BaseModel):
    macd_line: Optional[float] = None
    signal_line: Optional[float] = None
    histogram: Optional[float] = None
    cross: Optional[str] = Field(
        None,
        description="bullish_cross | bearish_cross | none | unclear",
    )
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class IndicatorBundle(BaseModel):
    """Minimum required: RSI and MACD; other indicators may appear as extras."""

    rsi: RSIState = Field(default_factory=RSIState)
    macd: MACDState = Field(default_factory=MACDState)
    extras: Dict[str, Any] = Field(default_factory=dict)


class FieldConfidence(BaseModel):
    ticker: float = Field(0.0, ge=0.0, le=1.0)
    timeframe: float = Field(0.0, ge=0.0, le=1.0)
    indicators: float = Field(0.0, ge=0.0, le=1.0)
    levels: float = Field(0.0, ge=0.0, le=1.0)
    bias: float = Field(0.0, ge=0.0, le=1.0)
    price: float = Field(0.0, ge=0.0, le=1.0)
    overall: float = Field(0.0, ge=0.0, le=1.0)


class ChartExtractionResult(BaseModel):
    """
    Strict schema for vision output from a TradingView (or similar) chart image.

    Numerical fields from images must never set hard risk limits without MCP
    cross-validation.
    """

    ticker: str = Field(..., min_length=1, max_length=16)
    timeframe: str = Field(..., min_length=1, description="e.g. 5m, 1H, 1D, 1W")
    indicators: IndicatorBundle = Field(default_factory=IndicatorBundle)
    levels: List[PriceLevel] = Field(default_factory=list)
    bias: Bias = Bias.UNCLEAR
    confidence: FieldConfidence = Field(default_factory=FieldConfidence)
    raw_model_description: str = ""
    extraction_warnings: List[str] = Field(default_factory=list)

    # Optional soft price read from the chart (never authoritative)
    image_last_price: Optional[float] = Field(None, gt=0)
    provider: Optional[str] = None
    model: Optional[str] = None

    @field_validator("ticker", mode="before")
    @classmethod
    def _normalize_ticker(cls, v: Any) -> str:
        s = str(v or "").strip().upper()
        # Strip common exchange prefixes shown on TV (NASDAQ:AAPL → AAPL)
        if ":" in s:
            s = s.split(":")[-1]
        if not s:
            raise ValueError("ticker must be non-empty")
        return s

    @field_validator("timeframe", mode="before")
    @classmethod
    def _normalize_tf(cls, v: Any) -> str:
        s = str(v or "").strip()
        if not s:
            raise ValueError("timeframe must be non-empty")
        return s

    @field_validator("bias", mode="before")
    @classmethod
    def _normalize_bias(cls, v: Any) -> str:
        if v is None or v == "":
            return Bias.UNCLEAR.value
        if isinstance(v, Bias):
            return v.value
        s = str(v).strip().lower()
        # Enum stringification sometimes yields "Bias.BULLISH"
        if s.startswith("bias."):
            s = s.split(".", 1)[1]
        aliases = {
            "bull": "bullish",
            "long": "bullish",
            "buy": "bullish",
            "up": "bullish",
            "bear": "bearish",
            "short": "bearish",
            "sell": "bearish",
            "down": "bearish",
            "sideways": "neutral",
            "range": "neutral",
            "unknown": "unclear",
            "n/a": "unclear",
        }
        s = aliases.get(s, s)
        if s not in {b.value for b in Bias}:
            return Bias.UNCLEAR.value
        return s

    @model_validator(mode="after")
    def _default_overall_confidence(self) -> "ChartExtractionResult":
        c = self.confidence
        if c.overall <= 0:
            parts = [c.ticker, c.timeframe, c.indicators, c.levels, c.bias, c.price]
            known = [p for p in parts if p > 0]
            if known:
                c.overall = float(sum(known) / len(known))
        return self

    def to_tv_signal_dict(self) -> Dict[str, Any]:
        """
        Map extraction into the same shape used by ``tv_integration`` /
        TradingView webhook bridge so MC can reuse existing wiring.
        """
        rsi_val = self.indicators.rsi.value
        return {
            "source": "chart_vision",
            "ticker": self.ticker,
            "symbol": self.ticker,
            "price": self.image_last_price,
            "close": self.image_last_price,
            "trend": self.bias.value,
            "direction": self.bias.value,
            "momentum": rsi_val,
            "rsi": rsi_val,
            "timeframe": self.timeframe,
            "tf": self.timeframe,
            "strategy": "chart_vision",
            "parse_status": "vision",
            "confidence_overall": self.confidence.overall,
            "levels": [lv.model_dump(mode="json") for lv in self.levels],
            "macd": self.indicators.macd.model_dump(mode="json"),
            "extraction_warnings": list(self.extraction_warnings),
        }


class MCPMarketSnapshot(BaseModel):
    """Authoritative market context from Robinhood MCP (or a mock)."""

    ticker: str
    last_price: Optional[float] = Field(None, gt=0)
    bid: Optional[float] = None
    ask: Optional[float] = None
    previous_close: Optional[float] = None
    realized_vol_annual: Optional[float] = Field(
        None, ge=0.0, description="Annualized realized vol from historicals"
    )
    portfolio_equity: Optional[float] = None
    buying_power: Optional[float] = None
    existing_position_qty: Optional[float] = None
    raw_quotes: Optional[Dict[str, Any]] = None
    raw_historicals: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)


class ValidationDiscrepancy(BaseModel):
    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"
    image_value: Optional[Any] = None
    mcp_value: Optional[Any] = None


class ValidationResult(BaseModel):
    status: ValidationStatus
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0)
    adjusted_confidence: float = Field(0.0, ge=0.0, le=1.0)
    discrepancies: List[ValidationDiscrepancy] = Field(default_factory=list)
    price_rel_error: Optional[float] = None
    ticker_confirmed: bool = False
    notes: List[str] = Field(default_factory=list)

    @property
    def accepted_for_recommendation(self) -> bool:
        return self.status in (
            ValidationStatus.PASSED,
            ValidationStatus.DOWNWEIGHTED,
            ValidationStatus.SKIPPED,
        )


class RiskSummary(BaseModel):
    """Monte Carlo risk metrics for the decision payload."""

    n_paths: int = 0
    prob_profit: Optional[float] = None
    prob_loss: Optional[float] = None
    prob_flat: Optional[float] = None
    avg_pnl_pct: Optional[float] = None
    median_pnl_pct: Optional[float] = None
    worst_pnl_pct: Optional[float] = None
    best_pnl_pct: Optional[float] = None
    var_95_pct: Optional[float] = None  # 5th percentile of pnl_pct (loss convention)
    var_99_pct: Optional[float] = None
    es_95_pct: Optional[float] = None
    max_drawdown_p50: Optional[float] = None
    max_drawdown_p95: Optional[float] = None
    ruin_probability: Optional[float] = None  # P(hit stop) as proxy if no equity ruin model
    stop_hit_rate: Optional[float] = None
    take_profit_rate: Optional[float] = None
    pnl_p05: Optional[float] = None
    pnl_p95: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class PositionSizeSuggestion(BaseModel):
    shares: float = 0.0
    notional_usd: float = 0.0
    risk_budget_usd: float = 0.0
    method: str = "none"
    capped_by: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ChartTradeDecision(BaseModel):
    """
    Full pipeline decision object (vision + validation + MC + sizing).

    This is a *recommendation* by default. Execution still requires
    SafeRobinhoodClient gates and review_* tools.
    """

    ticker: str
    action: Action = Action.FLAT
    suggested_side: Optional[Literal["long", "short"]] = None
    position_size: PositionSizeSuggestion = Field(default_factory=PositionSizeSuggestion)
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None

    risk: RiskSummary = Field(default_factory=RiskSummary)
    vision: Optional[ChartExtractionResult] = None
    validation: Optional[ValidationResult] = None
    mcp: Optional[MCPMarketSnapshot] = None

    mc_paths: int = 100_000
    mc_horizon_days: int = 5
    mc_seed: Optional[int] = None
    starting_price: Optional[float] = None
    annual_volatility: Optional[float] = None
    annual_drift: Optional[float] = None

    vision_confidence: float = 0.0
    validation_status: ValidationStatus = ValidationStatus.SKIPPED
    execution_mode: Literal["log_only", "recommendation_only", "gated_execution"] = (
        "recommendation_only"
    )
    executable: bool = False  # True only if mode allows and validation passed hard gates

    rationale: str = ""
    notes: List[str] = Field(default_factory=list)
    audit_id: Optional[str] = None

    def summary_text(self) -> str:
        lines = [
            "=== Chart Vision Trade Decision ===",
            f"Ticker:              {self.ticker}",
            f"Action:              {self.action.value}",
            f"Validation:          {self.validation_status.value}",
            f"Vision confidence:   {self.vision_confidence:.2f}",
            f"Execution mode:      {self.execution_mode}",
            f"Executable:          {self.executable}",
            f"Starting price:      {self.starting_price}",
            f"Suggested notional:  ${self.position_size.notional_usd:.2f}",
            f"Stop / TP (pct):     {self.stop_loss_pct} / {self.take_profit_pct}",
            f"MC paths:            {self.mc_paths:,}",
            f"P(profit):           {self.risk.prob_profit}",
            f"VaR 95 (pnl%):       {self.risk.var_95_pct}",
            f"ES 95 (pnl%):        {self.risk.es_95_pct}",
            f"Ruin/stop proxy:     {self.risk.ruin_probability}",
            "",
            "Rationale:",
            self.rationale or "(none)",
        ]
        if self.notes:
            lines.append("")
            lines.append("Notes:")
            lines.extend(f"  * {n}" for n in self.notes)
        return "\n".join(lines)
