# Design note: Chart vision confidence & secondary channel

## Role in the system

| Channel | Reliability | Role |
|---------|-------------|------|
| TradingView **webhook** JSON | High | **Primary** signal path (`tv_webhook_bridge` → `tv_integration`) |
| Robinhood **MCP** quotes / historicals | High | **Authoritative** price, vol, portfolio context |
| Chart **image** vision | Low–medium | **Secondary / convenience** path only |

Image extraction never replaces the webhook path. It exists so an operator (or Hermes agent) can drop a TradingView screenshot and still obtain a *structured recommendation* that is cross-checked against MCP and stress-tested with tactical Monte Carlo (default **100,000** paths).

## Why image numbers are lower confidence

1. **OCR / model error** — RSI labels, axis ticks, and last price text are easy to misread by a few points or dollars.
2. **Chart configuration** — Scaling, indicators, and overlays vary; the model may invent levels that are visual artifacts.
3. **Staleness** — A screenshot has no guaranteed timestamp; webhooks and MCP are fresher and auditable.
4. **No cryptographic integrity** — Anyone can edit a PNG; webhooks + MCP OAuth provide stronger provenance.

Therefore:

* Image-derived **prices never set hard risk limits** without MCP confirmation.
* Image **stop/TP levels** are soft suggestions, clamped to safe bands and widened when confidence is low.
* Image **RSI/MACD** may influence *conditional drift* only through existing calibration tables (when present) or a **conservative soft drift** scaled by adjusted confidence.
* **Starting price** and **realized volatility** prefer MCP; image price is used only as fallback and is flagged.

## Confidence handling

Each extraction carries per-field scores and an overall score in \([0,1]\).

After vision:

1. Call MCP (`get_equity_quotes`, `get_equity_historicals`, portfolio as available).
2. **Reject** or **down-weight** when:
   * ticker cannot be confirmed,
   * \(|P_{\text{image}} - P_{\text{MCP}}| / P_{\text{MCP}}\) exceeds a configurable threshold (default 2%),
   * overall confidence is below a minimum (default 0.45 reject / 0.60 full weight).
3. Log every discrepancy to the audit trail.

Adjusted confidence multiplies:

* soft drift magnitude,
* position size,
* whether `executable` can ever be true (still requires `gated_execution` mode + safety gates).

## Execution modes (feature flags)

| Mode | Behavior |
|------|----------|
| `log_only` | Extract + validate + MC; **no** trade recommendation sizing for execution |
| `recommendation_only` (default) | Full decision object; **never** auto-places orders |
| `gated_execution` | May mark `executable=true`; agent must still call `SafeRobinhoodClient` + `review_*` |

Live `place_*` is always blocked unless `RH_LIVE_TRADING_ENABLED=1` and all safety gates pass.

## Module boundaries

* **Monte-Carlo-Sim** — schema, mapping, MC run, scoring (`chart_vision_*`).
* **Robinhood-Bot plugin** — vision backends, MCP validation, Hermes tool `analyze_tradingview_chart`, FastAPI, audit.

## Non-goals

* Do not fine-tune VLMs in-repo.
* Do not bypass safety gates or webhook primary path.
* Do not treat vision as a standalone trading signal without MCP when MCP is configured.
