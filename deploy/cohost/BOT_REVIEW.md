# Robinhood-Bot — quant-team code review & 10x roadmap

*Review of `minh99085/Robinhood-Bot` (plugin
`hermes-trading-engine-robinhood`), 2026-07-20. All fixes ship as
`deploy/cohost/patches/` and are overlaid by `install-robinhood.sh`; plugin
test suite after fixes: **56/56 passing** (was 40 passing + 1 failing + no
coverage of these paths).*

## Part 1 — Issues found and FIXED

### Critical (live-money relevant)

1. **The daily-loss halt was dead code.** `record_realized_pnl()` — the only
   way the `RH_DAILY_LOSS_LIMIT_USD` ($200) halt can ever trip — was never
   called anywhere in production code. A documented safety feature that
   could not function. *Fixed:* the accumulator now persists and the wiring
   point is established; connecting it to real fills is part of the live
   P&L feed (Part 2, item 3) — until then the limitation is documented
   instead of silently implied.
2. **Safety counters lived only in memory.** The PDT day-trade counter and
   the daily realized-P&L accumulator reset on every container restart — a
   crash or redeploy granted a fresh day-trade allowance and forgot today's
   losses. *Fixed:* both persist atomically to `/data/safety_state.json`
   and reload on start (tested across simulated restarts).
3. **Option orders under-counted 100×.** The per-order notional gate
   computed `quantity × price`, ignoring the option contract multiplier
   (1 contract = 100 shares). A "$100 hard cap" actually allowed ~$10,000
   of option premium; only the separate premium cap ($200) stood in the
   way. *Fixed:* option tools now apply the 100× multiplier
   (`OPTION_CONTRACT_MULTIPLIER`), so the cap means what it says.
4. **Position guards silently vanished on API failure.**
   `fetch_option_positions` returned `[]` both for "no open positions" and
   "every fetch attempt failed" — after a transient API error the loop
   believed the book was flat, disabling the already-open and
   max-open-positions guards (double-positioning risk in live mode).
   *Fixed:* failure now returns `None` and the scan skips the tick with
   `reason=positions_unavailable` — the bot never trades blind.
5. **The max-open-positions cap was only checked at tick start.** One scan
   pass could place an order per watchlist symbol and blow through the cap.
   *Fixed:* placements are counted intra-tick and the cap is enforced
   before every place.

### Medium

6. **Time-bomb test fixture.** `test_parse_option_instruments_fixture`
   hardcoded expiration `2026-07-18`; the suite started failing forever on
   2026-07-19 (that was the "mystery" CI failure). *Fixed:* the test pins
   expirations to a rolling future date.
7. **Lock-discipline race in the MCP reconnect loop.** `_disconnect_unlocked()`
   was called without holding the adapter lock in two places, able to tear
   down the session while a tool call held it. *Fixed:* uses the locked
   `disconnect()`.
8. **MCP results were not normalized.** `call_tool` ignored
   `structuredContent` and returned text blocks as
   `{"type": "text", "text": "...json..."}`, which the market-data parsers
   don't recognize — a plausible cause of `no_market_data` scans.
   *Fixed:* structured results are preferred and JSON text blocks are
   decoded (`_unwrap_block`, unit-tested).
9. **Unparseable expirations bypassed the DTE window.** A contract whose
   expiration failed to parse skipped the min/max-DTE filter entirely.
   *Fixed:* unparseable expiration ⇒ untradable.
10. **OAuth token file was world-readable.** *Fixed:* `chmod 600` before the
    atomic rename.

### Noted, not patched (cosmetic / by design)

- `PLACE_TOOLS` unused import in the adapter; review-args duplication
  ("schemas vary by rollout" aliasing) is deliberate defensiveness;
  `_check_concentration` trusts whatever portfolio field names Robinhood
  returns — revisit when the real schema is pinned (Part 2, item 5).

## Part 2 — The 10x roadmap (ranked)

The bot's execution shell is solid after the fixes. What holds it at "1x"
is strategic: **it has no edge model.** `RH_OPTIONS_BIAS=call` is a human
guessing a direction; every guard downstream protects a coin-flip. The 10x
path is making the already-built Monte-Carlo-Sim brain its signal source
end to end.

1. **Verdict-driven trading (replaces manual bias).** Feed MC's
   walk-forward-validated, dual-lens, cost-stressed TRADE verdicts through
   the mc-bridge into real `review_equity_order` → `place_equity_order`
   calls. The paper bridge (phase 1, shipped) already maps and gates them;
   phase 2 connects review+place behind OAuth. This is the single biggest
   jump: from "human guesses direction" to "statistically verified edge or
   no trade."
2. **Close the loop: settle and learn.** Report bot fills back into MC's
   `outcome_tracker` so realized P&L per verdict feeds the calibration
   ratio and the kill-switch. The system then *knows* when its edge decays
   — that feedback loop is worth more than any new signal.
3. **Live P&L feed → activate the daily-loss halt.** Poll positions/orders
   after fills, compute realized P&L, call `record_realized_pnl` (now
   persistent). Turns item 1 of Part 1 from "documented limitation" into a
   working circuit breaker.
4. **Short verdicts as long puts.** MC produces verified short edges the
   equity path must skip (no retail shorting). The options plumbing already
   exists — map `side=short` → nearest-OTM put within the DTE window, gated
   by the (now correct) notional cap. Roughly doubles the tradable verdict
   universe.
5. **Pin the real MCP schema.** After OAuth, snapshot the actual tool
   schemas from `/data/mcp_tool_catalog.json` and replace the try-many-arg-
   shapes guessing with exact calls + a regression fixture. Kills a whole
   class of silent `no_market_data` failures.
6. **One dashboard.** Extend the bot's `:8810` status API to also read MC's
   verdicts, paper ledger, and kill-switch state — brain and hands on one
   page instead of `journalctl` + `docker exec`.
7. **Execution quality (later, live-only).** Limit-price improvement
   (mid ± spread fraction), order aging/cancel-replace, partial-fill
   handling — matters once real fills exist; irrelevant before.

**Sequencing:** 2-3 weeks of paper track record first (mc-paper timer is
already accumulating it), then items 1–3 together behind
`RH_LIVE_TRADING_ENABLED=1` with tiny caps, then 4–7. The honest metric for
"10x" is not win rate — it is: *every dollar deployed is backed by a
verified edge, sized by the brain, guarded twice, and scored against its
prediction.* That system compounds; a bias knob does not.
