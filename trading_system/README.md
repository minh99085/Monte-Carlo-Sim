# Meta-labeled trading system (v2)

The rebuild: **brain** = a trained model that judges each TradingView signal
(trade/no-trade + size); **hand** = the existing Robinhood bot executor;
**Monte Carlo = risk overlay only** (drawdown distribution + kill levels —
never direction).

Old design's flaw, plainly: GBM Monte Carlo has no directional predictive
power, so v1 was trading raw TradingView alerts minus costs. v2 instead
*learns from history which alerts were worth taking* (López de Prado
meta-labeling) and refuses the rest. **A correct refusal to trade is the
system working, not failing.**

## Layout

| File | Role |
|---|---|
| `config.yaml` | Every knob. `DRY_RUN: true` is the master safety. |
| `data.py` | OHLC adapter (yfinance now; Robinhood MCP slot-in later). |
| `primary.py` | The primary signal stream: EMA 9/21 cross flips, exactly reconstructible from history + live TV archive merge. |
| `barriers.py` | Triple-barrier labeler (profit/stop/vertical, next-open entry, net of costs). |
| `features.py` | As-of-signal features. Lookahead is impossible by test. |
| `model.py` | Purged walk-forward logistic + isotonic calibration. |
| `sizing.py` | A4 (threshold) and B1 (de Prado bet size) multipliers on fixed-% risk. |
| `gauntlet.py` | The six go/no-go gates. |
| `decision.py` | Live path: one signal → verdict JSON (same schema the bot's bridge already consumes). |

## Run the gauntlet (on the VPS — needs market data)

```
cd /opt/monte-carlo-sim
.venv/bin/python -m trading_system.gauntlet            # gates 1–5
.venv/bin/python -m trading_system.gauntlet --holdout  # gate 6, EXACTLY ONCE
```

Report: `outputs/gauntlet_report.json`. Trial count accumulates in
`outputs/trials.json` — every configuration ever tried deflates the Sharpe
bar. The holdout can only be evaluated once; a marker file enforces it.

## Decide on the latest TradingView signal (paper)

```
.venv/bin/python -m trading_system.decision \
    --signal-file tv_data/latest_signal.json --verdict-dir outputs/verdicts
```

The verdict file feeds the existing `mc_bridge` → Robinhood-bot safety
gates → paper ledger, unchanged.

## Going live — the exact two-step procedure

1. The gauntlet report must show **all six gates PASS** (including the
   one-shot holdout). If any gate fails: fix nothing, tune nothing — the
   system is telling you the signal has no edge.
2. Only then: set `DRY_RUN: false` in `config.yaml` **and**
   `RH_LIVE_TRADING_ENABLED=1` in the bot's `.env`, and confirm the risk
   caps in both configs match `RISK_BUDGET.md`.

Until both steps happen, nothing can place an order anywhere.

## Honest status

Built and ground-truth tested (synthetic worlds: planted edges are found,
no-edge worlds are refused, holdout runs once, lookahead is impossible).
**Not yet run on real market data** — the sandbox has no market-data
access; the gauntlet's first real run happens on the VPS. Expect the
insufficient-history stop or failed gates as likely first outcomes; that is
the design protecting capital, not a bug.
