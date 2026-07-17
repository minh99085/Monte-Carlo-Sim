# Phase 3 — TradingView → Monte Carlo webhook bridge

This phase adds a **simple, local webhook bridge** so trend and momentum
signals from **TradingView** can be captured and stored for the Monte Carlo
tactical simulator.

It does **not** place trades. It only **receives and saves** alert data.

---

## How the bridge works

```
TradingView chart (Pine script)
        │  alert fires with JSON message
        ▼
TradingView servers  ──POST──►  your public URL /webhook?secret=...
                                        │
                                        ▼
                          tv_webhook_bridge.py  (local Python server)
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
         tv_data/latest_signal.json            tv_data/signal_history.jsonl
         (overwrite each time)                 (append one line per alert)
```

1. A Pine Script on TradingView computes **trend** (EMA cross) and **momentum** (RSI).
2. When an alert fires, TradingView POSTs the alert message to your webhook URL.
3. `tv_webhook_bridge.py` checks the **secret key**, parses the JSON, and saves it.
4. Later tools (or you) can read `tv_data/latest_signal.json` as input context
   for a short-horizon tactical Monte Carlo run.

---

## Files added in Phase 3

| File | Purpose |
| --- | --- |
| `tv_webhook_bridge.py` | Local HTTP server (stdlib only) |
| `tradingview_alert_template.pine` | Ready-to-paste Pine Script |
| `PHASE3_README.md` | This guide |
| `tv_data/` | Created at runtime (signals; gitignored) |

---

## What data is sent and received

### From TradingView (JSON alert message)

Example body (what the Pine template builds):

```json
{
  "ticker": "AAPL",
  "price": 198.42,
  "trend": "bullish",
  "momentum": 61.35,
  "timeframe": "5",
  "strategy": "mc_bridge_ema_rsi"
}
```

| Field | Meaning |
| --- | --- |
| `ticker` | Symbol, e.g. `AAPL` |
| `price` | Bar close when the alert fired |
| `trend` | `bullish` or `bearish` (EMA fast vs slow) |
| `momentum` | RSI value (0–100) |
| `timeframe` | Chart timeframe code (e.g. `5`, `60`, `D`) |
| `strategy` | Label for which script sent it |

### What the bridge stores

Each accepted webhook becomes a richer record:

```json
{
  "received_at_utc": "2026-07-17T12:34:56+00:00",
  "source": "tradingview",
  "source_ip": "52.x.x.x",
  "ticker": "AAPL",
  "price": 198.42,
  "trend": "bullish",
  "momentum": 61.35,
  "timeframe": "5",
  "strategy": "mc_bridge_ema_rsi",
  "parse_status": "json",
  "raw": { "...original payload..." }
}
```

- **Latest only:** `tv_data/latest_signal.json`  
- **Full history:** `tv_data/signal_history.jsonl` (one JSON object per line)

---

## Step-by-step setup

### A. Run the local bridge

```powershell
cd Monte-Carlo-Sim

# Pick a long random secret (do not commit it)
python tv_webhook_bridge.py --secret "replace-with-a-long-random-string" --port 5001
```

You should see:

```text
TradingView webhook bridge listening on http://0.0.0.0:5001
  POST webhook : http://0.0.0.0:5001/webhook?secret=<your-secret>
  GET  health  : http://0.0.0.0:5001/health
  GET  latest  : http://0.0.0.0:5001/latest?secret=<your-secret>
```

Environment variables (optional):

| Variable | Meaning |
| --- | --- |
| `TV_BRIDGE_SECRET` | Shared secret |
| `TV_BRIDGE_PORT` | Port (default `5001`) |
| `TV_BRIDGE_HOST` | Bind host (default `0.0.0.0`) |
| `TV_BRIDGE_DATA_DIR` | Output folder (default `tv_data`) |

### B. Make the bridge reachable from the internet

TradingView’s servers cannot call `localhost` on your PC. Use one of:

1. **ngrok** (quick test):
   ```powershell
   ngrok http 5001
   ```
   Copy the `https://....ngrok.io` URL.

2. **Cloudflare Tunnel**, **localtunnel**, or a small **VPS** with the port open.

Your public webhook URL will look like:

```text
https://YOUR_PUBLIC_HOST/webhook?secret=replace-with-a-long-random-string
```

### C. Install the Pine Script on TradingView

1. Open [TradingView](https://www.tradingview.com) → chart for your symbol.  
2. **Pine Editor** → new indicator.  
3. Paste the contents of `tradingview_alert_template.pine`.  
4. **Save** → **Add to chart**.  
5. You should see EMAs, a tinted background, and a small status table.

### D. Create the alert + webhook

1. Click **Alert** on the chart.  
2. Condition: this script’s alert (`Any alert() function call` or the named flip conditions).  
3. Options: e.g. **Once Per Bar Close** (matches the script).  
4. **Notifications** → enable **Webhook URL**.  
5. Paste:
   ```text
   https://YOUR_PUBLIC_HOST/webhook?secret=replace-with-a-long-random-string
   ```
6. Message: if the UI asks for a message and you use `alert()`, the script already
   supplies JSON. For `alertcondition` rows, the message template is already JSON.  
7. Save the alert.

### E. Confirm data arrives

- Watch the bridge terminal for `Saved signal #…`  
- Or open:
  ```powershell
  Get-Content .\tv_data\latest_signal.json
  ```
- Or:
  ```powershell
  curl "http://127.0.0.1:5001/latest?secret=replace-with-a-long-random-string"
  ```

---

## How to test the bridge (without TradingView)

With the server running in one terminal:

```powershell
python tv_webhook_bridge.py --secret testsecret --port 5001
```

In another terminal:

```powershell
# Health check (no secret required)
curl http://127.0.0.1:5001/health

# Fake TradingView webhook
curl -X POST "http://127.0.0.1:5001/webhook?secret=testsecret" `
  -H "Content-Type: text/plain" `
  -d '{"ticker":"AAPL","price":190.5,"trend":"bullish","momentum":58.2,"timeframe":"5","strategy":"manual_test"}'

# Read latest (secret required)
curl "http://127.0.0.1:5001/latest?secret=testsecret"
```

**Wrong secret** should return `401 unauthorized` and **not** write a file.

**Windows PowerShell** alternative to curl:

```powershell
Invoke-RestMethod -Method POST `
  -Uri "http://127.0.0.1:5001/webhook?secret=testsecret" `
  -ContentType "application/json" `
  -Body '{"ticker":"MSFT","price":420.1,"trend":"bearish","momentum":41.0,"timeframe":"15","strategy":"manual_test"}'
```

---

## Security (simple but important)

| Practice | Why |
| --- | --- |
| Long random `--secret` | Stops strangers from POSTing fake signals |
| Put secret in **URL query** or header, not in public Pine | Avoid leaking keys if the script is shared |
| HTTPS public URL (ngrok / tunnel / VPS TLS) | Protects the secret in transit |
| Do not commit `tv_data/` | May contain operational signals |

Accepted secret locations:

1. Query: `?secret=...`  
2. Header: `X-Webhook-Secret: ...`  
3. Header: `Authorization: Bearer ...`  
4. JSON field: `"secret": "..."` (stripped before save)

---

## Reading signals in the tactical simulator (integration)

The tactical runner can load the bridge file automatically:

```python
from tactical_config import preset_5_day, TradingRule
from tactical_simulator import run_tactical_simulation

cfg = preset_5_day("AAPL", paths=10_000, annual_volatility=0.25).with_rule(
    TradingRule(
        name="TV-aware",
        entry_condition="Enter with TV trend",
        exit_condition="stop/tp/hold",
        stop_loss_pct=0.02,
        take_profit_pct=0.03,
        max_holding_days=5,
    )
)
result = run_tactical_simulation(cfg, use_tradingview=True)
print(result.summary_text())  # shows Used TradingView: YES/NO
```

Or from the command line:

```powershell
python run_tactical_with_tv.py --demo          # offline fake signal
python run_tactical_with_tv.py                 # real tv_data/latest_signal.json
python monte_carlo_gbm.py AAPL --tactical --tactical-tv --paths 10000 --no-chart
```

See the main **README** section *How to use the full tactical system with TradingView*.

---

## Endpoints summary

| Method | Path | Secret? | Purpose |
| --- | --- | --- | --- |
| `GET` | `/` or `/health` | No | Liveness + last ticker |
| `POST` | `/webhook` | **Yes** | Receive TradingView alert |
| `GET` | `/latest` | **Yes** | Read last saved signal |

---

## Troubleshooting

| Symptom | What to try |
| --- | --- |
| Alert fires but nothing saved | Check public URL, port forward/tunnel, and secret match |
| `401 unauthorized` | Same secret in bridge CLI and webhook URL |
| `parse_status: text` | Alert message is not JSON — use the provided Pine template |
| TradingView “webhook failed” | Your tunnel/VPS must accept HTTPS POSTs from the internet |
| No `tv_data` folder | It is created on the first successful save |

---

## Phase 3 scope / non-goals

**In scope:** receive → authenticate → parse → save JSON.  
**Not in scope (later):** auto-run Monte Carlo on every alert, broker execution,
cloud multi-user auth, Plotly dashboards of TV feeds.

---

## Disclaimer

Research plumbing only. Alert data can be late, wrong, or missing. This bridge
does not trade and is not investment advice.
