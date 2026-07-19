# VPS deployment

Production scaffolding for running the TradingView → calibrated-drift
pipeline on a VPS: TLS termination, a supervised webhook bridge, and
scheduled calibration / decision / settlement jobs. None of this changes
the trading logic — it's the operational shell around
`tv_webhook_bridge.py`, `run_weekly_from_tv.py`, `signal_calibration.py`
and `outcome_tracker.py`.

## Architecture

```
TradingView alert ──HTTPS──▶ Caddy (:443, Let's Encrypt TLS)
                               │  reverse_proxy /webhook, /health
                               ▼
                     tv-bridge.service ──▶ tv_data/latest_signal.json
                     (127.0.0.1:5001, systemd)
                               │
   mc-weekly.timer ──▶ run_weekly_from_tv.py ──▶ outputs/verdicts/*.json
   (Mon 09:35 ET)                                outputs/trade_log.jsonl
                               │
   mc-settle.timer ──▶ outcome_tracker.py settle + report  (daily)
   mc-calibrate.timer ──▶ signal_calibration.py <tickers>  (monthly)
```

The bridge binds to loopback only; Caddy is the sole public listener, so the
shared secret and HMAC key never cross the network in the clear.

## One-shot install

On a fresh Debian/Ubuntu VPS as root:

```bash
git clone https://github.com/minh99085/Monte-Carlo-Sim.git
sudo bash Monte-Carlo-Sim/deploy/install.sh
```

`install.sh` creates the `mcsim` service user, a venv, runtime dirs, installs
the units, and enables the timers. It then prints the manual steps below
(they need your secrets and DNS, which can't be automated).

## Manual steps

1. **Secrets + watchlist** — `sudoedit /etc/monte-carlo-sim/mcsim.env`
   (copied from `mcsim.env.example`). Set `TV_BRIDGE_SECRET` (`openssl rand
   -hex 32`) and `MC_TICKERS` to every symbol your alerts can name.
2. **TLS** — point a DNS A/AAAA record at the VPS, set the domain + email in
   `Caddyfile`, install it to `/etc/caddy/Caddyfile`, `systemctl reload
   caddy`.
3. **Seed + start**:
   ```bash
   systemctl start mc-calibrate.service          # build calibration tables
   systemctl enable --now tv-bridge.service
   ```
4. **TradingView alert** — set the webhook URL to
   `https://YOUR_DOMAIN/webhook?secret=YOUR_TV_BRIDGE_SECRET`
   (or send the secret as the `X-Webhook-Secret` header). Message body =
   the JSON the Pine template emits.

## Network policy (important)

`signal_calibration.py` and `run_weekly_from_tv.py` fetch price history from
Yahoo Finance via `yfinance`. The VPS must allow outbound HTTPS to
`*.yahoo.com` / `query*.finance.yahoo.com`. With no market data,
`run_weekly_from_tv.py` **hard-fails and writes no verdict** rather than
inventing prices — that's a safety feature, but it means a locked-down
egress policy silently blocks every decision.

## Operating notes

- **NO_TRADE is the normal case.** The decision layer only fires TRADE when a
  bucket has statistically real edge above breakeven; expect mostly
  NO_TRADE. `mc-weekly.service` treats exit codes 0/2/3 as success so those
  don't register as failures.
- **Kill-switch.** After 20 settled TRADE verdicts with negative mean P&L,
  new TRADEs are refused until you pass `--override-killswitch`. Watch it
  with `outcome_tracker.py report` (also logged daily by `mc-settle`).
- **Calibration staleness.** Tables warn at 30 days, hard-error at 120. The
  monthly timer keeps them fresh; a signal for an uncalibrated ticker gets
  NO_TRADE with the reason recorded.
- **Backups.** `outputs/trade_log.jsonl` is the settlement ledger and
  `calibration/*.json` the fitted edges — back these up; everything else
  regenerates.

## Verify / observe

```bash
systemctl status tv-bridge
curl https://YOUR_DOMAIN/health
journalctl -u tv-bridge -f            # live webhook receipts
journalctl -u mc-weekly -n 50         # last verdict
systemctl list-timers 'mc-*'          # next scheduled runs
```

## Files

| File | Purpose |
|---|---|
| `install.sh` | Idempotent provisioner |
| `Caddyfile` | TLS + reverse proxy (edit domain/email) |
| `mcsim.env.example` | Env template → `/etc/monte-carlo-sim/mcsim.env` |
| `systemd/tv-bridge.service` | Supervised webhook bridge |
| `systemd/mc-weekly.{service,timer}` | Weekly decision |
| `systemd/mc-settle.{service,timer}` | Daily settlement + report |
| `systemd/mc-calibrate.{service,timer}` | Monthly recalibration |

## Alternative: event-driven decisions

The timer model runs the decision on a fixed weekly schedule and consumes
the freshest signal (freshness-guarded). If you'd rather decide the moment a
signal lands, drop a systemd `path` unit watching
`tv_data/latest_signal.json` and have it start `mc-weekly.service` on change.
The timer approach is the default because the tool is a *weekly* decision and
the fixed cadence composes cleanly with settlement/reporting.
