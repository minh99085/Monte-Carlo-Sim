# VPS deployment

Production scaffolding for running the TradingView → calibrated-drift
pipeline on a VPS: a supervised webhook bridge and scheduled calibration /
decision / settlement jobs. None of this changes the trading logic — it's
the operational shell around `tv_webhook_bridge.py`,
`run_weekly_from_tv.py`, `signal_calibration.py` and `outcome_tracker.py`.

## Two supported setups

1. **Default (simple)** — the bridge listens directly on the server's
   public IP: `http://YOUR_IP:5001/webhook?secret=...`. No domain, no
   certificate. This is the same pattern many TradingView bot setups use.
   The tradeoff: traffic (including the `?secret=...` in the URL) is
   unencrypted.
2. **Optional (HTTPS)** — Caddy terminates TLS in front of the bridge on a
   real domain: `https://your-domain.com/webhook?secret=...`. More setup,
   encrypted traffic.

Both use the exact same Python code and systemd units; only the network
front door differs. Start with the default — you can add HTTPS later
without reinstalling anything.

## Architecture (default setup)

```
TradingView alert ──HTTP──▶ tv-bridge.service (0.0.0.0:5001, systemd)
                                      │
                                      ▼
                          tv_data/latest_signal.json
                                      │
   mc-weekly.timer ──▶ run_weekly_from_tv.py ──▶ outputs/verdicts/*.json
   (Mon 09:35 ET)                                outputs/trade_log.jsonl
                                      │
   mc-settle.timer ──▶ outcome_tracker.py settle + report  (daily)
   mc-calibrate.timer ──▶ signal_calibration.py <tickers>  (monthly)
```

## One-shot install

On a fresh Debian/Ubuntu VPS, connect over SSH as root and run:

```bash
git clone https://github.com/minh99085/Monte-Carlo-Sim.git
sudo bash Monte-Carlo-Sim/deploy/install.sh
```

`install.sh` creates the `mcsim` service user, a Python environment, runtime
directories, installs the systemd units, opens the firewall for SSH + port
5001, and enables the scheduled timers. It prints the remaining manual
steps (they need your secret and watchlist, which can't be automated) —
summarized below.

## Manual steps

1. **Secrets + watchlist** — edit `/etc/monte-carlo-sim/mcsim.env` (created
   from `mcsim.env.example`). Set `TV_BRIDGE_SECRET` (generate one with
   `openssl rand -hex 32`) and `MC_TICKERS` to every symbol your alerts can
   name.
2. **Seed + start**:
   ```bash
   systemctl start mc-calibrate.service          # build calibration tables
   systemctl enable --now tv-bridge.service
   systemctl enable --now mc-weekly.timer mc-settle.timer mc-calibrate.timer
   ```
3. **TradingView alert** — set the webhook URL to
   `http://YOUR_SERVER_IP:5001/webhook?secret=YOUR_TV_BRIDGE_SECRET`
   (install.sh prints your server's IP at the end). The alert message body
   should be the JSON the Pine template emits.

## Optional: add HTTPS later

If you later want an encrypted `https://` address (e.g. a domain-based
TradingView setup, or just tighter security), install Caddy as a reverse
proxy in front of the bridge:

```bash
apt-get install -y caddy
```

Then edit `deploy/Caddyfile` (set your real domain + email), install it:
```bash
cp deploy/Caddyfile /etc/caddy/Caddyfile
systemctl reload caddy
```

Finally, change `/etc/monte-carlo-sim/mcsim.env`:
```
TV_BRIDGE_HOST=127.0.0.1
```
and restart the bridge (`systemctl restart tv-bridge`) so it only accepts
connections from Caddy, not the whole internet. Your TradingView webhook
URL becomes `https://your-domain.com/webhook?secret=...`.

This requires owning a domain name and pointing its DNS A record at the
VPS's IP first — see any domain registrar (Namecheap, Porkbun, etc.).

## Network policy (important either way)

`signal_calibration.py` and `run_weekly_from_tv.py` fetch price history from
Yahoo Finance via `yfinance`. The VPS must allow **outbound** HTTPS to
`*.yahoo.com` / `query*.finance.yahoo.com` (this is a normal default on most
VPS providers — only relevant if you've locked down egress yourself). With
no market data, `run_weekly_from_tv.py` **hard-fails and writes no
verdict** rather than inventing prices — that's a safety feature, but it
means a locked-down egress policy silently blocks every decision.

## Operating notes

- **NO_TRADE is the normal case.** The decision layer only fires TRADE when a
  bucket has statistically real edge above breakeven; expect mostly
  NO_TRADE. `mc-weekly.service` treats exit codes 0/3 as success so those
  don't register as failures; exit 2 (no market data — usually a network
  problem) and exit 1 remain real failures so they show up in the logs.
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
curl http://127.0.0.1:5001/health
journalctl -u tv-bridge -f            # live webhook receipts
journalctl -u mc-weekly -n 50         # last verdict
systemctl list-timers 'mc-*'          # next scheduled runs
```

## Files

| File | Purpose |
|---|---|
| `install.sh` | Idempotent provisioner |
| `mcsim.env.example` | Env template → `/etc/monte-carlo-sim/mcsim.env` |
| `Caddyfile` | Optional TLS + reverse proxy (only if adding HTTPS) |
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
