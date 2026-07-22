# Co-hosting: Monte-Carlo-Sim + Robinhood bot on one VPS

This runs the **brain** (Monte-Carlo-Sim — decides what to trade) and the
**hands** (the Robinhood bot — places orders) on the *same* server. They stay
separate programs that share only decision files; nothing about the trading
logic changes.

> Live trading is **OFF** by default and stays off until you deliberately turn
> it on. Setting this up places no orders and reviews no real orders. Do the
> paper-training first (see `../README.md`), and only connect live much later.

## Why one VPS is fine

| | Monte-Carlo-Sim | Robinhood bot |
|---|---|---|
| Runs as | systemd + Python venv | Docker container |
| Directory | `/opt/monte-carlo-sim` | `/opt/Robinhood-Bot` |
| Port | `5001` (TradingView webhook) | `8810` (health API, localhost only) |
| Extra port | — | `53682` (one-time OAuth, localhost only) |

Different ports, isolated runtimes, separate folders — no collision. Both are
lightweight. The bot's API and OAuth ports are bound to `127.0.0.1` (not the
public internet), so no firewall change is needed.

## Before you start (check RAM)

Adding Docker + a container needs a little headroom. On the VPS:

```
free -m
```

If you have **less than ~1.5 GB free**, resize the VPS to ~2 GB first (Vultr:
power off → resize → power on). Everything here is light, but Docker builds want
some room.

## Step 1 — install the bot next to the sim

On the VPS (this does not touch your running sim):

```
sudo bash /opt/monte-carlo-sim/deploy/cohost/install-robinhood.sh
```

This installs Docker if needed, clones `Robinhood-Bot` to `/opt/Robinhood-Bot`,
writes a safe `.env` (live trading OFF, API localhost-only), builds the
container, and health-checks it. Safe to re-run.

When it finishes it should say the API is healthy. If not, it prints the log
command to look at.

## Step 2 — one-time Robinhood login (OAuth)

The bot needs to log in to your Robinhood account once. Robinhood requires a
real desktop browser, so you bridge it with an SSH tunnel.

1. **On your laptop**, open a tunnel (keep this window open):
   ```
   ssh -L 53682:127.0.0.1:53682 root@45.76.65.21
   ```
2. **In that same SSH window (now on the VPS)**, run:
   ```
   bash /opt/monte-carlo-sim/deploy/cohost/oauth-login.sh
   ```
3. It prints a URL. **Open that URL in your laptop's browser** and approve. The
   approval returns through the tunnel and the bot saves your token.
4. Restart the bot so it picks up the token:
   ```
   cd /opt/Robinhood-Bot/hermes-agent-main/plugins/hermes-trading-engine-robinhood
   docker compose --profile robinhood up -d
   ```

You only do this once (until the token expires or you revoke it).

## Step 3 — check both are healthy

```
bash /opt/monte-carlo-sim/deploy/cohost/status.sh
```

You should see the sim's `tv-bridge` active, the `mc-*` timers listed, the
Robinhood containers `Up`, the API answering, and `live trading: 0 (OFF)`.

## What YOU need to do manually (I can't do these)

- Run Step 1 (Docker install needs root on your server).
- Do the OAuth login in Step 2 — it needs your Robinhood account and your
  browser; no one else can complete it for you.
- Decide, much later, when to enable live trading (flip
  `RH_LIVE_TRADING_ENABLED=1` in the bot's `.env` and restart). Don't do this
  until paper training has built a solid track record.

## Note on these scripts

They were written and syntax-checked, but **not run end-to-end from here** —
Docker and your Robinhood account don't exist in the build environment. The
verification you run on the VPS (Step 1 health check + Step 3 status) is the
real end-to-end test. If anything doesn't match what's described, paste the
output back and it'll get fixed.

## Reviewed bot fixes (now native in the bot repo)

A full quant-team review of the Robinhood plugin found and fixed 10 issues —
including a dead daily-loss halt, safety counters that reset on restart, and
option orders whose dollar exposure was under-counted 100× by the per-order
cap. **These fixes now live natively on `minh99085/Robinhood-Bot` `main`**, so
`git pull` in the bot checkout brings them — there is no longer a file
overlay. (Earlier, before this session had push access to the bot repo, the
fixes shipped as a `deploy/cohost/patches/` overlay; that mechanism has been
retired because overlaying tracked files also caused `git pull` collisions on
the VPS.) Full findings and the improvement roadmap:
[`BOT_REVIEW.md`](BOT_REVIEW.md). Plugin test suite: 56/56 passing.

## The MC → bot paper bridge (included)

The installer also wires the first connection between the two programs: a
**paper bridge** (`hermes-mc-bridge` container) that watches the sim's
verdict files (`outputs/verdicts/` and `outputs/paper_verdicts/`) and, for
every fresh `TRADE`, rehearses it through the bot's own safety gates —
notional cap, daily-loss halt, day-trade limit — then records the outcome to
an append-only paper ledger.

What it deliberately does **not** do (phase 1):

- **No Robinhood calls at all.** No OAuth needed; nothing can be placed or
  even reviewed against your account. It is a full-dress rehearsal on paper.
- **No shorts.** Robinhood can't short shares; `short` verdicts are logged
  and skipped (turning them into long puts is a possible later phase).
- **No repeats.** Each verdict file is processed exactly once, even across
  restarts.
- **No stale orders.** Verdicts older than 48 hours are skipped.

Watch what it's deciding:

```
docker exec hermes-mc-bridge sh -c 'tail -n 5 /data/mc_bridge_ledger.jsonl'
```

Each line shows the verdict, how it mapped to an order plan (symbol, buy,
quantity, limit price), the gate result, and the outcome. Note: the bot's
per-order cap (`RH_MAX_ORDER_NOTIONAL_USD`, default $100) is far below one
share of expensive symbols like SPY — those show up as skips with a message
telling you what to raise the cap to. That's the cap working, not a bug.

**Where the bridge code lives:** natively in the Robinhood-Bot repo
(`engine/robinhood/mc_bridge.py`, `scripts/run_mc_bridge.py`, and the
`mc-bridge` service in the plugin's own `docker-compose.yml`). `git pull` in
the bot checkout brings it; the installer no longer overlays any files.

## Next milestone (not built yet)

Connecting the bridge's *paper plans* to Robinhood's real `review_*` /
`place_*` calls — only after OAuth is done, the paper ledger looks healthy
over multiple weekly cycles, and you deliberately enable live trading.
