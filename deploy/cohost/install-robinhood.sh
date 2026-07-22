#!/usr/bin/env bash
# install-robinhood.sh — put the Robinhood execution bot on the SAME VPS as
# Monte-Carlo-Sim, safely and idempotently. Run as root on the VPS:
#
#     sudo bash /opt/monte-carlo-sim/deploy/cohost/install-robinhood.sh
#
# What it does:
#   1. Installs Docker if it isn't already present (the bot runs in a container;
#      Monte-Carlo-Sim keeps running as systemd + venv, untouched).
#   2. Clones (or updates) minh99085/Robinhood-Bot into /opt/Robinhood-Bot.
#   3. Creates the bot's .env from .env.example if missing, enforcing SAFE
#      defaults: live trading OFF, API bound to localhost only.
#   4. Builds and starts the bot container(s) behind the `robinhood` profile.
#   5. Health-checks the local API and prints the one-time OAuth next steps.
#
# It never opens a firewall port (the bot's 8810 API and 53682 OAuth callback
# stay on 127.0.0.1) and never enables live trading. Safe to re-run.

set -euo pipefail

BOT_DIR="${BOT_DIR:-/opt/Robinhood-Bot}"
BOT_REPO="${BOT_REPO:-https://github.com/minh99085/Robinhood-Bot.git}"
PLUGIN_SUBDIR="hermes-agent-main/plugins/hermes-trading-engine-robinhood"
PLUGIN_DIR="$BOT_DIR/$PLUGIN_SUBDIR"

log() { printf '\n==> %s\n' "$*"; }

if [[ "$(id -u)" -ne 0 ]]; then
	echo "ERROR: run as root (sudo bash ...)." >&2
	exit 1
fi

# ---------------------------------------------------------------------------
# 1. Docker
# ---------------------------------------------------------------------------
if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
	log "Docker + compose already present ($(docker --version))"
else
	log "Installing Docker (get.docker.com)"
	curl -fsSL https://get.docker.com | sh
	systemctl enable --now docker
fi

# ---------------------------------------------------------------------------
# 2. Clone or update the bot repo
# ---------------------------------------------------------------------------
if [[ -d "$BOT_DIR/.git" ]]; then
	log "Updating existing clone at $BOT_DIR"
	git -C "$BOT_DIR" pull --ff-only || {
		echo "NOTE: 'git pull' was not fast-forward; leaving the checkout as-is." >&2
	}
else
	log "Cloning $BOT_REPO -> $BOT_DIR"
	git clone "$BOT_REPO" "$BOT_DIR"
fi

if [[ ! -d "$PLUGIN_DIR" ]]; then
	echo "ERROR: expected plugin dir not found: $PLUGIN_DIR" >&2
	echo "       The Robinhood-Bot layout may have changed." >&2
	exit 1
fi

# ---------------------------------------------------------------------------
# 3. .env with SAFE defaults (never overwrite an existing one)
# ---------------------------------------------------------------------------
ENV_FILE="$PLUGIN_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
	log "Keeping existing $ENV_FILE (your settings/secrets)"
else
	log "Creating $ENV_FILE from .env.example (safe defaults)"
	cp "$PLUGIN_DIR/.env.example" "$ENV_FILE"
fi

# Enforce the two safety-critical settings regardless of what the example ships:
#   * live trading OFF   * API published only on localhost
set_env() {  # set_env KEY VALUE  — idempotent in-place upsert
	local key="$1" val="$2"
	if grep -qE "^${key}=" "$ENV_FILE"; then
		sed -i "s|^${key}=.*|${key}=${val}|" "$ENV_FILE"
	else
		printf '%s=%s\n' "$key" "$val" >>"$ENV_FILE"
	fi
}
set_env RH_LIVE_TRADING_ENABLED 0
set_env RH_API_PUBLISH 127.0.0.1:8810
chmod 600 "$ENV_FILE"
log "Safety defaults enforced: RH_LIVE_TRADING_ENABLED=0, RH_API_PUBLISH=127.0.0.1:8810"

# ---------------------------------------------------------------------------
# 4. The reviewed fixes and the MC→bot paper bridge now live natively in the
#    Robinhood-Bot repo (on main), so the `git pull` in step 2 already brought
#    them — the plugin's own docker-compose.yml defines the `mc-bridge`
#    profile. No file overlay: the old deploy/cohost/{patches,bridge} copy
#    step was retired once this session gained push access to the bot repo
#    (overlaying tracked files was also what caused `git pull` collisions on
#    the VPS). A stale docker-compose.override.yml from an earlier install is
#    now redundant and is removed so it can't shadow the committed compose.
# ---------------------------------------------------------------------------
STALE_OVERRIDE="$PLUGIN_DIR/docker-compose.override.yml"
if [[ -f "$STALE_OVERRIDE" ]]; then
	log "Removing redundant docker-compose.override.yml (bridge is now in the bot's own compose)"
	rm -f "$STALE_OVERRIDE"
fi

# ---------------------------------------------------------------------------
# 5. Build + start the container(s) — bot API + agent + MC paper bridge
# ---------------------------------------------------------------------------
log "Building and starting the bot + bridge"
cd "$PLUGIN_DIR"
docker compose --profile robinhood --profile mc-bridge up -d --build

# ---------------------------------------------------------------------------
# 6. Health check + next steps
# ---------------------------------------------------------------------------
log "Waiting for the API health endpoint (up to ~40s)"
ok=0
for _ in $(seq 1 20); do
	if curl -fsS http://127.0.0.1:8810/api/health >/dev/null 2>&1; then
		ok=1
		break
	fi
	sleep 2
done

echo
if [[ "$ok" -eq 1 ]]; then
	echo "OK: Robinhood bot API is healthy at http://127.0.0.1:8810/api/health"
else
	echo "NOTE: API did not answer yet. Check: docker compose --profile robinhood logs --tail=50"
fi

cat <<'EOF'

Robinhood bot is installed alongside Monte-Carlo-Sim. Live trading is OFF.

The MC→bot paper bridge is also running: it reads the sim's verdict files,
rehearses each fresh TRADE through the bot's safety gates, and records the
outcome to a paper ledger — placing NOTHING. Watch it with:
  docker exec hermes-mc-bridge sh -c 'tail -n 5 /data/mc_bridge_ledger.jsonl'

Remaining manual steps (only you can do these):

  1. One-time Robinhood OAuth (needs your login + a desktop browser).
     From YOUR laptop, open an SSH tunnel:
       ssh -L 53682:127.0.0.1:53682 root@45.76.65.21
     Then, in that SSH session on the VPS, run:
       bash /opt/monte-carlo-sim/deploy/cohost/oauth-login.sh
     Open the printed URL in your laptop browser and approve. The callback
     returns through the tunnel and seeds the bot's token.

  2. Check both systems any time:
       bash /opt/monte-carlo-sim/deploy/cohost/status.sh

  Live trading stays OFF until you deliberately set RH_LIVE_TRADING_ENABLED=1
  in the bot's .env and restart it — do that only after paper training looks
  good. This installer never turns it on.
EOF
