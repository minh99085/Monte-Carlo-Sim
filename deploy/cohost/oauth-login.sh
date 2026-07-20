#!/usr/bin/env bash
# oauth-login.sh — one-time Robinhood OAuth login for the co-hosted bot.
#
# Robinhood's Trading MCP requires an OAuth login through a real desktop
# browser. On a headless VPS you bridge the callback with an SSH tunnel from
# your laptop, then run this script ON THE VPS.
#
# Step 1 (on YOUR laptop):
#     ssh -L 53682:127.0.0.1:53682 root@45.76.65.21
#
# Step 2 (in that SSH session, on the VPS):
#     bash /opt/monte-carlo-sim/deploy/cohost/oauth-login.sh
#
# A URL is printed — open it in your laptop browser and approve. The redirect
# comes back through the tunnel to 127.0.0.1:53682 and the bot stores the
# token in its shared /data volume. You only do this once (until the token
# expires or is revoked).

set -euo pipefail

BOT_DIR="${BOT_DIR:-/opt/Robinhood-Bot}"
PLUGIN_DIR="$BOT_DIR/hermes-agent-main/plugins/hermes-trading-engine-robinhood"

if [[ ! -d "$PLUGIN_DIR" ]]; then
	echo "ERROR: bot not installed at $PLUGIN_DIR." >&2
	echo "       Run deploy/cohost/install-robinhood.sh first." >&2
	exit 1
fi

cd "$PLUGIN_DIR"

echo "==> Starting one-time OAuth login container"
echo "    (open the printed URL in your LAPTOP browser; the callback returns"
echo "     through your 'ssh -L 53682:127.0.0.1:53682' tunnel)"
echo

# --rm: throwaway container; -p publishes the callback port for the tunnel.
docker compose --profile robinhood run --rm -p 53682:53682 \
	hermes-robinhood-agent python scripts/robinhood_oauth_login.py

echo
echo "==> If the token was saved, restart the bot so the agent picks it up:"
echo "    cd $PLUGIN_DIR && docker compose --profile robinhood up -d"
