#!/usr/bin/env bash
# status.sh — one-paste health of BOTH co-hosted systems on this VPS:
# Monte-Carlo-Sim (the brain) and the Robinhood bot (the hands).
# Usage:  bash /opt/monte-carlo-sim/deploy/cohost/status.sh

set -uo pipefail  # not -e: keep going so every section prints

MC_DIR="${MC_DIR:-/opt/monte-carlo-sim}"
BOT_DIR="${BOT_DIR:-/opt/Robinhood-Bot}"
PLUGIN_DIR="$BOT_DIR/hermes-agent-main/plugins/hermes-trading-engine-robinhood"

echo "================ Monte-Carlo-Sim (brain) ================"
printf 'tv-bridge: '
systemctl is-active tv-bridge 2>/dev/null || echo "not running"
echo "timers:"
systemctl list-timers 'mc-*' --no-pager 2>/dev/null | sed -n '1,6p' || echo "  (none)"
PAPER_LOG="$MC_DIR/outputs/paper_log.jsonl"
if [[ -f "$PAPER_LOG" ]]; then
	echo "paper log: $(wc -l <"$PAPER_LOG") verdicts logged"
else
	echo "paper log: none yet (run paper_train.py or wait for mc-paper.timer)"
fi

echo
echo "================ Robinhood bot (hands) ================"
if command -v docker >/dev/null 2>&1; then
	echo "containers:"
	docker ps --filter "name=hermes-robinhood" \
		--format '  {{.Names}}  {{.Status}}' 2>/dev/null || echo "  (docker error)"
	printf 'api health: '
	if curl -fsS http://127.0.0.1:8810/api/health 2>/dev/null; then
		echo
	else
		echo "no response (check: cd $PLUGIN_DIR && docker compose --profile robinhood logs --tail=50)"
	fi
	if [[ -f "$PLUGIN_DIR/.env" ]]; then
		live="$(grep -E '^RH_LIVE_TRADING_ENABLED=' "$PLUGIN_DIR/.env" 2>/dev/null | cut -d= -f2)"
		echo "live trading: ${live:-unknown}  (0 = OFF / paper-safe)"
	fi
	echo
	echo "-------- MC → bot paper bridge --------"
	if docker ps --filter "name=hermes-mc-bridge" --format '{{.Names}}' 2>/dev/null | grep -q .; then
		docker ps --filter "name=hermes-mc-bridge" --format '  {{.Names}}  {{.Status}}'
		echo "last bridge ledger entries (paper decisions, no orders placed):"
		docker exec hermes-mc-bridge sh -c \
			'tail -n 3 /data/mc_bridge_ledger.jsonl 2>/dev/null' \
			| sed 's/^/  /' || echo "  (ledger empty — no verdicts processed yet)"
	else
		echo "  bridge container not running (re-run install-robinhood.sh)"
	fi
else
	echo "docker not installed — run deploy/cohost/install-robinhood.sh"
fi
echo "======================================================="
