#!/usr/bin/env bash
# quick_check.sh — run this on the VPS to see a real verdict in one paste.
# Usage:  bash /opt/monte-carlo-sim/deploy/quick_check.sh
set -euo pipefail

APP_DIR="/opt/monte-carlo-sim"
PY="$APP_DIR/.venv/bin/python"

echo "==> Bridge status"
systemctl is-active tv-bridge && echo "    OK: bridge is running" || echo "    NOT RUNNING — see 'systemctl status tv-bridge'"

echo
echo "==> Latest received signal"
if [[ -f "$APP_DIR/tv_data/latest_signal.json" ]]; then
	cat "$APP_DIR/tv_data/latest_signal.json"
else
	echo "    No signal received yet."
fi

echo
echo "==> Running the weekly decision on it"
"$PY" "$APP_DIR/run_weekly_from_tv.py" \
	--calibration-dir "$APP_DIR/calibration" \
	--data-dir "$APP_DIR/tv_data"
