#!/usr/bin/env bash
# install.sh — provision Monte-Carlo-Sim on a fresh Debian/Ubuntu VPS.
# Run as root:  sudo bash deploy/install.sh
#
# Idempotent: safe to re-run. It creates the service user, a venv, installs
# the units + env template, and enables the timers. It does NOT overwrite an
# existing /etc/monte-carlo-sim/mcsim.env (your secrets), and it does NOT
# start the bridge until you've filled that in.

set -euo pipefail

APP_DIR="${APP_DIR:-/opt/monte-carlo-sim}"
ETC_DIR="/etc/monte-carlo-sim"
SVC_USER="mcsim"
REPO_URL="${REPO_URL:-https://github.com/minh99085/Monte-Carlo-Sim.git}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Installing system packages"
apt-get update -qq
# Core packages only. Caddy (TLS reverse proxy) is optional and installed
# separately below — most deployments now use a plain http://IP:PORT/webhook
# address (no domain, no TLS), matching common TradingView bot setups, so a
# missing/unavailable 'caddy' package must never block the Python install.
apt-get install -y -qq python3 python3-venv python3-pip git ufw

if [[ "${WITH_CADDY:-0}" == "1" ]]; then
	echo "==> Installing Caddy (WITH_CADDY=1: TLS reverse proxy requested)"
	apt-get install -y -qq caddy || {
		echo "NOTE: 'caddy' not found in apt. Install it from"
		echo "      https://caddyserver.com/docs/install then re-run,"
		echo "      or skip it entirely — see deploy/README.md for the"
		echo "      no-domain/no-TLS setup (default path)."
	}
else
	echo "==> Skipping Caddy (default: plain http://IP:PORT setup)."
	echo "    Re-run with WITH_CADDY=1 sudo bash deploy/install.sh for HTTPS."
fi

echo "==> Service user ($SVC_USER)"
id -u "$SVC_USER" &>/dev/null || useradd --system --home "$APP_DIR" \
	--shell /usr/sbin/nologin "$SVC_USER"

echo "==> Application at $APP_DIR"
if [[ ! -d "$APP_DIR/.git" ]]; then
	git clone "$REPO_URL" "$APP_DIR"
else
	git -C "$APP_DIR" pull --ff-only
fi

echo "==> Python virtualenv + dependencies"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install --upgrade pip -q
"$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

echo "==> Runtime directories"
install -d -o "$SVC_USER" -g "$SVC_USER" \
	"$APP_DIR/tv_data" "$APP_DIR/outputs" "$APP_DIR/outputs/verdicts" \
	"$APP_DIR/calibration"
chown -R "$SVC_USER:$SVC_USER" "$APP_DIR"

echo "==> Environment file ($ETC_DIR/mcsim.env)"
install -d -m 750 "$ETC_DIR"
if [[ ! -f "$ETC_DIR/mcsim.env" ]]; then
	install -m 600 "$HERE/mcsim.env.example" "$ETC_DIR/mcsim.env"
	echo "    Created from template — EDIT IT: set TV_BRIDGE_SECRET and MC_TICKERS."
else
	echo "    Exists — left untouched."
fi

echo "==> Firewall (ufw)"
if command -v ufw &>/dev/null; then
	# Always allow SSH first so we can never lock ourselves out.
	ufw allow OpenSSH >/dev/null 2>&1 || ufw allow 22/tcp >/dev/null 2>&1
	ufw allow 5001/tcp >/dev/null 2>&1
	# TradingView permits only port 80 for plain-HTTP webhook URLs; allow it
	# so TV_BRIDGE_PORT=80 works (harmless if the bridge stays on 5001).
	ufw allow 80/tcp >/dev/null 2>&1
	if ufw status | grep -q "Status: active"; then
		echo "    ufw already active — SSH + port 5001 allowed."
	else
		echo "    ufw installed but not enabled. Enable it yourself with:"
		echo "      ufw enable   (it will ask to confirm — say yes)"
		echo "    SSH (22) and port 5001 are pre-allowed either way."
	fi
else
	echo "    ufw not found — check your VPS provider's firewall/security"
	echo "    group in their web dashboard and allow inbound TCP 5001."
fi

echo "==> systemd units"
install -m 644 "$HERE"/systemd/*.service "$HERE"/systemd/*.timer \
	/etc/systemd/system/
systemctl daemon-reload

echo "==> Enabling timers (not the bridge yet — fill in the secret first)"
systemctl enable mc-weekly.timer mc-settle.timer mc-calibrate.timer mc-paper.timer

SERVER_IP="$(curl -s -4 ifconfig.me 2>/dev/null || echo YOUR_SERVER_IP)"

cat <<EOF

Done. Remaining manual steps:

  1. Edit secrets + watchlist:
       nano $ETC_DIR/mcsim.env
     (set TV_BRIDGE_SECRET to a long random value, and MC_TICKERS to your
     watchlist. Save with Ctrl+O, Enter, then exit with Ctrl+X.)

  2. Seed calibration and start the bridge:
       systemctl start mc-calibrate.service   # one-time seed
       systemctl enable --now tv-bridge.service
       systemctl enable --now mc-weekly.timer mc-settle.timer mc-calibrate.timer
       systemctl enable --now mc-paper.timer   # paper-training track record

  3. Point the TradingView alert webhook at (replace YOUR_SECRET with what
     you set in step 1):
       http://$SERVER_IP:5001/webhook?secret=YOUR_SECRET

     This is a plain (unencrypted) address — same pattern as many
     TradingView bot setups. For an HTTPS version instead, see
     deploy/README.md's "optional: add HTTPS" section.

  Verify:
       systemctl status tv-bridge
       curl http://127.0.0.1:5001/health
       journalctl -u mc-weekly -f
EOF
