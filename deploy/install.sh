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
apt-get install -y -qq python3 python3-venv python3-pip git caddy || {
	echo "NOTE: 'caddy' not found in apt. Install it from"
	echo "      https://caddyserver.com/docs/install then re-run."
}

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

echo "==> systemd units"
install -m 644 "$HERE"/systemd/*.service "$HERE"/systemd/*.timer \
	/etc/systemd/system/
systemctl daemon-reload

echo "==> Enabling timers (not the bridge yet — fill in the secret first)"
systemctl enable mc-weekly.timer mc-settle.timer mc-calibrate.timer

cat <<EOF

Done. Remaining manual steps:

  1. Edit secrets + watchlist:
       sudoedit $ETC_DIR/mcsim.env

  2. Point a DNS record at this host, edit domain/email in the Caddyfile,
     then install it:
       cp $HERE/Caddyfile /etc/caddy/Caddyfile
       systemctl reload caddy

  3. Seed calibration and start the bridge:
       systemctl start mc-calibrate.service   # one-time seed
       systemctl enable --now tv-bridge.service
       systemctl start mc-weekly.timer mc-settle.timer mc-calibrate.timer

  4. Point the TradingView alert webhook at:
       https://YOUR_DOMAIN/webhook?secret=YOUR_TV_BRIDGE_SECRET

  Verify:
       systemctl status tv-bridge
       curl https://YOUR_DOMAIN/health
       journalctl -u mc-weekly -f
EOF
