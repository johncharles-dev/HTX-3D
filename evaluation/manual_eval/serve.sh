#!/usr/bin/env bash
# Start (or restart) the blinded scoring / comparison viewer.
#
#   ./serve.sh          start on :8899
#   ./serve.sh 9000     start on another port
#   ./serve.sh stop     stop it
#
# Serves _serve/ ONLY — score.html, the blinded panels, the blinded GLBs, the
# vendored three.js and key.json. blind_key.csv is deliberately outside the
# served root so the answer key can't be reached from the browser.

set -euo pipefail
cd "$(dirname "$0")"

PORT="${1:-8899}"
LOG=/tmp/htx3d-score-server.log

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "serve.py" 2>/dev/null && echo "stopped." || echo "nothing running."
  exit 0
fi

pkill -f "serve.py ${PORT}" 2>/dev/null || true
pkill -f "http.server ${PORT}" 2>/dev/null || true
sleep 0.4

nohup python3 serve.py "$PORT" >"$LOG" 2>&1 < /dev/null &
disown 2>/dev/null || true
sleep 1.2

if ! curl -sf -o /dev/null "http://127.0.0.1:${PORT}/"; then
  echo "FAILED to start — see $LOG"; exit 1
fi

LAN=$(hostname -I | awk '{print $1}')
TS=$(tailscale ip -4 2>/dev/null | head -1 || true)
TSNAME=$(tailscale status --json 2>/dev/null \
         | python3 -c "import json,sys;print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))" 2>/dev/null || true)

HOST="${TSNAME:-${TS:-$LAN}}"
echo "serving on :${PORT}  (/api proxied to :8000 · log: $LOG)"
echo
echo "  Blinded scoring     http://${HOST}:${PORT}/"
echo "  Live comparison     http://${HOST}:${PORT}/live.html"
echo
echo "  LAN                 http://${LAN}:${PORT}/"
[[ -n "$TS" ]] && echo "  Tailscale IP        http://${TS}:${PORT}/"
echo "  SSH tunnel          ssh -L ${PORT}:localhost:${PORT} $(whoami)@${HOST}"
echo "                      then open http://localhost:${PORT}/"
echo
if curl -sf -o /dev/null --max-time 3 "http://127.0.0.1:${PORT}/api/gallery"; then
  echo "  backend :8000       reachable ✓  (live generation will work)"
else
  echo "  backend :8000       NOT reachable ✗  (live.html cannot generate)"
fi
