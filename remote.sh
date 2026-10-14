#!/usr/bin/env bash
# Run the chat on a rented GPU and use it from your own computer through an SSH tunnel.
#
#   ./remote.sh "ssh -p PORT root@HOST" --addon <hf-user>/Qwen2.5-32B-self      # start (or reconnect) + tunnel
#   ./remote.sh "ssh -p PORT root@HOST" stop                                      # stop the server on the GPU
#
# Options: --addon ID_OR_FOLDER (required to start), --self my_self.md (default selves/horus.md),
#          --port 8080 (the port on your computer and on the GPU).
# What it does: copies this folder to ~/horus-chat on the GPU, installs requirements, starts server.py there
# (listening on the GPU's 127.0.0.1 only), waits until the model is loaded, then opens the tunnel. Keep this
# window open while you chat at http://localhost:8080; Ctrl+C closes the tunnel, the server keeps running.
set -euo pipefail

if [ $# -lt 1 ]; then sed -n '2,11p' "$0"; exit 1; fi
read -r -a SSH <<< "$1"; shift
if [ "${SSH[0]:-}" != "ssh" ]; then echo "the first argument is the ssh command, e.g. \"ssh -p 12345 root@1.2.3.4\""; exit 1; fi
ADDON="" SELF="selves/horus.md" PORT=8080 ACTION=start
while [ $# -gt 0 ]; do
  case "$1" in
    --addon) ADDON="$2"; shift 2 ;;
    --self) SELF="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    stop) ACTION=stop; shift ;;
    *) echo "unknown option: $1"; exit 1 ;;
  esac
done
HERE="$(cd "$(dirname "$0")" && pwd)"
run() { ssh -o ConnectTimeout=20 "${SSH[@]:1}" "$@"; }       # options must come before the host

if [ "$ACTION" = stop ]; then
  run 'cd ~/horus-chat 2>/dev/null && [ -f server.pid ] && kill "$(cat server.pid)" 2>/dev/null && rm -f server.pid && echo "stopped" || echo "no server running"'
  exit 0
fi
[ -n "$ADDON" ] || { echo "--addon is required, e.g. --addon <hf-user>/Qwen2.5-32B-self"; exit 1; }
[ -f "$SELF" ] || { echo "no self document at $SELF"; exit 1; }

echo "1/4 copying the app to the GPU ..."
(cd "$HERE" && COPYFILE_DISABLE=1 tar czf - --exclude .git --exclude __pycache__ .) | run 'mkdir -p ~/horus-chat && cd ~/horus-chat && tar xzf - 2>/dev/null'
SELF_NAME="selves/$(basename "$SELF")"
case "$(cd "$(dirname "$SELF")" && pwd)/$(basename "$SELF")" in
  "$HERE/selves/"*) ;;                                                    # already copied with the app
  *) run "cat > ~/horus-chat/$SELF_NAME" < "$SELF" ;;
esac

echo "2/4 installing requirements (first time: a minute or two) ..."
run 'cd ~/horus-chat && { [ -f /venv/main/bin/activate ] && . /venv/main/bin/activate; true; } && python3 -m pip install -q -r requirements.txt 2>&1 | grep -v -i "warning" | tail -3 || true'

echo "3/4 starting the server ..."
run "cd ~/horus-chat && { [ -f /venv/main/bin/activate ] && . /venv/main/bin/activate; true; } && \
  if [ -f server.pid ] && kill -0 \"\$(cat server.pid)\" 2>/dev/null; then echo '   already running - reconnecting'; else \
  nohup python3 server.py --addon '$ADDON' --self '$SELF_NAME' --port $PORT > server.log 2>&1 < /dev/null & echo \$! > server.pid; fi"

echo "   waiting for the model to load (the first time it downloads the base model: 9-18 GB) ..."
LAST=""
until run "curl -s -m 5 http://127.0.0.1:$PORT/api/info" 2>/dev/null | grep -q '"ready": true'; do
  if ! run 'cd ~/horus-chat && kill -0 "$(cat server.pid)" 2>/dev/null'; then
    echo "   the server stopped - its last lines:"; run 'tail -15 ~/horus-chat/server.log'; exit 1
  fi
  NOW="$(run 'tail -c 300 ~/horus-chat/server.log | tr "\r" "\n" | grep -v "^\s*$" | tail -1' 2>/dev/null || true)"
  [ "$NOW" != "$LAST" ] && [ -n "$NOW" ] && echo "   $NOW" && LAST="$NOW"
  sleep 10
done

echo "4/4 ready. Open http://localhost:$PORT in your browser. Keep this window open; Ctrl+C closes the tunnel."
exec ssh -N -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes -L "$PORT:127.0.0.1:$PORT" "${SSH[@]:1}"
