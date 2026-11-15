#!/usr/bin/env bash
# 第二套 AppWorld server 集群（用于 GPU 0-3 并行跑 seed1），不杀已有集群。
# 端口 9000-9159（160 个），端口表 /tmp/codewiz/appworld_ports2.ports
set -u

N=${1:-160}
BASE_PORT=9000
LOG_DIR=/tmp/codewiz/appworld_servers2
PORT_FILE=/tmp/codewiz/appworld_ports2.ports
ROOT=/mnt/tidalfs-hssh01
APPWORLD_BIN=/root/venv-appworld/bin/appworld

mkdir -p "$LOG_DIR"
> "$PORT_FILE"

echo "[$(date)] starting $N appworld servers on ports $BASE_PORT-$((BASE_PORT+N-1))"
for i in $(seq 0 $((N-1))); do
    port=$((BASE_PORT+i))
    setsid nohup "$APPWORLD_BIN" serve environment --port "$port" --no-show-usage --root "$ROOT" \
        > "$LOG_DIR/env_$port.log" 2>&1 < /dev/null &
    echo "$port" >> "$PORT_FILE"
    [ $((i % 8)) -eq 7 ] && sleep 2
done

echo "[$(date)] waiting for servers..."
ok=0; tries=0
while [ $tries -lt 90 ]; do
    ok=0
    for i in $(seq 0 $((N-1))); do
        port=$((BASE_PORT+i))
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 "http://0.0.0.0:$port/docs" 2>/dev/null)
        [ "$code" = "200" ] && ok=$((ok+1))
    done
    [ $ok -eq $N ] && break
    tries=$((tries+1))
    sleep 10
done
echo "[$(date)] $ok/$N servers ready (tries=$tries)"
[ $ok -lt $N ] && { echo "FAILED: only $ok/$N up"; exit 1; }
echo "OK"
