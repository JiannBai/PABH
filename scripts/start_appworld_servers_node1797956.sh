#!/usr/bin/env bash
# AppWorld server 集群批量启动（ai-1797956，2026-10-15）
# 训练规格：train 16 tasks × group 8 = 128 workers + val 32 = 160 实例
# 端口规划：8000-8159（160 个）
# 用法：bash start_appworld_servers.sh [N_INSTANCES]（默认 160）
# 输出：端口表 /tmp/codewiz/appworld_ports.ports（供 APPWORLD_PORT_FILE）

N=${1:-160}
BASE_PORT=8000
LOG_DIR=/tmp/codewiz/appworld_servers
ROOT=/mnt/tidalfs-hssh01
mkdir -p $LOG_DIR
APPWORLD_BIN=/root/venv-appworld/bin/appworld

# 清旧实例
pkill -f "appworld serve environment" 2>/dev/null
sleep 3

echo "[$(date)] starting $N appworld servers on ports $BASE_PORT-$((BASE_PORT+N-1))"
> /tmp/codewiz/appworld_ports.ports
for i in $(seq 0 $((N-1))); do
    port=$((BASE_PORT+i))
    setsid nohup $APPWORLD_BIN serve environment --port $port --no-show-usage --root $ROOT \
        > $LOG_DIR/env_$port.log 2>&1 < /dev/null &
    echo $port >> /tmp/codewiz/appworld_ports.ports
    # 每 8 个一批错峰（避免 fork 风暴）
    [ $((i % 8)) -eq 7 ] && sleep 2
done

echo "[$(date)] waiting for servers to come up..."
ok=0; tries=0
while [ $tries -lt 60 ]; do
    ok=0
    for i in $(seq 0 $((N-1))); do
        port=$((BASE_PORT+i))
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://0.0.0.0:$port/docs 2>/dev/null)
        [ "$code" = "200" ] && ok=$((ok+1))
    done
    [ $ok -eq $N ] && break
    tries=$((tries+1))
    sleep 10
done
echo "[$(date)] $ok/$N servers ready (tries=$tries)"
[ $ok -lt $N ] && { echo "FAILED: not all servers up"; exit 1; }
echo "OK"
