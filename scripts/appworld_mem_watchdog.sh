#!/usr/bin/env bash
# AppWorld server 内存看门狗
# 某单个 server 会因任务在 IPython REPL/内存 DB 里累积而泄漏到数十 GB，
# 把整机内存打爆触发 Ray OOM。这里每 2 分钟扫一次，RSS > 15GB 的 server
# 直接杀掉并在原端口重启（端口表 /tmp/codewiz/appworld_ports.ports 不变）。

THRESHOLD_GB=15
ROOT=/mnt/tidalfs-hssh01
BIN=/root/venv-appworld/bin/appworld
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/appworld_mem_watchdog.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

# 单例保护：只允许一个看门狗实例（防止多次拉起导致互踩）
LOCKFILE=/tmp/appworld_mem_watchdog.pid
if [ -f "$LOCKFILE" ] && kill -0 "$(cat "$LOCKFILE" 2>/dev/null)" 2>/dev/null; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] another appworld mem watchdog already running, exit" >> "$LOG"
  exit 0
fi
echo $$ > "$LOCKFILE"

log "appworld mem watchdog started, threshold=${THRESHOLD_GB}GB"
while true; do
  for pid in $(pgrep -f 'appworld serve environment'); do
    rss_kb=$(awk '/VmRSS/{print $2}' /proc/$pid/status 2>/dev/null)
    [ -z "$rss_kb" ] && continue
    rss_gb=$((rss_kb / 1048576))
    if [ "$rss_gb" -gt "$THRESHOLD_GB" ]; then
      port=$(ps -o args= -p "$pid" 2>/dev/null | grep -oE '\-\-port [0-9]+' | awk '{print $2}')
      log "server pid=$pid port=$port RSS=${rss_gb}GB exceeds threshold, restarting"
      kill -9 "$pid" 2>/dev/null
      sleep 3
      [ -n "$port" ] && setsid nohup "$BIN" serve environment --port "$port" \
        --no-show-usage --no-docker --root "$ROOT" \
        > /tmp/codewiz/appworld_servers/env_$port.log 2>&1 < /dev/null &
      log "restarted server on port $port"
    fi
  done
  sleep 120
done
