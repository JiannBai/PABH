#!/usr/bin/env bash
# (Re)start gpu_idle_monitor.py in supervisor mode (all physical GPUs).
# The monitor only stresses IDLE GPUs and yields to any real compute
# process, so it is safe to keep running while experiments execute.
set -u

MON_DIR=/mnt/tidalfs-hssh01/dataset/bxd
MON="$MON_DIR/gpu_idle_monitor.py"
SUP_LOG="$MON_DIR/gpu_idle_monitor_supervisor.log"
PY=/usr/bin/python3

restart_monitor() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] idle_monitor_restart: stopping existing monitors"
  # SIGTERM the supervisor and its per-GPU children; each child stops its
  # stress subprocess via its SIGTERM handler.
  pkill -TERM -f 'gpu_idle_monitor.py' 2>/dev/null
  sleep 12
  pkill -TERM -f 'gpu_stress.py' 2>/dev/null
  sleep 3

  cd "$MON_DIR" || exit 1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] idle_monitor_restart: starting supervisor"
  nohup setsid "$PY" "$MON" >> "$SUP_LOG" 2>&1 &
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] idle_monitor_restart: supervisor launched pid=$!"
}

restart_monitor
exit 0
