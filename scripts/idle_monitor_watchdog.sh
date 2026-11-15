#!/usr/bin/env bash
# Keep the gpu_idle_monitor supervisor alive forever.
#
# The supervisor's command line is exactly "python3 gpu_idle_monitor.py"
# (no --gpu-index), so `pgrep -f 'gpu_idle_monitor.py$'` matches only the
# supervisor and not its per-GPU child monitors.
set -u

BASE_DIR=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE
RESTART="$BASE_DIR/scripts/idle_monitor_restart.sh"
LOG="$BASE_DIR/verl_log/idle_monitor_watchdog.log"

is_supervisor_alive() {
  pgrep -f 'gpu_idle_monitor.py$' >/dev/null 2>&1
}

echo "[$(date '+%Y-%m-%d %H:%M:%S')] idle_monitor_watchdog started" >> "$LOG"
while true; do
  if ! is_supervisor_alive; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] supervisor not alive, restarting" >> "$LOG"
    bash "$RESTART" >> "$LOG" 2>&1
  fi
  sleep 60
done
