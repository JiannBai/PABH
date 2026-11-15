#!/usr/bin/env bash
# 清理已结束实验的中间 checkpoint，只保留 best 与 last。
# 自动跳过所有正在运行的实验与 SFT 起点目录。
set -u

BASE=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_checkpoints
LOG=/mnt/tidalfs-hssh01/dataset/bxd/research/BiPACE/verl_log/ckpt_prune_best_last.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }
is_alive() { pgrep -f "trainer.experiment_name=$1" > /dev/null 2>&1; }

for expdir in $(find "$BASE" -mindepth 2 -maxdepth 2 -type d); do
  name=$(basename "$expdir")
  [ "$name" = "sft_full" ] && continue
  is_alive "$name" && continue

  steps=$(find "$expdir" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null)
  [ -z "$steps" ] && continue
  n=$(printf '%s\n' "$steps" | wc -l)
  [ "$n" -le 1 ] && continue

  last=$(printf '%s\n' "$steps" | sed -E 's#.*/global_step_([0-9]+)#\1#' | sort -n | tail -1)
  best=""
  if [ -f "$expdir/best_checkpoint.txt" ]; then
    best=$(head -n1 "$expdir/best_checkpoint.txt" | sed -E 's/.*global_step_([0-9]+).*/\1/')
    printf '%s' "$best" | grep -qE '^[0-9]+$' || best=""
  fi

  keep="$last"
  [ -n "$best" ] && keep="$keep $best"

  for gs in $steps; do
    s=$(basename "$gs" | sed -E 's#global_step_([0-9]+)#\1#')
    ok=0
    for k in $keep; do [ "$k" = "$s" ] && ok=1; done
    if [ "$ok" = "0" ]; then
      log "deleting $gs"
      find "$gs" -depth -delete 2>>"$LOG"
    fi
  done
done

log "prune finished"
