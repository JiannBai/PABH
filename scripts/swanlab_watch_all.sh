#!/usr/bin/env bash
# Spin up one swanlab dashboard per local logdir on adjacent ports.
# swanlab local mode doesn't merge logdirs — this is the next-best thing.
#
# Usage:
#   bash scripts/swanlab_watch_all.sh                # default: scan ./swanlog* dirs
#   bash scripts/swanlab_watch_all.sh dir1 dir2 ...  # specify dirs explicitly
#
# Stop everything:
#   pkill -f 'swanlab watch'

set -e
START_PORT=${START_PORT:-5092}

if [ "$#" -gt 0 ]; then
    DIRS=("$@")
else
    DIRS=()
    for d in "$PWD"/swanlog*; do
        [ -d "$d" ] && [ -f "$d/runs.swanlab" ] && DIRS+=("$d")
    done
fi

if [ "${#DIRS[@]}" -eq 0 ]; then
    echo "no swanlog* dirs with runs.swanlab found under $PWD"
    exit 1
fi

echo "starting $((${#DIRS[@]})) swanlab dashboards..."
PORT=$START_PORT
for d in "${DIRS[@]}"; do
    name=$(basename "$d")
    echo "  $name → http://127.0.0.1:$PORT"
    nohup swanlab watch -l "$d" --port "$PORT" \
        > /tmp/swanlab_${name}.out 2>&1 &
    PORT=$((PORT + 1))
done
echo
echo "logs: /tmp/swanlab_<name>.out"
echo "stop: pkill -f 'swanlab watch'"
