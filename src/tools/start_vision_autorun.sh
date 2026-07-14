#!/bin/bash
set -u

PROJECT_ROOT="/home/robot96/vision_project"
LOG_DIR="$PROJECT_ROOT/data/logs/terminal"
LOG_FILE="$LOG_DIR/main_vp_runtime.log"
MAX_BYTES=$((20 * 1024 * 1024))

cd "$PROJECT_ROOT/src" || exit 1
mkdir -p "$LOG_DIR"

if [ -f "$LOG_FILE" ]; then
    SIZE=$(stat -c%s "$LOG_FILE" 2>/dev/null || echo 0)
    if [ "$SIZE" -ge "$MAX_BYTES" ]; then
        rm -f "${LOG_FILE}.1"
        mv "$LOG_FILE" "${LOG_FILE}.1"
    fi
fi

{
    echo
    echo "===== VISION START $(date '+%Y-%m-%d %H:%M:%S') pid=$$ ====="
} >> "$LOG_FILE"

# Explicit RUN mode for boot/autostart. Manual terminal execution of
# `python3 main_vp.py` remains EDIT by default.
exec stdbuf -oL -eL python3 -u main_vp.py --startup-mode run \
    >> "$LOG_FILE" 2>&1
