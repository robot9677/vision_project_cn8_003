#!/bin/bash
set -u

PROJECT_ROOT="/home/robot96/vision_project"
cd "$PROJECT_ROOT/src" || exit 1

# Explicit RUN mode for boot/autostart. Manual terminal execution of
# `python3 main_vp.py` remains EDIT by default.
exec python3 main_vp.py --startup-mode run
