@echo off

 git add .gitignore
 git add src
 git add data/config
 git add data/roi

 git reset -- data/roi/roi.json
 git reset -- data/roi/profiles/*_roi.json

 git commit -m "auto deploy"
 git push origin main

ssh -i %USERPROFILE%\.ssh\deploy_for_gha robot96@172.30.1.92 "cd ~/vision_project && mkdir -p /tmp/vision_runtime_bak && cp -f data/roi/roi.json /tmp/vision_runtime_bak/roi.json 2>/dev/null || true && cp -f data/roi/align_template.png /tmp/vision_runtime_bak/align_template.png 2>/dev/null || true && git fetch origin main && git reset --hard origin/main && cp -f /tmp/vision_runtime_bak/roi.json data/roi/roi.json 2>/dev/null || true && cp -f /tmp/vision_runtime_bak/align_template.png data/roi/align_template.png 2>/dev/null || true && git rev-parse --short HEAD && ls -l data/roi/roi.json 2>/dev/null || true"

 echo.
 echo ===== CODE DEPLOY COMPLETE =====
