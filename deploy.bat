@echo off

 git add .
 git commit -m "auto deploy"
 git push origin main

 ssh -i %USERPROFILE%\.ssh\deploy_for_gha robot96@192.168.0.14 "cd ~/vision_project && git fetch origin main && git reset --hard origin/main && git rev-parse --short HEAD && ls -l data/roi/roi.json 2>/dev/null || true"

 echo.
 echo ===== CODE DEPLOY COMPLETE =====
