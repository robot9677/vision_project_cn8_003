@echo off

git add .
git commit -m "auto deploy"
git push origin main

ssh -i %USERPROFILE%\.ssh\deploy_for_gha robot96@192.168.0.14 ^
"cd ~/vision_project && git fetch origin main && git reset --hard origin/main && sudo systemctl restart vision && sleep 1 && systemctl status vision --no-pager | head -n 15"
echo.
echo ===== DEPLOY COMPLETE =====
pause