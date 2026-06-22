@echo off
scp -i %USERPROFILE%\.ssh\deploy_for_gha robot96@172.30.1.92:~/vision_project/data/roi/roi.json data/roi/roi.json
scp -i %USERPROFILE%\.ssh\deploy_for_gha robot96@172.30.1.92:~/vision_project/data/roi/align_template.png data/roi/align_template.png 2>nul

echo.
echo ===== ROI PULL COMPLETE =====
pause
