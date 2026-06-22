@echo off
scp -i %USERPROFILE%\.ssh\deploy_for_gha data/roi/roi.json robot96@172.30.1.92:~/vision_project/data/roi/roi.json
if exist data\roi\align_template.png scp -i %USERPROFILE%\.ssh\deploy_for_gha data/roi/align_template.png robot96@172.30.1.92:~/vision_project/data/roi/align_template.png

echo.
echo ===== ROI PUSH COMPLETE =====
pause
