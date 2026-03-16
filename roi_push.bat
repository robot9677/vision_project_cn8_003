@echo off
scp -i %USERPROFILE%\.ssh\deploy_for_gha data/roi/roi.json robot96@192.168.0.14:~/vision_project/data/roi/roi.json
if exist data\roi\align_template.png scp -i %USERPROFILE%\.ssh\deploy_for_gha data/roi/align_template.png robot96@192.168.0.14:~/vision_project/data/roi/align_template.png

echo.
echo ===== ROI PUSH COMPLETE =====
pause
