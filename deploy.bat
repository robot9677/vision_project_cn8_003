@echo off

git add .
git commit -m "auto deploy"
git push origin main

ssh -i %USERPROFILE%\.ssh\deploy_for_gha robot96@192.168.0.14 ^
"cd ~/vision_project && \
mkdir -p /tmp/vision_roi_bak && \
cp -f data/roi/roi.json /tmp/vision_roi_bak/roi.json 2>/dev/null || true && \
cp -f data/roi/align_template.png /tmp/vision_roi_bak/align_template.png 2>/dev/null || true && \
cp -f src/inspection/recipe_static.json /tmp/vision_roi_bak/recipe_static.json 2>/dev/null || true && \
git fetch origin main && \
git reset --hard origin/main && \
cp -f /tmp/vision_roi_bak/roi.json data/roi/roi.json 2>/dev/null || true && \
cp -f /tmp/vision_roi_bak/align_template.png data/roi/align_template.png 2>/dev/null || true && \
cp -f /tmp/vision_roi_bak/recipe_static.json src/inspection/recipe_static.json 2>/dev/null || true && \
git rev-parse --short HEAD && \
ls -l data/roi/roi.json 2>/dev/null || true && \
ls -l src/inspection/recipe_static.json 2>/dev/null || true"

echo.
echo ===== CODE DEPLOY COMPLETE =====