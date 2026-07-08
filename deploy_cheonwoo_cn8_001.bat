@echo off
set TARGET=robot96@daol-vision-cheonwoo-cn8-001

echo ================================
echo Deploy to Cheonwoo CN8 Vision 001
echo ================================

git add .
git commit -m "deploy cheonwoo cn8 001"
git push origin main

ssh %TARGET% "cd ~/vision_project && mkdir -p /tmp/vision_roi_bak && cp -a data/roi /tmp/vision_roi_bak/roi_bak 2>/dev/null || true && git fetch origin main && git reset --hard origin/main && rm -rf data/roi && cp -a /tmp/vision_roi_bak/roi_bak data/roi && echo DEPLOY_DONE"

pause