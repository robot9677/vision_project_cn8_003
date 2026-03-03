git add .
git commit -m "auto deploy"
git push origin main
ssh -i ~/.ssh/deploy_for_gha robot96@192.168.0.14 "cd ~/vision_project && git fetch origin && git reset --hard origin/main && git rev-parse --short HEAD"