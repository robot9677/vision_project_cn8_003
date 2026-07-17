@echo off
setlocal EnableExtensions

rem ============================================================
rem Daol Vision - Single Equipment Deploy Script
rem Each source folder must have its own deploy.bat.
rem Jetson path is always: /home/robot96/vision_project
rem ============================================================

set SSH_KEY=%USERPROFILE%\.ssh\deploy_for_gha

rem ===== Change only these 2 lines per equipment source folder =====
set "BOARD_NAME=CN8-001"
set "TARGET=robot96@daol-vision-cheonwoo-cn8-001"

rem ===== Git deploy source =====
set "GIT_REMOTE=origin"
set "GIT_BRANCH=main"
set "DEPLOY_REF=%GIT_REMOTE%/%GIT_BRANCH%"

echo.
echo ============================================
echo Deploy %BOARD_NAME%
echo TARGET     = %TARGET%
echo DEPLOY_REF = %DEPLOY_REF%
echo ============================================
echo.

call :COMMIT_AND_PUSH
if errorlevel 1 exit /b 1

call :DEPLOY_TO_TARGET
if errorlevel 1 exit /b 1

echo.
echo ===== %BOARD_NAME% DEPLOY COMPLETE =====
pause
exit /b 0


:COMMIT_AND_PUSH
echo.
echo [1/3] Stage source files

git add .gitignore 2>nul
git add src
git add data/config
git add data/roi

rem ===== Do not commit equipment runtime files =====
git reset -q -- data/roi/roi.json 2>nul
git reset -q -- data/roi/align_template.png 2>nul
git reset -q -- data/roi/profiles/*_roi.json 2>nul
git reset -q -- data/roi/profiles/align_template_*.png 2>nul
git reset -q -- data/roi/baseline_profile.json 2>nul
git reset -q -- data/roi/recipe_static.json.save 2>nul
git reset -q -- data/roi/recipes/recipe_auto.json 2>nul
git reset -q -- logs 2>nul
git reset -q -- data/logs 2>nul
git reset -q -- data/dataset 2>nul

echo.
echo [2/3] Commit if changed

git diff --cached --quiet
if errorlevel 1 (
    git commit -m "auto deploy %BOARD_NAME%"
) else (
    echo No staged changes. Skip commit.
)

echo.
echo [3/3] Push %GIT_REMOTE% %GIT_BRANCH%
git push %GIT_REMOTE% %GIT_BRANCH%
exit /b %errorlevel%


:DEPLOY_TO_TARGET
echo.
echo Deploy %DEPLOY_REF% to %TARGET%

ssh -i "%SSH_KEY%" %TARGET% "cd ~/vision_project && mkdir -p /tmp/vision_runtime_bak && cp -f data/roi/roi.json /tmp/vision_runtime_bak/roi.json 2>/dev/null || true && mkdir -p /tmp/vision_runtime_bak/profiles && cp -a data/roi/profiles/. /tmp/vision_runtime_bak/profiles/ 2>/dev/null || true && cp -f data/roi/align_template.png /tmp/vision_runtime_bak/align_template.png 2>/dev/null || true && git fetch %GIT_REMOTE% %GIT_BRANCH% && git reset --hard %DEPLOY_REF% && cp -f /tmp/vision_runtime_bak/roi.json data/roi/roi.json 2>/dev/null || true && mkdir -p data/roi/profiles && cp -a /tmp/vision_runtime_bak/profiles/. data/roi/profiles/ 2>/dev/null || true && cp -f /tmp/vision_runtime_bak/align_template.png data/roi/align_template.png 2>/dev/null || true && git rev-parse --short HEAD && echo DEPLOY_DONE_%BOARD_NAME%"

exit /b %errorlevel%
