@echo off
setlocal enabledelayedexpansion

:: 2_run_prediction.bat
:: Action: Morning Prediction for News Strategy
:: Using the data fetched last night. No retraining.

set TARGET_DATE=%1

if "%TARGET_DATE%"=="" (
    for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
    set TARGET_DATE=!datetime:~0,8!
)

echo ==========================================================
echo [STEP 2] Generating Dragon Picks for: %TARGET_DATE%
echo ==========================================================

python production_predict.py %TARGET_DATE%

echo.
echo ==========================================================
echo Prediction Complete.
echo ==========================================================
pause
