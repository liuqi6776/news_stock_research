@echo off
setlocal enabledelayedexpansion

:: 1_fetch_data.bat
:: Action: Daily News Strategy Data Sync
:: Running after market close to fetch today's features

set TARGET_DATE=%1

if "%TARGET_DATE%"=="" (
    for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
    set TARGET_DATE=!datetime:~0,8!
)

echo ==========================================================
echo [STEP 1] Data Sync for News Strategy: %TARGET_DATE%
echo ==========================================================

python data_fetcher.py %TARGET_DATE%

echo.
echo ==========================================================
echo Sync Complete.
echo ==========================================================
pause
