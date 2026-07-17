@echo off
cd /d "c:\Users\liuqi\quant_system_v2"
echo ================================================== >> daily_evening_run.log
echo [Start Evening Sync] %date% %time% >> daily_evening_run.log
python daily_evening_pipeline.py >> daily_evening_run.log 2>&1
echo [End Evening Sync] %date% %time% >> daily_evening_run.log
