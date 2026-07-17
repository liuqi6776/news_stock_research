@echo off
cd /d "c:\Users\liuqi\quant_system_v2"
echo ================================================== >> daily_pipeline_run.log
echo [Start] %date% %time% >> daily_pipeline_run.log
python daily_morning_pipeline.py >> daily_pipeline_run.log 2>&1
echo [End] %date% %time% >> daily_pipeline_run.log
