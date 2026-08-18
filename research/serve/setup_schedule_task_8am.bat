@echo off
chcp 65001 > nul
echo ================================================================================
echo    注册 Windows 计划任务: 每日 08:00 AM 自动启动量化策略服务
echo ================================================================================

set TASK_NAME=QuantStrategyDailyService_8AM
set BAT_PATH=%~dp0start_service.bat

echo 正在创建计划任务 [%TASK_NAME%] ...
schtasks /create /tn "%TASK_NAME%" /tr "\"%BAT_PATH%\"" /sc daily /st 08:00 /f

if %errorlevel% equ 0 (
    echo.
    echo ✅ 成功注册 Windows 计划任务！
    echo 任务名称: %TASK_NAME%
    echo 触发时间: 每日早晨 08:00 AM
    echo 执行文件: %BAT_PATH%
) else (
    echo.
    echo ⚠️ 注册计划任务需要管理员权限，请右键选择 "以管理员身份运行" 此脚本。
)

echo.
pause
