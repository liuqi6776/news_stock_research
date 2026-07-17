@echo off
echo ========================================
echo  doubao策略 - 每日选股完整流程
echo  (含TS综合评分过滤 + 空仓判断)
echo ========================================
echo.

set DATE=%1
if "%DATE%"=="" (
    for /f "tokens=2 delims==" %%a in ('wmic os get localdatetime /value') do set dt=%%a
    set DATE=%dt:~0,8%
)

echo 目标日期: %DATE%
echo.

echo [1/4] 分析新闻...
python 1_analyze_news.py
if errorlevel 1 (
    echo 错误: 新闻分析失败
    pause
    exit /b 1
)

echo.
echo [2/4] 下载数据...
python 2_process_data.py %DATE%
if errorlevel 1 (
    echo 错误: 数据下载失败
    pause
    exit /b 1
)

echo.
echo [3/4] 检查模型...
if not exist "models\doubao_t1t2_model.joblib" (
    echo 模型不存在，开始训练...
    python 3_train_model.py
    if errorlevel 1 (
        echo 错误: 模型训练失败
        pause
        exit /b 1
    )
) else (
    echo 模型已存在，跳过训练
)

echo.
echo [4/4] 预测和选股 (含TS过滤 + 空仓判断)...
python 4_predict_select.py %DATE%
if errorlevel 1 (
    echo 错误: 预测选股失败
    pause
    exit /b 1
)

echo.
echo ========================================
echo  完成！查看 prediction_%DATE%.json
echo ========================================
pause
