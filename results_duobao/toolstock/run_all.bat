@echo off
echo ========================================
echo  每日选股完整流程
echo ========================================
echo.

set DATE=%1
if "%DATE%"=="" (
    for /f "tokens=1-3 delims=-/. " %%a in ("%date%") do (
        set Y=%%c
        set M=%%a
        set D=%%b
    )
    set DATE=%Y%%M%%D%
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
if not exist "models\daily_t1_model.joblib" (
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
echo [4/4] 预测和选股...
python 4_predict_select.py %DATE%

echo.
echo ========================================
echo  完成！
echo ========================================
pause
