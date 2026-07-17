@echo off
chcp 65001 >nul
echo ========================================
echo 韭研公社数据分析一键运行
echo ========================================
echo.

set PYTHON=C:\Users\liuqi\anaconda3\envs\iquant\python.exe
set SCRAPER=%~dp0scraper.py
set ANALYZER=%~dp0analyzer.py

echo [1/3] 下载第一页文章链接到 list_id ...
echo.
%PYTHON% %SCRAPER% --step1
echo.

echo [2/3] 下载文章HTML内容到 data ...
echo.
%PYTHON% %SCRAPER% --step2
echo.

echo [3/3] 分析最新下载的HTML ...
echo.
%PYTHON% %ANALYZER% --latest
echo.

echo ========================================
echo 完成！
echo ========================================
pause
