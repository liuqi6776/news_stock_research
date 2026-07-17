# 新闻处理工具集（盘前机要 news_major1）

## 脚本说明

### 1. `scraper.py`（下载 HTML）
- **功能**: 从韭研公社下载新闻 HTML 文件
- **输出**: `C:\Users\liuqi\clowspace\quant_system_v2\jiayo-analysis\data\`
- **使用**: 
  - `python scraper.py --step1` - 获取文章列表
  - `python scraper.py --step2` - 下载 HTML

### 2. `analyzer.py`（分析新闻）
- **功能**: 分析韭研公社新闻并生成 news_major1 格式（盘前机要）
- **输入**: `C:\Users\liuqi\clowspace\quant_system_v2\jiayo-analysis\data\` (HTML 文件)
- **输出**: `D:\iquant_data\data_v2\news_major1\analysis_YYYY-MM-DD.json`
- **使用**: 
  - `python analyzer.py` - 分析所有未分析的新闻
  - `python analyzer.py --latest` - 分析最新的新闻

### 3. `fetch_jiayo_news.py`
- **功能**: 从韭研公社 API 获取新闻列表（测试脚本）

## 数据流程（盘前机要 news_major1）

```
韭研公社
    ↓
[scraper.py] 下载 HTML
    ↓
[analyzer.py] 智谱AI分析
    ↓
news_major1 (分析后的新闻)
```

## 当前状态

- **news_major1 最新日期**: 2026-04-01
- **今天**: 2026-04-07
- **HTML 输入目录最新**: 2026-03-27

## 下一步操作

1. 运行 `scraper.py` 获取最新新闻 HTML
2. 运行 `analyzer.py` 分析新闻
3. 检查 news_major1 是否更新到 2026-04-07
