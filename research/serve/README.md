# A股综合复合量化策略监控服务 (Quant Strategy Service)

本服务提供经过全面审计与验证的最优综合量化策略（全市场 ENS 选股 Top40 + S123 三档宏观估值择时 + 净值熔断降档 -10%×0.5 + IM 期货低基差对冲）的**本地生产级服务、每日早晨 08:00 AM 自动定时调度与现代化 Web 监控仪表盘**。

---

## 一、 快速启动与定时调度 (Quick Start & 8:00 AM Automation)

### 方式 1: 常驻运行（内置 08:00 AM 自动调度）
直接双击运行 **`research/serve/start_service.bat`**：
- 启动本地 FastAPI Web 监控仪表盘（`http://127.0.0.1:8000`）；
- 后台常驻定时器：**每日早晨 08:00 AM 自动重新计算全市场最新信号**，在开盘前（09:30）准备就绪今日操作建议。

### 方式 2: 注册 Windows 系统级 08:00 AM 自动启动任务
右键以管理员身份运行 **`research/serve/setup_schedule_task_8am.bat`**：
- 自动向 Windows 系统注册名为 `QuantStrategyDailyService_8AM` 的计划任务；
- 无论电脑何时开机，系统都会在**每天早晨 08:00 AM 准时自动拉起策略服务**并执行信号生成。

### 方式 3: 命令行直接启动
```bash
# 进入仓库根目录
cd c:\Users\liuqi\quant_system_v2

# 启动服务与调度器
python research/serve/scheduler.py
```

浏览器访问: **`http://127.0.0.1:8000`**

---

## 二、 核心模块与文件清单

| 文件/目录 | 功能描述 |
| :--- | :--- |
| `research/serve/scheduler.py` | 策略自动化调度器，负责拉起 Web 服务并在每日 08:00 AM 触发信号生成 |
| `research/serve/app.py` | FastAPI Web 后端服务主入口，提供 RESTful API 与前端页面托管 |
| `research/serve/composite_signal_generator.py` | 前向实时信号生成引擎（计算 S123、Top40 选股、V8 避险与 IM 对冲） |
| `research/serve/templates/index.html` | 现代化暗黑高颜值交互式监控仪表盘前端 |
| `research/serve/assets/` | 静态图表与全景回测绩效看板 (`integrated_composite_nav.png`) |
| `research/serve/data/daily/` | 每日信号历史 JSON 快照存储目录（按 `YYYY-MM-DD.json` 归档） |
| `research/serve/start_service.bat` | 一键启动服务批处理脚本 |
| `research/serve/setup_schedule_task_8am.bat` | 注册 Windows 每日 08:00 AM 自动启动计划任务脚本 |

---

## 三、 API 接口列表

| 请求方式 | 路由 | 说明 |
| :--- | :--- | :--- |
| `GET` | `/` | 策略交互式监控仪表盘主页 |
| `GET` | `/api/today` | 今日最新复合量化信号（包含宏观 S123、Top40 选股、V8 避险与 IM 对冲） |
| `GET` | `/api/history` | 历史调仓信号时间轴（默认近 90 天） |
| `GET` | `/api/picks?date=YYYY-MM-DD` | 指定调仓日的持仓明细与目标金额 |
| `GET` | `/api/performance` | 策略全历史回测绩效指标与年度收益表 |
| `POST` | `/api/run_signal` | 异步触发后台立即重新计算最新信号 |
| `GET` | `/api/chart` | 获取高清三栏全景回测与对冲实测看板图像 |
