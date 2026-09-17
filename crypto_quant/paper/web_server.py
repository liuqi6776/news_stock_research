# -*- coding: utf-8 -*-
"""
Web Dashboard & Monitoring Server: Crypto Structural Trend Paper Service (Phase 22)
===================================================================================
Hosts an institutional, responsive real-time web dashboard displaying 10,000 USDT
ETH Paper Trading results with interactive refresh buttons and automated daily
morning email dispatch to 568701293@qq.com.
"""

from datetime import datetime, time, timezone, timedelta
import json
import os
from pathlib import Path
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import threading
import time as time_lib
from typing import Any, Dict, Optional
import urllib.request

from flask import Flask, jsonify, render_template_string, request
import pandas as pd
from dotenv import load_dotenv

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
dotenv_paths = [
    root_dir / ".env",
    Path(r"C:\Users\liuqi\quant_system_v2\.env"),
    Path.home() / ".env",
]
for dp in dotenv_paths:
    if dp.exists():
        load_dotenv(dp)
        break

from crypto_quant.paper.config import (
    CONFIG_HASH,
    DOCS_DIR,
    ENABLE_REAL_ORDERS,
    EXPERIMENT_ID,
    JOURNAL_PATH,
    LOG_DIR,
    PAPER_MODE,
    STATE_PATH,
    STRATEGY_NAME,
    TIMEFRAME,
    get_code_commit,
)
from crypto_quant.paper.journal import PaperJournal
from crypto_quant.paper.service import PaperService
from crypto_quant.paper.state import PortfolioPaperState
from crypto_quant.paper.strategy import StructuralTrendPaperStrategy

# ============================================================================
# CONSTANTS & CAPITAL CONFIGURATION
# ============================================================================
INITIAL_CAPITAL_USDT: float = 10000.0  # 10,000 USDT base capital
DEFAULT_RECIPIENT: str = "568701293@qq.com"
PORT: int = 8088

app = Flask(__name__)
service_lock = threading.Lock()
DEFAULT_PUBLIC_URL: str = "https://percolate-zipfile-corned.ngrok-free.dev"
PUBLIC_TUNNEL_URL: Optional[str] = None


def get_effective_public_url() -> str:
    """
    Dynamically resolves external public HTTPS URL:
    1. Query local ngrok API (http://127.0.0.1:4040/api/tunnels).
    2. Check PUBLIC_TUNNEL_URL global.
    3. Check PUBLIC_TUNNEL_URL or NGROK_URL in environment.
    4. Fallback to DEFAULT_PUBLIC_URL ('https://percolate-zipfile-corned.ngrok-free.dev').
    Never falls back to localhost / 127.0.0.1 for external reports.
    """
    global PUBLIC_TUNNEL_URL
    # 1. Try querying local ngrok inspection endpoint
    try:
        req = urllib.request.Request("http://127.0.0.1:4040/api/tunnels", headers={"User-Agent": "PaperRunner/1.0"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode())
            for t in data.get("tunnels", []):
                u = t.get("public_url", "")
                if u.startswith("https://"):
                    PUBLIC_TUNNEL_URL = u
                    return u
    except Exception:
        pass

    # 2. Check global
    if PUBLIC_TUNNEL_URL and PUBLIC_TUNNEL_URL.startswith("https://"):
        return PUBLIC_TUNNEL_URL

    # 3. Check environment
    env_url = os.getenv("PUBLIC_TUNNEL_URL") or os.getenv("NGROK_URL")
    if env_url and env_url.startswith("https://"):
        PUBLIC_TUNNEL_URL = env_url
        return env_url

    # 4. Fallback to known persistent dev domain
    PUBLIC_TUNNEL_URL = DEFAULT_PUBLIC_URL
    return DEFAULT_PUBLIC_URL


# ============================================================================
# DATA & STATUS EXTRACTION HELPERS
# ============================================================================
def get_current_dashboard_data() -> Dict[str, Any]:
    """Extracts complete operational and strategy state scaled to 10,000 USDT."""
    state = PortfolioPaperState.load(STATE_PATH) if STATE_PATH.exists() else PortfolioPaperState.create_initial()
    eth_s = state.eth_state

    # 1. Capital calculations (Scaled to 10,000 USDT)
    equity_usdt = eth_s.total_equity * INITIAL_CAPITAL_USDT
    peak_usdt = eth_s.peak_equity * INITIAL_CAPITAL_USDT
    pnl_usdt = (eth_s.total_equity - 1.0) * INITIAL_CAPITAL_USDT
    pnl_pct = (eth_s.total_equity - 1.0) * 100.0
    fees_usdt = eth_s.total_fees * INITIAL_CAPITAL_USDT

    # 2. Position calculations
    is_in_pos = (eth_s.position == 1)
    pos_direction = "LONG" if is_in_pos else "FLAT"
    pos_size_mult = eth_s.position_size if is_in_pos else 0.0
    pos_value_usdt = pos_size_mult * equity_usdt if is_in_pos else 0.0

    # 3. Strategy Indicators computation
    strat = StructuralTrendPaperStrategy()
    service = PaperService()
    df = service.fetcher.fetch_closed_klines("ETHUSDT", interval=TIMEFRAME, limit=250, check_freshness=False)
    ind = strat.compute_indicators(df)

    curr_close = ind["close"]
    bb_upper = ind["bb_upper"]
    bb_mid = ind["bb_mid"]
    bb_lower = ind["bb_lower"]
    ema200 = ind["ema200"]
    atr14 = ind["atr"]
    swing_low = ind["swing_low"]
    macro_mult = ind["macro_mult"]

    # Trigger distances
    dist_to_buy_pct = ((bb_upper - curr_close) / curr_close) * 100.0
    dist_to_stop_pct = ((curr_close - eth_s.trailing_stop_price) / curr_close) * 100.0 if is_in_pos else 0.0

    # Read recent events
    journal = PaperJournal(JOURNAL_PATH)
    events = journal.read_all_events()
    trade_events = [e for e in events if e.get("event_type") == "POSITION_CLOSED"][-5:]
    signal_events = [e for e in events if e.get("event_type") == "SIGNAL_CREATED"][-5:]

    now_utc = datetime.now(timezone.utc)
    now_bjt = now_utc + timedelta(hours=8)

    last_bar_time = eth_s.last_processed_bar_time or str(df.index[-1])

    return {
        "timestamp_utc": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "timestamp_bjt": now_bjt.strftime("%Y-%m-%d %H:%M:%S BJT"),
        "experiment_id": EXPERIMENT_ID,
        "strategy_name": STRATEGY_NAME,
        "public_url": get_effective_public_url(),
        "capital": {
            "initial_usdt": round(INITIAL_CAPITAL_USDT, 2),
            "current_equity_usdt": round(equity_usdt, 2),
            "peak_usdt": round(peak_usdt, 2),
            "pnl_usdt": round(pnl_usdt, 2),
            "pnl_pct": round(pnl_pct, 2),
            "drawdown_pct": round(eth_s.drawdown * 100.0, 2),
            "total_fees_usdt": round(fees_usdt, 2),
            "normalized_equity": round(eth_s.total_equity, 6),
        },
        "position": {
            "symbol": "ETHUSDT",
            "direction": pos_direction,
            "size_mult": pos_size_mult,
            "nominal_usdt": round(pos_value_usdt, 2),
            "entry_price": round(eth_s.entry_price, 2) if is_in_pos else None,
            "entry_time": eth_s.entry_time,
            "trailing_stop": round(eth_s.trailing_stop_price, 2) if is_in_pos else None,
            "highest_price": round(eth_s.highest_price_since_entry, 2) if is_in_pos else None,
            "bars_in_pos": eth_s.bars_in_position,
            "dist_to_stop_pct": round(dist_to_stop_pct, 2) if is_in_pos else None,
            "pending_order": eth_s.pending_order,
        },
        "market": {
            "symbol": "ETHUSDT",
            "curr_price": round(curr_close, 2),
            "last_closed_bar": last_bar_time,
            "bb_upper": round(bb_upper, 2),
            "bb_mid": round(bb_mid, 2),
            "bb_lower": round(bb_lower, 2),
            "ema200": round(ema200, 2),
            "atr14": round(atr14, 2),
            "swing_low": round(swing_low, 2),
            "macro_status": "强多头顺势 (1.0x)" if macro_mult == 1.0 else "弱势/盘整防守 (0.5x)",
            "dist_to_buy_pct": round(dist_to_buy_pct, 2),
        },
        "trades_history": trade_events,
        "signals_history": signal_events,
        "system": {
            "code_commit": get_code_commit(),
            "config_hash": CONFIG_HASH,
            "paper_mode": PAPER_MODE,
            "enable_real_orders": ENABLE_REAL_ORDERS,
        },
    }


# ============================================================================
# EMAIL SENDER
# ============================================================================
def send_email_report(to_email: str = DEFAULT_RECIPIENT, is_manual: bool = False) -> bool:
    """Sends institutional HTML summary email with clickable public dashboard link."""
    sender_email = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASSWORD")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.qq.com")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))

    if not sender_email or not password:
        print("[EMAIL ERROR] Missing SMTP_USER or SMTP_PASSWORD in environment!")
        return False

    data = get_current_dashboard_data()
    cap = data["capital"]
    pos = data["position"]
    mkt = data["market"]
    pub_url = data["public_url"]

    subject_prefix = "[手动触发]" if is_manual else "[每日早报]"
    pnl_sign = "+" if cap["pnl_usdt"] >= 0 else ""
    subject = f"{subject_prefix} ETH 纯结构趋势监控 | 净值: ${cap['current_equity_usdt']:,.2f} ({pnl_sign}{cap['pnl_pct']}%)"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background-color: #0d1117; color: #c9d1d9; padding: 20px; }}
            .container {{ max-width: 650px; margin: 0 auto; background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 24px; }}
            .header {{ border-bottom: 1px solid #30363d; padding-bottom: 16px; margin-bottom: 20px; }}
            .title {{ font-size: 20px; font-weight: 700; color: #58a6ff; margin: 0; }}
            .badge {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; background: #238636; color: #ffffff; margin-top: 8px; }}
            .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 16px 0; }}
            .card {{ background: #21262d; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }}
            .card-title {{ font-size: 12px; color: #8b949e; text-transform: uppercase; margin-bottom: 6px; }}
            .card-value {{ font-size: 20px; font-weight: 700; color: #f0f6fc; }}
            .green {{ color: #3fb950; }}
            .red {{ color: #f85149; }}
            .btn {{ display: block; width: 100%; text-align: center; background: #1f6feb; color: #ffffff !important; padding: 14px 0; border-radius: 8px; font-weight: 700; text-decoration: none; font-size: 16px; margin: 24px 0 16px; }}
            .footer {{ font-size: 11px; color: #8b949e; border-top: 1px solid #30363d; padding-top: 12px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="title">⚡ ETH 纯结构趋势 Paper Signal 监控早报</div>
                <div class="badge">PAPER MODE (10,000 USDT 本金)</div>
                <div style="font-size: 12px; color: #8b949e; margin-top: 6px;">时间: {data['timestamp_bjt']}</div>
            </div>

            <a href="{pub_url}" class="btn" target="_blank">👉 点击打开实时在线监控网页 (随时刷新数据)</a>
            <div style="text-align: center; margin-top: -6px; margin-bottom: 20px; font-size: 13px; color: #8b949e;">
                外部公网访问直达地址: <a href="{pub_url}" target="_blank" style="color: #58a6ff; text-decoration: underline; word-break: break-all;">{pub_url}</a>
            </div>

            <div class="stat-grid">
                <div class="card">
                    <div class="card-title">当前账户净值 (USDT)</div>
                    <div class="card-value">${cap['current_equity_usdt']:,.2f}</div>
                    <div style="font-size: 12px; margin-top: 4px; color: {'#3fb950' if cap['pnl_usdt'] >= 0 else '#f85149'};">
                        累计盈亏: {pnl_sign}${cap['pnl_usdt']:,.2f} ({pnl_sign}{cap['pnl_pct']}%)
                    </div>
                </div>
                <div class="card">
                    <div class="card-title">ETH 当前持仓状态</div>
                    <div class="card-value" style="color: {'#3fb950' if pos['direction'] == 'LONG' else '#8b949e'};">
                        {pos['direction']} {f"({pos['size_mult']}x / ${pos['nominal_usdt']:,.0f})" if pos['direction'] == 'LONG' else '(100% 现金观望)'}
                    </div>
                    <div style="font-size: 12px; margin-top: 4px; color: #8b949e;">
                        最大回撤: {cap['drawdown_pct']}%
                    </div>
                </div>
            </div>

            <div class="card" style="margin-bottom: 16px;">
                <div class="card-title">关键指标与触发边界 (4h K线)</div>
                <table style="width: 100%; font-size: 13px; color: #c9d1d9; border-collapse: collapse;">
                    <tr><td style="padding: 4px 0;">当前 ETH 收盘价:</td><td style="text-align: right; font-weight: 700; color: #f0f6fc;">${mkt['curr_price']:,.2f}</td></tr>
                    <tr><td style="padding: 4px 0;">120 布林上轨 (突破买点):</td><td style="text-align: right; color: #58a6ff;">${mkt['bb_upper']:,.2f} (差 {mkt['dist_to_buy_pct']}%)</td></tr>
                    <tr><td style="padding: 4px 0;">120 布林中轨 (离场参考):</td><td style="text-align: right; color: #8b949e;">${mkt['bb_mid']:,.2f}</td></tr>
                    <tr><td style="padding: 4px 0;">EMA 200 宏观过滤均线:</td><td style="text-align: right; color: #d29922;">${mkt['ema200']:,.2f} ({mkt['macro_status']})</td></tr>
                    <tr><td style="padding: 4px 0;">ATR 14 波动率:</td><td style="text-align: right;">${mkt['atr14']:,.2f}</td></tr>
                    {f"<tr><td style='padding: 4px 0; color: #f85149;'>当前单调移动止损线:</td><td style='text-align: right; font-weight: 700; color: #f85149;'>${pos['trailing_stop']:,.2f} (距止损 {pos['dist_to_stop_pct']}%)</td></tr>" if pos['direction'] == 'LONG' else ""}
                </table>
            </div>

            <div class="footer">
                <div>免责声明: 本监控邮件仅为量化策略前向模拟交易测试 (Paper Only)，严禁作为投资建议。</div>
                <div>配置哈希: {data['system']['config_hash'][:16]}... | 源码 Commit: {data['system']['code_commit'][:8]}</div>
            </div>
        </div>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=15)
        server.login(sender_email, password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        print(f"[EMAIL SUCCESS] Report successfully sent to {to_email}")
        return True
    except Exception as e:
        print(f"[EMAIL FAILURE] Failed to send email to {to_email}: {e}")
        return False


# ============================================================================
# EMBEDDED MODERN HTML DASHBOARD TEMPLATE
# ============================================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETH 结构趋势 Paper Trading 实时监控看板 (10,000 USDT)</title>
    <style>
        :root {
            --bg-color: #090d16;
            --surface-color: #121826;
            --card-color: #1a2234;
            --border-color: #2a3449;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --accent-yellow: #f59e0b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background-color: var(--bg-color); color: var(--text-primary); padding: 24px; }
        .container { max-width: 1100px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; padding-bottom: 16px; border-bottom: 1px solid var(--border-color); flex-wrap: wrap; gap: 12px; }
        .brand { display: flex; align-items: center; gap: 12px; }
        .brand-icon { width: 36px; height: 36px; background: linear-gradient(135deg, #38bdf8, #2563eb); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 18px; color: white; }
        .brand-text h1 { font-size: 20px; font-weight: 700; color: var(--text-primary); }
        .brand-text p { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }
        .btn-refresh { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; border: none; padding: 10px 20px; border-radius: 8px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 8px; font-size: 14px; transition: all 0.2s; }
        .btn-refresh:hover { opacity: 0.9; transform: translateY(-1px); }
        .btn-refresh:disabled { opacity: 0.5; cursor: not-allowed; }
        .grid-3 { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-bottom: 20px; }
        .card { background-color: var(--surface-color); border: 1px solid var(--border-color); border-radius: 12px; padding: 20px; }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
        .card-title { font-size: 13px; font-weight: 600; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; }
        .metric-big { font-size: 28px; font-weight: 800; color: #ffffff; }
        .pill { padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 700; }
        .pill-green { background: rgba(34, 197, 94, 0.15); color: var(--accent-green); border: 1px solid rgba(34, 197, 94, 0.3); }
        .pill-red { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.3); }
        .pill-gray { background: rgba(148, 163, 184, 0.15); color: var(--text-secondary); border: 1px solid rgba(148, 163, 184, 0.3); }
        .row-item { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 13px; }
        .row-item:last-child { border-bottom: none; }
        .row-label { color: var(--text-secondary); }
        .row-val { font-weight: 600; color: var(--text-primary); }
        .banner-alert { background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 8px; padding: 12px 16px; font-size: 13px; color: var(--accent-blue); margin-bottom: 20px; display: flex; align-items: center; justify-content: space-between; }
        .footer { margin-top: 30px; text-align: center; font-size: 12px; color: var(--text-secondary); line-height: 1.6; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="brand">
                <div class="brand-icon">Ξ</div>
                <div class="brand-text">
                    <h1>ETH 纯结构趋势 Paper Trading 实时监控</h1>
                    <p>策略: structural_trend_v1 (4h 布林 120 + EMA200 + 3 ATR) | 本金规模: 10,000 USDT</p>
                </div>
            </div>
            <div>
                <button class="btn-refresh" id="refresh-btn" onclick="triggerRefresh()">
                    <span id="btn-text">🔄 立即刷新数据</span>
                </button>
            </div>
        </div>

        <div class="banner-alert" id="status-banner">
            <span>🟢 <b>系统正常运行中</b> | 最新 4h K线: <span id="last-bar-time">{{ data.market.last_closed_bar }}</span></span>
            <span style="font-size: 11px;">刷新时间: <span id="sync-time">{{ data.timestamp_bjt }}</span></span>
        </div>

        <div class="grid-3">
            <!-- 账户资产卡片 -->
            <div class="card">
                <div class="card-header">
                    <span class="card-title">账户净值总览</span>
                    <span class="pill {{ 'pill-green' if data.capital.pnl_usdt >= 0 else 'pill-red' }}">
                        {{ '+' if data.capital.pnl_usdt >= 0 else '' }}{{ data.capital.pnl_pct }}%
                    </span>
                </div>
                <div class="metric-big" id="equity-val">${{ "{:,.2f}".format(data.capital.current_equity_usdt) }}</div>
                <div style="font-size: 13px; color: var(--text-secondary); margin-top: 6px; margin-bottom: 16px;">
                    净收益: <b style="color: {{ '#22c55e' if data.capital.pnl_usdt >= 0 else '#ef4444' }}">${{ "{:+,.2f}".format(data.capital.pnl_usdt) }} USDT</b>
                </div>
                <div class="row-item"><span class="row-label">初始基准本金:</span><span class="row-val">$10,000.00 USDT</span></div>
                <div class="row-item"><span class="row-label">历史最高净值:</span><span class="row-val">${{ "{:,.2f}".format(data.capital.peak_usdt) }}</span></div>
                <div class="row-item"><span class="row-label">当前回撤:</span><span class="row-val">{{ data.capital.drawdown_pct }}%</span></div>
                <div class="row-item"><span class="row-label">累计交易摩擦:</span><span class="row-val">${{ "{:,.2f}".format(data.capital.total_fees_usdt) }} (8 bps)</span></div>
            </div>

            <!-- 当前持仓卡片 -->
            <div class="card">
                <div class="card-header">
                    <span class="card-title">ETH 当前持仓状态</span>
                    <span class="pill {{ 'pill-green' if data.position.direction == 'LONG' else 'pill-gray' }}">
                        {{ data.position.direction }}
                    </span>
                </div>
                <div class="metric-big" style="color: {{ '#22c55e' if data.position.direction == 'LONG' else '#94a3b8' }};">
                    {{ data.position.direction }} {{ f"({data.position.size_mult}x)" if data.position.direction == 'LONG' else "(100% 现金)" }}
                </div>
                <div style="font-size: 13px; color: var(--text-secondary); margin-top: 6px; margin-bottom: 16px;">
                    当前持仓市值: <b>${{ "{:,.2f}".format(data.position.nominal_usdt) }} USDT</b>
                </div>
                <div class="row-item"><span class="row-label">入场成交价:</span><span class="row-val">{{ f"${data.position.entry_price:,.2f}" if data.position.entry_price else "—" }}</span></div>
                <div class="row-item"><span class="row-label">入场时间:</span><span class="row-val">{{ data.position.entry_time or "—" }}</span></div>
                <div class="row-item"><span class="row-label">单调跟踪止损位:</span><span class="row-val" style="color: #ef4444;">{{ f"${data.position.trailing_stop:,.2f}" if data.position.trailing_stop else "—" }}</span></div>
                <div class="row-item"><span class="row-label">持仓 K 线数:</span><span class="row-val">{{ data.position.bars_in_pos }} 根 ({{ data.position.bars_in_pos * 4 }} 小时)</span></div>
            </div>

            <!-- 市场指标诊断卡片 -->
            <div class="card">
                <div class="card-header">
                    <span class="card-title">4h 结构指标与触发线</span>
                    <span class="pill pill-gray">ETH: ${{ "{:,.2f}".format(data.market.curr_price) }}</span>
                </div>
                <div class="row-item"><span class="row-label">120 布林上轨 (突破买点):</span><span class="row-val" style="color: var(--accent-blue);">${{ "{:,.2f}".format(data.market.bb_upper) }}</span></div>
                <div class="row-item"><span class="row-label">距突破买点差距:</span><span class="row-val">+{{ data.market.dist_to_buy_pct }}%</span></div>
                <div class="row-item"><span class="row-label">120 布林中轨 (离场底线):</span><span class="row-val">${{ "{:,.2f}".format(data.market.bb_mid) }}</span></div>
                <div class="row-item"><span class="row-label">EMA 200 宏观过滤均线:</span><span class="row-val" style="color: var(--accent-yellow);">${{ "{:,.2f}".format(data.market.ema200) }}</span></div>
                <div class="row-item"><span class="row-label">宏观状态乘数:</span><span class="row-val">{{ data.market.macro_status }}</span></div>
                <div class="row-item"><span class="row-label">ATR 14 波动率:</span><span class="row-val">${{ "{:,.2f}".format(data.market.atr14) }}</span></div>
            </div>
        </div>

        <div class="footer">
            <p>🛡️ <b>硬性安全声明</b>: 本系统运行于 <code>PAPER_MODE=True</code>，严格禁止且无法下达任何真实订单（<code>ENABLE_REAL_ORDERS=False</code>）。杠杆严格限制为 1.0x。</p>
            <p>每天早间 08:00 (BJT) 自动从币安公共接口拉取最新关闭 K 线，计算最新指标并推送外部访问链接至 <code>568701293@qq.com</code>。</p>
            <p style="font-family: monospace; font-size: 11px; margin-top: 6px;">Config Hash: {{ data.system.config_hash[:16] }}... | Commit: {{ data.system.code_commit[:8] }}</p>
        </div>
    </div>

    <script>
        async function triggerRefresh() {
            const btn = document.getElementById('refresh-btn');
            const text = document.getElementById('btn-text');
            btn.disabled = true;
            text.innerText = "⏳ 正在从币安获取最新 K 线并重新计算...";

            try {
                const resp = await fetch('/api/refresh', { method: 'POST' });
                const res = await resp.json();
                if (res.success) {
                    location.reload();
                } else {
                    alert('刷新失败: ' + res.error);
                }
            } catch (err) {
                alert('网络通信异常: ' + err);
            } finally {
                btn.disabled = false;
                text.innerText = "🔄 立即刷新数据";
            }
        }

        // Auto-refresh page every 60 seconds
        setInterval(() => {
            fetch('/api/status').then(r => r.json()).then(data => {
                document.getElementById('sync-time').innerText = data.timestamp_bjt;
            }).catch(() => {});
        }, 60000);
    </script>
</body>
</html>
"""


# ============================================================================
# FLASK ROUTES
# ============================================================================
@app.route("/")
def index():
    data = get_current_dashboard_data()
    return render_template_string(HTML_TEMPLATE, data=data)


@app.route("/api/status")
def api_status():
    return jsonify(get_current_dashboard_data())


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    with service_lock:
        try:
            service = PaperService()
            service.run_once()
            data = get_current_dashboard_data()
            return jsonify({"success": True, "data": data})
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/send_email", methods=["POST"])
def api_send_email():
    """Manual endpoint to test sending report email."""
    req_data = request.get_json(silent=True) or {}
    recipient = req_data.get("email", DEFAULT_RECIPIENT)
    success = send_email_report(to_email=recipient, is_manual=True)
    return jsonify({"success": success, "recipient": recipient})


# ============================================================================
# BACKGROUND SCHEDULER: DAILY MORNING 08:02 REFRESH & EMAIL PUSH
# ============================================================================
def background_morning_scheduler():
    """
    Background worker that runs daily at 08:02 Beijing Time (00:02 UTC).
    1. Triggers data refresh.
    2. Sends email with public URL to 568701293@qq.com.
    """
    last_sent_date = None
    print("[SCHEDULER] Daily morning scheduler initialized (Target: 08:02 BJT / 00:02 UTC)...")

    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            now_bjt = now_utc + timedelta(hours=8)
            current_date_str = now_bjt.strftime("%Y-%m-%d")

            # Trigger at 08:00 - 08:05 BJT once per day
            if now_bjt.hour == 8 and 0 <= now_bjt.minute <= 5:
                if last_sent_date != current_date_str:
                    print(f"[SCHEDULER] Morning trigger fired at {now_bjt.strftime('%Y-%m-%d %H:%M:%S')} BJT!")
                    with service_lock:
                        service = PaperService()
                        service.run_once()
                    send_email_report(to_email=DEFAULT_RECIPIENT, is_manual=False)
                    last_sent_date = current_date_str
        except Exception as e:
            print(f"[SCHEDULER ERROR] {e}")

        time_lib.sleep(30)


def start_server(port: int = PORT, public_url: Optional[str] = None):
    """Starts the Flask dashboard server and background morning scheduler."""
    global PUBLIC_TUNNEL_URL
    if public_url:
        PUBLIC_TUNNEL_URL = public_url

    # Start background scheduler thread
    scheduler_thread = threading.Thread(target=background_morning_scheduler, daemon=True)
    scheduler_thread.start()

    print(f"[DASHBOARD] Starting Paper Dashboard on port {port} (Public URL: {PUBLIC_TUNNEL_URL})...")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    start_server()
