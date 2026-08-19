# -*- coding: utf-8 -*-
"""邮件通知模块 (复用项目 .env 中的 SMTP 配置)

SMTP_USER / SMTP_PASSWORD / RECEIVER_EMAIL / SMTP_SERVER / SMTP_PORT
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))
except Exception:
    pass


def send_email_html(subject, html_body):
    sender = os.getenv("SMTP_USER")
    receiver = os.getenv("RECEIVER_EMAIL")
    password = os.getenv("SMTP_PASSWORD")
    server = os.getenv("SMTP_SERVER", "smtp.qq.com")
    port = int(os.getenv("SMTP_PORT", "465"))

    if not all([sender, receiver, password]):
        print("[notify] SMTP 配置缺失 (SMTP_USER/RECEIVER_EMAIL/SMTP_PASSWORD 需在 .env)")
        return False

    msg = MIMEMultipart("alternative")
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html", "utf-8"))
    try:
        s = smtplib.SMTP_SSL(server, port, timeout=30)
        s.login(sender, password)
        s.sendmail(sender, receiver, msg.as_string())
        s.quit()
        print(f"[notify] 邮件已成功发送 -> {receiver}")
        return True
    except Exception as e:
        print(f"[notify] 邮件发送失败: {e}")
        return False


def build_signal_email_html(sig):
    """根据今日结构化信号构建高颜值 HTML 邮件正文"""
    macro = sig.get("macro_timing", {})
    risk = sig.get("risk_control", {})
    hedge = sig.get("im_futures_hedge", {})
    alloc = sig.get("allocation_summary", {})
    picks = sig.get("holdings_picks", [])
    defensive = sig.get("defensive_picks", [])
    
    score = macro.get("s123_total_score", 0)
    if score >= 3:
        status_color = "#10b981"
        status_tag = "S3 · 满仓进攻 100%"
    elif score == 2:
        status_color = "#f59e0b"
        status_tag = "S2 · 均衡配置 50%"
    else:
        status_color = "#ef4444"
        status_tag = "S1/S0 · 防守避险 0%"

    stock_w_pct = alloc.get("stock_exposure_pct", "0.0%")
    def_w_pct = alloc.get("defensive_exposure_pct", "100.0%")

    # 构建持仓表格行 (前 15 只展示，避免邮件过长)
    rows_html = ""
    for p in picks[:15]:
        rows_html += f"""
        <tr style="border-bottom:1px solid #e2e8f0;">
          <td style="padding:8px;text-align:center;color:#64748b;">{p.get('rank')}</td>
          <td style="padding:8px;font-family:monospace;font-weight:bold;color:#2563eb;">{p.get('ts_code')}</td>
          <td style="padding:8px;font-weight:600;color:#1e293b;">{p.get('name')}</td>
          <td style="padding:8px;color:#475569;">{p.get('industry')}</td>
          <td style="padding:8px;text-align:right;font-family:monospace;font-weight:bold;color:#10b981;">{p.get('target_weight', 0)*100:.2f}%</td>
          <td style="padding:8px;text-align:right;font-family:monospace;">¥{p.get('target_amount', 0):,.0f}</td>
        </tr>
        """

    def_rows_html = ""
    for d in defensive:
        def_rows_html += f"""
        <tr style="border-bottom:1px solid #e2e8f0;">
          <td style="padding:8px;font-family:monospace;font-weight:bold;color:#2563eb;">{d.get('ts_code')}</td>
          <td style="padding:8px;font-weight:600;color:#1e293b;">{d.get('name')}</td>
          <td style="padding:8px;color:#475569;">{d.get('category')}</td>
          <td style="padding:8px;text-align:right;font-family:monospace;font-weight:bold;color:#10b981;">{d.get('target_weight', 0)*100:.2f}%</td>
          <td style="padding:8px;text-align:right;font-family:monospace;">¥{d.get('target_amount', 0):,.0f}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;background-color:#f1f5f9;margin:0;padding:24px;">
      <div style="max-width:680px;margin:0 auto;background:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 4px 16px rgba(0,0,0,0.06);border:1px solid #e2e8f0;">
        
        <!-- Header -->
        <div style="background:linear-gradient(135deg, #1e293b 0%, #0f172a 100%);padding:24px 28px;color:#ffffff;">
          <div style="display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;background:rgba(59,130,246,0.2);color:#60a5fa;margin-bottom:8px;">
            ⚡ A股综合复合量化策略 · 每日晨报
          </div>
          <h1 style="margin:0;font-size:22px;font-weight:700;letter-spacing:-0.3px;">今日调仓决策与信号清单</h1>
          <div style="margin-top:6px;font-size:13px;color:#94a3b8;">
            信号日期: <strong>{sig.get('signal_date')}</strong> (早晨 07:00 自动生成) | 调仓基准: {sig.get('rebalance_date')}
          </div>
        </div>

        <div style="padding:24px 28px;">
          
          <!-- Action Banner -->
          <div style="background:#f8fafc;border-left:4px solid {status_color};border-radius:6px;padding:16px 20px;margin-bottom:20px;">
            <div style="font-size:13px;color:#64748b;font-weight:600;text-transform:uppercase;">今日核心操作指令</div>
            <div style="font-size:24px;font-weight:800;color:{status_color};margin:6px 0;">{status_tag}</div>
            <div style="font-size:14px;color:#334155;line-height:1.5;">
              • 股票端目标仓位: <strong>{stock_w_pct}</strong> (全市场 Top 40 分散选股)<br>
              • 稳健避险资金池: <strong>{def_w_pct}</strong> (V8 短债/信用债/黄金等权)<br>
              • IM 股指期货对冲建议: <strong>{hedge.get('recommended_lots', 0):.1f} 手</strong> (对冲面值约 {hedge.get('target_hedge_notional', 0):,.0f} 元)
            </div>
          </div>

          <!-- Macro Status -->
          <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px 18px;margin-bottom:20px;font-size:13px;color:#334155;">
            <strong style="color:#0f172a;">📊 S123 宏观估值状态指标：</strong><br>
            - S1 (PE 分位 < 20%): {'✅ 满足' if macro.get('s1_pe_low') else '❌ 未满足'}<br>
            - S2 (10Y 国债 ERP > 1σ): {'✅ 满足' if macro.get('s2_erp_high') else '❌ 未满足'}<br>
            - S3 (自高点回撤 ≤ -25%): {'✅ 满足' if macro.get('s3_dd_deep') else '❌ 未满足'}<br>
            - 组合净值回撤熔断: {'⚠️ 已触发 (-10%降档)' if risk.get('is_degraded') else '🟢 正常 (未触发)'}
          </div>

          <!-- Defensive Holdings (if any) -->
          {f'''
          <div style="margin-bottom:24px;">
            <h3 style="font-size:15px;color:#0f172a;margin-bottom:10px;">🛡️ V8 避险资金配置清单 ({def_w_pct})</h3>
            <table style="width:100%;border-collapse:collapse;font-size:12.5px;">
              <thead>
                <tr style="background:#f1f5f9;color:#475569;">
                  <th style="padding:6px 8px;text-align:left;">代码</th>
                  <th style="padding:6px 8px;text-align:left;">资产名称</th>
                  <th style="padding:6px 8px;text-align:left;">类别</th>
                  <th style="padding:6px 8px;text-align:right;">权重</th>
                  <th style="padding:6px 8px;text-align:right;">配置金额</th>
                </tr>
              </thead>
              <tbody>{def_rows_html}</tbody>
            </table>
          </div>
          ''' if defensive else ''}

          <!-- Stock Picks -->
          <div>
            <h3 style="font-size:15px;color:#0f172a;margin-bottom:10px;">📈 股票端 ENS Top 40 目标持仓 (前 15 只预览)</h3>
            <table style="width:100%;border-collapse:collapse;font-size:12.5px;">
              <thead>
                <tr style="background:#f1f5f9;color:#475569;">
                  <th style="padding:6px 8px;text-align:center;">#</th>
                  <th style="padding:6px 8px;text-align:left;">代码</th>
                  <th style="padding:6px 8px;text-align:left;">名称</th>
                  <th style="padding:6px 8px;text-align:left;">细分行业</th>
                  <th style="padding:6px 8px;text-align:right;">权重</th>
                  <th style="padding:6px 8px;text-align:right;">建议金额</th>
                </tr>
              </thead>
              <tbody>{rows_html if rows_html else '<tr><td colspan="6" style="padding:16px;text-align:center;color:#94a3b8;">当前为防守避险段，建议空仓股票并持有 V8 稳健资产。</td></tr>'}</tbody>
            </table>
          </div>

          <!-- Footer Link -->
          <div style="margin-top:28px;padding-top:16px;border-top:1px solid #e2e8f0;text-align:center;font-size:12px;color:#94a3b8;">
            <a href="http://127.0.0.1:8000" style="color:#2563eb;text-decoration:none;font-weight:600;">👉 点击访问本地实时监控仪表盘 (http://127.0.0.1:8000)</a><br>
            <span style="display:inline-block;margin-top:6px;">策略服务由 Antigravity Quant Engine 驱动</span>
          </div>

        </div>
      </div>
    </body>
    </html>
    """
    return html


if __name__ == "__main__":
    test_sig = {
        "signal_date": "2026-08-19",
        "rebalance_date": "2026-08-03",
        "macro_timing": {"s123_total_score": 1, "s1_pe_low": 0, "s2_erp_high": 1, "s3_dd_deep": 0},
        "risk_control": {"is_degraded": False, "target_stock_exposure": 0.0, "target_defensive_exposure": 1.0},
        "allocation_summary": {"stock_exposure_pct": "0.0%", "defensive_exposure_pct": "100.0%"},
        "defensive_picks": [
            {"ts_code": "511990.SH", "name": "华宝添益ETF", "category": "货币/流动性", "target_weight": 0.3333, "target_amount": 333333},
            {"ts_code": "511260.SH", "name": "十年国债ETF", "category": "债券避险", "target_weight": 0.3333, "target_amount": 333333},
            {"ts_code": "518880.SH", "name": "黄金ETF", "category": "大宗商品", "target_weight": 0.3333, "target_amount": 333334}
        ],
        "holdings_picks": []
    }
    body = build_signal_email_html(test_sig)
    send_email_html("【量化策略晨报】2026-08-19 操作决策与持仓清单", body)
