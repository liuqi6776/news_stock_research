"""
daily_morning_pipeline.py
每日晨间 8:00 A股量化策略调度与邮件推送主程序。
流程：
1. 增量抓取最新期权数据并同步。
2. 抓取最新的韭研公社盘前热点纪要网页，调用 GLM-4 进行 NLP 解析并存入 news_major1 目录。
3. 读取最新交易日（T-1日，如周五收盘后数据）对应的量化预测数据。
4. 解析出 [筛选前 Top 10]、[筛选后正式买入推荐] 与 [筛选规则说明]。
5. 自动抓取最新期权 Z-Score 与 PCR 指标，进行大盘健康评估诊断。
6. 拼装成极其精美的机构级 HTML 量化晨报，并发送至 568701293@qq.com 邮箱。
"""
import os
import sys
import json
import smtplib
import pandas as pd
import numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Set path environment
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv
load_dotenv()

def sync_all_data():
    """Step 1 & 2: 同步最新的期权与盘前新闻数据"""
    print("[INFO] Syncing options data...")
    try:
        from data.update_options_data import main as sync_options
        sync_options()
    except Exception as e:
        print(f"[WARNING] Options sync failed: {e}")

    print("[INFO] Scraping latest pre-market news...")
    try:
        jiayo_path = os.path.join(ROOT_DIR, "jiayo-analysis")
        if jiayo_path not in sys.path:
            sys.path.insert(0, jiayo_path)
        from quick_sync_news import sync_latest_news
        sync_latest_news()
    except Exception as e:
        print(f"[WARNING] News scrape failed: {e}")

def get_latest_pcr_indicators():
    """获取最新交易日的期权指标"""
    pcr_csv = r"D:\iquant_data\data_v2\qiquan\historical_pcr.csv"
    if not os.path.exists(pcr_csv):
        return None
    try:
        df = pd.read_csv(pcr_csv)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        last_row = df.iloc[-1]
        
        # We can also load QVIX live to get Z-score
        import akshare as ak
        df_qvix = ak.index_option_50etf_qvix()
        df_qvix['ma'] = df_qvix['close'].rolling(20).mean()
        df_qvix['std'] = df_qvix['close'].rolling(20).std()
        df_qvix['zscore'] = (df_qvix['close'] - df_qvix['ma']) / df_qvix['std']
        
        return {
            'date': last_row['date'].strftime('%Y-%m-%d'),
            'qvix': float(df_qvix['close'].iloc[-1]),
            'qvix_zscore': float(df_qvix['zscore'].iloc[-1]),
            'pcr_50': float(last_row['pcr_50']),
            'oi_pcr_50': float(last_row['oi_pcr_50']),
            'pcr_300': float(last_row['pcr_300']),
            'oi_pcr_300': float(last_row['oi_pcr_300']),
        }
    except Exception as e:
        print(f"[WARNING] Failed to load latest option indicators: {e}")
        return None

def build_html_report(date_str, rules, filtered_sigs, top10_raw, opt):
    """拼装机构级的 HTML 邮件正文"""
    # Options Health Audit
    health_status = "STABLE / NORMAL"
    health_color = "#2E7D32" # Green
    if opt:
        if opt['qvix_zscore'] >= 2.0:
            health_status = "Oversold Spike - Rebound Signal Triggered!"
            health_color = "#D32F2F" # Red (rebound buy opportunity)
        elif opt['qvix_zscore'] <= -1.5:
            health_status = "Extreme Complacency - Watch out for Pullbacks"
            health_color = "#ED6C02" # Orange
        elif opt['pcr_50'] >= 1.09:
            health_status = "Panic Extreme - contrarian Rebound Zone"
            health_color = "#1976D2" # Blue
            
    opt_html = ""
    if opt:
        opt_html = f"""
        <table class="metrics-table">
            <tr>
                <th>QVIX (恐慌指数)</th>
                <td>{opt['qvix']:.2f}</td>
                <th>QVIX Z-Score</th>
                <td><strong style="color: {health_color}">{opt['qvix_zscore']:.2f}</strong></td>
            </tr>
            <tr>
                <th>50ETF PCR (成交量)</th>
                <td>{opt['pcr_50']:.2f}</td>
                <th>50ETF PCR (持仓量)</th>
                <td>{opt['oi_pcr_50']:.2f}</td>
            </tr>
            <tr>
                <th>300ETF PCR (成交量)</th>
                <td>{opt['pcr_300']:.2f}</td>
                <th>300ETF PCR (持仓量)</th>
                <td>{opt['oi_pcr_300']:.2f}</td>
            </tr>
        </table>
        <p><strong>大盘情绪健康诊断</strong>: <span style="background-color: {health_color}1a; color: {health_color}; padding: 4px 8px; border-radius: 4px; font-weight: bold;">{health_status}</span></p>
        """
    else:
        opt_html = "<p>暂无最新期权大盘指标数据。</p>"

    # Filtered Signals HTML
    filtered_html = ""
    if not filtered_sigs:
        filtered_html = """
        <div class="empty-box">
            今日无满足硬性防御规则的选股信号。系统自动保持空仓防御或进行板块避险。
        </div>
        """
    else:
        filtered_html = """
        <table class="signals-table">
            <thead>
                <tr>
                    <th>标的代码</th>
                    <th>上涨期望得分 (Score)</th>
                    <th>开盘入场参考价 (Entry)</th>
                    <th>操作指令</th>
                    <th>止损 / 止盈</th>
                </tr>
            </thead>
            <tbody>
        """
        for sig in filtered_sigs:
            entry_p = f"¥{sig['entry_price']:.2f}" if sig['entry_price'] else "开盘市价"
            filtered_html += f"""
                <tr>
                    <td><strong style="color: #1976D2;">{sig['ts_code']}</strong></td>
                    <td class="score-cell">{sig['prob']:.4f}</td>
                    <td>{entry_p}</td>
                    <td>{sig['action']}</td>
                    <td><span style="color: #D32F2F;">{sig['stop_loss']}</span> / <span style="color: #2E7D32;">{sig['take_profit']}</span></td>
                </tr>
            """
        filtered_html += "</tbody></table>"

    # Top 10 Raw HTML
    top10_html = """
    <table class="raw-table">
        <thead>
            <tr>
                <th>全市场排序</th>
                <th>标的代码</th>
                <th>基础得分</th>
                <th>参考买入价</th>
                <th>板块与风控参数详情</th>
            </tr>
        </thead>
        <tbody>
    """
    for item in top10_raw:
        ep = f"¥{item['entry_price']:.2f}" if item['entry_price'] else "开盘市价"
        top10_html += f"""
            <tr>
                <td class="rank-cell">No.{item['rank']}</td>
                <td><strong>{item['ts_code']}</strong></td>
                <td>{item['prob']:.4f}</td>
                <td>{ep}</td>
                <td style="color: #555; font-size: 13px;">{item['info']}</td>
            </tr>
        """
    top10_html += "</tbody></table>"

    # Rules HTML format
    rules_html = "<ul>" + "".join([f"<li>{r.strip()}</li>" for r in rules.split('\n') if r.strip()]) + "</ul>"

    # Main CSS Template
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; line-height: 1.6; color: #333; margin: 0; padding: 20px; background-color: #F4F6F9; }}
            .container {{ max-width: 800px; margin: 0 auto; background-color: #FFFFFF; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); overflow: hidden; border: 1px solid #E0E4EC; }}
            .header {{ background: linear-gradient(135deg, #1A237E, #0D47A1); color: #FFFFFF; padding: 25px; text-align: center; }}
            .header h1 {{ margin: 0; font-size: 24px; font-weight: bold; letter-spacing: 1px; }}
            .header p {{ margin: 5px 0 0 0; opacity: 0.8; font-size: 14px; }}
            .content {{ padding: 25px; }}
            h2 {{ color: #1A237E; font-size: 18px; border-left: 4px solid #1E88E5; padding-left: 10px; margin-top: 30px; margin-bottom: 15px; }}
            .metrics-table {{ width: 100%; border-collapse: collapse; margin-bottom: 15px; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #E0E4EC; padding: 10px; text-align: left; font-size: 14px; }}
            .metrics-table th {{ background-color: #F8F9FA; color: #555; width: 25%; font-weight: 600; }}
            .metrics-table td {{ width: 25%; }}
            .signals-table, .raw-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            .signals-table th, .signals-table td, .raw-table th, .raw-table td {{ border: 1px solid #E2E8F0; padding: 12px; text-align: left; }}
            .signals-table th {{ background-color: #E3F2FD; color: #0D47A1; font-weight: bold; }}
            .raw-table th {{ background-color: #F5F5F5; color: #333; font-weight: bold; }}
            .score-cell {{ font-weight: bold; color: #2E7D32; }}
            .rank-cell {{ font-weight: bold; color: #E65100; }}
            .empty-box {{ background-color: #FFF9C4; border: 1px dashed #FBC02D; color: #F57F17; padding: 15px; border-radius: 4px; text-align: center; font-weight: 500; }}
            .footer {{ background-color: #F8F9FA; padding: 20px; text-align: center; font-size: 12px; color: #888; border-top: 1px solid #E0E4EC; }}
            ul {{ padding-left: 20px; color: #555; font-size: 14px; }}
            li {{ margin-bottom: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>A股量化择时晨报（期权大局观版）</h1>
                <p>预测分析基准日: {date_str} | 信号执行建议日: {datetime.now().strftime('%Y-%m-%d')} 盘前</p>
            </div>
            <div class="content">
                <h2>📈 大盘健康诊断与期权隐含波动率</h2>
                {opt_html}

                <h2>🎯 本日推荐买入标的（XGBoost 期权筛选后）</h2>
                {filtered_html}

                <h2>📊 全行业期望值排行 Top 10（筛选前）</h2>
                {top10_html}

                <h2>🛠️ 本日量化过滤与中性化筛选规则</h2>
                {rules_html}
            </div>
            <div class="footer">
                <p>本晨报由 Antigravity 智能量化交易系统每日晨间 8:00 自动运算并生成。</p>
                <p><strong>风险提示</strong>：策略选股模型与期权择时结果仅供参考，不作为正式投资建议。股市有风险，投资需谨慎。</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def is_today_trading_day():
    """判断今天是否为A股开盘日"""
    now = datetime.now()
    # 周六、周日直接排除
    if now.weekday() in (5, 6):
        print(f"[INFO] Today ({now.strftime('%Y-%m-%d')}) is weekend. Skip prediction & email on non-trading days.")
        return False

    # 用 akshare 获取 A 股交易日历进行精准过滤
    import akshare as ak
    try:
        df_cal = ak.tool_trade_date_hist_sina()
        cal_dates = set(df_cal['trade_date'].astype(str).str.replace('-', '').tolist())
        today_str = now.strftime('%Y%m%d')
        if today_str in cal_dates:
            return True
        else:
            print(f"[INFO] Today ({now.strftime('%Y-%m-%d')}) is a holiday/non-trading day. Skip prediction & email.")
            return False
    except Exception as e:
        print(f"[WARNING] Failed to fetch trading calendar: {e}. Defaulting to allowing weekdays.")
        return True

def get_t_minus_1_date():
    import akshare as ak
    from datetime import datetime
    now = datetime.now()
    try:
        df_cal = ak.tool_trade_date_hist_sina()
        today_str = now.strftime('%Y-%m-%d')
        df_cal['trade_date'] = pd.to_datetime(df_cal['trade_date']).dt.strftime('%Y-%m-%d')
        trading_dates = sorted(df_cal['trade_date'].tolist())
        
        prior_dates = [d for d in trading_dates if d < today_str]
        if prior_dates:
            return prior_dates[-1].replace('-', '')
    except Exception as e:
        print(f"[WARNING] Failed to fetch T-1 date from Sina: {e}")
    
    from datetime import timedelta
    target = now - timedelta(days=1)
    if target.weekday() == 5:
        target = target - timedelta(days=1)
    elif target.weekday() == 6:
        target = target - timedelta(days=2)
    return target.strftime('%Y%m%d')

def get_stock_name_map():
    """获取本地缓存或在线拉取的股票 Ticker -> Name 映射表"""
    cache_path = os.path.join(ROOT_DIR, "stock_name_map.parquet")
    if os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return dict(zip(df['ts_code'], df['name']))
        except Exception:
            pass

    print("[INFO] Fetching stock list for Ticker-to-Name mapping...")
    try:
        from infra_data.fetcher import DataFetcher
        fetcher = DataFetcher()
        df_basic = fetcher.get_stock_list()
        if df_basic is not None and not df_basic.empty:
            df_basic = df_basic[['ts_code', 'name']].drop_duplicates()
            df_basic.to_parquet(cache_path)
            print(f"[SUCCESS] Ticker-to-Name mapping cached to {cache_path}")
            return dict(zip(df_basic['ts_code'], df_basic['name']))
    except Exception as e:
        print(f"[WARNING] Failed to fetch online stock list mapping: {e}")
        
    return {}

def pre_trade_audit(row):
    """Pre-Trade Audit: 过滤面值退市股、筹码冻结股与极端崩盘异常股"""
    # 1. 过滤低于 1.5 元的仙股（避免面值退市风险）
    if row.get('close', 0) < 1.5:
        return False, "股价低于1.5元(仙股/退市风险)"
    # 2. 过滤单日暴跌幅度超过 11% 的异常股（如异常复牌或严重踩雷）
    if row.get('pct_chg', 0) < -11.0:
        return False, f"单日异常暴跌 {row.get('pct_chg', 0):.2f}%"
    # 3. 过滤每日流动性过低的股票（成交额低于 500 万人民币）
    if row.get('amount', 0) < 5000:
        return False, "成交额低于 500 万元(流动性匮乏)"
    return True, "通过审核"

def run_pipeline_and_send_email():
    print(f"=== Starting Daily Pipeline Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # 0. Check if today is a trading day
    if not is_today_trading_day():
        print("[INFO] Today is a non-trading day. Exiting pipeline early without prediction or email.")
        return
        
    # 1. Sync PCR & News dynamically
    sync_all_data()

    # 1.5. Calculate T-1 date & fetch raw data
    t_minus_1_date = get_t_minus_1_date()
    print(f"[INFO] T-1 trading date determined: {t_minus_1_date}")
    
    print(f"[INFO] Fetching daily raw data for {t_minus_1_date}...")
    try:
        import subprocess
        subprocess.run([sys.executable, "fetch_latest_data.py", t_minus_1_date], check=True)
        print(f"[SUCCESS] Raw data fetched successfully.")
    except Exception as e:
        print(f"[WARNING] Failed to fetch daily raw data: {e}")

    # 2. Run real-time prediction
    print(f"[INFO] Running real-time daily dragon predictions for {t_minus_1_date}...")
    try:
        from daily_dragon_predict import predict_for_date
        picks_top3, picks_top10 = predict_for_date(t_minus_1_date)
        
        if picks_top10 is None or picks_top10.empty:
            print("[ERROR] Prediction failed or returned empty results.")
            return

        # 获取股票名称映射表
        name_map = get_stock_name_map()

        filtered_sigs = []
        top10_raw = []

        # 3. Apply Pre-Trade Audit / Double-Model Shield Filter
        idx = 1
        for _, row in picks_top10.iterrows():
            passed, reason = pre_trade_audit(row)
            ts_code = row["ts_code"]
            name = name_map.get(ts_code, "未知名称")
            display_name = f"{ts_code} ({name})"
            
            # Format top10_raw
            top10_raw.append({
                "rank":        idx,
                "ts_code":     display_name,
                "prob":        round(float(row["prob"]), 4),
                "entry_price": round(float(row["close"]), 2) if pd.notna(row["close"]) else None,
                "info":        f"昨日涨跌: {row['pct_chg']:.2f}% | 流通市值: {row['circ_mv']/10000:.2f}亿 | 舆情权重: {row['news_stock_impact']:.1f} | 审计: {reason}",
            })
            
            # If passed and we haven't reached max positions, add to filtered signals
            if passed and len(filtered_sigs) < 3:
                filtered_sigs.append({
                    "ts_code":     display_name,
                    "prob":        round(float(row["prob"]), 4),
                    "entry_price": round(float(row["close"]), 2) if pd.notna(row["close"]) else None,
                    "action":      f"T+1 以开盘价买入 | 昨收 ¥{row['close']:.2f} ({row['pct_chg']:.2f}%) | 舆情评分: {row['news_stock_impact']:.1f}",
                    "stop_loss":   "-5%",
                    "take_profit": "+6% 固定止盈"
                })
            idx += 1

        rules = (
            "1. 筛选基础: XGBoost 实时上涨概率高分排序\n"
            "2. 审计过滤: 排除股价 < ¥1.5 的面值退市风险股，排除单日暴跌 > 11% 异常复牌股\n"
            "3. 仓位控制: 每日最多买入推荐前 3 只股票 (Max Positions = 3)\n"
            "4. 风控管理: 严格执行 -5% 止损 与 +6% 固定止盈离场策略"
        )
        
        # Get options indicators
        opt = get_latest_pcr_indicators()
        
        # Generate HTML content
        html_body = build_html_report(
            date_str=f"{t_minus_1_date[:4]}-{t_minus_1_date[4:6]}-{t_minus_1_date[6:8]}",
            rules=rules,
            filtered_sigs=filtered_sigs,
            top10_raw=top10_raw,
            opt=opt
        )
        
        # Send Email
        sender_email = os.getenv("SMTP_USER", "568701293@qq.com") # Default to user's
        receiver_email = "568701293@qq.com"
        password = os.getenv("SMTP_PASSWORD")
        smtp_server = os.getenv("SMTP_SERVER", "smtp.qq.com")
        smtp_port = int(os.getenv("SMTP_PORT", "465"))

        if not password:
            print("[ERROR] SMTP_PASSWORD is not configured in .env!")
            print("Please add 'SMTP_PASSWORD=<your_qq_mail_authorization_code>' to your .env file.")
            return

        subject = f"【量化晨报】A股期权增强型选股信号建议 ({datetime.now().strftime('%m-%d')})"
        
        msg = MIMEMultipart()
        msg['From'] = f"Antigravity Quant <{sender_email}>"
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(html_body, 'html'))
        
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        login_user = sender_email.split("@")[0] if "163.com" in sender_email else sender_email
        print(f"[INFO] Logging in SMTP server as: {login_user}")
        server.login(login_user, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print("[SUCCESS] Quant signal report email sent to 568701293@qq.com!")
        
    except Exception as e:
        print(f"[ERROR] Pipeline run failed: {e}")

if __name__ == "__main__":
    run_pipeline_and_send_email()
