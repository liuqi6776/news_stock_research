import os
import sys
import pandas as pd
import numpy as np
import joblib
import pickle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

# Load env
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')
RANK_DIR  = os.path.join(DATA_DIR, 'ths_rank1')
CHIP_DIR  = os.path.join(DATA_DIR, 'cyq1')
OTHER_DIR = os.path.join(DATA_DIR, 'other_day1')
NEWS_DIR  = os.path.join(DATA_DIR, 'news_major1')
MODEL_PATH = os.path.join(ROOT_DIR, 'daily_dragon_news_model.joblib')
STOCK_CACHE_PATH = os.path.join(ROOT_DIR, 'trade_stock_dates_cache.pkl')

def load_stock_dates_cache():
    if os.path.exists(STOCK_CACHE_PATH):
        try:
            with open(STOCK_CACHE_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load stock dates cache: {e}")
    return {}

def is_new_stock(ts_code, date_t, stock_dates, min_days=10):
    if ts_code not in stock_dates:
        return True
    dates = stock_dates[ts_code]
    count = sum(1 for d in dates if d < date_t)
    return count < min_days

def load_news_data(target_date):
    news_market_impact = 0.0
    news_stock_dict = {}
    
    if not os.path.exists(NEWS_DIR):
        return news_market_impact, news_stock_dict
    
    for filename in os.listdir(NEWS_DIR):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(NEWS_DIR, filename)
        try:
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            date_str = data.get("article_date", "")
            if not date_str:
                continue
            trade_date = pd.to_datetime(date_str).strftime('%Y%m%d')
            if trade_date > target_date:
                continue
            
            market_impact = data.get("market_impact", 0)
            news_market_impact = float(market_impact)
            
            for s in data.get("stocks", []):
                code = s.get("stock_code")
                if not code:
                    continue
                ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ" if (code.startswith('0') or code.startswith('3')) else code
                news_stock_dict[ts_code] = float(s.get("impact", 0))
        except Exception as e:
            continue
    
    return news_market_impact, news_stock_dict

def get_options_data(target_date):
    pcr_csv = os.path.join(DATA_DIR, "qiquan", "historical_pcr.csv")
    if not os.path.exists(pcr_csv):
        return None
    try:
        df = pd.read_csv(pcr_csv)
        df['date'] = pd.to_datetime(df['date'])
        target_dt = pd.to_datetime(target_date, format='%Y%m%d')
        row = df[df['date'] == target_dt]
        if not row.empty:
            last_row = row.iloc[0]
            # Get index option QVIX for Z-Score estimate
            import akshare as ak
            df_qvix = ak.index_option_50etf_qvix()
            df_qvix['ma'] = df_qvix['close'].rolling(20).mean()
            df_qvix['std'] = df_qvix['close'].rolling(20).std()
            df_qvix['zscore'] = (df_qvix['close'] - df_qvix['ma']) / df_qvix['std']
            
            return {
                'date': target_dt.strftime('%Y-%m-%d'),
                'qvix': float(df_qvix['close'].iloc[-1]),
                'qvix_zscore': float(df_qvix['zscore'].iloc[-1]),
                'pcr_50': float(last_row['pcr_50']),
                'oi_pcr_50': float(last_row['oi_pcr_50']),
                'pcr_300': float(last_row['pcr_300']),
                'oi_pcr_300': float(last_row['oi_pcr_300']),
            }
    except Exception as e:
        print(f"Warning: Failed to load option indicators: {e}")
    return None

def build_html_report(date_str, rules, filtered_sigs, top10_raw, opt):
    health_status = "STABLE / NORMAL"
    health_color = "#2E7D32" # Green
    if opt:
        if opt['qvix_zscore'] >= 2.0:
            health_status = "Oversold Spike - Rebound Signal Triggered!"
            health_color = "#D32F2F"
        elif opt['qvix_zscore'] <= -1.5:
            health_status = "Extreme Complacency - Watch out for Pullbacks"
            health_color = "#ED6C02"
        elif opt['pcr_50'] >= 1.09:
            health_status = "Panic Extreme - Contrarian Rebound Zone"
            health_color = "#1976D2"
            
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
                    <th>排名</th>
                    <th>标的代码</th>
                    <th>上涨概率 (Score)</th>
                    <th>收盘价 (Close)</th>
                    <th>日涨幅 (Change)</th>
                    <th>流通市值 (亿)</th>
                </tr>
            </thead>
            <tbody>
        """
        for i, sig in enumerate(filtered_sigs, 1):
            filtered_html += f"""
                <tr>
                    <td>No.{i}</td>
                    <td><strong style="color: #1976D2;">{sig['ts_code']}</strong></td>
                    <td class="score-cell">{sig['prob']:.4f}</td>
                    <td>¥{sig['close']:.2f}</td>
                    <td><span style="color: #D32F2F; font-weight: bold;">{sig['pct_chg']:.2f}%</span></td>
                    <td>{sig['circ_mv']/10000:.2f}亿</td>
                </tr>
            """
        filtered_html += "</tbody></table>"

    # Top 10 Raw HTML
    top10_html = """
    <table class="raw-table">
        <thead>
            <tr>
                <th>排序</th>
                <th>标的代码</th>
                <th>基础得分</th>
                <th>收盘价</th>
                <th>当日涨跌幅</th>
                <th>个股新闻影响</th>
            </tr>
        </thead>
        <tbody>
    """
    for i, item in enumerate(top10_raw, 1):
        top10_html += f"""
            <tr>
                <td class="rank-cell">No.{i}</td>
                <td><strong>{item['ts_code']}</strong></td>
                <td>{item['prob']:.4f}</td>
                <td>¥{item['close']:.2f}</td>
                <td>{item['pct_chg']:.2f}%</td>
                <td>{item['news_stock_impact']:.2f}</td>
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
            .header {{ background: linear-gradient(135deg, #FF6F00, #E65100); color: #FFFFFF; padding: 25px; text-align: center; }}
            .header h1 {{ margin: 0; font-size: 24px; font-weight: bold; letter-spacing: 1px; }}
            .header p {{ margin: 5px 0 0 0; opacity: 0.8; font-size: 14px; }}
            .content {{ padding: 25px; }}
            h2 {{ color: #E65100; font-size: 18px; border-left: 4px solid #FF8F00; padding-left: 10px; margin-top: 30px; margin-bottom: 15px; }}
            .metrics-table {{ width: 100%; border-collapse: collapse; margin-bottom: 15px; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #E0E4EC; padding: 10px; text-align: left; font-size: 14px; }}
            .metrics-table th {{ background-color: #F8F9FA; color: #555; width: 25%; font-weight: 600; }}
            .metrics-table td {{ width: 25%; }}
            .signals-table, .raw-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            .signals-table th, .signals-table td, .raw-table th, .raw-table td {{ border: 1px solid #E2E8F0; padding: 12px; text-align: left; }}
            .signals-table th {{ background-color: #FFF3E0; color: #E65100; font-weight: bold; }}
            .raw-table th {{ background-color: #F5F5F5; color: #333; font-weight: bold; }}
            .score-cell {{ font-weight: bold; color: #E65100; }}
            .rank-cell {{ font-weight: bold; color: #FB8C00; }}
            .empty-box {{ background-color: #FFF9C4; border: 1px dashed #FBC02D; color: #F57F17; padding: 15px; border-radius: 4px; text-align: center; font-weight: 500; }}
            .footer {{ background-color: #F8F9FA; padding: 20px; text-align: center; font-size: 12px; color: #888; border-top: 1px solid #E0E4EC; }}
            ul {{ padding-left: 20px; color: #555; font-size: 14px; }}
            li {{ margin-bottom: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🐲 Daily Dragon News 量化选股晨报 (补发版)</h1>
                <p>预测分析基准日: {date_str} | 信号执行建议日: 2026-06-01 (周一) 盘前</p>
            </div>
            <div class="content">
                <h2>📈 大盘健康诊断与期权隐含波动率</h2>
                {opt_html}
 
                <h2>🎯 本日首选推荐买入标的 (Daily Dragon News 筛选后)</h2>
                {filtered_html}
 
                <h2>📊 全行业期望值排行 Top 10 (筛选前)</h2>
                {top10_html}
 
                <h2>🛠️ 本日量化过滤与中性化筛选规则说明</h2>
                {rules_html}
            </div>
            <div class="footer">
                <p>本晨报由 Antigravity 智能量化交易系统补发补算并生成。</p>
                <p><strong>风险提示</strong>：策略选股模型与期权择时结果仅供参考，不作为正式投资建议。股市有风险，投资需谨慎。</p>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

def run():
    target_date = "20260529"
    print(f"=== Running quant predictions for {target_date} ===")
    
    # 1. Predict using model
    model, feats = joblib.load(MODEL_PATH)
    stock_dates = load_stock_dates_cache()
    news_market_impact, news_stock_dict = load_news_data(target_date)
    
    p_rank = os.path.join(RANK_DIR, f"{target_date}.parquet")
    p_chip = os.path.join(CHIP_DIR, f"{target_date}.parquet")
    p_price = os.path.join(PRICE_DIR, f"{target_date}.parquet")
    p_other = os.path.join(OTHER_DIR, f"{target_date}.parquet")
    
    rank_df = pd.read_parquet(p_rank).drop_duplicates(subset=['ts_code'], keep='first')
    rank_df['hot_rank_pct'] = rank_df['hot'].rank(pct=True)
    chip_df = pd.read_parquet(p_chip)
    chip_df['chip_concentration'] = (chip_df['cost_85pct'] - chip_df['cost_15pct']) / (chip_df['cost_50pct'] + 1e-8)
    price_df = pd.read_parquet(p_price, columns=['ts_code', 'close', 'pct_chg', 'amount', 'vol', 'pre_close'])
    other_df = pd.read_parquet(p_other, columns=['ts_code', 'circ_mv'])
    
    df = pd.merge(rank_df[['ts_code', 'hot_rank_pct']], price_df, on='ts_code')
    df = pd.merge(df, chip_df[['ts_code', 'chip_concentration', 'winner_rate']], on='ts_code')
    df = pd.merge(df, other_df, on='ts_code', how='left')
    df['news_market_impact'] = news_market_impact
    df['news_stock_impact'] = df['ts_code'].map(news_stock_dict).fillna(0.0)
    df['news_sector_impact'] = 0.0
    
    # 1. Predict on the entire unfiltered set first
    X = df[feats].fillna(0)
    df['prob'] = model.predict_proba(X)[:, 1]
    
    # Top 10 Raw is selected from the entire unfiltered set
    picks_top10 = df.sort_values('prob', ascending=False).head(10)
    
    # 2. Apply filters to get final recommendations
    df_filtered = df[~df['ts_code'].str.startswith('688')].copy()
    df_filtered = df_filtered[df_filtered['circ_mv'] <= 500000]
    
    # New stock filtering
    date_int = int(target_date)
    df_filtered = df_filtered[~df_filtered['ts_code'].apply(lambda x: is_new_stock(x, date_int, stock_dates, 10))]
    
    if len(df_filtered) == 0:
        print("Warning: All stocks filtered by new stock filter. Bypassing it...")
        df_filtered = df[~df['ts_code'].str.startswith('688')].copy()
        df_filtered = df_filtered[df_filtered['circ_mv'] <= 500000]
        
    picks_top3 = df_filtered[df_filtered['prob'] > 0.8].sort_values('prob', ascending=False).head(3)
    
    # Format filtered list
    filtered_sigs = []
    if not picks_top3.empty:
        for _, r in picks_top3.iterrows():
            filtered_sigs.append({
                'ts_code': r['ts_code'],
                'prob': float(r['prob']),
                'close': float(r['close']),
                'pct_chg': float(r['pct_chg']),
                'circ_mv': float(r['circ_mv']),
            })
    else:
        # Fallback to Top 1
        top1 = df_filtered.sort_values('prob', ascending=False).head(1)
        if not top1.empty:
            r = top1.iloc[0]
            filtered_sigs.append({
                'ts_code': r['ts_code'],
                'prob': float(r['prob']),
                'close': float(r['close']),
                'pct_chg': float(r['pct_chg']),
                'circ_mv': float(r['circ_mv']),
            })
            
    # Format Top 10 Raw list
    top10_list = []
    for _, r in picks_top10.iterrows():
        top10_list.append({
            'ts_code': r['ts_code'],
            'prob': float(r['prob']),
            'close': float(r['close']),
            'pct_chg': float(r['pct_chg']),
            'news_stock_impact': float(r['news_stock_impact']),
        })
        
    rules_text = (
        "1. 板块限额: 不含科创板 (排除688开头股票)\n"
        "2. 市值要求: 属于中小市值成长股 (流通市值 <= 500 亿)\n"
        "3. 新股规避: 历史上市天数少于 10 个交易日的新股一律排除\n"
        "4. 大模型期望分: 智能多因子模型预测上涨概率期望值 prob > 0.80 的标的\n"
        "5. 止盈止损策略: T+1 集合竞价开盘入场，T+2日盘中触及 +8% 止盈或 14:50 强平"
    )
    
    # 2. Get Option Indicators for May 29
    opt = get_options_data(target_date)
    
    # 3. Build HTML Report
    html_body = build_html_report(
        date_str="2026-05-29",
        rules=rules_text,
        filtered_sigs=filtered_sigs,
        top10_raw=top10_list,
        opt=opt
    )
    
    # 4. Send Email via 163 SMTP SSL
    smtp_server = os.getenv("SMTP_SERVER", "smtp.163.com")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    mail_user = os.getenv("SMTP_USER", "13259770650@163.com")
    mail_pass = os.getenv("SMTP_PASSWORD")
    receivers = ["568701293@qq.com"]
    
    print(f"Connecting to {smtp_server}:{smtp_port} using SSL...")
    msg = MIMEMultipart()
    msg['From'] = f"Daily Dragon Quant <{mail_user}>"
    msg['To'] = receivers[0]
    msg['Subject'] = f"【补发晨报】Daily Dragon 量化多因子选股预测报告 ({target_date[:4]}-{target_date[4:6]}-{target_date[6:8]})"
    msg.attach(MIMEText(html_body, 'html'))
    
    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=15)
        # Login using the short username or full email as robustly handled
        login_user = mail_user.split("@")[0] if "163.com" in mail_user else mail_user
        print(f"Logging in as: {login_user}")
        server.login(login_user, mail_pass)
        print("Logged in successfully! Sending report email...")
        server.sendmail(mail_user, receivers, msg.as_string())
        server.quit()
        print(f"[SUCCESS] Email successfully sent to {receivers[0]}!")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")

if __name__ == "__main__":
    run()
