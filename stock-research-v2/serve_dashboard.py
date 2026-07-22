# -*- coding: utf-8 -*-
"""
serve_dashboard.py - 可转债 10 万实盘/试盘本地 UI 控制台
=====================================================

运行本脚本后，会在本地启动一个 Web 服务器：
  打开浏览器访问: http://127.0.0.1:8500

页面功能：
1. 【10万资金分配仪表盘】：明确标注 6万元 (60%) 买入转债 Top 20，4万元 (40%) 存入货币基金。
2. 【今日具体买什么、怎么买】：实时计算 20 只目标转债的代码、名称、现价与建议买入张数（每只约 3,000 元）。
3. 【我的持仓监控与一键诊断】：读取并管理 `my_holdings.json`，自动检测 -5% 止损与 130 元强平线，高亮报警指示。
4. 【交互式网页控制台】：支持在网页上直接录入/修改真实持仓，一键重新生成信号。
"""

import os
import sys
import json
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "research", "studies", "study_006_cb_doublelow"))
try:
    from backtest_cb_doublelow import load_data
except ImportError:
    from research.studies.study_006_cb_doublelow.backtest_cb_doublelow import load_data

HOLDINGS_FILE = os.path.join(SCRIPT_DIR, "my_holdings.json")

def get_live_target_signals(capital_cb=60000.0, top_n=20):
    df_pit = load_data()
    latest_date = df_pit['trade_date'].max()
    df_today = df_pit[df_pit['trade_date'] == latest_date].copy()
    
    rating_ranks = {'AAA': 6, 'AA+': 5, 'AA': 4, 'AA-': 3, 'A+': 2, 'A': 1}
    def get_rating_rank(r):
        if pd.isna(r): return 0
        r_str = str(r).upper().strip()
        for key in rating_ranks:
            if r_str.startswith(key): return rating_ranks[key]
        return 0
    df_today['rating_rank'] = df_today['rating'].apply(get_rating_rank)
    
    df_active = df_today.dropna(subset=['close', 'premium']).copy()
    df_active = df_active[~df_active['stock_name'].str.contains('ST', na=False)]
    df_active = df_active[df_active['issue_size'] >= 1.0]
    df_active = df_active[df_active['years_to_maturity'] >= 0.5]
    df_active = df_active[df_active['rating_rank'] >= 1]
    
    r_dl = df_active['double_low'].rank(pct=True, ascending=True)
    r_prem = df_active['premium'].rank(pct=True, ascending=True)
    mom_filled = df_active['stock_mom_20'].fillna(df_active['stock_mom_20'].median())
    vol_filled = df_active['stock_vol_20'].fillna(df_active['stock_vol_20'].median())
    r_mom = mom_filled.rank(pct=True, ascending=False)
    r_vol = vol_filled.rank(pct=True, ascending=True)
    r_scale = df_active['issue_size'].rank(pct=True, ascending=True)
    r_ytm = df_active['ytm'].rank(pct=True, ascending=False)
    r_dist = df_active['dist_redempt'].rank(pct=True, ascending=False)
    
    df_active['score'] = (
        0.30 * r_dl + 0.30 * r_prem + 0.10 * r_mom + 0.10 * r_vol + 0.10 * r_scale + 0.05 * r_ytm + 0.05 * r_dist
    )
    df_top = df_active.sort_values('score').head(top_n).copy()
    
    single_alloc = capital_cb / top_n
    items = []
    for _, row in df_top.iterrows():
        close_p = float(row['close'])
        # 10张整倍数向下取整，最少10张
        shares = max(10, (int(single_alloc / close_p / 10)) * 10)
        items.append({
            'code': str(row['ts_code']),
            'name': str(row['name']) if 'name' in row and pd.notna(row['name']) else str(row['ts_code']),
            'close': round(close_p, 2),
            'premium': round(float(row['premium']), 2),
            'double_low': round(float(row['double_low']), 2),
            'rating': str(row['rating']),
            'alloc_amount': round(single_alloc, 2),
            'suggested_shares': shares,
            'est_cost': round(shares * close_p, 2)
        })
    return items, latest_date.strftime('%Y-%m-%d')

def get_my_holdings_status():
    if not os.path.exists(HOLDINGS_FILE):
        demo = {
            "113681.SH": {"buy_price": 138.0, "shares": 400},
            "113042.SH": {"buy_price": 116.0, "shares": 500}
        }
        with open(HOLDINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(demo, f, indent=4, ensure_ascii=False)
            
    with open(HOLDINGS_FILE, 'r', encoding='utf-8') as f:
        holdings = json.load(f)
        
    df_pit = load_data()
    latest_dt = df_pit['trade_date'].max()
    df_today = df_pit[df_pit['trade_date'] == latest_dt].set_index('ts_code')
    
    items = []
    tot_cost = 0.0
    tot_val = 0.0
    alerts = []
    
    for code, info in holdings.items():
        clean_code = code.split('.')[0]
        matched = [c for c in df_today.index if clean_code in c]
        if matched:
            tc = matched[0]
            row = df_today.loc[tc]
            curr_price = round(float(row['close']), 2)
            name = str(row['name']) if 'name' in df_today.columns else tc
            buy_price = round(float(info['buy_price']), 2)
            shares = int(info['shares'])
            
            pnl_pct = round((curr_price / buy_price - 1.0) * 100, 2)
            cost_val = round(buy_price * shares, 2)
            curr_val = round(curr_price * shares, 2)
            
            tot_cost += cost_val
            tot_val += curr_val
            
            action_tag = "HOLD"
            action_desc = "持仓正常，安心持有"
            badge_class = "badge-ok"
            
            if curr_price <= 0.95 * buy_price:
                action_tag = "STOP_LOSS"
                action_desc = "🔴 触发 -5% 止损！建议立刻卖出并拉黑 20 天"
                badge_class = "badge-danger"
                alerts.append(f"止损预警：{name}({tc}) 成本 {buy_price} -> 现价 {curr_price} (亏损 {pnl_pct}%)")
            elif curr_price >= 130.0:
                action_tag = "PROFIT_TAKE"
                action_desc = "🟢 触发 130元 强平线！建议止盈卖出"
                badge_class = "badge-success"
                alerts.append(f"止盈预警：{name}({tc}) 现价 {curr_price} >= 130.0元")
                
            items.append({
                'code': tc, 'name': name, 'buy_price': buy_price, 'curr_price': curr_price,
                'shares': shares, 'cost_val': cost_val, 'curr_val': curr_val,
                'pnl_pct': pnl_pct, 'action_tag': action_tag, 'action_desc': action_desc,
                'badge_class': badge_class
            })
            
    tot_pnl_pct = round(((tot_val / tot_cost - 1.0) * 100), 2) if tot_cost > 0 else 0.0
    return {
        'items': items,
        'tot_cost': round(tot_cost, 2),
        'tot_val': round(tot_val, 2),
        'tot_pnl_pct': tot_pnl_pct,
        'alerts': alerts,
        'latest_date': latest_dt.strftime('%Y-%m-%d')
    }

class DashboardHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/api/data':
            try:
                targets, latest_date = get_live_target_signals(capital_cb=60000.0, top_n=20)
                holdings_data = get_my_holdings_status()
                resp = {
                    'status': 'success',
                    'capital': {
                        'total': 100000.0,
                        'cb_capital': 60000.0,
                        'cash_capital': 40000.0
                    },
                    'targets': targets,
                    'holdings': holdings_data,
                    'latest_date': latest_date
                }
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(resp, ensure_ascii=False).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode('utf-8'))
        else:
            # Serve HTML UI
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_UI.encode('utf-8'))

    def do_POST(self):
        if self.path == '/api/save_holdings':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                with open(HOLDINGS_FILE, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success'}).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode('utf-8'))

HTML_UI = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>量化转债 10万实盘/试盘控制台 | Quant CB Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: #111827;
            --card-border: #1f2937;
            --primary: #3b82f6;
            --primary-hover: #2563eb;
            --text-main: #f9fafb;
            --text-sub: #9ca3af;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --accent-purple: #8b5cf6;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', system-ui, -apple-system, sans-serif; }
        body { background-color: var(--bg-color); color: var(--text-main); line-height: 1.5; padding: 24px; }
        .container { max-width: 1300px; margin: 0 auto; }

        /* Header */
        header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid var(--card-border); margin-bottom: 24px; }
        h1 { font-size: 24px; font-weight: 700; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .sub-title { font-size: 14px; color: var(--text-sub); margin-top: 4px; }
        .date-badge { background: #1e293b; padding: 6px 14px; border-radius: 20px; font-size: 13px; color: var(--text-sub); border: 1px solid var(--card-border); }

        /* Top Banner Action Steps */
        .banner { background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1)); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; padding: 20px; margin-bottom: 24px; }
        .banner-title { font-size: 18px; font-weight: 600; color: #93c5fd; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
        .steps-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-top: 12px; }
        .step-card { background: rgba(17, 24, 39, 0.8); border: 1px solid var(--card-border); border-radius: 8px; padding: 16px; }
        .step-num { font-size: 12px; font-weight: 700; color: var(--primary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
        .step-text { font-size: 15px; font-weight: 600; color: var(--text-main); }
        .step-desc { font-size: 13px; color: var(--text-sub); margin-top: 4px; }

        /* Metric Cards */
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .metric-card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 18px; position: relative; overflow: hidden; }
        .metric-card::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: var(--primary); }
        .metric-card.cash::before { background: var(--success); }
        .metric-card.cb::before { background: var(--accent-purple); }
        .metric-card.pnl::before { background: var(--warning); }
        .metric-label { font-size: 13px; color: var(--text-sub); }
        .metric-val { font-size: 26px; font-weight: 700; margin-top: 6px; }
        .metric-hint { font-size: 12px; color: var(--text-sub); margin-top: 4px; }

        /* Tables & Sections */
        .section-grid { display: grid; grid-template-columns: 1fr; gap: 24px; }
        .card { background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 12px; padding: 20px; }
        .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
        .card-title { font-size: 16px; font-weight: 600; }
        
        table { width: 100%; border-collapse: collapse; text-align: left; font-size: 14px; }
        th { padding: 12px; background: #1f2937; color: var(--text-sub); font-weight: 500; font-size: 12px; text-transform: uppercase; }
        td { padding: 12px; border-bottom: 1px solid var(--card-border); }
        tr:hover td { background: rgba(255, 255, 255, 0.02); }

        /* Badges & Buttons */
        .badge { padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; display: inline-block; }
        .badge-ok { background: rgba(16, 185, 129, 0.15); color: #34d399; }
        .badge-danger { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
        .badge-success { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
        .btn { background: var(--primary); color: white; border: none; padding: 8px 16px; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; transition: 0.2s; }
        .btn:hover { background: var(--primary-hover); }

        .alerts-box { background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; }
        .alert-item { color: #f87171; font-size: 14px; font-weight: 500; margin-bottom: 4px; }
        .alert-item:last-child { margin-bottom: 0; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>量化转债 10万实盘/试盘控制台</h1>
                <div class="sub-title">10W Capital Quant Convertible Bond Live Execution Dashboard</div>
            </div>
            <div class="date-badge" id="dateBadge">行情数据计算中...</div>
        </header>

        <!-- Top Action Banner -->
        <div class="banner">
            <div class="banner-title">
                💡 手上 10 万元资金，今日该怎么买、怎么做？ (Action Directives)
            </div>
            <div class="steps-grid">
                <div class="step-card">
                    <div class="step-num">步骤 1 · 现金拨备 (40%)</div>
                    <div class="step-text">4.0 万元 存入货币基金 / 国债逆回购</div>
                    <div class="step-desc">选 银华日利(511880) 或 支付宝/微信理财，提供无风险 3% 底座。</div>
                </div>
                <div class="step-card">
                    <div class="step-num">步骤 2 · 选券建仓 (60%)</div>
                    <div class="step-text">6.0 万元 平均买入下表 TOP 20 转债</div>
                    <div class="step-desc">单只买入约 3,000 元（约 20-30 张/整手），在券商软件按买入张数挂单。</div>
                </div>
                <div class="step-card">
                    <div class="step-num">步骤 3 · 盘中盯防与月度调仓</div>
                    <div class="step-text">日常盯防 -5% 止损 & 130元强平</div>
                    <div class="step-desc">若持仓触发红/绿预警则卖出；月末最后一个交易日一键刷新调仓。</div>
                </div>
            </div>
        </div>

        <!-- Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">总投资资金 (Total Capital)</div>
                <div class="metric-val">￥100,000</div>
                <div class="metric-hint">资产配置比例 60/40</div>
            </div>
            <div class="metric-card cb">
                <div class="metric-label">转债策略预算 (CB Allocation)</div>
                <div class="metric-val">￥60,000</div>
                <div class="metric-hint">共 20 只，单只 ￥3,000</div>
            </div>
            <div class="metric-card cash">
                <div class="metric-label">理财现金预算 (Cash Anchor)</div>
                <div class="metric-val">￥40,000</div>
                <div class="metric-hint">无风险 3.0% 年化收益底座</div>
            </div>
            <div class="metric-card pnl">
                <div class="metric-label">试盘持仓浮动盈亏 (Trial PnL)</div>
                <div class="metric-val" id="trialPnl">--</div>
                <div class="metric-hint" id="trialCostHint">成本: --</div>
            </div>
        </div>

        <div class="section-grid">
            <!-- Current Holdings Alert & Table -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">📌 我的真实试盘持仓与风控诊断 (My Live Holdings Status)</div>
                    <button class="btn" onclick="fetchData()">刷新最新行情</button>
                </div>
                <div id="alertsContainer"></div>
                <table>
                    <thead>
                        <tr>
                            <th>代码</th>
                            <th>名称</th>
                            <th>买入成本(元)</th>
                            <th>最新现价(元)</th>
                            <th>持仓张数</th>
                            <th>持仓市值(元)</th>
                            <th>盈亏%</th>
                            <th>风控诊断指示 (Operational Action)</th>
                        </tr>
                    </thead>
                    <tbody id="holdingsTbody">
                        <tr><td colspan="8" style="text-align:center;">加载中...</td></tr>
                    </tbody>
                </table>
            </div>

            <!-- Target Top 20 Signal List -->
            <div class="card">
                <div class="card-header">
                    <div class="card-title">🎯 10 万元资金【今日具体买入清单】(Target Top 20 Portfolio)</div>
                    <div style="font-size:13px; color:var(--text-sub);">按单只 3,000 元自动整手换算</div>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>序号</th>
                            <th>代码</th>
                            <th>转债名称</th>
                            <th>最新现价</th>
                            <th>溢价率%</th>
                            <th>双低值</th>
                            <th>评级</th>
                            <th>建议分配金额</th>
                            <th>建议买入手数/张数</th>
                        </tr>
                    </thead>
                    <tbody id="targetsTbody">
                        <tr><td colspan="9" style="text-align:center;">计算中...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        async function fetchData() {
            try {
                const res = await fetch('/api/data');
                const data = await res.json();
                if(data.status === 'success') {
                    renderDashboard(data);
                } else {
                    alert('获取数据失败: ' + data.message);
                }
            } catch(e) {
                console.error(e);
            }
        }

        function renderDashboard(data) {
            document.getElementById('dateBadge').innerText = '行情最新计算日期: ' + data.latest_date;
            
            // Render PnL
            const h = data.holdings;
            const pnlEl = document.getElementById('trialPnl');
            pnlEl.innerText = (h.tot_pnl_pct >= 0 ? '+' : '') + h.tot_pnl_pct + '%';
            pnlEl.style.color = h.tot_pnl_pct >= 0 ? 'var(--success)' : 'var(--danger)';
            document.getElementById('trialCostHint').innerText = '市值: ￥' + h.tot_val + ' | 成本: ￥' + h.tot_cost;

            // Render Alerts
            const alertBox = document.getElementById('alertsContainer');
            if(h.alerts && h.alerts.length > 0) {
                let html = '<div class="alerts-box">';
                h.alerts.forEach(a => {
                    html += `<div class="alert-item">⚠️ ${a}</div>`;
                });
                html += '</div>';
                alertBox.innerHTML = html;
            } else {
                alertBox.innerHTML = '';
            }

            // Render Holdings Table
            const hTbody = document.getElementById('holdingsTbody');
            if(h.items.length === 0) {
                hTbody.innerHTML = '<tr><td colspan="8" style="text-align:center;">暂无持仓，请在下表中根据建议建仓！</td></tr>';
            } else {
                let html = '';
                h.items.forEach(item => {
                    html += `<tr>
                        <td><b>${item.code}</b></td>
                        <td>${item.name}</td>
                        <td>￥${item.buy_price}</td>
                        <td>￥${item.curr_price}</td>
                        <td>${item.shares} 张 (${item.shares/10}手)</td>
                        <td>￥${item.curr_val}</td>
                        <td style="color:${item.pnl_pct >= 0 ? 'var(--success)' : 'var(--danger)'}; font-weight:600;">
                            ${item.pnl_pct >= 0 ? '+' : ''}${item.pnl_pct}%
                        </td>
                        <td><span class="badge ${item.badge_class}">${item.action_desc}</span></td>
                    </tr>`;
                });
                hTbody.innerHTML = html;
            }

            // Render Targets Table
            const tTbody = document.getElementById('targetsTbody');
            let tHtml = '';
            data.targets.forEach((t, idx) => {
                tHtml += `<tr>
                    <td><b>${idx + 1}</b></td>
                    <td><b style="color:var(--primary);">${t.code}</b></td>
                    <td><b>${t.name}</b></td>
                    <td>￥${t.close}</td>
                    <td>${t.premium}%</td>
                    <td>${t.double_low}</td>
                    <td><span class="badge badge-ok">${t.rating}</span></td>
                    <td>￥${t.alloc_amount}</td>
                    <td><b style="color:var(--success); font-size:15px;">${t.suggested_shares} 张 (${t.suggested_shares/10}手)</b> (约￥${t.est_cost})</td>
                </tr>`;
            });
            tTbody.innerHTML = tHtml;
        }

        fetchData();
    </script>
</body>
</html>
"""

def main():
    port = 8500
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, DashboardHandler)
    url = f"http://127.0.0.1:{port}"
    print("\n" + "="*80)
    print(f"       🚀 10万资金转债实盘控制台 Web 服务已启动！")
    print(f"       请在浏览器打开访问: {url}")
    print("="*80)
    webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nWeb 服务已停止。")

if __name__ == "__main__":
    main()
