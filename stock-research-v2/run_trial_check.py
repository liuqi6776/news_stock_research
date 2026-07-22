# -*- coding: utf-8 -*-
"""
run_trial_check.py - 实盘/试盘每日快速风控与持仓诊断工具
=====================================================

使用说明 / Usage:
1. 建立一个简单的持仓文件 `my_holdings.json`：
   {
       "113681.SH": {"buy_price": 135.0, "shares": 400},
       "118071.SH": {"buy_price": 102.0, "shares": 600}
   }
2. 运行本脚本：
   python run_trial_check.py
3. 脚本会自动检查：
   - 哪些持仓触发了 -5% 止损（提示卖出，并记录 20 天黑名单）；
   - 哪些持仓触发了 130 元强平线（提示止盈卖出）；
   - 当前试盘组合的总浮动盈亏。
"""

import os
import sys
import json
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "research", "studies", "study_006_cb_doublelow"))
try:
    from backtest_cb_doublelow import load_data
except ImportError:
    from research.studies.study_006_cb_doublelow.backtest_cb_doublelow import load_data

def check_trial_status(holdings_file='my_holdings.json'):
    if not os.path.exists(holdings_file):
        # 创建模版文件
        demo_holdings = {
            "113681": {"buy_price": 138.0, "shares": 430},
            "113042": {"buy_price": 116.0, "shares": 510}
        }
        with open(holdings_file, 'w', encoding='utf-8') as f:
            json.dump(demo_holdings, f, indent=4, ensure_ascii=False)
        print(f"[提示] 未找到持仓文件，已自动在本地创建模版持仓文件: {holdings_file}")

    with open(holdings_file, 'r', encoding='utf-8') as f:
        holdings = json.load(f)

    print("正在获取最新行情数据...")
    df_pit = load_data()
    latest_dt = df_pit['trade_date'].max()
    df_today = df_pit[df_pit['trade_date'] == latest_dt].set_index('ts_code')

    print("\n" + "="*85)
    print(f"               【试盘持仓每日风控诊断报告】")
    print(f"                行情最新日期: {latest_dt.strftime('%Y-%m-%d')}")
    print("="*85)

    alerts = []
    total_val = 0.0
    total_cost = 0.0

    for code, info in holdings.items():
        clean_code = code.split('.')[0]
        # 寻找匹配代码
        matched = [c for c in df_today.index if clean_code in c]
        if matched:
            target_code = matched[0]
            row = df_today.loc[target_code]
            curr_price = row['close']
            name = row['name'] if 'name' in df_today.columns else target_code
            
            buy_price = info['buy_price']
            shares = info['shares']
            
            pnl_pct = (curr_price / buy_price - 1.0)
            cost_val = buy_price * shares
            curr_val = curr_price * shares
            
            total_cost += cost_val
            total_val += curr_val
            
            status = "持仓正常 (OK)"
            if curr_price <= 0.95 * buy_price:
                status = "[止损预警] 触发 -5% 止损！建议清仓并拉黑20天"
                alerts.append(f"止损预警: {name}({target_code}) 成本 {buy_price:.2f} 元 -> 现价 {curr_price:.2f} 元 (跌幅 {pnl_pct:.2%})")
            elif curr_price >= 130.0:
                status = "[止盈预警] 触发 130元强平线！建议止盈卖出"
                alerts.append(f"止盈预警: {name}({target_code}) 现价 {curr_price:.2f} 元 >= 130元")

            print(f" - {name:10s} ({target_code}): 成本 {buy_price:6.2f} | 现价 {curr_price:6.2f} | 盈亏 {pnl_pct:+6.2%} | 状态: {status}")
        else:
            print(f" - 代码 {code}: 未在最新行情中找到数据")

    print("-"*85)
    tot_pnl_pct = (total_val / total_cost - 1.0) if total_cost > 0 else 0.0
    print(f" 试盘组合总持仓市值: {total_val/10000:.2f} 万元 | 总持仓成本: {total_cost/10000:.2f} 万元 | 累计浮动盈亏: {tot_pnl_pct:+6.2%}")
    print("="*85)

    if alerts:
        print("\n[!] 紧急风控交易提示:")
        for a in alerts:
            print(f"   - {a}")
    else:
        print("\n[OK] 今日持仓全线正常，无止损或强平触发，请继续安心持有！")

if __name__ == "__main__":
    check_trial_status()
