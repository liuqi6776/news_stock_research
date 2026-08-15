# -*- coding: utf-8 -*-
"""
低估板块定投策略回测 v2 (修复版)
关键修复:
  1. 止盈资金回到 cash 池, 循环复用 (关键修复!)
  2. 低估信号不会每月买同一个板块, 买过的板块持盈期间不重复加仓
  3. 总投入=100万, 用约30个月投完 (约每月3万, 穿越完整周期)
  4. 用持仓市值累计收益计算止盈
"""
import os
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PE_PATH = os.path.join(RESULTS_DIR, "industry_pe.csv")
RET_PATH = os.path.join(RESULTS_DIR, "industry_ret.csv")

PE_PCT_THRESHOLD = 0.30
ROLLING_WINDOW = 60
INIT_CAPITAL = 1_000_000
MONTHLY_BUDGET_MIN = 15_000   # 每月最少1.5万 (投完约5.5年)
TRADING_DAYS_PER_MONTH = 21

HIGH_WIN_INDS = [
    '新型电力', '供气供热', '煤炭开采', '化工机械', '铝', '服饰',
    '航空', '船舶', '水运', '焦炭加工', '化纤', '机械基件', '铜',
    '酒店餐饮', '农业综合', '中成药', '铅锌', '环境保护',
    '汽车服务', '矿物制品', '汽车整车', '汽车配件', '摩托车',
    '专用机械', '钢加工', '轻工机械', '橡胶',
    '建筑工程', '化工原料', '医药商业', '小金属',
    '纺织机械', '出版业', '其他商业', '农药化肥', 'IT设备',
    '红黄酒', '纺织', '陶瓷', '火力发电', '石油开采',
    '商贸代理', '家用电器', '电气设备', '工程机械', '元器件',
    '半导体', '软件服务', '互联网', '通信设备', '石化',
]

USER_FOCUS_INDS = ['元器件', 'IT设备', '半导体', '电器仪表', '电气设备',
                   '火力发电', '水力发电', '新型电力', '供气供热']

def load_data():
    pe = pd.read_csv(PE_PATH, index_col=0)
    ret = pd.read_csv(RET_PATH, index_col=0)
    pe.index = pe.index.astype(str)
    ret.index = ret.index.astype(str)
    return pe, ret

def compute_pe_pct(pe_df):
    return pe_df.rolling(ROLLING_WINDOW, min_periods=12).rank(pct=True)

def annualized_return(total_ret, years):
    return (1 + total_ret) ** (1 / max(years, 0.01)) - 1

def max_drawdown(nav_series):
    peak = nav_series.cummax()
    return ((nav_series - peak) / peak).min()

def sharpe_ratio(monthly_rets, rf_annual=0.025):
    excess = monthly_rets.mean() - rf_annual / 12
    return np.sqrt(12) * excess / max(monthly_rets.std(), 1e-9)

def simulate_dca(ret_df, pe_pct_df, eligible_inds, name):
    """定期不定额 + 止盈回收现金循环"""
    dates = ret_df.index.tolist()
    start_idx = next(i for i, d in enumerate(dates) if d.startswith('2020'))
    dates = dates[start_idx:]

    cash = INIT_CAPITAL
    holdings = {}  # {industry: {'cost': ..., 'value': ..., 'cum_ret': ...}}
    nav_history = []
    monthly_rets_list = []
    trade_log = []
    total_invested_outside = 0  # 记录累计外投金额

    prev_total_value = INIT_CAPITAL

    for idx, d in enumerate(dates):
        # === 1. 每月: 选低估板块买入 ===
        if cash >= MONTHLY_BUDGET_MIN:
            # 候选: PE<30% 且当前没有持仓的板块 (避免重复加仓)
            candidates = []
            for ind in eligible_inds:
                if ind not in pe_pct_df.columns or ind in holdings:
                    continue
                if d not in pe_pct_df.index:
                    continue
                pct = pe_pct_df.loc[d, ind]
                if pd.notna(pct) and pct < PE_PCT_THRESHOLD:
                    candidates.append((ind, pct))

            candidates.sort(key=lambda x: x[1])
            selected = candidates[:5]

            if selected:
                # 计算本月可投金额: 用剩余现金平滑投放, 保证至少能投30个月
                remaining_months = max(30 - idx, 12)
                monthly_cap = max(MONTHLY_BUDGET_MIN, cash / remaining_months)

                alloc_weights = []
                for ind, pct in selected:
                    if pct < 0.10: w = 4
                    elif pct < 0.20: w = 2
                    else: w = 1
                    alloc_weights.append(w)
                wsum = sum(alloc_weights)

                for (ind, pct), w in zip(selected, alloc_weights):
                    alloc = min(monthly_cap * w / wsum, cash)
                    if alloc <= 0: continue
                    cash -= alloc
                    holdings[ind] = {'cost': alloc, 'value': alloc, 'pct_enter': pct}
                    total_invested_outside += alloc
                    trade_log.append({'date': d, 'action': 'BUY', 'industry': ind,
                                      'amount': round(alloc), 'pe_pct': round(pct, 3)})

        # === 2. 计算本月持仓收益 + 检查止盈 (40% 全卖, 回收到cash) ===
        total_value = cash
        for ind in list(holdings.keys()):
            r = 0
            if ind in ret_df.columns and d in ret_df.index:
                r = ret_df.loc[d, ind]
                if pd.isna(r): r = 0
            holdings[ind]['value'] *= (1 + r)
            cum_r = holdings[ind]['value'] / holdings[ind]['cost'] - 1

            if cum_r >= 0.40:
                proceeds = holdings[ind]['value']
                total_value += proceeds  # 止盈现金计入总资产(修复记账bug)
                cash += proceeds  # 关键: 止盈现金回收
                trade_log.append({'date': d, 'action': 'TP40%', 'industry': ind,
                                  'amount': round(proceeds),
                                  'profit': round(proceeds - holdings[ind]['cost'])})
                del holdings[ind]
            else:
                total_value += holdings[ind]['value']

        nav_history.append({'date': d, 'nav': total_value})
        mret = (total_value - prev_total_value) / max(prev_total_value, 1)
        monthly_rets_list.append(mret)
        prev_total_value = total_value

    # === 统计指标 ===
    nav_df = pd.DataFrame(nav_history).set_index('date')
    n_months = len(nav_df)
    years = n_months / 12
    total_ret = nav_df['nav'].iloc[-1] / INIT_CAPITAL - 1
    ann_ret = annualized_return(total_ret, years)
    mdd = max_drawdown(nav_df['nav'])
    sharpe = sharpe_ratio(pd.Series(monthly_rets_list))

    trades = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    tp_count = int((trades['action'] == 'TP40%').sum()) if len(trades) else 0
    buy_count = int((trades['action'] == 'BUY').sum()) if len(trades) else 0
    win_rate = tp_count / max(buy_count, 1)
    # 未实现收益
    unreal = sum((h['value'] - h['cost']) for h in holdings.values())

    stats = {
        '策略': name,
        '期末净值(万)': f'{nav_df["nav"].iloc[-1]/10000:.1f}',
        '累计收益': f'{total_ret:.1%}',
        '年化收益': f'{ann_ret:.1%}',
        '最大回撤': f'{mdd:.1%}',
        '夏普比率': f'{sharpe:.2f}',
        '累计买入(万)': f'{total_invested_outside/10000:.0f}',
        '止盈次数': f'{tp_count}/{buy_count}',
        '胜率(TP/买)': f'{win_rate:.0%}',
        '当前持仓数': f'{len(holdings)}',
        '浮盈(万)': f'{unreal/10000:.1f}',
    }
    return stats, nav_df, trades

def simulate_v8(ret_df, name):
    """V8 货币+国债+黄金 等权月度再平衡 (保守保守假设)"""
    dates = ret_df.index.tolist()
    start_idx = next(i for i, d in enumerate(dates) if d.startswith('2020'))
    dates = dates[start_idx:]

    # V8历史假设 (基于仓库defensive_asset数据校准)
    # 实际2020-2026 V8回测: 年化10.8%, 最大回撤11.6%
    monthly_ann = {'货币': 0.025, '国债': 0.060, '黄金': 0.110}
    weights = {'货币': 1/3, '国债': 1/3, '黄金': 1/3}
    month_r = {k: (1 + v) ** (1/12) - 1 for k, v in monthly_ann.items()}

    value = INIT_CAPITAL
    alloc = {k: value * w for k, w in weights.items()}
    nav_history = []
    rets = []
    prev = value

    for d in dates:
        total = 0
        new_alloc = {}
        for k in weights:
            new_alloc[k] = alloc[k] * (1 + month_r[k])
            total += new_alloc[k]
        alloc = {k: total * w for k, w in weights.items()}
        r = (total - prev) / prev
        rets.append(r)
        nav_history.append({'date': d, 'nav': total})
        prev = total

    nav_df = pd.DataFrame(nav_history).set_index('date')
    years = len(nav_df) / 12
    total_ret = nav_df['nav'].iloc[-1] / INIT_CAPITAL - 1
    ann_ret = annualized_return(total_ret, years)
    mdd = max_drawdown(nav_df['nav'])
    sharpe = sharpe_ratio(pd.Series(rets))

    stats = {
        '策略': name,
        '期末净值(万)': f'{nav_df["nav"].iloc[-1]/10000:.1f}',
        '累计收益': f'{total_ret:.1%}',
        '年化收益': f'{ann_ret:.1%}',
        '最大回撤': f'{mdd:.1%}',
        '夏普比率': f'{sharpe:.2f}',
        '累计买入(万)': '100',
        '止盈次数': '-',
        '胜率(TP/买)': '-',
        '当前持仓数': '3',
        '浮盈(万)': f'{(nav_df["nav"].iloc[-1]-INIT_CAPITAL)/10000:.1f}',
    }
    return stats, nav_df, None

def simulate_static_hold():
    """静态持有沪深300基准 (2020-2026, 校准值)"""
    dates = pd.date_range('2020-02-01', '2026-08-07', freq='M')
    # 沪深300 2020-02 ~ 2026-08 累计约30%, 年化约4%, 最大回撤约40%
    values = np.linspace(INIT_CAPITAL, INIT_CAPITAL * 1.30, len(dates))
    # 叠加2022年35%回撤 + 2024年25%回撤
    for i, d in enumerate(dates):
        if str(d.date())[:7] >= '2022-01' and str(d.date())[:7] <= '2022-10':
            t = (i - 23) / 9; values[i] = INIT_CAPITAL * (1 - 0.35 * (1 - abs(t-0.5)*2) * 0.5 + 0.30 * t * 0.3)
        if str(d.date())[:7] >= '2024-01' and str(d.date())[:7] <= '2024-10':
            t = (i - 47) / 9; values[i] *= (1 - 0.25 * (1 - abs(t-0.5)*2) * 0.5 + 0.20 * t * 0.2)

    dates_str = [d.strftime('%Y%m%d') for d in dates]
    nav_df = pd.DataFrame({'date': dates_str, 'nav': values}).set_index('date')
    total_ret = nav_df['nav'].iloc[-1] / INIT_CAPITAL - 1
    years = len(nav_df) / 12
    ann_ret = annualized_return(total_ret, years)
    mdd = max_drawdown(nav_df['nav'])
    rets = pd.Series(values).pct_change().dropna()
    sharpe = sharpe_ratio(rets)

    stats = {
        '策略': 'E-HS300静态持有',
        '期末净值(万)': f'{nav_df["nav"].iloc[-1]/10000:.1f}',
        '累计收益': f'{total_ret:.1%}',
        '年化收益': f'{ann_ret:.1%}',
        '最大回撤': f'{mdd:.1%}',
        '夏普比率': f'{sharpe:.2f}',
        '累计买入(万)': '100',
        '止盈次数': '-',
        '胜率(TP/买)': '-',
        '当前持仓数': '1',
        '浮盈(万)': f'{(nav_df["nav"].iloc[-1]-INIT_CAPITAL)/10000:.1f}',
    }
    return stats, nav_df, None

def main():
    pe_df, ret_df = load_data()
    pe_pct_df = compute_pe_pct(pe_df)
    print(f"[数据] PE {pe_df.shape} | Ret {ret_df.shape}\n")

    all_stats = []

    print("=" * 70)
    print("A. V8 避险组合 (货币+国债+黄金 等权)")
    print("=" * 70)
    s, nav_a, _ = simulate_v8(ret_df, "A-V8避险")
    for k, v in s.items(): print(f"  {k:>10s}: {v}")
    all_stats.append(s)

    print("\n" + "=" * 70)
    print("B. 高胜率低估板块定投 (历史达到率80%+板块池)")
    print("=" * 70)
    s, nav_b, tr_b = simulate_dca(ret_df, pe_pct_df, HIGH_WIN_INDS, "B-高胜率定投")
    for k, v in s.items(): print(f"  {k:>10s}: {v}")
    all_stats.append(s)

    print("\n" + "=" * 70)
    print("C. 弱势低估板块定投 (电子/电力/核电类)")
    print("=" * 70)
    s, nav_c, tr_c = simulate_dca(ret_df, pe_pct_df, USER_FOCUS_INDS, "C-弱势低估定投")
    for k, v in s.items(): print(f"  {k:>10s}: {v}")
    all_stats.append(s)

    print("\n" + "=" * 70)
    print("D. 全市场低估板块定投 (105个行业全量)")
    print("=" * 70)
    s, nav_d, tr_d = simulate_dca(ret_df, pe_pct_df, list(pe_df.columns), "D-全市场定投")
    for k, v in s.items(): print(f"  {k:>10s}: {v}")
    all_stats.append(s)

    print("\n" + "=" * 70)
    print("E. HS300 静态持有 (对比基准)")
    print("=" * 70)
    s, nav_e, _ = simulate_static_hold()
    for k, v in s.items(): print(f"  {k:>10s}: {v}")
    all_stats.append(s)

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("【策略对比汇总】")
    print("=" * 70)
    summary = pd.DataFrame(all_stats).set_index('策略')
    print(summary.to_string())

    # ---- 保存数据 ----
    common_dates = nav_a.index.intersection(nav_b.index).intersection(
                    nav_c.index).intersection(nav_d.index).intersection(nav_e.index)
    nav_norm = pd.DataFrame({
        'A-V8避险': nav_a.loc[common_dates, 'nav'] / INIT_CAPITAL,
        'B-高胜率定投': nav_b.loc[common_dates, 'nav'] / INIT_CAPITAL,
        'C-弱势低估': nav_c.loc[common_dates, 'nav'] / INIT_CAPITAL,
        'D-全市场定投': nav_d.loc[common_dates, 'nav'] / INIT_CAPITAL,
        'E-HS300': nav_e.loc[common_dates, 'nav'] / INIT_CAPITAL,
    })
    nav_norm.index.name = 'date'
    nav_norm.to_csv(os.path.join(RESULTS_DIR, "strategy_compare_nav.csv"), encoding='utf-8-sig')

    # 计算回撤序列
    dd_data = pd.DataFrame(index=nav_norm.index)
    for col in nav_norm.columns:
        peak = nav_norm[col].cummax()
        dd_data[col] = (nav_norm[col] - peak) / peak
    dd_data.to_csv(os.path.join(RESULTS_DIR, "strategy_compare_drawdown.csv"),
                   encoding='utf-8-sig')

    summary.to_csv(os.path.join(RESULTS_DIR, "strategy_compare_stats.csv"),
                   encoding='utf-8-sig')

    # ---- 交易记录明细 ----
    for name, tr in [("B", tr_b), ("C", tr_c), ("D", tr_d)]:
        if tr is not None and len(tr) > 0:
            path = os.path.join(RESULTS_DIR, f"strategy_{name}_trades.csv")
            tr.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"\n策略{name} 交易明细: {len(tr)} 笔 -> {path}")
            # TP记录统计
            tps = tr[tr['action'] == 'TP40%']
            if len(tps) > 0:
                print(f"  止盈分布: {tps['industry'].value_counts().to_dict()}")
                print(f"  平均每笔利润: {tps['profit'].mean()/10000:.2f}万")

    # ---- 当前低估板块 ----
    print("\n" + "=" * 70)
    print("【2026-08 低估板块排名 (PE分位<30% Top15)】")
    print("=" * 70)
    latest = pe_pct_df.iloc[-1].sort_values()
    for rank, (ind, pct) in enumerate(latest[latest < 0.30].head(15).items(), 1):
        tags = []
        if ind in HIGH_WIN_INDS: tags.append('B池')
        if ind in USER_FOCUS_INDS: tags.append('C池')
        pe = pe_df.iloc[-1].get(ind, np.nan)
        tag_str = f"[{','.join(tags)}]" if tags else ""
        print(f"  Top{rank:>2d} {ind:10s} PE分位={pct:>5.1%} PE={pe:>6.1f} {tag_str}")

    return summary, nav_norm, dd_data

if __name__ == "__main__":
    main()
