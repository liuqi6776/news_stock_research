# -*- coding: utf-8 -*-
"""
满仓 + 止盈/止损/强平 参数网格搜索
目标: 在高胜率板块池上, 找收益与回撤的最优平衡
参数:
  - 止盈线 tp: 20% / 30% / 40% / 50%
  - 止损线 sl: 无 / -10% / -20%
  - 强平月数 mh: 无 / 12月 / 24月
"""
import os
import numpy as np
import pandas as pd
import itertools

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PE_PATH = os.path.join(RESULTS_DIR, "industry_pe.csv")
RET_PATH = os.path.join(RESULTS_DIR, "industry_ret.csv")

PE_PCT_THRESHOLD = 0.30
ROLLING_WINDOW = 60
INIT_CAPITAL = 1_000_000

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

def simulate(ret_df, pe_pct_df, eligible_inds,
             tp_target, stop_loss, max_hold_months, cash_buf=0.0):
    """满仓模拟: 低估信号全买, 止盈/止损/强平三机制"""
    dates = ret_df.index.tolist()
    start_idx = next(i for i, d in enumerate(dates) if d.startswith('2020'))
    dates = dates[start_idx:]

    cash = INIT_CAPITAL
    holdings = {}  # {industry: {'cost','value','buy_idx'}}
    nav_history = []
    monthly_rets = []
    prev_total = INIT_CAPITAL
    tp_cnt = sl_cnt = mh_cnt = 0

    for idx, d in enumerate(dates):
        # 满仓买入: 低估板块全部买入
        if cash > 1000:
            candidates = []
            for ind in eligible_inds:
                if ind not in pe_pct_df.columns or ind in holdings:
                    continue
                if d not in pe_pct_df.index:
                    continue
                pct = pe_pct_df.loc[d, ind]
                if pd.notna(pct) and pct < PE_PCT_THRESHOLD:
                    candidates.append(ind)
            if candidates:
                per = cash / len(candidates)
                for ind in candidates:
                    if per <= 0: break
                    cash -= per
                    holdings[ind] = {'cost': per, 'value': per, 'buy_idx': idx}

        # 更新收益 + 止盈止损强平
        total_value = cash
        for ind in list(holdings.keys()):
            r = 0
            if ind in ret_df.columns and d in ret_df.index:
                r = ret_df.loc[d, ind]
                if pd.isna(r): r = 0
            holdings[ind]['value'] *= (1 + r)
            cum_r = holdings[ind]['value'] / holdings[ind]['cost'] - 1

            # 注意: 止盈/止损/强平回收的现金也要计入 total_value (月末总资产)
            if cum_r >= tp_target:
                total_value += holdings[ind]['value']
                cash += holdings[ind]['value']; tp_cnt += 1; del holdings[ind]; continue
            if stop_loss is not None and cum_r <= stop_loss:
                total_value += holdings[ind]['value']
                cash += holdings[ind]['value']; sl_cnt += 1; del holdings[ind]; continue
            if max_hold_months is not None:
                if idx - holdings[ind]['buy_idx'] >= max_hold_months:
                    total_value += holdings[ind]['value']
                    cash += holdings[ind]['value']; mh_cnt += 1; del holdings[ind]; continue
            total_value += holdings[ind]['value']

        nav_history.append(total_value)
        mret = (total_value - prev_total) / max(prev_total, 1)
        monthly_rets.append(mret)
        prev_total = total_value

    nav = pd.Series(nav_history, index=dates)
    total_ret = nav.iloc[-1] / INIT_CAPITAL - 1
    years = len(nav) / 12
    ann = annualized_return(total_ret, years)
    mdd = max_drawdown(nav)
    mrets = pd.Series(monthly_rets)
    excess = mrets.mean() - 0.025 / 12
    sharpe = np.sqrt(12) * excess / max(mrets.std(), 1e-9)
    return {
        '总收益': total_ret, '年化': ann, '最大回撤': mdd,
        '夏普': sharpe, '止盈': tp_cnt, '止损': sl_cnt, '强平': mh_cnt,
        '期末持仓': len(holdings), '剩余现金(万)': cash / 10000,
        'nav': nav,
    }

def main():
    pe_df, ret_df = load_data()
    pe_pct_df = compute_pe_pct(pe_df)
    print(f"[数据] PE {pe_df.shape} | Ret {ret_df.shape} | 板块池 {len(HIGH_WIN_INDS)} 个")

    tp_list = [0.20, 0.30, 0.40, 0.50]
    sl_list = [None, -0.10, -0.20]
    mh_list = [None, 12, 24]

    results = []
    n_total = len(tp_list) * len(sl_list) * len(mh_list)
    for i, (tp, sl, mh) in enumerate(itertools.product(tp_list, sl_list, mh_list)):
        r = simulate(ret_df, pe_pct_df, HIGH_WIN_INDS, tp, sl, mh)
        tp_s = f"{tp:.0%}" if tp else "无"
        sl_s = f"{sl:.0%}" if sl is not None else "无"
        mh_s = f"{mh}月" if mh else "无"
        results.append({
            '止盈': tp_s, '止损': sl_s, '强平': mh_s,
            '年化': r['年化'], '最大回撤': r['最大回撤'], '夏普': r['夏普'],
            '累计收益': r['总收益'],
        })
        print(f"[{i+1}/{n_total}] 止盈={tp_s:>4s} 止损={sl_s:>4s} 强平={mh_s:>4s} "
              f"年化={r['年化']:>6.1%} 回撤={r['最大回撤']:>7.1%} 夏普={r['夏普']:.2f}")

    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("【按夏普排序 Top15】")
    print("=" * 80)
    print(df.sort_values('夏普', ascending=False).head(15).to_string(index=False))

    print("\n" + "=" * 80)
    print("【回撤可控(<=-25%)中年化最高 Top10】")
    print("=" * 80)
    mask = df['最大回撤'] >= -0.25
    print(df[mask].sort_values('年化', ascending=False).head(10).to_string(index=False))

    # 保存
    df.to_csv(os.path.join(RESULTS_DIR, "fullinvest_param_grid.csv"),
              index=False, encoding='utf-8-sig')
    print(f"\n网格结果已保存: results/fullinvest_param_grid.csv")

if __name__ == "__main__":
    main()
