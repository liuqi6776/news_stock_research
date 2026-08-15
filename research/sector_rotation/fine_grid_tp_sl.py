# -*- coding: utf-8 -*-
"""
精细网格搜索: 止盈 × 止损 (2%步长)
止盈: 20% ~ 60%, 步长2%
止损: 无 ~ -30%, 步长2%
数据: 2020-2026, 高胜率板块池51个, 满仓策略
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

def simulate(ret_df, pe_pct_df, tp_target, stop_loss, max_hold_months=None):
    """满仓模拟: 止盈/止损/可选强平, 修复版(止盈现金计入总资产)"""
    dates = ret_df.index.tolist()
    start_idx = next(i for i, d in enumerate(dates) if d.startswith('2020'))
    dates = dates[start_idx:]

    cash = INIT_CAPITAL
    holdings = {}  # {ind: {'cost','value','buy_idx'}}
    nav_history = []
    prev_total = INIT_CAPITAL
    tp_cnt = sl_cnt = mh_cnt = 0

    for idx, d in enumerate(dates):
        # 满仓买入低估板块
        if cash > 1000:
            candidates = []
            for ind in HIGH_WIN_INDS:
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

        total_value = cash
        for ind in list(holdings.keys()):
            r = 0
            if ind in ret_df.columns and d in ret_df.index:
                r = ret_df.loc[d, ind]
                if pd.isna(r): r = 0
            holdings[ind]['value'] *= (1 + r)
            cum_r = holdings[ind]['value'] / holdings[ind]['cost'] - 1

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
        prev_total = total_value

    nav = np.array(nav_history, dtype=float)
    total_ret = nav[-1] / INIT_CAPITAL - 1
    years = len(nav) / 12
    ann = (1 + total_ret) ** (1 / years) - 1
    peak = np.maximum.accumulate(nav)
    mdd = ((nav - peak) / peak).min()
    # 夏普
    nav_s = pd.Series(nav_history)
    mrets = nav_s.pct_change().dropna()
    excess = mrets.mean() - 0.025 / 12
    sharpe = np.sqrt(12) * excess / max(mrets.std(), 1e-9)

    return {'止盈': tp_target, '止损': stop_loss if stop_loss is not None else 0.0,
            '年化': ann, '最大回撤': mdd, '夏普': sharpe,
            '累计收益': total_ret, '止盈次数': tp_cnt, '止损次数': sl_cnt,
            '强平次数': mh_cnt}

def main():
    pe_df, ret_df = load_data()
    pe_pct_df = compute_pe_pct(pe_df)

    tp_list = [round(0.20 + i * 0.02, 2) for i in range(21)]      # 20%~60%
    sl_list = [None] + [round(-0.04 - i * 0.02, 2) for i in range(14)]  # 无 + -4%~-30%

    results = []
    n = len(tp_list) * len(sl_list)
    print(f"网格: {len(tp_list)} 止盈 × {len(sl_list)} 止损 = {n} 组合\n")
    for i, (tp, sl) in enumerate(itertools.product(tp_list, sl_list)):
        r = simulate(ret_df, pe_pct_df, tp, sl)
        r['止损'] = 0.0 if sl is None else sl
        results.append(r)
        if (i + 1) % 50 == 0 or i + 1 == n:
            print(f"  已跑 {i+1}/{n}")

    df = pd.DataFrame(results)
    df['止损'] = df['止损'].astype(float)
    df.to_csv(os.path.join(RESULTS_DIR, "fine_grid_tp_sl.csv"), index=False, encoding='utf-8-sig')

    print("\n" + "=" * 90)
    print("【按夏普排序 Top20】")
    print("=" * 90)
    top_s = df.sort_values('夏普', ascending=False).head(20)
    for _, r in top_s.iterrows():
        print(f"  止盈{r['止盈']:.0%} 止损{r['止损']:+.0%} "
              f"年化{r['年化']:6.1%} 回撤{r['最大回撤']:7.1%} 夏普{r['夏普']:.3f}")

    print("\n" + "=" * 90)
    print("【回撤可控(>=-25%) 年化最高 Top15】")
    print("=" * 90)
    mask = df['最大回撤'] >= -0.25
    top_m = df[mask].sort_values('年化', ascending=False).head(15)
    for _, r in top_m.iterrows():
        print(f"  止盈{r['止盈']:.0%} 止损{r['止损']:+.0%} "
              f"年化{r['年化']:6.1%} 回撤{r['最大回撤']:7.1%} 夏普{r['夏普']:.3f}")

    print("\n" + "=" * 90)
    print("【综合评分 Top15 (年化 - 1.5*回撤, 兼顾收益与风险)】")
    print("=" * 90)
    df['综合'] = df['年化'] + 1.5 * df['最大回撤']
    top_c = df.sort_values('综合', ascending=False).head(15)
    for _, r in top_c.iterrows():
        print(f"  止盈{r['止盈']:.0%} 止损{r['止损']:+.0%} "
              f"年化{r['年化']:6.1%} 回撤{r['最大回撤']:7.1%} 夏普{r['夏普']:.3f} 综合{r['综合']:.3f}")

if __name__ == "__main__":
    main()
