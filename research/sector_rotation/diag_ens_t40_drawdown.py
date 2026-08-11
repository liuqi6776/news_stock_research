# -*- coding: utf-8 -*-
"""ENS_T40 最大回撤 -31.05% 归因诊断

从 4 个维度拆解回撤来源：
1. 时段归因：回撤发生在哪些月份 / s123 状态是什么（在市/避险）
2. 行业归因：回撤期行业贡献（Top40 各行业仓位×行业收益）
3. 个股归因：回撤期贡献度最大的 10 只亏损股（权重×收益）
4. 集中度归因：Top5/行业≤4 约束是否被突破 / HHI 集中度
"""
import os, time, sys, pickle
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))
PKL = os.path.join(ROOT, "research", "sector_rotation", "results", "stock_gbdt_s123_results.pkl")

t0 = time.time()
# 用主引擎重跑拿 log（因为 pkl 只有 nav，没有持仓明细）
# 直接 import 跑一次获取 hold_log / trade_log
from stock_gbdt_s123_backtest import run_backtest, s123_signal_history, GBDT_FEATS, GBDT_PARAMS, CANDIDATE_INDS, V8_COMP

# 手动复刻策略: ENS_T40_S123_ONLY_S123
print("[引擎] 重跑 ENS_T40_S123_ONLY_S123 获取持仓明细...")
res = run_backtest(
    model_key="ENS", top_n=40, mode="ONLY_S123", tv=None,
    hold_threshold=0, gbdt_feats=GBDT_FEATS, gbdt_params=GBDT_PARAMS,
    candidate_inds=CANDIDATE_INDS, ind_max=4, v8_comps=V8_COMP,
    verbose=0
)
nav_s = pd.Series(res["nav"], index=res["date"]).sort_index()
hold_log = res.get("hold_log", [])
trade_log = res.get("trade_log", [])
s123_hist = s123_signal_history()
# 重建 state_ts: 每日 s123 state
state_ts = pd.Series(index=s123_hist["date"], data=s123_hist["state"].values).sort_index()
state_ts = state_ts.reindex(nav_s.index, method="ffill")
print(f"  {len(nav_s)} 交易日, {len(hold_log)} 持仓快照")

# ========== 1. 识别回撤区间 ==========
print("\n=== 1. 回撤区间识别（逐日复利） ===")
cum = nav_s / nav_s.iloc[0]
peak = cum.cummax()
dd = cum / peak - 1.0
max_dd = dd.min()
# 最大回撤期: 先找最低点日期，再找之前最高点日期
dd_end_date = dd.idxmin()
dd_start_date = peak.loc[:dd_end_date].idxmax()
print(f"  最大回撤 = {max_dd*100:.2f}%")
print(f"  回撤起点: {dd_start_date}, 峰值 NAV = {peak.loc[dd_end_date]:.4f}")
print(f"  回撤终点: {dd_end_date}, 当时 NAV = {cum.loc[dd_end_date]:.4f}")
# 区间日收益归因
ret_s = nav_s.pct_change().fillna(0)
window = ret_s[(ret_s.index >= dd_start_date) & (ret_s.index <= dd_end_date)]
state_window = state_ts[(state_ts.index >= dd_start_date) & (state_ts.index <= dd_end_date)]
print(f"  区间 {len(window)} 交易日, s123 状态分布: {dict(pd.Series(state_window).value_counts().to_dict())}")
print(f"  在市(状态='in') 交易日数: {(state_window == 'in').sum()}")
print(f"  避险(状态='out') 交易日数: {(state_window == 'out').sum()}")
# 避险期间应持有 V8，所以把区间日内收益按状态拆分
in_ret = (ret_s[(state_ts == 'in')] + 1).prod() - 1
out_ret = (ret_s[(state_ts == 'out')] + 1).prod() - 1
dd_in_ret = (window[state_window == 'in'] + 1).prod() - 1
dd_out_ret = (window[state_window == 'out'] + 1).prod() - 1
print(f"  全时段: 在市累计收益 {in_ret*100:.2f}%, 避险累计 {out_ret*100:.2f}%")
print(f"  回撤期: 在市累计收益 {dd_in_ret*100:.2f}%, 避险累计 {dd_out_ret*100:.2f}%")

# ========== 2. 月度行业归因 ==========
# 先拿到每个调仓日（月末）持仓明细 → 下月行业贡献 ≈ 行业仓位 × 下月行业收益
print("\n=== 2. 回撤期逐月行业归因（仓位×下月行业收益） ===")
# 从 hold_log 取月末快照
hold_df = pd.DataFrame(hold_log)  # 应有 cols: date/ts_code/weight/industry
# 把 date 转成和 nav_s 兼容的 int
if "date" not in hold_df.columns or len(hold_df) == 0:
    print("  hold_log 不可用，跳过行业归因")
else:
    # 只取调仓日（月末）快照，按 trade_date 分组
    reb_dates = sorted(hold_df["date"].unique())
    # 行业收益: 用中证1000成分中，对应行业的等权下月收益
    panel = pd.read_parquet(os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet"))
    # 从 panel 构造下月行业等权收益
    panel_mon = panel[["trade_date","ts_code","industry","ret_1m"]].dropna().copy()
    # 注意: panel 里的 ret_1m 是"过去1月收益"，我们需要"未来1月收益"
    # 重新构造 fwd_1m
    panel_mon["fwd_1m"] = panel_mon.groupby("ts_code")["ret_1m"].shift(-1)
    ind_ret = panel_mon.groupby(["trade_date","industry"])["fwd_1m"].mean().reset_index()
    ind_ret.columns = ["trade_date","industry","ind_fwd"]
    # 回撤期覆盖的 trade_date 区间
    dd_st_mon = int(f"{str(dd_start_date)[:6]}01") if len(str(dd_start_date))==8 else dd_start_date
    dd_ed_mon = int(f"{str(dd_end_date)[:6]}01") if len(str(dd_end_date))==8 else dd_end_date
    reb_in_dd = [d for d in reb_dates if (d >= dd_start_date and d <= dd_end_date)]
    # 对每个 reb_date: 仓位×下月行业收益 = 行业贡献度
    contrib_rows = []
    for rd in reb_in_dd:
        h = hold_df[hold_df["date"]==rd]
        ind_pos = h.groupby("industry")["weight"].sum()
        # 找下一个 trade_date（调仓后月度）
        idx = reb_dates.index(rd) if rd in reb_dates else -1
        next_m = reb_dates[idx+1] if idx+1 < len(reb_dates) else None
        if next_m is None: continue
        ir = ind_ret[ind_ret["trade_date"]==rd].set_index("industry")["ind_fwd"]
        for ind in ind_pos.index:
            w = ind_pos[ind]
            r = ir.get(ind, np.nan)
            if pd.notna(r):
                contrib_rows.append({"date": rd, "industry": ind, "weight": w, "ind_next_month_ret": r, "contrib": w*r})
    contrib_df = pd.DataFrame(contrib_rows)
    if len(contrib_df) > 0:
        # 行业总贡献
        ind_total = contrib_df.groupby("industry")["contrib"].sum().sort_values()
        print("  回撤期各行业累计贡献度（负=拖累，正=支撑）:")
        for ind, c in ind_total.items():
            print(f"    {ind:>10}: 贡献 {c*100:+.2f}%")
        top_neg = ind_total.head(5)
        print(f"\n  ⚠️ 最大拖累 TOP5 行业合计: {top_neg.sum()*100:.2f}%")
        # 集中度 HHI
        first_rd = reb_in_dd[0] if reb_in_dd else reb_dates[0]
        hh = hold_df[hold_df["date"]==first_rd]
        pos = hh.groupby("industry")["weight"].sum()
        hhi = (pos**2).sum()
        print(f"\n  起始调仓日行业HHI: {hhi:.4f} (HHI<0.1 低集中, 0.1-0.18 中等, >0.18 高度集中)")
        print(f"  初始行业数: {len(pos)}, Top3 行业仓位: {dict(pos.sort_values(ascending=False).head(3).round(3).to_dict())}")
    else:
        print("  行业归因无数据")

# ========== 3. 回撤期个股贡献 TOP10 ==========
print("\n=== 3. 回撤期个股贡献 TOP10（权重×累计收益近似） ===")
# 近似: 对每只持有中的股票，用其 ret_1m(未来1月) × 调仓时权重
if len(hold_df) > 0 and "industry" in hold_df.columns:
    # 构造 fwd_1m 到个股
    fwd_map = panel_mon.groupby(["trade_date","ts_code"])["fwd_1m"].first()
    stock_rows = []
    for rd in reb_in_dd:
        h = hold_df[hold_df["date"]==rd]
        for _, r in h.iterrows():
            r_fwd = fwd_map.get((rd, r["ts_code"]), np.nan)
            if pd.notna(r_fwd):
                stock_rows.append({"date": rd, "ts_code": r["ts_code"], "industry": r["industry"],
                                   "weight": r["weight"], "stock_ret": r_fwd, "contrib": r["weight"]*r_fwd})
    sdf = pd.DataFrame(stock_rows)
    if len(sdf) > 0:
        stk_contrib = sdf.groupby("ts_code")["contrib"].sum().sort_values()
        print("  TOP10 亏损贡献个股（回撤期合计）:")
        for code, c in stk_contrib.head(10).items():
            ind = sdf[sdf["ts_code"]==code]["industry"].iloc[0]
            n_months = (sdf["ts_code"]==code).sum()
            print(f"    {code} ({ind:>6}): {c*100:+.2f}% (持有{n_months}个月)")
        print(f"  TOP10 亏损贡献合计: {stk_contrib.head(10).sum()*100:.2f}%")
        print(f"  TOP10 盈利贡献合计: {stk_contrib.tail(10).sum()*100:.2f}%")

# ========== 4. s123 择时有效性（回测全期） ==========
print("\n=== 4. s123 择时有效性（全期 in/out 收益对比） ===")
# 中证1000指数日收益作为对照
import akshare as ak
# 本地已有中证1000数据，用 000852 直接算
try:
    df1000 = panel_mon.copy()  # 没 000852 指数，用 panel 的股票等权近似
    # 用 hold_log 里 V8 资产推断: out 期收益与 V8 组合直接比较
    ret_d = ret_s.copy()
    in_dates = state_ts[state_ts == "in"].index.intersection(ret_d.index)
    out_dates = state_ts[state_ts == "out"].index.intersection(ret_d.index)
    # 年化: (累计+1)**(252/n)-1
    def ann_ret(s):
        if len(s) < 20: return np.nan
        return (s+1).prod()**(252.0/len(s)) - 1
    def ann_vol(s):
        if len(s) < 20: return np.nan
        return s.std() * np.sqrt(252)
    r_in = ret_d.loc[in_dates]
    r_out = ret_d.loc[out_dates]
    print(f"  In 期: {len(in_dates)} 日, 年化收益 {ann_ret(r_in)*100:.2f}%, 年化波动 {ann_vol(r_in)*100:.2f}%")
    print(f"  Out 期: {len(out_dates)} 日, 年化收益 {ann_ret(r_out)*100:.2f}%, 年化波动 {ann_vol(r_out)*100:.2f}%")
    # 若 s123 有效: In 期收益 >> Out 期（或波动显著低）
    print(f"  择时前(假设始终满仓In): 等效年化 {ann_ret(ret_d)*100:.2f}% (实际就是策略)")
except Exception as e:
    print(f"  s123有效性跳过: {e}")

print(f"\n总耗时 {time.time()-t0:.0f}s")
