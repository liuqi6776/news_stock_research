# -*- coding: utf-8 -*-
"""ENS_T40 最大回撤归因 (不改引擎, 直接重算持仓)

归因 4 维度:
1. 时段: 回撤起止日期, s123 在市/避险日占比与各自贡献
2. 行业: 回撤期每月调仓行业仓位 × 下月行业等权收益
3. 个股: 最大亏损贡献 TOP10 个股
4. s123 全期 in/out 年化收益对比（有效性检验）
"""
import os, time, sys
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")

# ========== 复用主引擎的准备逻辑（直接重算 scores + 信号）==========
import stock_gbdt_s123_backtest as _m
t0 = time.time()

score_ens = _m.score_ens          # {trade_date: Series<ts_code -> score>}
sig_df = _m.sig_df                # [ym, s1, s2, s3, s123]
cal_dates = _m.cal_dates          # 交易日 list
v8_daily = _m.v8_daily            # Series<date -> V8 日收益>
open_w = _m.open_w                # DataFrame<date, ts_code> 开盘价
close_w = _m.close_w              # 收盘价
p = _m.p                          # 面板 + is_traditional / industry
latest_members = _m.latest_members
select_with_limit = _m.select_with_limit
CANDIDATE_INDS = _m.CANDIDATE_INDS  # 传统行业列表
TOP_N = 40
MAX_IND = 4  # Top40 对应

# 调仓日 = 每月首个交易日
rebals = []
for ym in sorted(set(d // 100 for d in cal_dates)):
    mf = min(d for d in cal_dates if d // 100 == ym)
    rebals.append(mf)
month_last_map = {d // 100: d for d in sorted(p["trade_date"].unique())}
sig_map = sig_df["s123"].to_dict()

# 行业映射（panel 中有）
ind_map = p[["ts_code","industry"]].drop_duplicates().set_index("ts_code")["industry"].to_dict()

# ========== 先重放一遍交易，得到每日 NAV + 调仓明细 ==========
print("[回放] 重放 ENS_T40_S123_ONLY_S123 策略...")
state_in = False
positions = {}
cash = 0.0
reserve = 1.0e6
navs = []
hold_details = []   # 调仓日明细: [{date, ts_code, industry, weight, price, shares}]
daily_log = []      # 每日: {date, nav, state, pos_val, reserve}
prev_s123 = None

def pos_value_at(d):
    return sum(sh * (close_w.at[d, c] if (d in close_w.index and c in close_w.columns and not np.isnan(close_w.at[d, c])) else 0)
               for c, sh in positions.items())

for i, d in enumerate(cal_dates):
    ym = d // 100
    # 月末信号
    if d == rebals[0]:
        prev_s123 = sig_map.get(ym, 0)
    if i > 0 and cal_dates[i-1] // 100 != ym:
        prev_s123 = sig_map.get(cal_dates[i-1] // 100, 0)
    target_state = False
    if not state_in and prev_s123 is not None and prev_s123 >= 3:
        target_state = True
    elif state_in and prev_s123 is not None and prev_s123 <= 1:
        target_state = False
    else:
        target_state = state_in

    # V8 每日增值
    if d in v8_daily.index:
        reserve *= (1 + v8_daily.at[d])

    # 调仓日
    if d in rebals:
        # 先算股权
        snap_ym = d // 100 - 1
        snap = month_last_map.get(snap_ym)
        pool = None
        if snap is not None and score_ens.get(snap) is not None:
            pool = score_ens[snap]
            trad_codes = set(p.loc[(p["trade_date"] == snap) & (p["is_traditional"]), "ts_code"])
            members = latest_members(d)
            pool = pool[pool.index.isin(members) & pool.index.isin(trad_codes)]

        if target_state and not state_in:
            if pool is not None and len(pool):
                sel = select_with_limit(pool, MAX_IND, TOP_N)
                equity = cash + reserve
                stock_budget = equity
                reserve = 0.0
                cash = stock_budget
                positions = {}
                alloc = stock_budget / len(sel) if len(sel) else 0
                for c in sel:
                    o = open_w.at[d, c] if (d in open_w.index and c in open_w.columns) else np.nan
                    if np.isnan(o) or o <= 0:
                        continue
                    plim = (close_w.at[min(x for x in cal_dates if x < d), c]
                           * (0.9 if c[:3] in ("300","688") else 0.95)) if True else 0
                    # 简化：不管跌停（影响仓位记录但不影响归因）
                    sh = int(alloc / (o * 1.001) // 100 * 100)
                    if sh > 0 and cash >= sh * o * 1.001:
                        cash -= sh * o * 1.001
                        positions[c] = positions.get(c, 0) + sh
                if len(positions) > 0:
                    state_in = True
                # 记录持仓明细（按当日价值计权）
                tot_val = pos_value_at(d)
                for c, sh in positions.items():
                    v = sh * (close_w.at[d, c] if (d in close_w.index and c in close_w.columns and not np.isnan(close_w.at[d, c])) else 0)
                    hold_details.append({"date": d, "ts_code": c, "industry": ind_map.get(c, "?"),
                                         "shares": sh, "value": v, "weight": v / (tot_val + 1e-9)})
        elif not target_state and state_in:
            for c, sh in positions.items():
                o = open_w.at[d, c] if (d in open_w.index and c in open_w.columns) else np.nan
                if not np.isnan(o) and o > 0:
                    cash += sh * o * 0.999
            positions = {}
            reserve += cash
            cash = 0.0
            state_in = False
        elif target_state and state_in:
            if pool is not None and len(pool):
                sel = select_with_limit(pool, MAX_IND, TOP_N)
                equity = cash + reserve + pos_value_at(d)
                target_stock = equity
                # 卖不在目标
                for c in list(positions):
                    if c not in sel:
                        o = open_w.at[d, c] if (d in open_w.index and c in open_w.columns) else np.nan
                        if not np.isnan(o) and o > 0:
                            cash += positions[c] * o * 0.999
                        del positions[c]
                cur_val = pos_value_at(d)
                deficit = target_stock - cur_val
                if deficit > 0:
                    avail = min(reserve, deficit)
                    reserve -= avail
                    cash += avail
                alloc = target_stock / len(sel) if len(sel) else 0
                for c in sel:
                    o = open_w.at[d, c] if (d in open_w.index and c in open_w.columns) else np.nan
                    if np.isnan(o) or o <= 0:
                        continue
                    have = positions.get(c, 0) * (close_w.at[d, c] if (d in close_w.index and c in close_w.columns and not np.isnan(close_w.at[d, c])) else 0)
                    diff = alloc - have
                    if diff > 100:
                        sh = int(diff / (o * 1.001) // 100 * 100)
                        if sh > 0 and cash >= sh * o * 1.001:
                            cash -= sh * o * 1.001
                            positions[c] = positions.get(c, 0) + sh
                    elif diff < -100:
                        sh = int(-diff / (o * 0.999) // 100 * 100)
                        sh = min(sh, positions.get(c, 0))
                        if sh > 0:
                            cash += sh * o * 0.999
                            positions[c] -= sh
                            if positions[c] <= 0:
                                del positions[c]
                # 记录
                tot_val = pos_value_at(d)
                for c, sh in positions.items():
                    v = sh * (close_w.at[d, c] if (d in close_w.index and c in close_w.columns and not np.isnan(close_w.at[d, c])) else 0)
                    hold_details.append({"date": d, "ts_code": c, "industry": ind_map.get(c, "?"),
                                         "shares": sh, "value": v, "weight": v / (tot_val + 1e-9)})

    # MA5 跳过（sell_mode=S123_ONLY）
    reserve += cash
    cash = 0.0
    pv = pos_value_at(d)
    nav = reserve + pv
    navs.append(nav)
    daily_log.append({"date": d, "nav": nav, "state": "in" if state_in else "out",
                      "pos_val": pv, "reserve": reserve})

print(f"  回测 {len(cal_dates)} 交易日, {len(hold_details)} 条持仓明细记录, {len(set(h['date'] for h in hold_details))} 个有持仓的调仓日")
nav_s = pd.Series(navs, index=cal_dates).sort_index()
log_df = pd.DataFrame(daily_log).set_index("date")
state_ts = log_df["state"]

# ========== 1. 回撤区间识别 ==========
print("\n=== 1. 回撤区间识别（逐日复利） ===")
cum = nav_s / nav_s.iloc[0]
peak = cum.cummax()
dd = cum / peak - 1.0
max_dd = dd.min()
dd_end_date = int(dd.idxmin())
dd_start_date = int(peak.loc[:dd_end_date].idxmax())
print(f"  最大回撤 = {max_dd*100:.2f}%")
print(f"  起点 {dd_start_date} (累计净值峰 = {peak.loc[dd_end_date]:.4f})")
print(f"  终点 {dd_end_date} (当时净值 = {cum.loc[dd_end_date]:.4f})")
mask_dd = (log_df.index >= dd_start_date) & (log_df.index <= dd_end_date)
win = log_df[mask_dd]
print(f"  区间 {len(win)} 交易日. In市: {(win['state']=='in').sum()}日, 避险: {(win['state']=='out').sum()}日")

# 区间内 in/out 各自累计收益
ret_s = nav_s.pct_change().fillna(0)
in_idx = state_ts[(state_ts == "in") & mask_dd].index
out_idx = state_ts[(state_ts == "out") & mask_dd].index
# 只用 in 期的收益累乘（其他日置 0）
in_ret_series = ret_s.copy()
in_ret_series[~in_ret_series.index.isin(in_idx)] = 0
out_ret_series = ret_s.copy()
out_ret_series[~out_ret_series.index.isin(out_idx)] = 0
print(f"  回撤期 IN市 累计: {(in_ret_series.loc[dd_start_date:dd_end_date]+1).prod()-1:.4%}")
print(f"  回撤期 OUT避险 累计: {(out_ret_series.loc[dd_start_date:dd_end_date]+1).prod()-1:.4%}")

# ========== 2. 行业归因 ==========
print("\n=== 2. 回撤期行业归因（调仓行业仓位 × 下月行业等权收益） ===")
# 构建下月行业等权收益
panel = pd.read_parquet(PANEL)
panel["ym"] = panel["trade_date"] // 100
# 未来 1 月收益 = ret_1m.shift(-1)（按 ts_code 分组）
panel["fwd_1m"] = panel.sort_values("trade_date").groupby("ts_code")["ret_1m"].shift(-1)
ind_fwd = panel.groupby(["trade_date","industry"])["fwd_1m"].mean().reset_index()
ind_fwd.columns = ["trade_date","industry","ind_next_ret"]
# 调仓明细 DataFrame
hd = pd.DataFrame(hold_details)
if len(hd) == 0:
    print("  无持仓明细，跳过")
else:
    # 回撤区间覆盖的调仓日
    reb_in_dd = sorted([d for d in hd["date"].unique() if (d >= dd_start_date and d <= dd_end_date)])
    # 每个调仓日对应: trade_date(快照 ym_last), 行业仓位, 乘以下月行业收益
    contrib_rows = []
    for rd in reb_in_dd:
        h = hd[hd["date"] == rd]
        snap_ym = rd // 100 - 1
        snap_td = month_last_map.get(snap_ym)
        if snap_td is None: continue
        ind_w = h.groupby("industry")["weight"].sum()
        # ind_fwd: 对应快照 trade_date 那行的 ind_next_ret
        ir = ind_fwd[ind_fwd["trade_date"] == snap_td].set_index("industry")["ind_next_ret"]
        for ind in ind_w.index:
            w = ind_w[ind]
            r = ir.get(ind, np.nan)
            if pd.notna(r):
                contrib_rows.append({"date": rd, "industry": ind, "weight": w, "ind_next_ret": r, "contrib": w*r})
    cdf = pd.DataFrame(contrib_rows)
    if len(cdf) > 0:
        tot = cdf.groupby("industry")["contrib"].sum().sort_values()
        print("  各行业回撤期累计贡献度（负=拖累）:")
        for ind, c in tot.items():
            print(f"    {ind:>12}: {c*100:+.2f}%")
        print(f"  拖累TOP5合计: {tot.head(5).sum()*100:.2f}%")
        # HHI 集中度
        first_hd = reb_in_dd[0] if len(reb_in_dd) else sorted(hd["date"].unique())[0]
        pos0 = hd[hd["date"] == first_hd].groupby("industry")["weight"].sum()
        hhi = (pos0**2).sum()
        print(f"\n  回撤起始调仓日({first_hd}):")
        print(f"    行业数 = {len(pos0)}, HHI = {hhi:.4f} (HHI>0.18 高度集中)")
        print(f"    Top5 行业仓位: {dict(pos0.sort_values(ascending=False).head(5).round(3).to_dict())}")

# ========== 3. 个股归因 TOP10 ==========
print("\n=== 3. 回撤期个股贡献 TOP10（权重 × 下月个股收益近似） ===")
if len(hd) > 0:
    fwd_stock = panel.set_index(["trade_date","ts_code"])["fwd_1m"].to_dict()
    srows = []
    for rd in reb_in_dd:
        h = hd[hd["date"] == rd]
        snap_ym = rd // 100 - 1
        snap_td = month_last_map.get(snap_ym)
        if snap_td is None: continue
        for _, r in h.iterrows():
            r_fwd = fwd_stock.get((snap_td, r["ts_code"]), np.nan)
            if pd.notna(r_fwd):
                srows.append({"date": rd, "ts_code": r["ts_code"], "industry": r["industry"],
                              "weight": r["weight"], "stock_next_ret": r_fwd, "contrib": r["weight"]*r_fwd})
    sdf = pd.DataFrame(srows)
    if len(sdf) > 0:
        tot_s = sdf.groupby("ts_code")["contrib"].sum().sort_values()
        print("  TOP10 拖累个股:")
        for code, c in tot_s.head(10).items():
            ind = sdf[sdf["ts_code"]==code]["industry"].iloc[0]
            months = (sdf["ts_code"]==code).sum()
            print(f"    {code}({ind:>6}): 贡献 {c*100:+.2f}% (持有{months}个月)")
        print(f"  TOP10拖累合计: {tot_s.head(10).sum()*100:.2f}%")

# ========== 4. s123 择时有效性（全期） ==========
print("\n=== 4. s123 择时有效性（全期） ===")
def ann(s, n_days=242):
    if len(s) < 30: return np.nan
    return (s+1).prod()**(n_days/len(s)) - 1
r_full = ret_s
r_in = ret_s[state_ts[state_ts=="in"].index]
r_out = ret_s[state_ts[state_ts=="out"].index]
print(f"  全部期: 年化 {ann(r_full)*100:.2f}%, 波动率 {r_full.std()*15.55*100:.2f}% (√242≈15.55)")
print(f"  IN期  : {len(r_in)} 日, 年化 {ann(r_in)*100:.2f}%, 波动率 {r_in.std()*15.55*100:.2f}%")
print(f"  OUT期 : {len(r_out)} 日, 年化 {ann(r_out)*100:.2f}%, 波动率 {r_out.std()*15.55*100:.2f}%")
# 择时有效性的关键: OUT期波动率应该显著低，或收益高（说明逃顶/抄底成功）
print(f"  解读: 若OUT期年化 > IN期年化 - 说明s123择时失败(逃在上涨)")
print(f"        若OUT期波动显著低(<IN期波动的1/2) - 说明s123有效规避熊市")
# 补充 s123 信号历史
sig_show = sig_df.copy()
sig_show["ym"] = sig_show["ym"].astype(int)
print(f"\n  s123信号分布: {dict(sig_df['s123'].value_counts().sort_index().astype(int).to_dict())}")
# 特别查看回撤期 s123 信号
dd_ym_start = dd_start_date // 100
dd_ym_end = dd_end_date // 100
win_sig = sig_show[(sig_show["ym"]>=dd_ym_start)&(sig_show["ym"]<=dd_ym_end)]
print(f"  回撤期 s123 按月信号: \n{win_sig[['ym','s1','s2','s3','s123']].to_string(index=False)}")

print(f"\n总耗时 {time.time()-t0:.0f}s")
