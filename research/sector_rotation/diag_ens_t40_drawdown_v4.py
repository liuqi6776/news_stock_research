# -*- coding: utf-8 -*-
"""回撤归因 v4 (只读 pkl + panel，不 import 主引擎)"""
import os, time, pickle
import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
PKL = os.path.join(ROOT, "research", "sector_rotation", "results", "stock_gbdt_s123_results.pkl")
PANEL = os.path.join(ROOT, "research", "sector_rotation", "stock_ml_panel_72m.parquet")
t0 = time.time()

with open(PKL, "rb") as f:
    data = pickle.load(f)
res = data["results"]
t7 = data["t7"]
print(f"[读取] {len(res)} 个策略结果")

# ---------- 1. ENS_T40 + T7 + ENS_T60_TV12 的 nav_dated ----------
tags = ["ENS_T40_S123_ONLY_S123", "ENS_T60_S123_TV12", "ENH_T40_S123_ONLY_S123", "GBDT_T40_S123_ONLY_S123"]
navs = {}
logs = {}
for t in tags:
    r = res[t]
    nd = r["nav_dated"].sort_index() if hasattr(r["nav_dated"], "sort_index") else r["nav_dated"]
    navs[t] = nd
    logs[t] = r["log"]
    lcols = list(r["log"].columns) if r["log"] is not None else None
    print(f"  {t}: len(nav)={len(nd)}, 初始={nd.iloc[0]:.2f}, 期末={nd.iloc[-1]:.2f}, log_cols={lcols}")
    if lcols is None and len(nd):
        print(f"     (log 为 None，使用 nav_s.index 推断每日 state，state 归因将跳过敏捷状态拆分)

# ---------- 2. 回撤识别 + in/out 状态拆分 ----------
def _fmt(x):
    if hasattr(x, "strftime"): return x.strftime("%Y-%m-%d")
    s = str(int(x)) if not pd.isna(x) else str(x)
    return f"{s[:4]}-{s[4:6]}-{s[6:]}" if len(s)==8 else s

def diag_dd(tag, nav_s, log_df):
    print(f"\n{'='*60}\n=== {tag} 回撤归因 ===\n{'='*60}")
    cum = nav_s / nav_s.iloc[0]
    peak = cum.cummax()
    dd = cum / peak - 1.0
    maxdd = dd.min()
    dd_end = dd.idxmin()
    dd_start = peak.loc[:dd_end].idxmax()
    # 日收益
    ret_s = nav_s.pct_change().fillna(0)
    # state
    if log_df is not None and len(log_df):
        if "state" in log_df.columns:
            state_col = "state"
        elif "timing" in log_df.columns:
            state_col = "timing"
        else:
            state_col = None
        if state_col:
            # 找到 date 列 (log 里 date 是 int 20200203)
            ldf = log_df.copy()
            if "date" in ldf.columns:
                ldf = ldf.set_index("date")
            # 转成 bool: 不管 True/False 还是 'in'/'out'
            s = ldf[state_col]
            s_bool = s.astype(bool) if s.dtype == bool else s.isin(["in", True, 1, "True"])
            # reindex 对齐 nav_s
            state = s_bool.reindex(nav_s.index, method="ffill").fillna(False)
        else:
            state = pd.Series(True, index=nav_s.index)
    else:
        state = pd.Series(True, index=nav_s.index)
    n_in = state.sum()
    n_out = (~state).sum()
    print(f"  最大回撤 = {maxdd*100:.2f}%")
    print(f"  区间  : {_fmt(dd_start)} → {_fmt(dd_end)}")
    print(f"  In市日数: {n_in}, 避险日数: {n_out}")
    # 全期 in/out 年化
    def ann(ss, n=242):
        if len(ss) < 30: return np.nan
        return (ss+1).prod()**(n/len(ss)) - 1
    r_in = ret_s[state[state].index]
    r_out = ret_s[state[~state].index]
    print(f"\n  [全期]")
    print(f"    In期  : 年化 {ann(r_in)*100:.2f}%, 波动率 {r_in.std()*15.55*100:.2f}%, 天数 {len(r_in)}")
    print(f"    Out期 : 年化 {ann(r_out)*100:.2f}%, 波动率 {r_out.std()*15.55*100:.2f}%, 天数 {len(r_out)}")
    if (n_in + n_out) > 0:
        print(f"    解读: Out波动/In波动 = {r_out.std()/(r_in.std()+1e-9):.2f}倍 (<0.5 说明避险有效降低波动)")
        print(f"          In年化 - Out年化 = {(ann(r_in)-ann(r_out))*100:+.2f}% (正=择时正确吃到涨/避开跌)")
    # 回撤期内拆分
    mask = (ret_s.index >= min(dd_start, dd_end)) & (ret_s.index <= max(dd_start, dd_end))
    rw_ret = ret_s[mask]
    rw_state_in = state[(state)&mask].index
    rw_state_out = state[(~state)&mask].index
    rwi = ret_s.copy(); rwi[~rwi.index.isin(rw_state_in)] = 0
    rwo = ret_s.copy(); rwo[~rwo.index.isin(rw_state_out)] = 0
    in_cum = (rwi[mask]+1).prod()-1
    out_cum = (rwo[mask]+1).prod()-1
    print(f"\n  [回撤期 {_fmt(dd_start)}~{_fmt(dd_end)}]")
    print(f"    In市  累计收益: {in_cum*100:.2f}% (占该交易日 {(state[mask]).sum()}/{len(mask)} 日)")
    print(f"    避险  累计收益: {out_cum*100:.2f}% (占该交易日 {(~state[mask]).sum()}/{len(mask)} 日)")
    print(f"    净值总变化:     {(rw_ret+1).prod()-1:.4%}")
    return {"tag": tag, "maxdd": maxdd, "dd_start": dd_start, "dd_end": dd_end,
            "in_days": n_in, "out_days": n_out,
            "in_ann": ann(r_in), "out_ann": ann(r_out),
            "in_vol": r_in.std(), "out_vol": r_out.std()}

# 逐个诊断
outs = []
for t in tags:
    outs.append(diag_dd(t, navs[t], logs[t]))

# T7 单独（它是月频，没有 state，但能看回撤）
if hasattr(t7, "index"):
    t7nav = t7.sort_index() if hasattr(t7, "sort_index") else t7
    cum = t7nav / t7nav.iloc[0]
    peak = cum.cummax()
    dd = cum / peak - 1.0
    print(f"\n{'='*60}\n=== T7 ETF对照 回撤 ===\n{'='*60}")
    print(f"  最大回撤 = {dd.min()*100:.2f}%")
    print(f"  区间 {_fmt(peak.loc[:dd.idxmin()].idxmax())} → {_fmt(dd.idxmin())}")

# ---------- 3. 参考组合对比（策略回撤 vs 市场β） ----------
print(f"\n{'='*60}\n=== 参考组合对比：是市场β？还是策略选股问题？ ===\n{'='*60}")
panel = pd.read_parquet(PANEL)
# 行业列表
inds = sorted(panel["industry"].dropna().unique())
TECH = {"电子","计算机","通信","传媒","国防军工","医药","电力设备及新能源","机械","汽车"}
TRAD = [i for i in inds if i not in TECH]
panel["is_trad"] = panel["industry"].isin(TRAD)
# 构建 3 个参考组合的月度 fwd_20
cols = {
    "全A等权": (lambda g: g["fwd_20"].mean()),
    "传统行业等权": (lambda g: g[g["is_trad"]]["fwd_20"].mean()),
    "传统行业Top40等权(按 momentum_20 排序)": (lambda g: g[g["is_trad"]].nlargest(40,"momentum_20")["fwd_20"].mean() if len(g[g["is_trad"]])>=40 else np.nan),
}
grp = panel.dropna(subset=["fwd_20"]).groupby("trade_date")
res_rows = []
for dt, g in grp:
    row = {"trade_date": dt}
    for k, fn in cols.items():
        try: row[k] = fn(g)
        except: row[k] = np.nan
    res_rows.append(row)
bm = pd.DataFrame(res_rows).set_index("trade_date").sort_index().dropna(how="all")
# 累积
cum_df = (1 + bm.fillna(0)).cumprod()
# 回撤
dd_df = cum_df / cum_df.cummax() - 1.0
print(f"  参考组合 MaxDD (全期 72 月):")
for c in bm.columns:
    print(f"    {c}: MaxDD={dd_df[c].min()*100:.2f}%, 全期收益={(cum_df[c].iloc[-1]-1)*100:.2f}%")

# 同一回撤期（取 ENS_T40 的回撤期）下，参考组合跌多少
r0 = outs[0]  # ENS_T40
st, ed = r0["dd_start"], r0["dd_end"]
# 对齐月份（取最小的包含区间）
bm_win_mask = (bm.index.astype(int) >= int(st)) & (bm.index.astype(int) <= int(ed))
if bm_win_mask.sum() == 0:
    # 因为 panel 的 trade_date 是月末（如 20220429），而 dd_start/end 可能是任意日（如 20211206）
    # 转 ym 取整
    st_ym = int(str(int(st))[:6]) if not hasattr(st,"year") else st.year*100+st.month
    ed_ym = int(str(int(ed))[:6]) if not hasattr(ed,"year") else ed.year*100+ed.month
    bm_ym = (bm.index // 100).astype(int)
    bm_win_mask = (bm_ym >= st_ym) & (bm_ym <= ed_ym)
win_cum = (1 + bm[bm_win_mask].fillna(0)).cumprod()
# 起点=1（归一化到回撤起点），然后看最低点
print(f"\n  与 ENS_T40 同期 ({_fmt(st)}~{_fmt(ed)}):")
for c in bm.columns:
    wc = win_cum[c] / win_cum[c].iloc[0]
    dd = (wc / wc.cummax() - 1).min()
    chg = wc.iloc[-1] - 1
    print(f"    {c}: 同期MaxDD={dd*100:.2f}%, 期末累计={chg*100:.2f}%")
print(f"\n  ENS_T40 同期回撤: {r0['maxdd']*100:.2f}%")
print(f"\n  解读: 如果策略回撤 ≈ 传统行业等权回撤 → 是市场β（普跌）")
print(f"        如果策略回撤显著更小 → 说明选股/择时有效抗跌")
print(f"        如果策略回撤显著更大 → 回撤来自行业集中度/个股选得差")

print(f"\n总耗时 {time.time()-t0:.0f}s")
