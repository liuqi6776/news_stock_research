# -*- coding: utf-8 -*-
"""回撤归因 v5（最终版：不依赖 log state，用 nav_dated + 面板参考组合 + ALWAYS 对照）

回答用户问题 "回撤为什么这么高？"：
1. 回撤区间日期 + 同期 ALWAYS(无s123) 回撤对比 → 看 s123 有没有帮降低回撤
2. 同期参考组合(传统行业等权/全A等权)回撤 → 看是不是市场β
3. T7 ETF 同期回撤对比 → 看是不是行业池问题
4. 逐年收益表，看哪些年拖后腿
"""
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
t7_nav = data["t7"]
TAGS = ["ENS_T40_S123_ONLY_S123", "ENS_T40_S123_ONLY_ALWAYS",
        "ENH_T40_S123_ONLY_S123", "GBDT_T40_S123_ONLY_S123",
        "ENS_T60_S123_TV12", "ENS_T60_S123_ONLY_S123"]

# ---------- 工具 ----------
def _fmt(x):
    s = str(int(x)) if not pd.isna(x) else str(x)
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"

def maxdd_info(nav_s, name):
    cum = nav_s / nav_s.iloc[0]
    peak = cum.cummax()
    dd = cum / peak - 1
    end = dd.idxmin()
    start = peak.loc[:end].idxmax()
    return {"name": name, "maxdd": dd.min(), "start": int(start), "end": int(end),
            "start_NAV": float(peak.loc[end]), "end_NAV": float(cum.loc[end]),
            "n_days": int(((pd.Timestamp(_fmt(end))-pd.Timestamp(_fmt(start))).days))}

# 载入所有需要的 nav
navs = {}
for t in TAGS:
    nd = res[t]["nav_dated"].copy().sort_index()
    navs[t] = nd.astype(float)
# T7（可能是 DataFrame 有 nav/ret 多列，也可能是 Series；index 可能是 "202001"字符串 / ym int）
if hasattr(t7_nav, "columns"):
    # DataFrame: 取 nav 列，把 index 变成 int ym
    t7_s = t7_nav["nav"].copy().astype(float)
else:
    t7_s = t7_nav.copy().astype(float)
# index 统一化：如果是字符串如 "202001" → int 202001
if isinstance(t7_s.index[0], str):
    try:
        t7_s.index = [int(s) for s in t7_s.index]
    except:
        pass
t7_idx = t7_s.index
print(f"[载入] {len(TAGS)} 策略, T7 长度={len(t7_s)}, T7 index type={type(t7_idx[0])}, sample={list(t7_idx[:3])}")

# ---------- 1. 回撤基本信息（策略+对照） ----------
print("\n" + "="*72)
print("=== 1. 各策略最大回撤区间对比（看 s123/TV 是否有效降低回撤） ===")
print("="*72)
rows_dd = []
for t in TAGS:
    info = maxdd_info(navs[t], t)
    rows_dd.append(info)
# T7
cum = t7_s / t7_s.iloc[0]
peak = cum.cummax()
dd = cum / peak - 1
end = dd.idxmin()
start = peak.loc[:end].idxmax()
rows_dd.append({"name": "T7_ETF", "maxdd": dd.min(), "start": start, "end": end,
                "start_NAV": float(peak.loc[end]), "end_NAV": float(cum.loc[end]),
                "n_days": 0})  # T7 月频，天数无意义
df_dd = pd.DataFrame(rows_dd).sort_values("maxdd")
df_dd[["maxdd_pct"]] = df_dd[["maxdd"]] * 100
show = df_dd[["name","maxdd_pct","start","end","n_days"]].copy()
show["start"] = show["start"].apply(_fmt)
show["end"]   = show["end"].apply(_fmt)
show.columns = ["策略","最大回撤(%)","起点","终点","持续天数"]
print(show.to_string(index=False, float_format="%.2f"))
print(f"\n  关键对照:")
print(f"    S123 vs ALWAYS 回撤差距: {df_dd[df_dd['name']=='ENS_T40_S123_ONLY_S123']['maxdd_pct'].iloc[0]-df_dd[df_dd['name']=='ENS_T40_S123_ONLY_ALWAYS']['maxdd_pct'].iloc[0]:.2f}pp (正=S123更优)")
print(f"    ENS vs ENH    回撤差距: {df_dd[df_dd['name']=='ENS_T40_S123_ONLY_S123']['maxdd_pct'].iloc[0]-df_dd[df_dd['name']=='ENH_T40_S123_ONLY_S123']['maxdd_pct'].iloc[0]:.2f}pp (正=ENS更优)")
print(f"    TV12 vs 无TV  回撤差距: {df_dd[df_dd['name']=='ENS_T60_S123_TV12']['maxdd_pct'].iloc[0]-df_dd[df_dd['name']=='ENS_T60_S123_ONLY_S123']['maxdd_pct'].iloc[0]:.2f}pp (正=TV更优)")

# ---------- 2. 参考组合同期回撤（以 ENS_T40 的回撤期为准） ----------
print("\n" + "="*72)
print("=== 2. 参考组合同期回撤对比（以 ENS_T40_S123 回撤期为基准）===")
print("="*72)
r0 = df_dd[df_dd["name"]=="ENS_T40_S123_ONLY_S123"].iloc[0]
st, ed = int(r0["start"]), int(r0["end"])
st_ym, ed_ym = st//100, ed//100
print(f"  基准区间: {_fmt(st)} ~ {_fmt(ed)} (ym {st_ym}→{ed_ym})")
# 加载面板构建参考组合
panel = pd.read_parquet(PANEL)
inds = sorted(panel["industry"].dropna().unique())
TECH = {"电子","计算机","通信","传媒","国防军工","医药","电力设备及新能源","机械","汽车"}
panel["is_trad"] = ~panel["industry"].isin(TECH)
print(f"  行业总数={len(inds)}, 传统行业={panel['is_trad'].sum()/len(panel)*100:.1f}%")

# 各参考组合月度 fwd_20 收益
res_rows = []
for dt, g in panel.dropna(subset=["fwd_20"]).groupby("trade_date"):
    tg = g[g["is_trad"]]
    t40 = tg.nlargest(40, "momentum_20")["fwd_20"].mean() if len(tg)>=40 else np.nan
    res_rows.append({
        "trade_date": int(dt),
        "全A等权": g["fwd_20"].mean(),
        "传统行业等权": tg["fwd_20"].mean(),
        "传统Top40动量": t40,
    })
bm = pd.DataFrame(res_rows).set_index("trade_date").sort_index()

# 取同期，计算回撤
bm_ym = bm.index // 100
mask_bm = (bm_ym >= st_ym) & (bm_ym <= ed_ym)
if mask_bm.sum() == 0:
    # 扩大一个月前后
    mask_bm = (bm_ym >= st_ym-1) & (bm_ym <= ed_ym+1)
bm_win = (1 + bm[mask_bm].fillna(0)).cumprod()
# 对照策略在同期回撤（取对应日频）
mask_strat = (navs["ENS_T40_S123_ONLY_S123"].index.astype(int) >= st) & (navs["ENS_T40_S123_ONLY_S123"].index.astype(int) <= ed)

def _win_dd(nav_s, s, e):
    mask = (nav_s.index.astype(int) >= s) & (nav_s.index.astype(int) <= e)
    win = nav_s[mask]
    if len(win)==0: return np.nan, np.nan
    wc = win / win.iloc[0]
    dd = (wc / wc.cummax() - 1).min()
    chg = wc.iloc[-1] - 1
    return float(dd), float(chg)

rows_comp = []
for col in bm.columns:
    wc = bm_win[col] / bm_win[col].iloc[0] if len(bm_win) > 0 else bm_win[col]
    dd = (wc / wc.cummax() - 1).min() if len(wc) else np.nan
    chg = wc.iloc[-1] - 1 if len(wc) else np.nan
    rows_comp.append({"name": f"参考/{col}", "dd": dd, "chg": chg})
# 策略
for t in ["ENS_T40_S123_ONLY_S123","ENS_T40_S123_ONLY_ALWAYS"]:
    dd, chg = _win_dd(navs[t], st, ed)
    rows_comp.append({"name": t, "dd": dd, "chg": chg})
# T7 同月频 (按 ym 取交集)
mask7 = (t7_s.index >= st_ym) & (t7_s.index <= ed_ym)
win = t7_s[mask7]
if len(win) > 0:
    wc = win / win.iloc[0]
    dd7 = (wc / wc.cummax() - 1).min()
    chg7 = wc.iloc[-1] - 1
    rows_comp.append({"name": "T7 ETF", "dd": dd7, "chg": chg7})

dfc = pd.DataFrame(rows_comp).sort_values("dd")
dfc["dd_pct"] = dfc["dd"]*100
dfc["chg_pct"] = dfc["chg"]*100
print("\n  同期回撤/收益对比（越小越优）:")
print(dfc[["name","dd_pct","chg_pct"]].to_string(index=False, float_format="%.2f", col_space=30))
print(f"""
  解读:
    · 如果策略 ENS_T40 回撤 ≈ 传统行业等权  → 回撤是 β 性质（行业池普跌，非策略原因）
    · 如果 ALWAYS 回撤显著大 → 说明 s123 有效降低了回撤
    · 如果参考/全A等权 回撤显著小 → 说明传统行业池本身比全市场脆弱
""")

# ---------- 3. 逐年收益/回撤（看哪年拖累） ----------
print("="*72)
print("=== 3. 逐年收益（判断哪一年拖后腿导致 MaxDD）===")
print("="*72)
def yearly(nav_s, name):
    out = []
    if hasattr(nav_s.index[0], "year"):
        years = [x.year for x in nav_s.index]
    else:
        years = [int(str(int(x))[:4]) for x in nav_s.index]
    df = pd.DataFrame({"nav": nav_s.values, "y": years})
    for y, g in df.groupby("y"):
        if len(g) < 10: continue
        ret = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1
        cum = g["nav"] / g["nav"].iloc[0]
        dd = (cum / cum.cummax() - 1).min()
        out.append({"name": name, "year": y, "ret": ret, "dd": dd})
    return out

rows_y = []
for t in ["ENS_T40_S123_ONLY_S123","ENS_T40_S123_ONLY_ALWAYS","ENS_T60_S123_TV12"]:
    rows_y.extend(yearly(navs[t], t))
# T7
if len(t7_s):
    years = [int(str(int(x))[:6])//100 if isinstance(x,(int,np.integer)) and len(str(int(x)))>=6 else (x.year if hasattr(x,'year') else int(str(x)[:4])) for x in t7_s.index]
    df = pd.DataFrame({"nav": t7_s.values, "y": years})
    for y, g in df.groupby("y"):
        if len(g) < 2: continue
        ret = g["nav"].iloc[-1] / g["nav"].iloc[0] - 1
        cum = g["nav"] / g["nav"].iloc[0]
        dd = (cum / cum.cummax() - 1).min()
        rows_y.append({"name": "T7 ETF", "year": y, "ret": ret, "dd": dd})

dy = pd.DataFrame(rows_y)
dy_p = dy.pivot(index="year", columns="name", values="ret")
dy_p = (dy_p * 100).round(2)
dy_dd = dy.pivot(index="year", columns="name", values="dd")
dy_dd = (dy_dd * 100).round(2)
print("\n  [年度收益 %]")
print(dy_p.fillna("-").to_string())
print("\n  [年度最大回撤 %]")
print(dy_dd.fillna("-").to_string())

# 找出 ENS_T40 最坏年
et = dy[dy["name"]=="ENS_T40_S123_ONLY_S123"].sort_values("ret")
if len(et):
    worst = et.iloc[0]
    print(f"\n  ENS_T40 最坏年份: {int(worst['year'])} 年收益 {worst['ret']*100:.2f}%, 年内MaxDD {worst['dd']*100:.2f}%")
    # 再找 ALWAYS 在同年
    wa = dy[(dy["name"]=="ENS_T40_S123_ONLY_ALWAYS") & (dy["year"]==worst["year"])]
    wt7 = dy[(dy["name"]=="T7 ETF") & (dy["year"]==worst["year"])]
    if len(wa):
        print(f"  同年 ALWAYS 收益: {wa.iloc[0]['ret']*100:.2f}%, 年内MaxDD {wa.iloc[0]['dd']*100:.2f}% → s123 差值 = {(wa.iloc[0]['ret']-worst['ret'])*100:+.2f}%")
    if len(wt7):
        print(f"  同年 T7 收益   : {wt7.iloc[0]['ret']*100:.2f}%, 年内MaxDD {wt7.iloc[0]['dd']*100:.2f}%")
    print(f"  → 判断最坏年份的行情结构，通常是熊市/政策底/风格切换")

print(f"\n总耗时 {time.time()-t0:.0f}s")
