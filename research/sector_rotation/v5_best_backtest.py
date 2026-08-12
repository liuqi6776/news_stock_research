# -*- coding: utf-8 -*-
"""
v5 最终版：网格搜索最优参数 (K3_ROE12_PEG2.0_CHIP50_MV50_YR3)
全区间 2020-01 ~ 2025-12 回测，输出指标 + 净值曲线
"""
import os, sys, time
import numpy as np
import pandas as pd

# ============================================================
# 1. 加载 v5 数据和函数（跳过配置执行部分）
# ============================================================
v5_path = os.path.join(os.path.dirname(__file__), 'backtest_stock_picking_v5.py')
v5_code = open(v5_path, encoding='utf-8').read()
marker = '# ============================================================\n# 8. v5'
code_before = v5_code.split(marker)[0]
exec(code_before)

# ============================================================
# 2. 网格搜索最优参数
# ============================================================
BEST = dict(
    global_top_k=3,                 # Top3
    max_same_sector=2,              # 同板块最多2只
    max_pe=60,
    min_turnover_pct=0.5,
    min_circ_mv_yi=50,              # 市值下限50亿
    max_circ_mv_yi=2000,
    max_peg=2.0,                    # PEG上限
    peg_preferred=1.5,
    min_roe_pct=12.0,               # ROE门槛12%
    chip_conc_pctl_threshold=0.50,  # 筹码前50%
    min_list_years=3,               # 上市满3年
    preferred_weight=1.2,
)
LABEL = "V5-BEST: Top3+ROE12%+PEG<2+筹码50%+MV50亿+上市3年"

print(f"\n{'='*80}")
print(f"v5 最终版回测: {LABEL}")
print(f"区间: 2020-01 ~ 2025-12 (数据截至)")
print(f"{'='*80}")

t0 = time.time()
nv, trs, picks = run_strategy_v5(**BEST)
print(f"回测完成, 耗时{time.time()-t0:.0f}s")

# ============================================================
# 3. 指标
# ============================================================
def full_metrics(nav_s, trades, label):
    if len(nav_s) < 30:
        return None
    tr = nav_s.iloc[-1] / INIT - 1
    yrs = (nav_s.index[-1] - nav_s.index[0]).days / 365.25
    ann = (1 + tr) ** (1/yrs) - 1 if yrs > 0 else np.nan
    pk = nav_s.cummax()
    mdd = ((nav_s - pk) / pk).min()
    ret = nav_s.pct_change().dropna()
    shp = ret.mean() / ret.std(ddof=1) * np.sqrt(252) if len(ret) > 1 and ret.std(ddof=1) > 0 else np.nan
    buys = sum(1 for t in trades if t[1]=="BUY")
    tps = sum(1 for t in trades if t[1]=="TP")
    tls = sum(1 for t in trades if t[1] not in ("BUY","TP"))
    hds = [t[5] for t in trades if t[1] != "BUY"]
    calmar = ann / abs(mdd) if mdd != 0 else np.nan
    return {"配置": label, "年化": ann, "累计": tr, "回撤": mdd, "夏普": shp,
            "Calmar": calmar, "买入": buys, "止盈": tps, "时间止损": tls,
            "止盈率": tps/(buys+1e-9), "平均持仓天": np.mean(hds) if hds else 0,
            "期末(万)": nav_s.iloc[-1]/1e4}

m = full_metrics(nv, trs, LABEL)
print(f"\n{'='*60}")
print(f"  {LABEL}")
print(f"  年化 {m['年化']:.1%}  累计 {m['累计']:.1%}  回撤 {m['回撤']:.1%}  夏普 {m['夏普']:.2f}  Calmar {m['Calmar']:.2f}")
print(f"  买入{m['买入']}  止盈{m['止盈']}  时间止损{m['时间止损']}  止盈率{m['止盈率']:.0%}  平均持仓{m['平均持仓天']:.0f}天  期末{m['期末(万)']:.0f}万")
print(f"{'='*60}")

# 分年度
print(f"\n  分年度表现:")
years = sorted(set(d.year for d in nv.index))
for y in years:
    yv = nv[nv.index.year == y]
    if len(yv) == 0: continue
    y_tr = yv.iloc[-1] / (yv.iloc[0] if len(yv) > 1 else INIT) - 1
    y_pk = yv.cummax()
    y_mdd = ((yv - y_pk) / y_pk).min()
    print(f"    {y}: 收益{y_tr:+.1%}  最大回撤{y_mdd:.1%}")

# ============================================================
# 4. 画净值曲线
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei","Microsoft YaHei","Arial Unicode MS"]
rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
# 净值
s_norm = nv / nv.iloc[0]
axes[0].plot(s_norm.index, s_norm.values, lw=2, color="#c0392b")
axes[0].set_title(f"v5 网格最优策略净值曲线 (初始100万)\n{LABEL}\n"
                  f"年化{m['年化']:.1%} | 回撤{m['回撤']:.1%} | 夏普{m['夏普']:.2f} | 期末{m['期末(万)']:.0f}万", fontsize=12)
axes[0].axhline(1.0, color="gray", lw=0.6, ls=":")
axes[0].set_ylabel("净值(倍数)")
axes[0].grid(alpha=0.3)
# 回撤
dd = s_norm / s_norm.cummax() - 1
axes[1].fill_between(dd.index, dd.values, 0, color="#e74c3c", alpha=0.4)
axes[1].set_title(f"回撤曲线 (最大回撤 {m['回撤']:.1%})", fontsize=12)
axes[1].set_ylabel("回撤")
axes[1].grid(alpha=0.3)
plt.tight_layout()
fig_path = os.path.join(OUT_DIR, "v5_best_nav_2020_2026.png")
plt.savefig(fig_path, dpi=120, bbox_inches="tight")
print(f"\n净值曲线: {fig_path}")

# 保存指标
pd.DataFrame([m]).to_csv(os.path.join(OUT_DIR, "v5_best_metrics.csv"),
                         index=False, encoding="utf-8-sig")

# 保存净值序列
nv.to_csv(os.path.join(OUT_DIR, "v5_best_nav.csv"), encoding="utf-8-sig")

# 选股明细
picks_rows = []
for ym, codes in sorted(picks.items()):
    if codes:
        for c in codes:
            picks_rows.append({"ym": ym, "ts_code": c})
if picks_rows:
    pd.DataFrame(picks_rows).to_csv(os.path.join(OUT_DIR, "v5_best_picks.csv"),
                                    index=False, encoding="utf-8-sig")
    print(f"选股明细: {os.path.join(OUT_DIR, 'v5_best_picks.csv')} ({len(picks_rows)}条)")

print(f"\n总耗时 {time.time()-t0:.0f}s")
