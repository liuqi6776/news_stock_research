# -*- coding: utf-8 -*-
"""
v5 真正的 OOS 测试
- 选参只用训练期(2020-2024) Sharpe，完全不看 2025
- 冻结后直接在 2025 上评估
- 与之前的"稳健分"(train×val, 含2025信息)选参做对照
"""
import os, sys, time
import numpy as np
import pandas as pd

t0 = time.time()
SR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SR, "results")

# ============================================================
# 1. 读网格搜索结果
# ============================================================
grid_csv = os.path.join(OUT, "v5_grid_search.csv")
df = pd.read_csv(grid_csv)
print(f"网格搜索结果: {len(df)} 组配置")

# ============================================================
# 2. 两种选参方式
# ============================================================
# A. 真正OOS: 只按训练期Sharpe选（不看2025）
df_train = df.sort_values("训练_夏普", ascending=False).reset_index(drop=True)
best_oos = df_train.iloc[0]

# B. 之前的"稳健分"(含2025信息)
df_robust = df[df["稳健分"] > 0].sort_values("稳健分", ascending=False).reset_index(drop=True)
best_robust = df_robust.iloc[0]

print(f"\n{'='*80}")
print("【选参方式对比】")
print(f"{'='*80}")
print(f"\nA. 真正OOS（仅训练Sharpe选参）: {best_oos['配置']}")
print(f"   训练: 年化{best_oos['训练_年化']:.1%} 回撤{best_oos['训练_回撤']:.1%} 夏普{best_oos['训练_夏普']:.2f}")
print(f"   验证: 年化{best_oos['验证_年化']:.1%} 回撤{best_oos['验证_回撤']:.1%} 夏普{best_oos['验证_夏普']:.2f}")

print(f"\nB. 稳健分选参（含2025信息）: {best_robust['配置']}")
print(f"   训练: 年化{best_robust['训练_年化']:.1%} 回撤{best_robust['训练_回撤']:.1%} 夏普{best_robust['训练_夏普']:.2f}")
print(f"   验证: 年化{best_robust['验证_年化']:.1%} 回撤{best_robust['验证_回撤']:.1%} 夏普{best_robust['验证_夏普']:.2f}")

# ============================================================
# 3. 解析配置字符串 -> 参数
# ============================================================
def parse_config(label):
    """K3_ROE12_PEG2.0_CHIP50_MV50_YR3 -> dict"""
    parts = label.split("_")
    p = {}
    for part in parts:
        if part.startswith("K"):
            p["global_top_k"] = int(part[1:])
        elif part.startswith("ROE"):
            p["min_roe_pct"] = int(part[3:])
        elif part.startswith("PEG"):
            p["max_peg"] = float(part[3:])
        elif part.startswith("CHIP"):
            p["chip_conc_pctl_threshold"] = int(part[4:]) / 100
        elif part.startswith("MV"):
            p["min_circ_mv_yi"] = int(part[2:])
        elif part.startswith("YR"):
            p["min_list_years"] = int(part[2:])
    return p

# ============================================================
# 4. 加载 v5 引擎并回测两种配置
# ============================================================
v5_path = os.path.join(SR, 'backtest_stock_picking_v5.py')
v5_code = open(v5_path, encoding='utf-8').read()
marker = '# ============================================================\n# 8. v5'
code_before = v5_code.split(marker)[0]
ns = {}
exec(code_before, ns)
run_strategy_v5 = ns["run_strategy_v5"]
INIT = ns["INIT"]

configs = {
    "OOS-TrainOnly": parse_config(best_oos["配置"]),
    "Robust-含2025": parse_config(best_robust["配置"]),
}

SPLIT = pd.Timestamp('2025-01-01')

print(f"\n{'='*80}")
print("【回测两种配置的全期净值，拆出2025 OOS】")
print(f"{'='*80}\n")

all_nav = {}

for name, p in configs.items():
    print(f"回测 {name}: K{p['global_top_k']} ROE{p['min_roe_pct']} PEG{p['max_peg']} "
          f"CHIP{int(p['chip_conc_pctl_threshold']*100)} MV{p['min_circ_mv_yi']} YR{p['min_list_years']}")
    nv, trs, _ = run_strategy_v5(
        global_top_k=p['global_top_k'],
        max_same_sector=2,
        max_pe=60,
        min_turnover_pct=0.5,
        min_circ_mv_yi=p['min_circ_mv_yi'],
        max_circ_mv_yi=2000,
        max_peg=p['max_peg'],
        peg_preferred=1.5,
        min_roe_pct=p['min_roe_pct'],
        chip_conc_pctl_threshold=p['chip_conc_pctl_threshold'],
        min_list_years=p['min_list_years'],
        preferred_weight=1.2,
        verbose=False,
    )
    all_nav[name] = nv

    # 全期 (rebase to 1.0 for CAGR)
    nv_rb = nv / nv.iloc[0]
    yrs_full = (nv.index[-1] - nv.index[0]).days / 365.25
    cagr_full = nv_rb.iloc[-1] ** (1/yrs_full) - 1
    mdd_full = ((nv_rb - nv_rb.cummax()) / nv_rb.cummax()).min()

    # 训练期 (2020-01 ~ 2024-12)
    tr_mask = nv.index < SPLIT
    nv_tr = nv[tr_mask]
    yrs_tr = (nv_tr.index[-1] - nv_tr.index[0]).days / 365.25
    cagr_tr = (nv_tr.iloc[-1] / nv_tr.iloc[0]) ** (1/yrs_tr) - 1
    mdd_tr = ((nv_tr - nv_tr.cummax()) / nv_tr.cummax()).min()
    ret_tr = nv_tr.pct_change().dropna()
    shp_tr = ret_tr.mean() / ret_tr.std(ddof=1) * np.sqrt(252) if len(ret_tr) > 1 else np.nan

    # 2025 OOS — rebase到1.0
    nv_oos = nv[~tr_mask].copy()
    if len(nv_oos) > 0:
        nv_oos = nv_oos / nv_oos.iloc[0]
        yrs_oos = (nv_oos.index[-1] - nv_oos.index[0]).days / 365.25
        cagr_oos = nv_oos.iloc[-1] ** (1/yrs_oos) - 1 if yrs_oos > 0 else np.nan
        mdd_oos = ((nv_oos - nv_oos.cummax()) / nv_oos.cummax()).min()
        ret_oos = nv_oos.pct_change().dropna()
        shp_oos = ret_oos.mean() / ret_oos.std(ddof=1) * np.sqrt(252) if len(ret_oos) > 1 else np.nan
        cum_oos = nv_oos.iloc[-1] - 1
    else:
        cagr_oos = mdd_oos = shp_oos = cum_oos = np.nan

    print(f"  全期:  年化{cagr_full:.1%} 回撤{mdd_full:.1%} 净值{nv.iloc[-1]:.3f}")
    print(f"  训练:  年化{cagr_tr:.1%} 回撤{mdd_tr:.1%} 夏普{shp_tr:.2f}")
    print(f"  OOS:   年化{cagr_oos:.1%} 回撤{mdd_oos:.1%} 夏普{shp_oos:.2f} 累计{cum_oos:.1%}")
    print()

# ============================================================
# 5. 训练Top5 的 2025 OOS 汇总（不只看第一名）
# ============================================================
print(f"\n{'='*80}")
print("【训练Sharpe Top5 在 2025 OOS 的表现】")
print(f"{'='*80}\n")
print(f"{'配置':<40} {'训练Sharpe':>10} {'训练年化':>10} {'OOS年化':>10} {'OOS回撤':>10} {'OOS夏普':>10}")
print("-" * 95)

for i in range(min(5, len(df_train))):
    row = df_train.iloc[i]
    cfg = parse_config(row["配置"])
    # 从网格CSV直接读验证期指标（已算好）
    print(f"{row['配置']:<40} {row['训练_夏普']:>10.2f} {row['训练_年化']:>10.1%} "
          f"{row['验证_年化']:>10.1%} {row['验证_回撤']:>10.1%} {row['验证_夏普']:>10.2f}")

# ============================================================
# 6. 稳健分Top5 对照
# ============================================================
print(f"\n{'='*80}")
print("【稳健分 Top5 在 2025 的表现（对照：含2025信息选参）】")
print(f"{'='*80}\n")
print(f"{'配置':<40} {'训练Sharpe':>10} {'训练年化':>10} {'OOS年化':>10} {'OOS回撤':>10} {'OOS夏普':>10} {'稳健分':>10}")
print("-" * 100)

for i in range(min(5, len(df_robust))):
    row = df_robust.iloc[i]
    print(f"{row['配置']:<40} {row['训练_夏普']:>10.2f} {row['训练_年化']:>10.1%} "
          f"{row['验证_年化']:>10.1%} {row['验证_回撤']:>10.1%} {row['验证_夏普']:>10.2f} {row['稳健分']:>10.2f}")

# ============================================================
# 7. 存月度净值供画图
# ============================================================
comb = pd.DataFrame(all_nav)
comb.index = pd.to_datetime(comb.index)
comb_m = comb.resample("M").last()
comb_m.to_csv(os.path.join(OUT, "v5_oos_monthly.csv"))
print(f"\n月度净值已存: v5_oos_monthly.csv")

# 存 OOS 指标汇总
oos_summary = pd.DataFrame([
    {
        "选参方式": "OOS-TrainOnly",
        "配置": best_oos["配置"],
        "训练_年化": best_oos["训练_年化"],
        "训练_夏普": best_oos["训练_夏普"],
        "OOS_年化": best_oos["验证_年化"],
        "OOS_回撤": best_oos["验证_回撤"],
        "OOS_夏普": best_oos["验证_夏普"],
    },
    {
        "选参方式": "Robust-含2025",
        "配置": best_robust["配置"],
        "训练_年化": best_robust["训练_年化"],
        "训练_夏普": best_robust["训练_夏普"],
        "OOS_年化": best_robust["验证_年化"],
        "OOS_回撤": best_robust["验证_回撤"],
        "OOS_夏普": best_robust["验证_夏普"],
    },
])
oos_summary.to_csv(os.path.join(OUT, "v5_oos_summary.csv"), index=False, encoding='utf-8-sig')
print(f"OOS汇总已存: v5_oos_summary.csv")

print(f"\n总耗时 {time.time()-t0:.0f}s" if 't0' in dir() else "")
