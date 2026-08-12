# -*- coding: utf-8 -*-
"""
v5 网格搜索：2018-2024训练 → 2025验证
（实际数据从2020开始，故用2020-2024训练 + 2025验证）
复用 backtest_stock_picking_v5.py 的数据加载和回测引擎
"""
import os, sys, time, itertools
import numpy as np
import pandas as pd

# ============================================================
# 1. 加载 v5 数据和函数（跳过配置执行部分）
# ============================================================
v5_path = os.path.join(os.path.dirname(__file__), 'backtest_stock_picking_v5.py')
v5_code = open(v5_path, encoding='utf-8').read()
# 在 "# 8. v5 组合测试" 处截断，只执行数据加载 + 函数定义
marker = '# ============================================================\n# 8. v5'
code_before = v5_code.split(marker)[0]
exec(code_before)

# ============================================================
# 2. 网格参数空间
# ============================================================
GRID = {
    'global_top_k':           [3, 5, 8],        # 持仓数
    'min_roe_pct':            [8, 10, 12, 15],  # ROE硬门槛
    'max_peg':                [1.5, 2.0, 2.5],  # PEG硬上限
    'chip_conc_pctl_threshold': [0.40, 0.50, 0.60],  # 筹码集中度分位
    'min_circ_mv_yi':         [50, 100],         # 市值下限(亿)
    'min_list_years':         [2, 3],            # 上市年限
}
# 固定参数: max_same_sector=2, max_pe=60, min_turnover=0.5,
#           max_circ_mv=2000, peg_preferred=1.5, preferred_weight=1.2

SPLIT = pd.Timestamp('2025-01-01')
TRAIN_START = pd.Timestamp('2020-01-01')

# ============================================================
# 3. 分段指标计算
# ============================================================
def calc_period(nav_s, trades, start, end):
    """计算 [start, end) 区间的指标。验证期rebase到INIT。"""
    mask = (nav_s.index >= start) & (nav_s.index < end)
    sub = nav_s[mask].copy()
    if len(sub) < 5:
        return {"年化": np.nan, "累计": np.nan, "回撤": np.nan,
                "夏普": np.nan, "买入": 0, "止盈": 0, "时间止损": 0}
    # 验证期rebase
    if start > nav_s.index[0]:
        before = nav_s[nav_s.index < start]
        if len(before):
            sub = sub / before.iloc[-1] * INIT
        else:
            sub = sub / sub.iloc[0] * INIT
    else:
        sub = sub / sub.iloc[0] * INIT
    tr = sub.iloc[-1] / INIT - 1
    yrs = (sub.index[-1] - sub.index[0]).days / 365.25
    ann = (1 + tr) ** (1/yrs) - 1 if yrs > 0 else np.nan
    pk = sub.cummax()
    mdd = ((sub - pk) / pk).min()
    ret = sub.pct_change().dropna()
    shp = ret.mean() / ret.std(ddof=1) * np.sqrt(252) if len(ret) > 1 and ret.std(ddof=1) > 0 else np.nan
    t_trades = [t for t in trades if start <= t[0] < end]
    buys = sum(1 for t in t_trades if t[1] == "BUY")
    tps = sum(1 for t in t_trades if t[1] == "TP")
    tls = sum(1 for t in t_trades if t[1] not in ("BUY", "TP"))
    return {"年化": ann, "累计": tr, "回撤": mdd, "夏普": shp,
            "买入": buys, "止盈": tps, "时间止损": tls}

# ============================================================
# 4. 网格搜索
# ============================================================
keys = list(GRID.keys())
vals = list(GRID.values())
total = 1
for v in vals: total *= len(v)
print(f"\n{'='*80}")
print(f"v5 网格搜索: {total} 组配置")
print(f"训练期: 2020-01 ~ 2024-12 | 验证期: 2025-01 ~ 2025-12")
print(f"{'='*80}\n")

results = []
t_start = time.time()

for i, combo in enumerate(itertools.product(*vals)):
    p = dict(zip(keys, combo))
    label = (f"K{p['global_top_k']}_ROE{p['min_roe_pct']}"
             f"_PEG{p['max_peg']}_CHIP{int(p['chip_conc_pctl_threshold']*100)}"
             f"_MV{p['min_circ_mv_yi']}_YR{p['min_list_years']}")

    try:
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
    except Exception as e:
        continue

    if len(nv) < 30:
        continue

    # 全区间
    full_m = calc_period(nv, trs, TRAIN_START, pd.Timestamp('2026-01-01'))
    # 训练期
    train_m = calc_period(nv, trs, TRAIN_START, SPLIT)
    # 验证期
    val_m = calc_period(nv, trs, SPLIT, pd.Timestamp('2026-01-01'))

    row = {"配置": label}
    row.update({f"训练_{k}": v for k, v in train_m.items()})
    row.update({f"验证_{k}": v for k, v in val_m.items()})
    row.update({f"全期_{k}": v for k, v in full_m.items()})
    # 稳健性评分：训练夏普 × 验证夏普（都为正才有意义）
    ts = train_m["夏普"]; vs = val_m["夏普"]
    if not np.isnan(ts) and not np.isnan(vs) and ts > 0 and vs > 0:
        row["稳健分"] = ts * vs
    else:
        row["稳健分"] = 0.0
    results.append(row)

    if (i+1) % 50 == 0:
        elapsed = time.time() - t_start
        print(f"  进度: {i+1}/{total} ({(i+1)/total*100:.0f}%) 已耗时{elapsed:.0f}s 预计{elapsed/(i+1)*total:.0f}s")

elapsed = time.time() - t_start
print(f"\n网格搜索完成: {len(results)}/{total} 组有效, 耗时{elapsed:.0f}s")

# ============================================================
# 5. 输出结果
# ============================================================
df = pd.DataFrame(results)

# 按训练夏普排名
df = df.sort_values("训练_夏普", ascending=False).reset_index(drop=True)

# 输出完整CSV
csv_path = os.path.join(OUT_DIR, "v5_grid_search.csv")
df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"完整结果: {csv_path}")

# ===== 训练期Top20 =====
print(f"\n{'='*120}")
print("【训练期 Top20】(按训练夏普排名)")
print(f"{'='*120}")
cols_show = ["配置", "训练_年化", "训练_回撤", "训练_夏普", "训练_买入", "训练_止盈",
             "验证_年化", "验证_回撤", "验证_夏普", "验证_买入", "验证_止盈", "稳健分"]
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(df[cols_show].head(20).to_string(index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)))

# ===== 稳健分Top20（训练+验证都好） =====
df_robust = df[df["稳健分"] > 0].sort_values("稳健分", ascending=False).head(20)
print(f"\n{'='*120}")
print("【稳健分 Top20】(训练夏普×验证夏普, 仅含两者都为正的配置)")
print(f"{'='*120}")
if len(df_robust) > 0:
    print(df_robust[cols_show].to_string(index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)))
else:
    print("  无配置在训练和验证期夏普同时为正")

# ===== 验证期Top10（看哪些配置在样本外最好） =====
df_val = df.dropna(subset=["验证_夏普"]).sort_values("验证_夏普", ascending=False).head(10)
print(f"\n{'='*120}")
print("【验证期 Top10】(按验证夏普排名 — 样本外表现)")
print(f"{'='*120}")
print(df_val[cols_show].to_string(index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)))

# ===== 训练Top5 vs 验证对照 =====
print(f"\n{'='*120}")
print("【训练Top5 在验证期的表现对照】")
print(f"{'='*120}")
top5 = df.head(5)
for _, r in top5.iterrows():
    ta = r["训练_年化"]; va = r["验证_年化"]
    ts = r["训练_夏普"]; vs = r["验证_夏普"]
    td = r["训练_回撤"]; vd = r["验证_回撤"]
    print(f"  {r['配置']}")
    print(f"    训练: 年化{ta:.1%} 回撤{td:.1%} 夏普{ts:.2f} | "
          f"验证: 年化{va:.1%} 回撤{vd:.1%} 夏普{vs:.2f}")
    if not np.isnan(va) and not np.isnan(ta):
        print(f"    年化变化: {va-ta:+.1%}  夏普变化: {vs-ts:+.2f}")
    print()

print(f"\n总耗时 {elapsed:.0f}s")
