# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diversify_pool import *

rets = load_returns()
corr = corr_matrix(rets, TRAIN_START, TRAIN_END)
must = ["纯债", "黄金", "纳指", "沪深300"]
sel6 = greedy_select(corr, max_n=6, must_include=must, max_pair_corr=0.85)
print("6资产池:", sel6)

tr = pd.DataFrame({c: rets[c] for c in sel6})
tr = tr[(tr.index >= TRAIN_START) & (tr.index <= TRAIN_END)].fillna(0)
te = pd.DataFrame({c: rets[c] for c in sel6})
te = te[(te.index >= TEST_START) & (te.index <= TEST_END)].fillna(0)
w = np.ones(len(sel6)) / len(sel6)

r_te = (te * w).sum(axis=1)
st = portfolio_stats(te, w)
print(f"验证期: 年化 {st['ann']:.2%} 回撤 {st['mdd']:.2%} 波动 {st['vol']:.2%} 夏普 {st['sharpe']:.2f}")
yr = yearly_returns(r_te)
r1y = rolling_1y(r_te)
print("逐年:", " ".join(f"{y.year} {v:>5.1%}" for y, v in yr.items()))
print(f"滚动1y: 中位 {r1y.median():.2%} 负占比 {(r1y<0).mean():.1%} min {r1y.min():.2%} max {r1y.max():.2%}")

st_tr = portfolio_stats(tr, w)
print(f"\n训练期: 年化 {st_tr['ann']:.2%} 回撤 {st_tr['mdd']:.2%} 夏普 {st_tr['sharpe']:.2f}")
r_tr = (tr * w).sum(axis=1)
yr_tr = yearly_returns(r_tr)
print("训练逐年:", " ".join(f"{y.year} {v:>5.1%}" for y, v in yr_tr.items()))

# 相关性矩阵
sc = corr.loc[sel6, sel6]
sa = sc.abs().values
sa = sa[~np.eye(len(sel6), dtype=bool)].mean()
print(f"\n子集平均|corr| = {sa:.3f}")
print("相关性矩阵:")
for a in sel6:
    print("  ", a, " ".join(f"{sc.loc[a,b]:+.2f}" for b in sel6))

# 单资产验证期
print("\n单资产验证期:")
for c in sel6:
    r = te[c]
    nav = (1 + r).cumprod()
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(nav)) - 1
    mdd = float((nav / nav.cummax() - 1).min())
    vol = r.std() * np.sqrt(252)
    print(f"  {c:8s} 年化{ann:>7.2%} 回撤{mdd:>7.2%} 波动{vol:>7.2%}")
