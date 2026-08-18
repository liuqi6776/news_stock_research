# -*- coding: utf-8 -*-
"""
多资产低相关分散化策略: 训练(2018-2020) / 验证(2021-2026) 严格分离
====================================================================
Step 1: 加载 ~30 只跨资产/跨板块场外基金净值
Step 2: 用 2018-2020 日收益算相关性矩阵
Step 3: 贪心选低相关子集 (~12-15 只)
Step 4: 多策略回测: 等权 / 最小方差 / 风险平价 / 最大夏普
Step 5: 训练期优化权重 → 验证期直接代入, 打印年化/回撤/逐年/滚动1y

用法: C:/Users/liuqi/anaconda3/python.exe studies/rotation_dingtou/diversify_pool.py
"""
import os, sys, time, itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

TRAIN_START = "2018-01-01"
TRAIN_END   = "2020-12-31"
TEST_START  = "2021-01-01"
TEST_END    = "2026-08-06"
NAV_DIR = r"D:\iquant_data\data_v2\fund2\nav"

# ---- 30 只候选基金: 跨资产/跨板块 ----
CANDIDATES = {
    # 债类
    "纯债":       ("000015", "华夏纯债A"),
    "国债":       ("001512", "易方达3-5年国债"),
    "可转债":     ("100051", "富国可转债"),
    "QDII债":     ("004998", "长信全球债券"),
    # A股宽基
    "沪深300":    ("050002", "博时沪深300"),
    "中证500":    ("160119", "南方中证500"),
    "中证1000":   ("003646", "创金合信中证1000"),
    "创业板":     ("001879", "长城创业板增强"),
    # A股板块
    "红利":       ("100032", "富国中证红利"),
    "食品饮料":   ("160222", "国泰食品饮料"),
    "医药":       ("165519", "中信保诚医药"),
    "军工":       ("161024", "富国军工"),
    "新能源":     ("164905", "交银新能源"),
    "煤炭":       ("161724", "招商煤炭"),
    "有色":       ("165520", "中信保诚有色"),
    "钢铁":       ("168203", "国联钢铁"),
    "金融地产":   ("161211", "国投金融地产"),
    "电子信息":   ("080012", "长盛电子信息"),
    "民生科技":   ("002683", "民生加银前沿科技"),
    # 港股
    "恒生":       ("000071", "华夏恒生ETF联接"),
    "港股红利":   ("004098", "前海开源港股通红利"),
    # 海外
    "纳指":       ("000834", "大成纳指100"),
    "标普500":    ("050025", "博时标普500"),
    "油气":       ("162411", "华宝标普油气"),
    # 商品
    "黄金":       ("000216", "华安黄金"),
    "白银":       ("161226", "国投白银"),
    "原油":       ("501018", "南方原油"),
    "大宗商品":   ("161715", "招商大宗商品"),
}

_AC = {}
def acc_nav(code):
    if code not in _AC:
        p = os.path.join(NAV_DIR, f"{code}.parquet")
        if not os.path.exists(p):
            _AC[code] = None; return None
        df = pd.read_parquet(p, columns=["date", "acc_nav"])
        s = pd.Series(df["acc_nav"].to_numpy(dtype=float),
                      index=pd.to_datetime(df["date"]))
        s = s[~s.index.duplicated(keep="last")].sort_index().dropna()
        s = s[(s.index >= pd.Timestamp(TRAIN_START) - pd.Timedelta(days=400)) &
              (s.index <= pd.Timestamp(TEST_END))]
        _AC[code] = s
    return _AC[code]


def load_returns():
    """加载所有候选基金的日收益, 返回 dict[cat] = Series(daily_ret)"""
    rets = {}
    for cat, (code, name) in CANDIDATES.items():
        s = acc_nav(code)
        if s is None or len(s) < 300:
            print(f"  跳过 {cat}: {code} 数据不足")
            continue
        r = s.pct_change().dropna()
        r = r[(r >= -0.2) & (r <= 0.2)]
        rets[cat] = r
    return rets


def corr_matrix(rets, start, end):
    """计算指定时段的相关性矩阵"""
    df = pd.DataFrame({k: v[(v.index >= pd.Timestamp(start)) &
                            (v.index <= pd.Timestamp(end))] for k, v in rets.items()})
    return df.corr()


def greedy_select(corr, max_n=15, must_include=None, max_pair_corr=0.85):
    """贪心选低相关子集: 从must_include开始, 每次选与已选集平均相关性最低的
    新增去重: 如果与已选任一资产 |corr| > max_pair_corr 则跳过"""
    cats = list(corr.columns)
    if must_include is None:
        must_include = []
    if not must_include:
        avg_corr = corr.abs().mean()
        must_include = [avg_corr.idxmin()]
    selected = list(must_include)
    remaining = [c for c in cats if c not in selected]
    while len(selected) < max_n and remaining:
        best_cat, best_score = None, 999
        for c in remaining:
            # 如果与已选任一资产相关性太高, 跳过
            if any(abs(corr.loc[c, s]) > max_pair_corr for s in selected):
                continue
            score = corr.loc[selected, c].abs().mean()
            if score < best_score:
                best_score = score
                best_cat = c
        if best_cat is None:
            break  # 没有更多低相关资产可加
        selected.append(best_cat)
        remaining.remove(best_cat)
    return selected


def portfolio_stats(returns_df, weights):
    """计算组合统计"""
    r = (returns_df * weights).sum(axis=1).dropna()
    if len(r) < 60:
        return {"ann": 0, "mdd": 0, "vol": 0, "sharpe": 0}
    nav = (1.0 + r).cumprod()
    ann = (nav.iloc[-1] / nav.iloc[0]) ** (252.0 / len(nav)) - 1
    vol = r.std() * np.sqrt(252)
    mdd = float((nav / nav.cummax() - 1).min())
    sharpe = (r.mean() * 252.0) / vol if vol > 0 else 0
    return {"ann": ann, "mdd": mdd, "vol": vol, "sharpe": sharpe}


def min_variance_weights(cov):
    """最小方差权重 (long-only, sum=1)"""
    n = len(cov)
    def obj(w):
        return w @ cov @ w
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 0.4)] * n  # 单资产上限 40%
    w0 = np.ones(n) / n
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x


def max_sharpe_weights(cov, mu, rf=0.0):
    """最大夏普权重 (long-only, sum=1)"""
    n = len(cov)
    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-8:
            return 0
        return -(port_ret - rf) / port_vol
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 0.4)] * n
    w0 = np.ones(n) / n
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x


def risk_parity_weights(cov):
    """风险平价: 每个资产贡献相等风险 (近似: inverse vol)"""
    vol = np.sqrt(np.diag(cov))
    w = 1.0 / vol
    w /= w.sum()
    w = np.minimum(w, 0.4)
    w /= w.sum()
    return w


def inverse_vol_weights(cov):
    """逆波动率加权 (与风险平价类似但更简单)"""
    vol = np.sqrt(np.diag(cov))
    w = 1.0 / vol
    w /= w.sum()
    return w


def yearly_returns(r):
    nav = (1.0 + r).cumprod()
    yr = nav.resample("Y").last().pct_change().dropna()
    return yr


def rolling_1y(r):
    nav = (1.0 + r).cumprod()
    return nav.pct_change(252).dropna()


def main():
    t0 = time.time()
    print("=" * 100)
    print("多资产低相关分散化策略 — 训练(2018-2020) / 验证(2021-2026) 严格分离")
    print("=" * 100)

    # Step 1: 加载
    print("\n1) 加载候选基金净值 ...")
    rets = load_returns()
    print(f"   成功加载 {len(rets)} 只基金")
    cats = list(rets.keys())

    # Step 2: 相关性矩阵 (训练期)
    print("\n2) 训练期 (2018-2020) 相关性矩阵:")
    corr = corr_matrix(rets, TRAIN_START, TRAIN_END)
    avg_abs_corr = corr.abs().values
    avg_abs_corr = avg_abs_corr[~np.eye(len(cats), dtype=bool)].mean()
    print(f"   全池平均 |corr| = {avg_abs_corr:.3f}")

    print("\n   高相关对 (|corr| > 0.7), 需去重:")
    pairs = []
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            c = corr.iloc[i, j]
            if abs(c) > 0.7:
                pairs.append((cats[i], cats[j], c))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    for a, b, c in pairs:
        print(f"     {a:10s} — {b:10s}: {c:+.3f}")

    # Step 3: 贪心选低相关子集 (去重: pair_corr > 0.85 跳过)
    print("\n3) 贪心选择低相关子集 (去重阈值 0.85):")
    must = ["纯债", "黄金", "纳指", "沪深300"]
    must = [m for m in must if m in cats]

    all_results = []  # (pool_size, strategy, train_stats, test_stats, weights, selected, yr, r1y)

    for pool_size in [6, 8, 10, 12, 15]:
        selected = greedy_select(corr, max_n=pool_size, must_include=must, max_pair_corr=0.85)
        if len(selected) < 4:
            continue
        sub_corr = corr.loc[selected, selected]
        sub_avg = sub_corr.abs().values
        sub_avg = sub_avg[~np.eye(len(selected), dtype=bool)].mean()

        train_rets = pd.DataFrame({c: rets[c] for c in selected})
        train_rets = train_rets[(train_rets.index >= TRAIN_START) & (train_rets.index <= TRAIN_END)].fillna(0.0)
        test_rets = pd.DataFrame({c: rets[c] for c in selected})
        test_rets = test_rets[(test_rets.index >= TEST_START) & (test_rets.index <= TEST_END)].fillna(0.0)

        n = len(selected)
        train_cov = train_rets.cov().values * 252
        train_mu = train_rets.mean().values * 252

        strategies = {
            "等权":         np.ones(n) / n,
            "逆波动":       inverse_vol_weights(train_cov),
            "风险平价":     risk_parity_weights(train_cov),
            "最小方差":     min_variance_weights(train_cov),
            "最大夏普":     max_sharpe_weights(train_cov, train_mu),
        }

        for sname, w in strategies.items():
            st_tr = portfolio_stats(train_rets, w)
            st_te = portfolio_stats(test_rets, w)
            r_te = (test_rets * w).sum(axis=1)
            yr = yearly_returns(r_te)
            r1y = rolling_1y(r_te)
            all_results.append((pool_size, sname, st_tr, st_te, w, selected,
                                sub_avg, yr, r1y))

    # 打印汇总表
    print("\n" + "=" * 120)
    print("4) 全部组合汇总 (训练期优化 → 验证期代入)")
    print("=" * 120)
    print(f"{'池大小':>5s} {'策略':8s} | {'训练年化':>8s} {'训练回撤':>8s} {'训练夏普':>6s} | "
          f"{'验证年化':>8s} {'验证回撤':>8s} {'验证夏普':>6s} {'子集|corr|':>8s} | 达标")
    print("-" * 120)
    # 按 验证夏普 排序
    all_results.sort(key=lambda x: x[3]["sharpe"], reverse=True)
    for ps, sn, st_tr, st_te, w, sel, sa, yr, r1y in all_results:
        hit = "Y" if st_te["ann"] >= 0.10 and st_te["mdd"] >= -0.08 else " "
        print(f"{ps:>5d} {sn:8s} | {st_tr['ann']:>7.2%} {st_tr['mdd']:>7.2%} {st_tr['sharpe']:>5.2f} | "
              f"{st_te['ann']:>7.2%} {st_te['mdd']:>7.2%} {st_te['sharpe']:>5.2f} {sa:>7.3f} |  {hit}")

    # Top 3 详细
    print("\n" + "=" * 120)
    print("5) 验证期夏普 Top 3 详细:")
    print("=" * 120)
    for rank, (ps, sn, st_tr, st_te, w, sel, sa, yr, r1y) in enumerate(all_results[:3], 1):
        print(f"\n  #{rank} 池大小={ps} 策略={sn}  子集|corr|={sa:.3f}")
        print(f"     资产: {', '.join(sel)}")
        print(f"     训练期: 年化 {st_tr['ann']:.2%} 回撤 {st_tr['mdd']:.2%} 夏普 {st_tr['sharpe']:.2f}")
        print(f"     验证期: 年化 {st_te['ann']:.2%} 回撤 {st_te['mdd']:.2%} 夏普 {st_te['sharpe']:.2f}")
        print(f"     逐年: " + " ".join(f"{y.year} {v:>5.1%}" for y, v in yr.items()))
        print(f"     滚动1y: 中位 {r1y.median():.2%} 负占比 {(r1y<0).mean():.1%} min {r1y.min():.2%}")
        w_str = " ".join(f"{c}={wi:.0%}" for c, wi in zip(sel, w) if wi > 0.01)
        print(f"     权重: {w_str}")

    # 最佳等权 (用户最关心的简单方案)
    eq_results = [r for r in all_results if r[1] == "等权"]
    eq_results.sort(key=lambda x: x[3]["sharpe"], reverse=True)
    if eq_results:
        ps, sn, st_tr, st_te, w, sel, sa, yr, r1y = eq_results[0]
        print(f"\n  最佳等权方案: 池大小={ps}")
        print(f"     资产: {', '.join(sel)}")
        print(f"     验证期: 年化 {st_te['ann']:.2%} 回撤 {st_te['mdd']:.2%} 夏普 {st_te['sharpe']:.2f}")
        print(f"     逐年: " + " ".join(f"{y.year} {v:>5.1%}" for y, v in yr.items()))
        print(f"     滚动1y: 中位 {r1y.median():.2%} 负占比 {(r1y<0).mean():.1%}")

    # 单资产验证期表现
    print("\n6) 单资产验证期表现 (2021-2026):")
    print(f"   {'资产':12s} {'年化':>8s} {'回撤':>8s} {'夏普':>6s}")
    for c in cats:
        r = rets[c]
        r_te = r[(r.index >= pd.Timestamp(TEST_START)) & (r.index <= pd.Timestamp(TEST_END))]
        if len(r_te) < 60:
            continue
        nav = (1.0 + r_te).cumprod()
        ann = (nav.iloc[-1] / nav.iloc[0]) ** (252.0 / len(r_te)) - 1
        mdd = float((nav / nav.cummax() - 1).min())
        vol = r_te.std() * np.sqrt(252)
        sh = (r_te.mean() * 252) / vol if vol > 0 else 0
        print(f"   {c:12s} {ann:>7.2%} {mdd:>7.2%} {sh:>5.2f}")

    print(f"\n总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
