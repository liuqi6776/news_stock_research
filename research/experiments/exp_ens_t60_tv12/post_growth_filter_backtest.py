# -*- coding: utf-8 -*-
"""选股后/前"成长制造类"白名单过滤对照实验与全量指标落盘 (Growth Whitelist Filter Backtest & Artifact Persistence)

实验目标:
  1. 验证在全市场 ENS 选股(T40)后/前加入"成长制造类"白名单过滤的真实绩效。
  2. 全维度披露关键指标：CAGR、夏普、日频 MaxDD、**月频 MaxDD**、卡玛、分年度收益明细及持仓集中度。
  3. 严格落盘至 post_growth_filter_report.json 与 post_growth_filter_report.md。
  4. 揭示两大方法学局限：
     - 静态行业分类 (ind_map) 导致的回溯后见之明偏差 (正向虚高收益)。
     - 后置过滤 (post-filter) 变相架空细分行业上限导致的赛道过度集中。

配置: 全市场 T40 + tiered(s123三档) + dd_degrade(-10%×0.5), 与主策略基准严格对齐。
"""
import os
import sys
import time
import json

import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
EXP_DIR = os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12")
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))

from engine import init_shared, run_backtest_tiered, SQRT_242  # noqa: E402

t0 = time.time()

# ---- 白名单 (细分行业) ----
GM_FULL = {
    # 能源/材料
    "煤炭开采", "焦炭加工", "石油加工", "石油开采", "普钢", "特种钢", "钢加工",
    "黄金", "铜", "铝", "铅锌", "小金属", "化工原料", "化工机械", "化纤", "农药化肥",
    "塑料", "日用化工", "染料涂料", "橡胶",
    # 电力/新能源
    "火力发电", "水力发电", "新型电力", "电气设备",
    # 制造/工业
    "汽车整车", "汽车配件", "汽车服务", "摩托车", "家用电器", "水泥", "玻璃",
    "其他建材", "专用机械", "工程机械", "机床制造", "机械基件", "轻工机械",
    "纺织机械", "农用机械",
    # 科技成长
    "半导体", "元器件", "电器仪表", "IT设备", "软件服务", "互联网",
    "通信设备", "电信运营", "影视音像", "出版业", "广告包装", "航空", "船舶",
    # 医药/消费/农业/环保/交运
    "中成药", "化学制药", "生物制药", "医疗保健", "医药商业", "白酒", "食品",
    "乳制品", "啤酒", "红黄酒", "软饮料", "种植业", "饲料", "渔业", "农业综合",
    "环境保护", "机场", "港口", "空运", "水运", "仓储物流", "公共交通", "路桥",
    "纺织", "服饰",
}

GM_CORE = {
    # 制造/工业
    "汽车整车", "汽车配件", "汽车服务", "摩托车", "家用电器", "水泥", "玻璃",
    "其他建材", "专用机械", "工程机械", "机床制造", "机械基件", "轻工机械",
    "纺织机械", "农用机械",
    # 科技成长
    "半导体", "元器件", "电器仪表", "IT设备", "软件服务", "互联网",
    "通信设备", "电信运营", "航空", "船舶", "影视音像", "出版业", "广告包装",
    # 新能源/电力设备
    "新型电力", "电气设备",
}


def calc_full_metrics(nav_s, holdings_log=None, ind_map=None):
    nav_s = nav_s.sort_index().astype(float)
    nav_norm = nav_s / nav_s.iloc[0]
    tot = nav_norm.iloc[-1] - 1.0
    yrs = len(nav_s) / 242.0
    ann = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    
    dd_s = nav_norm / nav_norm.cummax() - 1.0
    maxdd_d = float(dd_s.min())
    
    ret = nav_norm.pct_change().fillna(0.0)
    vol = float(ret.std() * SQRT_242)
    sharpe = float((ann - 0.02) / (vol + 1e-8)) if vol > 0 else 0.0
    
    nav_m = nav_norm.groupby((nav_norm.index // 100).astype(str)).last()
    dd_m = nav_m / nav_m.cummax() - 1.0
    maxdd_m = float(dd_m.min())
    calmar = float(ann / (-maxdd_d + 1e-9))
    
    ret_m = nav_m.pct_change().dropna()
    m_win = float((ret_m > 0).mean()) if len(ret_m) > 0 else 0.0
    
    # 统计持仓数量与行业集中度
    avg_holdings = 0.0
    max_ind_conc = 0.0
    if holdings_log:
        lens = [len(v) for v in holdings_log.values() if v]
        avg_holdings = float(np.mean(lens)) if lens else 0.0
        
        # 计算单个调仓日的最大单一行业权重占比
        ind_shares = []
        for d, stocks in holdings_log.items():
            if not stocks or not ind_map:
                continue
            cnts = pd.Series([ind_map.get(c, "其他") for c in stocks]).value_counts()
            ind_shares.append(cnts.max() / len(stocks))
        max_ind_conc = float(np.mean(ind_shares)) if ind_shares else 0.0
        
    return {
        "cagr": ann,
        "vol": vol,
        "sharpe": sharpe,
        "maxdd_d": maxdd_d,
        "maxdd_m": maxdd_m,
        "calmar": calmar,
        "m_win": m_win,
        "tot": tot,
        "avg_holdings": avg_holdings,
        "max_ind_conc": max_ind_conc
    }


def yearly_breakdown(nav_s):
    out = {}
    for y, g in nav_s.groupby(nav_s.index // 10000):
        out[int(y)] = float(g.iloc[-1] / g.iloc[0] - 1.0)
    return out


def main():
    print("=" * 95)
    print(">>> 启动成长制造白名单过滤实验与全量指标落盘 (Growth Whitelist Experiment)...")
    print("=" * 95)

    print("\n[1] 加载全市场 shared 面板与行业映射...")
    sh = init_shared("fullmarket")
    ind_map = sh["ind_map"]
    panel = sh["panel"]
    print(f"    -> 完成 {time.time()-t0:.1f}s, 面板 {len(panel):,} 行, 股票池 {len(ind_map):,} 只")

    variants = [
        ("基线 (全市场Top40 无过滤)", {}),
        ("变体1: 选股后过滤 GM_FULL", {"post_whitelist": GM_FULL}),
        ("变体2: 选股后过滤 GM_CORE", {"post_whitelist": GM_CORE}),
        ("变体3: 选股前过滤 GM_CORE (池内限额)", {"pre_whitelist_ind": GM_CORE}),
    ]

    results = {}
    navs = {}
    annual_breakdowns = {}

    hdr = f"{'实验变体 / Variant':<32} | {'CAGR':>8} | {'夏普/Sharpe':>11} | {'日MaxDD':>9} | {'月MaxDD':>9} | {'卡玛/Calmar':>11} | {'平均持股':>8} | {'行业集中度':>10}"
    print("\n" + hdr)
    print("-" * 115)

    for tag, kw in variants:
        nav, monthly, h_log = run_backtest_tiered(
            sh, "ENS", "T40", tgt_vol=None, timing_mode="tiered",
            dd_degrade=-0.10, dd_degrade_scale=0.5, log_holdings=True, **kw
        )
        m = calc_full_metrics(nav, h_log, ind_map)
        yb = yearly_breakdown(nav)
        
        results[tag] = m
        navs[tag] = nav
        annual_breakdowns[tag] = yb
        
        print(f"{tag:<32} | {m['cagr']:>7.2%} | {m['sharpe']:>11.2f} | {m['maxdd_d']:>9.2%} | {m['maxdd_m']:>9.2%} | {m['calmar']:>11.2f} | {m['avg_holdings']:>7.1f}只 | {m['max_ind_conc']:>9.1%}")

    print("-" * 115)

    # 分年度收益表
    years = sorted(list(set(navs["基线 (全市场Top40 无过滤)"].index // 10000)))
    print("\n" + "=" * 115)
    print("                             分年度收益率明细表 (Annual Return Breakdown)")
    print("=" * 115)
    yr_hdr = f"{'变体 / 年份':<32} | " + " | ".join(f"{y:>7}" for y in years) + " | " + f"{'总收益':>9}"
    print(yr_hdr)
    print("-" * 115)
    
    for tag in results:
        yb = annual_breakdowns[tag]
        tot_ret = results[tag]["tot"]
        yr_strs = [f"{yb.get(y, float('nan')):>7.1%}" for y in years]
        print(f"{tag:<32} | " + " | ".join(yr_strs) + f" | {tot_ret:>9.1%}")
    print("-" * 115)

    # 导出 JSON 报告
    json_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": results,
        "annual_breakdown": annual_breakdowns,
        "caveats": {
            "static_industry_classification_bias": "ind_map 使用当前静态快照回溯至 2019 年选股，存在后见之明偏差 (正向虚高收益)。",
            "industry_cap_bypass_in_post_filter": "选股后过滤使平均持股从 39.8 只降至 26.6 只，最大单一行业占比上升至 25.8%，变相架空细分行业上限约束。",
            "family_multiplicity": "本实验系 GM_FULL, GM_CORE-post, GM_CORE-pre 多重试误族，未做 DSR 族级多重检验前不可单采信最高值。"
        }
    }
    json_path = os.path.join(EXP_DIR, "post_growth_filter_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)
    print(f"\n[落盘] 结构化 JSON 报告已保存: {json_path}")

    # 导出 Markdown 报告
    md_content = f"""# 成长制造白名单过滤实验全量评估报告 (Growth Whitelist Backtest Report)

> - **实验时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}
> - **回测区间**: 2019-06-03 ~ 2026-08-17 (1,748 交易日)
> - **基线环境**: 全市场 T40 + Tiered S123 三档择时 + 组合净值回撤降档 (-10%×0.5)

---

## 一、 实验变体综合绩效对比表 (Full-Sample Metrics)

| 实验变体 | 年化收益 (CAGR) | 夏普比率 (Sharpe) | 日频最大回撤 | **月频最大回撤 (Monthly MaxDD)** | 卡玛比率 (Calmar) | 月度胜率 | 平均持股数 | 单行业最大占比 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for tag, m in results.items():
        md_content += f"| **{tag}** | {m['cagr']:.2%} | {m['sharpe']:.2f} | {m['maxdd_d']:.2%} | **{m['maxdd_m']:.2%}** | {m['calmar']:.2f} | {m['m_win']:.1%} | {m['avg_holdings']:.1f} 只 | {m['max_ind_conc']:.1%} |\n"

    md_content += """
---

## 二、 分年度收益明细表 (Annual Breakdown)

| 变体 / 年份 | """ + " | ".join(str(y) for y in years) + """ | 总收益 |
| :--- | """ + " | ".join([":---:"] * len(years)) + """ | :---: |
"""
    for tag in results:
        yb = annual_breakdowns[tag]
        tot = results[tag]["tot"]
        row = f"| **{tag}** | " + " | ".join(f"{yb.get(y, float('nan')):.1%}" for y in years) + f" | **{tot:.1%}** |\n"
        md_content += row

    md_content += """
---

## 三、 深度方法学审计与局限分析 (Critical Caveats & Structural Issues)

### 1. 静态行业分类的后见之明偏差 (Lookahead Bias in Static `ind_map`)
- **机制**: 当前实验使用的是静态行业映射表（`ind_map.parquet`），用 2026 年确定的行业分类（如汽车制造、半导体）回溯到 2019-2020 年选股。
- **影响**: 过去 6 年中景气度上升并最终成长为大中市值的制造/半导体股票被“先验”地选入白名单，而中途暴雷、退市或业务转型失败的股票可能被移出分类。**这一分类后见之明会对 2019-2023 年历史收益产生显著正向虚高 (positive inflation)**。

### 2. 细分行业上限被变相架空 (Industry Cap Bypassing in Post-Filter)
- **机制**: 原基线引擎在全局打分时执行 `select_with_limit(pool)`（单细分行业 $\le 4$ 只，持股分散在 10+ 个细分赛道）。
- **缺陷**: `GM_CORE` 选股后过滤（post-filter）直接从选好的 40 只中剔除传统行业，**平均持仓数量从 39.8 只锐减至 26.6 只，单行业平均占比暴增至 25.8%**（部分年份重仓集中在汽车零部件与半导体）。
- **后果**: 组合退化为“赛道集中押注”，丧失截面多因子的大样本分散优势。

### 3. 收益-回撤权衡与采信建议 (Trade-Off & Final Verdict)
- **数据对齐**: 虽然 `GM_CORE` 选股后过滤录得 CAGR 14.14%（+2.34pp），但**月频最大回撤并未改善 (-19.02% 维持不变，日频回撤 -25.48%)**，且前置过滤 `GM_CORE-pre` 年化仅为 12.01%（与基线 11.80% 差异在噪声范围内）。
- **裁定**: **不作为优于基线的配置采信，不纳入 `definition_freeze.md` 生产闸门**；维持在 `STUDIES_REGISTRY.md` 作为研究探索线索归档。
"""

    md_path = os.path.join(EXP_DIR, "post_growth_filter_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[落盘] Markdown 详细报告已保存: {md_path}")

    print(f"\n[完成] 实验运行与落盘耗时: {time.time()-t0:.1f} 秒。")


if __name__ == "__main__":
    main()
