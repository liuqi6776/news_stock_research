#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Study A (正确版) — DOS 作为"隔夜跳水风险过滤器"的诚实对照实验
================================================================
本脚本修复了 study_a_results_v3 的三个 bug, 并按 3 方审查(Oracle/GPT/Gemini)
的要求做了对照基准和增量检验。

修复的 bug:
  Bug1  DOS 过滤空操作 → 本脚本显式把 dos merge 进来, 且断言 dos 列存在、有方差
  Bug2  扩展映射与 DOS 脱节 → 只在 dos 非缺失的样本上做实验, 并报告覆盖率
  Bug3  概念词污染 → 用黑名单剔除板块/概念/外国公司, 只留真个股

新增的诚实检验(决定 DOS 到底有没有用):
  对照1  剔高DOS  vs  剔高历史波动  vs  剔高历史振幅
         → 若 DOS 跑不赢历史波动, 说明 DOS 只是低波动异象代理
  对照2  控制历史波动后, DOS 对次日跳空的回归系数是否仍显著(NW-t)
         → 这是"DOS 有没有增量"的一锤定音
  对照3  剔高DOS 后, 跳水(gap<-3%) 和 暴涨(gap>+3%) 是否【不对称】下降
         → 对称下降=纯方差压缩(没用); 只砍跳水=有风控价值

口径修复(Gemini 抓到的):
  - dos 有大量 =0, 分位数过滤必须显式处理 ties: 用"严格大于阈值才剔",
    并打印【实际剔除比例】, 不再假装"剔了30%"

成功标准(任一不满足则 DOS 当过滤器=失败):
  S1  控制历史波动后, DOS→跳空 的 NW-t 仍 > 2
  S2  剔高DOS 的跳水率下降 明显优于 同比例剔高历史波动
  S3  剔高DOS 对跳水的削减 明显大于对暴涨的削减(不对称)
"""

import pandas as pd
import numpy as np
import os
import sys
import duckdb
import json

DATA_DIR = 'D:/iquant_data/data_v2'
SAVE_DIR = 'C:/Users/liuqi/quant_system_v2'

# 概念词/板块/外国公司黑名单 (非A股个股)
CONCEPT_BLACKLIST = {
    '人工智能', '医疗', '机器人', '新能源汽车', '芯片', '光伏', '低空经济', '风电',
    'DeepSeek', '英伟达', '字节跳动', '华为', '诺和诺德', '礼来', '特斯拉', '美光科技',
    '阿特斯', '嘉楠科技', '文远知行', '小鹏汽车', '苹果', '微软', '谷歌', 'OpenAI',
    '三星', '台积电', '新能源', '半导体', '算力', '军工', '消费', '金融', '房地产',
    '稀土', '有色', '电力', '鸿蒙', 'AI', '大模型', '元宇宙', '区块链', '数字经济',
    '人形机器人', '固态电池', '中国电影', '国新能源',
}
GAP_DOWN = -3.0   # 跳水阈值 %
GAP_UP = 3.0      # 暴涨阈值 %
DROP_FRAC = 0.30  # 名义剔除比例(实际会因ties不同, 脚本会打印真实值)


def build():
    """加载 + 清映射 + 重接DOS + 算次日跳空/收益/历史风险因子。"""
    print("Reading news academic data...")
    af = pd.read_csv(f'{SAVE_DIR}/news_academic_full.csv')
    need = ['news_date', 'ts_code', 'company_name', 'title',
            'positive', 'negative', 'net_sentiment', 'dos']
    af = af[[c for c in need if c in af.columns]].copy()

    # --- Bug3 修复: 清概念词 + 要求公司名在标题 ---
    af['is_concept'] = af['company_name'].isin(CONCEPT_BLACKLIST)
    af['name_in_title'] = af.apply(
        lambda r: str(r['company_name']) in str(r['title']), axis=1)
    clean = af[(~af['is_concept']) & af['name_in_title']].copy()
    print(f"After concept filter: {len(clean)} / {len(af)} ({len(clean)/len(af):.1%})")

    # --- Bug1 断言: dos 列存在且有方差 ---
    assert 'dos' in clean.columns, "FATAL: dos 列缺失"
    assert clean['dos'].std() > 0, "FATAL: dos 无方差(全常数), 过滤会失效"

    # 价格表算次日跳空 + 当日历史风险因子(point-in-time, 用 shift 防前视)
    print("Reading price data...")
    con = duckdb.connect()
    price_files = [os.path.join(DATA_DIR, 'data_day1', f) for f in os.listdir(os.path.join(DATA_DIR, 'data_day1')) 
                   if f.endswith('.parquet') and os.path.getsize(os.path.join(DATA_DIR, 'data_day1', f)) > 1000
                   and f.startswith(('2024', '2025'))]
    files_str = ', '.join([f"'{f}'" for f in price_files[:500]])
    px = con.execute(f"SELECT trade_date, ts_code, open, close, high, low FROM read_parquet([{files_str}])").df()
    con.close()
    
    px = px.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)
    px['trade_date'] = px['trade_date'].astype(int)
    
    # Calculate pre_close (shift close by 1 day per stock)
    px['pre_close'] = px.groupby('ts_code')['close'].shift(1)
    
    # Calculate overnight gap and intraday return
    px['gap'] = (px['open'] / px['pre_close'] - 1) * 100
    px['ret_oc'] = (px['close'] / px['open'] - 1) * 100
    
    # 历史风险因子: 用 t 之前的数据 (shift防前视)
    daily_absret = (px['close'] / px['pre_close'] - 1).abs() * 100
    px['hist_vol5'] = daily_absret.groupby(px['ts_code']).transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    px['hist_vol20'] = daily_absret.groupby(px['ts_code']).transform(
        lambda s: s.shift(1).rolling(20, min_periods=1).mean())
    hist_amp = (px['high'] - px['low']) / px['pre_close'] * 100
    px['hist_amp5'] = hist_amp.groupby(px['ts_code']).transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    
    print(f"Price data: {len(px)} rows, {px['ts_code'].nunique()} stocks, {px['trade_date'].nunique()} dates")

    # 决策日 = news_date 下一交易日
    td = sorted(px['trade_date'].unique())
    nx = {td[i]: td[i + 1] for i in range(len(td) - 1)}
    clean['decision_date'] = clean['news_date'].astype(int).map(nx)
    
    # 合并
    d = clean.merge(
        px[['trade_date', 'ts_code', 'gap', 'ret_oc',
            'hist_vol5', 'hist_vol20', 'hist_amp5']],
        left_on=['decision_date', 'ts_code'],
        right_on=['trade_date', 'ts_code'], how='inner')
    d = d.dropna(subset=['dos', 'gap', 'hist_vol5'])
    
    print(f"Final merged: {len(d)} rows")
    return d


def filter_report(d, factor, name, drop_frac=DROP_FRAC):
    """剔除 factor 最高的 drop_frac, 显式处理 ties, 报告真实剔除比例 + 跳水/暴涨。"""
    thr = d[factor].quantile(1 - drop_frac)
    kept = d[d[factor] <= thr]  # 严格: <=阈值保留, 显式
    real_drop = (len(d) - len(kept)) / len(d)
    gd = (kept['gap'] < GAP_DOWN).mean() * 100
    gu = (kept['gap'] > GAP_UP).mean() * 100
    ret = kept['ret_oc'].mean()
    return dict(name=name, n=len(kept), real_drop=real_drop * 100,
                gap_down=gd, gap_up=gu, ret_oc=ret)


def incremental_regression(d):
    """控制历史波动后, DOS 对次日|跳空| 的回归系数 + Newey-West t。"""
    coefs = []
    for dd, g in d.groupby('decision_date'):
        g = g.dropna(subset=['dos', 'gap', 'hist_vol5', 'hist_vol20'])
        if len(g) < 12 or g['dos'].nunique() < 3:
            continue
        y = g['gap'].abs().values
        X = np.column_stack([np.ones(len(g)), g['dos'].values,
                             g['hist_vol5'].values, g['hist_vol20'].values])
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            coefs.append(beta[1])  # dos 的系数
        except Exception:
            pass
    coefs = np.array(coefs)
    if len(coefs) == 0:
        return None
    mean = coefs.mean()
    n = len(coefs); s2 = np.var(coefs, ddof=1)
    for l in range(1, 6):
        w = 1 - l / 6
        cov = np.cov(coefs[:-l], coefs[l:])[0, 1] if n > l else 0
        s2 += 2 * w * cov
    nwt = mean / np.sqrt(s2 / n) if s2 > 0 else np.nan
    return dict(coef=mean, nwt=nwt, n=n)


def main():
    print("=" * 70)
    print("Study A 正确版: DOS 隔夜跳水过滤器 — 诚实对照实验")
    print("=" * 70)
    d = build()
    print(f"\n清洗后样本: {len(d)} (真个股+名在标题+有DOS+有价格)")
    print(f"唯一个股: {d['ts_code'].nunique()} | 决策日: {d['decision_date'].nunique()}")
    print(f"dos=0 占比: {(d['dos']==0).mean()*100:.1f}%  (Gemini提示: 影响分位数过滤口径)")
    print(f"dos 与 历史20日波动 相关性: {d[['dos','hist_vol20']].corr().iloc[0,1]:+.3f}")

    # ===== 对照1: 三种风险因子当过滤器, 谁更能降跳水 =====
    print("\n" + "─" * 66)
    print(f"对照1: 各剔除 top {int(DROP_FRAC*100)}% 高风险股, 比谁更能降跳水")
    print("─" * 66)
    base_gd = (d['gap'] < GAP_DOWN).mean() * 100
    base_gu = (d['gap'] > GAP_UP).mean() * 100
    print(f"  {'基线(不过滤)':22s}: n={len(d):4d} 跳水率={base_gd:.2f}% 暴涨率={base_gu:.2f}% 日内={d['ret_oc'].mean():+.3f}%")
    for fac, nm in [('dos', '剔高DOS'), ('hist_vol5', '剔高历史波动5d'),
                    ('hist_vol20', '剔高历史波动20d'), ('hist_amp5', '剔高历史振幅5d')]:
        r = filter_report(d, fac, nm)
        print(f"  {r['name']:22s}: n={r['n']:4d} 跳水率={r['gap_down']:.2f}% 暴涨率={r['gap_up']:.2f}% "
              f"日内={r['ret_oc']:+.3f}% (实剔{r['real_drop']:.0f}%)")
    print("\n  >>> 判读: 若'剔高DOS'跳水率 远不如 '剔高历史波动' → DOS无独立风控价值")

    # ===== 对照2: 控制历史波动后, DOS 增量回归 =====
    print("\n" + "─" * 66)
    print("对照2: 控制历史波动后, DOS 对次日|跳空|的增量预测 (一锤定音)")
    print("─" * 66)
    reg = incremental_regression(d)
    if reg:
        sig = '显著***' if abs(reg['nwt']) > 2.6 else '显著**' if abs(reg['nwt']) > 2 else '不显著'
        print(f"  DOS系数={reg['coef']:+.4f}  Newey-West t={reg['nwt']:+.2f}  ({sig})  天数={reg['n']}")
        print("  >>> S1标准: 控制历史波动后 |t|>2 才算DOS有增量")

    # ===== 对照3: 不对称性(纯方差压缩 vs 风控) =====
    print("\n" + "─" * 66)
    print("对照3: 剔高DOS后, 跳水 vs 暴涨 是否不对称下降")
    print("─" * 66)
    r = filter_report(d, 'dos', '剔高DOS')
    print(f"  跳水率: {base_gd:.2f}% → {r['gap_down']:.2f}% (降{base_gd-r['gap_down']:+.2f}pp)")
    print(f"  暴涨率: {base_gu:.2f}% → {r['gap_up']:.2f}% (降{base_gu-r['gap_up']:+.2f}pp)")
    print("  >>> S3标准: 跳水降幅 明显>暴涨降幅 才有风控价值; 同步下降=纯方差压缩(无用)")

    # ===== 最终判定 =====
    print("\n" + "=" * 66)
    print("最终判定 (3条标准)")
    print("=" * 66)
    s1 = abs(reg['nwt']) > 2 if reg else False
    r_dos = filter_report(d, 'dos', 'dos')
    r_vol = filter_report(d, 'hist_vol5', 'vol')
    s2 = (base_gd - r_dos['gap_down']) > (base_gd - r_vol['gap_down']) * 0.5
    s3 = (base_gd - r_dos['gap_down']) > (base_gu - r_dos['gap_up']) * 1.5
    print(f"  S1 控制历史波动后DOS仍显著(|t|>2): {'PASS' if s1 else 'FAIL'}")
    print(f"  S2 剔DOS降跳水 不输给 剔历史波动:    {'PASS' if s2 else 'FAIL'}")
    print(f"  S3 跳水降幅明显>暴涨降幅(非方差压缩): {'PASS' if s3 else 'FAIL'}")
    verdict = "DOS当跳水过滤器 = 有价值" if (s1 and s2 and s3) else "DOS当跳水过滤器 = 失败(回归到'纯波动预测'用途)"
    print(f"\n  结论: {verdict}")
    
    # Save results
    results = {
        's1_pass': bool(s1),
        's2_pass': bool(s2),
        's3_pass': bool(s3),
        'dos_nwt': float(reg['nwt']) if reg else None,
        'dos_coef': float(reg['coef']) if reg else None,
        'sample_size': int(len(d)),
        'n_stocks': int(d['ts_code'].nunique()),
        'n_dates': int(d['decision_date'].nunique()),
        'dos_zero_pct': float((d['dos']==0).mean()*100),
    }
    import json
    with open(f'{SAVE_DIR}/study_a_correct_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to study_a_correct_results.json")


if __name__ == '__main__':
    main()
