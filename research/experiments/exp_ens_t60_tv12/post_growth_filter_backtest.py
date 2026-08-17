# -*- coding: utf-8 -*-
"""选股后"成长制造类"白名单过滤对照实验。

用户提问: 在全市场 ENS 选股(T40)之后, 把"成长制造类"股票单独筛出来交易,
是否比全量 Top40 有更高收益?

方法 (选股后过滤, post-filter):
  - 基线: ENS 打分 → select_with_limit → Top40 (当前最优配置)
  - 变体: 同上选 Top40 后, 再只保留白名单细分行业的股票 (等权分配到剩余持仓)
两个白名单:
  - GM_FULL: 成长/制造 + 科技成长 + 能源/材料 + 医药/消费/农业 (PREFERRED 全展开)
  - GM_CORE: 严格"成长制造+科技" (制造/工业 + 科技成长 + 新能源, 剔除医药消费/能源材料/周期)

对照组 (选股前过滤, pre-filter): 打分池直接限制为 GM_CORE 再选 Top40,
用于回答"过滤时机"是否影响结论。

配置: 全市场 T40 + tiered(s123三档) + dd_degrade(-10%×0.5), 与当前最优基线一致。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = r"c:\Users\liuqi\quant_system_v2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research", "experiments", "exp_ens_t60_tv12"))
sys.path.insert(0, os.path.join(ROOT, "research", "sector_rotation"))

from engine import init_shared, run_backtest_tiered  # noqa: E402

t0 = time.time()
SQRT_242 = np.sqrt(242.0)

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
    # 制造/工业 (用户重点: 汽车制造/工业)
    "汽车整车", "汽车配件", "汽车服务", "摩托车", "家用电器", "水泥", "玻璃",
    "其他建材", "专用机械", "工程机械", "机床制造", "机械基件", "轻工机械",
    "纺织机械", "农用机械",
    # 科技成长
    "半导体", "元器件", "电器仪表", "IT设备", "软件服务", "互联网",
    "通信设备", "电信运营", "航空", "船舶", "影视音像", "出版业", "广告包装",
    # 新能源/电力设备
    "新型电力", "电气设备",
}


def metrics(nav_s):
    nav_s = nav_s.sort_index().astype(float)
    tot = nav_s.iloc[-1] / nav_s.iloc[0] - 1.0
    yrs = len(nav_s) / 242.0
    ann = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    dd_s = nav_s / nav_s.cummax() - 1.0
    ret = nav_s.pct_change().fillna(0.0)
    sharpe = ret.mean() / (ret.std() + 1e-8) * SQRT_242
    nav_m = nav_s.groupby((nav_s.index // 100).astype(str)).last()
    dd_m = nav_m / nav_m.cummax() - 1.0
    return {"ann": ann, "maxdd": dd_s.min(), "maxdd_m": dd_m.min(),
            "calmar": ann / (-dd_s.min() + 1e-9), "sharpe": sharpe}


def main():
    print("[1] 加载全市场 shared...", flush=True)
    sh = init_shared("fullmarket")
    ind_map = sh["ind_map"]
    print(f"    完成 {time.time()-t0:.0f}s, panel {len(sh['panel']):,} 行", flush=True)

    # 检查白名单覆盖
    inds = set(ind_map.values())
    print(f"    细分行业数 {len(inds)}, GM_FULL 命中 {len(GM_FULL & inds)}/"
          f"{len(GM_FULL)}, GM_CORE 命中 {len(GM_CORE & inds)}/{len(GM_CORE)}", flush=True)

    hdr = f"{'config':<26} {'CAGR':>8} {'Sharpe':>7} {'日MaxDD':>9} {'月MaxDD':>9} {'Calmar':>6}"
    print("\n" + hdr)
    print("-" * 76)

    def show(tag, kw):
        nav, _ = run_backtest_tiered(sh, "ENS", "T40", tgt_vol=None, timing_mode="tiered",
                                     dd_degrade=-0.10, dd_degrade_scale=0.5, **kw)
        m = metrics(nav)
        print(f"{tag:<26} {m['ann']:7.2%} {m['sharpe']:7.2f} {m['maxdd']:8.2%} "
              f"{m['maxdd_m']:8.2%} {m['calmar']:6.2f}", flush=True)
        return nav

    nav_base = show("基线 (无过滤)", {})
    print("-" * 76)
    nav_full = show("选股后过滤 GM_FULL", {"post_whitelist": GM_FULL})
    nav_core = show("选股后过滤 GM_CORE", {"post_whitelist": GM_CORE})
    print("-" * 76)
    nav_pre = show("选股前过滤 GM_CORE", {"pre_whitelist_ind": GM_CORE})

    # 分年度对比
    def yearly(nav):
        out = {}
        for y, g in nav.groupby(nav.index // 10000):
            out[y] = g.iloc[-1] / g.iloc[0] - 1.0
        return out

    yrs = sorted(set(nav_base.index // 10000))
    print(f"\n# 分年度收益")
    print(f"{'year':<6} {'基线':>8} {'GM_FULL':>9} {'GM_CORE':>9} {'PRE_CORE':>9}")
    for y in yrs:
        print(f"{y:<6} {yearly(nav_base).get(y, float('nan')):>8.1%} "
              f"{yearly(nav_full).get(y, float('nan')):>9.1%} "
              f"{yearly(nav_core).get(y, float('nan')):>9.1%} "
              f"{yearly(nav_pre).get(y, float('nan')):>9.1%}")

    print(f"\n[完成] 总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
