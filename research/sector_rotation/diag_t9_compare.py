# -*- coding: utf-8 -*-
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "fund_research", "studies", "rotation_dingtou"))

from diag_t9_exit import run_backtest, monthly_view  # noqa: E402
from etf_optimize_backtest2 import (  # noqa: E402
    INDUSTRY_ETFS, load_industry_daily, load_hv_daily, build_series,
    hv_monthly_ret, monthly_from_daily, calc_stats,
)
from sector_rotation_traditional import TRADITIONAL_ETFS, build_signals4, run_graded  # noqa: E402

# 月频 run_graded T7
panel = load_industry_daily()
trad_codes = [c for _, c in TRADITIONAL_ETFS]
tp = {c: s for c, s in panel.items() if c in set(trad_codes)}
ew_trad = build_series(tp)
plain_trad_m = monthly_from_daily(ew_trad)
monthly_nav = {}
for code, s in panel.items():
    monthly_nav[code] = (1 + s).cumprod().groupby(s.index.str[:6]).last()
nav_panel = pd.DataFrame(monthly_nav).sort_index() if False else __import__("pandas").DataFrame(monthly_nav).sort_index()
hv = load_hv_daily()
v8_m = hv_monthly_ret(hv)
sig = build_signals4(list(nav_panel.index), nav_panel, trad_codes)
nv_m = run_graded(nav_panel, sig, plain_trad_m, v8_m, use_v8=True, mode="strict",
                  entry_sig=3, exit_sig=1, sig_col="s123")

# 日频 run_backtest T7
dd = run_backtest("s123le1")
m = monthly_view(dd)

print("关键时点 NAV 对比 (月频run_graded vs 日频run_backtest):")
print(f"{'ym':<8} {'月频T7':>10} {'日频T7':>10}")
for ym in ["201901", "201903", "202204", "202207", "202210", "202302", "202304",
           "202401", "202404", "202409", "202501", "202508", "202603", "202608"]:
    a = nv_m["nav"].get(ym, float("nan"))
    b = m["nav"].get(ym, float("nan"))
    print(f"{ym:<8} {a:>10.4f} {b:>10.4f}")
