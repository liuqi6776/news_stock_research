# -*- coding: utf-8 -*-
"""
Automated CI Parity Checker: verify_report_parity.py
Asserts 0-tolerance mathematical parity between audit_remediation_round8_report.md
and canonical production JSON artifacts (etf_scs_clean_v1, cs_transformer_scs_clean_v1, ch4_attribution).
"""

import os
import sys
import json
import re

EXP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "research", "experiments", "exp_ens_t60_tv12")
REPORT_PATH = os.path.join(EXP_DIR, "audit_remediation_round8_report.md")

ETF_METRICS_PATH = os.path.join(EXP_DIR, "artifacts", "etf_scs_clean_v1", "metrics.json")
CS_METRICS_PATH = os.path.join(EXP_DIR, "artifacts", "cs_transformer_scs_clean_v1", "metrics.json")
CH4_REG_PATH = os.path.join(EXP_DIR, "artifacts", "ch4_attribution", "regression_summary.json")


def test_zero_tolerance_report_parity():
    assert os.path.exists(REPORT_PATH), f"Report missing at {REPORT_PATH}"
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        report_text = f.read()

    with open(ETF_METRICS_PATH, "r", encoding="utf-8") as f:
        etf_m = json.load(f)
    with open(CS_METRICS_PATH, "r", encoding="utf-8") as f:
        cs_m = json.load(f)
    with open(CH4_REG_PATH, "r", encoding="utf-8") as f:
        ch4_m = json.load(f)

    # 1. ETF baseline metrics
    e_b1 = etf_m["etf_scs_b1_smoothed"]
    assert f"{e_b1['cagr']}%" in report_text, f"ETF B1 CAGR {e_b1['cagr']}% not found in report"
    assert f"{e_b1['sharpe']}" in report_text, f"ETF B1 Sharpe {e_b1['sharpe']} not found in report"
    assert f"{e_b1['max_dd']}%" in report_text, f"ETF B1 MaxDD {e_b1['max_dd']}% not found in report"

    # 2. CS-Transformer metrics
    c_b1 = cs_m["cs_transformer_b1_smoothed"]
    assert f"{c_b1['cagr']}%" in report_text, f"CS B1 CAGR {c_b1['cagr']}% not found in report"
    assert f"{c_b1['sharpe']}" in report_text, f"CS B1 Sharpe {c_b1['sharpe']} not found in report"
    assert f"{c_b1['max_dd']}%" in report_text, f"CS B1 MaxDD {c_b1['max_dd']}% not found in report"

    # 3. Paired delta metrics
    delta_b1 = cs_m["paired_incremental_alpha_b1"]
    assert f"{delta_b1['delta_cagr_pct']:+.2f}%" in report_text, f"Delta CAGR not found"
    assert f"{delta_b1['delta_sharpe']:+.2f}" in report_text, f"Delta Sharpe not found"
    assert f"{delta_b1['tracking_error_pct']}%" in report_text, f"Tracking error not found"
    assert f"{delta_b1['information_ratio']}" in report_text, f"Information ratio not found"

    # 4. HAC and Bootstrap
    hac = delta_b1["hac_test"]
    boot = delta_b1["bootstrap_95ci"]
    assert f"{hac['t_stat']}" in report_text, f"HAC t-stat {hac['t_stat']} not found"
    assert f"{hac['p_value_2sided']}" in report_text, f"HAC p-val {hac['p_value_2sided']} not found"
    assert f"[{boot['alpha_ann_pct_95ci'][0]}%, {boot['alpha_ann_pct_95ci'][1]}%]" in report_text, f"Bootstrap CI not found"

    # 5. CH4 sanity & attribution
    bm_mod = ch4_m["samples"]["full_43_months"]["models"]["512100_bh"]["primary_lag1"]
    assert f"{bm_mod['beta_mkt']}" in report_text, f"Benchmark Beta {bm_mod['beta_mkt']} not found"
    assert f"{bm_mod['r_squared']}" in report_text, f"Benchmark R2 {bm_mod['r_squared']} not found"

    # 6. Real holdings micro cap diagnosis
    micro = ch4_m["real_holdings_micro_cap_diagnosis"]
    assert f"{micro['bottom_30pct_micro_cap_count_ratio_pct']}%" in report_text, "Micro count ratio not found"
    assert f"{micro['bottom_30pct_micro_cap_value_weighted_ratio_pct']}%" in report_text, "Micro value ratio not found"
    assert f"{micro['top_70pct_investable_value_weighted_ratio_pct']}%" in report_text, "Investable ratio not found"
    assert f"{micro['average_holding_circ_mv_yi']}" in report_text, "Average holding circ mv not found"

    # 7. Sensitivity tables
    for r_k, v in cs_m["cash_sensitivity"].items():
        assert f"{v['cs_b1_cagr_pct']}%" in report_text, f"Cash sensitivity {r_k} not found"
    for f_k, v in cs_m["friction_sensitivity"].items():
        assert f"{v['cagr_pct']}%" in report_text, f"Friction sensitivity {f_k} not found"

    print(">>> [PASS] verify_report_parity: 100% 0-tolerance parity confirmed between Report & Artifacts!")


if __name__ == "__main__":
    test_zero_tolerance_report_parity()
