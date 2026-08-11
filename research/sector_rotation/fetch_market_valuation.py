# -*- coding: utf-8 -*-
"""拉取并缓存 沪深300 PE-TTM + 10年国债 数据（S1/S2/S3 信号所需）"""
import os
import sys

DINGTOU_DIR = "c:/Users/liuqi/quant_system_v2/research/fund_research/studies/rotation_dingtou"
sys.path.insert(0, DINGTOU_DIR)
sys.path.insert(0, os.path.join(DINGTOU_DIR, "..", ".."))

from timing_dingtou import fetch_pe_csi300, fetch_bond10y

if __name__ == "__main__":
    try:
        pe = fetch_pe_csi300()
        print(f"PE OK: {len(pe)} rows")
    except Exception as e:
        print(f"PE FAIL: {e}")
    try:
        bond = fetch_bond10y()
        print(f"BOND OK: {len(bond)} rows")
    except Exception as e:
        print(f"BOND FAIL: {e}")
