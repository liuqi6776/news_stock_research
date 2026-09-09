# -*- coding: utf-8 -*-
"""真实流动性与换手率波动计算器 (True Amihud Illiquidity & Turnover Volatility Engine)

实现《A股20日收益目标：论文、因子与GitHub实现的增强研究 (2026-09-08)》：
- 严格遵循经典 Amihud (1997/2002) 定义：基于日收益绝对值与真实成交金额构建非流动性度量；
- 提取真实换手率标准差 (turnover_vol_20) 与换手率放大比率 (turnover_surge)；
- 遵循审计原则：仅用于容量约束、不可交易风险诊断与卖压条件分类，严禁直接对低流动性股票赋予加分。
"""
import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = r"D:\iquant_data\data_v2"
DAY1_DIR = os.path.join(DATA_DIR, "data_day1")
OTHER1_DIR = os.path.join(DATA_DIR, "other_day1")


class TrueLiquidityManager:
    """真实流动性与交易摩擦度量管理器"""

    def __init__(self, day1_dir=DAY1_DIR, other1_dir=OTHER1_DIR):
        self.day1_dir = day1_dir
        self.other1_dir = other1_dir
        
        # 获取所有可用交易日文件列表并排序
        day_fps = sorted(glob.glob(os.path.join(self.day1_dir, "*.parquet")))
        self.available_dates = [int(os.path.splitext(os.path.basename(f))[0]) for f in day_fps]
        self.date_to_idx = {d: i for i, d in enumerate(self.available_dates)}

    def compute_liquidity_metrics_for_date(self, decision_date: int, window: int = 20) -> pd.DataFrame:
        """为决策日 T 计算过去 window 个交易日的真实 Amihud 与换手特征
        严格 PIT: 仅使用 <= decision_date 的交易日
        """
        if decision_date not in self.date_to_idx:
            # 找到 <= decision_date 的最后一个交易日
            valid_dates = [d for d in self.available_dates if d <= decision_date]
            if not valid_dates:
                return pd.DataFrame()
            idx = self.date_to_idx[valid_dates[-1]]
        else:
            idx = self.date_to_idx[decision_date]

        start_idx = max(0, idx - window + 1)
        sub_dates = self.available_dates[start_idx : idx + 1]

        day_dfs = []
        for d in sub_dates:
            fp1 = os.path.join(self.day1_dir, f"{d}.parquet")
            fp2 = os.path.join(self.other1_dir, f"{d}.parquet")
            if not (os.path.exists(fp1) and os.path.exists(fp2)):
                continue
            
            df1 = pd.read_parquet(fp1, columns=["ts_code", "trade_date", "pct_chg", "amount"])
            df2 = pd.read_parquet(fp2, columns=["ts_code", "trade_date", "turnover_rate", "circ_mv"])
            m = pd.merge(df1, df2, on=["ts_code", "trade_date"], how="inner")
            day_dfs.append(m)

        if not day_dfs:
            return pd.DataFrame()

        panel_20 = pd.concat(day_dfs, ignore_index=True)
        # amount 单位通常是千元或元，换算为亿元: amount / 1e8
        # amihud ratio: |pct_chg| / (amount_yi + 1e-4)
        panel_20["amount_yi"] = panel_20["amount"] / 1e8
        panel_20["daily_illiq"] = panel_20["pct_chg"].abs() / (panel_20["amount_yi"] + 1e-4)

        agg_dict = {
            "daily_illiq": "mean",
            "turnover_rate": ["std", "mean", "last"],
            "circ_mv": "last",
            "amount_yi": "mean"
        }
        res = panel_20.groupby("ts_code").agg(agg_dict)
        res.columns = [
            "true_amihud_illiq_20",
            "true_turnover_vol_20",
            "true_avg_turnover_20",
            "last_day_turnover",
            "circ_mv_last",
            "avg_daily_amount_yi"
        ]
        res = res.reset_index()
        res["trade_date"] = decision_date
        # 换手率异动比率 (当日换手 / 20日均值)
        res["turnover_surge_ratio"] = res["last_day_turnover"] / (res["true_avg_turnover_20"] + 1e-4)
        
        return res


if __name__ == "__main__":
    mgr = TrueLiquidityManager()
    res = mgr.compute_liquidity_metrics_for_date(20230428)
    print("Sample True Liquidity Metrics for 20230428:")
    print(res.head(3))
