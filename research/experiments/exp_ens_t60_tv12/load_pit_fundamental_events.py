# -*- coding: utf-8 -*-
"""PIT 基本面事件加载器与消息驱动下跌识别器 (Point-in-Time Fundamental Event Loader)

实现《A股20日收益目标：论文、因子与GitHub实现的增强研究 (2026-09-08)》方向A：
- 严格遵循 PIT (Point-in-Time) 原则：决策日 T 仅可使用 ann_date <= T 的已公告财报/快报；
- 识别“严重业绩负面事件”：净利润同比暴跌 (netprofit_yoy < -30%)、发生亏损 (roe < 0 或 dt_eps < 0)；
- 区分“消息驱动下跌 (News-Driven Drop)”与“暂时卖压下跌 (Liquidity Drop)”。
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = r"D:\iquant_data\data_v2"
FINA_CACHE_FP = os.path.join(DATA_DIR, "fundamental1", "fina_indicator_cache.parquet")


class PITFundamentalEventManager:
    """PIT 基本面事件管理与条件反转过滤器"""

    def __init__(self, fina_fp=FINA_CACHE_FP):
        if not os.path.exists(fina_fp):
            raise FileNotFoundError(f"财务指标缓存文件未找到: {fina_fp}")
        self.fina_df = pd.read_parquet(fina_fp)
        self.fina_df["ann_int"] = self.fina_df["ann_date"].astype(int)
        self.fina_df["end_int"] = self.fina_df["end_date"].astype(int)
        # 预计算单条公告的严重负面标记
        self.fina_df["is_bad_news"] = (
            (self.fina_df["netprofit_yoy"] < -30.0) |
            (self.fina_df["roe"] < 0.0) |
            (self.fina_df["dt_eps"] < 0.0)
        )
        print(f"[PITManager] 加载财务记录 {len(self.fina_df)} 条, 覆盖代码 {self.fina_df['ts_code'].nunique()} 只.")

    def get_negative_news_stocks(self, decision_date: int, lookback_calendar_days: int = 30) -> set:
        """获取在决策日 T 过去 lookback_calendar_days 天内公告严重负面事件的股票集合
        严格 PIT: ann_date <= decision_date
        """
        d_dt = pd.to_datetime(str(decision_date))
        start_dt = d_dt - pd.Timedelta(days=lookback_calendar_days)
        start_int = int(start_dt.strftime("%Y%m%d"))

        # 严格切片已公告区间
        mask = (
            (self.fina_df["ann_int"] <= decision_date) &
            (self.fina_df["ann_int"] >= start_int) &
            (self.fina_df["is_bad_news"])
        )
        bad_df = self.fina_df[mask]
        return set(bad_df["ts_code"].unique())

    def classify_losers(self, current_panel: pd.DataFrame, decision_date: int, lookback_calendar_days: int = 30):
        """对截面上的下跌股票 (ret_1m < 0) 进行二元归因:
        - news_driven_drop: 过去 30 天内有负面公告，下跌属于业绩预期恶化驱动
        - liquidity_drop: 过去 30 天内无负面公告，下跌属于无基本面利空的交易卖压驱动 (符合反转补偿假说)
        """
        bad_news_set = self.get_negative_news_stocks(decision_date, lookback_calendar_days)
        
        df = current_panel.copy()
        df["has_bad_news"] = df["ts_code"].isin(bad_news_set)
        
        # 判断下跌
        ret_col = "ret_1m" if "ret_1m" in df.columns else "momentum_20"
        df["is_loser"] = df[ret_col] < 0.0
        
        df["is_news_drop"] = df["is_loser"] & df["has_bad_news"]
        df["is_liquidity_drop"] = df["is_loser"] & (~df["has_bad_news"])
        
        return df


if __name__ == "__main__":
    mgr = PITFundamentalEventManager()
    bad_20230428 = mgr.get_negative_news_stocks(20230428, lookback_calendar_days=30)
    print(f"20230428 过去 30 天公告暴雷股票数: {len(bad_20230428)}")
