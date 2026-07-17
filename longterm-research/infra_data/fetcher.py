"""
数据获取模块

从 Tushare API 获取 A股行情、技术指标、筹码分布等数据
重构自原 data_extraction.py
"""

import os
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

import pandas as pd
import tushare as ts
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from utils.retry_utils import retry, RetryContext
from utils.date_utils import get_latest_date_file, format_date, get_today


class DataFetcher:
    """
    数据获取器
    
    从 Tushare API 获取各类 A股数据并存储为 parquet 文件
    
    Example:
        fetcher = DataFetcher()
        fetcher.fetch_all(end_date='20251118')
    """
    
    def __init__(self, token: Optional[str] = None, api_url: Optional[str] = None):
        """
        初始化数据获取器
        
        Args:
            token: Tushare API Token，默认从配置读取
            api_url: Tushare API URL，默认从配置读取
        """
        self.token = token or settings.TUSHARE_TOKEN
        self.api_url = api_url or getattr(settings, 'TUSHARE_API_URL', None)
        
        # 初始化 Tushare API
        self.pro = ts.pro_api(self.token)
        if self.api_url:
            self.pro._DataApi__http_url = self.api_url
        
        # 数据存储路径
        self.data_path = settings.DATA_PATH
        
        # 确保目录存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保所有数据目录存在"""
        directories = [
            settings.daily_data_path,
            settings.other_data_path,
            settings.skill_data_path,
            settings.recommend_data_path,
            settings.ths_rank_path,
            settings.money_flow_path,
            settings.moneyflow_path,
            settings.board_path,
            settings.cyq_path,
        ]
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
    
    @retry(max_attempts=10, delay=7)
    def _api_call(self, method: str, **kwargs) -> pd.DataFrame:
        """统一的 API 调用方法"""
        if hasattr(self.pro, method):
            func = getattr(self.pro, method)
            return func(**kwargs)
        else:
            return self.pro.query(method, **kwargs)
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        return self._api_call(
            'stock_basic',
            exchange='',
            list_status='L',
            fields='ts_code,symbol,name,area,industry,list_date'
        )
    
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """获取交易日期列表"""
        df = self._api_call('daily', ts_code='000001.SZ', 
                           start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            raise ValueError(f"无法获取交易日期: {start_date} - {end_date}")
        return sorted(df['trade_date'].unique().tolist())
    
    def fetch_daily_quotes(self, dates: List[str], show_progress: bool = True):
        """
        获取日线行情数据
        
        ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
        """
        save_path = settings.daily_data_path
        existing_files = set(os.listdir(save_path))
        
        iterator = tqdm(dates, desc="获取日线数据") if show_progress else dates
        failed = []
        
        for date in iterator:
            if f"{date}.parquet" in existing_files:
                continue
            
            with RetryContext(max_attempts=10, delay=7) as ctx:
                while ctx.should_continue():
                    try:
                        df = self.pro.query('daily', ts_code='', trade_date=date)
                        df.to_parquet(os.path.join(save_path, f"{date}.parquet"))
                        break
                    except Exception as e:
                        ctx.record_failure(e)
                
                if ctx.attempt >= ctx.max_attempts:
                    failed.append(date)
        
        return failed
    
    def fetch_daily_basic(self, dates: List[str], show_progress: bool = True):
        """
        获取每日指标数据
        
        ts_code, trade_date, close, turnover_rate, volume_ratio, pe, pb, circ_mv
        """
        save_path = settings.other_data_path
        existing_files = set(os.listdir(save_path))
        
        iterator = tqdm(dates, desc="获取每日指标") if show_progress else dates
        failed = []
        
        for date in iterator:
            if f"{date}.parquet" in existing_files:
                continue
            
            with RetryContext(max_attempts=10, delay=7) as ctx:
                while ctx.should_continue():
                    try:
                        df = self.pro.daily_basic(
                            ts_code='', trade_date=date,
                            fields='ts_code,trade_date,close,turnover_rate,volume_ratio,pe,pb,circ_mv'
                        )
                        df.to_parquet(os.path.join(save_path, f"{date}.parquet"))
                        break
                    except Exception as e:
                        ctx.record_failure(e)
                
                if ctx.attempt >= ctx.max_attempts:
                    failed.append(date)
        
        return failed
    
    def fetch_technical_factors(self, dates: List[str], show_progress: bool = True):
        """
        获取技术因子数据
        
        ts_code, trade_date, macd, vol, amount, rsi_6, rsi_12, rsi_24, 
        boll_upper, boll_mid, boll_lower, cci, macd_dea, kdj_k, kdj_d, kdj_j
        """
        save_path = settings.skill_data_path
        existing_files = set(os.listdir(save_path))
        
        iterator = tqdm(dates, desc="获取技术因子") if show_progress else dates
        failed = []
        
        for date in iterator:
            if f"{date}.parquet" in existing_files:
                continue
            
            with RetryContext(max_attempts=10, delay=7) as ctx:
                while ctx.should_continue():
                    try:
                        df = self.pro.stk_factor(
                            ts_code='', trade_date=date,
                            fields='ts_code,trade_date,macd,vol,amount,rsi_6,rsi_12,rsi_24,'
                                   'boll_upper,boll_mid,boll_lower,cci,macd_dea,kdj_k,kdj_d,kdj_j'
                        )
                        df.to_parquet(os.path.join(save_path, f"{date}.parquet"))
                        break
                    except Exception as e:
                        ctx.record_failure(e)
                
                if ctx.attempt >= ctx.max_attempts:
                    failed.append(date)
        
        return failed
    
    def fetch_broker_recommend(self, months: List[str], show_progress: bool = True):
        """获取券商推荐数据"""
        save_path = settings.recommend_data_path
        existing_files = set(os.listdir(save_path))
        
        unique_months = sorted(set([d[:6] for d in months]))
        iterator = tqdm(unique_months, desc="获取券商推荐") if show_progress else unique_months
        failed = []
        
        for month in iterator:
            if f"{month}.parquet" in existing_files:
                continue
            
            with RetryContext(max_attempts=10, delay=7) as ctx:
                while ctx.should_continue():
                    try:
                        df = self.pro.broker_recommend(month=month)
                        df.to_parquet(os.path.join(save_path, f"{month}.parquet"))
                        break
                    except Exception as e:
                        ctx.record_failure(e)
                
                if ctx.attempt >= ctx.max_attempts:
                    failed.append(month)
        
        return failed
    
    def fetch_ths_hot(self, dates: List[str], show_progress: bool = True):
        """获取同花顺热榜数据"""
        save_path = settings.ths_rank_path
        existing_files = set(os.listdir(save_path))
        
        iterator = tqdm(dates, desc="获取同花顺热榜") if show_progress else dates
        failed = []
        
        for date in iterator:
            if f"{date}.parquet" in existing_files:
                continue
            
            with RetryContext(max_attempts=10, delay=7) as ctx:
                while ctx.should_continue():
                    try:
                        df = self.pro.ths_hot(
                            trade_date=date, market='热股',
                            fields='ts_code,ts_name,hot,concept'
                        )
                        df.to_parquet(os.path.join(save_path, f"{date}.parquet"))
                        break
                    except Exception as e:
                        ctx.record_failure(e)
                
                if ctx.attempt >= ctx.max_attempts:
                    failed.append(date)
        
        return failed
    
    def fetch_money_flow(self, dates: List[str], show_progress: bool = True):
        """获取资金流向数据"""
        save_path = settings.moneyflow_path
        existing_files = set(os.listdir(save_path))
        
        iterator = tqdm(dates, desc="获取资金流向") if show_progress else dates
        failed = []
        
        for date in iterator:
            if f"{date}.parquet" in existing_files:
                continue
            
            with RetryContext(max_attempts=10, delay=7) as ctx:
                while ctx.should_continue():
                    try:
                        df = self.pro.moneyflow(trade_date=date)
                        df.to_parquet(os.path.join(save_path, f"{date}.parquet"))
                        break
                    except Exception as e:
                        ctx.record_failure(e)
                
                if ctx.attempt >= ctx.max_attempts:
                    failed.append(date)
        
        return failed
    
    def fetch_cyq_data(self, end_date: str, start_date: str = None, show_progress: bool = True):
        """
        获取筹码分布数据
        
        这个方法按股票遍历获取，然后按日期保存
        """
        save_path = settings.cyq_path
        
        # 获取最新已有日期
        latest = get_latest_date_file(save_path)
        if latest and latest >= end_date:
            print("筹码数据已是最新")
            return []
        
        # 获取股票列表
        stock_data = self._api_call('daily', ts_code='', start_date=end_date, end_date=end_date)
        ts_codes = stock_data['ts_code'].unique()
        
        # 获取交易日期
        dates_df = self._api_call('daily', ts_code='000001.SZ',
                                  start_date=start_date or '20200101', end_date=end_date)
        
        if latest:
            dates_df = dates_df[dates_df['trade_date'] >= latest]
        
        date_range = sorted(dates_df['trade_date'].unique())
        cache = {d: [] for d in date_range}
        failed = []
        
        iterator = tqdm(ts_codes, desc="获取筹码数据") if show_progress else ts_codes
        
        for ts_code in iterator:
            with RetryContext(max_attempts=3, delay=7) as ctx:
                while ctx.should_continue():
                    try:
                        cyq_df = self.pro.cyq_perf(
                            ts_code=ts_code,
                            start_date=str(date_range[0]),
                            end_date=str(date_range[-1])
                        )
                        if cyq_df.empty:
                            break
                        
                        for trade_date, g in cyq_df.groupby('trade_date'):
                            if trade_date in cache:
                                cache[trade_date].append(g)
                        break
                    except Exception as e:
                        ctx.record_failure(e)
                
                if ctx.attempt >= ctx.max_attempts:
                    failed.append(ts_code)
        
        # 写入文件
        for trade_date, df_list in tqdm(cache.items(), desc="保存筹码数据"):
            if not df_list:
                continue
            out_file = os.path.join(save_path, f'{trade_date}.parquet')
            if os.path.exists(out_file):
                continue
            pd.concat(df_list, ignore_index=True).to_parquet(out_file, index=False)
        
        return failed
    
    def fetch_industry_data(self, dates: List[str], show_progress: bool = True):
        """获取行业板块数据"""
        save_path = settings.board_path
        existing_files = set(os.listdir(save_path))
        
        iterator = tqdm(dates, desc="获取行业数据") if show_progress else dates
        failed = []
        
        for date in iterator:
            if f"{date}.parquet" in existing_files:
                continue
            
            with RetryContext(max_attempts=10, delay=7) as ctx:
                while ctx.should_continue():
                    try:
                        df = self.pro.ths_daily(ts_code='', trade_date=date)
                        df.to_parquet(os.path.join(save_path, f"{date}.parquet"))
                        break
                    except Exception as e:
                        ctx.record_failure(e)
                
                if ctx.attempt >= ctx.max_attempts:
                    failed.append(date)
        
        return failed
    
    def fetch_vix_data(self, start_date: str, end_date: str, show_progress: bool = True):
        """
        获取中国波指(000188.SH)数据
        """
        save_path = settings.vix_data_path
        os.makedirs(save_path, exist_ok=True)
        
        # 尝试使用不同的方法获取中国波指数据
        print("尝试获取中国波指数据...")
        
        try:
            # 方法1: 使用index_daily接口
            print("方法1: 使用index_daily接口")
            df = self.pro.index_daily(
                ts_code='000130.SH',
                start_date=start_date,
                end_date=end_date
            )
            print(f"方法1结果: {df.shape}")
            if not df.empty:
                print(f"数据内容: {df.head()}")
                # 按日期保存
                for date in df['trade_date'].unique():
                    date_df = df[df['trade_date'] == date]
                    date_df.to_parquet(os.path.join(save_path, f"{date}.parquet"))
                print(f"成功保存 {len(df['trade_date'].unique())} 天的数据")
                return []
        except Exception as e:
            print(f"方法1失败: {str(e)}")
        
        try:
            # 方法2: 使用daily接口（可能中国波指作为普通股票处理）
            print("方法2: 使用daily接口")
            df = self.pro.daily(
                ts_code='000130.SH',
                start_date=start_date,
                end_date=end_date
            )
            print(f"方法2结果: {df.shape}")
            if not df.empty:
                print(f"数据内容: {df.head()}")
                # 按日期保存
                for date in df['trade_date'].unique():
                    date_df = df[df['trade_date'] == date]
                    date_df.to_parquet(os.path.join(save_path, f"{date}.parquet"))
                print(f"成功保存 {len(df['trade_date'].unique())} 天的数据")
                return []
        except Exception as e:
            print(f"方法2失败: {str(e)}")
        
        try:
            # 方法3: 使用通用行情接口
            print("方法3: 使用通用行情接口")
            df = self.pro.query('index_daily', 
                               ts_code='000130.SH',
                               start_date=start_date,
                               end_date=end_date)
            print(f"方法3结果: {df.shape}")
            if not df.empty:
                print(f"数据内容: {df.head()}")
                # 按日期保存
                for date in df['trade_date'].unique():
                    date_df = df[df['trade_date'] == date]
                    date_df.to_parquet(os.path.join(save_path, f"{date}.parquet"))
                print(f"成功保存 {len(df['trade_date'].unique())} 天的数据")
                return []
        except Exception as e:
            print(f"方法3失败: {str(e)}")
        
        print("所有方法都失败，无法获取中国波指数据")
        return []
    
    def fetch_margin_data(self, dates: List[str], show_progress: bool = True):
        """
        获取融资融券交易汇总数据
        """
        save_path = settings.margin_data_path
        os.makedirs(save_path, exist_ok=True)
        
        existing_files = set(os.listdir(save_path))
        
        iterator = tqdm(dates, desc="获取融资融券数据") if show_progress else dates
        failed = []
        
        for date in iterator:
            if f"{date}.parquet" in existing_files:
                continue
            
            with RetryContext(max_attempts=10, delay=7) as ctx:
                while ctx.should_continue():
                    try:
                        df = self.pro.margin(trade_date=date)
                        df.to_parquet(os.path.join(save_path, f"{date}.parquet"))
                        break
                    except Exception as e:
                        ctx.record_failure(e)
                
                if ctx.attempt >= ctx.max_attempts:
                    failed.append(date)
        
        return failed
    
    def fetch_all(self, end_date: str = None, start_date: str = "20200101"):
        """
        获取所有数据
        
        Args:
            end_date: 结束日期，默认今天
            start_date: 开始日期，默认 20200101
        """
        if end_date is None:
            end_date = get_today()
        
        print(f"开始获取数据: {start_date} - {end_date}")
        
        # 获取交易日期
        dates = self.get_trading_dates(start_date, end_date)
        print(f"共 {len(dates)} 个交易日")
        
        # 依次获取各类数据
        results = {
            'daily_quotes': self.fetch_daily_quotes(dates),
            'daily_basic': self.fetch_daily_basic(dates),
            'technical_factors': self.fetch_technical_factors(dates),
            'broker_recommend': self.fetch_broker_recommend(dates),
            'ths_hot': self.fetch_ths_hot(dates),
            'money_flow': self.fetch_money_flow(dates),
            'industry': self.fetch_industry_data(dates),
            'cyq': self.fetch_cyq_data(end_date, start_date),
            'vix': self.fetch_vix_data(start_date, end_date),
            'margin': self.fetch_margin_data(dates),
        }
        
        # 统计失败情况
        total_failed = sum(len(v) for v in results.values())
        if total_failed > 0:
            print(f"\n警告: 共有 {total_failed} 项数据获取失败")
            for key, failed in results.items():
                if failed:
                    print(f"  - {key}: {len(failed)} 项失败")
        else:
            print("\n所有数据获取成功!")
        
        return results


def main(end_date: str = None):
    """主函数入口（兼容旧接口）"""
    fetcher = DataFetcher()
    fetcher.fetch_all(end_date=end_date)


if __name__ == "__main__":
    main()
