"""
Strict T+1 Strategy Backtest

核心逻辑 (The user's defined plan):
1. T日 (Day 1): 盘后基于迄今为止(包括T日收盘)的数据进行模型预测，生成T+1日买入信号。
2. T+1日 (Day 2): 按照T日生成的信号，在开盘价(Open)位置买入。如果开盘涨停则无法买入。
3. T+2日 (Day 3): 根据设定的风控或简单的持有期设定，在开盘价(Open)位置卖出此前按T+1买入的股票。
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infra_data.storage import DataStorage
from features.enhanced_factors import calculate_all_enhanced_features
from processing.cleaner import filter_stock_codes
from processing.merger import merge_dataframes
import sys
import os
import importlib.util

# 稳妥起见，直接从绝对路径加载 iquant 的 process.py 模块
iquant_root = r"C:\Users\liuqi\iquant\quant_trading_system"
if iquant_root not in sys.path:
    sys.path.insert(0, iquant_root)

pr_path = os.path.join(iquant_root, "data", "process.py")
config_path = os.path.join(iquant_root, "config.py")

# 强制将 'config' 映射到 iquant 下的 config.py，防止被 quant_system_v2/config 文件夹拦截
if os.path.exists(config_path):
    spec_config = importlib.util.spec_from_file_location("config", config_path)
    config_mod = importlib.util.module_from_spec(spec_config)
    sys.modules["config"] = config_mod
    spec_config.loader.exec_module(config_mod)

spec = importlib.util.spec_from_file_location("iquant_process", pr_path)
pr = importlib.util.module_from_spec(spec)
sys.modules["iquant_process"] = pr
spec.loader.exec_module(pr)

warnings.filterwarnings('ignore')

class StrictT1Backtest:
    def __init__(
        self,
        initial_capital: float = 100000,
        top_n: int = 5,
        commission_rate: float = 0.0003,
        stamp_duty: float = 0.0005,
        slippage: float = 0.002,
        signal_threshold: float = 0.75,
        entry_timing: str = 'open', # 'open' or 'close'
        exit_timing: str = 'open',    # 'open' or 'close'
        stop_loss: float = -0.035     # -3.5% Hard Stop-Loss
    ):
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty
        self.slippage = slippage
        self.signal_threshold = signal_threshold
        self.entry_timing = entry_timing
        self.exit_timing = exit_timing
        self.stop_loss = stop_loss
        
        self.storage = DataStorage()
        self.capital = initial_capital
        
        # 记录持仓: {ts_code: {'shares': 100, 'buy_price': 10.0, 'buy_date': 'YYYYMMDD'}}
        self.positions = {} 
        self.equity_curve = []
        self.trades = []

    def simple_t1_labeling(self, df):
        """
        修正后的 T+1 收益标签：
        如果 entry='open', exit='open', 收益 = (T+2_Open - T+1_Open) / T+1_Open
        如果 entry='close', exit='close', 收益 = (T+2_Close - T+1_Close) / T+1_Close
        """
        print(f"生成 T+1 收益标签 (模式: Entry={self.entry_timing}, Exit={self.exit_timing})...")
        df = df.sort_values(['ts_code', 'trade_date'])
        
        if self.entry_timing == 'open':
            # T+1 开盘
            df['target_entry'] = df.groupby('ts_code')['open'].shift(-1)
        else:
            # T+1 收盘
            df['target_entry'] = df.groupby('ts_code')['close'].shift(-1)
            
        if self.exit_timing == 'open':
            # T+2 开盘
            df['target_exit'] = df.groupby('ts_code')['open'].shift(-2)
        else:
            # T+2 收盘
            df['target_exit'] = df.groupby('ts_code')['close'].shift(-2)
        
        # 计算实际收益
        df['t1_trade_return'] = (df['target_exit'] - df['target_entry']) / df['target_entry']
        
        # 标签设定：收益率 > 0.005 记为 1 (降低阈值以捕捉更多 Alpha)
        df['return_label'] = 0
        df.loc[df['t1_trade_return'] > 0.005, 'return_label'] = 1
        return df

    def add_advanced_features_and_labels(self, df):
        # 1. 剔除ST股
        df = pr.remove_st_stocks(df)
        
        # 2. 计算 Alpha 101 因子
        df = pr.calculate_alpha101_factors(df)
        
        # 3. 三阻碍标签 (Triple Barrier) - 调整 Horizon 到 1 天
        # 使训练目标与 Day T+1 买，Day T+2 卖的现实完全一致
        df = pr.triple_barrier_labeling(df, horizon=1, pt=2.0, sl=1.0)
        
        # 4. 扩展窗口标准化 (消除前视偏差)
        cols_to_std = [
            'turnover_rate', 'volume_ratio', 'pe', 'pb', 'circ_mv',
            'macd', 'rsi_6', 'rsi_12', 'rsi_24', 'cci', 'kdj_k', 'kdj_d', 'kdj_j',
            'alpha_006', 'alpha_009', 'alpha_012', 'alpha_023', 'volatility_20'
        ]
        df = pr.apply_expanding_standardization(df, cols_to_std)
        
        return df

    def load_data(self, start_date, end_date):
        print(f"1. 加载数据 {start_date} - {end_date}...")
        daily = self.storage.load_daily_data(start_date, end_date)
        other = self.storage.load_daily_basic(start_date, end_date)
        skill = self.storage.load_technical_factors(start_date, end_date)
        money_flow = self.storage.load_money_flow(start_date, end_date)
        chip_data = self.storage.load_chip_data(start_date, end_date)
        
        dfs = [daily]
        if not other.empty: dfs.append(other)
        if not skill.empty: dfs.append(skill)
        
        df = merge_dataframes(dfs)
        df = filter_stock_codes(df, patterns=['^60', '^00'])
        
        print("2. 计算基础增强特征...")
        df = calculate_all_enhanced_features(df, money_flow, chip_data)
        
        print("3. 集成 Alpha 101 与三阻碍标签逻辑...")
        df = self.add_advanced_features_and_labels(df)
        
        print("4. 生成回归对照标签...")
        df = self.simple_t1_labeling(df)
        return df

    def get_index_filter(self):
        """加载上证指数并计算买入开关"""
        if not os.path.exists('sse_index_2023.csv'):
            print("Warning: sse_index_2023.csv not found. No market filter will be applied.")
            return {}
        
        index_df = pd.read_csv('sse_index_2023.csv')
        index_df['date'] = pd.to_datetime(index_df['date'])
        index_df = index_df.sort_values('date')
        index_df['ma20'] = index_df['close'].rolling(20).mean()
        index_df['bull_market'] = index_df['close'] > index_df['ma20']
        
        # 转换为 Dict 方便快速查询 [YYYYMMDD] -> Boolean
        index_df['date_str'] = index_df['date'].dt.strftime('%Y%m%d')
        return dict(zip(index_df['date_str'], index_df['bull_market']))

    def get_features(self, df):
        exclude = ['ts_code', 'trade_date', 'return_label', 't1_trade_return', 
                   'next_open', 'next_next_open', 'next_close', 'future_date', 'future_close', 
                   'growth_percentage', 'tb_label', 'name']
        # 动态捕捉所有 _norm 列作为主特征
        norms = [c for c in df.columns if c.endswith('_norm')]
        if len(norms) > 5:
            return norms
        return [c for c in df.columns if c not in exclude and df[c].dtype not in ['object', 'datetime64[ns]']]

    def train_model(self, train_df, features):
        print(f"    Train Data Size: {len(train_df)}")
        # 回归到最直接的 T+1 收益标签
        train_df = train_df.dropna(subset=['return_label'])
        
        # 标签分布
        majority_class = train_df[train_df['return_label'] == 0]
        minority_class = train_df[train_df['return_label'] == 1]
        
        print(f"    Label Distribution: 1(Positive):{len(minority_class)}, 0(Negative):{len(majority_class)}")
        
        # Sub-Sample to avoid large imbalance
        if len(minority_class) > 0 and len(majority_class) > len(minority_class):
            from sklearn.utils import resample
            majority_downsampled = resample(majority_class,
                                            replace=False, 
                                            n_samples=len(minority_class), 
                                            random_state=42)
            balanced_df = pd.concat([majority_downsampled, minority_class])
            train_df = balanced_df
            
        X = train_df[features].fillna(0).replace([np.inf, -np.inf], 0)
        y = train_df['return_label']
        
        # 简单的特征选择与缩放
        selector = SelectKBest(f_classif, k=min(60, len(features)))
        selector.fit(X, y)
        selected_features = [features[i] for i in selector.get_support(indices=True)]
        
        X = X[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            random_state=42, n_jobs=-1, use_label_encoder=False, 
            eval_metric='logloss', tree_method='hist'
        )
        model.fit(X_scaled, y)
        return model, scaler, selected_features

    def run(self, data_start='20200101', test_start='20230101', test_end='20260101'):
        df = self.load_data(data_start, test_end)
        features = self.get_features(df)
        
        df_test = df[df['trade_date'] >= test_start].copy()
        trade_dates = sorted(df_test['trade_date'].unique())
        
        # 缓存分组数据
        df_grouped = dict(tuple(df.groupby('trade_date')))
        
        # 记录待买入目标 (当日收盘后计算，次日买入)
        # 格式: [ts_code1, ts_code2...]
        targets_for_tomorrow = []
        
        # 记录模型状态
        last_train_week = -1
        model, scaler, selected_features = None, None, None
        
        # 获取大盘滤网
        market_filter = self.get_index_filter()
        
        # 回测主循环
        for current_date in tqdm(trade_dates, desc="T+1 Strategy Backtest"):
            # 是否处于大盘反弹波段
            is_bull = market_filter.get(current_date, True) # 默认允许交易
            
            day_data = df_grouped.get(current_date)
            if day_data is None or day_data.empty: continue
            
            day_data_dict = day_data.set_index('ts_code').to_dict('index')
            opens = dict(zip(day_data['ts_code'], day_data['open']))
            closes = dict(zip(day_data['ts_code'], day_data['close']))
            
            # -----------------------------------------------------------------
            # 阶段 A: 开盘时。执行上一日的指令 (卖出与买入)
            # -----------------------------------------------------------------
            
            # A1. 卖出 (根据 exit_timing)
            for ts_code in list(self.positions.keys()):
                if ts_code not in closes: continue
                
                pos = self.positions[ts_code]
                if pos['buy_date'] == current_date:
                    continue # T+1 以前买的才能卖 (遵循 T+1 规则)
                
                # 1. 触发硬止损判断 (按当前价格计算，如果还没到卖出时点但跌超了，也要考虑出局)
                # 简化逻辑：如果在卖出时点发现累计收益低于止损位，执行卖出
                current_pnl_pct = (closes[ts_code] - pos['buy_price']) / pos['buy_price']
                trigger_stop_loss = current_pnl_pct <= self.stop_loss
                
                # 2. 正常离场判断 (T+2 以后)
                # (保持现状：T+2 卖出)
                
                # 确定卖出价
                if trigger_stop_loss:
                    sell_price_base = closes[ts_code] # 盘中触发，以收盘计
                elif self.exit_timing == 'open':
                    sell_price_base = opens.get(ts_code, closes[ts_code])
                else:
                    sell_price_base = closes[ts_code]
                
                # 跌停限制
                pct_chg = day_data_dict[ts_code].get('pct_chg', 0)
                if pct_chg < -9.7: # 留点余量
                    continue 

                sell_price = sell_price_base * (1 - self.slippage)
                revenue = pos['shares'] * sell_price
                commission = max(5, revenue * self.commission_rate)
                stamp = revenue * self.stamp_duty
                net_income = revenue - commission - stamp
                
                self.capital += net_income
                pnl = net_income - (pos['shares'] * pos['buy_price'])
                
                self.trades.append({
                    'sell_date': current_date, 'buy_date': pos['buy_date'],
                    'code': ts_code, 'action': 'SELL', 'price': sell_price, 'shares': pos['shares'],
                    'pnl': pnl, 'pnl_pct': pnl / (pos['shares'] * pos['buy_price'])*100
                })
                del self.positions[ts_code]

            # A2. 买入 (根据昨晚决定的 target_for_tomorrow)
            # 头寸计算：总资产 / top_n
            nav_estimation = self.capital + sum([p['shares'] * closes.get(c, p['buy_price']) for c, p in self.positions.items()])
            slot_value = nav_estimation / self.top_n
            
            for ts_code in targets_for_tomorrow:
                # 大盘过滤逻辑：如果大盘趋势向下，禁止买入
                if not is_bull: break
                
                if len(self.positions) >= self.top_n: break
                if ts_code in self.positions: continue
                if ts_code not in closes: continue
                
                # 确定买入价
                if self.entry_timing == 'open':
                    buy_price_base = opens.get(ts_code, closes[ts_code])
                    # 开盘涨停无法买入限制
                    if day_data_dict[ts_code].get('pct_chg', 0) > 9.7:
                        continue
                else:
                    buy_price_base = closes[ts_code]
                    # 收盘涨停通常也能通过集合竞价买入，但保守起见也加限制
                    if day_data_dict[ts_code].get('pct_chg', 0) > 9.8:
                        continue
                
                buy_price = buy_price_base * (1 + self.slippage)
                if buy_price <= 0: continue
                
                # 计算股数
                shares_can_buy = int(slot_value / buy_price / 100) * 100
                if shares_can_buy < 100: continue
                
                cost = shares_can_buy * buy_price
                commission = max(5, cost * self.commission_rate)
                
                if self.capital >= (cost + commission):
                    self.capital -= (cost + commission)
                    self.positions[ts_code] = {
                        'shares': shares_can_buy,
                        'buy_price': buy_price,
                        'buy_date': current_date
                    }
                    self.trades.append({
                        'sell_date': '', 'buy_date': current_date,
                        'code': ts_code, 'action': 'BUY', 'price': buy_price, 'shares': shares_can_buy,
                        'pnl': 0, 'pnl_pct': 0
                    })
            
            # 清空目标，因为只能买一天
            targets_for_tomorrow = []
            
            # -----------------------------------------------------------------
            # 阶段 B: 收盘后。记录净值，滚动训练，预测标的
            # -----------------------------------------------------------------
            nav = self.capital + sum([p['shares'] * closes.get(c, p['buy_price']) for c, p in self.positions.items()])
            self.equity_curve.append({'date': current_date, 'nav': nav})
            
            # B1. 降低训练频率或复用模型
            dt_obj = pd.to_datetime(current_date)
            # 为了加快回测速度，我们改为每月重新训练一次
            current_month = dt_obj.month
            if current_month != getattr(self, 'last_train_month', -1):
                train_end = dt_obj - timedelta(days=1)
                train_mask = (df['trade_date'] < train_end.strftime('%Y%m%d')) & \
                             (df['trade_date'] >= (train_end - timedelta(days=365*2)).strftime('%Y%m%d'))
                sub_df = df[train_mask]
                
                # 训练模型需要确保有标签
                if len(sub_df.dropna(subset=['return_label'])) > 500:
                    model, scaler, selected_features = self.train_model(sub_df, features)
                    self.last_train_month = current_month

            # B2. 预测明早可以买的票
            if model is not None:
                X_day = day_data[selected_features].fillna(0).replace([np.inf, -np.inf], 0)
                if not X_day.empty:
                    X_day_scaled = scaler.transform(X_day)
                    probs = model.predict_proba(X_day_scaled)[:, 1]
                    day_data['prob'] = probs
                    
                    # 选出 Top 标的
                    # 提高 signal_threshold 到 0.60 减少烂交易
                    candidates = day_data[day_data['prob'] > max(self.signal_threshold, 0.60)].sort_values('prob', ascending=False)
                    
                    if len(candidates) > 0:
                        targets_for_tomorrow = candidates.head(self.top_n)['ts_code'].tolist()
                    else:
                        targets_for_tomorrow = []
                    
        # Save the final model and scaler for live signal generation
        if model is not None:
            import joblib
            model_data = {
                'model': model,
                'scaler': scaler,
                'features': selected_features,
                'timing': {'entry': self.entry_timing, 'exit': self.exit_timing}
            }
            joblib.dump(model_data, 'latest_model.joblib')
            print("\nFinal model saved to latest_model.joblib")
        
        self.print_summary(df)

    def print_summary(self, df):
        if not self.equity_curve: return
        
        eq_df = pd.DataFrame(self.equity_curve)
        final_nav = eq_df.iloc[-1]['nav']
        ret = (final_nav - self.initial_capital) / self.initial_capital
        
        eq_df['max_nav'] = eq_df['nav'].cummax()
        eq_df['dd'] = (eq_df['max_nav'] - eq_df['nav']) / eq_df['max_nav']
        max_dd = eq_df['dd'].max()
        
        print("\n" + "="*50)
        print(f"  Strict T+1 Strategy Backtest Summary (Entry={self.entry_timing}, Exit={self.exit_timing})")
        print("="*50)
        print(f"Initial Capital: {self.initial_capital:,.2f}")
        print(f"Final NAV:       {final_nav:,.2f}")
        print(f"Total Return:    {ret*100:.2f}%")
        print(f"Max Drawdown:    {max_dd*100:.2f}%")
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        print(f"Total Buy Trades:{len(buy_trades)}")
        
        if len(self.trades) > 0:
            trades_df = pd.DataFrame(self.trades)
            win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df[trades_df['action'] == 'SELL']) if len(trades_df[trades_df['action'] == 'SELL']) > 0 else 0
            print(f"Win Rate:        {win_rate*100:.2f}%")
            
            # Save trades to csv
            filename = f"trades_{self.entry_timing}_{self.exit_timing}.csv"
            trades_df.to_csv(filename, index=False)
            print(f"Saved trade history to {filename}")
            
        print("="*50)

if __name__ == "__main__":
    # 模式 A: Open 买, Open 卖 (基准 - 恢复策略)
    print("\n[Profit Recovery] 运行模式: T+1 Close 买, T+2 Close 卖 + 大盘过滤 + 硬止损")
    bt = StrictT1Backtest(
        top_n=3, 
        signal_threshold=0.75, 
        slippage=0.001, 
        entry_timing='close', 
        exit_timing='close',
        stop_loss=-0.035
    )
    bt.run(data_start='20210101', test_start='20230101', test_end='20240101')
