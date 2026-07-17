"""
PTrade 仿真回测脚本 V5 (Ptrade Emulation Backtest)

功能:
1. 严格复刻 Ptrade 实盘逻辑 (ptrade_client_v5.py)
2. 包含大盘状态检测 (Bull/Bear) -> 动态调整持仓数量 (Top 10 vs Top 3)
3. 包含严格风控 (止损 -8%, 止盈 +25%, 信号消失卖出)
4. 数据源: 本地 Tushare 数据 (无需连接 Ptrade)
5. 回测周期: 2023-01-01 至今
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
sys.path.insert(0, r'C:\Users\liuqi\quant_system_v2')

from infra_data.storage import DataStorage
from features.enhanced_factors import calculate_all_enhanced_features
from features.labeling import simple_return_labeling
from processing.cleaner import filter_stock_codes
from processing.merger import merge_dataframes
from config.settings import settings

warnings.filterwarnings('ignore')

class PtradeEmulationBacktest:
    def __init__(
        self,
        initial_capital: float = 1000000, # 100万起步
        commission_rate: float = 0.0003,
        stamp_duty: float = 0.0005,
        slippage: float = 0.002,
        stop_loss: float = 0.08,
        take_profit: float = 0.25,
        max_position_pct: float = 0.10, # 单只最大仓位 10%
        trailing_stop: float = 0.10, # Opt5: 回撤止盈 10% (放宽，避免震出强势股)
        min_hold_days: int = 5, # Opt2: 最低持仓 5 天
        signal_threshold: float = 0.60, # Opt3: 信号阈值提高到 0.60
        gap_limit: float = 0.03, # Opt4: 跳空超过 3% 不买入
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty
        self.slippage = slippage
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_position_pct = max_position_pct
        self.trailing_stop = trailing_stop
        self.min_hold_days = min_hold_days
        self.signal_threshold = signal_threshold
        self.gap_limit = gap_limit
        
        self.storage = DataStorage()
        self.capital = initial_capital
        self.positions = {} # {ts_code: {'shares': 100, 'buy_price': 10.0, 'buy_date': '20230101'}}
        self.equity_curve = []
        self.trades = []
        
        # 缓存大盘数据
        self.index_data = None 

    def load_data(self, start_date, end_date):
        print(f"1. 加载个股数据 {start_date} - {end_date}...")
        daily = self.storage.load_daily_data(start_date, end_date)
        other = self.storage.load_daily_basic(start_date, end_date)
        skill = self.storage.load_technical_factors(start_date, end_date)
        money_flow = self.storage.load_money_flow(start_date, end_date)
        chip_data = self.storage.load_chip_data(start_date, end_date)
        
        dfs = [daily]
        if not other.empty: dfs.append(other)
        if not skill.empty: dfs.append(skill)
        
        df = merge_dataframes(dfs)
        df = filter_stock_codes(df, patterns=['^60', '^00']) # 只做主板
        
        print("2. 计算增强特征...")
        df = calculate_all_enhanced_features(df, money_flow, chip_data)
        
        # Opt8: Add 'open_gap' feature (Precise Open Indicator)
        # Using pre_close if available, else calc from close.shift(1)
        if 'pre_close' in df.columns:
            df['open_gap'] = df['open'] / df['pre_close'] - 1
        else:
            df['sort_date'] = df['trade_date']
            df = df.sort_values(['ts_code', 'sort_date'])
            df['pre_close_calc'] = df.groupby('ts_code')['close'].shift(1)
            df['open_gap'] = df['open'] / df['pre_close_calc'] - 1
            del df['sort_date'], df['pre_close_calc']
        
        print("3. 生成训练标签...")
        df = simple_return_labeling(df, forward_days=5, threshold=0.03, price_col='open') # Opt7: 10d -> 5d horizon
        df = df.sort_values(['ts_code', 'trade_date'])
        
        # 加载大盘指数 (用于 Bear/Bull 判断)
        try:
            print("4. 加载大盘指数 (000001.SH)...")
            import tushare as ts
            pro = ts.pro_api(settings.TUSHARE_TOKEN)
            # 获取更早一点的数据以便计算均线
            idx_start = (pd.to_datetime(start_date) - timedelta(days=120)).strftime('%Y%m%d')
            self.index_data = pro.index_daily(ts_code='000001.SH', start_date=idx_start, end_date=end_date)
            self.index_data = self.index_data.sort_values('trade_date').set_index('trade_date')
            self.index_data['MA20'] = self.index_data['close'].rolling(20).mean()
            self.index_data['MA60'] = self.index_data['close'].rolling(60).mean()
        except Exception as e:
            print(f"警告: 无法加载大盘指数: {e}")
            self.index_data = pd.DataFrame()
            
        return df

    def get_market_status(self, date):
        """判断市场状态: BULL or BEAR"""
        if self.index_data is None or date not in self.index_data.index:
            return 'BULL' # 默认做多
        
        row = self.index_data.loc[date]
        if np.isnan(row['MA20']) or np.isnan(row['MA60']):
            return 'BULL'
            
        return 'BEAR' if row['MA20'] < row['MA60'] else 'BULL'

    def train_model(self, train_df, features):
        X = train_df[features].fillna(0).replace([np.inf, -np.inf], 0)
        y = (train_df['return_label'] == 1).astype(int)
        
        # 特征选择
        selector = SelectKBest(f_classif, k=min(80, len(X.columns)))
        selector.fit(X, y)
        selected_idx = selector.get_support(indices=True)
        selected_features = [features[i] for i in selected_idx]
        
        X = X[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist' # 关键优化: 使用直方图算法加速训练 (5-10x fast)
        )
        model.fit(X_scaled, y)
        return model, scaler, selected_features

    def run(self, start_date='20200101', retrain_freq='daily'):
        end_date = datetime.now().strftime('%Y%m%d')
        
        # Determine data loading start date (2 years prior to start_date for initial training)
        start_dt = pd.to_datetime(start_date)
        data_start_dt = start_dt - timedelta(days=730) 
        data_start_load = data_start_dt.strftime('%Y%m%d')
        
        df = self.load_data(data_start_load, end_date)
        
        # 特征列表
        exclude = ['ts_code', 'trade_date', 'return_label', 'future_return', 
                   'pct_label', 'return_rank', 'tb_label', 'year_month']
        features = [c for c in df.columns if c not in exclude and df[c].dtype not in ['object', 'datetime64[ns]']]
        
        # 初始训练 (用 start_date 前2年数据训练)
        print(f">>> 初始模型训练 ({data_start_load} - {start_date})...")
        train_mask = (df['trade_date'] >= data_start_load) & (df['trade_date'] < start_date)
        train_df = df[train_mask]
        train_df = train_df[train_df['return_label'].notna()]
        
        if train_df.empty:
            # Fallback if specific range is empty (e.g. holidays), try to get whatever is before start_date
            train_mask_backup = df['trade_date'] < start_date
            train_df = df[train_mask_backup]
            if train_df.empty:
                raise ValueError("没有足够的训练数据")
            
        model, scaler, selected_features = self.train_model(train_df, features)
        
        # 开始回测循环
        print(f">>> 开始 Ptrade 仿真回测: {start_date} - {end_date} (Retrain: {retrain_freq})")
        backtest_df = df[df['trade_date'] >= start_date].copy()
        trade_dates = sorted(backtest_df['trade_date'].unique())
        
        # 按日期分组加速
        df_grouped = dict(tuple(backtest_df.groupby('trade_date')))
        
        last_week = -1
        
        # 初始 Target List (第一天无信号，不操作)
        next_day_targets = [] 
        
        for current_date in tqdm(trade_dates, desc="Backtest Loop"):
            day_data = df_grouped.get(current_date)
            if day_data is None or day_data.empty: continue
            
            # --- 0. 准备价格数据 (用于成交) ---
            # 今天的 Open/Low/High/Close
            # 交易执行: 使用 Open 价
            # 风控监测: 使用 Low/High
            # 信号生成: 使用 Close (收盘后)
            
            day_data_dict = day_data.set_index('ts_code').to_dict('index')
            opens        = dict(zip(day_data['ts_code'], day_data['open']))
            
            # --- 1. 执行交易 (基于昨日的 next_day_targets) ---
            # 卖出逻辑: 如果持仓股票不在今日目标中 -> 开盘卖出
            # 注意: T+1 规则
            
            executed_sells = []
            
            # 先更新 max_price (用于风控)
            for ts_code in list(self.positions.keys()):
                if ts_code in day_data_dict:
                    current_high = day_data_dict[ts_code]['high']
                    self.positions[ts_code]['max_price'] = max(self.positions[ts_code]['max_price'], current_high)
            
            # 1.1 卖出 (Signal Lost) at OPEN
            for ts_code in list(self.positions.keys()):
                # 停牌跳过
                if ts_code not in opens: continue
                # 跌停跳过 (使用 Open 价判断? 假设 Open 不跌停即可卖? 严格点用 Low)
                # 假设只要开盘没有一字跌停就可以卖
                # 这里简化: 如果 Open < YesterdayClose * 0.9, 跳过? 
                # 我们用 day_data_dict check
                pct_chg = day_data_dict[ts_code]['pct_chg']
                if pct_chg < -9.5: continue # 跌停无法卖出
                
                pos = self.positions[ts_code]
                
                # T+1 检查
                if pos['buy_date'] == current_date: continue
                
                # Opt2: 最低持仓期检查
                hold_days = (pd.to_datetime(current_date) - pd.to_datetime(pos['buy_date'])).days
                if hold_days < self.min_hold_days:
                    continue  # 持仓不足 min_hold_days 天，不因信号消失而卖出
                
                # 如果不在目标池中 -> 卖出 (Signal Lost)
                if ts_code not in next_day_targets:
                     self._sell(ts_code, opens[ts_code], current_date, "SIGNAL_LOST")
                     executed_sells.append(ts_code)
            
            # 1.2 买入 (Signal Gain) at OPEN
            # 资金管理: 10% per stock
            market_value = 0
            for c, p in self.positions.items():
                price_lookup = opens.get(c, p['buy_price']) # Use Open for valuation
                market_value += p['shares'] * price_lookup
                
            total_assets = self.capital + market_value
            target_pos_value = total_assets * self.max_position_pct
            
            for ts_code in next_day_targets:
                if ts_code in self.positions: continue
                if ts_code in executed_sells: continue
                
                if ts_code not in opens: continue
                price = opens[ts_code] # Buy at Open
                
                # 涨停无法买入
                if day_data_dict[ts_code]['pct_chg'] > 9.5: continue
                
                # Opt4: 跳空过滤 - 高开超过 gap_limit 不追
                if 'pre_close' in day_data_dict[ts_code]:
                    pre_close = day_data_dict[ts_code]['pre_close']
                    if pre_close > 0:
                        open_gap = (price - pre_close) / pre_close
                        if open_gap > self.gap_limit:
                            continue
                
                # Check Cash
                if self.capital < 5000: break # Cash limit
                
                # Buy
                cost_est = price * (1 + self.slippage)
                available_cash = min(self.capital, target_pos_value)
                if available_cash < 5000: continue
                
                self._buy(ts_code, price, available_cash, current_date)

            # --- 2. 盘中风控 (Stop Loss / Take Profit / Trailing Stop) ---
            # 检查 Low/High
            for ts_code in list(self.positions.keys()):
                if ts_code not in day_data_dict: continue
                if ts_code in executed_sells: continue # Already sold
                
                pos = self.positions[ts_code]
                if pos['buy_date'] == current_date: continue # T+1 Rule: Cannot sell today's buy
                
                row = day_data_dict[ts_code]
                low = row['low']
                high = row['high']
                
                # 1. Stop Loss -8%
                if low <= pos['buy_price'] * (1 - self.stop_loss):
                    sell_p = max(row['open'], pos['buy_price'] * (1 - self.stop_loss)) # gap protection
                    self._sell(ts_code, sell_p, current_date, "STOP_LOSS")
                    executed_sells.append(ts_code)
                    continue
                
                # 2. Take Profit +25%
                if high >= pos['buy_price'] * (1 + self.take_profit):
                    sell_p = max(row['open'], pos['buy_price'] * (1 + self.take_profit))
                    self._sell(ts_code, sell_p, current_date, "TAKE_PROFIT")
                    executed_sells.append(ts_code)
                    continue
                    
                # 3. Trailing Stop
                stop_price = pos['max_price'] * (1 - self.trailing_stop)
                if low <= stop_price:
                    sell_p = max(row['open'], stop_price)
                    self._sell(ts_code, sell_p, current_date, "TRAILING_STOP")
                    executed_sells.append(ts_code)
                    continue
            
            # --- 3. 盘后: 生成明日信号 (Signal Generation for Next Day) ---
            # 模型重训练 (Retrain Logic)
            dt_obj = pd.to_datetime(current_date)
            should_retrain = False
            
            if retrain_freq == 'daily':
                should_retrain = True
            elif retrain_freq == 'weekly':
                current_week = dt_obj.isocalendar().week
                if current_week != last_week:
                    should_retrain = True
                    last_week = current_week
            
            if should_retrain:
                train_end_dt = dt_obj - timedelta(days=1)
                train_start_dt = train_end_dt - timedelta(days=730)
                sub_train = df[(df['trade_date'] >= train_start_dt.strftime('%Y%m%d')) & 
                               (df['trade_date'] <= train_end_dt.strftime('%Y%m%d'))]
                sub_train = sub_train[sub_train['return_label'].notna()]
                if len(sub_train) > 1000:
                    model, scaler, selected_features = self.train_model(sub_train, features)
            
            # 预测 (Prediction)
            market_status = self.get_market_status(current_date)
            target_count = 3 if market_status == 'BEAR' else 10
            
            X_day = day_data[selected_features].fillna(0).replace([np.inf, -np.inf], 0)
            X_day_scaled = scaler.transform(X_day)
            probs = model.predict_proba(X_day_scaled)[:, 1]
            day_data['prob'] = probs
            
            if 'mf_signal' in day_data.columns:
                day_data['score'] = day_data['prob'] + 0.1 * day_data['mf_signal'].fillna(0)
            else:
                day_data['score'] = day_data['prob']
                
            candidates = day_data[day_data['prob'] > self.signal_threshold].sort_values('score', ascending=False)
            
            # Opt6: 周度调仓 - 只在周一更新目标池，其余时间保持不变
            is_monday = dt_obj.weekday() == 0
            if is_monday or len(next_day_targets) == 0:
                next_day_targets = candidates.head(target_count)['ts_code'].tolist()
            
            # --- 4. 记录净值 (End of Day) ---
            prices = dict(zip(day_data['ts_code'], day_data['close'])) # Record using Close
            self._record_equity(current_date, prices)

        # 结束
        self.print_summary()

    def _buy(self, ts_code, price, money, date):
        actual_price = price * (1 + self.slippage)
        shares = int(money / actual_price / 100) * 100
        if shares < 100: return
        
        cost = shares * actual_price
        commission = max(5, cost * self.commission_rate)
        total_cost = cost + commission
        
        if self.capital >= total_cost:
            self.capital -= total_cost
            self.positions[ts_code] = {
                'shares': shares,
                'buy_price': actual_price,
                'buy_date': date,
                'max_price': actual_price # Initialize max_price for trailing stop
            }
            self.trades.append({
                'date': date,
                'code': ts_code,
                'action': 'BUY',
                'price': actual_price,
                'shares': shares,
                'info': 'SIGNAL'
            })

    def _sell(self, ts_code, price, date, reason):
        actual_price = price * (1 - self.slippage)
        pos = self.positions[ts_code]
        revenue = pos['shares'] * actual_price
        commission = max(5, revenue * self.commission_rate)
        stamp = revenue * self.stamp_duty
        net_income = revenue - commission - stamp
        
        self.capital += net_income
        pnl = net_income - (pos['shares'] * pos['buy_price'])
        pnl_pct = pnl / (pos['shares'] * pos['buy_price'])
        
        del self.positions[ts_code]
        
        self.trades.append({
            'date': date,
            'code': ts_code,
            'action': 'SELL',
            'price': actual_price,
            'shares': pos['shares'],
            'info': f"{reason} ({pnl_pct:.1%})"
        })

    def _record_equity(self, date, prices):
        mv = 0
        for c, p in self.positions.items():
            if c in prices:
                mv += p['shares'] * prices[c]
            else:
                mv += p['shares'] * p['buy_price']
        
        total = self.capital + mv
        status = self.get_market_status(date)
        
        self.equity_curve.append({
            'date': date,
            'assets': total,
            'market': status,
            'positions': len(self.positions)
        })

    def print_summary(self):
        if not self.equity_curve: return
        
        df = pd.DataFrame(self.equity_curve)
        final_assets = df.iloc[-1]['assets']
        ret = (final_assets - self.initial_capital) / self.initial_capital
        
        # Max Drawdown
        df['max_assets'] = df['assets'].cummax()
        df['dd'] = (df['max_assets'] - df['assets']) / df['max_assets']
        max_dd = df['dd'].max()
        
        print("\n" + "="*50)
        print("  PTrade Logic Backtest Summary (2023-Now)")
        print("="*50)
        print(f"Initial Capital: {self.initial_capital:,.2f}")
        print(f"Final Assets:    {final_assets:,.2f}")
        print(f"Total Return:    {ret*100:.2f}%")
        print(f"Max Drawdown:    {max_dd*100:.2f}%")
        print(f"Total Trades:    {len(self.trades)}")
        print("="*50)
        
        # 保存结果
        out_dir = r"C:\Users\liuqi\quant_system_v2\backtesting"
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(os.path.join(out_dir, "ptrade_equity.csv"), index=False)
        pd.DataFrame(self.trades).to_csv(os.path.join(out_dir, "ptrade_trades.csv"), index=False)
        print(f"Results saved to: {out_dir}")

if __name__ == "__main__":
    # T+1 优化版: 周度调仓 + 日频风控 + Open-to-Open 标签
    bt = PtradeEmulationBacktest(
        initial_capital=1000000,
        trailing_stop=0.10,    # Opt5: 放宽回撤止盈
        min_hold_days=5,       # Opt2: 最低持仓 5 天
        signal_threshold=0.60, # Opt3: 信号阈值提高
        gap_limit=0.03,        # Opt4: 跳空 >3% 不追
    )
    bt.run(start_date='20200101', retrain_freq='weekly')
