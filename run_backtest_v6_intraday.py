"""
PTrade T+0 (Intraday Close) 仿真回测脚本 V6
Strategy: 尾盘热门股策略 (Hot Stocks Intraday)

Core Logic:
1. Universe: 每日成交额 (Amount) 前 100 名的活跃股 (Hot Money)
2. Execution: 尾盘 (Close) 买入，次日 (Close) 或之后卖出
3. Label: Close-to-Close (T+1) 收益率
4. Slippage: 0.5% (千分之五) —— 模拟 14:50 进场到 15:00 收盘的潜在波动成本

Target: 验证 "14:50 进场" 是否比 "T+1 Open 进场" 更具优势
"""

import sys
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

# Add project root to path
sys.path.insert(0, r'C:\Users\liuqi\quant_system_v2')

from infra_data.storage import DataStorage
from features.enhanced_factors import calculate_all_enhanced_features
from processing.merger import merge_dataframes
from config.settings import settings

warnings.filterwarnings('ignore')

def close_return_labeling(df, forward_days=1, threshold=0.02):
    """
    Standard Close-to-Close Labeling for Intraday Strategy
    If buying at Close T, we care about Close T+N vs Close T
    """
    group = df.groupby('ts_code')
    
    # Calculate forward return using Close price
    # (Close_{t+n} - Close_t) / Close_t
    df['future_return'] = group['close'].pct_change(forward_days).shift(-forward_days)
    
    # Label: 1 if return > threshold, else 0
    df['return_label'] = (df['future_return'] > threshold).astype(int)
    return df

class IntradayCloseBacktest:
    def __init__(
        self,
        initial_capital=1000000,
        slippage=0.005,      # 0.5% (High slippage for approximation)
        commission=0.0003,
        stamp_duty=0.0005,
        hold_days=1,         # T+1 selling (or longer)
        top_k_amount=100,    # Universe: Top 100 Turnover
        n_features=20        # Only Top 20 features (Speed + Robustness)
    ):
        self.capital = initial_capital
        self.positions = {}
        self.slippage = slippage
        self.commission = commission
        self.stamp_duty = stamp_duty
        self.hold_days = hold_days
        self.top_k_amount = top_k_amount
        self.n_features = n_features
        
        self.storage = DataStorage()
        self.equity_curve = []
        self.trades = []

    def load_data(self, start_date='20230101', end_date='20260218'):
        print(f"Loading data {start_date} - {end_date}...")
        daily = self.storage.load_daily_data(start_date, end_date)
        
        # Filter: Only keep stocks that were in Top 100 Amount roughly
        # To avoid loading too much, we just load basic now and filter daily in loop?
        # No, better pre-filter or load all and filter in memory.
        # Given memory, let's load all and filter.
        
        # Merge other data for features
        chip = self.storage.load_chip_data(start_date, end_date)
        money = self.storage.load_money_flow(start_date, end_date)
        vix = self.storage.load_vix_data(start_date, end_date)
        margin = self.storage.load_margin_data(start_date, end_date)
        
        df = daily.copy()
        
        # Merge VIX data (market-level feature)
        if not vix.empty:
            print("Merging VIX data...")
            vix = vix.rename(columns={'close': 'vix_close', 'open': 'vix_open', 'high': 'vix_high', 'low': 'vix_low', 'pct_chg': 'vix_pct_chg'})
            df = pd.merge(df, vix[['trade_date', 'vix_close', 'vix_pct_chg']], on='trade_date', how='left')
        
        # Merge margin data
        if not margin.empty:
            print("Merging margin data...")
            margin = margin.rename(columns={'rzye': 'margin_rzye', 'rzmre': 'margin_rzmre', 'rqye': 'margin_rqye', 'rzrqye': 'margin_rzrqye'})
            df = pd.merge(df, margin[['trade_date', 'margin_rzye', 'margin_rzmre', 'margin_rqye', 'margin_rzrqye']], on='trade_date', how='left')
        
        # Calculate Features
        print("Calculating Top Features...")
        df = calculate_all_enhanced_features(df, money, chip)
        
        # Labeling (Close-to-Close)
        print("Generating Labels (Close-to-Close)...")
        df = close_return_labeling(df, forward_days=self.hold_days, threshold=0.02)
        
        return df.sort_values(['trade_date', 'ts_code'])

    def train_model(self, train_df, features):
        X = train_df[features].fillna(0)
        y = train_df['return_label']
        
        # Select Top N Features
        if len(features) > self.n_features:
            selector = SelectKBest(f_classif, k=self.n_features)
            selector.fit(X, y)
            cols = list(np.array(features)[selector.get_support()])
            X = X[cols]
        else:
            cols = features
            
        model = xgb.XGBClassifier(
            n_estimators=50,      # Faster
            max_depth=3,          # Simpler
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        model.fit(X, y)
        return model, cols

    def run(self):
        # 1. Prepare Data
        # Use 2023 for training, 2024-Now for testing
        full_df = self.load_data('20220101', '20260218')
        
        # Feature columns (exclude meta)
        exclude = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 
                  'change', 'pct_chg', 'vol', 'amount', 'return_label', 'future_return']
        all_features = [c for c in full_df.columns if c not in exclude]
        
        # Train/Test Split
        split_date = '20240101'
        train_df = full_df[full_df['trade_date'] < split_date].dropna(subset=['return_label'])
        test_df = full_df[full_df['trade_date'] >= split_date].copy()
        
        print(f"Training Model ({len(train_df)} rows)...")
        model, selected_features = self.train_model(train_df, all_features)
        print(f"Top Features: {selected_features}")
        
        # Backtest Loop
        dates = sorted(test_df['trade_date'].unique())
        
        for date in tqdm(dates, desc="Intraday Backtest"):
            day_data = test_df[test_df['trade_date'] == date]
            
            # --- 1. Sell Existing (T+N Rule) at CLOSE ---
            # We assume we sell at Close today for simplicity, 
            # effectively fulfilling "Close-to-Close" return.
            
            pnl_daily = 0
            open_positions = list(self.positions.keys())
            
            for code in open_positions:
                pos = self.positions[code]
                if code in day_data['ts_code'].values:
                    row = day_data[day_data['ts_code'] == code].iloc[0]
                    curr_price = row['close']
                    
                    # Logic: Hold for `hold_days`. If reached, sell.
                    # Or Stop Loss / Take Profit? Keep it simple: Time-based exit for alpha validation.
                    # If we predicted T+1 return, we sell at T+1 Close.
                    
                    # Check holding period
                    held_days = (pd.to_datetime(date) - pd.to_datetime(pos['date'])).days
                    if held_days >= self.hold_days:
                        # SELL
                        sell_val = pos['shares'] * curr_price
                        cost = sell_val * (self.commission + self.stamp_duty + self.slippage) # Exit Slippage
                        net_cash = sell_val - cost
                        
                        self.capital += net_cash
                        raw_ret = (curr_price - pos['price']) / pos['price']
                        
                        self.trades.append({
                            'code': code,
                            'buy_date': pos['date'],
                            'sell_date': date,
                            'buy_price': pos['price'],
                            'sell_price': curr_price,
                            'return': raw_ret,
                            'profit': net_cash - (pos['shares'] * pos['price'])
                        })
                        del self.positions[code]
            
            # --- 2. Buy New (At Close,模拟 14:50) ---
            # Filter: Top 100 Turnover (Hot Stocks)
            # "Amount" is technically whole day, but at 14:50 rank is stable.
            
            top_100 = day_data.sort_values('amount', ascending=False).head(100)
            
            # Predict
            X_curr = top_100[selected_features].fillna(0)
            probs = model.predict_proba(X_curr)[:, 1]
            top_100['score'] = probs
            
            # Select Top 3 with Score > 0.55
            candidates = top_100[top_100['score'] > 0.55].sort_values('score', ascending=False).head(3)
            
            # Exec Buy
            # Position Sizing: 1/10 of current capital (or max 3 stocks total)
            target_pos_count = 5
            max_per_pos = self.capital / max(1, (target_pos_count - len(self.positions)))
            
            for idx, row in candidates.iterrows():
                code = row['ts_code']
                if code in self.positions: continue
                
                if self.capital < 10000: break # No cash
                
                buy_price = row['close'] # Buying at Close
                cost_basis = buy_price * (1 + self.slippage + self.commission) # Entry Slippage
                
                buy_amount = min(max_per_pos, self.capital)
                shares = int(buy_amount / cost_basis / 100) * 100
                
                if shares >= 100:
                    cost = shares * cost_basis
                    self.capital -= cost
                    self.positions[code] = {
                        'date': date,
                        'price': buy_price,
                        'shares': shares
                    }
            
            # Record Equity usually using Close prices
            # (Positions are marked to market at Close)
            pos_val = 0
            for code, pos in self.positions.items():
                if code in day_data['ts_code'].values:
                    # Update close price for valuation
                    price = day_data[day_data['ts_code'] == code].iloc[0]['close']
                else:
                    price = pos['price'] # Fallback
                pos_val += pos['shares'] * price
                
            total_equity = self.capital + pos_val
            self.equity_curve.append({'date': date, 'equity': total_equity})

        # Summary
        final_eq = self.equity_curve[-1]['equity'] if self.equity_curve else self.capital
        ret = (final_eq - 1000000) / 1000000 * 100
        print(f"\n=== V6 Intraday Result (2024-Now) ===")
        print(f"Final Assets: {final_eq:,.2f}")
        print(f"Total Return: {ret:.2f}%")
        print(f"Trades: {len(self.trades)}")
        
        # Calc Drawdown
        eq_df = pd.DataFrame(self.equity_curve)
        eq_df['max'] = eq_df['equity'].cummax()
        eq_df['dd'] = (eq_df['equity'] - eq_df['max']) / eq_df['max']
        mdd = eq_df['dd'].min() * 100
        print(f"Max Drawdown: {mdd:.2f}%")

if __name__ == "__main__":
    bt = IntradayCloseBacktest()
    bt.run()
