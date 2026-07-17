"""
滚动训练回测脚本 V5 - 日频增强版 (Daily Loop)

基于用户需求:
1. 日级别回测 (Daily Loop)
2. 模型每周更新 (Weekly Retraining)
3. 买卖价格滑点 (Slippage)
4. 涨跌停限制 (Price Limit)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, r'C:\Users\liuqi\quant_system_v2')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

from infra_data.storage import DataStorage
from features.enhanced_factors import calculate_all_enhanced_features
from features.labeling import simple_return_labeling
from processing.cleaner import filter_stock_codes
from processing.merger import merge_dataframes

class RollingBacktestV5Daily:
    def __init__(
        self,
        initial_capital: float = 100000,
        top_n: int = 10,
        commission_rate: float = 0.0003,
        stamp_duty: float = 0.0005,
        slippage: float = 0.002,      # 滑点 0.2%
        stop_loss: float = 0.08,
        take_profit: float = 0.25,
        trailing_stop: float = 0.05,
        max_features: int = 80,
    ):
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty
        self.slippage = slippage  # 新增滑点
        
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop = trailing_stop
        self.max_features = max_features
        
        self.storage = DataStorage()
        self.capital = initial_capital
        self.positions = {}
        self.equity_curve = []
        self.trades = []
        self.high_watermarks = {}
        self.market_trend = 'NORMAL'

    def load_data(self, start_date, end_date):
        print("加载数据...")
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
        
        print("计算增强特征...")
        df = calculate_all_enhanced_features(df, money_flow, chip_data)
        
        print("生成标签...")
        # 标签依然是用未来收益，但回测是逐日
        df = simple_return_labeling(df, forward_days=10, threshold=0.03) 
        df = df.sort_values(['ts_code', 'trade_date'])
        return df

    def train_model(self, train_df, features):
        X = train_df[features].fillna(0).replace([np.inf, -np.inf], 0)
        y = (train_df['return_label'] == 1).astype(int)
        
        # 特征选择
        selector = SelectKBest(f_classif, k=min(self.max_features, len(X.columns)))
        selector.fit(X, y)
        selected_idx = selector.get_support(indices=True)
        selected_features = [features[i] for i in selected_idx]
        
        X = X[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = xgb.XGBClassifier(
            n_estimators=100, # 稍微减少以加快周频训练速度
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_scaled, y)
        return model, scaler, selected_features

    def get_features(self, df):
        exclude = ['ts_code', 'trade_date', 'return_label', 'future_return', 
                   'pct_label', 'return_rank', 'tb_label', 'year_month']
        return [c for c in df.columns if c not in exclude and df[c].dtype not in ['object', 'datetime64[ns]']]

    def buy_stock(self, ts_code, price, amount, date):
        # 涨停限制: 如果涨停（涨幅>9.5%），无法买入
        # 注意: 这里的price通常是收盘价。如果收盘涨停，我们假设买不进去。
        # 实际上日频回测通常假设以收盘价或次日开盘价买入。
        # 简单起见，如果当日涨幅 > 9.5%，则禁止买入
        
        # 滑点成本
        actual_price = price * (1 + self.slippage)
        
        if actual_price <= 0: return False
        
        shares = int(amount / actual_price / 100) * 100
        if shares < 100: return False
        
        cost = shares * actual_price
        commission = max(cost * self.commission_rate, 5)
        total_cost = cost + commission
        
        if total_cost > self.capital: return False
        
        self.capital -= total_cost
        self.positions[ts_code] = {
            'shares': shares,
            'buy_price': actual_price,
            'buy_date': date
        }
        self.high_watermarks[ts_code] = actual_price
        
        self.trades.append({
            'date': date,
            'ts_code': ts_code,
            'type': 'BUY',
            'price': actual_price,
            'shares': shares,
            'cost': total_cost,
            'pnl': 0
        })
        return True

    def sell_stock(self, ts_code, price, date, reason):
        # 跌停限制
        # 如果跌幅 < -9.5%，无法卖出
        
        # 滑点成本
        actual_price = price * (1 - self.slippage)
        
        pos = self.positions[ts_code]
        revenue = pos['shares'] * actual_price
        commission = max(revenue * self.commission_rate, 5)
        stamp = revenue * self.stamp_duty
        net_revenue = revenue - commission - stamp
        
        self.capital += net_revenue
        pnl = net_revenue - (pos['shares'] * pos['buy_price'])  # 简单盈亏计算
        
        self.trades.append({
            'date': date,
            'ts_code': ts_code,
            'type': 'SELL',
            'reason': reason,
            'price': actual_price,
            'shares': pos['shares'],
            'revenue': net_revenue,
            'pnl': pnl
        })
        
        del self.positions[ts_code]
        if ts_code in self.high_watermarks:
            del self.high_watermarks[ts_code]

    def run(self, data_start='20180101', train_end='20191231'):
        today = datetime.now().strftime('%Y%m%d')
        print(f"开始日频回测: {data_start} - {today}")
        print(f"滑点: {self.slippage*100}% | 涨停无法买入 | 跌停无法卖出")
        print(f"模型更新频率: 每周")
        
        df = self.load_data(data_start, today)
        features = self.get_features(df)
        
        # 初始训练
        print("初始模型训练...")
        train_end_dt = pd.to_datetime(train_end)
        train_df = df[df['trade_date'] <= train_end_dt]
        train_df = train_df[train_df['return_label'].notna()]
        model, scaler, selected_features = self.train_model(train_df, features)
        
        # 准备回测数据
        backtest_df = df[df['trade_date'] > train_end_dt].copy()
        trade_dates = sorted(backtest_df['trade_date'].unique())
        
        print(f"回测交易日: {len(trade_dates)} 天")
        
        # 缓存数据加速
        df_grouped = dict(tuple(backtest_df.groupby('trade_date')))
        
        model_retrain_needed = False
        last_week = -1
        
        for current_date in tqdm(trade_dates, desc="Daily Loop"):
            day_data = df_grouped.get(current_date)
            if day_data is None or day_data.empty: continue
            
            # --- 1. 每周重新训练 ---
            current_week = current_date.isocalendar().week
            if current_week != last_week:
                # 新的一周，重新训练
                # 为了性能，我们只在周一(或这周的第一个交易日)训练
                train_mask = df['trade_date'] < current_date
                # 限制训练集大小(最近2年)，避免越来越慢
                min_train_date = current_date - timedelta(days=730)
                sub_train = df[train_mask & (df['trade_date'] >= min_train_date)]
                sub_train = sub_train[sub_train['return_label'].notna()]
                
                if len(sub_train) > 1000:
                    model, scaler, selected_features = self.train_model(sub_train, features)
                    # print(f"模型已更新: {current_date.date()}")
                
                last_week = current_week
            
            # --- 2. 每日行情与风控 ---
            prices = dict(zip(day_data['ts_code'], day_data['close']))
            pct_chgs = dict(zip(day_data['ts_code'], day_data['pct_chg']))
            highs = dict(zip(day_data['ts_code'], day_data['high']))
            lows = dict(zip(day_data['ts_code'], day_data['low']))
            
            # 检查持仓的风控 (止损/止盈/移动止损)
            # 必须主要: 跌停无法卖出
            for ts_code in list(self.positions.keys()):
                if ts_code not in prices: continue
                
                price = prices[ts_code]
                if price <= 0.01 or np.isnan(price): continue
                
                buy_price = self.positions[ts_code]['buy_price']
                pct_chg = pct_chgs.get(ts_code, 0)
                
                # 更新最高价
                self.high_watermarks[ts_code] = max(self.high_watermarks.get(ts_code, price), price)
                high_price = self.high_watermarks[ts_code]
                
                # 跌停限制 (-9.5%)
                if pct_chg < -9.5:
                    continue # 无法卖出
                
                # 策略卖出逻辑
                reason = None
                pnl_pct = (price - buy_price) / buy_price
                
                if pnl_pct <= -self.stop_loss:
                    reason = 'STOP_LOSS'
                elif pnl_pct >= self.take_profit:
                    reason = 'TAKE_PROFIT'
                else:
                    drawdown = (high_price - price) / high_price
                    if drawdown >= self.trailing_stop and pnl_pct > 0:
                        reason = 'TRAILING_STOP'
                
                if reason:
                    self.sell_stock(ts_code, price, current_date, reason)
            
            # --- 3. 每日选股 (使用这周的模型) ---
            # 预测
            X_day = day_data[selected_features].fillna(0).replace([np.inf, -np.inf], 0)
            X_day_scaled = scaler.transform(X_day)
            probs = model.predict_proba(X_day_scaled)[:, 1]
            day_data['prob'] = probs
            
            # 资金流分数修正
            if 'mf_signal' in day_data.columns:
                day_data['score'] = day_data['prob'] + 0.1 * day_data['mf_signal'].fillna(0)
            else:
                day_data['score'] = day_data['prob']
            
            candidates = day_data[
                (day_data['prob'] > 0.55)
            ].sort_values('score', ascending=False).head(self.top_n)
            
            target_codes = set(candidates['ts_code'].tolist())
            
            # --- 4. 调仓: 卖出不在榜单的 (每日换股逻辑? 或者是持有逻辑?) ---
            # V5 原逻辑是持有直到卖出信号。这里如果是日频，我们可以采用"持有直到风控或跌出榜单"
            # 既然是日频，如果今天不在TopN，是否卖出？
            # 激进策略: 是。 稳健策略: 否，只看风控。
            # 为了体现"日频回测"的活跃性，我们设定: 如果掉出 Top 20，则卖出。
            
            # 卖出逻辑补充
            for ts_code in list(self.positions.keys()):
                if ts_code not in prices: continue
                
                price = prices[ts_code]
                if price <= 0.01 or np.isnan(price): continue
                
                # 如果已经因为风控卖了，就不管
                # 否则，检查是否还在推荐列表 (放宽到 Top 15 避免频繁换手)
                # 另外，跌停不能卖
                if pct_chgs.get(ts_code, 0) < -9.5: continue
                
                # 如果不在当天的 Top 15，且持有超过 1 天，则换股
                # (T+1 规则: check buy_date)
                pos = self.positions[ts_code]
                if pos['buy_date'] == current_date: continue # T+0 无法卖
                
                # 检查排名
                # 简单做法: 如果不在 candidates (Top 10) 里，就卖？太频繁。
                # 让我们只卖出那些预测概率掉到 0.5 以下的
                row = day_data[day_data['ts_code'] == ts_code]
                if not row.empty:
                    prob = row.iloc[0]['prob']
                    if prob < 0.5:
                        self.sell_stock(ts_code, prices[ts_code], current_date, 'WEAK_SIGNAL')

            # --- 5. 买入逻辑 ---
            # 有空余仓位才买
            available_slots = self.top_n - len(self.positions)
            if available_slots > 0:
                # 计算可用资金
                # 简单均分: 总是试图保持 top_n 只股票
                # 每只股票目标资金 = 总资产 / top_n
                # 当前可用 cash
                
                for _, row in candidates.iterrows():
                    if available_slots <= 0: break
                    ts_code = row['ts_code']
                    
                    if ts_code in self.positions: continue
                    
                    # 涨停限制
                    if row['pct_chg'] > 9.5: continue
                    
                    # 买入
                    current_portfolio_value = self.capital + sum([self.positions[c]['shares'] * prices[c] for c in self.positions if c in prices])
                    target_pos_value = current_portfolio_value / self.top_n
                    
                    if self.capital >= target_pos_value * 0.9: # 资金足够
                        if self.buy_stock(ts_code, row['close'], target_pos_value, current_date):
                            available_slots -= 1

            # 记录净值
            total_val = self.capital
            for c, pos in self.positions.items():
                if c in prices:
                    total_val += pos['shares'] * prices[c]
                else:
                    # 停牌，用买入价或者最后已知价? 用买入价暂代
                    total_val += pos['shares'] * pos['buy_price']
            
            self.equity_curve.append({'date': current_date, 'value': self.get_total_value(prices)})
        
        self.print_results()
        return self.equity_curve, self.trades

    def get_total_value(self, prices):
        val = self.capital
        for c, pos in self.positions.items():
            if c in prices and prices[c] > 0.01:
                val += pos['shares'] * prices[c]
            else:
                # 停牌或数据异常，用买入价暂估 (或者应该用上一日收盘价，但这里简化)
                val += pos['shares'] * pos['buy_price']
        return val

    def print_results(self):
        if not self.equity_curve: return
        
        final_val = self.equity_curve[-1]['value']
        ret = (final_val / self.initial_capital - 1) * 100
        
        # Drawdown
        vals = [x['value'] for x in self.equity_curve]
        peak = vals[0]
        max_dd = 0
        for v in vals:
            peak = max(peak, v)
            dd = (peak - v) / peak * 100
            max_dd = max(max_dd, dd)
            
        print(f"\n{'='*40}")
        print(f"Daily Backtest Results (2020-Now)")
        print(f"{'='*40}")
        print(f"Final Capital: {final_val:,.0f}")
        print(f"Total Return:  {ret:.2f}%")
        print(f"Max Drawdown:  {max_dd:.2f}%")
        print(f"Trades:        {len(self.trades)}")
        print(f"{'='*40}\n")

if __name__ == "__main__":
    backtest = RollingBacktestV5Daily()
    # 扩大回测范围以覆盖更多市场周期
    backtest.run(data_start='20200101', train_end='20201231') 
    
    # 保存
    if backtest.equity_curve:
        pd.DataFrame(backtest.equity_curve).to_csv(r'C:\Users\liuqi\.gemini\antigravity\scratch\equity_curve_daily.csv', index=False)
        pd.DataFrame(backtest.trades).to_csv(r'C:\Users\liuqi\.gemini\antigravity\scratch\trades_daily.csv', index=False)

