"""
月度滚动回测框架 (Walk-Forward Analysis) - V2
支持保存模型和预测数据，便于阈值调整
"""
import pandas as pd
import numpy as np
import os
import sys
import pickle
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

# A股交易规则配置
COST_RATE = 0.003       # 交易费用 0.3% 双边
SLIPPAGE = 0.002        # 滑点 0.2%
LIMIT_THRESHOLD = 0.5   # 涨跌停阈值偏移 (10% -> 9.5%)


@dataclass
class WalkForwardResult:
    """月度滚动回测结果"""
    period: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_samples: int
    test_samples: int
    train_positive_ratio: float
    test_return: float
    test_sharpe: float
    test_max_dd: float
    n_trades: int
    win_rate: float
    avg_return: float
    skipped_limit_up: int
    skipped_limit_down: int
    top_features: List[str]


@dataclass
class PeriodPrediction:
    """每月的预测数据"""
    period: str
    dates: List[str]
    predictions: pd.DataFrame  # 包含 ts_code, date, prob, features
    model_path: str
    scaler_path: str
    feature_cols: List[str]


def is_main_board(ts_code: str) -> bool:
    """检查是否为主板/中小板"""
    return (ts_code.startswith('60') or 
            ts_code.startswith('00') or 
            ts_code.startswith('002') or 
            ts_code.startswith('003'))


def get_limit_pct(ts_code: str) -> float:
    """获取涨跌停幅度"""
    if ts_code.startswith('688') or ts_code.startswith('689'):
        return 20.0
    elif ts_code.startswith('30') or ts_code.startswith('301'):
        return 20.0
    elif ts_code.startswith('8') or ts_code.startswith('4'):
        return 30.0
    else:
        return 10.0


def calculate_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算简单特征（不依赖历史数据）- 增强版
    """
    # 价格位置特征
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # 涨跌幅特征
    df['price_change'] = (df['close'] - df['pre_close']) / df['pre_close']
    df['high_change'] = (df['high'] - df['pre_close']) / df['pre_close']
    df['low_change'] = (df['low'] - df['pre_close']) / df['pre_close']
    
    # 振幅
    df['amplitude'] = (df['high'] - df['low']) / df['pre_close']
    
    # 成交量特征
    df['vol_amount'] = df['close'] * df['vol']
    df['vol_ratio'] = df['vol'] / (df['vol'].mean() + 1e-8)
    
    # 开盘位置
    df['open_position'] = (df['open'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # 实体大小
    df['body_size'] = abs(df['close'] - df['open']) / df['pre_close']
    df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['pre_close']
    df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['pre_close']
    
    # 是否阳线
    df['is_yang'] = (df['close'] > df['open']).astype(int)
    
    # 新增特征：价格动量
    df['close_to_high'] = (df['close'] - df['high']) / df['pre_close']
    df['close_to_low'] = (df['close'] - df['low']) / df['pre_close']
    
    # 新增特征：开盘跳空
    df['gap'] = (df['open'] - df['pre_close']) / df['pre_close']
    
    # 新增特征：日内趋势强度
    df['intraday_trend'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    return df


class WalkForwardBacktestV2:
    """月度滚动回测类 - V2（支持保存模型和预测数据）"""
    
    def __init__(self,
                 data_dir: str = r'D:\iquant_data\data_v2',
                 output_dir: str = None,
                 model_dir: str = None):
        self.data_dir = data_dir
        self.price_dir = os.path.join(data_dir, 'data_day1')
        
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        else:
            self.output_dir = output_dir
        
        if model_dir is None:
            self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        else:
            self.model_dir = model_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 获取所有交易日
        self.all_dates = sorted([f.replace('.parquet', '') 
                                for f in os.listdir(self.price_dir) 
                                if f.endswith('.parquet')])
        
        print(f"数据目录: {self.price_dir}")
        print(f"总交易日数: {len(self.all_dates)}")
        print(f"日期范围: {self.all_dates[0]} 至 {self.all_dates[-1]}")
    
    def load_and_prepare_data(self, dates: List[str], label_threshold: float = 0.02) -> Tuple[pd.DataFrame, List[str]]:
        """
        加载并准备数据
        """
        all_data = []
        
        for i in range(len(dates) - 2):
            d_curr = dates[i]
            d_t1 = dates[i + 1]
            d_t2 = dates[i + 2]
            
            # 加载数据
            p_t = os.path.join(self.price_dir, f"{d_curr}.parquet")
            p_t1 = os.path.join(self.price_dir, f"{d_t1}.parquet")
            p_t2 = os.path.join(self.price_dir, f"{d_t2}.parquet")
            
            if not os.path.exists(p_t) or not os.path.exists(p_t1) or not os.path.exists(p_t2):
                continue
            
            try:
                df_t = pd.read_parquet(p_t)
                df_t1 = pd.read_parquet(p_t1)
                df_t2 = pd.read_parquet(p_t2)
            except Exception as e:
                continue
            
            # 过滤主板
            df_t = df_t[df_t['ts_code'].apply(is_main_board)]
            
            if len(df_t) == 0:
                continue
            
            # 计算简单特征
            df_t = calculate_simple_features(df_t)
            
            # 合并T+1价格
            df_t = df_t.merge(
                df_t1[['ts_code', 'open', 'pre_close']].rename(
                    columns={'open': 't1_open', 'pre_close': 't1_pre_close'}
                ),
                on='ts_code',
                how='left'
            )
            
            # 合并T+2价格
            df_t = df_t.merge(
                df_t2[['ts_code', 'close', 'low', 'open']].rename(
                    columns={'close': 't2_close', 'low': 't2_low', 'open': 't2_open'}
                ),
                on='ts_code',
                how='left'
            )
            
            # 过滤缺失值
            df_t = df_t.dropna(subset=['t1_open', 't2_close'])
            
            if len(df_t) == 0:
                continue
            
            # 计算标签
            df_t['label_ret'] = df_t['t2_close'] / df_t['t1_open'] - 1
            df_t['label'] = (df_t['label_ret'] > label_threshold).astype(int)
            
            all_data.append(df_t)
        
        if not all_data:
            return pd.DataFrame(), []
        
        result = pd.concat(all_data, ignore_index=True)
        
        # 选择特征列
        exclude_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 
                       'vol', 'amount', 'pre_close', 'label_ret', 'label',
                       't1_open', 't1_pre_close', 't2_close', 't2_low', 't2_open']
        feature_cols = [c for c in result.columns if c not in exclude_cols]
        
        return result, feature_cols
    
    def train_model(self, train_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[xgb.XGBClassifier, StandardScaler]:
        """训练模型"""
        X = train_df[feature_cols].fillna(0)
        y = train_df['label']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
            eval_metric='auc',
            n_jobs=-1,
            tree_method='hist',
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5
        )
        
        model.fit(X_scaled, y)
        
        return model, scaler
    
    def save_model(self, model, scaler, feature_cols, period_name):
        """保存模型和scaler"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"model_{period_name}_{timestamp}.pkl"
        scaler_filename = f"scaler_{period_name}_{timestamp}.pkl"
        feature_filename = f"features_{period_name}_{timestamp}.json"
        
        model_path = os.path.join(self.model_dir, model_filename)
        scaler_path = os.path.join(self.model_dir, scaler_filename)
        feature_path = os.path.join(self.model_dir, feature_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        with open(feature_path, 'w') as f:
            json.dump(feature_cols, f)
        
        return model_path, scaler_path, feature_path
    
    def load_model(self, model_path, scaler_path, feature_path):
        """加载模型和scaler"""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(feature_path, 'r') as f:
            feature_cols = json.load(f)
        
        return model, scaler, feature_cols
    
    def generate_predictions(self, model, scaler, feature_cols, test_dates):
        """生成预测数据（保存所有股票的预测概率）"""
        all_predictions = []
        
        for i in range(len(test_dates) - 2):
            d_curr = test_dates[i]
            
            # 加载T日数据
            p_t = os.path.join(self.price_dir, f"{d_curr}.parquet")
            if not os.path.exists(p_t):
                continue
            
            try:
                df_t = pd.read_parquet(p_t)
            except:
                continue
            
            # 过滤主板
            df_t = df_t[df_t['ts_code'].apply(is_main_board)]
            
            if len(df_t) == 0:
                continue
            
            # 计算特征
            df_t = calculate_simple_features(df_t)
            
            # 预测
            X = df_t[feature_cols].fillna(0)
            X_scaled = scaler.transform(X)
            df_t['prob'] = model.predict_proba(X_scaled)[:, 1]
            
            # 保存预测结果
            pred_df = df_t[['ts_code', 'trade_date', 'prob'] + feature_cols].copy()
            all_predictions.append(pred_df)
        
        if not all_predictions:
            return pd.DataFrame()
        
        return pd.concat(all_predictions, ignore_index=True)
    
    def backtest_with_predictions(self, predictions_df, test_dates, min_prob=0.6):
        """使用已保存的预测数据进行回测"""
        trades = []
        skipped_limit_up = 0
        skipped_limit_down = 0
        
        initial_capital = 100000.0
        capital = initial_capital
        daily_nav = []
        
        for i in range(len(test_dates) - 2):
            d_curr = test_dates[i]
            d_t1 = test_dates[i + 1]
            d_t2 = test_dates[i + 2]
            
            # 获取当日预测（确保日期类型一致）
            d_curr_str = str(d_curr)
            day_pred = predictions_df[predictions_df['trade_date'].astype(str) == d_curr_str]
            
            if len(day_pred) == 0:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            # 选择概率最高的股票
            best_idx = day_pred['prob'].idxmax()
            best_prob = day_pred.loc[best_idx, 'prob']
            
            if best_prob < min_prob:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            ts_code = day_pred.loc[best_idx, 'ts_code']
            
            # 加载T+1和T+2价格
            p_t1 = os.path.join(self.price_dir, f"{d_t1}.parquet")
            p_t2 = os.path.join(self.price_dir, f"{d_t2}.parquet")
            
            if not os.path.exists(p_t1) or not os.path.exists(p_t2):
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            try:
                df_t1 = pd.read_parquet(p_t1)
                df_t2 = pd.read_parquet(p_t2)
            except:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            # 获取T+1数据
            t1_data = df_t1[df_t1['ts_code'] == ts_code]
            if t1_data.empty:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            t1_open = float(t1_data.iloc[0]['open'])
            t1_pre = float(t1_data.iloc[0]['pre_close'])
            
            limit_pct = get_limit_pct(ts_code)
            
            # 涨停检查
            t1_open_chg = (t1_open - t1_pre) / t1_pre * 100
            if t1_open_chg >= (limit_pct - LIMIT_THRESHOLD):
                skipped_limit_up += 1
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            # 获取T+2数据
            t2_data = df_t2[df_t2['ts_code'] == ts_code]
            if t2_data.empty:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            t2_close = float(t2_data.iloc[0]['close'])
            t2_low = float(t2_data.iloc[0]['low'])
            t2_open = float(t2_data.iloc[0]['open'])
            
            # 跌停检查
            t2_low_chg = (t2_low - t1_open) / t1_open * 100
            if t2_low_chg <= -(limit_pct - LIMIT_THRESHOLD):
                sell_price = min(t2_open, t1_open * (1 - limit_pct/100))
                skipped_limit_down += 1
            else:
                sell_price = t2_close
            
            # 应用滑点和费用
            buy_price = t1_open * (1 + SLIPPAGE)
            sell_price = sell_price * (1 - SLIPPAGE)
            ret = sell_price / buy_price - 1 - COST_RATE
            
            capital *= (1 + ret)
            
            trades.append({
                'date_t': d_curr,
                'date_t1': d_t1,
                'date_t2': d_t2,
                'ts_code': ts_code,
                'prob': best_prob,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'return': ret
            })
            
            daily_nav.append({'date': d_t2, 'nav': capital})
        
        if len(trades) == 0:
            return {
                'return': 0, 'sharpe': 0, 'max_dd': 0, 'n_trades': 0,
                'win_rate': 0, 'avg_return': 0,
                'skipped_limit_up': skipped_limit_up,
                'skipped_limit_down': skipped_limit_down,
                'daily_nav': daily_nav, 'trades': pd.DataFrame()
            }
        
        trades_df = pd.DataFrame(trades)
        total_ret = capital / initial_capital - 1
        
        nav_df = pd.DataFrame(daily_nav)
        if len(nav_df) > 1:
            nav_df['ret'] = nav_df['nav'].pct_change()
            vol = nav_df['ret'].std() * np.sqrt(252)
            sharpe = (total_ret / (len(test_dates) / 252)) / vol if vol > 0 else 0
        else:
            sharpe = 0
        
        if len(nav_df) > 0:
            nav_df['cummax'] = nav_df['nav'].cummax()
            nav_df['dd'] = (nav_df['nav'] - nav_df['cummax']) / nav_df['cummax']
            max_dd = nav_df['dd'].min()
        else:
            max_dd = 0
        
        win_rate = (trades_df['return'] > 0).mean()
        avg_ret = trades_df['return'].mean()
        
        return {
            'return': total_ret, 'sharpe': sharpe, 'max_dd': max_dd,
            'n_trades': len(trades), 'win_rate': win_rate, 'avg_return': avg_ret,
            'skipped_limit_up': skipped_limit_up,
            'skipped_limit_down': skipped_limit_down,
            'daily_nav': daily_nav, 'trades': trades_df
        }
    
    def run_walk_forward(self, start_date='20230101', end_date='20260331',
                        train_months=24, test_months=1, min_prob=0.6,
                        save_predictions=True):
        """
        运行月度滚动回测
        
        Parameters:
        -----------
        save_predictions : bool
            是否保存预测数据，用于后续阈值调整
        """
        print("=" * 80)
        print("月度滚动回测 (Walk-Forward Analysis) - V2")
        print("=" * 80)
        print(f"回测期: {start_date} 至 {end_date}")
        print(f"训练期: {train_months} 个月")
        print(f"测试期: {test_months} 个月")
        print(f"买入阈值: {min_prob}")
        print("=" * 80)
        
        # 过滤日期
        dates = [d for d in self.all_dates if start_date <= d <= end_date]
        
        if len(dates) < train_months * 21 + test_months * 21:
            print("数据不足！")
            return []
        
        # 生成月份列表
        months = []
        current_month = dates[0][:6]
        month_dates = []
        
        for d in dates:
            if d[:6] == current_month:
                month_dates.append(d)
            else:
                months.append((current_month, month_dates))
                current_month = d[:6]
                month_dates = [d]
        
        if month_dates:
            months.append((current_month, month_dates))
        
        print(f"总月份数: {len(months)}")
        
        results = []
        all_equity = []
        all_trades = []
        all_predictions = []  # 保存所有预测数据
        
        for i in range(train_months, len(months), test_months):
            train_months_list = months[i-train_months:i]
            train_dates = []
            for _, month_dates in train_months_list:
                train_dates.extend(month_dates)
            
            test_months_list = months[i:i+test_months]
            test_dates = []
            for _, month_dates in test_months_list:
                test_dates.extend(month_dates)
            
            if len(test_dates) < 5:
                continue
            
            period_name = test_months_list[0][0]
            
            print(f"\n{'='*60}")
            print(f"回测月份: {period_name}")
            print(f"训练期: {train_dates[0]} 至 {train_dates[-1]} ({len(train_dates)} 天)")
            print(f"测试期: {test_dates[0]} 至 {test_dates[-1]} ({len(test_dates)} 天)")
            print(f"{'='*60}")
            
            # 准备训练数据
            print("准备训练数据...")
            train_df, feature_cols = self.load_and_prepare_data(train_dates)
            
            if len(train_df) == 0 or len(feature_cols) == 0:
                print("训练数据不足，跳过")
                continue
            
            print(f"训练样本: {len(train_df)}, 特征数: {len(feature_cols)}")
            print(f"正样本比例: {train_df['label'].mean():.2%}")
            
            # 训练模型
            print("训练模型...")
            model, scaler = self.train_model(train_df, feature_cols)
            
            # 获取特征重要性
            importance = model.feature_importances_
            top_features = [feature_cols[i] for i in np.argsort(importance)[-10:]]
            print(f"Top特征: {', '.join(top_features[:5])}")
            
            # 保存模型
            model_path, scaler_path, feature_path = self.save_model(
                model, scaler, feature_cols, period_name
            )
            print(f"模型已保存: {model_path}")
            
            # 生成并保存预测数据
            if save_predictions:
                print("生成预测数据...")
                predictions_df = self.generate_predictions(model, scaler, feature_cols, test_dates)
                
                if not predictions_df.empty:
                    pred_file = os.path.join(self.model_dir, f"predictions_{period_name}.csv")
                    predictions_df.to_csv(pred_file, index=False)
                    print(f"预测数据已保存: {pred_file}")
                    
                    # 保存预测元数据
                    pred_meta = {
                        'period': period_name,
                        'test_dates': test_dates,
                        'model_path': model_path,
                        'scaler_path': scaler_path,
                        'feature_path': feature_path,
                        'feature_cols': feature_cols,
                        'n_stocks': len(predictions_df['ts_code'].unique()),
                        'n_days': len(predictions_df['trade_date'].unique())
                    }
                    
                    meta_file = os.path.join(self.model_dir, f"meta_{period_name}.json")
                    with open(meta_file, 'w') as f:
                        json.dump(pred_meta, f, indent=2)
                    
                    # 使用预测数据进行回测
                    print("回测测试期...")
                    backtest_result = self.backtest_with_predictions(
                        predictions_df, test_dates, min_prob
                    )
                else:
                    backtest_result = {
                        'return': 0, 'sharpe': 0, 'max_dd': 0, 'n_trades': 0,
                        'win_rate': 0, 'avg_return': 0,
                        'skipped_limit_up': 0, 'skipped_limit_down': 0,
                        'daily_nav': [], 'trades': pd.DataFrame()
                    }
            else:
                # 直接回测（不保存预测数据）
                print("回测测试期...")
                backtest_result = self.backtest_month(model, scaler, feature_cols, test_dates, min_prob)
            
            print(f"测试期收益: {backtest_result['return']*100:.2f}%")
            print(f"夏普比率: {backtest_result['sharpe']:.2f}")
            print(f"最大回撤: {backtest_result['max_dd']*100:.2f}%")
            print(f"交易次数: {backtest_result['n_trades']}")
            print(f"胜率: {backtest_result['win_rate']*100:.2f}%")
            
            result = WalkForwardResult(
                period=period_name,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                test_start=test_dates[0],
                test_end=test_dates[-1],
                train_samples=len(train_df),
                test_samples=len(test_dates),
                train_positive_ratio=train_df['label'].mean(),
                test_return=backtest_result['return'],
                test_sharpe=backtest_result['sharpe'],
                test_max_dd=backtest_result['max_dd'],
                n_trades=backtest_result['n_trades'],
                win_rate=backtest_result['win_rate'],
                avg_return=backtest_result['avg_return'],
                skipped_limit_up=backtest_result['skipped_limit_up'],
                skipped_limit_down=backtest_result['skipped_limit_down'],
                top_features=top_features
            )
            
            results.append(result)
            
            all_equity.extend(backtest_result['daily_nav'])
            if backtest_result['n_trades'] > 0:
                all_trades.append(backtest_result['trades'])
        
        # 保存结果
        self.save_results(results, all_equity, all_trades)
        
        return results
    
    def run_with_saved_predictions(self, min_prob=0.6):
        """
        使用已保存的预测数据运行回测（无需重新训练模型）
        """
        print("=" * 80)
        print("使用已保存的预测数据运行回测")
        print(f"买入阈值: {min_prob}")
        print("=" * 80)
        
        # 查找所有预测数据文件
        pred_files = [f for f in os.listdir(self.model_dir) if f.startswith('predictions_') and f.endswith('.csv')]
        
        if not pred_files:
            print("未找到预测数据文件，请先运行完整回测！")
            return []
        
        print(f"找到 {len(pred_files)} 个预测数据文件")
        
        results = []
        all_equity = []
        all_trades = []
        
        for pred_file in sorted(pred_files):
            period_name = pred_file.replace('predictions_', '').replace('.csv', '')
            
            # 加载元数据
            meta_file = os.path.join(self.model_dir, f"meta_{period_name}.json")
            if not os.path.exists(meta_file):
                print(f"跳过 {period_name}: 未找到元数据")
                continue
            
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            
            test_dates = meta['test_dates']
            
            print(f"\n{'='*60}")
            print(f"回测月份: {period_name}")
            print(f"测试期: {test_dates[0]} 至 {test_dates[-1]} ({len(test_dates)} 天)")
            print(f"{'='*60}")
            
            # 加载预测数据
            pred_path = os.path.join(self.model_dir, pred_file)
            predictions_df = pd.read_csv(pred_path)
            
            print(f"预测数据: {len(predictions_df)} 条记录")
            
            # 使用预测数据进行回测
            backtest_result = self.backtest_with_predictions(
                predictions_df, test_dates, min_prob
            )
            
            print(f"测试期收益: {backtest_result['return']*100:.2f}%")
            print(f"夏普比率: {backtest_result['sharpe']:.2f}")
            print(f"最大回撤: {backtest_result['max_dd']*100:.2f}%")
            print(f"交易次数: {backtest_result['n_trades']}")
            print(f"胜率: {backtest_result['win_rate']*100:.2f}%")
            
            result = WalkForwardResult(
                period=period_name,
                train_start=meta.get('train_start', ''),
                train_end=meta.get('train_end', ''),
                test_start=test_dates[0],
                test_end=test_dates[-1],
                train_samples=meta.get('train_samples', 0),
                test_samples=len(test_dates),
                train_positive_ratio=meta.get('train_positive_ratio', 0),
                test_return=backtest_result['return'],
                test_sharpe=backtest_result['sharpe'],
                test_max_dd=backtest_result['max_dd'],
                n_trades=backtest_result['n_trades'],
                win_rate=backtest_result['win_rate'],
                avg_return=backtest_result['avg_return'],
                skipped_limit_up=backtest_result['skipped_limit_up'],
                skipped_limit_down=backtest_result['skipped_limit_down'],
                top_features=meta.get('top_features', [])
            )
            
            results.append(result)
            
            all_equity.extend(backtest_result['daily_nav'])
            if backtest_result['n_trades'] > 0:
                all_trades.append(backtest_result['trades'])
        
        # 保存结果
        self.save_results(results, all_equity, all_trades)
        
        return results
    
    def backtest_month(self, model, scaler, feature_cols, test_dates, min_prob=0.6):
        """回测单月（原始方法，用于不保存预测数据时）"""
        trades = []
        skipped_limit_up = 0
        skipped_limit_down = 0
        
        initial_capital = 100000.0
        capital = initial_capital
        daily_nav = []
        
        for i in range(len(test_dates) - 2):
            d_curr = test_dates[i]
            d_t1 = test_dates[i + 1]
            d_t2 = test_dates[i + 2]
            
            # 加载T日数据
            p_t = os.path.join(self.price_dir, f"{d_curr}.parquet")
            if not os.path.exists(p_t):
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            try:
                df_t = pd.read_parquet(p_t)
            except:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            # 过滤主板
            df_t = df_t[df_t['ts_code'].apply(is_main_board)]
            
            if len(df_t) == 0:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            # 计算特征
            df_t = calculate_simple_features(df_t)
            
            # 预测
            X = df_t[feature_cols].fillna(0)
            X_scaled = scaler.transform(X)
            df_t['prob'] = model.predict_proba(X_scaled)[:, 1]
            
            # 选择概率最高的股票
            best_idx = df_t['prob'].idxmax()
            best_prob = df_t.loc[best_idx, 'prob']
            
            if best_prob < min_prob:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            ts_code = df_t.loc[best_idx, 'ts_code']
            
            # 加载T+1和T+2价格
            p_t1 = os.path.join(self.price_dir, f"{d_t1}.parquet")
            p_t2 = os.path.join(self.price_dir, f"{d_t2}.parquet")
            
            if not os.path.exists(p_t1) or not os.path.exists(p_t2):
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            try:
                df_t1 = pd.read_parquet(p_t1)
                df_t2 = pd.read_parquet(p_t2)
            except:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            # 获取T+1数据
            t1_data = df_t1[df_t1['ts_code'] == ts_code]
            if t1_data.empty:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            t1_open = float(t1_data.iloc[0]['open'])
            t1_pre = float(t1_data.iloc[0]['pre_close'])
            
            limit_pct = get_limit_pct(ts_code)
            
            # 涨停检查
            t1_open_chg = (t1_open - t1_pre) / t1_pre * 100
            if t1_open_chg >= (limit_pct - LIMIT_THRESHOLD):
                skipped_limit_up += 1
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            # 获取T+2数据
            t2_data = df_t2[df_t2['ts_code'] == ts_code]
            if t2_data.empty:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            t2_close = float(t2_data.iloc[0]['close'])
            t2_low = float(t2_data.iloc[0]['low'])
            t2_open = float(t2_data.iloc[0]['open'])
            
            # 跌停检查
            t2_low_chg = (t2_low - t1_open) / t1_open * 100
            if t2_low_chg <= -(limit_pct - LIMIT_THRESHOLD):
                sell_price = min(t2_open, t1_open * (1 - limit_pct/100))
                skipped_limit_down += 1
            else:
                sell_price = t2_close
            
            # 应用滑点和费用
            buy_price = t1_open * (1 + SLIPPAGE)
            sell_price = sell_price * (1 - SLIPPAGE)
            ret = sell_price / buy_price - 1 - COST_RATE
            
            capital *= (1 + ret)
            
            trades.append({
                'date_t': d_curr,
                'date_t1': d_t1,
                'date_t2': d_t2,
                'ts_code': ts_code,
                'prob': best_prob,
                'buy_price': buy_price,
                'sell_price': sell_price,
                'return': ret
            })
            
            daily_nav.append({'date': d_t2, 'nav': capital})
        
        if len(trades) == 0:
            return {
                'return': 0, 'sharpe': 0, 'max_dd': 0, 'n_trades': 0,
                'win_rate': 0, 'avg_return': 0,
                'skipped_limit_up': skipped_limit_up,
                'skipped_limit_down': skipped_limit_down,
                'daily_nav': daily_nav, 'trades': pd.DataFrame()
            }
        
        trades_df = pd.DataFrame(trades)
        total_ret = capital / initial_capital - 1
        
        nav_df = pd.DataFrame(daily_nav)
        if len(nav_df) > 1:
            nav_df['ret'] = nav_df['nav'].pct_change()
            vol = nav_df['ret'].std() * np.sqrt(252)
            sharpe = (total_ret / (len(test_dates) / 252)) / vol if vol > 0 else 0
        else:
            sharpe = 0
        
        if len(nav_df) > 0:
            nav_df['cummax'] = nav_df['nav'].cummax()
            nav_df['dd'] = (nav_df['nav'] - nav_df['cummax']) / nav_df['cummax']
            max_dd = nav_df['dd'].min()
        else:
            max_dd = 0
        
        win_rate = (trades_df['return'] > 0).mean()
        avg_ret = trades_df['return'].mean()
        
        return {
            'return': total_ret, 'sharpe': sharpe, 'max_dd': max_dd,
            'n_trades': len(trades), 'win_rate': win_rate, 'avg_return': avg_ret,
            'skipped_limit_up': skipped_limit_up,
            'skipped_limit_down': skipped_limit_down,
            'daily_nav': daily_nav, 'trades': trades_df
        }
    
    def save_results(self, results, all_equity, all_trades):
        """保存结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 月度结果
        if results:
            results_df = pd.DataFrame([{
                'period': r.period,
                'train_start': r.train_start,
                'train_end': r.train_end,
                'test_start': r.test_start,
                'test_end': r.test_end,
                'train_samples': r.train_samples,
                'test_samples': r.test_samples,
                'train_positive_ratio': r.train_positive_ratio,
                'test_return': r.test_return,
                'test_sharpe': r.test_sharpe,
                'test_max_dd': r.test_max_dd,
                'n_trades': r.n_trades,
                'win_rate': r.win_rate,
                'avg_return': r.avg_return,
                'skipped_limit_up': r.skipped_limit_up,
                'skipped_limit_down': r.skipped_limit_down,
                'top_features': ','.join(r.top_features)
            } for r in results])
            
            results_file = os.path.join(self.output_dir, f'walk_forward_results_{timestamp}.csv')
            results_df.to_csv(results_file, index=False)
            print(f"\n月度结果已保存: {results_file}")
        
        # 权益曲线
        if all_equity:
            equity_df = pd.DataFrame(all_equity)
            equity_df = equity_df.drop_duplicates(subset=['date'])
            equity_df = equity_df.sort_values('date')
            
            equity_file = os.path.join(self.output_dir, f'equity_curve_{timestamp}.csv')
            equity_df.to_csv(equity_file, index=False)
            print(f"权益曲线已保存: {equity_file}")
            
            # 总体统计
            initial_nav = equity_df['nav'].iloc[0]
            final_nav = equity_df['nav'].iloc[-1]
            total_return = final_nav / initial_nav - 1
            
            equity_df['ret'] = equity_df['nav'].pct_change()
            vol = equity_df['ret'].std() * np.sqrt(252)
            
            years = len(equity_df) / 252
            ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            sharpe = ann_return / vol if vol > 0 else 0
            
            equity_df['cummax'] = equity_df['nav'].cummax()
            equity_df['dd'] = (equity_df['nav'] - equity_df['cummax']) / equity_df['cummax']
            max_dd = equity_df['dd'].min()
            
            print(f"\n{'='*60}")
            print("总体回测结果")
            print(f"{'='*60}")
            print(f"总收益: {total_return*100:.2f}%")
            print(f"年化收益: {ann_return*100:.2f}%")
            print(f"夏普比率: {sharpe:.2f}")
            print(f"最大回撤: {max_dd*100:.2f}%")
            print(f"交易天数: {len(equity_df)}")
            print(f"{'='*60}")
        
        # 交易记录
        if all_trades:
            trades_df = pd.concat(all_trades, ignore_index=True)
            trades_file = os.path.join(self.output_dir, f'all_trades_{timestamp}.csv')
            trades_df.to_csv(trades_file, index=False)
            print(f"交易记录已保存: {trades_file}")
            
            print(f"\n交易统计:")
            print(f"总交易次数: {len(trades_df)}")
            print(f"胜率: {(trades_df['return'] > 0).mean()*100:.2f}%")
            print(f"平均收益: {trades_df['return'].mean()*100:.2f}%")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='月度滚动回测')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'threshold'],
                       help='运行模式: full=完整回测, threshold=仅调整阈值')
    parser.add_argument('--min_prob', type=float, default=0.6,
                       help='买入概率阈值')
    parser.add_argument('--start_date', type=str, default='20230101',
                       help='回测开始日期')
    parser.add_argument('--end_date', type=str, default='20260331',
                       help='回测结束日期')
    parser.add_argument('--train_months', type=int, default=24,
                       help='训练期月数')
    parser.add_argument('--test_months', type=int, default=1,
                       help='测试期月数')
    
    args = parser.parse_args()
    
    backtest = WalkForwardBacktestV2()
    
    if args.mode == 'full':
        # 完整回测（训练模型并保存预测数据）
        results = backtest.run_walk_forward(
            start_date=args.start_date,
            end_date=args.end_date,
            train_months=args.train_months,
            test_months=args.test_months,
            min_prob=args.min_prob,
            save_predictions=True
        )
    else:
        # 仅调整阈值（使用已保存的预测数据）
        results = backtest.run_with_saved_predictions(min_prob=args.min_prob)
    
    print("\n" + "=" * 80)
    print("回测完成！")
    print(f"结果保存在: {backtest.output_dir}")
    print(f"模型保存在: {backtest.model_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
