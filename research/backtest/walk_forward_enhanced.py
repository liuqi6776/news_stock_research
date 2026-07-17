"""
增强版月度滚动回测框架 (Walk-Forward Analysis) - Enhanced
整合所有特征、添加历史特征、支持特征选择和重要性分析
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
from scipy import stats

# 导入因子库
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from factors.technical_factors import TechnicalFactors, ChanLunFactors
from factors.feature_optimizer import FeatureAnalyzer

# A股交易规则配置
COST_RATE = 0.003       # 交易费用 0.3% 双边
SLIPPAGE = 0.002        # 滑点 0.2%
LIMIT_THRESHOLD = 0.5   # 涨跌停阈值偏移


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


def calculate_all_features(df: pd.DataFrame, hist_data: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
    """
    计算所有特征（简单特征 + 技术指标 + 历史特征）
    
    Parameters:
    -----------
    df : pd.DataFrame
        当日数据
    hist_data : Dict[str, pd.DataFrame]
        历史数据，键为日期字符串
    """
    # 1. 简单特征（当日）
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['price_change'] = (df['close'] - df['pre_close']) / df['pre_close']
    df['high_change'] = (df['high'] - df['pre_close']) / df['pre_close']
    df['low_change'] = (df['low'] - df['pre_close']) / df['pre_close']
    df['amplitude'] = (df['high'] - df['low']) / df['pre_close']
    df['vol_amount'] = df['close'] * df['vol']
    df['vol_ratio'] = df['vol'] / (df['vol'].mean() + 1e-8)
    df['open_position'] = (df['open'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['body_size'] = abs(df['close'] - df['open']) / df['pre_close']
    df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['pre_close']
    df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['pre_close']
    df['is_yang'] = (df['close'] > df['open']).astype(int)
    df['close_to_high'] = (df['close'] - df['high']) / df['pre_close']
    df['close_to_low'] = (df['close'] - df['low']) / df['pre_close']
    df['gap'] = (df['open'] - df['pre_close']) / df['pre_close']
    df['intraday_trend'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # 2. 技术指标（需要历史数据）
    if hist_data is not None and len(hist_data) >= 60:
        # 构建历史价格DataFrame
        hist_list = []
        for date, hist_df in sorted(hist_data.items()):
            hist_df = hist_df.copy()
            hist_df['trade_date'] = date
            hist_list.append(hist_df[['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']])
        
        if hist_list:
            hist_panel = pd.concat(hist_list, ignore_index=True)
            
            # 为每只股票计算技术指标
            all_tech_features = []
            for ts_code in df['ts_code'].unique():
                stock_hist = hist_panel[hist_panel['ts_code'] == ts_code].sort_values('trade_date')
                if len(stock_hist) < 20:
                    continue
                
                # 添加当日数据
                today_data = df[df['ts_code'] == ts_code][['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']]
                if len(today_data) == 0:
                    continue
                
                stock_data = pd.concat([stock_hist, today_data], ignore_index=True)
                stock_data = stock_data.sort_values('trade_date').reset_index(drop=True)
                
                # 计算技术指标
                stock_data = TechnicalFactors.momentum(stock_data, windows=[5, 10, 20, 60])
                stock_data = TechnicalFactors.volatility(stock_data, windows=[5, 10, 20, 60])
                stock_data = TechnicalFactors.volume_features(stock_data, windows=[5, 10, 20])
                stock_data = TechnicalFactors.moving_average(stock_data, windows=[5, 10, 20, 60, 120])
                stock_data = TechnicalFactors.rsi(stock_data, windows=[6, 12, 24])
                stock_data = TechnicalFactors.macd(stock_data)
                stock_data = TechnicalFactors.bollinger_bands(stock_data)
                stock_data = TechnicalFactors.kdj(stock_data)
                stock_data = TechnicalFactors.williams_r(stock_data, windows=[10, 20])
                stock_data = TechnicalFactors.atr(stock_data)
                
                # 只保留最新一天的数据
                latest = stock_data.iloc[-1:].copy()
                all_tech_features.append(latest)
            
            if all_tech_features:
                tech_df = pd.concat(all_tech_features, ignore_index=True)
                # 合并到原始数据
                tech_cols = [c for c in tech_df.columns if c not in ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']]
                df = df.merge(tech_df[['ts_code'] + tech_cols], on='ts_code', how='left')
    
    return df


def select_features_by_importance(train_df: pd.DataFrame, feature_cols: List[str], 
                                  top_n: int = 30) -> List[str]:
    """
    使用XGBoost特征重要性选择特征
    """
    X = train_df[feature_cols].fillna(0)
    y = train_df['label']
    
    # 移除常数列
    valid_cols = []
    for col in feature_cols:
        if X[col].std() > 0:
            valid_cols.append(col)
    
    if len(valid_cols) == 0:
        return feature_cols
    
    X = X[valid_cols]
    
    # 快速训练XGBoost获取特征重要性
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    model.fit(X, y)
    
    # 获取重要性并排序
    importance = model.feature_importances_
    feature_importance = list(zip(valid_cols, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    selected = [f[0] for f in feature_importance[:top_n]]
    
    return selected


class EnhancedWalkForwardBacktest:
    """增强版月度滚动回测类"""
    
    def __init__(self,
                 data_dir: str = r'D:\iquant_data\data_v2',
                 output_dir: str = None,
                 model_dir: str = None,
                 use_all_features: bool = True,
                 use_feature_selection: bool = True,
                 top_n_features: int = 30):
        
        self.data_dir = data_dir
        self.price_dir = os.path.join(data_dir, 'data_day1')
        self.use_all_features = use_all_features
        self.use_feature_selection = use_feature_selection
        self.top_n_features = top_n_features
        
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
        print(f"使用全部特征: {use_all_features}")
        print(f"特征选择: {use_feature_selection} (Top {top_n_features})")
    
    def load_and_prepare_data(self, dates: List[str], label_threshold: float = 0.02) -> Tuple[pd.DataFrame, List[str]]:
        """
        加载并准备数据（增强版）
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
            
            # 加载历史数据用于计算技术指标
            hist_data = {}
            if self.use_all_features:
                # 获取过去60天的数据
                curr_idx = self.all_dates.index(d_curr) if d_curr in self.all_dates else -1
                if curr_idx >= 60:
                    for j in range(curr_idx - 60, curr_idx):
                        hist_date = self.all_dates[j]
                        hist_path = os.path.join(self.price_dir, f"{hist_date}.parquet")
                        if os.path.exists(hist_path):
                            try:
                                hist_df = pd.read_parquet(hist_path)
                                hist_df = hist_df[hist_df['ts_code'].apply(is_main_board)]
                                hist_data[hist_date] = hist_df
                            except:
                                continue
            
            # 计算所有特征
            df_t = calculate_all_features(df_t, hist_data if self.use_all_features else None)
            
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
        
        # 特征选择
        if self.use_feature_selection and len(feature_cols) > self.top_n_features:
            print(f"特征选择: 从 {len(feature_cols)} 个特征中选择 Top {self.top_n_features}...")
            selected_cols = select_features_by_importance(result, feature_cols, self.top_n_features)
            print(f"选择的特征: {', '.join(selected_cols[:10])}...")
            return result, selected_cols
        
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
    
    def analyze_features(self, train_df: pd.DataFrame, feature_cols: List[str], period_name: str):
        """分析特征重要性和IC"""
        print(f"\n{'='*60}")
        print(f"特征分析 - {period_name}")
        print(f"{'='*60}")
        
        # 1. XGBoost特征重要性
        X = train_df[feature_cols].fillna(0)
        y = train_df['label']
        
        model = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        importance = model.feature_importances_
        feature_imp = list(zip(feature_cols, importance))
        feature_imp.sort(key=lambda x: x[1], reverse=True)
        
        print("\nTop 20 XGBoost特征重要性:")
        for i, (feat, imp) in enumerate(feature_imp[:20]):
            print(f"{i+1:2d}. {feat:25s} {imp:.4f}")
        
        # 2. IC分析
        print("\nTop 20 IC分析:")
        ic_results = []
        for feat in feature_cols:
            valid_data = train_df[[feat, 'label_ret']].dropna()
            if len(valid_data) > 30:
                ic, p_value = stats.spearmanr(valid_data[feat], valid_data['label_ret'])
                ic_results.append((feat, ic, p_value))
        
        ic_results.sort(key=lambda x: abs(x[1]), reverse=True)
        for i, (feat, ic, p) in enumerate(ic_results[:20]):
            print(f"{i+1:2d}. {feat:25s} IC={ic:7.4f} p={p:.4f}")
        
        # 保存特征分析结果
        analysis_result = {
            'period': period_name,
            'xgboost_importance': [{'feature': f, 'importance': float(i)} for f, i in feature_imp[:30]],
            'ic_analysis': [{'feature': f, 'ic': float(ic), 'p_value': float(p)} for f, ic, p in ic_results[:30]]
        }
        
        analysis_file = os.path.join(self.model_dir, f"feature_analysis_{period_name}.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis_result, f, indent=2)
        
        print(f"\n特征分析已保存: {analysis_file}")
        
        return feature_imp[:10]
    
    def run_walk_forward(self, start_date='20230101', end_date='20260331',
                        train_months=12, test_months=1, min_prob=0.55,
                        save_predictions=True, analyze_features=True):
        """
        运行增强版月度滚动回测
        """
        print("=" * 80)
        print("增强版月度滚动回测 (Walk-Forward Analysis) - Enhanced")
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
            
            # 特征分析
            if analyze_features:
                top_features = self.analyze_features(train_df, feature_cols, period_name)
            else:
                # 获取特征重要性
                X = train_df[feature_cols].fillna(0)
                y = train_df['label']
                model_temp = xgb.XGBClassifier(n_estimators=50, max_depth=4, random_state=42)
                model_temp.fit(X, y)
                importance = model_temp.feature_importances_
                top_features = [feature_cols[i] for i in np.argsort(importance)[-10:]]
            
            # 训练模型
            print("训练模型...")
            model, scaler = self.train_model(train_df, feature_cols)
            
            # 保存模型
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(self.model_dir, f"model_{period_name}_{timestamp}.pkl")
            scaler_path = os.path.join(self.model_dir, f"scaler_{period_name}_{timestamp}.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            
            # 生成预测
            print("生成预测...")
            predictions = []
            for d_curr in test_dates[:-2]:
                p_t = os.path.join(self.price_dir, f"{d_curr}.parquet")
                if not os.path.exists(p_t):
                    continue
                
                try:
                    df_t = pd.read_parquet(p_t)
                except:
                    continue
                
                df_t = df_t[df_t['ts_code'].apply(is_main_board)]
                if len(df_t) == 0:
                    continue
                
                # 加载历史数据
                hist_data = {}
                if self.use_all_features:
                    curr_idx = self.all_dates.index(d_curr) if d_curr in self.all_dates else -1
                    if curr_idx >= 60:
                        for j in range(curr_idx - 60, curr_idx):
                            hist_date = self.all_dates[j]
                            hist_path = os.path.join(self.price_dir, f"{hist_date}.parquet")
                            if os.path.exists(hist_path):
                                try:
                                    hist_df = pd.read_parquet(hist_path)
                                    hist_df = hist_df[hist_df['ts_code'].apply(is_main_board)]
                                    hist_data[hist_date] = hist_df
                                except:
                                    continue
                
                df_t = calculate_all_features(df_t, hist_data if self.use_all_features else None)
                
                # 预测
                X = df_t[feature_cols].fillna(0)
                X_scaled = scaler.transform(X)
                df_t['prob'] = model.predict_proba(X_scaled)[:, 1]
                
                predictions.append(df_t[['ts_code', 'trade_date', 'prob']].copy())
            
            if not predictions:
                continue
            
            predictions_df = pd.concat(predictions, ignore_index=True)
            
            # 回测
            print("回测...")
            backtest_result = self.backtest_with_predictions(predictions_df, test_dates, min_prob)
            
            print(f"\n回测结果:")
            print(f"  收益率: {backtest_result['return']:.2%}")
            print(f"  夏普比率: {backtest_result['sharpe']:.2f}")
            print(f"  最大回撤: {backtest_result['max_dd']:.2%}")
            print(f"  交易次数: {backtest_result['n_trades']}")
            print(f"  胜率: {backtest_result['win_rate']:.2%}")
            
            # 保存结果
            result = {
                'period': period_name,
                'train_start': train_dates[0],
                'train_end': train_dates[-1],
                'test_start': test_dates[0],
                'test_end': test_dates[-1],
                'train_samples': len(train_df),
                'test_samples': len(test_dates),
                'train_positive_ratio': train_df['label'].mean(),
                'test_return': backtest_result['return'],
                'test_sharpe': backtest_result['sharpe'],
                'test_max_dd': backtest_result['max_dd'],
                'n_trades': backtest_result['n_trades'],
                'win_rate': backtest_result['win_rate'],
                'avg_return': backtest_result['avg_return'],
                'skipped_limit_up': backtest_result['skipped_limit_up'],
                'skipped_limit_down': backtest_result['skipped_limit_down'],
                'top_features': [f[0] for f in top_features[:10]]
            }
            results.append(result)
            
            # 保存交易记录
            if len(backtest_result['trades']) > 0:
                all_trades.append(backtest_result['trades'])
            
            # 保存权益曲线
            if backtest_result['daily_nav']:
                nav_df = pd.DataFrame(backtest_result['daily_nav'])
                all_equity.append(nav_df)
        
        # 保存汇总结果
        if results:
            results_df = pd.DataFrame(results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_df.to_csv(os.path.join(self.output_dir, f'enhanced_results_{timestamp}.csv'), index=False)
            
            if all_equity:
                equity_df = pd.concat(all_equity, ignore_index=True)
                equity_df.to_csv(os.path.join(self.output_dir, f'enhanced_equity_{timestamp}.csv'), index=False)
            
            if all_trades:
                trades_df = pd.concat(all_trades, ignore_index=True)
                trades_df.to_csv(os.path.join(self.output_dir, f'enhanced_trades_{timestamp}.csv'), index=False)
            
            print(f"\n{'='*60}")
            print("回测完成！")
            print(f"结果保存至: {self.output_dir}")
            print(f"{'='*60}")
        
        return results
    
    def backtest_with_predictions(self, predictions_df, test_dates, min_prob=0.6):
        """使用预测数据进行回测"""
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
            
            d_curr_str = str(d_curr)
            day_pred = predictions_df[predictions_df['trade_date'].astype(str) == d_curr_str]
            
            if len(day_pred) == 0:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            best_idx = day_pred['prob'].idxmax()
            best_prob = day_pred.loc[best_idx, 'prob']
            
            if best_prob < min_prob:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            ts_code = day_pred.loc[best_idx, 'ts_code']
            
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
            
            t1_data = df_t1[df_t1['ts_code'] == ts_code]
            if t1_data.empty:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            t1_open = float(t1_data.iloc[0]['open'])
            t1_pre = float(t1_data.iloc[0]['pre_close'])
            
            limit_pct = get_limit_pct(ts_code)
            
            t1_open_chg = (t1_open - t1_pre) / t1_pre * 100
            if t1_open_chg >= (limit_pct - LIMIT_THRESHOLD):
                skipped_limit_up += 1
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            t2_data = df_t2[df_t2['ts_code'] == ts_code]
            if t2_data.empty:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            t2_close = float(t2_data.iloc[0]['close'])
            t2_low = float(t2_data.iloc[0]['low'])
            t2_open = float(t2_data.iloc[0]['open'])
            
            t2_low_chg = (t2_low - t1_open) / t1_open * 100
            if t2_low_chg <= -(limit_pct - LIMIT_THRESHOLD):
                sell_price = min(t2_open, t1_open * (1 - limit_pct/100))
                skipped_limit_down += 1
            else:
                sell_price = t2_close
            
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


if __name__ == "__main__":
    # 运行增强版回测
    bt = EnhancedWalkForwardBacktest(
        use_all_features=True,
        use_feature_selection=True,
        top_n_features=30
    )
    
    results = bt.run_walk_forward(
        start_date='20230101',
        end_date='20260331',
        train_months=12,
        test_months=1,
        min_prob=0.55,
        analyze_features=True
    )
