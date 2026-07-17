"""
Study 003: 优化框架运行脚本（年级别Walk Forward）

改进点：
1. 修复未来函数
2. 支持多周期预测（2日/5日/10日）
3. 年级别更新，提升速度
4. 动态阈值和改进止损
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from shared.data_loader import get_all_dates, PRICE_DIR
from config import *

# ==================== 修复后的特征工程 ====================

def is_main_board(ts_code: str) -> bool:
    return ts_code.startswith(('60', '00', '002', '003'))

def calculate_features_no_lookahead(df: pd.DataFrame, hist_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    计算特征 - 确保无未来函数
    
    Args:
        df: 当日数据
        hist_data: 历史数据（用于计算历史统计量）
    """
    df = df.copy()
    
    # 基础价格特征（当日数据，无未来函数）
    df['pct_chg'] = (df['close'] - df['pre_close']) / df['pre_close']
    df['price_change'] = df['close'] - df['pre_close']
    df['amplitude'] = (df['high'] - df['low']) / df['pre_close']
    df['body_size'] = abs(df['close'] - df['open']) / df['pre_close']
    df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['pre_close']
    df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['pre_close']
    df['is_yang'] = (df['close'] > df['open']).astype(int)
    df['gap'] = (df['open'] - df['pre_close']) / df['pre_close']
    
    # 成交量特征（使用历史均值，非当日均值）
    if hist_data is not None and not hist_data.empty:
        hist_vol_mean = hist_data.groupby('ts_code')['vol'].mean().reset_index()
        hist_vol_mean.columns = ['ts_code', 'hist_vol_mean']
        df = pd.merge(df, hist_vol_mean, on='ts_code', how='left')
        df['vol_ratio_hist'] = df['vol'] / (df['hist_vol_mean'] + 1e-8)
    else:
        df['vol_ratio_hist'] = 1.0
    
    df['vol_amount'] = df['close'] * df['vol']
    
    return df


def calculate_momentum_features(df: pd.DataFrame, price_history: dict) -> pd.DataFrame:
    """
    计算动量特征 - 使用历史价格数据
    
    Args:
        df: 当日数据
        price_history: 历史价格字典 {ts_code: [price_list]}
    """
    df = df.copy()
    
    for w in [5, 10, 20, 60]:
        mom_col = f'mom_{w}d'
        df[mom_col] = np.nan
        
        for idx, row in df.iterrows():
            ts_code = row['ts_code']
            if ts_code in price_history and len(price_history[ts_code]) >= w:
                hist_prices = price_history[ts_code]
                df.at[idx, mom_col] = (row['close'] - hist_prices[-w]) / hist_prices[-w]
    
    return df


def calculate_volatility_features(df: pd.DataFrame, price_history: dict) -> pd.DataFrame:
    """计算波动率特征 - 使用历史数据"""
    df = df.copy()
    
    for w in [5, 10, 20]:
        vol_col = f'vol_{w}d'
        df[vol_col] = np.nan
        
        for idx, row in df.iterrows():
            ts_code = row['ts_code']
            if ts_code in price_history and len(price_history[ts_code]) >= w:
                hist_prices = price_history[ts_code]
                returns = [(hist_prices[i] - hist_prices[i-1]) / hist_prices[i-1] 
                          for i in range(1, len(hist_prices))]
                if len(returns) >= w:
                    df.at[idx, vol_col] = np.std(returns[-w:]) * (252 ** 0.5)
    
    return df


def load_news_features(date: str) -> pd.DataFrame:
    """加载新闻特征"""
    news_path = os.path.join(NEWS_DIR, f"{date}.parquet")
    if os.path.exists(news_path):
        return pd.read_parquet(news_path)
    return pd.DataFrame()


def load_rank_features(date: str) -> pd.DataFrame:
    """加载排名特征"""
    rank_path = os.path.join(RANK_DIR, f"{date}.parquet")
    if os.path.exists(rank_path):
        return pd.read_parquet(rank_path)
    return pd.DataFrame()


# ==================== 数据准备 ====================

def prepare_data_year_by_year(start_year: int, end_year: int) -> pd.DataFrame:
    """
    按年准备数据 - 年级别Walk Forward
    
    Returns:
        DataFrame with features and labels
    """
    all_dates = get_all_dates()
    
    # 过滤日期范围
    dates_in_range = [d for d in all_dates if start_year <= int(d[:4]) <= end_year]
    
    training_data = []
    price_history = {}  # 存储历史价格用于计算动量
    
    print(f"准备数据: {start_year}-{end_year}, 共{len(dates_in_range)}个交易日")
    
    for i in tqdm(range(len(dates_in_range)), desc="加载数据"):
        d_curr = dates_in_range[i]
        
        # 需要t+N的数据作为目标
        if i + PREDICT_HORIZON >= len(dates_in_range):
            break
        
        d_future = dates_in_range[i + PREDICT_HORIZON]
        
        # 加载当日数据
        p_curr = os.path.join(PRICE_DIR, f"{d_curr}.parquet")
        p_future = os.path.join(PRICE_DIR, f"{d_future}.parquet")
        
        if not os.path.exists(p_curr) or not os.path.exists(p_future):
            continue
        
        try:
            df_curr = pd.read_parquet(p_curr)
            df_future = pd.read_parquet(p_future)
        except:
            continue
        
        # 过滤主板
        df_curr = df_curr[df_curr['ts_code'].apply(is_main_board)]
        
        if df_curr.empty:
            continue
        
        # 更新价格历史
        for _, row in df_curr.iterrows():
            ts_code = row['ts_code']
            if ts_code not in price_history:
                price_history[ts_code] = []
            price_history[ts_code].append(row['close'])
            # 保持历史数据在合理长度
            if len(price_history[ts_code]) > 100:
                price_history[ts_code] = price_history[ts_code][-100:]
        
        # 计算特征（无未来函数）
        # 获取历史数据用于计算统计量
        hist_data = None
        if i > 20:
            hist_dates = dates_in_range[max(0, i-60):i]
            hist_frames = []
            for hd in hist_dates:
                hp = os.path.join(PRICE_DIR, f"{hd}.parquet")
                if os.path.exists(hp):
                    hdf = pd.read_parquet(hp)
                    hist_frames.append(hdf[['ts_code', 'vol']])
            if hist_frames:
                hist_data = pd.concat(hist_frames, ignore_index=True)
        
        features = calculate_features_no_lookahead(df_curr, hist_data)
        features = calculate_momentum_features(features, price_history)
        features = calculate_volatility_features(features, price_history)
        
        # 加载新闻和排名特征
        news_df = load_news_features(d_curr)
        rank_df = load_rank_features(d_curr)
        
        if not news_df.empty:
            news_cols = [c for c in news_df.columns if c not in ['ts_code', 'trade_date']]
            features = pd.merge(features, news_df[['ts_code'] + news_cols], on='ts_code', how='left')
        
        if not rank_df.empty:
            rank_cols = [c for c in rank_df.columns if c not in ['ts_code', 'trade_date']]
            features = pd.merge(features, rank_df[['ts_code'] + rank_cols], on='ts_code', how='left')
        
        # 计算目标变量 (t+1开盘买入, t+N收盘卖出)
        d_t1 = dates_in_range[i + 1]
        p_t1 = os.path.join(PRICE_DIR, f"{d_t1}.parquet")
        
        if os.path.exists(p_t1):
            df_t1 = pd.read_parquet(p_t1)
            t1_data = df_t1[['ts_code', 'open']].rename(columns={'open': 't1_open'})
            features = pd.merge(features, t1_data, on='ts_code', how='left')
        
        future_data = df_future[['ts_code', 'close']].rename(columns={'close': f't{PREDICT_HORIZON}_close'})
        features = pd.merge(features, future_data, on='ts_code', how='left')
        
        # 计算收益率
        if 't1_open' in features.columns:
            features['label_ret'] = features[f't{PREDICT_HORIZON}_close'] / features['t1_open'] - 1
        
        features['trade_date'] = d_curr
        training_data.append(features)
    
    if training_data:
        result = pd.concat(training_data, ignore_index=True)
        print(f"数据准备完成: {len(result)} 行")
        return result
    
    return pd.DataFrame()


# ==================== 模型训练 ====================

def train_model(train_df: pd.DataFrame, feature_cols: list) -> object:
    """训练XGBoost模型"""
    from xgboost import XGBClassifier
    
    # 准备标签
    train_df = train_df.copy()
    train_df['label'] = (train_df['label_ret'] > LABEL_THRESHOLD).astype(int)
    
    # 处理缺失值
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['label']
    
    print(f"训练数据: {len(X_train)} 行, 正样本: {y_train.sum()} ({y_train.mean():.2%})")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # 输出特征重要性
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 特征:")
    print(importance.head(10))
    
    return model


# ==================== 回测引擎 ====================

def get_market_state(date: str, price_dir: str, all_dates: list) -> dict:
    """
    获取市场状态 - 用于动态阈值
    
    Returns:
        dict with 'trend' and 'volatility'
    """
    try:
        date_idx = all_dates.index(date)
        if date_idx < 20:
            return {'trend': 0, 'volatility': 0}
        
        # 获取最近20天数据
        hist_dates = all_dates[date_idx-20:date_idx]
        market_returns = []
        
        for d in hist_dates:
            p = os.path.join(price_dir, f"{d}.parquet")
            if os.path.exists(p):
                df = pd.read_parquet(p)
                if not df.empty:
                    avg_ret = ((df['close'] - df['pre_close']) / df['pre_close']).mean()
                    market_returns.append(avg_ret)
        
        if len(market_returns) > 0:
            trend = np.mean(market_returns)
            volatility = np.std(market_returns)
            return {'trend': trend, 'volatility': volatility}
        
        return {'trend': 0, 'volatility': 0}
    except:
        return {'trend': 0, 'volatility': 0}


def run_backtest_optimized(predictions_df: pd.DataFrame,
                          test_dates: list,
                          price_dir: str,
                          all_dates: list) -> dict:
    """
    优化后的回测引擎
    
    改进：
    1. 动态阈值
    2. 跟踪止损
    3. 市场环境过滤
    """
    trades = []
    initial_capital = 100000.0
    capital = initial_capital
    daily_nav = []
    
    for i in tqdm(range(len(test_dates) - PREDICT_HORIZON), desc="回测"):
        d_curr = test_dates[i]
        d_t1 = test_dates[i + 1]
        d_tn = test_dates[i + PREDICT_HORIZON]
        
        # 市场环境判断
        market_state = get_market_state(d_curr, price_dir, all_dates)
        
        # 波动率过滤 - 高波动时降低仓位
        if USE_VOLATILITY_FILTER and market_state['volatility'] > 0.03:
            max_pos = max(1, MAX_POSITIONS // 2)
        else:
            max_pos = MAX_POSITIONS
        
        # 动态阈值 - 趋势好时降低阈值，趋势差时提高阈值
        if USE_DYNAMIC_THRESHOLD:
            dynamic_threshold = MIN_PROB - market_state['trend'] * THRESHOLD_ADJUSTMENT
            dynamic_threshold = max(0.5, min(0.7, dynamic_threshold))
        else:
            dynamic_threshold = MIN_PROB
        
        # 市场环境过滤
        if USE_MARKET_FILTER and market_state['trend'] < -0.02:
            daily_nav.append({'date': d_tn, 'nav': capital})
            continue
        
        # 获取当日预测
        day_pred = predictions_df[predictions_df['trade_date'].astype(str) == str(d_curr)]
        if len(day_pred) == 0:
            daily_nav.append({'date': d_tn, 'nav': capital})
            continue
        
        # 选择前N个最高概率的股票
        day_pred_sorted = day_pred.sort_values('prob', ascending=False)
        selected_stocks = day_pred_sorted.head(max_pos)
        
        daily_return = 0
        n_trades_today = 0
        
        for _, row in selected_stocks.iterrows():
            best_prob = row['prob']
            if best_prob < dynamic_threshold:
                continue
            
            ts_code = row['ts_code']
            
            # 加载t+1和t+N数据
            p_t1 = os.path.join(price_dir, f"{d_t1}.parquet")
            p_tn = os.path.join(price_dir, f"{d_tn}.parquet")
            if not os.path.exists(p_t1) or not os.path.exists(p_tn):
                continue
            
            try:
                df_t1 = pd.read_parquet(p_t1)
                df_tn = pd.read_parquet(p_tn)
            except:
                continue
            
            # 获取t+1数据
            t1_data = df_t1[df_t1['ts_code'] == ts_code]
            if t1_data.empty:
                continue
            
            t1_open = float(t1_data.iloc[0]['open'])
            t1_pre = float(t1_data.iloc[0]['pre_close'])
            
            # 跳过涨停开盘
            limit_pct = 20.0 if ts_code.startswith(('68', '30')) else 10.0
            t1_open_chg = (t1_open - t1_pre) / t1_pre * 100
            if t1_open_chg >= (limit_pct - 0.5):
                continue
            
            # 获取t+N数据
            tn_data = df_tn[df_tn['ts_code'] == ts_code]
            if tn_data.empty:
                continue
            
            tn_close = float(tn_data.iloc[0]['close'])
            
            # 加载中间日期数据用于跟踪止损
            sell_price = tn_close
            if PREDICT_HORIZON > 2:
                # 跟踪止损：加载中间数据检查是否触发止损
                for j in range(2, PREDICT_HORIZON):
                    d_mid = test_dates[i + j]
                    p_mid = os.path.join(price_dir, f"{d_mid}.parquet")
                    if os.path.exists(p_mid):
                        try:
                            df_mid = pd.read_parquet(p_mid)
                            mid_data = df_mid[df_mid['ts_code'] == ts_code]
                            if not mid_data.empty:
                                mid_low = float(mid_data.iloc[0]['low'])
                                current_ret = (mid_low - t1_open) / t1_open
                                if current_ret <= -STOP_LOSS:
                                    sell_price = t1_open * (1 - STOP_LOSS)
                                    break
                        except:
                            pass
            
            # 计算收益
            buy_price = t1_open * (1 + SLIPPAGE)
            sell_price = sell_price * (1 - SLIPPAGE)
            ret = sell_price / buy_price - 1 - COST_RATE
            
            # 分配资金
            position_size = 1.0 / max_pos
            daily_return += ret * position_size
            n_trades_today += 1
            
            trades.append({
                'date_t': d_curr,
                'date_entry': d_t1,
                'date_exit': d_tn,
                'ts_code': ts_code,
                'prob': best_prob,
                'threshold': dynamic_threshold,
                'market_trend': market_state['trend'],
                'buy_price': buy_price,
                'sell_price': sell_price,
                'return': ret
            })
        
        if n_trades_today > 0:
            capital *= (1 + daily_return)
        daily_nav.append({'date': d_tn, 'nav': capital})
    
    # 计算指标
    nav_df = pd.DataFrame(daily_nav)
    if len(nav_df) > 0:
        total_return = (nav_df['nav'].iloc[-1] / initial_capital) - 1
        daily_returns = nav_df['nav'].pct_change().dropna()
        sharpe = daily_returns.mean() / (daily_returns.std() + 1e-8) * (252 ** 0.5)
        max_dd = ((nav_df['nav'].cummax() - nav_df['nav']) / nav_df['nav'].cummax()).max()
    else:
        total_return = 0
        sharpe = 0
        max_dd = 0
    
    win_rate = len([t for t in trades if t['return'] > 0]) / len(trades) if trades else 0
    
    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'n_trades': len(trades),
        'win_rate': win_rate,
        'trades': trades,
        'nav': nav_df
    }


# ==================== 主流程 ====================

def run_study():
    """运行完整研究流程"""
    print("=" * 80)
    print(f"Study 003: 优化框架 - {PREDICT_HORIZON}日收益预测")
    print("=" * 80)
    print(f"预测周期: {PREDICT_HORIZON}日")
    print(f"目标阈值: {LABEL_THRESHOLD:.2%}")
    print(f"训练期: {TRAIN_YEARS}年")
    print(f"测试期: {TEST_YEARS}年")
    print(f"止损: {STOP_LOSS:.2%}")
    print(f"动态阈值: {USE_DYNAMIC_THRESHOLD}")
    print(f"市场过滤: {USE_MARKET_FILTER}")
    print("=" * 80)
    
    # 1. 准备数据
    features_path = os.path.join(DATA_DIR, f'features_h{PREDICT_HORIZON}.parquet')
    
    if os.path.exists(features_path):
        print(f"加载已存在的特征数据: {features_path}")
        features_df = pd.read_parquet(features_path)
    else:
        print("计算特征数据...")
        features_df = prepare_data_year_by_year(
            int(START_DATE[:4]), 
            int(END_DATE[:4])
        )
        if not features_df.empty:
            features_df.to_parquet(features_path)
            print(f"特征数据已保存: {features_path}")
    
    print(f"特征数据: {len(features_df)} 行")
    
    # 2. 选择特征列
    exclude_cols = ['ts_code', 'trade_date', 't1_open', f't{PREDICT_HORIZON}_close', 
                    'label_ret', 'label', 'hist_vol_mean']
    feature_cols = [c for c in features_df.columns if c not in exclude_cols]
    print(f"特征数量: {len(feature_cols)}")
    
    # 3. Walk Forward回测（年级别）
    all_dates = get_all_dates()
    years = sorted(list(set([d[:4] for d in all_dates if START_DATE <= d <= END_DATE])))
    
    print(f"\n年级别Walk Forward: {years}")
    
    all_predictions = []
    all_results = []
    
    for i in range(TRAIN_YEARS, len(years)):
        train_years = years[i-TRAIN_YEARS:i]
        test_year = years[i]
        
        print(f"\n{'='*60}")
        print(f"轮次 {i-TRAIN_YEARS+1}: 训练{train_years} -> 测试{test_year}")
        print(f"{'='*60}")
        
        # 分割数据
        train_mask = features_df['trade_date'].astype(str).str[:4].isin(train_years)
        test_mask = features_df['trade_date'].astype(str).str[:4] == test_year
        
        train_df = features_df[train_mask]
        test_df = features_df[test_mask]
        
        if train_df.empty or test_df.empty:
            print("数据不足，跳过")
            continue
        
        # 训练模型
        print("训练模型...")
        model = train_model(train_df, feature_cols)
        
        # 预测
        print("生成预测...")
        X_test = test_df[feature_cols].fillna(0)
        test_df = test_df.copy()
        test_df['prob'] = model.predict_proba(X_test)[:, 1]
        
        all_predictions.append(test_df[['trade_date', 'ts_code', 'prob', 'label_ret']])
        
        # 回测
        test_dates = sorted(test_df['trade_date'].unique().tolist())
        print(f"回测 {len(test_dates)} 个交易日...")
        
        results = run_backtest_optimized(
            test_df,
            test_dates,
            PRICE_DIR,
            all_dates
        )
        
        print(f"\n{test_year}年结果:")
        print(f"  收益率: {results['total_return']:.2%}")
        print(f"  夏普: {results['sharpe']:.2f}")
        print(f"  最大回撤: {results['max_drawdown']:.2%}")
        print(f"  交易次数: {results['n_trades']}")
        print(f"  胜率: {results['win_rate']:.2%}")
        
        all_results.append({
            'year': test_year,
            **results
        })
    
    # 4. 汇总结果
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)
    
    if all_results:
        total_trades = sum(r['n_trades'] for r in all_results)
        avg_return = np.mean([r['total_return'] for r in all_results])
        avg_sharpe = np.mean([r['sharpe'] for r in all_results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in all_results])
        avg_winrate = np.mean([r['win_rate'] for r in all_results])
        
        print(f"平均收益率: {avg_return:.2%}")
        print(f"平均夏普: {avg_sharpe:.2f}")
        print(f"平均最大回撤: {avg_drawdown:.2%}")
        print(f"总交易次数: {total_trades}")
        print(f"平均胜率: {avg_winrate:.2%}")
        
        # 保存预测
        if all_predictions:
            pred_df = pd.concat(all_predictions, ignore_index=True)
            pred_path = os.path.join(DATA_DIR, f'predictions_h{PREDICT_HORIZON}.parquet')
            pred_df.to_parquet(pred_path)
            print(f"\n预测数据已保存: {pred_path}")
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary = {
            'study_id': '003',
            'study_name': '优化框架',
            'predict_horizon': PREDICT_HORIZON,
            'timestamp': timestamp,
            'avg_return': float(avg_return),
            'avg_sharpe': float(avg_sharpe),
            'avg_drawdown': float(avg_drawdown),
            'total_trades': int(total_trades),
            'avg_winrate': float(avg_winrate),
            'yearly_results': [
                {
                    'year': r['year'],
                    'return': float(r['total_return']),
                    'sharpe': float(r['sharpe']),
                    'drawdown': float(r['max_drawdown']),
                    'trades': int(r['n_trades']),
                    'win_rate': float(r['win_rate'])
                }
                for r in all_results
            ]
        }
        
        summary_path = os.path.join(RESULTS_DIR, f'summary_{timestamp}.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"摘要已保存: {summary_path}")
        
        # 保存交易记录
        all_trades = []
        for r in all_results:
            all_trades.extend(r['trades'])
        if all_trades:
            trades_df = pd.DataFrame(all_trades)
            trades_df.to_csv(os.path.join(RESULTS_DIR, f'trades_{timestamp}.csv'), index=False)
    
    # 更新注册表
    print("\n更新研究注册表...")
    registry_script = os.path.join(os.path.dirname(__file__), '..', '..', 'update_registry.py')
    os.system(f"python {registry_script}")


if __name__ == '__main__':
    run_study()
