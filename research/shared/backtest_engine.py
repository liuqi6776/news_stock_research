"""
回测引擎 - 统一的回测逻辑
"""
import pandas as pd
import numpy as np
import os
from typing import Dict, List

# A股交易规则
COST_RATE = 0.003
SLIPPAGE = 0.002
LIMIT_THRESHOLD = 0.5


def get_limit_pct(ts_code: str) -> float:
    """获取涨跌停限制"""
    if ts_code.startswith('68') or ts_code.startswith('30'):
        return 20.0
    elif ts_code.startswith('8') or ts_code.startswith('4'):
        return 30.0
    return 10.0


def run_backtest(predictions_df: pd.DataFrame,
                test_dates: List[str],
                price_dir: str,
                min_prob: float = 0.55,
                stop_loss: float = 0.05,
                max_positions: int = 5,
                use_market_filter: bool = True) -> Dict:
    """
    运行回测
    
    Args:
        predictions_df: 预测结果DataFrame (包含ts_code, trade_date, prob)
        test_dates: 测试日期列表
        price_dir: 价格数据目录
        min_prob: 最小概率阈值
        stop_loss: 止损阈值
        max_positions: 最大持仓数
        use_market_filter: 是否使用市场环境过滤
    
    Returns:
        回测结果字典
    """
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
        
        # 市场环境过滤
        if use_market_filter:
            market_trend = _get_market_trend(d_curr, price_dir, test_dates)
            if market_trend < -0.02:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
        
        # 获取当日预测
        day_pred = predictions_df[predictions_df['trade_date'].astype(str) == d_curr_str]
        if len(day_pred) == 0:
            daily_nav.append({'date': d_t2, 'nav': capital})
            continue
        
        # 选择前N个最高概率的股票
        day_pred_sorted = day_pred.sort_values('prob', ascending=False)
        selected_stocks = day_pred_sorted.head(max_positions)
        
        daily_return = 0
        n_trades_today = 0
        
        for _, row in selected_stocks.iterrows():
            best_prob = row['prob']
            if best_prob < min_prob:
                continue
            
            ts_code = row['ts_code']
            
            # 加载t+1和t+2数据
            p_t1 = os.path.join(price_dir, f"{d_t1}.parquet")
            p_t2 = os.path.join(price_dir, f"{d_t2}.parquet")
            if not os.path.exists(p_t1) or not os.path.exists(p_t2):
                continue
            
            try:
                df_t1 = pd.read_parquet(p_t1)
                df_t2 = pd.read_parquet(p_t2)
            except:
                continue
            
            # 获取t+1数据
            t1_data = df_t1[df_t1['ts_code'] == ts_code]
            if t1_data.empty:
                continue
            
            t1_open = float(t1_data.iloc[0]['open'])
            t1_pre = float(t1_data.iloc[0]['pre_close'])
            limit_pct = get_limit_pct(ts_code)
            t1_open_chg = (t1_open - t1_pre) / t1_pre * 100
            
            # 跳过涨停开盘
            if t1_open_chg >= (limit_pct - LIMIT_THRESHOLD):
                skipped_limit_up += 1
                continue
            
            # 获取t+2数据
            t2_data = df_t2[df_t2['ts_code'] == ts_code]
            if t2_data.empty:
                continue
            
            t2_close = float(t2_data.iloc[0]['close'])
            t2_low = float(t2_data.iloc[0]['low'])
            
            # 止损逻辑
            current_ret = (t2_low - t1_open) / t1_open
            if current_ret <= -stop_loss:
                sell_price = t1_open * (1 - stop_loss)
            else:
                sell_price = t2_close
            
            # 计算收益
            buy_price = t1_open * (1 + SLIPPAGE)
            sell_price = sell_price * (1 - SLIPPAGE)
            ret = sell_price / buy_price - 1 - COST_RATE
            
            # 分配资金
            position_size = 1.0 / max_positions
            daily_return += ret * position_size
            n_trades_today += 1
            
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
        
        if n_trades_today > 0:
            capital *= (1 + daily_return)
        daily_nav.append({'date': d_t2, 'nav': capital})
    
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
        'nav': nav_df,
        'skipped_limit_up': skipped_limit_up,
        'skipped_limit_down': skipped_limit_down
    }


def _get_market_trend(date: str, price_dir: str, all_dates: List[str], window: int = 20) -> float:
    """计算市场趋势"""
    try:
        date_idx = all_dates.index(date)
        if date_idx < window:
            return 0.0
        
        hist_dates = all_dates[date_idx-window:date_idx]
        market_returns = []
        
        for d in hist_dates:
            p = os.path.join(price_dir, f"{d}.parquet")
            if os.path.exists(p):
                df = pd.read_parquet(p)
                if not df.empty:
                    avg_ret = ((df['close'] - df['pre_close']) / df['pre_close']).mean()
                    market_returns.append(avg_ret)
        
        if len(market_returns) > 0:
            return np.mean(market_returns)
        return 0.0
    except:
        return 0.0
