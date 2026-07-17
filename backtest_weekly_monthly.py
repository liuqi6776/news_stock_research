"""
方案 A：基于现有 dragon_rolling_predictions，测试周频 + 月频持仓策略

核心逻辑:
- 每周一（或每月初）：按上一周五（月末）日的 AI 分数，选 Top-N 股票买入
- 持有整周 / 整月，到下周一（月初）按开盘价清仓，再换仓
- 价格数据直接从 iquant data 存储读取

关键优势：
- 彻底回避 T+1 的日内 Alpha 衰减问题
- 每次持仓 5~20 天，AI 信号有更长的价值窗口
- 换仓频率低，摩擦成本极低
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# 配置参数
# ============================================================
PRED_FILE   = r"C:\Users\liuqi\iquant\quant_trading_system\dragon_rolling_predictions.parquet"
DATA_DIR    = r"D:\iquant_data\data_v2\data_day1"   # 日 K 数据目录（parquet 按日期存储）

INITIAL_CAP = 100_000.0
TOP_N       = 5          # 每次持仓股票数
COMMISSION  = 0.0003     # 单边手续费
STAMP_DUTY  = 0.0005     # 印花税（卖出）
SLIPPAGE    = 0.001      # 滑点
SCORE_THRESHOLD = 0.65   # 最低分数门槛（过滤低质量信号）
STOP_LOSS   = -0.08      # 单笔持仓止损线 (-8%)，防止持有期间系统性崩盘

# SSE Index 数据（用于大盘趋势过滤）
SSE_INDEX_FILE = r'C:\Users\liuqi\quant_system_v2\sse_index_2023.csv'


# ============================================================
# 数据加载
# ============================================================
def load_index_filter():
    """加载上证指数并构建大盘过滤字典 {date_str -> is_bull_bool}"""
    if not os.path.exists(SSE_INDEX_FILE):
        print("⚠️ SSE Index 文件不存在，跳过大盘过滤")
        return {}
    idx = pd.read_csv(SSE_INDEX_FILE)
    idx['date'] = pd.to_datetime(idx['date'])
    idx = idx.sort_values('date')
    idx['ma20'] = idx['close'].rolling(20).mean()
    idx['is_bull'] = idx['close'] > idx['ma20']
    return dict(zip(idx['date'].dt.strftime('%Y-%m-%d'), idx['is_bull']))


def load_predictions():
    print("加载预测数据...")
    df = pd.read_parquet(PRED_FILE)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    # 只保留沪深主板（以 00/60 开头）
    df = df[df['ts_code'].str.match(r'^(00|60)')]
    print(f"  预测记录数: {len(df):,}, 股票数: {df['ts_code'].nunique()}")
    return df


def load_price_data(start_date: str, end_date: str) -> pd.DataFrame:
    """加载日 K 价格数据（从 data_day1 按日期 parquet 读取）"""
    print(f"加载价格数据 {start_date} ~ {end_date}...")
    all_files = []
    sd = pd.to_datetime(start_date)
    ed = pd.to_datetime(end_date)

    if not os.path.exists(DATA_DIR):
        print(f"  ⚠️ 价格目录不存在: {DATA_DIR}")
        return pd.DataFrame()

    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith('.parquet'):
            continue
        date_str = fname.replace('.parquet', '')
        try:
            dt = pd.to_datetime(date_str)
        except Exception:
            continue
        if sd <= dt <= ed:
            fpath = os.path.join(DATA_DIR, fname)
            try:
                tmp = pd.read_parquet(fpath, columns=['ts_code', 'trade_date', 'open', 'close', 'pct_chg'])
                all_files.append(tmp)
            except Exception:
                pass

    if not all_files:
        print("  ⚠️ 未找到价格数据！")
        return pd.DataFrame()

    price = pd.concat(all_files, ignore_index=True)
    price['trade_date'] = pd.to_datetime(price['trade_date'].astype(str))
    # 只保留沪深主板
    price = price[price['ts_code'].str.match(r'^(00|60)')]
    print(f"  价格记录数: {len(price):,}")
    return price


# ============================================================
# 回测核心
# ============================================================
def run_backtest(
    pred_df: pd.DataFrame,
    price_df: pd.DataFrame,
    market_filter: dict,
    mode: str = 'weekly',   # 'weekly' | 'monthly'
    initial_capital: float = INITIAL_CAP,
    top_n: int = TOP_N,
    score_threshold: float = SCORE_THRESHOLD,
    stop_loss: float = STOP_LOSS,
) -> pd.DataFrame:
    """
    period-rebalance 回测：
      - 每周一 / 每月第一个交易日：确定换仓日
      - 用上一个换仓日前一天（周五 / 月末）的分数选股
      - 按换仓日开盘买入，持有到下一换仓日开盘卖出
    """
    print(f"\n{'='*55}")
    print(f"  [{mode.upper()}] 周期持仓回测")
    print(f"{'='*55}")

    # 准备价格查找字典  (date, ts_code) -> {'open': ..., 'pct_chg': ...}
    price_df = price_df.sort_values('trade_date')
    price_lookup = {}
    for _, row in price_df.iterrows():
        price_lookup[(row['trade_date'], row['ts_code'])] = {
            'open': row['open'], 'close': row['close'], 'pct_chg': row['pct_chg']
        }

    # 全部交易日列表
    all_trade_dates = sorted(price_df['trade_date'].unique())

    # 确定换仓日（重平衡日）
    date_series = pd.Series(all_trade_dates)
    if mode == 'weekly':
        # 每周一（weekday==0）
        rebal_dates = [d for d in all_trade_dates if d.weekday() == 0]
    else:  # monthly
        # 每月第一个交易日
        rebal_dates = list(
            date_series.groupby(date_series.dt.to_period('M')).first()
        )

    print(f"  换仓日数: {len(rebal_dates)}")

    # 回测状态
    capital   = initial_capital
    positions = {}   # {ts_code: {'shares': x, 'cost': y}}
    equity_curve = []
    trade_log    = []

    for i, rebal_date in enumerate(tqdm(rebal_dates, desc=f"{mode} rebalance")):
        # ---- 大盘趋势过滤 ----
        date_key = rebal_date.strftime('%Y-%m-%d')
        is_bull = market_filter.get(date_key, True)  # 找不到则默认允许

        # A. 卖出（清仓上期持仓）
        for ts_code, pos in list(positions.items()):
            key = (rebal_date, ts_code)
            if key not in price_lookup:
                continue
            close_px = price_lookup[key]['close']
            cur_ret   = (close_px - pos['cost']) / pos['cost']

            # 止损检测（用当期收盘价判断）
            if cur_ret <= stop_loss:
                sell_px = close_px * (1 - SLIPPAGE)  # 触止损，当天收盘出
            else:
                sell_px = price_lookup[key]['open'] * (1 - SLIPPAGE)

            if sell_px <= 0:
                continue
            revenue = pos['shares'] * sell_px
            fee = max(5, revenue * COMMISSION) + revenue * STAMP_DUTY
            net = revenue - fee
            capital += net
            pnl = net - pos['shares'] * pos['cost']
            trade_log.append({
                'date': rebal_date, 'code': ts_code, 'action': 'SELL',
                'price': sell_px, 'shares': pos['shares'], 'pnl': pnl,
                'trigger': 'stop' if cur_ret <= stop_loss else 'rebal'
            })
        positions = {}

        # B. 选股：用上一换仓日 → 本次换仓日之间，取最新一天分数
        if i == 0:
            score_start = all_trade_dates[0]
        else:
            score_start = rebal_dates[i - 1]
        score_end = rebal_date - pd.Timedelta(days=1)

        mask = (
            (pred_df['trade_date'] >= score_start) &
            (pred_df['trade_date'] <= score_end)
        )
        window_scores = pred_df[mask]
        if window_scores.empty:
            equity_curve.append({'date': rebal_date, 'nav': capital})
            continue

        # 取每只股票在窗口内最新一天的分数
        latest_scores = (
            window_scores.sort_values('trade_date')
            .groupby('ts_code')
            .last()
            .reset_index()
        )
        # 过滤低分
        candidates = latest_scores[latest_scores['dragon_ai_score'] >= score_threshold]
        candidates = candidates.sort_values('dragon_ai_score', ascending=False).head(top_n)

        # C. 买入（大盘下行期禁止买入）
        if len(candidates) > 0 and is_bull:
            slot_value = capital / top_n
            for _, row in candidates.iterrows():
                ts_code = row['ts_code']
                key = (rebal_date, ts_code)
                if key not in price_lookup:
                    continue
                px = price_lookup[key]['open']
                pct_chg = price_lookup[key].get('pct_chg', 0)
                # 涨停无法买入
                if pct_chg > 9.7:
                    continue
                buy_px = px * (1 + SLIPPAGE)
                if buy_px <= 0:
                    continue
                shares = int(slot_value / buy_px / 100) * 100
                if shares < 100:
                    continue
                cost = shares * buy_px
                fee  = max(5, cost * COMMISSION)
                if capital >= cost + fee:
                    capital -= (cost + fee)
                    positions[ts_code] = {'shares': shares, 'cost': buy_px}
                    trade_log.append({
                        'date': rebal_date, 'code': ts_code, 'action': 'BUY',
                        'price': buy_px, 'shares': shares, 'pnl': 0
                    })

        # D. 记录每日净值（用收盘价估算持仓市值）
        holding_val = sum(
            pos['shares'] * price_lookup.get((rebal_date, code), {}).get('close', pos['cost'])
            for code, pos in positions.items()
        )
        equity_curve.append({'date': rebal_date, 'nav': capital + holding_val})

    eq_df = pd.DataFrame(equity_curve)
    tr_df = pd.DataFrame(trade_log)

    # === 汇总统计 ===
    final_nav = eq_df.iloc[-1]['nav'] if len(eq_df) > 0 else initial_capital
    total_ret  = (final_nav - initial_capital) / initial_capital
    eq_df['max_nav'] = eq_df['nav'].cummax()
    eq_df['dd'] = (eq_df['max_nav'] - eq_df['nav']) / eq_df['max_nav']
    max_dd = eq_df['dd'].max()

    sells = tr_df[tr_df['action'] == 'SELL'] if len(tr_df) > 0 else pd.DataFrame()
    win_rate = (len(sells[sells['pnl'] > 0]) / len(sells)) if len(sells) > 0 else 0
    n_trades = len(sells)

    print(f"\n  初始本金:  {initial_capital:>12,.0f}")
    print(f"  期末净值:  {final_nav:>12,.0f}")
    print(f"  总收益率:  {total_ret*100:>+.2f}%")
    print(f"  最大回撤:  {max_dd*100:.2f}%")
    print(f"  换仓次数:  {len(rebal_dates)}")
    print(f"  完成交易:  {n_trades} 笔")
    print(f"  胜率:      {win_rate*100:.1f}%")

    return eq_df, tr_df


# ============================================================
# 可视化
# ============================================================
def plot_results(weekly_eq, monthly_eq):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Strategy Performance: Weekly vs Monthly Rebalancing\n(Based on Dragon AI Score)', 
                  fontsize=14, fontweight='bold', y=0.98)

    # -- 颜色方案 --
    colors = {'weekly': '#00C7B7', 'monthly': '#F4A548'}
    
    for ax, (eq_df, label, color) in zip(axes, [
        (weekly_eq,  'Weekly Rebalance (每周调仓)', colors['weekly']),
        (monthly_eq, 'Monthly Rebalance (每月调仓)', colors['monthly']),
    ]):
        eq_df = eq_df.copy()
        eq_df['date'] = pd.to_datetime(eq_df['date'])
        eq_df = eq_df.sort_values('date')
        
        nav_norm = eq_df['nav'] / INITIAL_CAP
        ret = (eq_df.iloc[-1]['nav'] - INITIAL_CAP) / INITIAL_CAP * 100
        eq_df['max_nav'] = eq_df['nav'].cummax()
        eq_df['dd'] = (eq_df['max_nav'] - eq_df['nav']) / eq_df['max_nav']
        max_dd = eq_df['dd'].max()

        ax.fill_between(eq_df['date'], nav_norm, 1.0, 
                        where=(nav_norm < 1.0), alpha=0.15, color='red')
        ax.fill_between(eq_df['date'], nav_norm, 1.0, 
                        where=(nav_norm >= 1.0), alpha=0.15, color=color)
        ax.plot(eq_df['date'], nav_norm, color=color, linewidth=2, label=label)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, label='Cost Basis')
        
        ax.set_title(f'{label}    总收益率: {ret:+.2f}%  |  最大回撤: {max_dd*100:.2f}%',
                     fontsize=11, pad=8)
        ax.set_ylabel('NAV (normalized)', fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.tick_params(axis='x', rotation=30)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    out_path = r'C:\Users\liuqi\quant_system_v2\weekly_monthly_backtest.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存至: {out_path}")
    return out_path


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    # 1. 加载预测数据
    pred_df = load_predictions()
    start_date = pred_df['trade_date'].min().strftime('%Y%m%d')
    end_date   = pred_df['trade_date'].max().strftime('%Y%m%d')

    # 2. 加载价格数据
    price_df = load_price_data(start_date, end_date)

    if price_df.empty:
        print("\n⚠️ 价格数据为空，无法进行回测！")
        sys.exit(1)

    # 2.5 加载大盘过滤
    market_filter = load_index_filter()
    bull_days = sum(1 for v in market_filter.values() if v)
    print(f"  大盘过滤: {bull_days}/{len(market_filter)} 天处于 MA20 多头区间")

    # 3. 运行回测
    weekly_eq,  weekly_trades  = run_backtest(pred_df, price_df, market_filter, mode='weekly')
    monthly_eq, monthly_trades = run_backtest(pred_df, price_df, market_filter, mode='monthly')

    # 4. 生成图表
    out_path = plot_results(weekly_eq, monthly_eq)

    # 5. 保存交易记录
    weekly_trades.to_csv(r'C:\Users\liuqi\quant_system_v2\trades_weekly.csv', index=False)
    monthly_trades.to_csv(r'C:\Users\liuqi\quant_system_v2\trades_monthly.csv', index=False)
    print("交易记录已保存。")
