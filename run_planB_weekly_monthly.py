"""
方案 B：从零开始训练周频 / 月频模型

核心设计：
1. 使用 data_day1/ 的日 K 数据构建特征
2. 标签 = 未来 N 根 K 线（N=5 周频, N=20 月频）的对数收益率是否 > 阈值
3. 滚动训练（Walk-Forward）：每季度重训一次，用前 2 年数据训练
4. 每周/月初根据最新模型分数换仓

特征工程（针对周/月级别优化）：
- 动量因子：过去 20/60 日回报率、相对于指数的超额收益
- 均线系统：5/10/20/60 日均线偏离度（BIAS）
- 趋势强度：ADX、布林带收窄（波动率压缩）
- 量价特征：成交量比（近5日/近20日）、委比
- 行业动量：行业相对强度评分
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
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ============================================================
# 配置
# ============================================================
DATA_DIR   = r'D:\iquant_data\data_v2\data_day1'
SSE_FILE   = r'C:\Users\liuqi\quant_system_v2\sse_index_2023.csv'
OUT_DIR    = r'C:\Users\liuqi\quant_system_v2'

INITIAL_CAP   = 100_000.0
TOP_N         = 5
COMMISSION    = 0.0003
STAMP_DUTY    = 0.0005
SLIPPAGE      = 0.001
STOP_LOSS_W   = -0.10    # 周策略止损 -10%
STOP_LOSS_M   = -0.15    # 月策略止损 -15%

# 训练/测试期设置
DATA_START    = '20210101'
TRAIN_START   = '20210101'
TEST_START    = '20230101'
TEST_END      = '20260101'

# ============================================================
# 数据加载
# ============================================================
def load_price_range(start: str, end: str) -> pd.DataFrame:
    """加载指定日期范围的日 K 数据"""
    files = []
    sd = pd.to_datetime(start)
    ed = pd.to_datetime(end)
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith('.parquet'):
            continue
        ds = fname.replace('.parquet', '')
        try:
            dt = pd.to_datetime(ds)
        except:
            continue
        if sd <= dt <= ed:
            try:
                cols = ['ts_code', 'trade_date', 'open', 'high', 'low',
                        'close', 'vol', 'amount', 'pct_chg']
                tmp = pd.read_parquet(os.path.join(DATA_DIR, fname), columns=cols)
                files.append(tmp)
            except Exception as e:
                pass
    if not files:
        return pd.DataFrame()
    df = pd.concat(files, ignore_index=True)
    df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))
    # 只保留沪深主板
    df = df[df['ts_code'].str.match(r'^(00|60)')]
    df = df.dropna(subset=['close', 'open', 'vol'])
    return df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)


def load_index_filter() -> dict:
    """返回 {date_str: is_bull}"""
    if not os.path.exists(SSE_FILE):
        return {}
    idx = pd.read_csv(SSE_FILE)
    idx['date'] = pd.to_datetime(idx['date'])
    idx = idx.sort_values('date')
    idx['ma20'] = idx['close'].rolling(20).mean()
    idx['is_bull'] = idx['close'] > idx['ma20']
    return dict(zip(idx['date'].dt.strftime('%Y-%m-%d'), idx['is_bull']))


# ============================================================
# 特征工程
# ============================================================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """在截面+时序上构建周/月级别的 Alpha 特征"""
    df = df.copy().sort_values(['ts_code', 'trade_date'])
    g = df.groupby('ts_code')['close']

    # 1. 动量（多窗口）
    for w in [5, 10, 20, 60]:
        df[f'mom_{w}'] = g.transform(lambda x: x / x.shift(w) - 1)

    # 2. 均线偏离度（BIAS）
    for w in [5, 10, 20, 60]:
        ma = g.transform(lambda x: x.rolling(w).mean())
        df[f'bias_{w}'] = (df['close'] - ma) / ma

    # 3. 成交量比
    vol_g = df.groupby('ts_code')['vol']
    df['vol_ratio_5_20'] = (
        vol_g.transform(lambda x: x.rolling(5).mean()) /
        vol_g.transform(lambda x: x.rolling(20).mean()).clip(1e-6)
    )

    # 4. 波动率（历史 20 日年化）
    df['log_ret']   = df.groupby('ts_code')['close'].transform(lambda x: np.log(x / x.shift(1)))
    df['vol_20']    = df.groupby('ts_code')['log_ret'].transform(lambda x: x.rolling(20).std() * np.sqrt(252))

    # 5. 布林带宽度（波动率压缩 = 即将突破）
    ma20  = g.transform(lambda x: x.rolling(20).mean())
    std20 = g.transform(lambda x: x.rolling(20).std())
    df['boll_width'] = (2 * std20) / ma20.clip(1e-6)

    # 6. 量价背离（涨价缩量 = 弱）
    df['price_vol_div'] = (
        df.groupby('ts_code')['pct_chg'].transform(lambda x: x.rolling(5).mean()) /
        df['vol_ratio_5_20'].clip(1e-6)
    )

    # 7. 高低价通道位置
    hi60 = df.groupby('ts_code')['high'].transform(lambda x: x.rolling(60).max())
    lo60 = df.groupby('ts_code')['low'].transform(lambda x: x.rolling(60).min())
    df['channel_pos'] = (df['close'] - lo60) / (hi60 - lo60 + 1e-6)

    # 8. RSI 简化版
    delta = df.groupby('ts_code')['close'].transform(lambda x: x.diff())
    up    = delta.clip(lower=0)
    down  = (-delta).clip(lower=0)
    rs    = (up.groupby(df['ts_code']).transform(lambda x: x.rolling(14).mean()) /
             down.groupby(df['ts_code']).transform(lambda x: x.rolling(14).mean()).clip(1e-6))
    df['rsi_14'] = 100 - 100 / (1 + rs)

    # 9. 截面排名（消除股价绝对值影响）
    for col in ['mom_20', 'mom_60', 'vol_ratio_5_20', 'bias_20', 'channel_pos']:
        df[f'{col}_rank'] = df.groupby('trade_date')[col].rank(pct=True)

    return df


def add_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    标签 = 未来 horizon 天的累积收益 > 阈值
    用 t+1 开盘买入，t+horizon 开盘卖出的实际收益
    """
    df = df.sort_values(['ts_code', 'trade_date'])
    g_open = df.groupby('ts_code')['open']

    # 入场价 = T+1 开盘
    entry = g_open.transform(lambda x: x.shift(-1))
    # 出场价 = T+horizon+1 开盘（持有 horizon 根 K 线）
    exit_  = g_open.transform(lambda x: x.shift(-horizon - 1))

    df[f'ret_{horizon}d'] = (exit_ - entry) / entry.clip(1e-6)

    # 正类 = 超过 0.5% 的收益（去除交易成本后的真实 Alpha 标准）
    df[f'label_{horizon}d'] = (df[f'ret_{horizon}d'] > 0.005).astype(int)
    return df


# ============================================================
# 训练模型
# ============================================================
FEATURE_COLS = [
    'mom_5', 'mom_10', 'mom_20', 'mom_60',
    'bias_5', 'bias_10', 'bias_20', 'bias_60',
    'vol_ratio_5_20', 'vol_20', 'boll_width',
    'price_vol_div', 'channel_pos', 'rsi_14',
    'mom_20_rank', 'mom_60_rank', 'vol_ratio_5_20_rank',
    'bias_20_rank', 'channel_pos_rank'
]

def train_model(train_df: pd.DataFrame, label_col: str):
    sub = train_df.dropna(subset=FEATURE_COLS + [label_col]).copy()
    X = sub[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    y = sub[label_col]

    # 平衡类别
    pos = sub[y == 1]
    neg = sub[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None, None
    n = min(len(pos), len(neg))
    balanced = pd.concat([pos.sample(n, random_state=42), neg.sample(n, random_state=42)])
    X_bal = balanced[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    y_bal = balanced[label_col]

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_bal)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1,
        eval_metric='logloss', tree_method='hist'
    )
    model.fit(X_scaled, y_bal)

    # 验证集 AUC
    X_val = X.replace([np.inf, -np.inf], 0).fillna(0)
    X_val_scaled = scaler.transform(X_val)
    auc = roc_auc_score(y, model.predict_proba(X_val_scaled)[:, 1])
    print(f"      Train AUC={auc:.3f}  |  样本={len(X_val):,}  |  正类比={y.mean():.2%}")
    return model, scaler


# ============================================================
# 回测（Walk-Forward）
# ============================================================
def run_period_backtest(
    df: pd.DataFrame,
    market_filter: dict,
    mode: str = 'weekly',   # 'weekly' | 'monthly'
    horizon: int = 5,
    stop_loss: float = STOP_LOSS_W,
) -> tuple:

    label_col = f'label_{horizon}d'
    print(f"\n{'='*60}")
    print(f"  [{mode.upper()}] 滚动训练 + 周期换仓回测  (horizon={horizon}d)")
    print(f"{'='*60}")

    test_df = df[df['trade_date'] >= TEST_START].copy()
    all_dates = sorted(test_df['trade_date'].unique())

    # 换仓日
    date_s = pd.Series(all_dates)
    if mode == 'weekly':
        rebal_dates = [d for d in all_dates if d.weekday() == 0]
    else:
        rebal_dates = list(date_s.groupby(date_s.dt.to_period('M')).first())

    print(f"  测试换仓日: {len(rebal_dates)} 个")

    # 价格快速查找
    price_map = {}
    for _, row in df.iterrows():
        price_map[(row['trade_date'], row['ts_code'])] = {
            'open': row['open'], 'close': row['close']
        }

    # 回测状态
    capital  = INITIAL_CAP
    positions = {}
    equity_curve = []
    trade_log    = []

    current_model  = None
    current_scaler = None
    last_train_quarter = None

    for i, rebal_date in enumerate(tqdm(rebal_dates, desc=f"{mode} backtest")):

        # ---- 大盘过滤 ----
        date_key = rebal_date.strftime('%Y-%m-%d')
        is_bull  = market_filter.get(date_key, True)

        # ---- 按季度滚动训练 ----
        q = (rebal_date.year, (rebal_date.month - 1) // 3)
        if q != last_train_quarter:
            train_end  = rebal_date - pd.Timedelta(days=1)
            train_start_dt = train_end - pd.Timedelta(days=365 * 2)
            train_mask = (
                (df['trade_date'] >= train_start_dt) &
                (df['trade_date'] <= train_end)
            )
            train_data = df[train_mask]
            if len(train_data.dropna(subset=[label_col])) > 500:
                print(f"\n  [重训] {rebal_date.date()} | 训练窗口: {train_start_dt.date()} ~ {train_end.date()}")
                current_model, current_scaler = train_model(train_data, label_col)
                last_train_quarter = q

        # ---- A. 卖出（清仓上期持仓 + 止损） ----
        for ts_code, pos in list(positions.items()):
            key = (rebal_date, ts_code)
            if key not in price_map:
                continue
            close_px = price_map[key]['close']
            cur_ret  = (close_px - pos['cost']) / pos['cost']

            if cur_ret <= stop_loss:
                sell_px = close_px * (1 - SLIPPAGE)
                trigger = 'stop'
            else:
                sell_px = price_map[key]['open'] * (1 - SLIPPAGE)
                trigger = 'rebal'

            if sell_px <= 0:
                continue
            revenue = pos['shares'] * sell_px
            fee = max(5, revenue * COMMISSION) + revenue * STAMP_DUTY
            net = revenue - fee
            capital += net
            pnl = net - pos['shares'] * pos['cost']
            trade_log.append({
                'date': rebal_date, 'code': ts_code, 'action': 'SELL',
                'price': sell_px, 'pnl': pnl, 'trigger': trigger
            })
        positions = {}

        # ---- B. 预测打分 ----
        if current_model is None or not is_bull:
            # 熊市：空仓观望
            nav = capital
            equity_curve.append({'date': rebal_date, 'nav': nav})
            continue

        day_data = test_df[test_df['trade_date'] == rebal_date]
        if day_data.empty:
            equity_curve.append({'date': rebal_date, 'nav': capital})
            continue

        X_today = day_data[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
        if X_today.empty:
            equity_curve.append({'date': rebal_date, 'nav': capital})
            continue

        X_scaled = current_scaler.transform(X_today)
        probs = current_model.predict_proba(X_scaled)[:, 1]
        day_data = day_data.copy()
        day_data['pred_prob'] = probs

        # 选 Top-N（分数 > 0.60）
        candidates = (
            day_data[day_data['pred_prob'] > 0.60]
            .sort_values('pred_prob', ascending=False)
            .head(TOP_N)
        )

        # ---- C. 买入 ----
        if len(candidates) > 0:
            slot_value = capital / TOP_N
            for _, row in candidates.iterrows():
                ts_code = row['ts_code']
                key = (rebal_date, ts_code)
                if key not in price_map:
                    continue
                buy_px = price_map[key]['open'] * (1 + SLIPPAGE)
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
                        'price': buy_px, 'pnl': 0, 'trigger': 'signal'
                    })

        # ---- D. 记录净值 ----
        hold_val = sum(
            pos['shares'] * price_map.get((rebal_date, code), {}).get('close', pos['cost'])
            for code, pos in positions.items()
        )
        equity_curve.append({'date': rebal_date, 'nav': capital + hold_val})

    eq_df = pd.DataFrame(equity_curve)
    tr_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame(columns=['action', 'pnl'])

    # 汇总
    if len(eq_df) > 0:
        final_nav  = eq_df.iloc[-1]['nav']
        total_ret  = (final_nav - INITIAL_CAP) / INITIAL_CAP
        eq_df['max_nav'] = eq_df['nav'].cummax()
        eq_df['dd']      = (eq_df['max_nav'] - eq_df['nav']) / eq_df['max_nav']
        max_dd = eq_df['dd'].max()
        sells  = tr_df[tr_df['action'] == 'SELL'] if 'action' in tr_df.columns else pd.DataFrame()
        wr     = (len(sells[sells['pnl'] > 0]) / len(sells)) if len(sells) > 0 else 0
        print(f"\n  初始本金:  {INITIAL_CAP:>12,.0f}")
        print(f"  期末净值:  {final_nav:>12,.0f}")
        print(f"  总收益率:  {total_ret*100:>+.2f}%")
        print(f"  最大回撤:  {max_dd*100:.2f}%")
        print(f"  换仓次数:  {len(rebal_dates)}")
        stops = len(tr_df[(tr_df['trigger'] == 'stop')]) if 'trigger' in tr_df.columns else 0
        print(f"  止损触发:  {stops} 次")
        print(f"  胜率:      {wr*100:.1f}%")

    return eq_df, tr_df


# ============================================================
# 可视化
# ============================================================
def plot_results(weekly_eq, monthly_eq):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Plan B: Weekly vs Monthly Strategy\n(Fresh Training with Proper Forward-Return Labels)',
                  fontsize=13, fontweight='bold')

    colors = ['#00C7B7', '#F4A548']
    for ax, (eq_df, label, color) in zip(axes, [
        (weekly_eq,  'Weekly Rebalance  每周调仓 (horizon=5d)', colors[0]),
        (monthly_eq, 'Monthly Rebalance 每月调仓 (horizon=20d)', colors[1]),
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
        ax.plot(eq_df['date'], nav_norm, color=color, linewidth=2.2, label=label)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

        ax.set_title(
            f'{label}\n总收益率: {ret:+.2f}%   最大回撤: {max_dd*100:.2f}%',
            fontsize=11, pad=8
        )
        ax.set_ylabel('NAV (归一化)', fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.tick_params(axis='x', rotation=30)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'planB_weekly_monthly.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: {out}")
    return out


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=== 方案 B: 周/月频重训练回测 ===\n")

    # 1. 加载原始价格数据
    print(f"加载数据 {DATA_START} ~ {TEST_END}...")
    df = load_price_range(DATA_START, TEST_END)
    print(f"  记录数: {len(df):,}  |  股票数: {df.ts_code.nunique()}")

    # 2. 特征工程
    print("\n计算特征...")
    df = build_features(df)

    # 3. 添加两种标签
    print("生成标签 (5日 / 20日)...")
    df = add_labels(df, horizon=5)
    df = add_labels(df, horizon=20)

    # 4. 大盘过滤
    market_filter = load_index_filter()
    print(f"大盘过滤: {sum(v for v in market_filter.values() if v)}/{len(market_filter)} 天多头")

    # 5. 周频回测
    weekly_eq, weekly_tr = run_period_backtest(
        df, market_filter, mode='weekly', horizon=5, stop_loss=STOP_LOSS_W
    )
    weekly_tr.to_csv(os.path.join(OUT_DIR, 'planB_trades_weekly.csv'), index=False)

    # 6. 月频回测
    monthly_eq, monthly_tr = run_period_backtest(
        df, market_filter, mode='monthly', horizon=20, stop_loss=STOP_LOSS_M
    )
    monthly_tr.to_csv(os.path.join(OUT_DIR, 'planB_trades_monthly.csv'), index=False)

    # 7. 绘图
    plot_results(weekly_eq, monthly_eq)
    print("\n完成！")
