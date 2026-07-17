"""
高级 T+1 回测引擎 v3
核心逻辑：当天开盘买（T日open），第二天开盘卖（T+1日open）
改进项：
  1. 大盘 MA20 择时过滤 —— 熊市不开新仓
  2. 严格概率阈值（0.65）控制出手频率
  3. 跌停日无法卖出时，延后至次日平仓
  4. 完整手续费模型（滑点+佣金+印花税）
  5. 上证指数抬升时才允许开仓
"""
import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 配置
# ============================================================
INITIAL_CAP   = 1_000_000.0
MAX_POSITIONS = 3            # 最多同时持仓数量
PROB_THRESHOLD = 0.65        # 模型胜率阈值，低于此不开仓

SLIPPAGE   = 0.001           # 双边滑点
COMMISSION = 0.0003          # 买卖佣金
STAMP_DUTY = 0.0005          # 卖出印花税（单边）

MARKET_CAP_LIMIT = 5_000_000 # 市值上限（单位万元，即500亿）
SSE_FILE = r'C:\Users\liuqi\quant_system_v2\sse_index_2023.csv'

OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(OUT_DIR, 'data', 'super_dataset.parquet')
DATA_DIR  = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')

# ============================================================
# 1. 大盘 MA20 择时过滤
# ============================================================
def load_market_filter() -> dict:
    """返回 {日期字符串YYYYMMDD: True/False} 表示是否处于多头市场"""
    result = {}
    
    # 尝试从价格数据目录加载上证指数日K
    # 文件名规律：data_day1 中以 000001.SH 为 ts_code 的条目
    try:
        all_files = sorted([f for f in os.listdir(PRICE_DIR) if f.endswith('.parquet')])
        idx_rows = []
        for fname in all_files:
            try:
                pdf = pd.read_parquet(os.path.join(PRICE_DIR, fname), columns=['ts_code', 'close'])
                row = pdf[pdf['ts_code'] == '000001.SH']
                if not row.empty:
                    dt = fname.replace('.parquet', '')
                    idx_rows.append({'trade_date': dt, 'close': float(row.iloc[0]['close'])})
            except Exception:
                continue
        
        if not idx_rows:
            raise ValueError("No SSE data in price dir, fallback to CSV")
        
        idx = pd.DataFrame(idx_rows).sort_values('trade_date')
        idx['ma20'] = idx['close'].rolling(20).mean()
        idx['is_bull'] = idx['close'] > idx['ma20']
        idx['is_bull'] = idx['is_bull'].bfill().fillna(True)
        result = dict(zip(idx['trade_date'], idx['is_bull']))
        print(f"[Macro Filter] 从价格数据加载上证指数，共 {len(result)} 天，"
              f"多头占比 {100*sum(result.values())/len(result):.1f}%")
        return result
    except Exception as e:
        pass

    # 降级：从 CSV 加载
    if os.path.exists(SSE_FILE):
        idx = pd.read_csv(SSE_FILE)
        date_col = 'trade_date' if 'trade_date' in idx.columns else 'date'
        idx[date_col] = pd.to_datetime(idx[date_col].astype(str)).dt.strftime('%Y%m%d')
        idx = idx.sort_values(date_col)
        idx['ma20'] = idx['close'].rolling(20).mean()
        idx['is_bull'] = idx['close'] > idx['ma20']
        idx['is_bull'] = idx['is_bull'].bfill().fillna(True)
        result = dict(zip(idx[date_col], idx['is_bull']))
        print(f"[Macro Filter] 从 CSV 加载上证指数，共 {len(result)} 天")
        return result
    
    print("[Macro Filter] 警告：未找到上证指数数据，大盘过滤禁用")
    return {}


market_filter = load_market_filter()


# ============================================================
# 2. 价格缓存（只加载 2024 年以后）
# ============================================================
price_cache = {}

def load_price_cache():
    print("加载价格缓存...")
    files = sorted([f for f in os.listdir(PRICE_DIR)
                    if f.endswith('.parquet') and f >= '20240101.parquet'])
    for f in tqdm(files, desc="Caching Prices"):
        dt = f.replace('.parquet', '')
        pdf = pd.read_parquet(
            os.path.join(PRICE_DIR, f),
            columns=['ts_code', 'open', 'high', 'low', 'close', 'pre_close']
        )
        price_cache[dt] = pdf.set_index('ts_code')


# ============================================================
# 3. 事件驱动回测引擎（严格 T+1：当天开盘买，第二天开盘卖）
# ============================================================
class Backtester:
    def __init__(self):
        self.cash = INITIAL_CAP
        # 待卖出队列: {code: {'shares': int, 'entry_price': float, 'buy_date': str}}
        self.pending_sells = {}
        self.nav_history   = []

    def _round_shares(self, price, alloc):
        """按资金和价格计算可买手数（100股整数倍），返回 (股数, 实际成本+手续费)"""
        max_gross = alloc / (1 + SLIPPAGE + COMMISSION)
        shares = int(max_gross // price) * 1  # 先取整股
        shares = (shares // 100) * 100         # 再取整手
        if shares == 0:
            return 0, 0.0
        cost = shares * price
        fee  = cost * (SLIPPAGE + COMMISSION)
        return shares, cost + fee

    def process_day(self, date: str, next_date, signal_df: pd.DataFrame):
        """
        date      : 今天（T日），在今天开盘买
        next_date : 明天（T+1日），昨天买入的仓位今天开盘卖
        signal_df : 模型预测结果，含 ts_code, prob
        """
        today_prices = price_cache.get(date)
        next_prices  = price_cache.get(next_date) if next_date else None

        # A. 先卖出昨天（T-1日）建仓的所有持仓 （在 today_prices 中查卖出开盘价）
        sold_val = 0.0
        still_pending = {}
        for code, pos in list(self.pending_sells.items()):
            if today_prices is None or code not in today_prices.index:
                # 停牌或无数据，延后至下一个有数据的日期
                still_pending[code] = pos
                continue
            p_data = today_prices.loc[code]
            p_open, p_close, p_pre_close = float(p_data['open']), float(p_data['close']), float(p_data['pre_close'])
            if pd.isna(p_open) or p_open <= 0.01:
                still_pending[code] = pos
                continue

            # 跌停封板，早盘无法卖出：延后
            limit_down = round(p_pre_close * (0.80 if code.startswith(('300', '68')) else 0.90), 2)
            if p_open <= limit_down and p_close <= limit_down:
                still_pending[code] = pos
                continue

            # 正常卖出：T+1 开盘价卖出
            sell_px = p_open
            gross   = pos['shares'] * sell_px
            fee     = gross * (SLIPPAGE + COMMISSION + STAMP_DUTY)
            self.cash += gross - fee

        self.pending_sells = still_pending

        # B. 当天开盘新建仓位
        open_slots = MAX_POSITIONS - len(self.pending_sells)

        # 大盘择时过滤
        is_bull = market_filter.get(date, True)
        if not is_bull:
            open_slots = 0  # 熊市不开新仓

        if open_slots > 0 and today_prices is not None and not signal_df.empty:
            valid = signal_df[signal_df['prob'] >= PROB_THRESHOLD].copy()
            valid = valid[~valid['ts_code'].isin(self.pending_sells.keys())]

            if not valid.empty:
                valid = valid.sort_values('prob', ascending=False).head(open_slots)
                alloc = (self.cash * 0.99) / max(len(valid), 1)

                for _, row in valid.iterrows():
                    code = row['ts_code']
                    if code not in today_prices.index:
                        continue
                    p_data = today_prices.loc[code]
                    p_open     = float(p_data['open'])
                    p_pre_close = float(p_data['pre_close'])
                    if pd.isna(p_open) or p_open <= 0.01:
                        continue
                    # 开盘涨停，买不到
                    limit_up = round(p_pre_close * (1.195 if code.startswith(('300', '68')) else 1.095), 2)
                    if p_open >= limit_up:
                        continue

                    shares, total_cost = self._round_shares(p_open, alloc)
                    if shares == 0 or self.cash < total_cost:
                        continue
                    self.cash -= total_cost
                    self.pending_sells[code] = {
                        'shares':      shares,
                        'entry_price': p_open,
                        'buy_date':    date
                    }

        # C. 按当天收盘价 Mark-to-Market
        hold_val = 0.0
        for code, pos in self.pending_sells.items():
            if today_prices is not None and code in today_prices.index:
                px = float(today_prices.loc[code, 'close'])
                if not pd.isna(px) and px > 0:
                    hold_val += pos['shares'] * px
                else:
                    hold_val += pos['shares'] * pos['entry_price']
            else:
                hold_val += pos['shares'] * pos['entry_price']

        nav = self.cash + hold_val
        self.nav_history.append({'date': pd.to_datetime(date), 'nav': nav})


# ============================================================
# 4. WFO（滚动重训练）
# ============================================================
def train_model(train_df, features):
    sub = train_df.dropna(subset=['label']).copy()
    if len(sub) < 200:
        return None, None
    X = sub[features].copy()
    if 'hot_rank_pct' in X.columns:
        X['hot_rank_pct'] = X['hot_rank_pct'].fillna(0.5)
    X = X.fillna(0)
    y = sub['label']
    scaler = RobustScaler()
    X_s = scaler.fit_transform(X)
    model = xgb.XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, tree_method='hist', n_jobs=4,
        eval_metric='logloss', scale_pos_weight=1
    )
    model.fit(X_s, y)
    return model, scaler


def run_t1_wfo(df, news_col, label_name):
    print(f"\n>>> 严格 T+1 WFO 回测：{label_name}")
    start_date = '20240101'
    end_date   = '20260331'

    scope = df[(df['buy_date'] >= start_date) & (df['buy_date'] <= end_date)].copy()
    test_dates = sorted(scope['buy_date'].unique())
    features   = ['hot_rank_pct', 'chip_concentration', 'winner_rate', news_col]

    backtester       = Backtester()
    current_model    = None
    cur_scaler       = None
    last_train_month = -1

    for i, d_buy in enumerate(tqdm(test_dates, desc=f"  Simulating {label_name}")):
        curr_dt    = pd.to_datetime(d_buy)
        next_date  = test_dates[i + 1] if i + 1 < len(test_dates) else None

        # 按月重训
        if curr_dt.month != last_train_month:
            cutoff = curr_dt.replace(day=1).strftime('%Y%m%d')
            train_data = df[df['buy_date'] < cutoff].copy()
            if not train_data.empty:
                m, s = train_model(train_data, features)
                if m is not None:
                    current_model, cur_scaler = m, s
                    print(f"  [Retrain] {d_buy}，训练样本数: {len(train_data):,}", flush=True)
            last_train_month = curr_dt.month

        # 构建今日候选池
        day_data = df[df['buy_date'] == d_buy].copy()
        day_data = day_data[
            (day_data['circ_mv'] <= MARKET_CAP_LIMIT) &
            (~day_data['ts_code'].str.startswith('688'))
        ]

        if current_model is not None and not day_data.empty:
            feat = day_data[features].copy()
            if 'hot_rank_pct' in feat.columns:
                feat['hot_rank_pct'] = feat['hot_rank_pct'].fillna(0.5)
            X_test = cur_scaler.transform(feat.fillna(0))
            day_data = day_data.copy()
            day_data['prob'] = current_model.predict_proba(X_test)[:, 1]
        else:
            day_data = pd.DataFrame()

        backtester.process_day(d_buy, next_date, day_data)

    return pd.DataFrame(backtester.nav_history)


# ============================================================
# 5. 主程序
# ============================================================
if __name__ == '__main__':
    if not os.path.exists(DATA_PATH):
        print("请先运行 1_build_dataset.py 生成 super_dataset.parquet")
        sys.exit(1)

    load_price_cache()

    df = pd.read_parquet(DATA_PATH)
    print(f"数据集 {len(df):,} 行，日期范围：{df['buy_date'].min()} ~ {df['buy_date'].max()}")

    eq_major  = run_t1_wfo(df, 'news_major_impact',  'News Major（盘后新闻）')
    eq_major1 = run_t1_wfo(df, 'news_major1_impact', 'News Major1（早盘新闻）')

    # 计算关键指标
    def calc_metrics(eq):
        if eq.empty:
            return {}
        eq = eq.sort_values('date').copy()
        final   = eq.iloc[-1]['nav']
        ret     = (final - INITIAL_CAP) / INITIAL_CAP
        eq['peak'] = eq['nav'].cummax()
        eq['dd']   = (eq['peak'] - eq['nav']) / eq['peak']
        mdd     = eq['dd'].max()
        n_years = (eq['date'].iloc[-1] - eq['date'].iloc[0]).days / 365.25
        annret  = (final / INITIAL_CAP) ** (1 / max(n_years, 0.01)) - 1
        return {'final': final, 'total_ret': ret, 'ann_ret': annret, 'mdd': mdd}

    m1 = calc_metrics(eq_major)
    m2 = calc_metrics(eq_major1)

    print(f"\n{'='*50}")
    print(f"News Major（盘后新闻）")
    print(f"  最终净值：{m1.get('final', 0):,.0f}  总收益：{m1.get('total_ret', 0)*100:+.2f}%")
    print(f"  年化收益：{m1.get('ann_ret', 0)*100:+.2f}%  最大回撤：{m1.get('mdd', 0)*100:.2f}%")
    print(f"\nNews Major1（早盘新闻）")
    print(f"  最终净值：{m2.get('final', 0):,.0f}  总收益：{m2.get('total_ret', 0)*100:+.2f}%")
    print(f"  年化收益：{m2.get('ann_ret', 0)*100:+.2f}%  最大回撤：{m2.get('mdd', 0)*100:.2f}%")

    # 绘图
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(eq_major['date'],  eq_major['nav'],  label='News Major（盘后新闻）', color='royalblue', lw=2)
    ax.plot(eq_major1['date'], eq_major1['nav'], label='News Major1（早盘新闻）', color='tomato',    lw=2)
    ax.axhline(INITIAL_CAP, color='gray', ls='--', lw=0.8)
    ax.set_title('严格 T+1 WFO 回测（当天开盘买 → 次日开盘卖）\n大盘 MA20 过滤 | 概率阈值 0.65 | 最多 3 仓', fontsize=13)
    ax.set_xlabel('日期')
    ax.set_ylabel('净值（元）')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, 'advanced_wfo_comparison.png')
    plt.savefig(out_png, dpi=150)
    print(f"\n图表已保存：{out_png}")
