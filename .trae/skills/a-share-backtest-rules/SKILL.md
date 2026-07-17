---
name: "a-share-backtest-rules"
description: "Strict A-share market backtesting rules and validation checklist. Invoke when writing or reviewing any stock backtesting code for Chinese A-share markets to prevent unrealistic simulation results."
---

# A-Share Backtesting Rules (A股回测规范)

## 强制规则 (MANDATORY)

所有A股回测代码必须遵守以下规则，否则回测结果无效。

### 1. 涨跌停限制 (Price Limit Rules)

**主板/中小板 (60/00/002/003开头):**
- 涨停幅度: +10%
- 跌停幅度: -10%
- ST股票: ±5%

**创业板 (300/301开头):**
- 涨停幅度: +20%
- 跌停幅度: -20%

**科创板 (688/689开头):**
- 涨停幅度: +20%
- 跌停幅度: -20%

**北交所 (8/43开头):**
- 涨停幅度: +30%
- 跌停幅度: -30%

#### 买入限制 (Buy Restrictions)
```python
# T+1开盘相对T日收盘涨幅 >= limit_pct - 0.5% 时，不能买入
# 原因: 涨停开盘买不到
t1_open_chg = (t1_open - t0_close) / t0_close * 100
if t1_open_chg >= (limit_pct - 0.5):
    skip_buy = True  # 跳过此交易
```

#### 卖出限制 (Sell Restrictions)
```python
# T+2最低价相对T+1开盘价跌幅 <= -(limit_pct - 0.5%) 时，按跌停价卖出
# 原因: 跌停时无法按收盘价卖出
t2_low_chg = (t2_low - t1_open) / t1_open * 100
if t2_low_chg <= -(limit_pct - 0.5):
    # 按跌停价或开盘价卖出（取更低）
    sell_price = min(t2_open, t1_open * (1 - limit_pct/100))
else:
    sell_price = t2_close
```

### 2. 板块过滤 (Board Filtering)

**必须过滤的板块:**
- 创业板 (300/301开头) — 除非策略明确针对创业板
- 科创板 (688/689开头) — 除非策略明确针对科创板
- ST股票 — 高风险，建议过滤
- 新股/次新股 — 上市不满60天建议过滤

```python
# 过滤代码示例
def should_filter_stock(ts_code, listing_date=None):
    """返回True表示应该过滤此股票"""
    if ts_code.startswith('300') or ts_code.startswith('301'):
        return True  # 过滤创业板
    if ts_code.startswith('688') or ts_code.startswith('689'):
        return True  # 过滤科创板
    if ts_code.startswith('8') or ts_code.startswith('43'):
        return True  # 过滤北交所
    # 可以添加ST检查、上市时间检查等
    return False
```

### 3. 交易时间规则 (Trading Time Rules)

**T+1交易制度:**
- T日买入的股票，T+1日才能卖出
- 回测中必须确保买入日和卖出日至少间隔1个交易日

**数据使用规则:**
- 预测T+1日走势时，只能使用T日及之前的数据
- 严禁使用未来数据 (Lookahead Bias)
- 特征计算必须使用滚动窗口，不能包含未来信息

### 4. 收益合理性检查 (Return Validation)

**单日收益上限:**
- 主板: 最高约20% (买入后涨停，次日涨停卖出，但这种情况极少)
- 实际合理上限: 10-15% (考虑交易费用)
- **任何单笔收益 > 20% 都应被视为异常并检查**

**检查代码:**
```python
def validate_returns(trades_df):
    """检查收益是否合理"""
    issues = []

    # 检查超高收益
    over_20pct = trades_df[trades_df['ret'] > 0.20]
    if len(over_20pct) > 0:
        issues.append(f"发现{len(over_20pct)}笔收益>20%的交易，这在A股主板不可能发生")

    # 检查创业板股票
    cyb = trades_df[trades_df['ts_code'].str.startswith('300') |
                    trades_df['ts_code'].str.startswith('301')]
    if len(cyb) > 0:
        issues.append(f"发现{len(cyb)}笔创业板交易")

    # 检查科创板股票
    kcb = trades_df[trades_df['ts_code'].str.startswith('688') |
                    trades_df['ts_code'].str.startswith('689')]
    if len(kcb) > 0:
        issues.append(f"发现{len(kcb)}笔科创板交易")

    return issues
```

### 5. 交易费用 (Transaction Costs)

**必须包含的费用:**
- 佣金: 0.025% (双向)
- 印花税: 0.05% (卖出时)
- 过户费: 0.001% (双向)
- **合计约 0.15-0.2% 单边，0.3-0.4% 双边**

```python
COST_RATE = 0.003  # 保守估计 0.3% 双边

# 计算收益时扣除费用
ret = sell_price / buy_price - 1 - COST_RATE
```

### 6. 滑点 (Slippage)

**建议设置:**
- 买入滑点: +0.1% ~ +0.3%
- 卖出滑点: -0.1% ~ -0.3%

```python
SLIPPAGE = 0.002  # 0.2% 滑点

buy_price = t1_open * (1 + SLIPPAGE)
sell_price = t2_close * (1 - SLIPPAGE)
```

## 回测代码模板 (Backtest Template)

```python
import pandas as pd
import numpy as np

# ============ 配置 ============
COST_RATE = 0.003       # 交易费用 0.3%
SLIPPAGE = 0.002        # 滑点 0.2%
LIMIT_THRESHOLD = 0.5   # 涨跌停阈值偏移 (10% -> 9.5%)

# 板块限制
SKIP_CYB = True         # 过滤创业板
SKIP_KCB = True         # 过滤科创板
SKIP_ST = True          # 过滤ST

# ============ 辅助函数 ============
def get_limit_pct(ts_code):
    """获取股票的涨跌停幅度"""
    if ts_code.startswith('300') or ts_code.startswith('301'):
        return 20.0  # 创业板
    elif ts_code.startswith('688') or ts_code.startswith('689'):
        return 20.0  # 科创板
    elif ts_code.startswith('8') or ts_code.startswith('43'):
        return 30.0  # 北交所
    else:
        return 10.0  # 主板/中小板

def should_skip_stock(ts_code):
    """检查是否应该跳过此股票"""
    if SKIP_CYB and (ts_code.startswith('300') or ts_code.startswith('301')):
        return True
    if SKIP_KCB and (ts_code.startswith('688') or ts_code.startswith('689')):
        return True
    # 可以添加ST检查、停牌检查等
    return False

def check_limit_up(t1_open, t0_close, limit_pct):
    """检查是否涨停开盘，涨停则不能买入"""
    open_chg = (t1_open - t0_close) / t0_close * 100
    return open_chg >= (limit_pct - LIMIT_THRESHOLD)

def check_limit_down(t2_low, t1_open, limit_pct):
    """检查T+2是否跌停，跌停则按跌停价卖出"""
    low_chg = (t2_low - t1_open) / t1_open * 100
    return low_chg <= -(limit_pct - LIMIT_THRESHOLD)

def calculate_return(buy_price, sell_price, cost_rate=COST_RATE):
    """计算扣除费用后的收益"""
    return sell_price / buy_price - 1 - cost_rate

# ============ 回测主循环 ============
def backtest_a_share(model, features_df, price_data, dates):
    """
    A股回测主函数

    参数:
        model: 预测模型
        features_df: 特征DataFrame
        price_data: 价格数据字典 {date: price_df}
        dates: 回测日期列表
    """
    trades = []
    equity = [100000.0]

    for i, d in enumerate(dates):
        if i + 2 >= len(dates):
            break

        d_t1 = dates[i + 1]  # T+1
        d_t2 = dates[i + 2]  # T+2

        # 1. 获取预测日的特征
        feat = features_df[features_df['date'] == d]
        if feat.empty:
            equity.append(equity[-1])
            continue

        # 2. 模型预测
        X = feat[feature_cols].fillna(0)
        feat['prob'] = model.predict_proba(X)[:, 1]

        # 3. 选择候选股票
        candidates = feat[feat['prob'] >= PROB_THRESH].sort_values('prob', ascending=False)
        if candidates.empty:
            equity.append(equity[-1])
            continue

        pick = candidates.iloc[0]
        ts_code = pick['ts_code']

        # 4. 板块过滤
        if should_skip_stock(ts_code):
            equity.append(equity[-1])
            continue

        # 5. 获取价格数据
        p_t0 = price_data[d]
        p_t1 = price_data[d_t1]
        p_t2 = price_data[d_t2]

        t0_row = p_t0[p_t0['ts_code'] == ts_code]
        t1_row = p_t1[p_t1['ts_code'] == ts_code]
        t2_row = p_t2[p_t2['ts_code'] == ts_code]

        if t0_row.empty or t1_row.empty or t2_row.empty:
            equity.append(equity[-1])
            continue

        t0_close = float(t0_row['close'].values[0])
        t1_open = float(t1_row['open'].values[0])
        t1_pre = float(t1_row['pre_close'].values[0]) if 'pre_close' in t1_row.columns else t0_close
        t2_close = float(t2_row['close'].values[0])
        t2_low = float(t2_row['low'].values[0])
        t2_open = float(t2_row['open'].values[0]) if 'open' in t2_row.columns else t2_close

        # 6. 获取涨跌停幅度
        limit_pct = get_limit_pct(ts_code)

        # 7. 涨停检查 (不能买入)
        if check_limit_up(t1_open, t1_pre, limit_pct):
            equity.append(equity[-1])
            continue

        # 8. 跌停检查 (按跌停价卖出)
        if check_limit_down(t2_low, t1_open, limit_pct):
            sell_price = min(t2_open, t1_open * (1 - limit_pct/100))
        else:
            sell_price = t2_close

        # 9. 应用滑点
        buy_price = t1_open * (1 + SLIPPAGE)
        sell_price = sell_price * (1 - SLIPPAGE)

        # 10. 计算收益
        ret = calculate_return(buy_price, sell_price)

        # 11. 记录交易
        trades.append({
            'date': d,
            'ts_code': ts_code,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'ret': ret,
        })

        # 12. 更新权益
        new_equity = equity[-1] * (1 + ret)
        equity.append(new_equity)

    return trades, equity

# ============ 结果验证 ============
def validate_backtest_results(trades_df, equity_series):
    """验证回测结果是否合理"""
    issues = []

    # 检查1: 超高收益
    over_20 = trades_df[trades_df['ret'] > 0.20]
    if len(over_20) > 0:
        issues.append(f"❌ 发现{len(over_20)}笔收益>20%的交易")

    # 检查2: 创业板/科创板
    cyb = trades_df[trades_df['ts_code'].str.startswith('300') |
                    trades_df['ts_code'].str.startswith('301')]
    kcb = trades_df[trades_df['ts_code'].str.startswith('688') |
                    trades_df['ts_code'].str.startswith('689')]
    if len(cyb) > 0:
        issues.append(f"⚠️  发现{len(cyb)}笔创业板交易")
    if len(kcb) > 0:
        issues.append(f"⚠️  发现{len(kcb)}笔科创板交易")

    # 检查3: 收益分布合理性
    avg_ret = trades_df['ret'].mean()
    if avg_ret > 0.05:
        issues.append(f"⚠️  平均收益{avg_ret*100:.2f}%过高，请检查")

    # 检查4: 胜率合理性
    win_rate = (trades_df['ret'] > 0).mean()
    if win_rate > 0.8:
        issues.append(f"⚠️  胜率{win_rate*100:.1f}%过高，请检查")

    if not issues:
        print("✅ 回测结果通过验证")
    else:
        print("❌ 回测结果存在问题:")
        for issue in issues:
            print(f"   {issue}")

    return len(issues) == 0
```

## 检查清单 (Checklist)

在提交回测代码前，必须确认以下事项:

- [ ] 涨停开盘的股票被正确过滤 (不能买入)
- [ ] 跌停日的股票按跌停价卖出 (不是收盘价)
- [ ] 创业板股票被过滤 (除非策略允许)
- [ ] 科创板股票被过滤 (除非策略允许)
- [ ] 交易费用已扣除 (至少0.3%双边)
- [ ] 滑点已考虑 (至少0.1%)
- [ ] 没有使用未来数据
- [ ] 买入日和卖出日至少间隔1个交易日
- [ ] 单笔收益不超过20% (主板)
- [ ] 回测结果已验证合理性

## 常见错误 (Common Mistakes)

1. **涨停买入**: 假设可以按涨停价买入，实际买不到
2. **跌停卖出**: 假设可以按收盘价卖出，实际只能按跌停价卖出
3. **忽略交易费用**: 未扣除佣金、印花税等
4. **未来函数**: 使用未来数据计算特征或标签
5. **板块错误**: 未区分主板/创业板/科创板的涨跌停幅度
6. **停牌处理**: 未处理停牌股票，假设可以正常交易

## 参考标准 (Reference Standards)

**合理的回测结果范围:**
- 年化收益: -20% ~ +50% (超过100%需高度怀疑)
- 夏普比率: -1.0 ~ +2.0 (超过3.0需高度怀疑)
- 最大回撤: 10% ~ 60% (超过80%风险过高)
- 胜率: 40% ~ 65% (超过70%需检查)
- 平均单笔收益: -2% ~ +3%

**如果回测结果远超以上范围，极可能存在回测失真。**
