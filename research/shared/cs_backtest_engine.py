"""
截面选股回测引擎 - Cross-Sectional Backtest Engine
支持：月度/周度调仓、截面排名、等权/加权组合、IC/IR计算、分域建模
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class CSBacktestEngine:
    """
    截面选股回测引擎
    
    核心逻辑：
    1. 在每个调仓日，对所有股票计算因子得分
    2. 按得分排序，选取 top N 股票
    3. 等权（或加权）构建组合
    4. 持有到下一个调仓日，计算收益
    5. 重复以上过程
    """
    
    def __init__(self,
                 rebalance_freq: str = 'monthly',  # 'weekly', 'monthly', 'quarterly'
                 top_n: int = 50,
                 weight_method: str = 'equal',  # 'equal', 'score_weighted', 'cap_weighted'
                 cost_rate: float = 0.003,  # 单边交易成本（含佣金+滑点）
                 min_stocks: int = 10,  # 最少选股数量
                 long_short: bool = False,  # 是否支持多空
                 industry_neutral: bool = True,  # 行业中性约束
                 max_industry_pct: float = 0.3,  # 单个行业最大占比
                 ):
        self.rebalance_freq = rebalance_freq
        self.top_n = top_n
        self.weight_method = weight_method
        self.cost_rate = cost_rate
        self.min_stocks = min_stocks
        self.long_short = long_short
        self.industry_neutral = industry_neutral
        self.max_industry_pct = max_industry_pct
        
        self.portfolio_history = []
        self.daily_nav = []
        self.factor_ic = []
        self.factor_ir = []
    
    def get_rebalance_dates(self, all_dates: List[str]) -> List[str]:
        """生成调仓日列表"""
        if self.rebalance_freq == 'monthly':
            # 每月第一个交易日
            rebalance_dates = []
            current_month = None
            for d in all_dates:
                month = d[:6]  # YYYYMM
                if month != current_month:
                    rebalance_dates.append(d)
                    current_month = month
            return rebalance_dates
        elif self.rebalance_freq == 'weekly':
            # 每周第一个交易日（简化：每5个交易日）
            return all_dates[::5]
        elif self.rebalance_freq == 'quarterly':
            # 每季度第一个交易日
            rebalance_dates = []
            current_q = None
            for d in all_dates:
                month = int(d[4:6])
                q = (month - 1) // 3 + 1
                q_str = f"{d[:4]}Q{q}"
                if q_str != current_q:
                    rebalance_dates.append(d)
                    current_q = q_str
            return rebalance_dates
        else:
            return all_dates[::20]  # 默认20日
    
    def calculate_ic(self, factor_scores: pd.Series, future_returns: pd.Series) -> float:
        """计算因子IC（秩相关系数）"""
        valid = ~(factor_scores.isna() | future_returns.isna())
        if valid.sum() < 10:
            return np.nan
        return factor_scores[valid].corr(future_returns[valid], method='spearman')
    
    def select_stocks(self, df: pd.DataFrame, score_col: str, date: str) -> pd.DataFrame:
        """
        截面选股：按得分排序，选取top N
        支持行业中性约束
        """
        day_df = df[df['trade_date'] == date].copy()
        if day_df.empty:
            return pd.DataFrame()
        
        # 基础过滤：剔除ST、停牌、涨停等（由数据预处理完成）
        day_df = day_df.dropna(subset=[score_col])
        
        if len(day_df) < self.min_stocks:
            return pd.DataFrame()
        
        # 按得分排序
        day_df = day_df.sort_values(score_col, ascending=False)
        
        if self.industry_neutral and 'industry' in day_df.columns:
            # 行业中性：每个行业选前K只，但总数不超过top_n
            selected = []
            n_industries = day_df['industry'].nunique()
            k_per_industry = max(1, self.top_n // n_industries)
            
            for industry, group in day_df.groupby('industry'):
                n_select = min(k_per_industry, int(len(group) * self.max_industry_pct) + 1)
                selected.append(group.head(n_select))
            
            selected_df = pd.concat(selected, ignore_index=True)
            selected_df = selected_df.sort_values(score_col, ascending=False).head(self.top_n)
        else:
            selected_df = day_df.head(self.top_n)
        
        return selected_df
    
    def calculate_weights(self, selected_df: pd.DataFrame, score_col: str) -> pd.Series:
        """计算组合权重"""
        if self.weight_method == 'equal':
            n = len(selected_df)
            weights = pd.Series(1.0 / n, index=selected_df.index)
        
        elif self.weight_method == 'score_weighted':
            # 按得分加权
            scores = selected_df[score_col].fillna(0)
            scores = scores - scores.min() + 1e-8  # 确保非负
            weights = scores / scores.sum()
        
        elif self.weight_method == 'cap_weighted':
            # 按市值加权（默认市值越大权重越高，但截面策略通常反向）
            if 'circ_mv' in selected_df.columns:
                cap = selected_df['circ_mv'].fillna(0)
                weights = cap / cap.sum()
            else:
                weights = pd.Series(1.0 / len(selected_df), index=selected_df.index)
        
        else:
            weights = pd.Series(1.0 / len(selected_df), index=selected_df.index)
        
        return weights
    
    def run_backtest(self,
                     df: pd.DataFrame,
                     score_col: str,
                     all_dates: List[str],
                     return_col: str = 'future_ret_20d',
                     industry_col: str = 'industry',
                     benchmark_returns: pd.Series = None) -> Dict:
        """
        运行截面选股回测（调仓日重新选股，持有到下一调仓日）
        """
        rebalance_dates = self.get_rebalance_dates(all_dates)
        
        trades = []
        ics = []
        
        initial_capital = 1000000.0
        capital = initial_capital
        daily_nav = []
        
        for i, reb_date in enumerate(rebalance_dates):
            # 获取当前调仓日数据
            day_df = df[df['trade_date'] == reb_date].copy()
            if day_df.empty:
                continue
            
            # 计算IC（因子值 vs 未来收益）
            if return_col in day_df.columns:
                ic = self.calculate_ic(day_df[score_col], day_df[return_col])
                ics.append({'date': reb_date, 'ic': ic})
            
            # 选股
            selected = self.select_stocks(df, score_col, reb_date)
            if selected.empty:
                continue
            
            # 计算权重
            weights = self.calculate_weights(selected, score_col)
            selected = selected.copy()
            selected['weight'] = weights.values
            
            # 确定持有期结束日
            if i + 1 < len(rebalance_dates):
                next_reb_date = rebalance_dates[i + 1]
            else:
                break  # 最后一个调仓日没有后续收益可计算
            
            # 获取持有期收益（每个股票从当前调仓日到下一调仓日的收益）
            # 使用 return_col 作为持有期收益
            hold_rets = []
            for _, row in selected.iterrows():
                ts_code = row['ts_code']
                weight = row['weight']
                
                # 在当前调仓日查找该股票的未来收益
                stock_data = day_df[day_df['ts_code'] == ts_code]
                if stock_data.empty or return_col not in stock_data.columns:
                    continue
                
                ret = stock_data[return_col].iloc[0]
                if pd.isna(ret):
                    continue
                
                hold_rets.append(ret * weight)
                
                trades.append({
                    'entry_date': reb_date,
                    'exit_date': next_reb_date,
                    'ts_code': ts_code,
                    'weight': weight,
                    'score': row[score_col],
                    'hold_return': ret
                })
            
            if not hold_rets:
                continue
            
            # 组合收益 = 加权平均收益 - 交易成本
            portfolio_ret = sum(hold_rets) - self.cost_rate
            capital *= (1 + portfolio_ret)
            
            # 记录每个交易日的净值（简单线性插值到持有期各天）
            try:
                start_idx = all_dates.index(reb_date)
                end_idx = all_dates.index(next_reb_date)
            except ValueError:
                continue
            
            hold_dates = all_dates[start_idx:end_idx+1]
            if len(hold_dates) > 1:
                # 简单假设：持有期收益在期内均匀分布
                daily_ret = portfolio_ret / (len(hold_dates) - 1)
                for j, d in enumerate(hold_dates[1:], 1):
                    daily_nav.append({
                        'date': d,
                        'nav': capital * (1 - (len(hold_dates) - 1 - j) * daily_ret / (1 + portfolio_ret)),
                        'portfolio_ret': daily_ret if j == len(hold_dates) - 1 else 0  # 只在最后记录完整收益
                    })
                # 修正：最后一个记录应为真实 capital
                if daily_nav:
                    daily_nav[-1]['nav'] = capital
                    daily_nav[-1]['portfolio_ret'] = portfolio_ret
        
        # 计算回测指标
        nav_df = pd.DataFrame(daily_nav)
        if nav_df.empty:
            return self._empty_results()
        
        nav_df = nav_df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
        
        # 总收益
        total_return = nav_df['nav'].iloc[-1] / initial_capital - 1
        
        # 日收益率
        daily_rets = nav_df['nav'].pct_change().dropna()
        
        # 年化收益
        n_days = len(nav_df)
        n_years = n_days / 252
        cagr = (1 + total_return) ** (1 / max(n_years, 1e-8)) - 1 if n_years > 0 else 0
        
        # 夏普比率
        if daily_rets.std() > 0:
            sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)
        else:
            sharpe = 0
        
        # 最大回撤
        cummax = nav_df['nav'].cummax()
        drawdown = (cummax - nav_df['nav']) / cummax
        max_drawdown = drawdown.max()
        
        # 胜率
        win_rate = (daily_rets > 0).mean()
        
        # 基准对比
        if benchmark_returns is not None and not benchmark_returns.empty:
            bench_rets = benchmark_returns.reindex(nav_df['date']).fillna(0)
            bench_cum = (1 + bench_rets).cumprod()
            
            # 超额收益
            excess_rets = daily_rets.values - bench_rets.values[:len(daily_rets)]
            excess_mean = np.mean(excess_rets)
            excess_std = np.std(excess_rets) + 1e-8
            information_ratio = excess_mean / excess_std * np.sqrt(252)
        else:
            bench_cum = None
            information_ratio = 0
        
        # IC统计
        ic_df = pd.DataFrame(ics)
        if not ic_df.empty:
            mean_ic = ic_df['ic'].mean()
            ic_std = ic_df['ic'].std() + 1e-8
            ir = mean_ic / ic_std if ic_std > 0 else 0
            ic_positive_ratio = (ic_df['ic'] > 0).mean()
        else:
            mean_ic = 0
            ir = 0
            ic_positive_ratio = 0
        
        # 月度收益
        nav_df['month'] = nav_df['date'].str[:6]
        monthly = nav_df.groupby('month').last()
        monthly['monthly_ret'] = monthly['nav'].pct_change()
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'information_ratio': information_ratio,
            'mean_ic': mean_ic,
            'ir': ir,
            'ic_positive_ratio': ic_positive_ratio,
            'n_trades': len(trades),
            'n_days': n_days,
            'nav_df': nav_df,
            'drawdown': drawdown,
            'monthly_returns': monthly['monthly_ret'].dropna(),
            'trades': trades,
            'ic_df': ic_df,
            'benchmark_cum': bench_cum
        }
    
    def _empty_results(self) -> Dict:
        """空结果"""
        return {
            'total_return': 0,
            'cagr': 0,
            'sharpe': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'information_ratio': 0,
            'mean_ic': 0,
            'ir': 0,
            'ic_positive_ratio': 0,
            'n_trades': 0,
            'n_days': 0,
            'nav_df': pd.DataFrame(),
            'drawdown': pd.Series(),
            'monthly_returns': pd.Series(),
            'trades': [],
            'ic_df': pd.DataFrame(),
            'benchmark_cum': None
        }
    
    def run_domain_backtest(self,
                           df: pd.DataFrame,
                           score_col: str,
                           all_dates: List[str],
                           domain_col: str = 'market_cap_bin',
                           return_col: str = 'future_ret_20d') -> Dict:
        """
        分域建模回测：在不同子域内独立回测
        """
        domain_results = {}
        
        for domain, group in df.groupby(domain_col):
            if len(group) < 100:
                continue
            
            result = self.run_backtest(group, score_col, all_dates, return_col)
            domain_results[domain] = result
        
        return domain_results


def generate_summary_report(results: Dict, domain_results: Dict = None) -> str:
    """生成回测报告"""
    lines = []
    lines.append("=" * 60)
    lines.append("截面选股策略回测报告")
    lines.append("=" * 60)
    
    lines.append(f"\n总收益: {results['total_return']*100:.2f}%")
    lines.append(f"年化收益(CAGR): {results['cagr']*100:.2f}%")
    lines.append(f"夏普比率: {results['sharpe']:.3f}")
    lines.append(f"最大回撤: {results['max_drawdown']*100:.2f}%")
    lines.append(f"日胜率: {results['win_rate']*100:.2f}%")
    lines.append(f"信息比率(IR): {results['ir']:.3f}")
    lines.append(f"平均IC: {results['mean_ic']:.4f}")
    lines.append(f"IC正率: {results['ic_positive_ratio']*100:.2f}%")
    lines.append(f"交易次数: {results['n_trades']}")
    lines.append(f"回测天数: {results['n_days']}")
    
    if not results['monthly_returns'].empty:
        lines.append(f"\n月度收益统计:")
        lines.append(f"  月胜率: {(results['monthly_returns'] > 0).mean()*100:.2f}%")
        lines.append(f"  平均月收益: {results['monthly_returns'].mean()*100:.2f}%")
        lines.append(f"  月收益标准差: {results['monthly_returns'].std()*100:.2f}%")
    
    if domain_results:
        lines.append(f"\n{'='*60}")
        lines.append("分域回测结果")
        lines.append(f"{'='*60}")
        for domain, res in domain_results.items():
            lines.append(f"\n【{domain}】")
            lines.append(f"  CAGR: {res['cagr']*100:.2f}% | Sharpe: {res['sharpe']:.3f} | MaxDD: {res['max_drawdown']*100:.2f}%")
    
    lines.append(f"\n{'='*60}")
    return "\n".join(lines)
