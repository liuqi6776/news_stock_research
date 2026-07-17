"""
月度滚动回测框架 (Walk-Forward Analysis)
训练期: 2年 (24个月)
测试期: 1个月
步长: 1个月
总回测期: 2023-2026 (至少3年)
"""
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from factors.technical_factors import calculate_all_factors

# A股交易规则配置
COST_RATE = 0.003       # 交易费用 0.3% 双边
SLIPPAGE = 0.002        # 滑点 0.2%
LIMIT_THRESHOLD = 0.5   # 涨跌停阈值偏移 (10% -> 9.5%)


@dataclass
class WalkForwardResult:
    """月度滚动回测结果"""
    period: str                    # 月份
    train_start: str               # 训练开始
    train_end: str                 # 训练结束
    test_start: str                # 测试开始
    test_end: str                  # 测试结束
    train_samples: int             # 训练样本数
    test_samples: int              # 测试样本数
    train_positive_ratio: float    # 训练集正样本比例
    test_return: float             # 测试期收益
    test_sharpe: float             # 测试期夏普
    test_max_dd: float             # 测试期最大回撤
    n_trades: int                  # 交易次数
    win_rate: float                # 胜率
    avg_return: float              # 平均收益
    skipped_limit_up: int          # 跳过涨停次数
    skipped_limit_down: int        # 跌停卖出次数
    model_auc: float               # 模型AUC
    top_features: List[str]        # 重要特征


class WalkForwardBacktest:
    """月度滚动回测类"""
    
    def __init__(self,
                 data_dir: str = r'D:\iquant_data\data_v2',
                 output_dir: str = None):
        self.data_dir = data_dir
        self.price_dir = os.path.join(data_dir, 'data_day1')
        self.rank_dir = os.path.join(data_dir, 'ths_rank1')
        self.chip_dir = os.path.join(data_dir, 'cyq1')
        self.other_dir = os.path.join(data_dir, 'other_day1')
        
        if output_dir is None:
            self.output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 获取所有交易日
        self.all_dates = sorted([f.replace('.parquet', '') 
                                for f in os.listdir(self.price_dir) 
                                if f.endswith('.parquet')])
    
    def is_main_board(self, ts_code: str) -> bool:
        """检查是否为主板/中小板"""
        return (ts_code.startswith('60') or 
                ts_code.startswith('00') or 
                ts_code.startswith('002') or 
                ts_code.startswith('003'))
    
    def get_limit_pct(self, ts_code: str) -> float:
        """获取涨跌停幅度"""
        if ts_code.startswith('688') or ts_code.startswith('689'):
            return 20.0
        elif ts_code.startswith('30') or ts_code.startswith('301'):
            return 20.0
        elif ts_code.startswith('8') or ts_code.startswith('4'):
            return 30.0
        else:
            return 10.0
    
    def load_features_for_date(self, date_str: str) -> Optional[pd.DataFrame]:
        """加载某一天的特征数据"""
        try:
            # 价格数据
            price_file = os.path.join(self.price_dir, f"{date_str}.parquet")
            if not os.path.exists(price_file):
                return None
            
            price_df = pd.read_parquet(price_file)
            
            # 过滤主板
            price_df = price_df[price_df['ts_code'].apply(self.is_main_board)]
            
            if len(price_df) == 0:
                return None
            
            # 计算基础因子
            price_df = calculate_all_factors(price_df)
            
            # 添加日期
            price_df['trade_date'] = date_str
            
            return price_df
            
        except Exception as e:
            print(f"加载 {date_str} 特征失败: {e}")
            return None
    
    def prepare_training_data(self, 
                             train_dates: List[str],
                             label_threshold: float = 0.04) -> Tuple[pd.DataFrame, List[str]]:
        """
        准备训练数据
        
        Parameters:
        -----------
        train_dates : List[str]
            训练日期列表
        label_threshold : float
            标签阈值 (T+2 close / T+1 open - 1 > threshold)
        """
        all_data = []
        
        for i in range(len(train_dates) - 2):
            d_curr = train_dates[i]
            d_t1 = train_dates[i + 1]
            d_t2 = train_dates[i + 2]
            
            # 加载T日特征
            df_t = self.load_features_for_date(d_curr)
            if df_t is None or len(df_t) == 0:
                continue
            
            # 加载T+1和T+2价格
            p_t1 = os.path.join(self.price_dir, f"{d_t1}.parquet")
            p_t2 = os.path.join(self.price_dir, f"{d_t2}.parquet")
            
            if not os.path.exists(p_t1) or not os.path.exists(p_t2):
                continue
            
            df_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open', 'pre_close'])
            df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'close', 'low', 'open'])
            
            # 合并数据
            df = df_t.merge(df_t1[['ts_code', 'open', 'pre_close']], on='ts_code', how='left')
            df = df.merge(df_t2[['ts_code', 'close', 'low', 'open']], on='ts_code', how='left',
                         suffixes=('', '_t2'))
            
            # 重命名列以便统一处理
            df = df.rename(columns={
                'open': 't1_open',
                'close': 't2_close',
                'low_t2': 't2_low',
                'open_t2': 't2_open'
            })
            
            # 确保需要的列存在
            required_cols = ['t1_open', 't2_close', 't2_low', 't2_open']
            for col in required_cols:
                if col not in df.columns:
                    df[col] = np.nan
            
            # 过滤缺失值
            df = df.dropna(subset=required_cols)
            
            if len(df) == 0:
                continue
            
            # 计算标签: T+2 close / T+1 open - 1 > threshold
            df['label_ret'] = df['t2_close'] / df['t1_open'] - 1
            df['label'] = (df['label_ret'] > label_threshold).astype(int)
            
            all_data.append(df)
        
        if not all_data:
            return pd.DataFrame(), []
        
        result = pd.concat(all_data, ignore_index=True)
        
        # 选择特征列 (排除价格、标签等非特征列)
        exclude_cols = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 
                       'vol', 'amount', 'pre_close', 'label_ret', 'label',
                       't1_open', 't2_close', 't2_low', 't2_open']
        feature_cols = [c for c in result.columns if c not in exclude_cols]
        
        return result, feature_cols
    
    def train_model(self, 
                   train_df: pd.DataFrame,
                   feature_cols: List[str]) -> Tuple[xgb.XGBClassifier, StandardScaler]:
        """
        训练XGBoost模型
        """
        # 准备数据
        X = train_df[feature_cols].fillna(0)
        y = train_df['label']
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练模型
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc',
            n_jobs=-1,
            tree_method='hist'
        )
        
        model.fit(X_scaled, y)
        
        return model, scaler
    
    def backtest_month(self,
                      model: xgb.XGBClassifier,
                      scaler: StandardScaler,
                      feature_cols: List[str],
                      test_dates: List[str],
                      min_prob: float = 0.6) -> Dict:
        """
        回测单月表现
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
            
            # 加载T日特征
            df_t = self.load_features_for_date(d_curr)
            if df_t is None or len(df_t) == 0:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
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
            
            df_t1 = pd.read_parquet(p_t1, columns=['ts_code', 'open', 'pre_close', 'low'])
            df_t2 = pd.read_parquet(p_t2, columns=['ts_code', 'close', 'low', 'open'])
            
            # 获取T+1数据
            t1_data = df_t1[df_t1['ts_code'] == ts_code]
            if t1_data.empty:
                daily_nav.append({'date': d_t2, 'nav': capital})
                continue
            
            t1_open = float(t1_data.iloc[0]['open'])
            t1_pre = float(t1_data.iloc[0]['pre_close'])
            
            # 获取涨跌停幅度
            limit_pct = self.get_limit_pct(ts_code)
            
            # 涨停检查 (不能买入)
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
            
            # 跌停检查 (按跌停价卖出)
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
            
            # 更新资金
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
        
        # 计算统计指标
        if len(trades) == 0:
            return {
                'return': 0,
                'sharpe': 0,
                'max_dd': 0,
                'n_trades': 0,
                'win_rate': 0,
                'avg_return': 0,
                'skipped_limit_up': skipped_limit_up,
                'skipped_limit_down': skipped_limit_down,
                'daily_nav': daily_nav
            }
        
        trades_df = pd.DataFrame(trades)
        
        # 收益
        total_ret = capital / initial_capital - 1
        
        # 夏普比率
        nav_df = pd.DataFrame(daily_nav)
        if len(nav_df) > 1:
            nav_df['ret'] = nav_df['nav'].pct_change()
            vol = nav_df['ret'].std() * np.sqrt(252)
            sharpe = (total_ret / (len(test_dates) / 252)) / vol if vol > 0 else 0
        else:
            sharpe = 0
        
        # 最大回撤
        if len(nav_df) > 0:
            nav_df['cummax'] = nav_df['nav'].cummax()
            nav_df['dd'] = (nav_df['nav'] - nav_df['cummax']) / nav_df['cummax']
            max_dd = nav_df['dd'].min()
        else:
            max_dd = 0
        
        # 胜率
        win_rate = (trades_df['return'] > 0).mean()
        avg_ret = trades_df['return'].mean()
        
        return {
            'return': total_ret,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'n_trades': len(trades),
            'win_rate': win_rate,
            'avg_return': avg_ret,
            'skipped_limit_up': skipped_limit_up,
            'skipped_limit_down': skipped_limit_down,
            'daily_nav': daily_nav,
            'trades': trades_df
        }
    
    def run_walk_forward(self,
                        start_date: str = '20230101',
                        end_date: str = '20260331',
                        train_months: int = 24,
                        test_months: int = 1) -> List[WalkForwardResult]:
        """
        运行月度滚动回测
        
        Parameters:
        -----------
        start_date : str
            回测开始日期
        end_date : str
            回测结束日期
        train_months : int
            训练期月数 (默认24个月=2年)
        test_months : int
            测试期月数 (默认1个月)
        """
        print("=" * 80)
        print("月度滚动回测 (Walk-Forward Analysis)")
        print("=" * 80)
        print(f"回测期: {start_date} 至 {end_date}")
        print(f"训练期: {train_months} 个月")
        print(f"测试期: {test_months} 个月")
        print(f"步长: 1 个月")
        print("=" * 80)
        
        # 过滤日期范围
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
        
        # 滚动回测
        results = []
        all_equity = []
        all_trades = []
        
        for i in range(train_months, len(months), test_months):
            # 训练期
            train_months_list = months[i-train_months:i]
            train_dates = []
            for _, month_dates in train_months_list:
                train_dates.extend(month_dates)
            
            # 测试期
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
            train_df, feature_cols = self.prepare_training_data(train_dates)
            
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
            
            # 回测测试期
            print("回测测试期...")
            backtest_result = self.backtest_month(model, scaler, feature_cols, test_dates)
            
            print(f"测试期收益: {backtest_result['return']*100:.2f}%")
            print(f"夏普比率: {backtest_result['sharpe']:.2f}")
            print(f"最大回撤: {backtest_result['max_dd']*100:.2f}%")
            print(f"交易次数: {backtest_result['n_trades']}")
            print(f"胜率: {backtest_result['win_rate']*100:.2f}%")
            
            # 保存结果
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
                model_auc=0,  # 可以添加AUC计算
                top_features=top_features
            )
            
            results.append(result)
            
            # 收集权益曲线和交易记录
            all_equity.extend(backtest_result['daily_nav'])
            if backtest_result['n_trades'] > 0:
                all_trades.append(backtest_result['trades'])
        
        # 保存所有结果
        self.save_results(results, all_equity, all_trades)
        
        return results
    
    def save_results(self, 
                    results: List[WalkForwardResult],
                    all_equity: List[Dict],
                    all_trades: List[pd.DataFrame]):
        """
        保存回测结果
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 保存月度结果
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
        
        # 2. 保存权益曲线
        if all_equity:
            equity_df = pd.DataFrame(all_equity)
            equity_df = equity_df.drop_duplicates(subset=['date'])
            equity_df = equity_df.sort_values('date')
            
            equity_file = os.path.join(self.output_dir, f'equity_curve_{timestamp}.csv')
            equity_df.to_csv(equity_file, index=False)
            print(f"权益曲线已保存: {equity_file}")
            
            # 计算总体统计
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
        
        # 3. 保存交易记录
        if all_trades:
            trades_df = pd.concat(all_trades, ignore_index=True)
            trades_file = os.path.join(self.output_dir, f'all_trades_{timestamp}.csv')
            trades_df.to_csv(trades_file, index=False)
            print(f"交易记录已保存: {trades_file}")
            
            # 交易统计
            print(f"\n交易统计:")
            print(f"总交易次数: {len(trades_df)}")
            print(f"胜率: {(trades_df['return'] > 0).mean()*100:.2f}%")
            print(f"平均收益: {trades_df['return'].mean()*100:.2f}%")
            print(f"最高单笔: {trades_df['return'].max()*100:.2f}%")
            print(f"最低单笔: {trades_df['return'].min()*100:.2f}%")


def main():
    """
    主函数
    """
    # 初始化回测器
    backtest = WalkForwardBacktest()
    
    # 运行月度滚动回测
    results = backtest.run_walk_forward(
        start_date='20230101',
        end_date='20260331',
        train_months=24,  # 2年训练
        test_months=1     # 1个月测试
    )
    
    print("\n" + "=" * 80)
    print("回测完成！")
    print(f"结果保存在: {backtest.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
