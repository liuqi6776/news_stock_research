"""
特征优化器 - 寻找最有效的因子组合
包含：因子IC分析、因子相关性、因子重要性评估
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression


class FeatureAnalyzer:
    """特征分析器"""
    
    def __init__(self, df: pd.DataFrame, forward_return_col: str = 'forward_return'):
        """
        Parameters:
        -----------
        df : DataFrame
            包含因子和前瞻收益的数据
        forward_return_col : str
            前瞻收益列名
        """
        self.df = df
        self.forward_return_col = forward_return_col
        self.factor_cols = [c for c in df.columns if c not in 
                           ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 
                            'vol', 'amount', forward_return_col]]
    
    def calculate_ic(self, factor_col: str, method: str = 'spearman') -> Tuple[float, float]:
        """
        计算因子IC值（信息系数）
        
        Returns:
        --------
        ic : float
            IC值
        p_value : float
            p值
        """
        valid_data = self.df[[factor_col, self.forward_return_col]].dropna()
        
        if len(valid_data) < 30:
            return 0, 1
        
        if method == 'spearman':
            ic, p_value = stats.spearmanr(valid_data[factor_col], valid_data[self.forward_return_col])
        else:
            ic, p_value = stats.pearsonr(valid_data[factor_col], valid_data[self.forward_return_col])
        
        return ic, p_value
    
    def calculate_ic_series(self, factor_col: str, date_col: str = 'trade_date') -> pd.DataFrame:
        """
        计算因子IC时间序列
        """
        ic_series = []
        
        for date, group in self.df.groupby(date_col):
            valid_data = group[[factor_col, self.forward_return_col]].dropna()
            if len(valid_data) >= 10:
                ic, p_value = stats.spearmanr(valid_data[factor_col], valid_data[self.forward_return_col])
                ic_series.append({
                    'date': date,
                    'ic': ic,
                    'p_value': p_value,
                    'n_samples': len(valid_data)
                })
        
        return pd.DataFrame(ic_series)
    
    def calculate_factor_stats(self, factor_col: str) -> Dict:
        """
        计算因子统计指标
        """
        ic, p_value = self.calculate_ic(factor_col)
        ic_series = self.calculate_ic_series(factor_col)
        
        if len(ic_series) == 0:
            return {
                'factor': factor_col,
                'ic_mean': 0,
                'ic_std': 0,
                'ir': 0,
                'ic_positive_ratio': 0,
                't_stat': 0,
                'p_value': 1
            }
        
        ic_mean = ic_series['ic'].mean()
        ic_std = ic_series['ic'].std()
        ir = ic_mean / ic_std if ic_std > 0 else 0
        ic_positive_ratio = (ic_series['ic'] > 0).mean()
        t_stat, p_val = stats.ttest_1samp(ic_series['ic'].dropna(), 0)
        
        return {
            'factor': factor_col,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ir': ir,
            'ic_positive_ratio': ic_positive_ratio,
            't_stat': t_stat,
            'p_value': p_val
        }
    
    def analyze_all_factors(self) -> pd.DataFrame:
        """
        分析所有因子
        """
        results = []
        for factor in self.factor_cols:
            stats = self.calculate_factor_stats(factor)
            results.append(stats)
        
        return pd.DataFrame(results).sort_values('ir', ascending=False)
    
    def calculate_factor_correlation(self) -> pd.DataFrame:
        """
        计算因子相关性矩阵
        """
        return self.df[self.factor_cols].corr()
    
    def select_factors_by_correlation(self, max_corr: float = 0.7) -> List[str]:
        """
        基于相关性筛选因子
        
        Parameters:
        -----------
        max_corr : float
            最大允许相关性
        """
        corr_matrix = self.calculate_factor_correlation()
        
        # 按IR排序
        factor_stats = self.analyze_all_factors()
        sorted_factors = factor_stats['factor'].tolist()
        
        selected = []
        for factor in sorted_factors:
            if not selected:
                selected.append(factor)
            else:
                # 检查与已选因子的相关性
                max_correlation = corr_matrix.loc[factor, selected].abs().max()
                if max_correlation < max_corr:
                    selected.append(factor)
        
        return selected
    
    def calculate_mutual_info(self, factor_col: str) -> float:
        """
        计算互信息
        """
        valid_data = self.df[[factor_col, self.forward_return_col]].dropna()
        
        if len(valid_data) < 30:
            return 0
        
        mi = mutual_info_regression(
            valid_data[[factor_col]], 
            valid_data[self.forward_return_col],
            random_state=42
        )[0]
        
        return mi
    
    def xgboost_feature_importance(self) -> pd.DataFrame:
        """
        使用XGBoost计算特征重要性
        """
        X = self.df[self.factor_cols].fillna(0)
        y = self.df[self.forward_return_col]
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练XGBoost
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_scaled, y)
        
        # 获取特征重要性
        importance = model.feature_importances_
        
        return pd.DataFrame({
            'factor': self.factor_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)


class FeatureOptimizer:
    """特征优化器"""
    
    def __init__(self, analyzer: FeatureAnalyzer):
        self.analyzer = analyzer
    
    def optimize_factor_weights(self, top_n: int = 10) -> Dict[str, float]:
        """
        优化因子权重
        
        Returns:
        --------
        weights : Dict[str, float]
            因子权重字典
        """
        # 获取因子统计
        factor_stats = self.analyzer.analyze_all_factors()
        
        # 选择Top N因子
        top_factors = factor_stats.head(top_n)
        
        # 基于IR计算权重
        ir_values = top_factors['ir'].values
        ir_values = np.maximum(ir_values, 0)  # 只使用正IR的因子
        
        if ir_values.sum() == 0:
            return {}
        
        weights = ir_values / ir_values.sum()
        
        return dict(zip(top_factors['factor'], weights))
    
    def create_composite_factor(self, weights: Dict[str, float]) -> pd.Series:
        """
        创建复合因子
        """
        composite = pd.Series(0, index=self.analyzer.df.index)
        
        for factor, weight in weights.items():
            if factor in self.analyzer.df.columns:
                # 标准化
                factor_values = self.analyzer.df[factor]
                factor_std = (factor_values - factor_values.rolling(60).mean()) / factor_values.rolling(60).std()
                composite += factor_std * weight
        
        return composite
    
    def evaluate_factor_combination(self, factors: List[str]) -> Dict:
        """
        评估因子组合效果
        """
        # 创建等权组合
        composite = pd.Series(0, index=self.analyzer.df.index)
        
        for factor in factors:
            if factor in self.analyzer.df.columns:
                factor_values = self.analyzer.df[factor]
                factor_std = (factor_values - factor_values.rolling(60).mean()) / factor_values.rolling(60).std()
                composite += factor_std
        
        composite = composite / len(factors)
        
        # 计算组合因子的IC
        valid_data = pd.DataFrame({
            'composite': composite,
            'forward_return': self.analyzer.df[self.analyzer.forward_return_col]
        }).dropna()
        
        if len(valid_data) < 30:
            return {'ic_mean': 0, 'ic_std': 0, 'ir': 0}
        
        ic, _ = stats.spearmanr(valid_data['composite'], valid_data['forward_return'])
        
        # 计算IC序列
        ic_series = []
        for date, group in self.analyzer.df.groupby('trade_date'):
            valid = group[[factors[0], self.analyzer.forward_return_col]].dropna()
            if len(valid) >= 10:
                ic_val, _ = stats.spearmanr(valid[factors[0]], valid[self.analyzer.forward_return_col])
                ic_series.append(ic_val)
        
        ic_series = pd.Series(ic_series)
        
        return {
            'ic_mean': ic_series.mean(),
            'ic_std': ic_series.std(),
            'ir': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
            'n_factors': len(factors)
        }


def run_feature_analysis(df: pd.DataFrame, forward_return_col: str = 'forward_return') -> Dict:
    """
    运行完整的特征分析
    
    Parameters:
    -----------
    df : DataFrame
        包含因子和前瞻收益的数据
    forward_return_col : str
        前瞻收益列名
    
    Returns:
    --------
    results : Dict
        分析结果
    """
    print("=" * 80)
    print("特征分析开始")
    print("=" * 80)
    
    # 初始化分析器
    analyzer = FeatureAnalyzer(df, forward_return_col)
    optimizer = FeatureOptimizer(analyzer)
    
    # 1. 计算所有因子的IC统计
    print("\n1. 计算因子IC统计...")
    factor_stats = analyzer.analyze_all_factors()
    print(f"   共分析 {len(factor_stats)} 个因子")
    print(f"   Top 5 因子:")
    for _, row in factor_stats.head(5).iterrows():
        print(f"     {row['factor']}: IR={row['ir']:.3f}, IC_mean={row['ic_mean']:.3f}")
    
    # 2. 因子相关性分析
    print("\n2. 计算因子相关性...")
    corr_matrix = analyzer.calculate_factor_correlation()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    print(f"   发现 {len(high_corr_pairs)} 对高相关性因子 (|r| > 0.8)")
    
    # 3. 筛选低相关性因子
    print("\n3. 筛选低相关性因子...")
    selected_factors = analyzer.select_factors_by_correlation(max_corr=0.7)
    print(f"   筛选后保留 {len(selected_factors)} 个因子")
    
    # 4. XGBoost特征重要性
    print("\n4. XGBoost特征重要性...")
    xgb_importance = analyzer.xgboost_feature_importance()
    print(f"   Top 5 重要因子:")
    for _, row in xgb_importance.head(5).iterrows():
        print(f"     {row['factor']}: {row['importance']:.4f}")
    
    # 5. 优化因子权重
    print("\n5. 优化因子权重...")
    optimal_weights = optimizer.optimize_factor_weights(top_n=10)
    print(f"   最优权重:")
    for factor, weight in list(optimal_weights.items())[:5]:
        print(f"     {factor}: {weight:.3f}")
    
    # 6. 评估组合效果
    print("\n6. 评估因子组合效果...")
    combination_stats = optimizer.evaluate_factor_combination(selected_factors[:5])
    print(f"   组合IR: {combination_stats['ir']:.3f}")
    print(f"   组合IC_mean: {combination_stats['ic_mean']:.3f}")
    
    print("\n" + "=" * 80)
    print("特征分析完成！")
    print("=" * 80)
    
    return {
        'factor_stats': factor_stats,
        'correlation_matrix': corr_matrix,
        'selected_factors': selected_factors,
        'xgb_importance': xgb_importance,
        'optimal_weights': optimal_weights,
        'combination_stats': combination_stats
    }


if __name__ == "__main__":
    # 示例用法
    print("特征优化器已加载")
    print("使用方法:")
    print("  from feature_optimizer import run_feature_analysis")
    print("  results = run_feature_analysis(your_dataframe)")
