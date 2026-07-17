"""
每日交易信号生成器 V5 - 增强特征版

功能:
1. 获取最新市场数据 (含资金流向、筹码、技术指标)
2. 计算148个增强特征
3. 使用V5模型预测明日推荐股票
4. 根据风控逻辑输出建议
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, r'C:\Users\liuqi\quant_system_v2')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import joblib

from sklearn.preprocessing import StandardScaler
try:
    import xgboost as xgb
except ImportError:
    print("请安装 xgboost: pip install xgboost")
    sys.exit(1)

from config.settings import settings
from infra_data.storage import DataStorage
from features.enhanced_factors import calculate_all_enhanced_features
from features.labeling import simple_return_labeling
from processing.cleaner import filter_stock_codes
from processing.merger import merge_dataframes


class DailySignalV5:
    """V5 增强特征版信号生成器"""
    
    MODEL_PATH = r'C:\Users\liuqi\quant_system_v2\models\v5_xgb_model.pkl'
    SCALER_PATH = r'C:\Users\liuqi\quant_system_v2\models\v5_scaler.pkl'
    FEATURES_PATH = r'C:\Users\liuqi\quant_system_v2\models\v5_features.pkl'
    
    def __init__(self):
        self.storage = DataStorage()
        self.model = None
        self.scaler = None
        self.features = None
    
    def load_recent_data(self, days: int = 120) -> tuple:
        """加载最近N天的数据 (含全量维度)"""
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        print(f"1. 加载数据 {start_date} - {end_date}...")
        
        daily = self.storage.load_daily_data(start_date, end_date)
        if daily.empty:
            raise ValueError("没有找到日线数据，请先运行 data/data_extraction.py 获取最新数据")
            
        other = self.storage.load_daily_basic(start_date, end_date)
        skill = self.storage.load_technical_factors(start_date, end_date)
        money_flow = self.storage.load_money_flow(start_date, end_date)
        chip_data = self.storage.load_chip_data(start_date, end_date)
        
        dfs = [daily]
        if not other.empty: dfs.append(other)
        if not skill.empty: dfs.append(skill)
        
        df = merge_dataframes(dfs)
        df = filter_stock_codes(df, patterns=['^60', '^00'])
        
        return df, money_flow, chip_data
    
    def train_v5_model(self):
        """重新训练V5模型 (同步回测的逻辑)"""
        print("\n" + "=" * 50)
        print("重新训练 V5 模型...")
        print("=" * 50)
        
        # 加载完整历史
        df, money_flow, chip_data = self.load_recent_data(days=365*4)
        
        print("2. 计算增强特征 (148个)...")
        df = calculate_all_enhanced_features(df, money_flow, chip_data)
        
        print("3. 生成训练标签...")
        df = simple_return_labeling(df, forward_days=20, threshold=0.05)
        
        # 获取特征列表 (排除标签和非数值列)
        exclude = ['ts_code', 'trade_date', 'return_label', 'future_return', 
                   'pct_label', 'return_rank', 'tb_label', 'year_month']
        features = [c for c in df.columns if c not in exclude and df[c].dtype not in ['object', 'datetime64[ns]']]
        
        # 准备数据
        train_df = df[df['return_label'].notna()].copy()
        X = train_df[features].fillna(0).replace([np.inf, -np.inf], 0)
        y = (train_df['return_label'] == 1).astype(int)
        
        # 特征选择 (Top 80)
        print("4. 智能特征选择...")
        selector = SelectKBest(f_classif, k=80)
        selector.fit(X, y)
        selected_features = [features[i] for i in selector.get_support(indices=True)]
        
        X_selected = X[selected_features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # 训练
        print("5. 训练 XGBoost 模型...")
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_scaled, y)
        
        # 保存
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        joblib.dump(model, self.MODEL_PATH)
        joblib.dump(scaler, self.SCALER_PATH)
        joblib.dump(selected_features, self.FEATURES_PATH)
        
        print(f"\nV5 模型保存成功: {self.MODEL_PATH}")
        self.model, self.scaler, self.features = model, scaler, selected_features
        return model, scaler, selected_features
    
    def update_data(self):
        """调用外部脚本更新数据"""
        print("\n" + "=" * 50)
        print("正在启动数据更新 (Direct Execution)...")
        print("=" * 50)
        
        try:
            # 直接导入并运行，确保进度条能显示
            from infra_data.fetcher import DataFetcher
            fetcher = DataFetcher()
            # 获取数据 (默认到今天)
            fetcher.fetch_all()
            print("数据更新完成。")
        except ImportError:
            print("Error: 无法导入 data.fetcher，请检查路径。")
        except Exception as e:
            print(f"数据更新运行时出错: {e}")

    def predict_next_day(self, top_n: int = 10):
        """生成预测"""
        if not os.path.exists(self.MODEL_PATH):
            self.train_v5_model()
        else:
            self.model = joblib.load(self.MODEL_PATH)
            self.scaler = joblib.load(self.SCALER_PATH)
            self.features = joblib.load(self.FEATURES_PATH)
        
        df, money_flow, chip_data = self.load_recent_data(days=60)
        latest_date = df['trade_date'].max()
        print(f"最新行情日期: {latest_date}")
        
        # 计算特征
        print("计算昨日特征...")
        df = calculate_all_enhanced_features(df, money_flow, chip_data)
        
        # 提取最新一天
        pred_data = df[df['trade_date'] == latest_date].copy()
        X = pred_data[self.features].fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X)
        
        # 预测
        pred_data['up_proba'] = self.model.predict_proba(X_scaled)[:, 1]
        
        # 结合 V5 的额外评分系统 (资金流 + 筹码)
        if 'mf_signal' in pred_data.columns:
            pred_data['score'] = pred_data['up_proba'] + 0.1 * pred_data['mf_signal'].fillna(0)
        else:
            pred_data['score'] = pred_data['up_proba']
            
        if 'profit_pressure' in pred_data.columns:
            pred_data['score'] -= 0.05 * pred_data['profit_pressure'].fillna(0)
            
        # 过滤并排序
        recommendations = pred_data[
            (pred_data['up_proba'] > 0.55) & 
            (pred_data['pct_chg'] < 9.8) &
            (pred_data['pct_chg'] > -9.8)
        ].sort_values('score', ascending=False).head(top_n)
        
        # 补充股票名称和行业信息
        try:
            from infra_data.fetcher import DataFetcher
            fetcher = DataFetcher()
            stock_basics = fetcher.get_stock_list()
            # 合并 info
            if not stock_basics.empty and all(c in stock_basics.columns for c in ['ts_code', 'name', 'industry']):
                recommendations = recommendations.merge(stock_basics[['ts_code', 'name', 'industry']], on='ts_code', how='left')
            else:
                 recommendations['name'] = 'Unknown'
                 recommendations['industry'] = 'Unknown'

        except Exception as e:
            print(f"警告: 无法获取股票名称/行业信息: {e}")
            
        # Final safety check
        for col in ['name', 'industry']:
            if col not in recommendations.columns:
                recommendations[col] = 'Unknown'

        return recommendations, latest_date

    def check_market_status(self):
        """检查大盘指数 (000001.SH)"""
        try:
            import tushare as ts
            pro = ts.pro_api(settings.TUSHARE_TOKEN)
            
            # 获取最近 60 天的上证指数
            df = pro.index_daily(ts_code='000001.SH', end_date=datetime.now().strftime('%Y%m%d'), limit=100)
            if df.empty:
                return "未知", 0, 0
            
            df = df.sort_values('trade_date')
            closes = df['close'].values
            
            if len(closes) < 60:
                return "数据不足", 0, 0
                
            ma20 = closes[-20:].mean()
            ma60 = closes[-60:].mean()
            
            status = "BEAR" if ma20 < ma60 else "BULL"
            return status, ma20, ma60
        except Exception as e:
            print(f"获取大盘数据失败: {e}")
            return "Error", 0, 0

    def display(self, top_n: int = 10):
        """控制台输出"""
        recommendations, last_date = self.predict_next_day(top_n)
        
        # --- 新增: 大盘状态检查 ---
        status, ma20, ma60 = self.check_market_status()
        status_cn = "[熊市] (防御模式)" if status == "BEAR" else "[牛市] (进攻模式)"
        
        print("\n" + "=" * 70)
        print(f"  V5 策略今日实战信号 ({last_date.strftime('%Y-%m-%d')})")
        print("=" * 70)
        print(f"【大盘环境】 {status_cn}")
        print(f"   - 上证指数 MA20: {ma20:.2f}")
        print(f"   - 上证指数 MA60: {ma60:.2f}")
        print(f"   - 策略提示: {'只买 Top 3，总仓位 < 30%' if status == 'BEAR' else '正常买入 Top 10，满仓操作'}")
        print("=" * 70)
        
        if recommendations.empty:
            print("本周无符合55%以上胜率的股票，建议空仓休息。")
            return
            
        print(f"{'代码':<10} {'名称':<8} {'行业':<8} {'收盘价':>8} {'涨跌':>8} {'资金':>6} {'评分':>6}")
        print("-" * 70)
        
        for _, row in recommendations.iterrows():
            mf_status = "流入" if row.get('mf_signal', 0) > 0 else "流出"
            name = row.get('name', 'Unknown')[:4] # 截断名称
            ind = row.get('industry', 'Unknown')[:4]
            
            print(f"{row['ts_code']:<10} {name:<8} {ind:<8} {row['close']:>8.2f} {row['pct_chg']:>7.2f}% {mf_status:>6} {row['score']:>6.2f}")
            
        print("-" * 70)
        print("\n【V5 实战建议】")
        print(f"1. 调仓时机: 本周一 (即 {datetime.now().strftime('%Y-%m-%d')} 或之后第一个交易日)")
        if status == 'BEAR':
             print(f"2. ⚠️ 当前为熊市，请严格执行【只买前3只】策略！")
        else:
             print("2. 仓位管理: 建议等权分配 10 份资金，每只股票投入 10%。")
        print("3. 严格风控:")
        print("   - 止损线: -8% (坚决执行)")
        print("   - 止盈线: +25% (分批止盈)")
        print("=" * 70)


from sklearn.feature_selection import SelectKBest, f_classif


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Reload and retrain V5 model')
    parser.add_argument('--update', action='store_true', help='Update data before generating signals')
    args = parser.parse_args()
    
    gen = DailySignalV5()
    
    if args.update:
        gen.update_data()
        
    if args.train:
        gen.train_v5_model()
        
    gen.display()
    
    # Export to Desktop
    # Export to Project Signals Directory
    try:
        recs, date = gen.predict_next_day(top_n=20)
        if not recs.empty:
            # Save to: C:\Users\liuqi\quant_system_v2\signals
            base_dir = r"C:\Users\liuqi\quant_system_v2"
            signal_dir = os.path.join(base_dir, "signals")
            os.makedirs(signal_dir, exist_ok=True)
            
            # Format date safely
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            filename = f"V5_Signals_{date_str}.csv"
            filepath = os.path.join(signal_dir, filename)
            
            # Format for CSV
            export_df = recs[['ts_code', 'name', 'industry', 'close', 'pct_chg', 'score', 'mf_signal']].copy()
            export_df.columns = ['代码', '名称', '行业', '收盘价', '涨跌幅', 'V5评分', '资金流']
            export_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"\n[提示] 信号已保存至: {filepath}")
    except Exception as e:
        print(f"导出CSV失败: {e}")
