"""
量化交易系统 V2

主入口文件
"""

import argparse
from datetime import datetime

from config.settings import settings
from infra_data.fetcher import DataFetcher
from processing.pipeline import DataPipeline
from models.trainer import ModelTrainer
from models.predictor import Predictor


def fetch_data(end_date: str = None, start_date: str = "20200101"):
    """获取数据"""
    print("=" * 50)
    print("数据获取")
    print("=" * 50)
    
    fetcher = DataFetcher()
    fetcher.fetch_all(end_date=end_date, start_date=start_date)


def process_data(start_date: str, end_date: str):
    """处理数据"""
    print("=" * 50)
    print("数据处理")
    print("=" * 50)
    
    pipeline = DataPipeline()
    df = pipeline.run(start_date=start_date, end_date=end_date)
    
    return df


def train_model(df=None, start_date: str = None, end_date: str = None):
    """训练模型"""
    print("=" * 50)
    print("模型训练")
    print("=" * 50)
    
    if df is None:
        pipeline = DataPipeline()
        df = pipeline.run(start_date=start_date, end_date=end_date)
    
    trainer = ModelTrainer()
    trainer.train(df)
    trainer.save('model')
    
    # 显示特征重要性
    importance = trainer.get_feature_importance()
    print("\n特征重要性 Top 10:")
    print(importance.head(10))


def predict(model_name: str = 'model'):
    """预测"""
    print("=" * 50)
    print("模型预测")
    print("=" * 50)
    
    predictor = Predictor()
    predictor.load(model_name)
    
    # 加载最新数据
    import pandas as pd
    df = pd.read_parquet("filtered_df.parquet")
    
    # 预测
    result = predictor.predict(df)
    
    # 获取推荐股票
    top_stocks = predictor.get_top_stocks(result)
    
    print("\n今日推荐股票:")
    print(top_stocks)
    
    return result, top_stocks


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='量化交易系统 V2')
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # fetch 命令
    fetch_parser = subparsers.add_parser('fetch', help='获取数据')
    fetch_parser.add_argument('--end-date', type=str, help='结束日期 (YYYYMMDD)')
    fetch_parser.add_argument('--start-date', type=str, default='20200101', help='开始日期')
    
    # process 命令
    process_parser = subparsers.add_parser('process', help='处理数据')
    process_parser.add_argument('--start-date', type=str, required=True, help='开始日期')
    process_parser.add_argument('--end-date', type=str, required=True, help='结束日期')
    
    # train 命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--start-date', type=str, help='开始日期')
    train_parser.add_argument('--end-date', type=str, help='结束日期')
    
    # predict 命令
    predict_parser = subparsers.add_parser('predict', help='预测')
    predict_parser.add_argument('--model', type=str, default='model', help='模型名称')
    
    # run 命令 (完整流程)
    run_parser = subparsers.add_parser('run', help='运行完整流程')
    run_parser.add_argument('--end-date', type=str, help='结束日期')
    
    args = parser.parse_args()
    
    if args.command == 'fetch':
        fetch_data(args.end_date, args.start_date)
    
    elif args.command == 'process':
        process_data(args.start_date, args.end_date)
    
    elif args.command == 'train':
        train_model(start_date=args.start_date, end_date=args.end_date)
    
    elif args.command == 'predict':
        predict(args.model)
    
    elif args.command == 'run':
        end_date = args.end_date or datetime.today().strftime('%Y%m%d')
        start_date = '20200101'
        
        # 1. 获取数据
        fetch_data(end_date, start_date)
        
        # 2. 处理数据
        df = process_data(start_date, end_date)
        
        # 3. 训练模型
        train_model(df)
        
        # 4. 预测
        predict()
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
