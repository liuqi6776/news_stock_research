import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_IDEA_DIR = os.path.join(BASE_DIR, 'new_idea')
DOUBAO_DIR = os.path.join(BASE_DIR, 'results_duobao')

src_trades = os.path.join(DOUBAO_DIR, 'preloaded_trades.csv')
dst_trades = os.path.join(NEW_IDEA_DIR, 'doubao_result_preloaded_trades.csv')

if os.path.exists(src_trades):
    shutil.copy(src_trades, dst_trades)
    print(f"已复制 doubao_result 交易记录到: {dst_trades}")

src_equity = os.path.join(DOUBAO_DIR, 'final_backtest_correct_equity.csv')
dst_equity = os.path.join(NEW_IDEA_DIR, 'doubao_result_equity.csv')

if os.path.exists(src_equity):
    shutil.copy(src_equity, dst_equity)
    print(f"已复制 doubao_result 净值数据到: {dst_equity}")

print("\n✅ 准备完成！")
