"""
检查 final_method 回测结果是否存在A股交易规则问题
"""
import os
import pandas as pd
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = r'D:\iquant_data\data_v2'
PRICE_DIR = os.path.join(DATA_DIR, 'data_day1')

# 检查是否有回测结果文件
print("=" * 80)
print("检查 final_method 回测结果")
print("=" * 80)

# 查找所有CSV文件
csv_files = [f for f in os.listdir(THIS_DIR) if f.endswith('.csv')]
print(f"\n找到 {len(csv_files)} 个CSV文件: {csv_files}")

# 查找包含交易记录的JSON
json_files = [f for f in os.listdir(THIS_DIR) if f.startswith('prediction_') and f.endswith('.json')]
print(f"找到 {len(json_files)} 个预测结果文件")

# 检查模型文件
model_path = os.path.join(THIS_DIR, 'models', 'doubao_t1t2_model.joblib')
if os.path.exists(model_path):
    print(f"✅ 模型文件存在: {model_path}")
else:
    print(f"❌ 模型文件不存在: {model_path}")

# 检查代码中的交易逻辑
print("\n" + "=" * 80)
print("检查代码逻辑")
print("=" * 80)

# 读取3_train_model.py检查标签计算
with open(os.path.join(THIS_DIR, '3_train_model.py'), 'r', encoding='utf-8') as f:
    train_code = f.read()

print("\n1. 标签计算逻辑:")
if "t2_close" in train_code and "t1_open" in train_code:
    print("   ✅ 标签使用 T+2 close / T+1 open - 1 (正确)")
else:
    print("   ⚠️  标签计算方式需检查")

print("\n2. 板块过滤:")
if "startswith('688')" in train_code:
    print("   ✅ 过滤科创板 (688)")
if "startswith('689')" in train_code:
    print("   ✅ 过滤科创板 (689)")
if "startswith('300')" in train_code or "startswith('301')" in train_code:
    print("   ✅ 过滤创业板")
else:
    print("   ⚠️  未过滤创业板 (300/301)")

# 读取4_predict_select.py检查预测逻辑
with open(os.path.join(THIS_DIR, '4_predict_select.py'), 'r', encoding='utf-8') as f:
    predict_code = f.read()

print("\n3. 选股逻辑:")
if "is_main_board" in predict_code:
    print("   ✅ 只选择主板股票 (60/00开头)")
else:
    print("   ⚠️  未限制主板")

print("\n4. 交易规则检查:")
# 检查是否有涨跌停处理
has_limit_check = False
if "limit" in predict_code.lower() or "涨停" in predict_code or "跌停" in predict_code:
    has_limit_check = True

if not has_limit_check:
    print("   ❌ 未处理涨跌停限制!")
    print("      - 涨停开盘的股票可能被选中")
    print("      - 跌停日的股票可能按收盘价卖出")
else:
    print("   ✅ 有涨跌停处理逻辑")

# 检查交易费用
if "cost" in predict_code.lower() or "fee" in predict_code.lower() or "费用" in predict_code:
    print("   ✅ 有交易费用处理")
else:
    print("   ⚠️  未明确处理交易费用")

print("\n" + "=" * 80)
print("问题总结")
print("=" * 80)

issues = []

# 检查是否有过滤创业板
if "startswith('300')" not in train_code and "startswith('301')" not in train_code:
    if "is_main_board" in predict_code:
        issues.append("训练时未过滤创业板，但预测时只选主板 - 一致")
    else:
        issues.append("⚠️ 未过滤创业板股票")

# 检查涨跌停
if not has_limit_check:
    issues.append("❌ 严重: 未处理涨跌停限制，回测可能失真")

# 检查交易费用
if "cost" not in predict_code.lower() and "fee" not in predict_code.lower():
    issues.append("⚠️ 未处理交易费用")

if not issues:
    print("✅ 未发现明显问题")
else:
    for issue in issues:
        print(f"  {issue}")

print("\n" + "=" * 80)
print("建议")
print("=" * 80)
print("1. 添加涨跌停检查:")
print("   - 涨停开盘 (>9.5%) 的股票不能买入")
print("   - 跌停日按跌停价卖出，不是收盘价")
print("\n2. 添加交易费用:")
print("   - 佣金 + 印花税 + 过户费 ≈ 0.3% 双边")
print("\n3. 添加滑点:")
print("   - 买入滑点 +0.1%~0.3%")
print("   - 卖出滑点 -0.1%~0.3%")
