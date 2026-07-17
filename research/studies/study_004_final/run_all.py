脚本研究,您说:"""
一键运行: 改进版超跌反弹策略完整流程

执行步骤:
1. 构建逆向特征 (build_contrarian_features.py)
2. 训练28日Walk-Forward模型 (train_contrarian.py)
3. 网格搜索止损止盈参数 (optimize_contrarian.py)
4. 详细回测+可视化 (backtest_contrarian.py)
5. 生成当日信号 (generate_signal.py)

使用方法:
    cd study_004_final
    python run_all.py

或分步运行:
    python build_contrarian_features.py
    python train_contrarian.py
    python optimize_contrarian.py
    python backtest_contrarian.py
    python generate_signal.py --date 20260101
"""
import os
import sys

# 确保当前目录在路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_step(name, module):
    print("\n" + "=" * 80)
    print(f" 开始执行: {name}")
    print("=" * 80)
    try:
        module.run()
        print(f" ✓ {name} 完成")
        return True
    except Exception as e:
        print(f" ✗ {name} 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 80)
    print("改进版超跌反弹策略 - 完整运行流程")
    print("=" * 80)

    steps = [
        ("Step 1: 构建逆向特征", "build_contrarian_features"),
        ("Step 2: 训练28日模型", "train_contrarian"),
        ("Step 3: 参数优化", "optimize_contrarian"),
        ("Step 4: 详细回测", "backtest_contrarian"),
    ]

    results = []
    for name, module_name in steps:
        module = __import__(module_name)
        success = run_step(name, module)
        results.append((name, success))

    print("\n" + "=" * 80)
    print("执行摘要")
    print("=" * 80)
    for name, success in results:
        status = "✓ 成功" if success else "✗ 失败"
        print(f" {status}: {name}")

    print("\n" + "=" * 80)
    print("下一步:")
    print("  生产环境信号生成: python generate_signal.py")
    print("  指定日期信号: python generate_signal.py --date 20260101")
    print("=" * 80)


if __name__ == '__main__':
    main()
