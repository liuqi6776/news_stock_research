"""
v2 Pipeline 主控脚本

一键运行完整研究流程，或单独运行某个步骤

用法:
  python run_v2_pipeline.py              # 运行全部步骤
  python run_v2_pipeline.py --step 1     # 只运行Step 1
  python run_v2_pipeline.py --step 5     # 只运行Step 5 (每日信号)
  python run_v2_pipeline.py --step 5 --date 20260504  # 指定日期
  python run_v2_pipeline.py --from 2     # 从Step 2开始运行
  python run_v2_pipeline.py --status     # 查看当前状态
"""
import os
import sys
import json
import argparse
from datetime import datetime

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))


def check_status():
    print("=" * 80)
    print("v2 Pipeline 当前状态")
    print("=" * 80)

    checks = [
        ("Step 1: 特征数据", "data/all_features_v2.parquet"),
        ("Step 2: WF预测", "predictions/predictions_1d_wf.parquet"),
        ("Step 2: 最新模型", "models/latest_wf_model.joblib"),
        ("Step 3: 最优参数", "results/optimized_params_v2.json"),
        ("Step 3: 网格搜索", "results/grid_search_v2_results.parquet"),
        ("Step 4: 逐日回测", "results/step4_daily_backtest.json"),
    ]

    for name, rel_path in checks:
        full_path = os.path.join(PIPELINE_DIR, rel_path)
        if os.path.exists(full_path):
            size_mb = os.path.getsize(full_path) / 1024 / 1024
            mtime = datetime.fromtimestamp(os.path.getmtime(full_path)).strftime('%Y-%m-%d %H:%M')
            print(f"  [OK] {name}: {size_mb:.1f}MB, 更新于 {mtime}")
        else:
            print(f"  [--] {name}: 不存在")

    signals_dir = os.path.join(PIPELINE_DIR, 'signals')
    if os.path.exists(signals_dir):
        signals = sorted([f for f in os.listdir(signals_dir) if f.endswith('.json')])
        if signals:
            latest_signal = signals[-1]
            print(f"  [OK] 最新信号: {latest_signal}")
        else:
            print(f"  [--] 信号目录为空")
    else:
        print(f"  [--] 信号目录不存在")

    if os.path.exists(os.path.join(PIPELINE_DIR, 'results', 'optimized_params_v2.json')):
        with open(os.path.join(PIPELINE_DIR, 'results', 'optimized_params_v2.json'), 'r') as f:
            params = json.load(f)
        print(f"\n  当前最优参数:")
        print(f"    threshold = {params['best_params']['threshold']:.2f}")
        print(f"    max_positions = {params['best_params']['max_positions']}")
        if params.get('opt_results'):
            print(f"    优化期 CAGR = {params['opt_results']['cagr']:.2%}")
            print(f"    优化期 Sharpe = {params['opt_results']['sharpe']:.2f}")


def run_step(step_num, extra_args=None):
    step_scripts = {
        1: ("Step 1: 构建特征数据", "step1_build_features.py"),
        2: ("Step 2: Walk-Forward训练+预测", "step2_walkforward_predict.py"),
        3: ("Step 3: 参数优化(无clip)", "step3_optimize_threshold.py"),
        4: ("Step 4: 逐日回测+分析", "step4_daily_backtest.py"),
        5: ("Step 5: 每日信号生成", "step5_daily_signal.py"),
    }

    if step_num not in step_scripts:
        print(f"错误: 无效步骤 {step_num}, 可选 1-5")
        return False

    name, script = step_scripts[step_num]
    print(f"\n{'='*80}")
    print(f"运行 {name}")
    print(f"{'='*80}")

    script_path = os.path.join(PIPELINE_DIR, script)
    if not os.path.exists(script_path):
        print(f"错误: 脚本不存在 {script_path}")
        return False

    cmd = f'python "{script_path}"'
    if extra_args:
        cmd += f" {extra_args}"

    ret = os.system(cmd)
    if ret != 0:
        print(f"错误: Step {step_num} 执行失败 (返回码={ret})")
        return False

    print(f"\nStep {step_num} 完成!")
    return True


def run_all(from_step=1):
    print("=" * 80)
    print("v2 Pipeline: 无clip, 只优化threshold + max_positions")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    for step in range(from_step, 6):
        if step == 5:
            print("\n注意: Step 5 (每日信号) 需要每天收盘后单独运行")
            print("跳过自动运行。请使用: python run_v2_pipeline.py --step 5")
            break

        success = run_step(step)
        if not success:
            print(f"\nPipeline 在 Step {step} 失败，停止执行")
            return False

    print(f"\n{'='*80}")
    print(f"Pipeline 完成! 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    check_status()
    return True


def main():
    parser = argparse.ArgumentParser(description='v2 Pipeline 主控脚本')
    parser.add_argument('--step', type=int, default=None, help='运行指定步骤 (1-5)')
    parser.add_argument('--from', type=int, default=1, dest='from_step', help='从指定步骤开始运行')
    parser.add_argument('--status', action='store_true', help='查看当前状态')
    parser.add_argument('--date', type=str, default=None, help='Step 5指定日期 (YYYYMMDD)')
    args = parser.parse_args()

    if args.status:
        check_status()
        return

    if args.step:
        extra = f"--date {args.date}" if args.date and args.step == 5 else ""
        run_step(args.step, extra)
    else:
        run_all(from_step=args.from_step)


if __name__ == '__main__':
    main()
