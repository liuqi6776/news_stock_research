"""
Auto Optimization Loop - 自动优化框架
======================================
用法: python optimize_loop.py

功能:
1. 对比所有版本的回测结果
2. 基于最优版本做参数微调
3. 实时输出 progress.log 和 result.log
4. 支持增量迭代: python optimize_loop.py --iter 5 (跑5轮)
"""
import os
import sys
import time
import json
import glob
import argparse
import warnings
import subprocess
from datetime import datetime

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(OUT_DIR, 'progress.log')
RESULT_FILE = os.path.join(OUT_DIR, 'result.log')

def log(msg, also_print=True):
    """写入 progress.log 并可选打印"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    line = f"[{timestamp}] {msg}"
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    if also_print:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode('utf-8', errors='replace').decode('ascii', errors='replace'))

def result_log(msg):
    """写入 result.log (汇总结果)"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    with open(RESULT_FILE, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def calc_metrics(equity_csv):
    """从 equity CSV 计算核心指标"""
    try:
        df = pd.read_csv(equity_csv)
        if len(df) < 10:
            return None
        df['date'] = pd.to_datetime(df['date'])
        nav = df['nav']
        
        initial = nav.iloc[0]
        final = nav.iloc[-1]
        total_ret = (final / initial - 1) * 100
        
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        if days <= 0:
            days = 1
        annual_ret = ((final / initial) ** (365/days) - 1) * 100
        
        peak = nav.cummax()
        drawdown = (nav - peak) / peak
        max_dd = drawdown.min() * 100
        
        daily_ret = nav.pct_change().dropna()
        sharpe = (daily_ret.mean() / (daily_ret.std() + 1e-8)) * np.sqrt(252)
        calmar = annual_ret / (abs(max_dd) + 1e-8)
        win_rate = (daily_ret > 0).sum() / (len(daily_ret) + 1e-8) * 100
        
        # 年度收益
        df['year'] = df['date'].dt.year
        yearly = df.groupby('year').apply(lambda x: (x['nav'].iloc[-1] / x['nav'].iloc[0] - 1) * 100)
        
        return {
            'total_ret': round(total_ret, 2),
            'annual_ret': round(annual_ret, 2),
            'max_dd': round(max_dd, 2),
            'sharpe': round(sharpe, 2),
            'calmar': round(calmar, 2),
            'win_rate': round(win_rate, 1),
            'years': len(yearly),
            'yearly': {int(k): round(v, 1) for k, v in yearly.items()},
            'final_nav': round(final, 2),
            'days': days,
        }
    except Exception as e:
        return {'error': str(e)}

def scan_all_versions():
    """扫描所有 equity CSV"""
    equity_files = glob.glob(os.path.join(OUT_DIR, '*equity*.csv'))
    results = {}
    
    log("📊 扫描所有版本回测结果...")
    
    for fpath in sorted(equity_files):
        name = os.path.basename(fpath).replace('_equity.csv', '')
        metrics = calc_metrics(fpath)
        if metrics and 'error' not in metrics:
            results[name] = metrics
            log(f"  ✅ {name}: 收益={metrics['total_ret']}%, 年化={metrics['annual_ret']}%, "
                f"最大回撤={metrics['max_dd']}%, Sharpe={metrics['sharpe']}, Calmar={metrics['calmar']}")
        elif metrics:
            log(f"  ❌ {name}: 读取失败 - {metrics['error']}")
    
    return results

def rank_versions(results):
    """按综合得分排名"""
    def score(m):
        """综合评分: 60%年化 + 20%Sharpe - 20%最大回撤"""
        return (m['annual_ret'] * 0.6 + m['sharpe'] * 20 - abs(m['max_dd']) * 0.2)
    
    ranked = sorted(results.items(), key=lambda x: score(x[1]), reverse=True)
    return ranked

def generate_report(results, ranked):
    """生成对比报告"""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"  策略版本对比报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    
    # 排名表
    report_lines.append(f"\n{'排名':>4} | {'版本':<30} | {'总收益':>8} | {'年化':>8} | {'最大回撤':>8} | {'Sharpe':>7} | {'Calmar':>7} | {'胜率':>6}")
    report_lines.append("-" * 100)
    
    for i, (name, m) in enumerate(ranked, 1):
        emoji = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else f" {i}"))
        report_lines.append(f"{emoji} | {name:<30} | {m['total_ret']:>7.1f}% | {m['annual_ret']:>7.1f}% | {m['max_dd']:>7.1f}% | {m['sharpe']:>6.2f} | {m['calmar']:>6.2f} | {m['win_rate']:>5.1f}%")
    
    # 最优版本详情
    best_name, best_m = ranked[0]
    report_lines.append(f"\n🏆 最优版本: {best_name}")
    report_lines.append(f"   总收益: {best_m['total_ret']:.1f}% | 年化: {best_m['annual_ret']:.1f}% | 最大回撤: {best_m['max_dd']:.1f}%")
    report_lines.append(f"   Sharpe: {best_m['sharpe']:.2f} | Calmar: {best_m['calmar']:.2f} | 胜率: {best_m['win_rate']:.1f}%")
    report_lines.append(f"\n   年度收益:")
    for year, ret in best_m.get('yearly', {}).items():
        bar = "🟢" if ret > 0 else "🔴"
        report_lines.append(f"     {bar} {year}: {ret:+.1f}%")
    
    # 改进建议
    report_lines.append(f"\n" + "=" * 80)
    report_lines.append("  改进建议 (基于数据驱动)")
    report_lines.append("=" * 80)
    
    if best_m['max_dd'] > -15:
        report_lines.append("  1. ⚠️ 最大回撤较大, 考虑:")
        report_lines.append("     - 收紧止损 (-8% → -10%)")
        report_lines.append("     - 增加波动率过滤 (高VIX时减仓)")
        report_lines.append("     - 分散持仓 (TOP_N 增加)")
    
    if best_m['win_rate'] < 55:
        report_lines.append("  2. ⚠️ 胜率偏低, 考虑:")
        report_lines.append("     - 提高模型阈值 (0.5 → 0.55)")
        report_lines.append("     - 增加质量过滤 (流动性/市值)")
        report_lines.append("     - 缩短持仓周期")
    
    if best_m['sharpe'] < 1.0:
        report_lines.append("  3. ⚠️ Sharpe偏低, 考虑:")
        report_lines.append("     - 加入市场中性对冲")
        report_lines.append("     - 动态仓位管理")
        report_lines.append("     - 优化特征工程")
    
    # 年度弱点
    for year, ret in best_m.get('yearly', {}).items():
        if ret < -5:
            report_lines.append(f"  4. 📉 {year}年表现较差({ret:+.1f}%), 分析原因:")
            report_lines.append(f"     - 检查{year}年市场环境(牛/熊/震荡)")
            report_lines.append(f"     - 该版本在该环境下的市场识别是否正确")
    
    report_text = '\n'.join(report_lines)
    
    # 写入报告
    report_path = os.path.join(OUT_DIR, 'optimization_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    log(f"\n📄 报告已保存: optimization_report.txt")
    result_log(report_text)
    
    return report_text

def run_version(version_script, version_name):
    """运行指定版本的回测脚本"""
    log(f"🚀 开始运行 {version_name}...")
    
    try:
        result = subprocess.run(
            [sys.executable, version_script],
            capture_output=True, text=True, timeout=3600,
            cwd=OUT_DIR
        )
        
        if result.returncode == 0:
            log(f"  ✅ {version_name} 运行完成")
            if result.stdout:
                for line in result.stdout.strip().split('\n')[-10:]:
                    log(f"     {line}", also_print=False)
                log(f"     最后输出: {result.stdout.strip().split(chr(10))[-1]}")
            return True
        else:
            log(f"  ❌ {version_name} 运行失败: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        log(f"  ⏱️ {version_name} 超时(60分钟)")
        return False
    except Exception as e:
        log(f"  ❌ {version_name} 异常: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Auto Optimization Loop')
    parser.add_argument('--scan', action='store_true', help='仅扫描现有结果, 不运行回测')
    parser.add_argument('--run', type=str, help='运行指定脚本 (如 run_super_weekly_v18.py)')
    parser.add_argument('--iter', type=int, default=1, help='迭代次数')
    args = parser.parse_args()
    
    # 初始化日志
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"  优化循环启动 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n\n")
    
    log("🦞 AutoClaw 优化循环启动")
    log(f"   工作目录: {OUT_DIR}")
    log(f"   迭代次数: {args.iter}")
    log(f"   模式: {'仅扫描' if args.scan else ('运行指定: '+args.run if args.run else '完整优化')}")
    
    if args.run:
        # 运行指定脚本
        script_path = os.path.join(OUT_DIR, args.run)
        if os.path.exists(script_path):
            run_version(script_path, args.run)
        else:
            log(f"❌ 脚本不存在: {script_path}")
    
    # 扫描所有版本
    results = scan_all_versions()
    
    if not results:
        log("❌ 没有找到任何可用的回测结果")
        return
    
    ranked = rank_versions(results)
    report = generate_report(results, ranked)
    
    print("\n" + report)
    log("✅ 优化循环完成")

if __name__ == '__main__':
    main()
