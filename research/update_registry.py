"""
研究注册表更新脚本

自动扫描所有研究目录，汇总结果到 STUDIES_REGISTRY.md
"""
import os
import pandas as pd
from datetime import datetime
import json


def scan_studies():
    """扫描所有研究目录"""
    studies_dir = os.path.join(os.path.dirname(__file__), 'studies')
    studies = []
    
    if not os.path.exists(studies_dir):
        return studies
    
    for study_name in sorted(os.listdir(studies_dir)):
        study_path = os.path.join(studies_dir, study_name)
        if not os.path.isdir(study_path):
            continue
        
        study_info = parse_study(study_name, study_path)
        studies.append(study_info)
    
    return studies


def parse_study(study_name, study_path):
    """解析单个研究目录"""
    info = {
        'id': study_name.split('_')[1] if '_' in study_name else '',
        'name': study_name,
        'path': study_path,
        'status': '⚪ 待开始',
        'features': '-',
        'target': '-',
        'backtest_method': '-',
        'return': '-',
        'sharpe': '-',
        'max_dd': '-',
        'win_rate': '-',
        'n_trades': '-',
        'last_run': '-'
    }
    
    # 读取README
    readme_path = os.path.join(study_path, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme = f.read()
        
        # 解析特征
        if '特征：' in readme:
            info['features'] = readme.split('特征：')[1].split('\n')[0].strip()
        
        # 解析目标
        if '目标：' in readme:
            info['target'] = readme.split('目标：')[1].split('\n')[0].strip()
        
        # 解析回测方式
        if '回测：' in readme:
            info['backtest_method'] = readme.split('回测：')[1].split('\n')[0].strip()
    
    # 读取结果
    results_dir = os.path.join(study_path, 'results')
    if os.path.exists(results_dir):
        # 找最新的结果文件
        result_files = [f for f in os.listdir(results_dir) if f.startswith('summary_')]
        if result_files:
            latest = sorted(result_files)[-1]
            summary_path = os.path.join(results_dir, latest)
            
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                info['return'] = f"{summary.get('total_return', 0):.2%}"
                info['sharpe'] = f"{summary.get('sharpe', 0):.2f}"
                info['max_dd'] = f"{summary.get('max_drawdown', 0):.2%}"
                info['win_rate'] = f"{summary.get('win_rate', 0):.2%}"
                info['n_trades'] = str(summary.get('n_trades', 0))
                info['status'] = '🟢 已完成' if summary.get('total_return', 0) > 0 else '🔴 亏损'
                info['last_run'] = latest.replace('summary_', '').replace('.json', '')
            except:
                pass
        else:
            info['status'] = '🟡 运行中/无结果'
    
    return info


def generate_registry(studies):
    """生成注册表Markdown"""
    lines = [
        "# 研究注册表",
        "",
        f"*最后更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## 研究列表",
        "",
        "| 编号 | 名称 | 状态 | 特征 | 目标 | 回测方式 | 收益率 | 夏普 | 最大回撤 | 胜率 | 交易次数 | 最后运行 |",
        "|------|------|------|------|------|----------|--------|------|----------|------|----------|----------|"
    ]
    
    for s in studies:
        lines.append(
            f"| {s['id']} | {s['name']} | {s['status']} | {s['features']} | {s['target']} | "
            f"{s['backtest_method']} | {s['return']} | {s['sharpe']} | {s['max_dd']} | "
            f"{s['win_rate']} | {s['n_trades']} | {s['last_run']} |"
        )
    
    lines.extend([
        "",
        "## 图例",
        "- 🟢 已完成且盈利",
        "- 🟡 进行中/待评估",
        "- 🔴 已完成但亏损",
        "- ⚪ 待开始",
        "",
        "## 详细说明",
        ""
    ])
    
    for s in studies:
        lines.extend([
            f"### Study {s['id']}: {s['name']}",
            f"- **路径**: `studies/{s['name']}/`",
            f"- **特征**: {s['features']}",
            f"- **目标**: {s['target']}",
            f"- **回测**: {s['backtest_method']}",
            f"- **状态**: {s['status']}",
            ""
        ])
    
    return '\n'.join(lines)


def main():
    """主函数"""
    print("扫描研究目录...")
    studies = scan_studies()
    print(f"发现 {len(studies)} 个研究")
    
    print("生成注册表...")
    registry = generate_registry(studies)
    
    output_path = os.path.join(os.path.dirname(__file__), 'STUDIES_REGISTRY.md')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(registry)
    
    print(f"注册表已更新: {output_path}")


if __name__ == '__main__':
    main()
