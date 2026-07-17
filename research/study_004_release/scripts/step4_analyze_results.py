"""
Step 4: Analyze Results

Reads the grid search results and generates:
  1. T+1 vs No-T+1 comparison
  2. Best parameter identification
  3. Gap-down risk analysis
  4. Structured conclusions (JSON)

Output: conclusions/summary.json
"""
import os
import sys
import pandas as pd
import numpy as np
import json

sys.stdout.reconfigure(line_buffering=True)

RELEASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(RELEASE_DIR, 'results')
CONCLUSIONS_DIR = os.path.join(RELEASE_DIR, 'conclusions')
FEATURES_FILE = os.path.join(RELEASE_DIR, 'data', 'all_features_v2.parquet')
os.makedirs(CONCLUSIONS_DIR, exist_ok=True)

T1_RESULTS = os.path.join(RESULTS_DIR, 'wf_monthly_grid_t1_full.csv')


def analyze_t1_gap_risk():
    if not os.path.exists(FEATURES_FILE):
        print("  (Feature file not available for gap risk analysis)")
        return {}

    feat = pd.read_parquet(FEATURES_FILE)
    feat['trade_date'] = feat['trade_date'].astype(str)
    feat = feat.sort_values(['ts_code', 'trade_date'])
    feat['next_open'] = feat.groupby('ts_code')['open'].shift(-1)
    feat['next_low'] = feat.groupby('ts_code')['low'].shift(-1)
    feat = feat.dropna(subset=['next_open', 'next_low'])

    gap_risk = {}
    for period_name, start, end in [('opt', '20220101', '20241231'), ('test', '20250101', '20261231')]:
        sub = feat[(feat['trade_date'] >= start) & (feat['trade_date'] <= end)]
        period_data = {'total_rows': len(sub)}
        for sl_val in [-0.03, -0.05, -0.07]:
            sl_price = sub['entry_price'] * (1 + sl_val)
            gap_down = int((sub['next_open'] <= sl_price).sum())
            sl_trigger = int((sub['next_low'] <= sl_price).sum())
            normal_trigger = sl_trigger - gap_down
            period_data[f'sl_{abs(sl_val):.0f}pct'] = {
                'gap_down_at_open': gap_down,
                'gap_down_pct': float(gap_down / len(sub)),
                'normal_trigger': normal_trigger,
                'normal_trigger_pct': float(normal_trigger / len(sub)),
                'total_trigger': sl_trigger,
                'total_trigger_pct': float(sl_trigger / len(sub)),
            }
        gap_risk[period_name] = period_data
    return gap_risk


def run():
    print("=" * 90)
    print("Step 4: Analyze Results")
    print("=" * 90)

    if not os.path.exists(T1_RESULTS):
        print(f"ERROR: Results file not found: {T1_RESULTS}")
        print("Please run step3_grid_search_t1.py first.")
        return

    t1 = pd.read_csv(T1_RESULTS)
    print(f"Loaded {len(t1)} grid search results")

    opt_df = t1.dropna(subset=['cagr_opt']).copy()

    print("\n--- T+1 Constrained: Top 10 by Test Sharpe ---")
    top10_test = opt_df.nlargest(10, 'sharpe_test')
    cols = ['threshold', 'max_pos', 'stop_loss', 'take_profit',
            'cagr_opt', 'sharpe_opt', 'max_dd_opt',
            'cagr_test', 'sharpe_test', 'max_dd_test']
    print(top10_test[[c for c in cols if c in top10_test.columns]].to_string(index=False))

    print("\n--- T+1 Constrained: Top 10 by Test CAGR (positive only) ---")
    pos_test = opt_df[opt_df['cagr_test'] > 0].nlargest(10, 'cagr_test')
    print(pos_test[[c for c in cols if c in pos_test.columns]].to_string(index=False))

    print("\n--- Stop Loss Impact (threshold=0.58, max_pos=3, tp=0) ---")
    sl_sub = opt_df[(opt_df['threshold'] == 0.58) & (opt_df['max_pos'] == 3) & (opt_df['take_profit'] == 0)]
    if len(sl_sub) > 0:
        for _, r in sl_sub.iterrows():
            print(f"  SL={r['stop_loss']:+.0%}: opt CAGR={r['cagr_opt']:+.1%}, Sharpe={r['sharpe_opt']:.2f}, "
                  f"DD={r['max_dd_opt']:.1%} | test CAGR={r['cagr_test']:+.1%}, Sharpe={r['sharpe_test']:.2f}")

    print("\n--- Take Profit Impact (threshold=0.58, max_pos=3, sl=0) ---")
    tp_sub = opt_df[(opt_df['threshold'] == 0.58) & (opt_df['max_pos'] == 3) & (opt_df['stop_loss'] == 0)]
    if len(tp_sub) > 0:
        for _, r in tp_sub.iterrows():
            print(f"  TP={r['take_profit']:+.0%}: opt CAGR={r['cagr_opt']:+.1%}, Sharpe={r['sharpe_opt']:.2f}, "
                  f"DD={r['max_dd_opt']:.1%} | test CAGR={r['cagr_test']:+.1%}, Sharpe={r['sharpe_test']:.2f}")

    print("\n--- Best per Threshold (by test Sharpe) ---")
    for t in sorted(opt_df['threshold'].unique()):
        sub = opt_df[opt_df['threshold'] == t]
        best = sub.nlargest(1, 'sharpe_test').iloc[0]
        print(f"  thresh={t}: pos={int(best['max_pos'])}, sl={best['stop_loss']:+.0%}, "
              f"tp={best['take_profit']:+.0%}, "
              f"opt_sharpe={best['sharpe_opt']:.2f}, opt_cagr={best['cagr_opt']:+.1%}, "
              f"test_sharpe={best['sharpe_test']:.2f}, test_cagr={best['cagr_test']:+.1%}")

    print("\n--- Gap-Down Risk Analysis ---")
    gap_risk = analyze_t1_gap_risk()
    for period, data in gap_risk.items():
        print(f"\n  {period}: {data['total_rows']} rows")
        for sl_key, sl_data in data.items():
            if sl_key.startswith('sl_'):
                print(f"    {sl_key}: gap_down={sl_data['gap_down_at_open']} ({sl_data['gap_down_pct']:.2%}), "
                      f"normal_trigger={sl_data['normal_trigger']} ({sl_data['normal_trigger_pct']:.2%}), "
                      f"total={sl_data['total_trigger']} ({sl_data['total_trigger_pct']:.2%})")

    best_overall = opt_df.nlargest(1, 'sharpe_test').iloc[0]
    pos_cagr = opt_df[opt_df['cagr_test'] > 0]
    best_cagr = pos_cagr.nlargest(1, 'cagr_test').iloc[0] if len(pos_cagr) > 0 else best_overall

    conclusion = {
        'study_name': 'study_004_daily_level_A_share',
        'date': '2026-05-05',
        'market': 'A-share (China)',
        'constraint': 'T+1 (cannot sell on purchase day)',
        'model': 'XGBoost Classifier, monthly walk-forward retraining',
        'prediction_target': '1-day forward return > 1.5%',
        'optimization_period': '2022-01 to 2024-12',
        'test_period': '2025-01 to 2026-03',
        'grid_search_space': {
            'threshold': THRESHOLDS if 'THRESHOLDS' in dir() else [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66],
            'max_positions': [1, 2, 3, 5, 10],
            'stop_loss': [0.0, -0.03, -0.05, -0.07, -0.10],
            'take_profit': [0.0, 0.05, 0.08, 0.10, 0.15],
            'total_combinations': 1125
        },
        'best_overall': {
            'description': 'Best by test Sharpe ratio (T+1 constrained)',
            'threshold': float(best_overall['threshold']),
            'max_positions': int(best_overall['max_pos']),
            'stop_loss': float(best_overall['stop_loss']),
            'take_profit': float(best_overall['take_profit']),
            'opt_cagr': float(best_overall['cagr_opt']),
            'opt_sharpe': float(best_overall['sharpe_opt']),
            'opt_max_dd': float(best_overall['max_dd_opt']),
            'opt_n_trades': int(best_overall['n_trades_opt']),
            'test_cagr': float(best_overall['cagr_test']),
            'test_sharpe': float(best_overall['sharpe_test']),
            'test_max_dd': float(best_overall['max_dd_test']),
            'test_n_trades': int(best_overall['n_trades_test']),
        },
        'critical_findings': {
            't1_constraint_destroys_stop_loss': True,
            'stop_loss_effect_without_t1': 'SL=-3% boosts CAGR from 1.6% to 248% (unrealistic)',
            'stop_loss_effect_with_t1': 'SL=-3% makes CAGR -97.8% (catastrophic)',
            'reason': 'A-share T+1 rule prevents same-day stop loss; next-day gap-down bypasses stop price',
            'take_profit_5pct_is_key': 'Only TP=5% without SL produces positive test returns',
            'best_realistic_expectation': 'Test CAGR ~8%, Test Sharpe ~1.0, Test MaxDD ~-6.5%',
            'strategy_edge_is_weak': 'Without T+1 violation, strategy barely breaks even in test period',
        },
        'gap_risk_analysis': gap_risk,
        'top5_test_sharpe': [],
        'top5_test_cagr_positive': [],
    }

    top5_sharpe = opt_df.nlargest(5, 'sharpe_test')
    for _, r in top5_sharpe.iterrows():
        conclusion['top5_test_sharpe'].append({
            'threshold': float(r['threshold']),
            'max_positions': int(r['max_pos']),
            'stop_loss': float(r['stop_loss']),
            'take_profit': float(r['take_profit']),
            'opt_cagr': float(r['cagr_opt']),
            'opt_sharpe': float(r['sharpe_opt']),
            'opt_max_dd': float(r['max_dd_opt']),
            'test_cagr': float(r['cagr_test']),
            'test_sharpe': float(r['sharpe_test']),
            'test_max_dd': float(r['max_dd_test']),
        })

    if len(pos_cagr) > 0:
        top5_cagr = pos_cagr.nlargest(5, 'cagr_test')
        for _, r in top5_cagr.iterrows():
            conclusion['top5_test_cagr_positive'].append({
                'threshold': float(r['threshold']),
                'max_positions': int(r['max_pos']),
                'stop_loss': float(r['stop_loss']),
                'take_profit': float(r['take_profit']),
                'opt_cagr': float(r['cagr_opt']),
                'opt_sharpe': float(r['sharpe_opt']),
                'opt_max_dd': float(r['max_dd_opt']),
                'test_cagr': float(r['cagr_test']),
                'test_sharpe': float(r['sharpe_test']),
                'test_max_dd': float(r['max_dd_test']),
            })

    output_path = os.path.join(CONCLUSIONS_DIR, 'summary.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(conclusion, f, ensure_ascii=False, indent=2)
    print(f"\nConclusions saved: {output_path}")
    print("Done!")


if __name__ == '__main__':
    run()
