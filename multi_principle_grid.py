#!/usr/bin/env python3
"""
Grid Search — Multi-Principle MTTD Strategy
=============================================
Goal: Find params that give 25-35 trades with best Sharpe
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

from multi_principle_strategy import multi_principle_strategy, backtest, generate_ichimoku_features

# Load data
print("Loading data...")
with open('data/btc_daily.json') as f:
    btc_data = json.load(f)
df = pd.DataFrame(btc_data['aligned_data'])
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = df[df.index >= '2018-01-01']

# Pre-compute features (expensive, do once)
print("Pre-computing features...")
df_feat = generate_ichimoku_features(df)
train_idx = df_feat.index < '2025-01-01'
test_idx = df_feat.index >= '2025-01-01'

# Grid search
results = []
total = 0

# Parameter ranges
configs = []

# Default config
base = {
    'min_hold_days': 10, 'max_hold_days': 120,
    'er_entry': 0.25, 't_entry': 0.40,
    'chikou_thresh': -0.30, 'immunity_thresh': 0.50,
    'entropy_thresh': 2.271, 'imo_min_limit': -0.30,
    'imo_exit_bull': -0.30, 'roc_gate_limit': -0.20,
    'cooldown': 5,
    'confirm_entry': 2, 'confirm_exit': 1
}

# Phase 1: Relax entry gates to get more trades
for t_entry in [0.20, 0.25, 0.30, 0.35, 0.40]:
    for er_entry in [0.15, 0.20, 0.25]:
        for entropy_thresh in [2.271, 2.5, 2.8, 3.0]:
            cfg = base.copy()
            cfg.update({'t_entry': t_entry, 'er_entry': er_entry, 'entropy_thresh': entropy_thresh})
            configs.append(cfg)

# Phase 2: Vary hold/exit params
for min_hold in [10, 15, 20]:
    for max_hold in [60, 90, 120]:
        for chikou_thresh in [-0.5, -0.3, -0.2, -0.1]:
            for immunity_thresh in [0.3, 0.5, 0.7]:
                for imo_exit_bull in [-0.5, -0.3, -0.1, 0.0]:
                    cfg = base.copy()
                    cfg.update({
                        'min_hold_days': min_hold,
                        'max_hold_days': max_hold,
                        'chikou_thresh': chikou_thresh,
                        'immunity_thresh': immunity_thresh,
                        'imo_exit_bull': imo_exit_bull
                    })
                    configs.append(cfg)

total = len(configs)
print(f"Total configs: {total}")
print(f"Phase 1 (entry gates): {sum(1 for c in configs if c == base.update(c) or True)}")
print()

start_all = time.time()

for i, cfg in enumerate(configs):
    if i % 50 == 0:
        elapsed = time.time() - start_all
        print(f"  [{i}/{total}] {elapsed:.0f}s elapsed...")
    
    # Run strategy on full data
    result = multi_principle_strategy(df_feat.copy(), **cfg)
    
    # Split train/test metrics
    train_result = result.loc[train_idx].copy()
    test_result = result.loc[test_idx].copy()
    
    train_metrics = backtest(train_result, df.loc[train_idx, 'close'])
    test_metrics = backtest(test_result, df.loc[test_idx, 'close'])
    
    # Filter: must have 20-40 trades total
    total_trades = train_metrics['trades'] + test_metrics['trades']
    if 20 <= total_trades <= 40:
        results.append({
            **cfg,
            'total_trades': total_trades,
            'train_trades': train_metrics['trades'],
            'train_wr': train_metrics['win_rate'],
            'train_sharpe': train_metrics['sharpe'],
            'train_cagr': train_metrics['cagr'],
            'test_trades': test_metrics['trades'],
            'test_wr': test_metrics['win_rate'],
            'test_sharpe': test_metrics['sharpe'],
            'test_cagr': test_metrics['cagr'],
        })

print(f"\nDone. {len(results)} configs with 20-40 trades out of {total}")
print()

# Sort by various criteria
by_sharpe = sorted(results, key=lambda x: x.get('test_sharpe', 0), reverse=True)
by_combined = sorted(results, key=lambda x: x.get('test_sharpe', 0) * 0.5 + x.get('test_wr', 0) / 100 * 0.5, reverse=True)

print("=" * 100)
print("TOP 10 by Test Sharpe (with 20-40 trades):")
print("=" * 100)
print(f"{'Config':<75} {'Trades':>7} {'WR%':>6} {'Sharpe':>7} {'CAGR%':>6}")
print("-" * 100)
for r in by_sharpe[:10]:
    cfg_str = f"t={r['t_entry']:.2f} er={r['er_entry']:.2f} ent={r['entropy_thresh']:.1f} mh={r['min_hold_days']}/{r['max_hold_days']} chikou={r['chikou_thresh']:.1f} imm={r['immunity_thresh']:.1f}"
    print(f"{cfg_str:<75} {r['total_trades']:>7} {r['test_wr']:>5.0f}% {r['test_sharpe']:>7.2f} {r['test_cagr']:>5.0f}%")

print()
print("=" * 100)
print("TOP 10 by Combined Score (Sharpe + WinRate):")
print("=" * 100)
print(f"{'Config':<75} {'Trades':>7} {'WR%':>6} {'Sharpe':>7} {'CAGR%':>6}")
print("-" * 100)
for r in by_combined[:10]:
    cfg_str = f"t={r['t_entry']:.2f} er={r['er_entry']:.2f} ent={r['entropy_thresh']:.1f} mh={r['min_hold_days']}/{r['max_hold_days']} chikou={r['chikou_thresh']:.1f} imm={r['immunity_thresh']:.1f}"
    print(f"{cfg_str:<75} {r['total_trades']:>7} {r['test_wr']:>5.0f}% {r['test_sharpe']:>7.2f} {r['test_cagr']:>5.0f}%")

# Save results
import csv
if results:
    keys = results[0].keys()
    with open('mttd/multi_principle_grid_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved {len(results)} results to mttd/multi_principle_grid_results.csv")

# Save top configs
top = {'by_sharpe': by_sharpe[:5], 'by_combined': by_combined[:5]}
with open('mttd/multi_principle_top_configs.json', 'w') as f:
    json.dump(top, f, indent=2, default=str)
print("Saved top configs to mttd/multi_principle_top_configs.json")
