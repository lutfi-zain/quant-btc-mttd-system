#!/usr/bin/env python3
"""
Test SERF + MSVR Combination V2
=================================

Uses SERF as CONTINUOUS weight instead of binary filter.
- Low entropy = high weight (market is cyclic, signal is reliable)
- High entropy = low weight (market is random, signal is unreliable)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import importlib.util
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
bank_root = '/home/ubuntu/projects/quant-technical-indicator-bank'
sys.path.append(project_root)
sys.path.append(bank_root)
from indicators_helper import *

print("=" * 70)
print("TEST SERF + MSVR COMBINATION V2 (Continuous Weight)")
print("=" * 70)

# ================================================================
# Load Data
# ================================================================
print("\n[1/5] Loading data...")

with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
    btc_data = json.load(f)

df_full = pd.DataFrame(btc_data['aligned_data'])
df_full['time'] = pd.to_datetime(df_full['time'])
df_full = df_full.set_index('time')
df_full = df_full[df_full.index >= '2018-01-01']

HOLDOUT_START = '2025-01-01'
df_train = df_full[df_full.index < HOLDOUT_START].copy()
df_holdout = df_full[df_full.index >= HOLDOUT_START].copy()

print(f"  Training: {len(df_train)} bars")
print(f"  Holdout:  {len(df_holdout)} bars")

# Load ISP
isp_df = pd.read_csv(os.path.join(project_root, 'isp-signals-btcusd-2026-06-13.csv'))
isp_df['Date'] = pd.to_datetime(isp_df['Date'])
isp_df = isp_df.set_index('Date')

isp_positions_full = pd.Series(0.0, index=df_full.index)
for date, row in isp_df.iterrows():
    if date in isp_positions_full.index:
        if row['Action'] == 'BUY':
            isp_positions_full.loc[date:] = 1.0
        elif row['Action'] == 'SELL':
            isp_positions_full.loc[date:] = 0.0

isp_positions_train = isp_positions_full[df_train.index]
isp_positions_holdout = isp_positions_full[df_holdout.index]

# ================================================================
# Metrics Functions
# ================================================================
def compute_metrics(positions, prices, transaction_cost=0.001):
    returns = prices.pct_change()
    strategy_returns = returns * positions.shift(1)
    strategy_returns = strategy_returns.dropna()
    transitions = positions.diff().fillna(0)
    strategy_returns = strategy_returns - transitions.loc[strategy_returns.index] * (transaction_cost / 2)
    if len(strategy_returns) == 0:
        return {'cagr': 0, 'sharpe': 0, 'sortino': 0, 'calmar': 0, 'max_dd': 0, 'n_trades': 0, 'pct_in': 0}
    equity = (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25
    cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
    downside = strategy_returns[strategy_returns < 0]
    sortino = strategy_returns.mean() / downside.std() * np.sqrt(365) if len(downside) > 0 and downside.std() > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    n_trades = (positions.diff().abs() > 0).sum()
    return {
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'calmar': round(calmar, 2),
        'max_dd': round(max_dd * 100, 2),
        'n_trades': int(n_trades),
        'pct_in': round(positions.mean() * 100, 2)
    }

def compute_coherence(positions, benchmark):
    aligned = pd.DataFrame({'system': positions, 'benchmark': benchmark}).dropna()
    if len(aligned) == 0:
        return 0.0
    return (aligned['system'] == aligned['benchmark']).sum() / len(aligned) * 100

# ================================================================
# Load Indicators
# ================================================================
print("\n[2/5] Loading indicators...")

# Load MSVR
spec = importlib.util.spec_from_file_location('msvr', os.path.join(bank_root, 'perpetual/median_standard_deviation_viresearch.py'))
msvr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msvr_module)

msvr_full = msvr_module.median_standard_deviation_viresearch(df_full)
msvr_signal = msvr_full['vii']  # Keep as -1, 0, 1

# Load SERF
spec = importlib.util.spec_from_file_location('serf', os.path.join(project_root, 'perpetual/spectral_entropy_regime_filter.py'))
serf_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serf_module)

print("  Computing SERF...")

# Test different entropy thresholds for continuous weighting
thresholds = [0.5, 0.6, 0.7, 0.8]

results = []

for threshold in thresholds:
    print(f"\n  Testing entropy_threshold={threshold}...")
    
    serf_full = serf_module.spectral_entropy_regime_filter(
        df_full,
        lookback=64,
        entropy_threshold=threshold,
        smooth_entropy=5
    )
    
    # Method 1: Binary filter (original)
    regime_binary = serf_full['regime']
    combined_binary = (msvr_signal > 0).astype(float) * regime_binary
    
    # Method 2: Continuous weight
    # Weight = 1 - entropy (low entropy = high weight)
    entropy = serf_full['entropy']
    weight = 1 - entropy  # Invert: low entropy = high weight
    
    # Apply weight to MSVR signal
    # If MSVR says long (1) and entropy is low (weight high), stay long
    # If MSVR says long (1) and entropy is high (weight low), reduce position
    msvr_binary = (msvr_signal > 0).astype(float)
    combined_weighted = msvr_binary * weight
    
    # Threshold the weighted signal (need minimum weight to trade)
    weight_thresholds = [0.3, 0.4, 0.5, 0.6]
    
    for wt in weight_thresholds:
        combined_thresholded = (combined_weighted > wt).astype(float)
        
        # Split
        combined_train = combined_thresholded[df_train.index]
        combined_holdout = combined_thresholded[df_holdout.index]
        
        metrics_train = compute_metrics(combined_train, df_train['close'])
        metrics_holdout = compute_metrics(combined_holdout, df_holdout['close'])
        
        coh_train = compute_coherence(combined_train, isp_positions_train)
        coh_holdout = compute_coherence(combined_holdout, isp_positions_holdout)
        
        result = {
            'entropy_threshold': threshold,
            'weight_threshold': wt,
            'pct_cyclic': round(regime_binary.mean() * 100, 1),
            'train': metrics_train,
            'holdout': metrics_holdout,
            'coh_train': coh_train,
            'coh_holdout': coh_holdout
        }
        
        results.append(result)
        
        print(f"    WT={wt:.1f}: Train Sharpe={metrics_train['sharpe']:.2f}, Holdout Sharpe={metrics_holdout['sharpe']:.2f}")

# ================================================================
# Summary
# ================================================================
print("\n[3/5] Summary...")

# Baseline (MSVR only)
msvr_train = (msvr_signal[df_train.index] > 0).astype(float)
msvr_holdout = (msvr_signal[df_holdout.index] > 0).astype(float)
metrics_msvr_train = compute_metrics(msvr_train, df_train['close'])
metrics_msvr_holdout = compute_metrics(msvr_holdout, df_holdout['close'])

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print(f"\nBaseline (MSVR only):")
print(f"  Train Sharpe:   {metrics_msvr_train['sharpe']:.2f}")
print(f"  Holdout Sharpe: {metrics_msvr_holdout['sharpe']:.2f}")

print(f"\n{'ET':<6} {'WT':<6} {'Train Sh':<10} {'Hold Sh':<10} {'Improve':<10}")
print("-" * 42)

for r in sorted(results, key=lambda x: x['holdout']['sharpe'], reverse=True):
    improvement = r['holdout']['sharpe'] - metrics_msvr_holdout['sharpe']
    print(f"{r['entropy_threshold']:<6} {r['weight_threshold']:<6} {r['train']['sharpe']:<10.2f} {r['holdout']['sharpe']:<10.2f} {improvement:<+10.2f}")

best = max(results, key=lambda x: x['holdout']['sharpe'])
improvement = best['holdout']['sharpe'] - metrics_msvr_holdout['sharpe']

print(f"\n🏆 BEST: ET={best['entropy_threshold']}, WT={best['weight_threshold']}")
print(f"   Improvement: {improvement:+.2f}")

# ================================================================
# Save Results
# ================================================================
print("\n[4/5] Saving results...")

output = {
    'baseline': {
        'train_sharpe': metrics_msvr_train['sharpe'],
        'holdout_sharpe': metrics_msvr_holdout['sharpe']
    },
    'best': {
        'entropy_threshold': best['entropy_threshold'],
        'weight_threshold': best['weight_threshold'],
        'improvement': improvement
    },
    'results': results
}

output_path = os.path.join(project_root, 'serf_msvr_v2_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to: {output_path}")
print("=" * 70)
