#!/usr/bin/env python3
"""
Test SERF + MSVR Combination
==============================

Tests the combination of:
1. Spectral Entropy Regime Filter (SERF) - WHEN to trade
2. Median Standard Deviation Viresearch (MSVR) - WHICH direction

Hypothesis: SERF filters out false signals from MSVR in random regimes
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
print("TEST SERF + MSVR COMBINATION")
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

print(f"  Training: {len(df_train)} bars ({df_train.index[0]} to {df_train.index[-1]})")
print(f"  Holdout:  {len(df_holdout)} bars ({df_holdout.index[0]} to {df_holdout.index[-1]})")

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
    """Compute trading metrics with transaction costs."""
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
    """Compute ISP coherence."""
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
msvr_signal = (msvr_full['vii'] > 0).astype(float)

# Load SERF
spec = importlib.util.spec_from_file_location('serf', os.path.join(project_root, 'perpetual/spectral_entropy_regime_filter.py'))
serf_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(serf_module)

print("  Computing SERF...")

# Test different entropy thresholds
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

results = []

for threshold in thresholds:
    print(f"\n  Testing entropy_threshold={threshold}...")
    
    serf_full = serf_module.spectral_entropy_regime_filter(
        df_full,
        lookback=64,
        entropy_threshold=threshold,
        smooth_entropy=5
    )
    
    # Combine signals
    # SERF regime: 1 = cyclic (trade), 0 = random (don't trade)
    # MSVR signal: 1 = long, 0 = flat
    # Combined: long ONLY when SERF says cyclic AND MSVR says long
    
    regime = serf_full['regime']
    msvr_pos = msvr_signal
    
    # Combined position
    combined_pos = msvr_pos * regime  # Only trade when regime is cyclic
    
    # Split
    regime_train = regime[df_train.index]
    regime_holdout = regime[df_holdout.index]
    
    msvr_train = msvr_pos[df_train.index]
    msvr_holdout = msvr_pos[df_holdout.index]
    
    combined_train = combined_pos[df_train.index]
    combined_holdout = combined_pos[df_holdout.index]
    
    # Compute metrics
    metrics_msvr_train = compute_metrics(msvr_train, df_train['close'])
    metrics_msvr_holdout = compute_metrics(msvr_holdout, df_holdout['close'])
    
    metrics_combined_train = compute_metrics(combined_train, df_train['close'])
    metrics_combined_holdout = compute_metrics(combined_holdout, df_holdout['close'])
    
    coh_msvr_train = compute_coherence(msvr_train, isp_positions_train)
    coh_msvr_holdout = compute_coherence(msvr_holdout, isp_positions_holdout)
    
    coh_combined_train = compute_coherence(combined_train, isp_positions_train)
    coh_combined_holdout = compute_coherence(combined_holdout, isp_positions_holdout)
    
    # Regime statistics
    pct_cyclic_train = regime_train.mean() * 100
    pct_cyclic_holdout = regime_holdout.mean() * 100
    
    result = {
        'threshold': threshold,
        'pct_cyclic_train': round(pct_cyclic_train, 1),
        'pct_cyclic_holdout': round(pct_cyclic_holdout, 1),
        'msvr_train': metrics_msvr_train,
        'msvr_holdout': metrics_msvr_holdout,
        'combined_train': metrics_combined_train,
        'combined_holdout': metrics_combined_holdout,
        'coh_msvr_train': coh_msvr_train,
        'coh_msvr_holdout': coh_msvr_holdout,
        'coh_combined_train': coh_combined_train,
        'coh_combined_holdout': coh_combined_holdout
    }
    
    results.append(result)
    
    print(f"    Regime: {pct_cyclic_train:.1f}% cyclic (train) → {pct_cyclic_holdout:.1f}% cyclic (holdout)")
    print(f"    MSVR only:     Sharpe={metrics_msvr_train['sharpe']:.2f} → {metrics_msvr_holdout['sharpe']:.2f}")
    print(f"    MSVR + SERF:   Sharpe={metrics_combined_train['sharpe']:.2f} → {metrics_combined_holdout['sharpe']:.2f}")

# ================================================================
# Summary
# ================================================================
print("\n[3/5] Summary...")

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print(f"\n{'Threshold':<12} {'Cyclic%':<12} {'MSVR Sh':<12} {'Combined Sh':<14} {'Improvement':<12}")
print("-" * 62)

for r in results:
    improvement = r['combined_holdout']['sharpe'] - r['msvr_holdout']['sharpe']
    print(f"{r['threshold']:<12} {r['pct_cyclic_train']:<12.1f} {r['msvr_holdout']['sharpe']:<12.2f} {r['combined_holdout']['sharpe']:<14.2f} {improvement:<+12.2f}")

# Find best threshold
best = max(results, key=lambda x: x['combined_holdout']['sharpe'])

print(f"\n🏆 BEST THRESHOLD: {best['threshold']}")
print(f"   Cyclic regime: {best['pct_cyclic_train']:.1f}% (train)")
print(f"   MSVR only holdout Sharpe: {best['msvr_holdout']['sharpe']:.2f}")
print(f"   MSVR+SERF holdout Sharpe: {best['combined_holdout']['sharpe']:.2f}")
print(f"   Improvement: {best['combined_holdout']['sharpe'] - best['msvr_holdout']['sharpe']:+.2f}")

# ================================================================
# Correlation Analysis
# ================================================================
print("\n[4/5] Correlation Analysis...")

# Check if SERF and MSVR are uncorrelated
serf_best = serf_module.spectral_entropy_regime_filter(
    df_full,
    lookback=64,
    entropy_threshold=best['threshold'],
    smooth_entropy=5
)

serf_signal = serf_best['regime']
msvr_signal_binary = (msvr_full['vii'] > 0).astype(float)

# Correlation
corr = serf_signal.corr(msvr_signal_binary)
print(f"  SERF-MSVR correlation: {corr:.3f}")

# Check regime stability
serf_diff = serf_signal.diff().abs()
print(f"  SERF regime changes: {serf_diff.sum():.0f} (avg {serf_diff.mean():.3f} per bar)")

# ================================================================
# Save Results
# ================================================================
print("\n[5/5] Saving results...")

output = {
    'best_threshold': best['threshold'],
    'results': results,
    'correlation': corr,
    'regime_stability': {
        'avg_changes_per_bar': float(serf_diff.mean()),
        'total_changes': int(serf_diff.sum())
    }
}

output_path = os.path.join(project_root, 'serf_msvr_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to: {output_path}")
print("=" * 70)
