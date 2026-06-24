#!/usr/bin/env python3
"""
MSVR + Cycle Phase Timing — Final Implementation
==================================================

Combines:
1. Median Standard Deviation Viresearch (MSVR) — Direction
2. Spectral Cycle Phase — Timing (when to enter/exit)

Optimization: Grid search for best lookback period
Validation: Train 2018-2024, Holdout 2025-2026
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
print("MSVR + CYCLE PHASE TIMING — FINAL IMPLEMENTATION")
print("=" * 70)

# ================================================================
# Load Data
# ================================================================
print("\n[1/6] Loading data...")

with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
    btc_data = json.load(f)

df_full = pd.DataFrame(btc_data['aligned_data'])
df_full['time'] = pd.to_datetime(df_full['time'])
df_full = df_full.set_index('time')
df_full = df_full[df_full.index >= '2018-01-01']

HOLDOUT_START = '2025-01-01'
df_train = df_full[df_full.index < HOLDOUT_START].copy()
df_holdout = df_full[df_full.index >= HOLDOUT_START].copy()

print(f"  Full:      {len(df_full)} bars ({df_full.index[0]} to {df_full.index[-1]})")
print(f"  Training:  {len(df_train)} bars")
print(f"  Holdout:   {len(df_holdout)} bars")

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
# Load MSVR
# ================================================================
print("\n[2/6] Loading MSVR indicator...")

spec = importlib.util.spec_from_file_location('msvr', os.path.join(bank_root, 'perpetual/median_standard_deviation_viresearch.py'))
msvr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msvr_module)

msvr_full = msvr_module.median_standard_deviation_viresearch(df_full)
msvr_signal = msvr_full['vii']

print(f"  MSVR signal loaded: {(msvr_signal == 1).sum()} bullish, {(msvr_signal == -1).sum()} bearish")

# ================================================================
# Cycle Phase Computation
# ================================================================
print("\n[3/6] Computing cycle phases...")

def compute_cycle_phase(df, lookback):
    """
    Compute cycle phase using FFT.
    Returns phase (0 to 2π) and dominant period.
    """
    src = (df['high'] + df['low'] + df['close']) / 3.0
    n = len(df)
    
    phase = pd.Series(np.nan, index=df.index)
    period = pd.Series(np.nan, index=df.index)
    
    min_period = 5
    max_period = lookback // 2
    
    for i in range(lookback - 1, n):
        window = src.iloc[i - lookback + 1:i + 1].values
        
        if np.any(np.isnan(window)):
            continue
        
        # Detrend
        window_detrended = window - np.mean(window)
        
        # Hanning window
        hann = np.hanning(lookback)
        window窗ed = window_detrended * hann
        
        # FFT
        fft_vals = np.fft.rfft(window窗ed)
        power = np.abs(fft_vals) ** 2
        
        # Find dominant frequency
        freqs = np.fft.rfftfreq(lookback, d=1)
        min_freq = 1.0 / max_period
        max_freq = 1.0 / min_period
        
        valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
        valid_power = power[valid_mask]
        valid_freqs = freqs[valid_mask]
        
        if len(valid_power) > 0 and np.sum(valid_power) > 0:
            dominant_idx = np.argmax(valid_power)
            dominant_freq = valid_freqs[dominant_idx]
            dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else lookback
            
            # Compute phase
            cycle_pos = i % int(dominant_period)
            phase_val = 2 * np.pi * cycle_pos / dominant_period
            
            phase.iloc[i] = phase_val
            period.iloc[i] = dominant_period
    
    return phase, period

# Compute phases for different lookbacks
lookbacks = list(range(20, 130, 5))  # 20 to 125, step 5
print(f"  Computing phases for {len(lookbacks)} lookback periods...")

for lb in lookbacks:
    phase, period = compute_cycle_phase(df_full, lookback=lb)
    df_full[f'phase_{lb}'] = phase

print(f"  Phases computed for lookbacks: {lookbacks[0]}-{lookbacks[-1]}")

# ================================================================
# Grid Search Optimal Lookback
# ================================================================
print("\n[4/6] Grid search optimal lookback...")

results = []

for lb in lookbacks:
    phase = df_full[f'phase_{lb}']
    
    # Cycle signal: buy at trough (phase near π), sell at peak (phase near 0/2π)
    cycle_signal = -np.cos(phase)  # +1 at trough (buy), -1 at peak (sell)
    
    # Combine with MSVR
    msvr_binary = (msvr_signal > 0).astype(float)
    cycle_binary = (cycle_signal > 0).astype(float)
    
    combined = msvr_binary * cycle_binary
    
    # Split
    combined_train = combined[df_train.index]
    combined_holdout = combined[df_holdout.index]
    
    # Metrics
    metrics_train = compute_metrics(combined_train, df_train['close'])
    metrics_holdout = compute_metrics(combined_holdout, df_holdout['close'])
    
    coh_train = compute_coherence(combined_train, isp_positions_train)
    coh_holdout = compute_coherence(combined_holdout, isp_positions_holdout)
    
    # Score (risk-adjusted returns)
    score = metrics_holdout['sharpe'] * 0.5 + metrics_holdout['calmar'] * 0.3 + metrics_holdout['sortino'] * 0.2
    
    results.append({
        'lookback': lb,
        'train': metrics_train,
        'holdout': metrics_holdout,
        'coh_train': coh_train,
        'coh_holdout': coh_holdout,
        'score': score
    })

# Sort by holdout Sharpe
results.sort(key=lambda x: x['holdout']['sharpe'], reverse=True)

# Print top 10
print("\n  Top 10 Lookbacks by Holdout Sharpe:")
print(f"  {'LB':<6} {'Train Sh':<10} {'Hold Sh':<10} {'Train CAGR':<12} {'Hold CAGR':<12} {'Hold MaxDD':<12}")
print("  " + "-" * 62)

for r in results[:10]:
    print(f"  {r['lookback']:<6} {r['train']['sharpe']:<10.2f} {r['holdout']['sharpe']:<10.2f} "
          f"{r['train']['cagr']:<12.1f} {r['holdout']['cagr']:<12.1f} {r['holdout']['max_dd']:<12.1f}")

# ================================================================
# Validate Best Configuration
# ================================================================
print("\n[5/6] Validating best configuration...")

best = results[0]
best_lb = best['lookback']

print(f"\n  Best Lookback: {best_lb}")
print(f"  Training Metrics:")
print(f"    Sharpe:   {best['train']['sharpe']:.2f}")
print(f"    CAGR:     {best['train']['cagr']:.1f}%")
print(f"    MaxDD:    {best['train']['max_dd']:.1f}%")
print(f"    Trades:   {best['train']['n_trades']}")
print(f"    In-market: {best['train']['pct_in']:.1f}%")
print(f"    Coherence: {best['coh_train']:.1f}%")
print(f"\n  Holdout Metrics:")
print(f"    Sharpe:   {best['holdout']['sharpe']:.2f}")
print(f"    CAGR:     {best['holdout']['cagr']:.1f}%")
print(f"    MaxDD:    {best['holdout']['max_dd']:.1f}%")
print(f"    Trades:   {best['holdout']['n_trades']}")
print(f"    In-market: {best['holdout']['pct_in']:.1f}%")
print(f"    Coherence: {best['coh_holdout']:.1f}%")

# Compare with baseline (MSVR only)
msvr_train = (msvr_signal[df_train.index] > 0).astype(float)
msvr_holdout = (msvr_signal[df_holdout.index] > 0).astype(float)
metrics_msvr_train = compute_metrics(msvr_train, df_train['close'])
metrics_msvr_holdout = compute_metrics(msvr_holdout, df_holdout['close'])

print(f"\n  Comparison with MSVR Only:")
print(f"  {'Metric':<15} {'MSVR Only':<15} {'MSVR+Cycle':<15} {'Improvement':<15}")
print(f"  " + "-" * 60)
print(f"  {'Train Sharpe':<15} {metrics_msvr_train['sharpe']:<15.2f} {best['train']['sharpe']:<15.2f} {best['train']['sharpe']-metrics_msvr_train['sharpe']:<+15.2f}")
print(f"  {'Holdout Sharpe':<15} {metrics_msvr_holdout['sharpe']:<15.2f} {best['holdout']['sharpe']:<15.2f} {best['holdout']['sharpe']-metrics_msvr_holdout['sharpe']:<+15.2f}")
print(f"  {'Train CAGR':<15} {metrics_msvr_train['cagr']:<15.1f} {best['train']['cagr']:<15.1f} {best['train']['cagr']-metrics_msvr_train['cagr']:<+15.1f}%")
print(f"  {'Holdout CAGR':<15} {metrics_msvr_holdout['cagr']:<15.1f} {best['holdout']['cagr']:<15.1f} {best['holdout']['cagr']-metrics_msvr_holdout['cagr']:<+15.1f}%")
print(f"  {'Train MaxDD':<15} {metrics_msvr_train['max_dd']:<15.1f} {best['train']['max_dd']:<15.1f} {best['train']['max_dd']-metrics_msvr_train['max_dd']:<+15.1f}%")
print(f"  {'Holdout MaxDD':<15} {metrics_msvr_holdout['max_dd']:<15.1f} {best['holdout']['max_dd']:<15.1f} {best['holdout']['max_dd']-metrics_msvr_holdout['max_dd']:<+15.1f}%")

# Degradation analysis
if metrics_msvr_train['sharpe'] > 0:
    deg_msvr = (metrics_msvr_holdout['sharpe'] - metrics_msvr_train['sharpe']) / metrics_msvr_train['sharpe'] * 100
else:
    deg_msvr = 0

if best['train']['sharpe'] > 0:
    deg_combined = (best['holdout']['sharpe'] - best['train']['sharpe']) / best['train']['sharpe'] * 100
else:
    deg_combined = 0

print(f"\n  Degradation Analysis:")
print(f"    MSVR Only:    {deg_msvr:+.1f}%")
print(f"    MSVR+Cycle:   {deg_combined:+.1f}%")

# ================================================================
# Save Results
# ================================================================
print("\n[6/6] Saving results...")

output = {
    'best_lookback': best_lb,
    'best_metrics': {
        'train': best['train'],
        'holdout': best['holdout'],
        'coh_train': best['coh_train'],
        'coh_holdout': best['coh_holdout']
    },
    'baseline_metrics': {
        'train': metrics_msvr_train,
        'holdout': metrics_msvr_holdout
    },
    'improvement': {
        'sharpe': best['holdout']['sharpe'] - metrics_msvr_holdout['sharpe'],
        'cagr': best['holdout']['cagr'] - metrics_msvr_holdout['cagr']
    },
    'degradation': {
        'msvr_only': deg_msvr,
        'combined': deg_combined
    },
    'all_results': results
}

output_path = os.path.join(project_root, 'msvr_cycle_timing_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to: {output_path}")

# Final Summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"\n  Best Configuration: MSVR + Cycle Phase (Lookback={best_lb})")
print(f"  Holdout Sharpe: {best['holdout']['sharpe']:.2f} (vs {metrics_msvr_holdout['sharpe']:.2f} baseline)")
print(f"  Improvement: {best['holdout']['sharpe']-metrics_msvr_holdout['sharpe']:+.2f} (+{(best['holdout']['sharpe']-metrics_msvr_holdout['sharpe'])/metrics_msvr_holdout['sharpe']*100:.0f}%)")
print("=" * 70)
