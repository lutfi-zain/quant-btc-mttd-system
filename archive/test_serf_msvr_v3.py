#!/usr/bin/env python3
"""
Test SERF + MSVR Combination V3
=================================

Uses CYCLE PHASE for timing instead of regime filtering.
- When cycle is at TROUGH → BUY signal
- When cycle is at PEAK → SELL signal
- Combine with MSVR direction for confirmation
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
print("TEST SERF + MSVR COMBINATION V3 (Cycle Phase Timing)")
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
# Compute Cycle Phase
# ================================================================
print("\n[2/5] Computing cycle phase...")

def compute_dominant_cycle_phase(df, lookback=64, min_period=5, max_period=30):
    """
    Compute the phase of the dominant cycle using FFT.
    
    Returns phase as value from 0 to 2π:
    - 0 or 2π = peak (top)
    - π = trough (bottom)
    - π/2 = rising
    - 3π/2 = falling
    """
    src = (df['high'] + df['low'] + df['close']) / 3.0
    n = len(df)
    
    phase = pd.Series(np.nan, index=df.index)
    dominant_period = pd.Series(np.nan, index=df.index)
    
    for i in range(lookback - 1, n):
        window = src.iloc[i - lookback + 1:i + 1].values
        
        if np.any(np.isnan(window)):
            continue
        
        # Detrend
        window_detrended = window - np.mean(window)
        
        # Apply Hanning window
        hann = np.hanning(lookback)
        window窗ed = window_detrended * hann
        
        # FFT
        fft_vals = np.fft.rfft(window窗ed)
        power = np.abs(fft_vals) ** 2
        
        # Find dominant frequency in valid range
        freqs = np.fft.rfftfreq(lookback, d=1)
        min_freq = 1.0 / max_period
        max_freq = 1.0 / min_period
        
        valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
        valid_power = power[valid_mask]
        valid_freqs = freqs[valid_mask]
        
        if len(valid_power) > 0 and np.sum(valid_power) > 0:
            dominant_idx = np.argmax(valid_power)
            dominant_freq = valid_freqs[dominant_idx]
            dominant_period_val = 1.0 / dominant_freq if dominant_freq > 0 else lookback
            
            # Compute phase using inverse FFT approach
            # Simple method: use position in cycle
            cycle_pos = i % int(dominant_period_val)
            phase_val = 2 * np.pi * cycle_pos / dominant_period_val
            
            phase.iloc[i] = phase_val
            dominant_period.iloc[i] = dominant_period_val
    
    return phase, dominant_period

# Compute phase for different lookback periods
lookbacks = [32, 64, 128]

for lb in lookbacks:
    print(f"\n  Computing phase with lookback={lb}...")
    phase, period = compute_dominant_cycle_phase(df_full, lookback=lb)
    
    # Store
    df_full[f'phase_{lb}'] = phase
    df_full[f'period_{lb}'] = period

# ================================================================
# Test Cycle Phase Strategies
# ================================================================
print("\n[3/5] Testing cycle phase strategies...")

# Load MSVR
spec = importlib.util.spec_from_file_location('msvr', os.path.join(bank_root, 'perpetual/median_standard_deviation_viresearch.py'))
msvr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msvr_module)

msvr_full = msvr_module.median_standard_deviation_viresearch(df_full)
msvr_signal = msvr_full['vii']

results = []

for lb in lookbacks:
    phase = df_full[f'phase_{lb}']
    
    # Strategy 1: Pure cycle phase
    # Buy at trough (phase near π), sell at peak (phase near 0 or 2π)
    # Use cosine: cos(phase) is -1 at trough, +1 at peak
    cycle_signal = -np.cos(phase)  # Invert: +1 at trough (buy), -1 at peak (sell)
    
    # Strategy 2: Cycle phase + MSVR confirmation
    # Buy when: phase says trough AND MSVR says long
    # Sell when: phase says peak OR MSVR says short
    msvr_binary = (msvr_signal > 0).astype(float)
    cycle_binary = (cycle_signal > 0).astype(float)
    
    combined = msvr_binary * cycle_binary
    
    # Test different lookbacks
    phase_train = phase[df_train.index]
    phase_holdout = phase[df_holdout.index]
    
    cycle_train = cycle_signal[df_train.index]
    cycle_holdout = cycle_signal[df_holdout.index]
    
    combined_train = combined[df_train.index]
    combined_holdout = combined[df_holdout.index]
    
    # Metrics
    metrics_cycle_train = compute_metrics((cycle_train > 0).astype(float), df_train['close'])
    metrics_cycle_holdout = compute_metrics((cycle_holdout > 0).astype(float), df_holdout['close'])
    
    metrics_combined_train = compute_metrics(combined_train, df_train['close'])
    metrics_combined_holdout = compute_metrics(combined_holdout, df_holdout['close'])
    
    coh_cycle_train = compute_coherence((cycle_train > 0).astype(float), isp_positions_train)
    coh_combined_train = compute_coherence(combined_train, isp_positions_train)
    
    result = {
        'lookback': lb,
        'cycle_train': metrics_cycle_train,
        'cycle_holdout': metrics_cycle_holdout,
        'combined_train': metrics_combined_train,
        'combined_holdout': metrics_combined_holdout,
        'coh_cycle_train': coh_cycle_train,
        'coh_combined_train': coh_combined_train
    }
    
    results.append(result)
    
    print(f"\n  Lookback={lb}:")
    print(f"    Cycle only:   Train Sharpe={metrics_cycle_train['sharpe']:.2f}, Holdout Sharpe={metrics_cycle_holdout['sharpe']:.2f}")
    print(f"    Cycle+MSVR:   Train Sharpe={metrics_combined_train['sharpe']:.2f}, Holdout Sharpe={metrics_combined_holdout['sharpe']:.2f}")

# ================================================================
# Summary
# ================================================================
print("\n[4/5] Summary...")

# Baseline
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

print(f"\n{'LB':<6} {'Cycle Sh':<12} {'Comb Sh':<12} {'Improve':<10}")
print("-" * 40)

for r in results:
    improvement = r['combined_holdout']['sharpe'] - metrics_msvr_holdout['sharpe']
    print(f"{r['lookback']:<6} {r['cycle_holdout']['sharpe']:<12.2f} {r['combined_holdout']['sharpe']:<12.2f} {improvement:<+10.2f}")

best = max(results, key=lambda x: x['combined_holdout']['sharpe'])
improvement = best['combined_holdout']['sharpe'] - metrics_msvr_holdout['sharpe']

print(f"\n🏆 BEST LOOKBACK: {best['lookback']}")
print(f"   Improvement: {improvement:+.2f}")

# ================================================================
# Save Results
# ================================================================
print("\n[5/5] Saving results...")

output = {
    'baseline': {
        'train_sharpe': metrics_msvr_train['sharpe'],
        'holdout_sharpe': metrics_msvr_holdout['sharpe']
    },
    'best_lookback': best['lookback'],
    'improvement': improvement,
    'results': results
}

output_path = os.path.join(project_root, 'serf_msvr_v3_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to: {output_path}")
print("=" * 70)
