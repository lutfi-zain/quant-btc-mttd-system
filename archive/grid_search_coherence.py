#!/usr/bin/env python3
"""
MTTD Grid Search - Optimize for Time-Coherence
Focus on reducing trade count to match ISP's ~13 transitions.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from indicators_helper import *
import importlib.util

print("=" * 70)
print("MTTD GRID SEARCH - TIME-COHERENCE OPTIMIZER")
print("=" * 70)

# Load data
with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
    btc_data = json.load(f)

df = pd.DataFrame(btc_data['aligned_data'])
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = df[df.index >= '2018-01-01']

# Load ISP benchmark
isp_df = pd.read_csv(os.path.join(project_root, '..', 'quant-technical-indicator-bank', 'isp-signals-btcusd-2026-06-13.csv'))
isp_df.columns = isp_df.columns.str.strip()
isp_df['Date'] = pd.to_datetime(isp_df['Date'])
isp_df = isp_df.set_index('Date')

# Create ISP positions series
isp_positions = pd.Series(0.0, index=df.index)
for date, row in isp_df.iterrows():
    if date in isp_positions.index:
        if row['Regime'] in ['Weak Bull', 'Strong Bull']:
            isp_positions.loc[date:] = 1.0
        elif row['Regime'] == 'Neutral':
            isp_positions.loc[date:] = 0.0

isp_transitions = (isp_positions.diff() != 0).sum()
print(f"ISP: {isp_transitions} transitions, {(isp_positions == 1.0).mean()*100:.1f}% in position")

# Load indicators with default params
TOP_INDICATORS = [
    ("Polynomial Deviation Bands", "perpetual", "polynomial_deviation_bands.py", "polynomial_deviation_bands"),
    ("Gaussian Smooth Trend | QuantEdgeB", "perpetual", "gaussian_smooth_trend_quantedgeb.py", "gaussian_smooth_trend_quantedgeb"),
    ("alma lag | viResearch", "perpetual", "alma_lag_viresearch.py", "alma_lag_viresearch"),
    ("Adaptive Regime Cloud", "perpetual", "adaptive_regime_cloud.py", "adaptive_regime_cloud"),
    ("Root Mean Square Deviation Trend", "perpetual", "root_mean_square_deviation_trend.py", "root_mean_square_deviation_trend"),
    ("P-Motion Trend | QuantEdgeB", "perpetual", "p_motion_trend_quantedgeb.py", "p_motion_trend_quantedgeb"),
    ("Z SMMA | QuantEdgeB", "oscillator", "z_smma_quantedgeb.py", "z_smma_quantedgeb"),
    ("Median RSI SD | QuantEdgeB", "oscillator", "median_rsi_sd_quantedgeb.py", "median_rsi_sd_quantedgeb"),
]

def load_indicator_func(indicator_file, category):
    module_path = os.path.join(project_root, category, indicator_file)
    spec = importlib.util.spec_from_file_location(indicator_file.replace('.py', ''), module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, indicator_file.replace('.py', ''))

def detect_direction(res_df):
    for col in ['dir', 'sig', 'direction', 'vii', 'qb', 'st_direction', 'trend_direction', 'trend']:
        if col in res_df.columns:
            return res_df[col]
    if 'long_signal' in res_df.columns and 'short_signal' in res_df.columns:
        direction = pd.Series(0.0, index=res_df.index)
        curr = 0.0
        for i in range(len(res_df)):
            l = bool(res_df['long_signal'].iloc[i]) if not pd.isna(res_df['long_signal'].iloc[i]) else False
            s = bool(res_df['short_signal'].iloc[i]) if not pd.isna(res_df['short_signal'].iloc[i]) else False
            if l and not s: curr = 1.0
            elif s and not l: curr = -1.0
            direction.iloc[i] = curr
        return direction
    if 'in_long_position' in res_df.columns and 'in_short_position' in res_df.columns:
        direction = pd.Series(0.0, index=res_df.index)
        direction[res_df['in_long_position'] == 1] = 1.0
        direction[res_df['in_short_position'] == 1] = -1.0
        return direction
    for col in res_df.columns:
        if 'direction' in col.lower() or 'signal' in col.lower() or 'trend' in col.lower():
            if len(res_df[col].dropna().unique()) <= 10:
                return res_df[col]
    return None

def compute_coherence(positions, benchmark):
    """Compute time-coherence with ISP benchmark."""
    aligned = pd.DataFrame({'system': positions, 'benchmark': benchmark}).dropna()
    if len(aligned) == 0:
        return 0.0
    matches = (aligned['system'] == aligned['benchmark']).sum()
    return matches / len(aligned) * 100

def compute_performance(positions, prices):
    """Compute performance metrics."""
    returns = prices.pct_change()
    strategy_returns = returns * positions.shift(1)
    strategy_returns = strategy_returns.dropna()
    equity = (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1 if years > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min() * 100
    return {'cagr': cagr * 100, 'sharpe': sharpe, 'max_dd': max_dd, 'pct_in': positions.mean() * 100}

# Compute all indicators
print("\nComputing indicators...")
indicator_directions = {}
for name, category, filename, func_name in TOP_INDICATORS:
    try:
        func = load_indicator_func(filename, category)
        res = func(df)
        direction = detect_direction(res)
        if direction is not None:
            binary = direction.apply(lambda x: 1.0 if x > 0 else -1.0)
            indicator_directions[func_name] = binary
            print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")

signal_df = pd.DataFrame(indicator_directions)
n_indicators = len(signal_df.columns)
print(f"\nActive indicators: {n_indicators}")

# ================================================================
# Grid Search: Threshold + EMA + Min Hold Period
# ================================================================
print("\n[Grid Search] Testing threshold, EMA, and min_hold combinations...")
print("-" * 70)

results = []

# Wide range of parameters
threshold_range = np.arange(-0.5, 0.8, 0.05)
ema_range = [10, 15, 20, 30, 50, 75, 100]
min_hold_range = [5, 10, 15, 20, 30, 50, 75, 100]

total = len(threshold_range) * len(ema_range) * len(min_hold_range)
count = 0

for threshold in threshold_range:
    for ema_len in ema_range:
        for min_hold in min_hold_range:
            count += 1
            
            # Compute ensemble
            ensemble_raw = signal_df.mean(axis=1)
            smoothed = ensemble_raw.ewm(span=ema_len).mean()
            
            # Convert to positions with minimum hold period
            raw_positions = pd.Series(0.0, index=df.index)
            raw_positions[smoothed > threshold] = 1.0
            
            # Apply minimum hold period
            positions = raw_positions.copy()
            last_change_idx = 0
            last_pos = positions.iloc[0]
            
            for i in range(1, len(positions)):
                if positions.iloc[i] != last_pos:
                    if i - last_change_idx >= min_hold:
                        last_change_idx = i
                        last_pos = positions.iloc[i]
                    else:
                        positions.iloc[i] = last_pos
            
            # Metrics
            coh = compute_coherence(positions, isp_positions)
            perf = compute_performance(positions, df['close'])
            trades = (positions.diff().abs() > 0).sum()
            
            # Score: heavily penalize low coherence and high trade count
            coherence_penalty = max(0, 90 - coh) * 3
            trade_penalty = max(0, trades - 20) * 2
            score = coh * 0.8 + perf['sharpe'] * 20 - coherence_penalty - trade_penalty
            
            results.append({
                'threshold': threshold,
                'ema_len': ema_len,
                'min_hold': min_hold,
                'coherence': coh,
                'cagr': perf['cagr'],
                'sharpe': perf['sharpe'],
                'max_dd': perf['max_dd'],
                'trades': trades,
                'pct_in': perf['pct_in'],
                'score': score
            })

# Sort by score
results.sort(key=lambda x: x['score'], reverse=True)

print(f"\nTotal combinations tested: {total}")
print("\nTop 15 Parameter Combinations:")
print("-" * 120)
print(f"{'Rank':>4} | {'Thresh':>6} | {'EMA':>4} | {'Hold':>5} | {'Coh%':>6} | {'CAGR%':>8} | {'Sharpe':>7} | {'MaxDD%':>8} | {'Trades':>6} | {'InPos%':>7} | {'Score':>7}")
print("-" * 120)

for i, r in enumerate(results[:15]):
    marker = " ★" if r['coherence'] >= 80 else ""
    print(f"{i+1:>4} | {r['threshold']:>6.2f} | {r['ema_len']:>4} | {r['min_hold']:>5} | {r['coherence']:>6.2f} | {r['cagr']:>8.2f} | {r['sharpe']:>7.2f} | {r['max_dd']:>8.2f} | {r['trades']:>6} | {r['pct_in']:>7.2f} | {r['score']:>7.2f}{marker}")

# Find best for different priorities
best_coherence = max(results, key=lambda x: x['coherence'])
best_sharpe = max(results, key=lambda x: x['sharpe'])
best_balance = results[0]  # Already sorted by score

print("\n" + "=" * 70)
print("BEST PARAMETERS BY PRIORITY:")
print("=" * 70)

print(f"\n1. Best Coherence:")
print(f"   Threshold: {best_coherence['threshold']:.2f}, EMA: {best_coherence['ema_len']}, MinHold: {best_coherence['min_hold']}")
print(f"   Coherence: {best_coherence['coherence']:.2f}%, CAGR: {best_coherence['cagr']:.2f}%, Sharpe: {best_coherence['sharpe']:.2f}")
print(f"   Max DD: {best_coherence['max_dd']:.2f}%, Trades: {best_coherence['trades']}, InPos: {best_coherence['pct_in']:.2f}%")

print(f"\n2. Best Sharpe:")
print(f"   Threshold: {best_sharpe['threshold']:.2f}, EMA: {best_sharpe['ema_len']}, MinHold: {best_sharpe['min_hold']}")
print(f"   Coherence: {best_sharpe['coherence']:.2f}%, CAGR: {best_sharpe['cagr']:.2f}%, Sharpe: {best_sharpe['sharpe']:.2f}")
print(f"   Max DD: {best_sharpe['max_dd']:.2f}%, Trades: {best_sharpe['trades']}, InPos: {best_sharpe['pct_in']:.2f}%")

print(f"\n3. Best Balanced (Score):")
print(f"   Threshold: {best_balance['threshold']:.2f}, EMA: {best_balance['ema_len']}, MinHold: {best_balance['min_hold']}")
print(f"   Coherence: {best_balance['coherence']:.2f}%, CAGR: {best_balance['cagr']:.2f}%, Sharpe: {best_balance['sharpe']:.2f}")
print(f"   Max DD: {best_balance['max_dd']:.2f}%, Trades: {best_balance['trades']}, InPos: {best_balance['pct_in']:.2f}%")

# Save results
output = {
    'isp_transitions': int(isp_transitions),
    'best_coherence': best_coherence,
    'best_sharpe': best_sharpe,
    'best_balance': best_balance,
    'top_15': results[:15]
}

with open(os.path.join(project_root, 'grid_search_coherence_results.json'), 'w') as f:
    json.dump(output, f, indent=2)

print("\nResults saved to: grid_search_coherence_results.json")
