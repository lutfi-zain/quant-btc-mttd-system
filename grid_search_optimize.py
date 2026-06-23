#!/usr/bin/env python3
"""
MTTD Grid Search Optimization
Searches for optimal parameters to maximize time-coherence and performance.
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

# ================================================================
# Load Data
# ================================================================
print("=" * 70)
print("MTTD GRID SEARCH OPTIMIZATION")
print("=" * 70)

with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
    btc_data = json.load(f)

df = pd.DataFrame(btc_data['aligned_data'])
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = df[df.index >= '2018-01-01']

print(f"Data loaded: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

# Load ISP benchmark
isp_df = pd.read_csv(os.path.join(project_root, '..', 'quant-technical-indicator-bank', 'isp-signals-btcusd-2026-06-13.csv'))
isp_df.columns = isp_df.columns.str.strip()
isp_df['Date'] = pd.to_datetime(isp_df['Date'])
isp_df = isp_df.set_index('Date')
isp_positions = pd.Series(0.0, index=df.index)
for date in isp_df.index:
    if date in isp_positions.index:
        isp_positions.loc[date] = 1.0
isp_positions = isp_positions.ffill().fillna(0.0)

# ================================================================
# Load Indicators from library.yaml
# ================================================================
lib_path = os.path.join(project_root, 'library.yaml')
with open(lib_path, 'r', encoding='utf-8') as f:
    import yaml
    library = yaml.safe_load(f)

# Top 10 indicators with parameter ranges to search
TOP_INDICATORS = [
    ("Polynomial Deviation Bands", "perpetual", "polynomial_deviation_bands.py", "polynomial_deviation_bands",
     {"length": [20, 30, 40, 50, 60], "std_dev": [1.5, 2.0, 2.5, 3.0]}),
    ("Gaussian Smooth Trend | QuantEdgeB", "perpetual", "gaussian_smooth_trend_quantedgeb.py", "gaussian_smooth_trend_quantedgeb",
     {"length": [20, 30, 40, 50, 60], "smooth": [5, 10, 15, 20]}),
    ("alma lag | viResearch", "perpetual", "alma_lag_viresearch.py", "alma_lag_viresearch",
     {"length": [20, 30, 40, 50, 60]}),
    ("Adaptive Regime Cloud", "perpetual", "adaptive_regime_cloud.py", "adaptive_regime_cloud",
     {"length": [20, 30, 40, 50, 60]}),
    ("Root Mean Square Deviation Trend", "perpetual", "root_mean_square_deviation_trend.py", "root_mean_square_deviation_trend",
     {"length": [20, 30, 40, 50, 60]}),
    ("P-Motion Trend | QuantEdgeB", "perpetual", "p_motion_trend_quantedgeb.py", "p_motion_trend_quantedgeb",
     {"length": [20, 30, 40, 50, 60]}),
    ("Z SMMA | QuantEdgeB", "oscillator", "z_smma_quantedgeb.py", "z_smma_quantedgeb",
     {"length": [20, 30, 40, 50, 60]}),
    ("Median RSI SD | QuantEdgeB", "oscillator", "median_rsi_sd_quantedgeb.py", "median_rsi_sd_quantedgeb",
     {"length": [20, 30, 40, 50, 60]}),
    ("DEMA Adjusted Average True Range", "perpetual", "dema_adjusted_average_true_range.py", "dema_adjusted_average_true_range",
     {"length": [20, 30, 40, 50, 60]}),
    ("Kalman Filtered RSI Oscillator", "oscillator", "kalman_filtered_rsi_oscillator.py", "kalman_filtered_rsi_oscillator",
     {"length": [20, 30, 40, 50, 60]}),
]

# ================================================================
# Helper Functions
# ================================================================
def load_indicator_func(indicator_file, category):
    """Load indicator function from file."""
    module_path = os.path.join(project_root, category, indicator_file)
    spec = importlib.util.spec_from_file_location(indicator_file.replace('.py', ''), module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func_name = indicator_file.replace('.py', '')
    return getattr(module, func_name)

def detect_direction(res_df):
    """Detect direction from indicator output."""
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
    
    # Equity curve
    equity = (1 + strategy_returns).cumprod()
    
    # CAGR
    years = len(strategy_returns) / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1 if years > 0 else 0
    
    # Sharpe
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
    
    # Max Drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min() * 100
    
    # Sortino
    downside = strategy_returns[strategy_returns < 0]
    downside_std = downside.std() * np.sqrt(365)
    sortino = strategy_returns.mean() / downside_std if downside_std > 0 else 0
    
    # Win rate (daily)
    win_rate = (strategy_returns > 0).sum() / len(strategy_returns) * 100
    
    # Trade count (transitions)
    trades = (positions.diff().abs() > 0).sum()
    
    return {
        'cagr': cagr * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'trades': int(trades),
        'pct_in': positions.mean() * 100
    }

# ================================================================
# Grid Search: Individual Indicator Parameters
# ================================================================
print("\n[Phase 1] Grid search individual indicator parameters...")
print("-" * 70)

best_params_per_indicator = {}

for name, category, filename, func_name, param_ranges in TOP_INDICATORS:
    print(f"\n  Testing: {name}")
    func = load_indicator_func(filename, category)
    
    # Generate parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    combinations = list(product(*param_values))
    
    best_coherence = 0
    best_params = {}
    
    for combo in combinations:
        params = dict(zip(param_names, combo))
        
        try:
            res = func(df, **params)
            direction = detect_direction(res)
            
            if direction is not None:
                # Convert to binary
                binary = direction.apply(lambda x: 1.0 if x > 0 else -1.0)
                # Coherence with ISP
                coh = compute_coherence(binary, isp_positions)
                
                if coh > best_coherence:
                    best_coherence = coh
                    best_params = params.copy()
        except Exception as e:
            pass
    
    best_params_per_indicator[func_name] = {
        'name': name,
        'params': best_params,
        'coherence': best_coherence
    }
    print(f"    Best params: {best_params} -> Coherence: {best_coherence:.2f}%")

# ================================================================
# Grid Search: Ensemble Threshold + EMA Length
# ================================================================
print("\n[Phase 2] Computing indicators with best params...")
print("-" * 70)

# Compute all indicators with best params
indicator_directions = {}
for name, category, filename, func_name, _ in TOP_INDICATORS:
    info = best_params_per_indicator[func_name]
    func = load_indicator_func(filename, category)
    
    try:
        res = func(df, **info['params'])
        direction = detect_direction(res)
        
        if direction is not None:
            binary = direction.apply(lambda x: 1.0 if x > 0 else -1.0)
            indicator_directions[func_name] = binary
            print(f"  ✓ {name}: coherence={info['coherence']:.2f}%")
        else:
            print(f"  ✗ {name}: no direction signal")
    except Exception as e:
        print(f"  ✗ {name}: {e}")

# Build signal matrix
signal_df = pd.DataFrame(indicator_directions)
n_indicators = len(signal_df.columns)
print(f"\n  Active indicators: {n_indicators}")

print("\n[Phase 3] Grid search ensemble parameters...")
print("-" * 70)

# Grid search threshold and EMA length
best_score = 0
best_threshold = 0
best_ema = 5
results = []

threshold_range = np.arange(-0.5, 0.6, 0.05)
ema_range = [3, 5, 7, 10, 15, 20]

total = len(threshold_range) * len(ema_range)
count = 0

for threshold in threshold_range:
    for ema_len in ema_range:
        count += 1
        
        # Compute ensemble
        ensemble_raw = signal_df.mean(axis=1)
        smoothed = ensemble_raw.ewm(span=ema_len).mean()
        
        # Convert to positions
        positions = pd.Series(0.0, index=df.index)
        positions[smoothed > threshold] = 1.0
        
        # Metrics
        coh = compute_coherence(positions, isp_positions)
        perf = compute_performance(positions, df['close'])
        
        # Score: prioritize coherence (90%) and performance (10%)
        # Target: coherence >= 95%
        coherence_penalty = max(0, 95 - coh) * 2  # Heavy penalty for low coherence
        score = coh * 0.9 + perf['sharpe'] * 10 - coherence_penalty
        
        results.append({
            'threshold': threshold,
            'ema_len': ema_len,
            'coherence': coh,
            'cagr': perf['cagr'],
            'sharpe': perf['sharpe'],
            'sortino': perf['sortino'],
            'max_dd': perf['max_dd'],
            'trades': perf['trades'],
            'pct_in': perf['pct_in'],
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_ema = ema_len
            best_coh = coh
            best_perf = perf

# ================================================================
# Results
# ================================================================
print("\n[Phase 4] Results...")
print("=" * 70)

# Sort by score
results.sort(key=lambda x: x['score'], reverse=True)

print("\nTop 10 Parameter Combinations:")
print("-" * 100)
print(f"{'Rank':>4} | {'Threshold':>8} | {'EMA':>4} | {'Coh%':>6} | {'CAGR%':>8} | {'Sharpe':>7} | {'MaxDD%':>8} | {'Trades':>6} | {'InPos%':>7} | {'Score':>7}")
print("-" * 100)

for i, r in enumerate(results[:10]):
    print(f"{i+1:>4} | {r['threshold']:>8.2f} | {r['ema_len']:>4} | {r['coherence']:>6.2f} | {r['cagr']:>8.2f} | {r['sharpe']:>7.2f} | {r['max_dd']:>8.2f} | {r['trades']:>6} | {r['pct_in']:>7.2f} | {r['score']:>7.2f}")

print("\n" + "=" * 70)
print(f"BEST PARAMETERS:")
print(f"  Threshold: {best_threshold:.2f}")
print(f"  EMA Length: {best_ema}")
print(f"  Coherence: {best_coh:.2f}%")
print(f"  CAGR: {best_perf['cagr']:.2f}%")
print(f"  Sharpe: {best_perf['sharpe']:.2f}")
print(f"  Max DD: {best_perf['max_dd']:.2f}%")
print(f"  Trades: {best_perf['trades']}")
print(f"  Position %: {best_perf['pct_in']:.2f}%")
print("=" * 70)

# Save results
output = {
    'best_params': {
        'threshold': best_threshold,
        'ema_len': best_ema,
        'indicator_params': {k: v['params'] for k, v in best_params_per_indicator.items()}
    },
    'best_metrics': {
        'coherence': best_coh,
        'cagr': best_perf['cagr'],
        'sharpe': best_perf['sharpe'],
        'sortino': best_perf['sortino'],
        'max_dd': best_perf['max_dd'],
        'trades': best_perf['trades'],
        'pct_in': best_perf['pct_in']
    },
    'top_10': results[:10]
}

with open(os.path.join(project_root, 'grid_search_results.json'), 'w') as f:
    json.dump(output, f, indent=2)

print("\nResults saved to: grid_search_results.json")
