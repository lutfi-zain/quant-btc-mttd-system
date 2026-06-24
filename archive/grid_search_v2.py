#!/usr/bin/env python3
"""
MTTD Grid Search V2 — Optimize for Trading Performance + ISP Coherence
=====================================================================

Two-phase approach:
Phase A: Optimize each indicator's parameters individually
Phase B: Optimize ensemble min_hold with all indicators using best params

Fitness: Trading metrics (Sharpe, Calmar, Sortino, MaxDD, CAGR) must
MATCH or EXCEED ISP benchmark, while maintaining ISP coherence ≥ 70%.
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
from ensemble_engine import compute_ensemble_signal
import importlib.util

print("=" * 70)
print("MTTD GRID SEARCH V2 — TRADING PERFORMANCE + ISP COHERENCE")
print("=" * 70)

# ================================================================
# Load Data
# ================================================================
print("\n[1/5] Loading data...")

with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
    btc_data = json.load(f)

df = pd.DataFrame(btc_data['aligned_data'])
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = df[df.index >= '2018-01-01']

# Load ISP benchmark
isp_df = pd.read_csv(os.path.join(project_root, 'isp-signals-btcusd-2026-06-13.csv'))
isp_df['Date'] = pd.to_datetime(isp_df['Date'])
isp_df = isp_df.set_index('Date')

# Build ISP position series
isp_positions = pd.Series(0.0, index=df.index)
for date, row in isp_df.iterrows():
    if date in isp_positions.index:
        if row['Action'] == 'BUY':
            isp_positions.loc[date:] = 1.0
        elif row['Action'] == 'SELL':
            isp_positions.loc[date:] = 0.0

isp_transitions = (isp_positions.diff() != 0).sum()
print(f"  Data: {len(df)} bars ({df.index[0]} to {df.index[-1]})")
print(f"  ISP: {isp_transitions} transitions, {(isp_positions == 1.0).mean()*100:.1f}% in position")

# ================================================================
# Compute ISP Benchmark Metrics
# ================================================================
print("\n[2/5] Computing ISP benchmark metrics...")

def compute_trading_metrics(positions, prices, initial_capital=100000.0):
    """Compute trading performance metrics."""
    returns = prices.pct_change()
    strategy_returns = returns * positions.shift(1)
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) == 0:
        return {
            'cagr': 0, 'sharpe': 0, 'sortino': 0, 'calmar': 0,
            'max_dd': 0, 'total_return': 0, 'n_trades': 0, 'pct_in': 0
        }

    equity = initial_capital * (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25

    # CAGR
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1 if years > 0 else 0

    # Total return
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

    # Sharpe
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0

    # Sortino
    downside = strategy_returns[strategy_returns < 0]
    sortino = strategy_returns.mean() / downside.std() * np.sqrt(365) if len(downside) > 0 and downside.std() > 0 else 0

    # Max Drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Trades
    n_trades = (positions.diff().abs() > 0).sum()

    return {
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'calmar': round(calmar, 2),
        'max_dd': round(max_dd * 100, 2),
        'total_return': round(total_return * 100, 2),
        'n_trades': int(n_trades),
        'pct_in': round(positions.mean() * 100, 2)
    }

def compute_isp_coherence(positions, benchmark):
    """Compute time-coherence with ISP benchmark."""
    aligned = pd.DataFrame({'system': positions, 'benchmark': benchmark}).dropna()
    if len(aligned) == 0:
        return 0.0
    matches = (aligned['system'] == aligned['benchmark']).sum()
    return matches / len(aligned) * 100

isp_metrics = compute_trading_metrics(isp_positions, df['close'])
print(f"  ISP CAGR:   {isp_metrics['cagr']:.2f}%")
print(f"  ISP Sharpe: {isp_metrics['sharpe']:.2f}")
print(f"  ISP Sortino:{isp_metrics['sortino']:.2f}")
print(f"  ISP Calmar: {isp_metrics['calmar']:.2f}")
print(f"  ISP MaxDD:  {isp_metrics['max_dd']:.2f}%")
print(f"  ISP Trades: {isp_metrics['n_trades']}")

# ================================================================
# Indicator Definitions with Tunable Parameters
# ================================================================
print("\n[3/5] Defining indicator search space...")

INDICATORS = [
    {
        'name': 'kalman_filtered_rsi_oscillator',
        'category': 'oscillator',
        'func_name': 'kalman_filtered_rsi_oscillator',
        'params': {'rsi_period': [10, 12, 14, 16, 18, 20]}
    },
    {
        'name': 'z_smma_quantedgeb',
        'category': 'oscillator',
        'func_name': 'z_smma_quantedgeb',
        'params': {'rma_length': [8, 10, 12, 14, 16], 'ema_length': [20, 25, 30, 35, 40], 'z_thresh': [0.05, 0.1, 0.15, 0.2]}
    },
    {
        'name': 'median_rsi_sd_quantedgeb',
        'category': 'oscillator',
        'func_name': 'median_rsi_sd_quantedgeb',
        'params': {'median_length': [5, 7, 10, 12, 15], 'rsi_length': [15, 18, 21, 24, 28]}
    },
    {
        'name': 'polynomial_deviation_bands',
        'category': 'perpetual',
        'func_name': 'polynomial_deviation_bands',
        'params': {'window': [10, 12, 14, 16, 20], 'dev_mult': [1.0, 1.2, 1.5, 1.8, 2.0]}
    },
    {
        'name': 'gaussian_smooth_trend_quantedgeb',
        'category': 'perpetual',
        'func_name': 'gaussian_smooth_trend_quantedgeb',
        'params': {'dema_length': [5, 6, 7, 8, 10], 'gaussian_sigma': [1.5, 2.0, 2.5, 3.0]}
    },
    {
        'name': 'alma_lag_viresearch',
        'category': 'perpetual',
        'func_name': 'alma_lag_viresearch',
        'params': {'alma_length': [60, 70, 78, 85, 100], 'alma_offset': [0.80, 0.85, 0.90]}
    },
    {
        'name': 'adaptive_regime_cloud',
        'category': 'perpetual',
        'func_name': 'adaptive_regime_cloud',
        'params': {'hurst_period': [30, 40, 50, 60, 70]}
    },
    {
        'name': 'root_mean_square_deviation_trend',
        'category': 'perpetual',
        'func_name': 'root_mean_square_deviation_trend',
        'params': {'length': [20, 24, 28, 32, 36, 40], 'ma_type': ['EMA', 'SMA', 'HMA', 'DEMA']}
    },
    {
        'name': 'p_motion_trend_quantedgeb',
        'category': 'perpetual',
        'func_name': 'p_motion_trend_quantedgeb',
        'params': {'dema_length': [5, 6, 7, 8, 10], 'ema_length': [15, 18, 21, 25, 30]}
    },
    {
        'name': 'dema_adjusted_average_true_range',
        'category': 'perpetual',
        'func_name': 'dema_adjusted_average_true_range',
        'params': {'period_dema': [5, 6, 7, 8, 10], 'factor_atr': [1.2, 1.5, 1.7, 2.0, 2.2]}
    }
]

def detect_direction(res_df):
    """Extract direction from indicator output."""
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

def load_indicator_func(indicator_name, category):
    """Dynamically load indicator function."""
    filename = f"{indicator_name}.py"
    module_path = os.path.join(project_root, category, filename)
    spec = importlib.util.spec_from_file_location(indicator_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, indicator_name)

def compute_indicator_with_params(indicator_def, params, df):
    """Compute indicator with given parameters and return binary signal."""
    try:
        func = load_indicator_func(indicator_def['name'], indicator_def['category'])
        # Filter params to only those accepted by the function
        import inspect
        sig = inspect.signature(func)
        valid_params = {k: v for k, v in params.items() if k in sig.parameters}
        res_df = func(df, **valid_params)
        direction = detect_direction(res_df)
        if direction is not None:
            return direction.apply(lambda x: 1.0 if x > 0 else -1.0)
    except Exception as e:
        pass
    return None

# ================================================================
# Phase A: Optimize Individual Indicator Parameters
# ================================================================
print("\n[4/5] Phase A: Optimizing individual indicator parameters...")

best_indicator_params = {}
indicator_isp_coherence = {}

for ind_def in INDICATORS:
    ind_name = ind_def['name']
    param_names = list(ind_def['params'].keys())
    param_values = list(ind_def['params'].values())
    combinations = list(product(*param_values))

    print(f"\n  {ind_name}: {len(combinations)} combinations")

    best_score = -999
    best_params = None
    best_coherence = 0

    for combo in combinations:
        params = dict(zip(param_names, combo))
        signal = compute_indicator_with_params(ind_def, params, df)

        if signal is None:
            continue

        # Compute ISP coherence
        coherence = compute_isp_coherence(signal, isp_positions)

        # Compute trading metrics
        # Convert signal to position (1 if bullish, 0 otherwise)
        position = (signal > 0).astype(float)
        metrics = compute_trading_metrics(position, df['close'])

        # Score: must exceed ISP metrics while maintaining coherence
        coherence_penalty = max(0, 70 - coherence) * 2  # Penalty if coherence < 70%

        # Compare trading metrics to ISP
        sharpe_ratio = metrics['sharpe'] / isp_metrics['sharpe'] if isp_metrics['sharpe'] > 0 else 0
        sortino_ratio = metrics['sortino'] / isp_metrics['sortino'] if isp_metrics['sortino'] > 0 else 0
        calmar_ratio = metrics['calmar'] / isp_metrics['calmar'] if isp_metrics['calmar'] > 0 else 0

        # Score combines coherence and trading performance
        score = coherence * 0.4 + sharpe_ratio * 20 + sortino_ratio * 10 + calmar_ratio * 5 - coherence_penalty

        if score > best_score:
            best_score = score
            best_params = params
            best_coherence = coherence

    if best_params is not None:
        best_indicator_params[ind_name] = best_params
        indicator_isp_coherence[ind_name] = best_coherence
        print(f"    Best: {best_params} → coherence={best_coherence:.1f}%")
    else:
        print(f"    No valid combination found, using defaults")
        best_indicator_params[ind_name] = {}
        indicator_isp_coherence[ind_name] = 0

# ================================================================
# Phase B: Optimize Ensemble Min Hold
# ================================================================
print("\n[5/5] Phase B: Optimizing ensemble min_hold...")

# Compute all indicators with optimized params
print("\n  Computing indicators with optimized params...")
signal_matrix_data = {}

for ind_def in INDICATORS:
    ind_name = ind_def['name']
    params = best_indicator_params.get(ind_name, {})
    signal = compute_indicator_with_params(ind_def, params, df)

    if signal is not None:
        signal_matrix_data[ind_name] = signal
        print(f"    ✓ {ind_name}")
    else:
        print(f"    ✗ {ind_name} (failed)")

signal_matrix = pd.DataFrame(signal_matrix_data)
n_indicators = len(signal_matrix.columns)
print(f"\n  Active indicators: {n_indicators}")

# Grid search min_hold
print("\n  Grid searching min_hold...")
min_hold_range = [1, 3, 5, 7, 10, 15, 20, 25, 30]
results = []

for min_hold in min_hold_range:
    # Compute ensemble
    ensemble_result = compute_ensemble_signal(signal_matrix, min_hold=min_hold)
    position = ensemble_result['position']

    # Compute metrics
    metrics = compute_trading_metrics(position, df['close'])
    coherence = compute_isp_coherence(position, isp_positions)

    # Score
    coherence_penalty = max(0, 70 - coherence) * 2

    sharpe_ratio = metrics['sharpe'] / isp_metrics['sharpe'] if isp_metrics['sharpe'] > 0 else 0
    sortino_ratio = metrics['sortino'] / isp_metrics['sortino'] if isp_metrics['sortino'] > 0 else 0
    calmar_ratio = metrics['calmar'] / isp_metrics['calmar'] if isp_metrics['calmar'] > 0 else 0

    score = coherence * 0.4 + sharpe_ratio * 20 + sortino_ratio * 10 + calmar_ratio * 5 - coherence_penalty

    results.append({
        'min_hold': min_hold,
        'coherence': coherence,
        'score': score,
        **metrics
    })

    print(f"    min_hold={min_hold:2d}: coherence={coherence:.1f}%, sharpe={metrics['sharpe']:.2f}, "
          f"sortino={metrics['sortino']:.2f}, calmar={metrics['calmar']:.2f}, "
          f"max_dd={metrics['max_dd']:.1f}%, cagr={metrics['cagr']:.1f}%")

# Sort by score
results.sort(key=lambda x: x['score'], reverse=True)
best = results[0]

print("\n" + "=" * 70)
print("GRID SEARCH RESULTS")
print("=" * 70)

print(f"\nBest min_hold: {best['min_hold']}")
print(f"  Coherence: {best['coherence']:.1f}%")
print(f"  CAGR:      {best['cagr']:.2f}% (ISP: {isp_metrics['cagr']:.2f}%)")
print(f"  Sharpe:    {best['sharpe']:.2f} (ISP: {isp_metrics['sharpe']:.2f})")
print(f"  Sortino:   {best['sortino']:.2f} (ISP: {isp_metrics['sortino']:.2f})")
print(f"  Calmar:    {best['calmar']:.2f} (ISP: {isp_metrics['calmar']:.2f})")
print(f"  Max DD:    {best['max_dd']:.2f}% (ISP: {isp_metrics['max_dd']:.2f}%)")
print(f"  Trades:    {best['n_trades']} (ISP: {isp_metrics['n_trades']})")

# Check if trading metrics match/exceed ISP
checks = {
    'Sharpe': best['sharpe'] >= isp_metrics['sharpe'],
    'Sortino': best['sortino'] >= isp_metrics['sortino'],
    'Calmar': best['calmar'] >= isp_metrics['calmar'],
    'MaxDD': abs(best['max_dd']) <= abs(isp_metrics['max_dd']),
    'CAGR': best['cagr'] >= isp_metrics['cagr'],
}

print(f"\nISP Benchmark Comparison:")
for metric, passed in checks.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {metric:10s}: {status}")

n_passed = sum(checks.values())
print(f"\n{n_passed}/{len(checks)} metrics match/exceed ISP")

# Save results
output = {
    'isp_metrics': isp_metrics,
    'best_indicator_params': best_indicator_params,
    'indicator_isp_coherence': indicator_isp_coherence,
    'best_min_hold': best['min_hold'],
    'ensemble_metrics': best,
    'all_results': results,
    'isp_benchmark_comparison': checks
}

output_path = os.path.join(project_root, 'grid_search_v2_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to: {output_path}")
