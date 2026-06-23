#!/usr/bin/env python3
"""
MTTD Grid Search V3 — REDUCED INDICATOR SET
=============================================

Uses only 4 indicators with genuine factor diversification:
1. Adaptive Regime Cloud (regime detection)
2. Kalman RSI (momentum)
3. ALMA Lag (trend direction)
4. RMSD Trend (trend quality + volatility)

Optimizes for risk-adjusted returns (NOT ISP coherence).
Includes 0.1% transaction costs.
Train on 2018-2024, holdout 2025-2026.
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
print("MTTD GRID SEARCH V3 — REDUCED INDICATOR SET")
print("=" * 70)
print("Indicators: 4 (Adaptive Cloud, Kalman RSI, ALMA Lag, RMSD Trend)")
print("Optimization: Risk-adjusted returns (NOT ISP coherence)")
print("Transaction costs: 0.1% round-trip")
print("Training: 2018-01-01 to 2024-12-31")
print("Holdout: 2025-01-01 to present")
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

# Split into training and holdout
HOLDOUT_START = '2025-01-01'
df_train = df_full[df_full.index < HOLDOUT_START].copy()
df_holdout = df_full[df_full.index >= HOLDOUT_START].copy()

print(f"  Full data:      {len(df_full)} bars ({df_full.index[0]} to {df_full.index[-1]})")
print(f"  Training data:  {len(df_train)} bars ({df_train.index[0]} to {df_train.index[-1]})")
print(f"  Holdout data:   {len(df_holdout)} bars ({df_holdout.index[0]} to {df_holdout.index[-1]})")

# Load ISP benchmark
isp_df = pd.read_csv(os.path.join(project_root, 'isp-signals-btcusd-2026-06-13.csv'))
isp_df['Date'] = pd.to_datetime(isp_df['Date'])
isp_df = isp_df.set_index('Date')

# Build ISP position series for full period
isp_positions_full = pd.Series(0.0, index=df_full.index)
for date, row in isp_df.iterrows():
    if date in isp_positions_full.index:
        if row['Action'] == 'BUY':
            isp_positions_full.loc[date:] = 1.0
        elif row['Action'] == 'SELL':
            isp_positions_full.loc[date:] = 0.0

# Split ISP positions
isp_positions_train = isp_positions_full[df_train.index]
isp_positions_holdout = isp_positions_full[df_holdout.index]

isp_transitions_train = (isp_positions_train.diff() != 0).sum()
isp_transitions_holdout = (isp_positions_holdout.diff() != 0).sum()

print(f"\n  ISP Training: {isp_transitions_train} transitions, {(isp_positions_train == 1.0).mean()*100:.1f}% in position")
print(f"  ISP Holdout:  {isp_transitions_holdout} transitions, {(isp_positions_holdout == 1.0).mean()*100:.1f}% in position")

# ================================================================
# Compute ISP Benchmark Metrics (Training Period) — Reference Only
# ================================================================
print("\n[2/6] Computing ISP benchmark metrics (reference only, NOT used in optimization)...")

def compute_trading_metrics(positions, prices, initial_capital=100000.0, transaction_cost=0.001):
    """Compute trading performance metrics with transaction costs."""
    returns = prices.pct_change()
    strategy_returns = returns * positions.shift(1)
    strategy_returns = strategy_returns.dropna()

    # Transaction costs: half the round-trip cost applied at each position transition
    transitions = positions.diff().fillna(0)
    cost_per_transition = transaction_cost / 2
    strategy_returns = strategy_returns - transitions.loc[strategy_returns.index] * cost_per_transition

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
    """Compute time-coherence with ISP benchmark (for reference only)."""
    aligned = pd.DataFrame({'system': positions, 'benchmark': benchmark}).dropna()
    if len(aligned) == 0:
        return 0.0
    matches = (aligned['system'] == aligned['benchmark']).sum()
    return matches / len(aligned) * 100

isp_metrics_train = compute_trading_metrics(isp_positions_train, df_train['close'])
print(f"  ISP Training (reference):")
print(f"    CAGR:   {isp_metrics_train['cagr']:.2f}%")
print(f"    Sharpe: {isp_metrics_train['sharpe']:.2f}")
print(f"    Sortino:{isp_metrics_train['sortino']:.2f}")
print(f"    Calmar: {isp_metrics_train['calmar']:.2f}")
print(f"    MaxDD:  {isp_metrics_train['max_dd']:.2f}%")
print(f"    Trades: {isp_metrics_train['n_trades']}")

# ================================================================
# REDUCED Indicator Definitions (4 indicators only)
# ================================================================
print("\n[3/6] Defining indicator search space (4 indicators only)...")

INDICATORS = [
    {
        'name': 'adaptive_regime_cloud',
        'category': 'perpetual',
        'func_name': 'adaptive_regime_cloud',
        'params': {'hurst_period': [30, 40, 50, 60, 70]},
        'rationale': 'Regime detection — THE core indicator'
    },
    {
        'name': 'kalman_filtered_rsi_oscillator',
        'category': 'oscillator',
        'func_name': 'kalman_filtered_rsi_oscillator',
        'params': {'rsi_period': [10, 12, 14, 16, 18, 20]},
        'rationale': 'Momentum factor — different from trend'
    },
    {
        'name': 'alma_lag_viresearch',
        'category': 'perpetual',
        'func_name': 'alma_lag_viresearch',
        'params': {'alma_length': [60, 70, 78, 85, 100], 'alma_offset': [0.80, 0.85, 0.90]},
        'rationale': 'Trend direction — clean implementation'
    },
    {
        'name': 'root_mean_square_deviation_trend',
        'category': 'perpetual',
        'func_name': 'root_mean_square_deviation_trend',
        'params': {'length': [20, 24, 28, 32, 36, 40], 'ma_type': ['EMA', 'SMA', 'HMA', 'DEMA']},
        'rationale': 'Trend quality + volatility adjustment'
    }
]

# Calculate total combinations
total_combos = 1
for ind in INDICATORS:
    combos = 1
    for v in ind['params'].values():
        combos *= len(v)
    total_combos *= combos
    print(f"  {ind['name']}: {combos} combinations — {ind['rationale']}")

print(f"\n  Total combinations: {total_combos}")

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
# Phase A: Optimize Individual Indicator Parameters (TRAINING ONLY)
# Optimization: Risk-adjusted returns (NOT ISP coherence)
# ================================================================
print("\n[4/6] Phase A: Optimizing individual indicator parameters (TRAINING DATA ONLY)...")
print("  Optimization target: Risk-adjusted returns (Sharpe × 0.5 + Calmar × 0.3 + Sortino × 0.2)")
print("  ISP coherence: NOT used in optimization (reference only)")

best_indicator_params = {}
indicator_scores = {}

for ind_def in INDICATORS:
    ind_name = ind_def['name']
    param_names = list(ind_def['params'].keys())
    param_values = list(ind_def['params'].values())
    combinations = list(product(*param_values))

    print(f"\n  {ind_name}: {len(combinations)} combinations")

    best_score = -999
    best_params = None
    best_metrics = None

    for combo in combinations:
        params = dict(zip(param_names, combo))
        signal = compute_indicator_with_params(ind_def, params, df_train)  # TRAINING ONLY

        if signal is None:
            continue

        # Compute trading metrics (TRAINING ONLY, with transaction costs)
        position = (signal > 0).astype(float)
        metrics = compute_trading_metrics(position, df_train['close'])

        # Score — Risk-adjusted returns ONLY (ISP coherence NOT used)
        score = metrics['sharpe'] * 0.5 + metrics['calmar'] * 0.3 + metrics['sortino'] * 0.2

        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    if best_params is not None:
        best_indicator_params[ind_name] = best_params
        indicator_scores[ind_name] = best_metrics
        print(f"    Best: {best_params}")
        print(f"    → Sharpe={best_metrics['sharpe']:.2f}, Calmar={best_metrics['calmar']:.2f}, "
              f"Sortino={best_metrics['sortino']:.2f}, CAGR={best_metrics['cagr']:.1f}%, "
              f"MaxDD={best_metrics['max_dd']:.1f}%")
    else:
        print(f"    No valid combination found, using defaults")
        best_indicator_params[ind_name] = {}

# ================================================================
# Phase B: Optimize Ensemble Min Hold (TRAINING ONLY)
# ================================================================
print("\n[5/6] Phase B: Optimizing ensemble min_hold (TRAINING DATA ONLY)...")

# Compute all indicators with optimized params on TRAINING data
print("\n  Computing indicators with optimized params on training data...")
signal_matrix_train = {}

for ind_def in INDICATORS:
    ind_name = ind_def['name']
    params = best_indicator_params.get(ind_name, {})
    signal = compute_indicator_with_params(ind_def, params, df_train)  # TRAINING ONLY

    if signal is not None:
        signal_matrix_train[ind_name] = signal
        print(f"    ✓ {ind_name}")
    else:
        print(f"    ✗ {ind_name} (failed)")

signal_matrix_train = pd.DataFrame(signal_matrix_train)
n_indicators = len(signal_matrix_train.columns)
print(f"\n  Active indicators: {n_indicators}")

# Compute expected correlations
print("\n  Computing actual indicator correlations...")
if n_indicators > 1:
    corr_matrix = signal_matrix_train.corr()
    print(f"\n  Correlation matrix:")
    print(f"  {corr_matrix.to_string()}")

# Grid search min_hold on TRAINING data
print("\n  Grid searching min_hold on training data...")
min_hold_range = [1, 3, 5, 7, 10, 15, 20, 25, 30]
train_results = []

for min_hold in min_hold_range:
    ensemble_result = compute_ensemble_signal(signal_matrix_train, min_hold=min_hold)
    position = ensemble_result['position']

    metrics = compute_trading_metrics(position, df_train['close'])
    coherence = compute_isp_coherence(position, isp_positions_train)  # reference only

    # Score — Risk-adjusted returns ONLY
    score = metrics['sharpe'] * 0.5 + metrics['calmar'] * 0.3 + metrics['sortino'] * 0.2

    train_results.append({
        'min_hold': min_hold,
        'coherence': coherence,
        'score': score,
        **metrics
    })

    print(f"    min_hold={min_hold:2d}: sharpe={metrics['sharpe']:.2f}, "
          f"sortino={metrics['sortino']:.2f}, calmar={metrics['calmar']:.2f}, "
          f"max_dd={metrics['max_dd']:.1f}%, cagr={metrics['cagr']:.1f}%, "
          f"coherence={coherence:.1f}%")

# Sort by score
train_results.sort(key=lambda x: x['score'], reverse=True)
best_train = train_results[0]

print("\n" + "=" * 70)
print("GRID SEARCH RESULTS (TRAINING PERIOD)")
print("=" * 70)

print(f"\nBest min_hold: {best_train['min_hold']}")
print(f"  CAGR:      {best_train['cagr']:.2f}%")
print(f"  Sharpe:    {best_train['sharpe']:.2f}")
print(f"  Sortino:   {best_train['sortino']:.2f}")
print(f"  Calmar:    {best_train['calmar']:.2f}")
print(f"  Max DD:    {best_train['max_dd']:.2f}%")
print(f"  Trades:    {best_train['n_trades']}")
print(f"  In-market: {best_train['pct_in']:.1f}%")
print(f"  Coherence: {best_train['coherence']:.1f}% (reference only)")

# ================================================================
# HOLDOUT EVALUATION
# ================================================================
print("\n[6/6] HOLDOUT EVALUATION (2025-2026)...")

# Compute indicators on FULL data for final evaluation
print("\n  Computing indicators on full data...")
signal_matrix_full = {}

for ind_def in INDICATORS:
    ind_name = ind_def['name']
    params = best_indicator_params.get(ind_name, {})
    signal = compute_indicator_with_params(ind_def, params, df_full)  # FULL DATA

    if signal is not None:
        signal_matrix_full[ind_name] = signal

signal_matrix_full = pd.DataFrame(signal_matrix_full)

# Compute ensemble on full data
ensemble_result_full = compute_ensemble_signal(signal_matrix_full, min_hold=best_train['min_hold'])
position_full = ensemble_result_full['position']

# Split positions
position_train = position_full[df_train.index]
position_holdout = position_full[df_holdout.index]

# Compute metrics for each period
metrics_train = compute_trading_metrics(position_train, df_train['close'])
metrics_holdout = compute_trading_metrics(position_holdout, df_holdout['close'])
metrics_full = compute_trading_metrics(position_full, df_full['close'])

# ISP metrics for full period
isp_metrics_full = compute_trading_metrics(isp_positions_full, df_full['close'])

# Coherence for each period
coherence_train = compute_isp_coherence(position_train, isp_positions_train)
coherence_holdout = compute_isp_coherence(position_holdout, isp_positions_holdout)
coherence_full = compute_isp_coherence(position_full, isp_positions_full)

# Print results
print("\n" + "=" * 70)
print("FINAL RESULTS — ALL PERIODS")
print("=" * 70)

print(f"\n{'Metric':<12} {'Training':<15} {'Holdout':<15} {'Full':<15} {'ISP Full':<15}")
print("-" * 72)
print(f"{'CAGR':<12} {metrics_train['cagr']:<15.2f} {metrics_holdout['cagr']:<15.2f} {metrics_full['cagr']:<15.2f} {isp_metrics_full['cagr']:<15.2f}")
print(f"{'Sharpe':<12} {metrics_train['sharpe']:<15.2f} {metrics_holdout['sharpe']:<15.2f} {metrics_full['sharpe']:<15.2f} {isp_metrics_full['sharpe']:<15.2f}")
print(f"{'Sortino':<12} {metrics_train['sortino']:<15.2f} {metrics_holdout['sortino']:<15.2f} {metrics_full['sortino']:<15.2f} {isp_metrics_full['sortino']:<15.2f}")
print(f"{'Calmar':<12} {metrics_train['calmar']:<15.2f} {metrics_holdout['calmar']:<15.2f} {metrics_full['calmar']:<15.2f} {isp_metrics_full['calmar']:<15.2f}")
print(f"{'MaxDD':<12} {metrics_train['max_dd']:<15.2f} {metrics_holdout['max_dd']:<15.2f} {metrics_full['max_dd']:<15.2f} {isp_metrics_full['max_dd']:<15.2f}")
print(f"{'Trades':<12} {metrics_train['n_trades']:<15} {metrics_holdout['n_trades']:<15} {metrics_full['n_trades']:<15} {isp_metrics_full['n_trades']:<15}")
print(f"{'In-market':<12} {metrics_train['pct_in']:<15.1f} {metrics_holdout['pct_in']:<15.1f} {metrics_full['pct_in']:<15.1f} {isp_metrics_full['pct_in']:<15.1f}")
print(f"{'Coherence':<12} {coherence_train:<15.1f} {coherence_holdout:<15.1f} {coherence_full:<15.1f} {'N/A':<15}")

# Degradation analysis
print("\n" + "=" * 70)
print("DEGRADATION ANALYSIS (Training → Holdout)")
print("=" * 70)

if metrics_train['sharpe'] > 0:
    sharpe_degradation = (metrics_holdout['sharpe'] - metrics_train['sharpe']) / metrics_train['sharpe'] * 100
    print(f"  Sharpe degradation:   {sharpe_degradation:+.1f}%")

if metrics_train['cagr'] > 0:
    cagr_degradation = (metrics_holdout['cagr'] - metrics_train['cagr']) / metrics_train['cagr'] * 100
    print(f"  CAGR degradation:     {cagr_degradation:+.1f}%")

coherence_degradation = coherence_holdout - coherence_train
print(f"  Coherence change:     {coherence_degradation:+.1f}%")

# Overfitting assessment
print("\n" + "=" * 70)
print("OVERFITTING ASSESSMENT")
print("=" * 70)

overfitting_score = 0
warnings_list = []

if metrics_holdout['sharpe'] < metrics_train['sharpe'] * 0.5:
    warnings_list.append("⚠️  Sharpe dropped by >50% in holdout")
    overfitting_score += 2
elif metrics_holdout['sharpe'] < metrics_train['sharpe'] * 0.7:
    warnings_list.append("⚠️  Sharpe dropped by >30% in holdout")
    overfitting_score += 1

if metrics_holdout['max_dd'] < metrics_train['max_dd'] * 1.5:
    warnings_list.append("⚠️  Max DD worsened by >50% in holdout")
    overfitting_score += 2
elif metrics_holdout['max_dd'] < metrics_train['max_dd'] * 1.2:
    warnings_list.append("⚠️  Max DD worsened by >20% in holdout")
    overfitting_score += 1

if coherence_holdout < coherence_train - 10:
    warnings_list.append("⚠️  Coherence dropped by >10% in holdout")
    overfitting_score += 1

if metrics_holdout['n_trades'] < 2:
    warnings_list.append("⚠️  Very few trades in holdout period")
    overfitting_score += 1

if overfitting_score == 0:
    print("  ✅ LOW overfitting risk — metrics stable across periods")
elif overfitting_score <= 2:
    print("  ⚠️  MODERATE overfitting risk")
else:
    print("  🔴 HIGH overfitting risk")

for w in warnings_list:
    print(f"  {w}")

# Save results
output = {
    'version': 'v3_reduced_4_indicators',
    'optimization_target': 'risk_adjusted_returns',
    'transaction_cost': 0.001,
    'holdout_start': HOLDOUT_START,
    'training_period': f"{df_train.index[0]} to {df_train.index[-1]}",
    'holdout_period': f"{df_holdout.index[0]} to {df_holdout.index[-1]}",
    'indicators_used': [ind['name'] for ind in INDICATORS],
    'best_indicator_params': best_indicator_params,
    'best_min_hold': best_train['min_hold'],
    'training_metrics': metrics_train,
    'holdout_metrics': metrics_holdout,
    'full_metrics': metrics_full,
    'isp_full_metrics': isp_metrics_full,
    'coherence': {
        'training': coherence_train,
        'holdout': coherence_holdout,
        'full': coherence_full
    },
    'overfitting_assessment': {
        'score': overfitting_score,
        'warnings': warnings_list
    },
    'train_results': train_results
}

output_path = os.path.join(project_root, 'grid_search_v3_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to: {output_path}")
print("\n" + "=" * 70)
print("GRID SEARCH V3 COMPLETE")
print("=" * 70)
