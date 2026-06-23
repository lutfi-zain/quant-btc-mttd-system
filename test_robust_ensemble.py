#!/usr/bin/env python3
"""
Test Robust Ensemble Configurations
=====================================

Tests different ensemble configurations with robustness features:
- Minimum agreement threshold
- Outlier rejection
- Min hold filter

Validates on training (2018-2024) and holdout (2025-2026)
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
from ensemble_robust import (
    compute_robust_ensemble, 
    apply_min_hold,
    ENSEMBLE_CONFIGS
)

print("=" * 70)
print("TEST ROBUST ENSEMBLE CONFIGURATIONS")
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

def load_indicator(indicator_name, category, df):
    """Load and compute indicator."""
    for base_path in [project_root, bank_root]:
        filename = f"{indicator_name}.py"
        module_path = os.path.join(base_path, category, filename)
        if os.path.exists(module_path):
            try:
                spec = importlib.util.spec_from_file_location(indicator_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                func = getattr(module, indicator_name)
                
                import inspect
                sig = inspect.signature(func)
                default_params = {}
                for name, param in sig.parameters.items():
                    if name == 'df':
                        continue
                    if param.default != inspect.Parameter.empty:
                        default_params[name] = param.default
                
                res_df = func(df, **default_params)
                direction = detect_direction(res_df)
                if direction is not None:
                    return direction.apply(lambda x: 1.0 if x > 0 else -1.0)
            except Exception as e:
                pass
    return None

# ================================================================
# Compute all indicator signals
# ================================================================
print("\n[2/5] Computing all indicator signals...")

# Get all indicator names from configs
all_indicators = set()
for config in ENSEMBLE_CONFIGS.values():
    all_indicators.update(config['indicators'])

# Map indicator names to categories
indicator_categories = {}
for ind_name in all_indicators:
    # Check local first
    if os.path.exists(os.path.join(project_root, 'oscillator', f'{ind_name}.py')):
        indicator_categories[ind_name] = 'oscillator'
    elif os.path.exists(os.path.join(project_root, 'perpetual', f'{ind_name}.py')):
        indicator_categories[ind_name] = 'perpetual'
    elif os.path.exists(os.path.join(bank_root, 'oscillator', f'{ind_name}.py')):
        indicator_categories[ind_name] = 'oscillator'
    elif os.path.exists(os.path.join(bank_root, 'perpetual', f'{ind_name}.py')):
        indicator_categories[ind_name] = 'perpetual'

print(f"  Computing {len(all_indicators)} indicators...")

signals_train = {}
signals_full = {}

for ind_name in all_indicators:
    category = indicator_categories.get(ind_name)
    if category:
        signal = load_indicator(ind_name, category, df_full)
        if signal is not None:
            signals_train[ind_name] = signal[df_train.index]
            signals_full[ind_name] = signal
            print(f"    ✓ {ind_name}")
        else:
            print(f"    ✗ {ind_name} (failed)")
    else:
        print(f"    ✗ {ind_name} (category not found)")

signal_matrix_train = pd.DataFrame(signals_train)
signal_matrix_full = pd.DataFrame(signals_full)

print(f"\n  Loaded {len(signals_train)} indicators")

# ================================================================
# Test each ensemble configuration
# ================================================================
print("\n[3/5] Testing ensemble configurations...")

results = []

for config_name, config in ENSEMBLE_CONFIGS.items():
    print(f"\n  === {config_name} ===")
    print(f"  {config['description']}")
    
    # Check which indicators are available
    available = [ind for ind in config['indicators'] if ind in signals_train]
    missing = [ind for ind in config['indicators'] if ind not in signals_train]
    
    if missing:
        print(f"  Missing indicators: {missing}")
    
    if len(available) < 3:
        print(f"  Not enough indicators available, skipping...")
        continue
    
    # Get signal matrix for this config
    train_matrix = signal_matrix_train[available]
    full_matrix = signal_matrix_full[available]
    
    print(f"  Available indicators: {len(available)}/{len(config['indicators'])}")
    
    # Compute ensemble on full data
    ensemble_full = compute_robust_ensemble(
        full_matrix,
        min_hold=config['min_hold'],
        min_agreement=config['min_agreement'],
        reject_outliers=config['reject_outliers']
    )
    
    position_full = ensemble_full['position']
    agreement_full = ensemble_full['agreement']
    
    # Split
    position_train = position_full[df_train.index]
    position_holdout = position_full[df_holdout.index]
    agreement_train = agreement_full[df_train.index]
    agreement_holdout = agreement_full[df_holdout.index]
    
    # Compute metrics
    metrics_train = compute_metrics(position_train, df_train['close'])
    metrics_holdout = compute_metrics(position_holdout, df_holdout['close'])
    metrics_full = compute_metrics(position_full, df_full['close'])
    
    coherence_train = compute_coherence(position_train, isp_positions_train)
    coherence_holdout = compute_coherence(position_holdout, isp_positions_holdout)
    
    # Degradation
    if metrics_train['sharpe'] > 0:
        sharpe_deg = (metrics_holdout['sharpe'] - metrics_train['sharpe']) / metrics_train['sharpe'] * 100
    else:
        sharpe_deg = 0
    
    result = {
        'name': config_name,
        'description': config['description'],
        'n_indicators': len(available),
        'min_agreement': config['min_agreement'],
        'min_hold': config['min_hold'],
        'train': metrics_train,
        'holdout': metrics_holdout,
        'full': metrics_full,
        'coherence_train': coherence_train,
        'coherence_holdout': coherence_holdout,
        'avg_agreement_train': round(agreement_train.mean() * 100, 1),
        'avg_agreement_holdout': round(agreement_holdout.mean() * 100, 1),
        'sharpe_degradation': round(sharpe_deg, 1)
    }
    
    results.append(result)
    
    print(f"  Train:    Sharpe={metrics_train['sharpe']:.2f}, CAGR={metrics_train['cagr']:.1f}%, MaxDD={metrics_train['max_dd']:.1f}%")
    print(f"  Holdout:  Sharpe={metrics_holdout['sharpe']:.2f}, CAGR={metrics_holdout['cagr']:.1f}%, MaxDD={metrics_holdout['max_dd']:.1f}%")
    print(f"  Degradation: {sharpe_deg:+.1f}%")
    print(f"  Avg agreement: {agreement_train.mean()*100:.1f}% (train) → {agreement_holdout.mean()*100:.1f}% (holdout)")

# ================================================================
# Compare with single indicator baseline
# ================================================================
print("\n[4/5] Comparing with single indicator baseline...")

# Use best single indicator as baseline
if 'median_standard_deviation_viresearch' in signals_train:
    baseline_signal = signals_train['median_standard_deviation_viresearch']
    baseline_position = (baseline_signal > 0).astype(float)
    baseline_train = compute_metrics(baseline_position, df_train['close'])
    baseline_holdout = compute_metrics(
        (signals_full['median_standard_deviation_viresearch'][df_holdout.index] > 0).astype(float),
        df_holdout['close']
    )
    print(f"  Baseline (median_std_dev):")
    print(f"    Train:   Sharpe={baseline_train['sharpe']:.2f}, CAGR={baseline_train['cagr']:.1f}%")
    print(f"    Holdout: Sharpe={baseline_holdout['sharpe']:.2f}, CAGR={baseline_holdout['cagr']:.1f}%")
else:
    baseline_train = None
    baseline_holdout = None

# ================================================================
# Save results
# ================================================================
print("\n[5/5] Saving results...")

output = {
    'ensemble_results': results,
    'baseline': {
        'train': baseline_train,
        'holdout': baseline_holdout
    } if baseline_train else None,
    'best_ensemble': max(results, key=lambda x: x['holdout']['sharpe']) if results else None
}

output_path = os.path.join(project_root, 'robust_ensemble_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

# Print summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n{'Config':<25} {'Ind':<5} {'Train Sh':<10} {'Hold Sh':<10} {'Deg%':<10} {'Train Coh':<10}")
print("-" * 70)

for r in sorted(results, key=lambda x: x['holdout']['sharpe'], reverse=True):
    print(f"{r['name']:<25} {r['n_indicators']:<5} {r['train']['sharpe']:<10.2f} {r['holdout']['sharpe']:<10.2f} {r['sharpe_degradation']:<+10.1f} {r['coherence_train']:<10.1f}")

if baseline_train:
    print(f"\n{'Baseline (single)':<25} {'1':<5} {baseline_train['sharpe']:<10.2f} {baseline_holdout['sharpe']:<10.2f}")

best = max(results, key=lambda x: x['holdout']['sharpe']) if results else None
if best:
    print(f"\n🏆 BEST: {best['name']}")
    print(f"   {best['description']}")
    print(f"   Train Sharpe: {best['train']['sharpe']:.2f}")
    print(f"   Holdout Sharpe: {best['holdout']['sharpe']:.2f}")
    print(f"   Degradation: {best['sharpe_degradation']:+.1f}%")

print(f"\nResults saved to: {output_path}")
print("=" * 70)
