#!/usr/bin/env python3
"""
Test ALL indicators from quant-technical-indicator-bank
Computes individual performance metrics and correlations
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
print("TEST ALL INDICATORS FROM BANK")
print("=" * 70)

# ================================================================
# Load Data
# ================================================================
print("\n[1/4] Loading data...")

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

print(f"  Full data: {len(df_full)} bars ({df_full.index[0]} to {df_full.index[-1]})")
print(f"  Training:  {len(df_train)} bars")
print(f"  Holdout:   {len(df_holdout)} bars")

# Load ISP benchmark
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

print(f"  ISP: {(isp_positions_train == 1.0).mean()*100:.1f}% in position")

# ================================================================
# Metrics Functions
# ================================================================
def compute_trading_metrics(positions, prices, transaction_cost=0.001):
    """Compute trading performance metrics with transaction costs."""
    returns = prices.pct_change()
    strategy_returns = returns * positions.shift(1)
    strategy_returns = strategy_returns.dropna()

    transitions = positions.diff().fillna(0)
    cost_per_transition = transaction_cost / 2
    strategy_returns = strategy_returns - transitions.loc[strategy_returns.index] * cost_per_transition

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
    """Compute time-coherence with ISP benchmark."""
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

def load_and_compute(indicator_name, category, df):
    """Try to load and compute indicator with default params."""
    try:
        # Try bank first, then local
        for base_path in [bank_root, project_root]:
            filename = f"{indicator_name}.py"
            module_path = os.path.join(base_path, category, filename)
            if os.path.exists(module_path):
                spec = importlib.util.spec_from_file_location(indicator_name, module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                func = getattr(module, indicator_name)
                
                # Try with default params
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
# Discover all indicators
# ================================================================
print("\n[2/4] Discovering indicators...")

indicators = []

# Oscillator folder
osc_path = os.path.join(bank_root, 'oscillator')
if os.path.exists(os.path.join(project_root, 'oscillator')):
    for f in os.listdir(os.path.join(project_root, 'oscillator')):
        if f.endswith('.py') and not f.startswith('__'):
            indicators.append({
                'name': f.replace('.py', ''),
                'category': 'oscillator',
                'source': 'local' if os.path.exists(os.path.join(project_root, 'oscillator', f)) else 'bank'
            })

# Perpetual folder
perp_path = os.path.join(bank_root, 'perpetual')
if os.path.exists(os.path.join(project_root, 'perpetual')):
    for f in os.listdir(os.path.join(project_root, 'perpetual')):
        if f.endswith('.py') and not f.startswith('__'):
            indicators.append({
                'name': f.replace('.py', ''),
                'category': 'perpetual',
                'source': 'local' if os.path.exists(os.path.join(project_root, 'perpetual', f)) else 'bank'
            })

# Also check bank for indicators not in local
for category in ['oscillator', 'perpetual']:
    bank_cat_path = os.path.join(bank_root, category)
    local_cat_path = os.path.join(project_root, category)
    if os.path.exists(bank_cat_path):
        for f in os.listdir(bank_cat_path):
            if f.endswith('.py') and not f.startswith('__'):
                ind_name = f.replace('.py', '')
                # Check if already in list
                existing = [i['name'] for i in indicators]
                if ind_name not in existing:
                    indicators.append({
                        'name': ind_name,
                        'category': category,
                        'source': 'bank_only'
                    })

print(f"  Found {len(indicators)} indicators")
print(f"  Oscillator: {sum(1 for i in indicators if i['category'] == 'oscillator')}")
print(f"  Perpetual: {sum(1 for i in indicators if i['category'] == 'perpetual')}")

# ================================================================
# Test all indicators
# ================================================================
print("\n[3/4] Testing all indicators...")

results = []
signals_for_corr = {}

for i, ind in enumerate(indicators):
    ind_name = ind['name']
    category = ind['category']
    
    print(f"\n  [{i+1}/{len(indicators)}] {ind_name} ({category})...", end=' ')
    
    signal = load_and_compute(ind_name, category, df_train)
    
    if signal is not None:
        position = (signal > 0).astype(float)
        metrics = compute_trading_metrics(position, df_train['close'])
        coherence = compute_coherence(signal, isp_positions_train)
        
        result = {
            'name': ind_name,
            'category': category,
            'source': ind['source'],
            'coherence': round(coherence, 1),
            **metrics
        }
        results.append(result)
        signals_for_corr[ind_name] = signal
        
        print(f"Sharpe={metrics['sharpe']:.2f}, CAGR={metrics['cagr']:.1f}%, Coherence={coherence:.1f}%")
    else:
        print("FAILED")

# Sort by Sharpe
results.sort(key=lambda x: x['sharpe'], reverse=True)

# ================================================================
# Compute correlations
# ================================================================
print("\n[4/4] Computing correlations...")

if len(signals_for_corr) > 1:
    sig_df = pd.DataFrame(signals_for_corr)
    corr_matrix = sig_df.corr()
    
    # Find high correlation pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append({
                    'ind1': corr_matrix.index[i],
                    'ind2': corr_matrix.columns[j],
                    'correlation': round(corr_matrix.iloc[i, j], 3)
                })
    
    high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
else:
    corr_matrix = None
    high_corr_pairs = []

# ================================================================
# Save results
# ================================================================
output = {
    'total_indicators': len(indicators),
    'successful': len(results),
    'failed': len(indicators) - len(results),
    'results': results,
    'high_correlation_pairs': high_corr_pairs[:50],  # Top 50
    'summary': {
        'best_sharpe': results[0] if results else None,
        'best_cagr': max(results, key=lambda x: x['cagr']) if results else None,
        'best_coherence': max(results, key=lambda x: x['coherence']) if results else None,
        'avg_sharpe': round(np.mean([r['sharpe'] for r in results]), 2) if results else 0,
        'avg_cagr': round(np.mean([r['cagr'] for r in results]), 2) if results else 0,
    }
}

output_path = os.path.join(project_root, 'all_indicators_test_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

# Print summary
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

print(f"\nTotal indicators: {len(indicators)}")
print(f"Successful: {len(results)}")
print(f"Failed: {len(indicators) - len(results)}")

if results:
    print(f"\n{'Rank':<5} {'Indicator':<45} {'Sharpe':<8} {'CAGR':<8} {'MaxDD':<8} {'Coherence':<10}")
    print("-" * 84)
    for i, r in enumerate(results[:20], 1):
        print(f"{i:<5} {r['name']:<45} {r['sharpe']:<8.2f} {r['cagr']:<8.1f} {r['max_dd']:<8.1f} {r['coherence']:<10.1f}")

    print(f"\nBest Sharpe:    {results[0]['name']} ({results[0]['sharpe']:.2f})")
    best_cagr = max(results, key=lambda x: x['cagr'])
    print(f"Best CAGR:      {best_cagr['name']} ({best_cagr['cagr']:.1f}%)")
    best_coh = max(results, key=lambda x: x['coherence'])
    print(f"Best Coherence: {best_coh['name']} ({best_coh['coherence']:.1f}%)")

if high_corr_pairs:
    print(f"\nTop 10 High Correlation Pairs (>0.7):")
    for pair in high_corr_pairs[:10]:
        print(f"  {pair['ind1']:<35} ↔ {pair['ind2']:<35} = {pair['correlation']:.3f}")

print(f"\nResults saved to: {output_path}")
print("=" * 70)
