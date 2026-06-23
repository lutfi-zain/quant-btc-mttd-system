#!/usr/bin/env python3
"""
Walk-Forward Validation — Best Config (MSVR_ONLY_ICH)
=======================================================

Comprehensive validation using lz-quant-researcher methodology:
1. Walk-forward with embargo
2. Statistical significance tests
3. Regime analysis
4. Cost sensitivity
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import importlib.util
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
bank_root = '/home/ubuntu/projects/quant-technical-indicator-bank'
sys.path.append(project_root)
sys.path.append(bank_root)
from indicators_helper import *
from ichimoku_quant import generate_ichimoku_features, generate_ichimoku_signals

print("=" * 70)
print("WALK-FORWARD VALIDATION — MSVR_ONLY_ICH")
print("=" * 70)

# ================================================================
# Load Data
# ================================================================
print("\n[1/7] Loading data...")

with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
    btc_data = json.load(f)

df_full = pd.DataFrame(btc_data['aligned_data'])
df_full['time'] = pd.to_datetime(df_full['time'])
df_full = df_full.set_index('time')
df_full = df_full[df_full.index >= '2018-01-01']

HOLDOUT_START = '2025-01-01'
df_train = df_full[df_full.index < HOLDOUT_START].copy()
df_holdout = df_full[df_full.index >= HOLDOUT_START].copy()

print(f"  Full: {len(df_full)} bars")
print(f"  Training: {len(df_train)} bars")
print(f"  Holdout: {len(df_holdout)} bars")

# ================================================================
# Load Indicators
# ================================================================
print("\n[2/7] Loading indicators...")

# MSVR
spec = importlib.util.spec_from_file_location('msvr', os.path.join(bank_root, 'perpetual/median_standard_deviation_viresearch.py'))
msvr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msvr_module)
msvr_full = msvr_module.median_standard_deviation_viresearch(df_full)
msvr_signal = msvr_full['vii']

# Ichimoku
df_ichimoku = generate_ichimoku_features(df_full)
df_ichimoku = generate_ichimoku_signals(df_ichimoku)

# Best config: MSVR only + Ichimoku filter
msvr_binary = (msvr_signal > 0).astype(float)
ichimoku_signal = df_ichimoku['Pos']
combined_signal = msvr_binary * ichimoku_signal

print(f"  MSVR signal: {(msvr_binary == 1).sum()} bullish")
print(f"  Ichimoku signal: {(ichimoku_signal == 1).sum()} in position")
print(f"  Combined: {(combined_signal == 1).sum()} in position")

# ================================================================
# Metrics Functions
# ================================================================
def compute_metrics(positions, prices, transaction_cost=0.001):
    """Compute comprehensive metrics."""
    returns = prices.pct_change()
    strategy_returns = returns * positions.shift(1)
    strategy_returns = strategy_returns.dropna()
    
    transitions = positions.diff().fillna(0)
    strategy_returns = strategy_returns - transitions.loc[strategy_returns.index] * (transaction_cost / 2)

    if len(strategy_returns) == 0:
        return {'cagr': 0, 'sharpe': 0, 'sortino': 0, 'calmar': 0, 'max_dd': 0, 'n_trades': 0, 'win_rate': 0}

    equity = (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25

    cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
    downside = strategy_returns[strategy_returns < 0]
    sortino = strategy_returns.mean() / downside.std() * np.sqrt(365) if len(downside) > 0 and downside.std() > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # Count trades
    changes = positions.diff().fillna(0)
    n_trades = (changes.abs() > 0).sum() // 2
    
    # Win rate
    in_position = False
    trade_returns = []
    for i, (date, pos) in enumerate(positions.items()):
        if pos == 1.0 and not in_position:
            in_position = True
            entry_price = prices.loc[date]
        elif pos == 0.0 and in_position:
            in_position = False
            exit_price = prices.loc[date]
            trade_ret = (exit_price - entry_price) / entry_price
            trade_returns.append(trade_ret)
    
    winning = sum(1 for r in trade_returns if r > 0)
    total = len(trade_returns)
    win_rate = winning / total * 100 if total > 0 else 0

    return {
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'calmar': round(calmar, 2),
        'max_dd': round(max_dd * 100, 2),
        'n_trades': n_trades,
        'win_rate': round(win_rate, 1),
        'equity': equity,
        'returns': strategy_returns
    }

# ================================================================
# 1. Walk-Forward Validation
# ================================================================
print("\n[3/7] Walk-forward validation (5 folds, 10-day embargo)...")

def walk_forward_validate(data, signal, n_folds=5, train_ratio=0.7, embargo_days=10):
    """Walk-forward validation with embargo gap."""
    results = []
    total_days = len(data)
    fold_size = total_days // n_folds
    
    for fold in range(n_folds):
        fold_start = fold * fold_size
        fold_end = min((fold + 1) * fold_size, total_days)
        
        train_end_idx = fold_start + int((fold_end - fold_start) * train_ratio)
        test_start_idx = train_end_idx + embargo_days
        
        if test_start_idx >= fold_end:
            continue
        
        train_idx = data.index[fold_start:train_end_idx]
        test_idx = data.index[test_start_idx:fold_end]
        
        train_signal = signal.reindex(train_idx).fillna(0)
        test_signal = signal.reindex(test_idx).fillna(0)
        
        train_returns = data['close'].pct_change().reindex(train_idx) * train_signal
        test_returns = data['close'].pct_change().reindex(test_idx) * test_signal
        
        is_sharpe = train_returns.dropna().mean() / train_returns.dropna().std() * np.sqrt(365) if train_returns.dropna().std() > 0 else 0
        oos_sharpe = test_returns.dropna().mean() / test_returns.dropna().std() * np.sqrt(365) if test_returns.dropna().std() > 0 else 0
        
        # Win rate
        test_positions = test_signal.reindex(test_idx).fillna(0)
        in_position = False
        trade_returns = []
        for i, (date, pos) in enumerate(test_positions.items()):
            if pos == 1.0 and not in_position:
                in_position = True
                entry_price = data.loc[date, 'close']
            elif pos == 0.0 and in_position:
                in_position = False
                exit_price = data.loc[date, 'close']
                trade_ret = (exit_price - entry_price) / entry_price
                trade_returns.append(trade_ret)
        
        winning = sum(1 for r in trade_returns if r > 0)
        total = len(trade_returns)
        win_rate = winning / total * 100 if total > 0 else 0
        
        results.append({
            'fold': fold,
            'train_period': f"{train_idx[0].date()} to {train_idx[-1].date()}",
            'test_period': f"{test_idx[0].date()} to {test_idx[-1].date()}",
            'is_sharpe': round(is_sharpe, 2),
            'oos_sharpe': round(oos_sharpe, 2),
            'oos_win_rate': round(win_rate, 1),
            'oos_trades': total,
            'decay': round((1 - oos_sharpe/is_sharpe)*100, 1) if is_sharpe > 0 else 0
        })
    
    return results

wf_results = walk_forward_validate(df_full, combined_signal, n_folds=5, embargo_days=10)

print(f"\n  Walk-Forward Results:")
print(f"  {'Fold':<6} {'Train Period':<30} {'Test Period':<30} {'IS Sharpe':<10} {'OOS Sharpe':<10} {'OOS Win%':<10} {'Decay':<10}")
print(f"  {'─'*106}")

for r in wf_results:
    print(f"  {r['fold']:<6} {r['train_period']:<30} {r['test_period']:<30} {r['is_sharpe']:<10.2f} {r['oos_sharpe']:<10.2f} {r['oos_win_rate']:<10.1f} {r['decay']:<10.1f}%")

avg_oos_sharpe = np.mean([r['oos_sharpe'] for r in wf_results])
avg_oos_winrate = np.mean([r['oos_win_rate'] for r in wf_results])
avg_decay = np.mean([r['decay'] for r in wf_results])

print(f"\n  Average OOS Sharpe:   {avg_oos_sharpe:.2f}")
print(f"  Average OOS Win Rate: {avg_oos_winrate:.1f}%")
print(f"  Average Decay:        {avg_decay:.1f}%")

# ================================================================
# 2. Training vs Holdout
# ================================================================
print("\n[4/7] Training vs Holdout comparison...")

train_metrics = compute_metrics(combined_signal[df_train.index], df_train['close'])
holdout_metrics = compute_metrics(combined_signal[df_holdout.index], df_holdout['close'])

print(f"\n  {'Metric':<20} {'Training':<15} {'Holdout':<15} {'Change':<15}")
print(f"  {'─'*65}")
print(f"  {'Sharpe':<20} {train_metrics['sharpe']:<15.2f} {holdout_metrics['sharpe']:<15.2f} {holdout_metrics['sharpe']-train_metrics['sharpe']:<+15.2f}")
print(f"  {'CAGR':<20} {train_metrics['cagr']:<15.1f}% {holdout_metrics['cagr']:<15.1f}% {holdout_metrics['cagr']-train_metrics['cagr']:<+15.1f}%")
print(f"  {'MaxDD':<20} {train_metrics['max_dd']:<15.1f}% {holdout_metrics['max_dd']:<15.1f}% {holdout_metrics['max_dd']-train_metrics['max_dd']:<+15.1f}%")
print(f"  {'Win Rate':<20} {train_metrics['win_rate']:<15.1f}% {holdout_metrics['win_rate']:<15.1f}% {holdout_metrics['win_rate']-train_metrics['win_rate']:<+15.1f}%")
print(f"  {'Trades':<20} {train_metrics['n_trades']:<15} {holdout_metrics['n_trades']:<15}")

# Degradation
if train_metrics['sharpe'] > 0:
    deg = (holdout_metrics['sharpe'] - train_metrics['sharpe']) / train_metrics['sharpe'] * 100
else:
    deg = 0
print(f"\n  Degradation: {deg:+.1f}%")

# ================================================================
# 3. Statistical Tests
# ================================================================
print("\n[5/7] Statistical significance tests...")

returns = df_full['close'].pct_change()
strategy_returns = returns * combined_signal.shift(1)
strategy_returns = strategy_returns.dropna()

t_stat, p_value = stats.ttest_1samp(strategy_returns, 0)
print(f"\n  t-test for mean return ≠ 0:")
print(f"    t-statistic: {t_stat:.4f}")
print(f"    p-value:     {p_value:.6f}")
print(f"    Significant: {'YES' if p_value < 0.05 else 'NO'} (α=0.05)")

# ================================================================
# 4. Regime Analysis
# ================================================================
print("\n[6/7] Regime analysis...")

sma50 = sma(df_full['close'], 50)
sma200 = sma(df_full['close'], 200)
regime = (sma50 > sma200).astype(float)

bull_returns = strategy_returns[regime.reindex(strategy_returns.index, fill_value=0) == 1]
bear_returns = strategy_returns[regime.reindex(strategy_returns.index, fill_value=0) == 0]

bull_sharpe = bull_returns.mean() / bull_returns.std() * np.sqrt(365) if bull_returns.std() > 0 else 0
bear_sharpe = bear_returns.mean() / bear_returns.std() * np.sqrt(365) if bear_returns.std() > 0 else 0

print(f"\n  Regime Analysis:")
print(f"    Bull market: {bull_sharpe:.2f} Sharpe ({len(bull_returns)} bars)")
print(f"    Bear market: {bear_sharpe:.2f} Sharpe ({len(bear_returns)} bars)")

# ================================================================
# 5. Cost Sensitivity
# ================================================================
print("\n[7/7] Cost sensitivity analysis...")

cost_scenarios = [0.0, 0.0005, 0.001, 0.002, 0.005]

print(f"\n  {'Cost':<10} {'Sharpe':<10} {'CAGR':<10} {'Win Rate':<10}")
print(f"  {'─'*40}")

for cost in cost_scenarios:
    metrics = compute_metrics(combined_signal[df_holdout.index], df_holdout['close'], transaction_cost=cost)
    print(f"  {cost*100:<10.2f}% {metrics['sharpe']:<10.2f} {metrics['cagr']:<10.1f}% {metrics['win_rate']:<10.1f}%")

# ================================================================
# Final Summary
# ================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\n  CONFIGURATION: MSVR_ONLY_ICH")
print(f"  Signal: MSVR binary × Ichimoku position")
print(f"\n  PERFORMANCE:")
print(f"    Training Sharpe:   {train_metrics['sharpe']:.2f}")
print(f"    Holdout Sharpe:    {holdout_metrics['sharpe']:.2f}")
print(f"    Walk-Forward OOS:  {avg_oos_sharpe:.2f}")
print(f"    Training Win Rate: {train_metrics['win_rate']:.1f}%")
print(f"    Holdout Win Rate:  {holdout_metrics['win_rate']:.1f}%")
print(f"    Walk-Forward OOS:  {avg_oos_winrate:.1f}%")

print(f"\n  VALIDATION:")
print(f"    t-test p-value:    {p_value:.6f} {'✓' if p_value < 0.05 else '✗'}")
print(f"    Regime robust:     {'YES' if abs(bull_sharpe - bear_sharpe) < 1.5 else 'NO'}")
print(f"    Cost insensitive:  {'YES' if abs(compute_metrics(combined_signal[df_holdout.index], df_holdout['close'], 0)['sharpe'] - compute_metrics(combined_signal[df_holdout.index], df_holdout['close'], 0.005)['sharpe']) < 0.1 else 'NO'}")

print(f"\n  VERDICT:")
if p_value < 0.05 and avg_oos_sharpe > 0.5:
    print(f"    ✅ STRATEGY PASSES RIGOROUS VALIDATION")
elif p_value < 0.1:
    print(f"    ⚠️ MARGINAL — needs more data")
else:
    print(f"    ❌ FAILS VALIDATION")

print("=" * 70)
