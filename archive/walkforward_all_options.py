#!/usr/bin/env python3
"""
Walk-Forward Validation — All Options
=======================================

Validate all recommended configs:
1. Best Win Rate: T100/150_BB15_3.0s_MH90
2. Most Robust: T75/250_BB25_2.0s_MH45
3. ISP-Like: T75/150_BB30_1.5s_MH60
4. Best Sharpe: T100/250_BB20_1.5s_MH90

Use walk-forward + holdout to find truly best performer.
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
print("WALK-FORWARD VALIDATION — ALL OPTIONS")
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

print(f"  Full: {len(df_full)} bars")
print(f"  Training: {len(df_train)} bars")
print(f"  Holdout: {len(df_holdout)} bars")

# ================================================================
# Load Indicators
# ================================================================
print("\n[2/5] Loading indicators...")

# MSVR
spec = importlib.util.spec_from_file_location('msvr', os.path.join(bank_root, 'perpetual/median_standard_deviation_viresearch.py'))
msvr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msvr_module)
msvr_full = msvr_module.median_standard_deviation_viresearch(df_full)
msvr_signal = msvr_full['vii']

# Cycle Phase
def compute_cycle_phase(df, lookback):
    src = (df['high'] + df['low'] + df['close']) / 3.0
    n = len(df)
    phase = pd.Series(np.nan, index=df.index)
    min_period = 5
    max_period = lookback // 2
    
    for i in range(lookback - 1, n):
        window = src.iloc[i - lookback + 1:i + 1].values
        if np.any(np.isnan(window)):
            continue
        window_detrended = window - np.mean(window)
        hann = np.hanning(lookback)
        window窗ed = window_detrended * hann
        fft_vals = np.fft.rfft(window窗ed)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(lookback, d=1)
        min_freq = 1.0 / max_period
        max_freq = 1.0 / min_period
        valid_mask = (freqs >= min_freq) & (freqs <= max_period)
        valid_power = power[valid_mask]
        valid_freqs = freqs[valid_mask]
        
        if len(valid_power) > 0 and np.sum(valid_power) > 0:
            dominant_idx = np.argmax(valid_power)
            dominant_freq = valid_freqs[dominant_idx]
            dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else lookback
            cycle_pos = i % int(dominant_period)
            phase.iloc[i] = 2 * np.pi * cycle_pos / dominant_period
    
    return phase

phase = compute_cycle_phase(df_full, lookback=40)
cycle_signal = -np.cos(phase)

# Basic combined signal
msvr_binary = (msvr_signal > 0).astype(float)
cycle_binary = (cycle_signal > 0).astype(float)
raw_combined = msvr_binary * cycle_binary

# ================================================================
# Define Configs
# ================================================================
print("\n[3/5] Defining configurations...")

configs = {
    'A_BEST_WINRATE': {
        'name': 'Best Win Rate (75%)',
        'trend_fast': 100,
        'trend_slow': 150,
        'bb_period': 15,
        'bb_std': 3.0,
        'min_hold': 90
    },
    'B_MOST_ROBUST': {
        'name': 'Most Robust (-0.5% deg)',
        'trend_fast': 75,
        'trend_slow': 250,
        'bb_period': 25,
        'bb_std': 2.0,
        'min_hold': 45
    },
    'C_ISP_LIKE': {
        'name': 'ISP-Like (19 trades, 62d)',
        'trend_fast': 75,
        'trend_slow': 150,
        'bb_period': 30,
        'bb_std': 1.5,
        'min_hold': 60
    },
    'D_BEST_SHARPE': {
        'name': 'Best Sharpe (0.67)',
        'trend_fast': 100,
        'trend_slow': 250,
        'bb_period': 20,
        'bb_std': 1.5,
        'min_hold': 90
    }
}

# ================================================================
# Helper Functions
# ================================================================
def apply_min_hold(signal, min_hold):
    result = signal.copy()
    in_position = False
    hold_count = 0
    for i in range(len(result)):
        if result.iloc[i] == 1.0 and not in_position:
            in_position = True
            hold_count = 0
        elif result.iloc[i] == 0.0 and in_position:
            if hold_count < min_hold:
                result.iloc[i] = 1.0
                hold_count += 1
            else:
                in_position = False
                hold_count = 0
        elif in_position:
            hold_count += 1
    return result

def compute_signal(config, df):
    """Compute signal for given config."""
    trend_fast = sma(df['close'], config['trend_fast'])
    trend_slow = sma(df['close'], config['trend_slow'])
    trend_filter = (trend_fast > trend_slow).astype(float)
    
    bb_mid = sma(df['close'], config['bb_period'])
    bb_std = df['close'].rolling(config['bb_period']).std()
    bb_upper = bb_mid + config['bb_std'] * bb_std
    bb_lower = bb_mid - config['bb_std'] * bb_std
    bb_signal = ((df['close'] > bb_lower) & (df['close'] < bb_upper)).astype(float)
    
    signal = raw_combined * trend_filter * bb_signal
    signal = apply_min_hold(signal, config['min_hold'])
    
    return signal

def compute_metrics(positions, prices):
    """Compute comprehensive metrics."""
    returns = prices.pct_change()
    strategy_returns = returns * positions.shift(1)
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) == 0:
        return {'cagr': 0, 'sharpe': 0, 'sortino': 0, 'calmar': 0, 'max_dd': 0, 'n_trades': 0, 'win_rate': 0, 'avg_hold': 0}

    equity = (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25

    cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    
    changes = positions.diff().fillna(0)
    n_trades = (changes.abs() > 0).sum() // 2
    
    in_position = False
    hold_start = None
    hold_periods = []
    trade_returns = []
    
    for i, (date, pos) in enumerate(positions.items()):
        if pos == 1.0 and not in_position:
            in_position = True
            hold_start = date
            entry_price = prices.loc[date]
        elif pos == 0.0 and in_position:
            in_position = False
            if hold_start is not None:
                hold_days = (date - hold_start).days
                hold_periods.append(hold_days)
                exit_price = prices.loc[date]
                trade_ret = (exit_price - entry_price) / entry_price
                trade_returns.append(trade_ret)
    
    winning = sum(1 for r in trade_returns if r > 0)
    total = len(trade_returns)
    win_rate = winning / total * 100 if total > 0 else 0
    avg_hold = np.mean(hold_periods) if hold_periods else 0

    return {
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd * 100, 2),
        'n_trades': n_trades,
        'win_rate': round(win_rate, 1),
        'avg_hold': round(avg_hold, 0),
        'equity': equity
    }

def walk_forward_validate(data, signal, n_folds=5, train_ratio=0.7, embargo_days=10):
    """Walk-forward validation."""
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
        
        test_idx = data.index[test_start_idx:fold_end]
        
        test_signal = signal.reindex(test_idx).fillna(0)
        test_returns = data['close'].pct_change().reindex(test_idx) * test_signal
        
        oos_sharpe = test_returns.dropna().mean() / test_returns.dropna().std() * np.sqrt(365) if test_returns.dropna().std() > 0 else 0
        
        # Win rate
        in_position = False
        trade_returns = []
        for i, (date, pos) in enumerate(test_signal.items()):
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
            'oos_sharpe': round(oos_sharpe, 2),
            'oos_winrate': round(win_rate, 1),
            'oos_trades': total
        })
    
    return results

# ================================================================
# Run Walk-Forward for All Configs
# ================================================================
print("\n[4/5] Running walk-forward validation...")

all_results = {}

for config_key, config in configs.items():
    print(f"\n  Testing: {config['name']}")
    
    # Compute signal
    signal = compute_signal(config, df_full)
    
    # Full metrics
    full_metrics = compute_metrics(signal, df_full['close'])
    
    # Training metrics
    train_signal = signal[df_train.index]
    train_metrics = compute_metrics(train_signal, df_train['close'])
    
    # Holdout metrics
    holdout_signal = signal[df_holdout.index]
    holdout_metrics = compute_metrics(holdout_signal, df_holdout['close'])
    
    # Walk-forward
    wf_results = walk_forward_validate(df_full, signal, n_folds=5, embargo_days=10)
    avg_oos_sharpe = np.mean([r['oos_sharpe'] for r in wf_results])
    avg_oos_winrate = np.mean([r['oos_winrate'] for r in wf_results])
    
    # Degradation
    if train_metrics['sharpe'] > 0:
        deg = (holdout_metrics['sharpe'] - train_metrics['sharpe']) / train_metrics['sharpe'] * 100
    else:
        deg = 0
    
    all_results[config_key] = {
        'config': config,
        'full': full_metrics,
        'train': train_metrics,
        'holdout': holdout_metrics,
        'wf': wf_results,
        'avg_oos_sharpe': avg_oos_sharpe,
        'avg_oos_winrate': avg_oos_winrate,
        'degradation': deg
    }
    
    print(f"    Full: Sharpe={full_metrics['sharpe']:.2f}, Win={full_metrics['win_rate']:.0f}%, Trades={full_metrics['n_trades']}")
    print(f"    Train: Sharpe={train_metrics['sharpe']:.2f}, Win={train_metrics['win_rate']:.0f}%")
    print(f"    Holdout: Sharpe={holdout_metrics['sharpe']:.2f}, Win={holdout_metrics['win_rate']:.0f}%")
    print(f"    WF OOS: Sharpe={avg_oos_sharpe:.2f}, Win={avg_oos_winrate:.0f}%")
    print(f"    Degradation: {deg:+.1f}%")

# ================================================================
# Final Comparison
# ================================================================
print("\n[5/5] Final comparison...")
print("\n" + "=" * 70)
print("FINAL COMPARISON — ALL OPTIONS")
print("=" * 70)

print(f"\n  {'Config':<25} {'Full Sh':<10} {'Train Sh':<10} {'Hold Sh':<10} {'WF OOS Sh':<10} {'Full Win':<10} {'Hold Win':<10} {'WF Win':<10} {'Deg%':<10}")
print(f"  {'─'*105}")

for key, result in all_results.items():
    print(f"  {result['config']['name']:<25} {result['full']['sharpe']:<10.2f} {result['train']['sharpe']:<10.2f} {result['holdout']['sharpe']:<10.2f} {result['avg_oos_sharpe']:<10.2f} {result['full']['win_rate']:<10.0f} {result['holdout']['win_rate']:<10.0f} {result['avg_oos_winrate']:<10.0f} {result['degradation']:<+10.1f}")

# Find best overall
print(f"\n  RECOMMENDATION:")
print(f"  {'─'*70}")

# Score each config
scores = {}
for key, result in all_results.items():
    # Score: WF OOS Sharpe + WF Win Rate + Robustness (low degradation)
    score = result['avg_oos_sharpe'] * 30 + result['avg_oos_winrate'] * 0.5 + (100 - abs(result['degradation'])) * 0.2
    scores[key] = score

best_key = max(scores, key=scores.get)
best_result = all_results[best_key]

print(f"\n  🏆 BEST OVERALL: {best_result['config']['name']}")
print(f"     Config: T{best_result['config']['trend_fast']}/{best_result['config']['trend_slow']}_BB{best_result['config']['bb_period']}_{best_result['config']['bb_std']}s_MH{best_result['config']['min_hold']}")
print(f"     WF OOS Sharpe: {best_result['avg_oos_sharpe']:.2f}")
print(f"     WF OOS Win Rate: {best_result['avg_oos_winrate']:.0f}%")
print(f"     Degradation: {best_result['degradation']:+.1f}%")
print(f"     Full Metrics: Sharpe={best_result['full']['sharpe']:.2f}, Win={best_result['full']['win_rate']:.0f}%, CAGR={best_result['full']['cagr']:.1f}%")

# Walk-forward details for best
print(f"\n  WALK-FORWARD DETAILS (Best Config):")
print(f"  {'─'*50}")
for r in best_result['wf']:
    print(f"    Fold {r['fold']}: OOS Sharpe={r['oos_sharpe']:.2f}, Win={r['oos_winrate']:.0f}%, Trades={r['oos_trades']}")
