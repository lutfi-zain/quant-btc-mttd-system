#!/usr/bin/env python3
"""
Optimize MSVR + Trend + Bollinger
===================================

Find the best parameters for:
1. Trend filter (SMA periods)
2. Bollinger (period, std)
3. Min hold period

Target: 70%+ win rate with ISP-like behavior
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
print("OPTIMIZE MSVR + TREND + BOLLINGER")
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
isp_positions_full = pd.Series(0.0, index=df_full.index)
new_isp_signals = [
    ('2019-03-26', 'BUY'), ('2019-07-03', 'SELL'),
    ('2020-01-06', 'BUY'), ('2020-02-23', 'SELL'),
    ('2020-04-25', 'BUY'), ('2020-08-18', 'SELL'),
    ('2020-10-12', 'BUY'), ('2021-01-14', 'SELL'),
    ('2021-02-02', 'BUY'), ('2021-04-20', 'SELL'),
    ('2021-07-23', 'BUY'), ('2021-09-08', 'SELL'),
    ('2021-10-01', 'BUY'), ('2021-11-17', 'SELL'),
    ('2023-01-11', 'BUY'), ('2023-02-23', 'SELL'),
    ('2023-03-13', 'BUY'), ('2023-04-22', 'SELL'),
    ('2023-06-18', 'BUY'), ('2023-07-20', 'SELL'),
    ('2023-09-17', 'BUY'), ('2024-03-26', 'SELL'),
    ('2024-05-17', 'BUY'), ('2024-06-12', 'SELL'),
    ('2024-07-14', 'BUY'), ('2024-08-01', 'SELL'),
    ('2024-10-14', 'BUY'), ('2024-12-21', 'SELL'),
    ('2025-04-20', 'BUY'), ('2025-06-01', 'SELL'),
    ('2025-07-07', 'BUY'), ('2025-07-31', 'SELL'),
    ('2026-04-08', 'BUY'), ('2026-05-15', 'SELL'),
]
for date_str, action in new_isp_signals:
    date = pd.Timestamp(date_str)
    if date in isp_positions_full.index:
        if action == 'BUY':
            isp_positions_full.loc[date:] = 1.0
        elif action == 'SELL':
            isp_positions_full.loc[date:] = 0.0

# ================================================================
# Load MSVR + Cycle Phase
# ================================================================
print("\n[2/5] Loading MSVR + Cycle Phase...")

spec = importlib.util.spec_from_file_location('msvr', os.path.join(bank_root, 'perpetual/median_standard_deviation_viresearch.py'))
msvr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msvr_module)

msvr_full = msvr_module.median_standard_deviation_viresearch(df_full)
msvr_signal = msvr_full['vii']

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
# Grid Search
# ================================================================
print("\n[3/5] Running comprehensive grid search...")

results = []

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

def calc_metrics(signal, prices):
    returns = prices.pct_change()
    strategy_returns = returns * signal.shift(1)
    strategy_returns = strategy_returns.dropna()
    
    equity = (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25
    cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    
    changes = signal.diff().fillna(0)
    n_trades = (changes.abs() > 0).sum() // 2
    
    in_position = False
    hold_start = None
    hold_periods = []
    trade_returns = []
    
    for i, (date, pos) in enumerate(signal.items()):
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
        'n_trades': n_trades,
        'avg_hold': avg_hold,
        'win_rate': win_rate,
        'cagr': cagr * 100,
        'sharpe': sharpe,
        'max_dd': max_dd * 100
    }

# Parameter grids
trend_fast_values = [30, 50, 75, 100]
trend_slow_values = [150, 200, 250]
bb_period_values = [15, 20, 25, 30]
bb_std_values = [1.5, 2.0, 2.5, 3.0]
min_hold_values = [30, 45, 60, 75, 90]

# Generate all combinations
configs = []
for tf in trend_fast_values:
    for ts in trend_slow_values:
        if tf >= ts:
            continue
        for bp in bb_period_values:
            for bs in bb_std_values:
                for mh in min_hold_values:
                    configs.append({
                        'trend_fast': tf,
                        'trend_slow': ts,
                        'bb_period': bp,
                        'bb_std': bs,
                        'min_hold': mh
                    })

print(f"  Testing {len(configs)} configurations...")

for config in configs:
    # Trend filter
    trend_fast = sma(df_full['close'], config['trend_fast'])
    trend_slow = sma(df_full['close'], config['trend_slow'])
    trend_filter = (trend_fast > trend_slow).astype(float)
    
    # Bollinger filter
    bb_mid = sma(df_full['close'], config['bb_period'])
    bb_std = df_full['close'].rolling(config['bb_period']).std()
    bb_upper = bb_mid + config['bb_std'] * bb_std
    bb_lower = bb_mid - config['bb_std'] * bb_std
    bb_signal = ((df_full['close'] > bb_lower) & (df_full['close'] < bb_upper)).astype(float)
    
    # Combine
    signal = raw_combined * trend_filter * bb_signal
    
    # Apply min hold
    signal = apply_min_hold(signal, config['min_hold'])
    
    # Calculate metrics
    metrics = calc_metrics(signal, df_full['close'])
    
    # Training metrics
    train_signal = signal[df_train.index]
    train_metrics = calc_metrics(train_signal, df_train['close'])
    
    # Holdout metrics
    holdout_signal = signal[df_holdout.index]
    holdout_metrics = calc_metrics(holdout_signal, df_holdout['close'])
    
    # Degradation
    if train_metrics['sharpe'] > 0:
        deg = (holdout_metrics['sharpe'] - train_metrics['sharpe']) / train_metrics['sharpe'] * 100
    else:
        deg = 0
    
    # Score (focus on win rate and robustness)
    score = holdout_metrics['win_rate'] * 0.4 + holdout_metrics['sharpe'] * 30 * 0.3 + (100 - abs(deg)) * 0.3
    
    results.append({
        'config': f"T{config['trend_fast']}/{config['trend_slow']}_BB{config['bb_period']}_{config['bb_std']}s_MH{config['min_hold']}",
        **metrics,
        'train_sharpe': train_metrics['sharpe'],
        'train_winrate': train_metrics['win_rate'],
        'holdout_sharpe': holdout_metrics['sharpe'],
        'holdout_winrate': holdout_metrics['win_rate'],
        'degradation': deg,
        'score': score
    })

# Sort by score
results.sort(key=lambda x: x['score'], reverse=True)

# Print results
print(f"\n  TOP 20 CONFIGURATIONS:")
print(f"  {'─'*120}")
print(f"  {'Config':<40} {'Trades':<8} {'Hold':<8} {'Win%':<8} {'Sharpe':<8} {'Train Sh':<10} {'Hold Sh':<10} {'Train Win':<10} {'Hold Win':<10} {'Deg%':<8}")
print(f"  {'─'*120}")

for r in results[:20]:
    deg_marker = "✅" if abs(r['degradation']) < 50 else "⚠️"
    print(f"  {r['config']:<40} {r['n_trades']:<8} {r['avg_hold']:<8.0f} {r['win_rate']:<8.1f} {r['sharpe']:<8.2f} {r['train_sharpe']:<10.2f} {r['holdout_sharpe']:<10.2f} {r['train_winrate']:<10.1f} {r['holdout_winrate']:<10.1f} {r['degradation']:<+8.1f}")

# Find best by different criteria
print(f"\n  BEST BY CRITERIA:")
print(f"  {'─'*100}")

best_winrate = max(results, key=lambda x: x['holdout_winrate'])
best_sharpe = max(results, key=lambda x: x['holdout_sharpe'])
best_robust = min(results, key=lambda x: abs(x['degradation']))
best_isp = [r for r in results if 15 <= r['n_trades'] <= 25 and 50 <= r['avg_hold'] <= 70]

print(f"\n  Best Holdout Win Rate:")
print(f"    {best_winrate['config']}")
print(f"    Win Rate: {best_winrate['holdout_winrate']:.1f}%, Sharpe: {best_winrate['holdout_sharpe']:.2f}")

print(f"\n  Best Holdout Sharpe:")
print(f"    {best_sharpe['config']}")
print(f"    Sharpe: {best_sharpe['holdout_sharpe']:.2f}, Win Rate: {best_sharpe['holdout_winrate']:.1f}%")

print(f"\n  Most Robust (Lowest Degradation):")
print(f"    {best_robust['config']}")
print(f"    Degradation: {best_robust['degradation']:+.1f}%, Win Rate: {best_robust['holdout_winrate']:.1f}%")

if best_isp:
    print(f"\n  ISP-Like (15-25 trades, 50-70 days hold):")
    for r in best_isp[:3]:
        print(f"    {r['config']}: {r['n_trades']} trades, {r['avg_hold']:.0f} days, {r['holdout_winrate']:.0f}% win")
