#!/usr/bin/env python3
"""
Grid Search — Find BEST Performance
=====================================

Test ALL combinations to find absolute best performance.
Focus on: Sharpe, CAGR, MaxDD, Win Rate
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
from ichimoku_quant import generate_ichimoku_features, generate_ichimoku_signals

print("=" * 70)
print("GRID SEARCH — FIND BEST PERFORMANCE")
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

HOLDOUT_START = '2025-01-01'
df_train = df_full[df_full.index < HOLDOUT_START].copy()
df_holdout = df_full[df_full.index >= HOLDOUT_START].copy()

print(f"  Training: {len(df_train)} bars")
print(f"  Holdout:  {len(df_holdout)} bars")

# ================================================================
# Load Indicators
# ================================================================
print("\n[2/4] Loading indicators...")

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

# Basic signals
msvr_binary = (msvr_signal > 0).astype(float)
cycle_binary = (cycle_signal > 0).astype(float)
raw_combined = msvr_binary * cycle_binary

# Ichimoku
df_ichimoku = generate_ichimoku_features(df_full)
df_ichimoku = generate_ichimoku_signals(df_ichimoku)

# Trend filters
trend_50_200 = (sma(df_full['close'], 50) > sma(df_full['close'], 200)).astype(float)
trend_100_200 = (sma(df_full['close'], 100) > sma(df_full['close'], 200)).astype(float)

# Bollinger
bb_mid = sma(df_full['close'], 20)
bb_std = df_full['close'].rolling(20).std()
bb_upper = bb_mid + 2 * bb_std
bb_lower = bb_mid - 2 * bb_std
bb_signal = ((df_full['close'] > bb_lower) & (df_full['close'] < bb_upper)).astype(float)

# ================================================================
# Grid Search
# ================================================================
print("\n[3/4] Running comprehensive grid search...")

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

# Test ALL combinations
configs = []

# 1. MSVR + Cycle Phase (baseline)
for mh in [0, 30, 45, 60, 75, 90]:
    for trend in [None, '50_200', '100_200']:
        for bb in [False, True]:
            for ichimoku in [False, True]:
                name = f"MSVR_CYCLE"
                if trend: name += f"_T{trend}"
                if bb: name += "_BB"
                if ichimoku: name += "_ICH"
                if mh > 0: name += f"_MH{mh}"
                
                configs.append({
                    'name': name,
                    'base': raw_combined,
                    'trend': trend,
                    'bb': bb,
                    'ichimoku': ichimoku,
                    'min_hold': mh
                })

# 2. MSVR only
for mh in [0, 30, 45, 60, 75, 90]:
    for trend in [None, '50_200', '100_200']:
        for bb in [False, True]:
            for ichimoku in [False, True]:
                name = f"MSVR_ONLY"
                if trend: name += f"_T{trend}"
                if bb: name += "_BB"
                if ichimoku: name += "_ICH"
                if mh > 0: name += f"_MH{mh}"
                
                configs.append({
                    'name': name,
                    'base': msvr_binary,
                    'trend': trend,
                    'bb': bb,
                    'ichimoku': ichimoku,
                    'min_hold': mh
                })

# 3. Cycle Phase only
for mh in [0, 30, 45, 60, 75, 90]:
    for trend in [None, '50_200', '100_200']:
        for bb in [False, True]:
            for ichimoku in [False, True]:
                name = f"CYCLE_ONLY"
                if trend: name += f"_T{trend}"
                if bb: name += "_BB"
                if ichimoku: name += "_ICH"
                if mh > 0: name += f"_MH{mh}"
                
                configs.append({
                    'name': name,
                    'base': cycle_binary,
                    'trend': trend,
                    'bb': bb,
                    'ichimoku': ichimoku,
                    'min_hold': mh
                })

# 4. Ichimoku only
for mh in [0, 10, 20, 30]:
    configs.append({
        'name': f"ICH_ONLY_MH{mh}",
        'base': df_ichimoku['Pos'],
        'trend': None,
        'bb': False,
        'ichimoku': False,
        'min_hold': mh
    })

# 5. Best combos
configs.append({
    'name': 'BEST_PREV',
    'base': raw_combined * trend_50_200 * bb_signal,
    'trend': None,
    'bb': False,
    'ichimoku': False,
    'min_hold': 60
})

print(f"  Testing {len(configs)} configurations...")

for config in configs:
    signal = config['base'].copy()
    
    # Apply trend filter
    if config['trend'] == '50_200':
        signal = signal * trend_50_200
    elif config['trend'] == '100_200':
        signal = signal * trend_100_200
    
    # Apply Bollinger
    if config['bb']:
        signal = signal * bb_signal
    
    # Apply Ichimoku
    if config['ichimoku']:
        signal = signal * df_ichimoku['Pos']
    
    # Apply min hold
    if config['min_hold'] > 0:
        signal = apply_min_hold(signal, config['min_hold'])
    
    # Calculate metrics
    metrics = calc_metrics(signal, df_full['close'])
    
    # Score (weight Sharpe and CAGR)
    score = metrics['sharpe'] * 0.4 + (metrics['cagr'] / 100) * 0.3 + (metrics['win_rate'] / 100) * 0.3
    
    results.append({
        'config': config['name'],
        **metrics,
        'score': score
    })

# Sort by score
results.sort(key=lambda x: x['score'], reverse=True)

# Print results
print(f"\n  TOP 20 CONFIGURATIONS:")
print(f"  {'─'*100}")
print(f"  {'Config':<45} {'Trades':<8} {'Hold':<8} {'Win%':<8} {'Sharpe':<8} {'CAGR':<8} {'MaxDD':<8}")
print(f"  {'─'*100}")

for r in results[:20]:
    print(f"  {r['config']:<45} {r['n_trades']:<8} {r['avg_hold']:<8.0f} {r['win_rate']:<8.1f} {r['sharpe']:<8.2f} {r['cagr']:<8.1f} {r['max_dd']:<8.1f}")

# Find best by different metrics
print(f"\n  BEST BY METRIC:")
print(f"  {'─'*80}")

best_sharpe = max(results, key=lambda x: x['sharpe'])
best_cagr = max(results, key=lambda x: x['cagr'])
best_winrate = max(results, key=lambda x: x['win_rate'])
best_maxdd = max(results, key=lambda x: x['max_dd'])  # Least negative

print(f"  Best Sharpe:    {best_sharpe['config']}: Sharpe={best_sharpe['sharpe']:.2f}, CAGR={best_sharpe['cagr']:.1f}%, Win={best_sharpe['win_rate']:.0f}%")
print(f"  Best CAGR:      {best_cagr['config']}: CAGR={best_cagr['cagr']:.1f}%, Sharpe={best_cagr['sharpe']:.2f}, Win={best_cagr['win_rate']:.0f}%")
print(f"  Best Win Rate:  {best_winrate['config']}: Win={best_winrate['win_rate']:.0f}%, Sharpe={best_winrate['sharpe']:.2f}, CAGR={best_winrate['cagr']:.1f}%")
print(f"  Best MaxDD:     {best_maxdd['config']}: MaxDD={best_maxdd['max_dd']:.1f}%, Sharpe={best_maxdd['sharpe']:.2f}, Win={best_maxdd['win_rate']:.0f}%")
