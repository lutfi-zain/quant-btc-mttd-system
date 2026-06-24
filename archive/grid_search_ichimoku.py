#!/usr/bin/env python3
"""
Grid Search — Ichimoku + MSVR + Cycle Phase
=============================================

Target: 70%+ win rate with ISP-like behavior (15-20 trades, 50-70 days hold)

Ichimoku provides:
- Ehler SuperSmoother (Family 2: Filtering)
- Shannon Entropy (Family 7: Entropy)
- Efficiency Ratio (Family 5: Fractal)
- Ichimoku Cloud (Trend)
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
from ichimoku_quant import generate_ichimoku_features, generate_ichimoku_signals, compute_ichimoku_metrics

print("=" * 70)
print("GRID SEARCH — ICHIMOKU + MSVR + CYCLE PHASE")
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

# Load ISP (new data)
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
# Generate Ichimoku Features
# ================================================================
print("\n[3/5] Generating Ichimoku features...")

df_ichimoku = generate_ichimoku_features(df_full)
df_ichimoku = generate_ichimoku_signals(df_ichimoku)

print(f"  Ichimoku signals generated")

# ================================================================
# Grid Search
# ================================================================
print("\n[4/5] Running grid search...")

results = []

# Parameter grids
min_hold_values = [45, 60, 75]
er_entry_values = [0.15, 0.20, 0.25, 0.30]
t_entry_values = [0.30, 0.40, 0.50]
entropy_thresh_values = [2.0, 2.271, 2.5]

# Test Ichimoku-only first
print(f"\n  Testing Ichimoku-only...")
ichimoku_metrics = compute_ichimoku_metrics(df_ichimoku, df_full['close'])
print(f"    Trades: {ichimoku_metrics['n_trades']}, Win Rate: {ichimoku_metrics['win_rate']}%, Sharpe: {ichimoku_metrics['sharpe']}")

# Test combined configs
configs = []

# Ichimoku + MSVR + Cycle
for mh in min_hold_values:
    for er in er_entry_values:
        for te in t_entry_values:
            configs.append({
                'name': f'ICH_MH{mh}_ER{er}_TE{te}',
                'type': 'ichimoku_msvr',
                'min_hold': mh,
                'er_entry': er,
                't_entry': te,
                'entropy_thresh': 2.271
            })

# Add entropy variations
for et in entropy_thresh_values:
    configs.append({
        'name': f'ICH_ENT{et}',
        'type': 'ichimoku_msvr',
        'min_hold': 60,
        'er_entry': 0.25,
        't_entry': 0.40,
        'entropy_thresh': et
    })

# Add pure Ichimoku with different params
configs.append({
    'name': 'ICH_PURE_DEFAULT',
    'type': 'ichimoku_only',
    'min_hold': 10,
    'er_entry': 0.25,
    't_entry': 0.40,
    'entropy_thresh': 2.271
})

configs.append({
    'name': 'ICH_PURE_CONSERVATIVE',
    'type': 'ichimoku_only',
    'min_hold': 30,
    'er_entry': 0.30,
    't_entry': 0.50,
    'entropy_thresh': 2.0
})

# Add MSVR + Bollinger (our best so far)
configs.append({
    'name': 'MSVR_BOLLINGER',
    'type': 'msvr_bollinger',
    'min_hold': 60,
    'er_entry': 0,
    't_entry': 0,
    'entropy_thresh': 0
})

# Add Ichimoku + Bollinger
configs.append({
    'name': 'ICH_BOLLINGER',
    'type': 'ichimoku_bollinger',
    'min_hold': 60,
    'er_entry': 0.25,
    't_entry': 0.40,
    'entropy_thresh': 2.271
})

def apply_min_hold(signal, min_hold):
    """Apply minimum holding period."""
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

for config in configs:
    if config['type'] == 'ichimoku_only':
        signal = df_ichimoku['Pos'].copy()
        signal = apply_min_hold(signal, config['min_hold'])
    elif config['type'] == 'ichimoku_msvr':
        # Combine Ichimoku + MSVR + Cycle
        ichimoku_signal = df_ichimoku['Pos']
        signal = raw_combined * ichimoku_signal
        signal = apply_min_hold(signal, config['min_hold'])
    elif config['type'] == 'msvr_bollinger':
        # Our best so far
        bb_period = 20
        bb_std = 2
        bb_mid = sma(df_full['close'], bb_period)
        bb_std_val = df_full['close'].rolling(bb_period).std()
        bb_upper = bb_mid + bb_std * bb_std_val
        bb_lower = bb_mid - bb_std * bb_std_val
        bb_signal = ((df_full['close'] > bb_lower) & (df_full['close'] < bb_upper)).astype(float)
        
        trend_filter = (sma(df_full['close'], 50) > sma(df_full['close'], 200)).astype(float)
        signal = raw_combined * trend_filter * bb_signal
        signal = apply_min_hold(signal, 60)
    elif config['type'] == 'ichimoku_bollinger':
        # Ichimoku + Bollinger
        bb_period = 20
        bb_std = 2
        bb_mid = sma(df_full['close'], bb_period)
        bb_std_val = df_full['close'].rolling(bb_period).std()
        bb_upper = bb_mid + bb_std * bb_std_val
        bb_lower = bb_mid - bb_std * bb_std_val
        bb_signal = ((df_full['close'] > bb_lower) & (df_full['close'] < bb_upper)).astype(float)
        
        trend_filter = (sma(df_full['close'], 50) > sma(df_full['close'], 200)).astype(float)
        ichimoku_signal = df_ichimoku['Pos']
        signal = raw_combined * trend_filter * bb_signal * ichimoku_signal
        signal = apply_min_hold(signal, 60)
    
    # Calculate metrics
    returns = df_full['close'].pct_change()
    strategy_returns = returns * signal.shift(1)
    strategy_returns = strategy_returns.dropna()
    
    equity = (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25
    cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    
    # Count trades and win rate
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
            entry_price = df_full.loc[date, 'close']
        elif pos == 0.0 and in_position:
            in_position = False
            if hold_start is not None:
                hold_days = (date - hold_start).days
                hold_periods.append(hold_days)
                exit_price = df_full.loc[date, 'close']
                trade_ret = (exit_price - entry_price) / entry_price
                trade_returns.append(trade_ret)
    
    winning = sum(1 for r in trade_returns if r > 0)
    total = len(trade_returns)
    win_rate = winning / total * 100 if total > 0 else 0
    
    avg_hold = np.mean(hold_periods) if hold_periods else 0
    
    # ISP coherence
    aligned = pd.DataFrame({'system': signal, 'benchmark': isp_positions_full}).dropna()
    coherence = (aligned['system'] == aligned['benchmark']).sum() / len(aligned) * 100 if len(aligned) > 0 else 0
    
    # Score (target: 70%+ win rate)
    trade_score = 1.0 if 15 <= n_trades <= 20 else max(0, 1 - abs(n_trades - 17) / 17)
    hold_score = 1.0 if 50 <= avg_hold <= 70 else max(0, 1 - abs(avg_hold - 60) / 60) if avg_hold > 0 else 0
    win_score = win_rate / 100
    
    # Higher weight on win rate
    score = sharpe * 0.2 + trade_score * 0.2 + hold_score * 0.2 + win_score * 0.4
    
    results.append({
        'config': config['name'],
        'type': config['type'],
        'n_trades': n_trades,
        'avg_hold': avg_hold,
        'win_rate': win_rate,
        'cagr': cagr * 100,
        'sharpe': sharpe,
        'max_dd': max_dd * 100,
        'coherence': coherence,
        'score': score,
        'n_winners': winning,
        'n_losers': total - winning
    })

# Sort by score
results.sort(key=lambda x: x['score'], reverse=True)

# Print results
print(f"\n  TOP 15 CONFIGURATIONS:")
print(f"  {'─'*110}")
print(f"  {'Config':<35} {'Type':<20} {'Trades':<8} {'Hold':<8} {'Win%':<8} {'Sharpe':<8} {'MaxDD':<8} {'Score':<8}")
print(f"  {'─'*110}")

for r in results[:15]:
    marker = "⭐" if r['win_rate'] >= 70 else "  "
    print(f"  {marker}{r['config']:<33} {r['type']:<20} {r['n_trades']:<8} {r['avg_hold']:<8.0f} {r['win_rate']:<8.1f} {r['sharpe']:<8.2f} {r['max_dd']:<8.1f} {r['score']:<8.3f}")

# Find best with 70%+ win rate
print(f"\n  TARGET: 70%+ win rate, 15-20 trades, 50-70 days hold")

best_70 = [r for r in results if r['win_rate'] >= 70]

if best_70:
    print(f"\n  ✅ 70%+ WIN RATE CONFIGURATIONS FOUND:")
    for r in best_70[:5]:
        print(f"    • {r['config']}: {r['n_trades']} trades, {r['avg_hold']:.0f} days hold, {r['win_rate']:.0f}% win, Sharpe {r['sharpe']:.2f}")
else:
    print(f"\n  ⚠️ No 70%+ win rate found, showing best:")
    for r in results[:5]:
        print(f"    • {r['config']}: {r['n_trades']} trades, {r['avg_hold']:.0f} days hold, {r['win_rate']:.0f}% win, Sharpe {r['sharpe']:.2f}")
