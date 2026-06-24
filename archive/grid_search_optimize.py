#!/usr/bin/env python3
"""
Grid Search — Optimize + Add New Indicators
=============================================

Option 1: Optimize min_hold + trend filter
Option 2: Add volume, momentum, volatility filters

Target: 15-20 trades, 50-70 days hold, ALL profitable
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
print("GRID SEARCH — OPTIMIZE + ADD NEW INDICATORS")
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

# New ISP signals from user
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

# ISP metrics
isp_returns = df_full['close'].pct_change() * isp_positions_full.shift(1)
isp_returns = isp_returns.dropna()
isp_sharpe = isp_returns.mean() / isp_returns.std() * np.sqrt(365) if isp_returns.std() > 0 else 0

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
        valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
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
# New Indicators
# ================================================================
print("\n[3/5] Computing new indicators...")

# 1. Volume Indicators
# OBV (On-Balance Volume)
obv = pd.Series(0.0, index=df_full.index)
for i in range(1, len(df_full)):
    if df_full['close'].iloc[i] > df_full['close'].iloc[i-1]:
        obv.iloc[i] = obv.iloc[i-1] + df_full['volume'].iloc[i]
    elif df_full['close'].iloc[i] < df_full['close'].iloc[i-1]:
        obv.iloc[i] = obv.iloc[i-1] - df_full['volume'].iloc[i]
    else:
        obv.iloc[i] = obv.iloc[i-1]

obv_signal = (obv > obv.rolling(20).mean()).astype(float)

# VWAP (simplified - price weighted by volume)
typical_price = (df_full['high'] + df_full['low'] + df_full['close']) / 3
vwap = (typical_price * df_full['volume']).cumsum() / df_full['volume'].cumsum()
vwap_signal = (df_full['close'] > vwap).astype(float)

# 2. Momentum Indicators
# RSI
def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

rsi = compute_rsi(df_full['close'], 14)
rsi_signal = ((rsi > 40) & (rsi < 70)).astype(float)  # Not overbought/oversold

# MACD
ema12 = df_full['close'].ewm(span=12).mean()
ema26 = df_full['close'].ewm(span=26).mean()
macd = ema12 - ema26
macd_signal_line = macd.ewm(span=9).mean()
macd_signal = (macd > macd_signal_line).astype(float)

# 3. Volatility Indicators
# ATR-based filter
def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

atr = compute_atr(df_full['high'], df_full['low'], df_full['close'], 14)
atr_pct = atr / df_full['close'] * 100
atr_signal = (atr_pct < atr_pct.rolling(50).median()).astype(float)  # Low volatility

# Bollinger Bands
bb_period = 20
bb_std = 2
bb_mid = df_full['close'].rolling(bb_period).mean()
bb_std_val = df_full['close'].rolling(bb_period).std()
bb_upper = bb_mid + bb_std * bb_std_val
bb_lower = bb_mid - bb_std * bb_std_val
bb_signal = ((df_full['close'] > bb_lower) & (df_full['close'] < bb_upper)).astype(float)

print(f"  Indicators computed:")
print(f"    OBV:       {obv_signal.mean()*100:.1f}% in market")
print(f"    VWAP:      {vwap_signal.mean()*100:.1f}% in market")
print(f"    RSI:       {rsi_signal.mean()*100:.1f}% in market")
print(f"    MACD:      {macd_signal.mean()*100:.1f}% in market")
print(f"    ATR:       {atr_signal.mean()*100:.1f}% in market")
print(f"    Bollinger: {bb_signal.mean()*100:.1f}% in market")

# ================================================================
# Grid Search
# ================================================================
print("\n[4/5] Running grid search...")

results = []

# Parameter grids
min_hold_values = [30, 45, 60, 75, 90]
trend_windows = [(50, 150), (50, 200), (100, 200)]
volume_filters = ['none', 'obv', 'vwap']
momentum_filters = ['none', 'rsi', 'macd']
volatility_filters = ['none', 'atr', 'bollinger']

# Test combinations
configs = []

# Base configs with different min_hold and trend
for mh in min_hold_values:
    for tf_fast, tf_slow in trend_windows:
        configs.append({
            'name': f'MH{mh}_T{tf_fast}/{tf_slow}',
            'min_hold': mh,
            'trend_fast': tf_fast,
            'trend_slow': tf_slow,
            'volume': 'none',
            'momentum': 'none',
            'volatility': 'none'
        })

# Add volume filter configs
for vol in ['obv', 'vwap']:
    configs.append({
        'name': f'MH60_T50/200_{vol.upper()}',
        'min_hold': 60,
        'trend_fast': 50,
        'trend_slow': 200,
        'volume': vol,
        'momentum': 'none',
        'volatility': 'none'
    })

# Add momentum filter configs
for mom in ['rsi', 'macd']:
    configs.append({
        'name': f'MH60_T50/200_{mom.upper()}',
        'min_hold': 60,
        'trend_fast': 50,
        'trend_slow': 200,
        'volume': 'none',
        'momentum': mom,
        'volatility': 'none'
    })

# Add volatility filter configs
for vol_filter in ['atr', 'bollinger']:
    configs.append({
        'name': f'MH60_T50/200_{vol_filter.upper()}',
        'min_hold': 60,
        'trend_fast': 50,
        'trend_slow': 200,
        'volume': 'none',
        'momentum': 'none',
        'volatility': vol_filter
    })

# Combined configs
configs.append({
    'name': 'BEST_COMBO_1',
    'min_hold': 60,
    'trend_fast': 50,
    'trend_slow': 200,
    'volume': 'obv',
    'momentum': 'rsi',
    'volatility': 'none'
})

configs.append({
    'name': 'BEST_COMBO_2',
    'min_hold': 75,
    'trend_fast': 50,
    'trend_slow': 200,
    'volume': 'obv',
    'momentum': 'macd',
    'volatility': 'none'
})

configs.append({
    'name': 'BEST_COMBO_3',
    'min_hold': 90,
    'trend_fast': 100,
    'trend_slow': 200,
    'volume': 'vwap',
    'momentum': 'rsi',
    'volatility': 'none'
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
    signal = raw_combined.copy()
    
    # Apply trend filter
    trend_fast = sma(df_full['close'], config['trend_fast'])
    trend_slow = sma(df_full['close'], config['trend_slow'])
    trend_filter = (trend_fast > trend_slow).astype(float)
    signal = signal * trend_filter
    
    # Apply volume filter
    if config['volume'] == 'obv':
        signal = signal * obv_signal
    elif config['volume'] == 'vwap':
        signal = signal * vwap_signal
    
    # Apply momentum filter
    if config['momentum'] == 'rsi':
        signal = signal * rsi_signal
    elif config['momentum'] == 'macd':
        signal = signal * macd_signal
    
    # Apply volatility filter
    if config['volatility'] == 'atr':
        signal = signal * atr_signal
    elif config['volatility'] == 'bollinger':
        signal = signal * bb_signal
    
    # Apply min hold
    signal = apply_min_hold(signal, config['min_hold'])
    
    # Analyze
    changes = signal.diff().fillna(0)
    n_trades = (changes.abs() > 0).sum() // 2  # Round trips
    
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
    
    # Win rate
    winning = sum(1 for r in trade_returns if r > 0)
    total = len(trade_returns)
    win_rate = winning / total * 100 if total > 0 else 0
    
    # ISP coherence
    aligned = pd.DataFrame({'system': signal, 'benchmark': isp_positions_full}).dropna()
    coherence = (aligned['system'] == aligned['benchmark']).sum() / len(aligned) * 100 if len(aligned) > 0 else 0
    
    # Score (target: 15-20 trades, 50-70 days hold)
    trade_score = 1.0 if 15 <= n_trades <= 20 else max(0, 1 - abs(n_trades - 17) / 17)
    hold_score = 1.0 if 50 <= np.mean(hold_periods) <= 70 else max(0, 1 - abs(np.mean(hold_periods) - 60) / 60) if hold_periods else 0
    win_score = win_rate / 100
    
    score = sharpe * 0.3 + trade_score * 0.25 + hold_score * 0.25 + win_score * 0.2
    
    results.append({
        'config': config['name'],
        'n_trades': n_trades,
        'avg_hold': np.mean(hold_periods) if hold_periods else 0,
        'win_rate': win_rate,
        'cagr': cagr * 100,
        'sharpe': sharpe,
        'max_dd': max_dd * 100,
        'coherence': coherence,
        'score': score,
        'trade_returns': trade_returns
    })

# Sort by score
results.sort(key=lambda x: x['score'], reverse=True)

# Print results
print(f"\n  TOP 15 CONFIGURATIONS:")
print(f"  {'─'*100}")
print(f"  {'Config':<30} {'Trades':<10} {'AvgHold':<10} {'WinRate':<10} {'Sharpe':<10} {'MaxDD':<10} {'Score':<10}")
print(f"  {'─'*100}")

for r in results[:15]:
    marker = "⭐" if 15 <= r['n_trades'] <= 20 and 50 <= r['avg_hold'] <= 70 else "  "
    print(f"  {marker}{r['config']:<28} {r['n_trades']:<10} {r['avg_hold']:<10.0f} {r['win_rate']:<10.1f}% {r['sharpe']:<10.2f} {r['max_dd']:<10.1f} {r['score']:<10.3f}")

# Find best that matches ISP behavior
print(f"\n  ISP BEHAVIOR TARGET:")
print(f"  ─────────────────────────────────────────")
print(f"  Target:  15-20 trades, 50-70 days hold, >60% win rate")

best_matches = [r for r in results if 15 <= r['n_trades'] <= 20 and 50 <= r['avg_hold'] <= 70]

if best_matches:
    print(f"\n  ✅ MATCHING CONFIGURATIONS FOUND:")
    for r in best_matches[:5]:
        print(f"    • {r['config']}: {r['n_trades']} trades, {r['avg_hold']:.0f} days hold, {r['win_rate']:.0f}% win, Sharpe {r['sharpe']:.2f}")
else:
    print(f"\n  ⚠️ No exact match, showing closest:")
    for r in results[:3]:
        print(f"    • {r['config']}: {r['n_trades']} trades, {r['avg_hold']:.0f} days hold, {r['win_rate']:.0f}% win")
