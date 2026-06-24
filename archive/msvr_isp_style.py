#!/usr/bin/env python3
"""
MSVR + Cycle Phase — ISP-Style Implementation
===============================================

User feedback: "Too noisy, not applicable"
ISP behavior: 16 trades in 8 years, 128 days avg hold

Changes:
1. min_hold = 60 days (force long holds)
2. Trend filter = 200 SMA (only trade with trend)
3. Signal smoothing = 20-day EMA (reduce noise)
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
print("MSVR + CYCLE PHASE — ISP-STYLE IMPLEMENTATION")
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

# ================================================================
# ISP-Style Filters
# ================================================================
print("\n[3/5] Applying ISP-style filters...")

# 1. Basic combined signal
msvr_binary = (msvr_signal > 0).astype(float)
cycle_binary = (cycle_signal > 0).astype(float)
raw_combined = msvr_binary * cycle_binary

# 2. Trend filter (200 SMA)
sma200 = sma(df_full['close'], 200)
trend_filter = (df_full['close'] > sma200).astype(float)

# 3. Apply trend filter
filtered = raw_combined * trend_filter

# 4. Signal smoothing (20-day EMA)
smoothed = filtered.ewm(span=20).mean()
smoothed_signal = (smoothed > 0.5).astype(float)

# 5. Min hold filter (60 days)
def apply_min_hold(positions, min_hold=60):
    """Force minimum holding period."""
    result = positions.copy()
    in_position = False
    hold_count = 0
    last_entry = 0
    
    for i in range(len(result)):
        if result.iloc[i] == 1.0 and not in_position:
            in_position = True
            hold_count = 0
            last_entry = i
        elif result.iloc[i] == 0.0 and in_position:
            if hold_count < min_hold:
                result.iloc[i] = 1.0  # Force hold
                hold_count += 1
            else:
                in_position = False
                hold_count = 0
        elif in_position:
            hold_count += 1
    
    return result

final_signal = apply_min_hold(smoothed_signal, min_hold=60)

# ================================================================
# Analyze ISP-Style Behavior
# ================================================================
print("\n[4/5] Analyzing ISP-style behavior...")

def analyze_behavior(signal, name):
    """Analyze trading behavior."""
    changes = signal.diff().fillna(0)
    n_trades = (changes.abs() > 0).sum()
    
    in_position = False
    hold_start = None
    hold_periods = []
    trade_dates = []
    
    for i, (date, pos) in enumerate(signal.items()):
        if pos == 1.0 and not in_position:
            in_position = True
            hold_start = date
            trade_dates.append(('BUY', date))
        elif pos == 0.0 and in_position:
            in_position = False
            if hold_start is not None:
                hold_days = (date - hold_start).days
                hold_periods.append(hold_days)
            trade_dates.append(('SELL', date))
    
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
    
    return {
        'name': name,
        'n_trades': n_trades,
        'avg_hold': np.mean(hold_periods) if hold_periods else 0,
        'median_hold': np.median(hold_periods) if hold_periods else 0,
        'in_market': signal.mean() * 100,
        'cagr': cagr * 100,
        'sharpe': sharpe,
        'max_dd': max_dd * 100,
        'trade_dates': trade_dates
    }

# Analyze all versions
isp_behavior = analyze_behavior(isp_positions_full, 'ISP')
raw_behavior = analyze_behavior(raw_combined, 'Raw Combined')
filtered_behavior = analyze_behavior(filtered, 'Filtered')
smoothed_behavior = analyze_behavior(smoothed_signal, 'Smoothed')
final_behavior = analyze_behavior(final_signal, 'Final (ISP-Style)')

# Print comparison
print(f"\n  BEHAVIOR COMPARISON:")
print(f"  {'─'*85}")
print(f"  {'Version':<20} {'Trades':<10} {'Avg Hold':<12} {'In-Market':<12} {'CAGR':<10} {'Sharpe':<10}")
print(f"  {'─'*85}")
print(f"  {'ISP':<20} {isp_behavior['n_trades']:<10} {isp_behavior['avg_hold']:<12.0f} {isp_behavior['in_market']:<12.1f}% {isp_behavior['cagr']:<10.1f}% {isp_behavior['sharpe']:<10.2f}")
print(f"  {'Raw Combined':<20} {raw_behavior['n_trades']:<10} {raw_behavior['avg_hold']:<12.0f} {raw_behavior['in_market']:<12.1f}% {raw_behavior['cagr']:<10.1f}% {raw_behavior['sharpe']:<10.2f}")
print(f"  {'Filtered':<20} {filtered_behavior['n_trades']:<10} {filtered_behavior['avg_hold']:<12.0f} {filtered_behavior['in_market']:<12.1f}% {filtered_behavior['cagr']:<10.1f}% {filtered_behavior['sharpe']:<10.2f}")
print(f"  {'Smoothed':<20} {smoothed_behavior['n_trades']:<10} {smoothed_behavior['avg_hold']:<12.0f} {smoothed_behavior['in_market']:<12.1f}% {smoothed_behavior['cagr']:<10.1f}% {smoothed_behavior['sharpe']:<10.2f}")
print(f"  {'Final (ISP-Style)':<20} {final_behavior['n_trades']:<10} {final_behavior['avg_hold']:<12.0f} {final_behavior['in_market']:<12.1f}% {final_behavior['cagr']:<10.1f}% {final_behavior['sharpe']:<10.2f}")

# Print trade history for final
print(f"\n  FINAL (ISP-STYLE) TRADE HISTORY:")
print(f"  {'Date':<15} {'Action':<8} {'BTC Price':<15}")
print(f"  {'─'*38}")
for action, date in final_behavior['trade_dates']:
    if date in df_full.index:
        price = df_full.loc[date, 'close']
        print(f"  {date.strftime('%Y-%m-%d'):<15} {action:<8} ${price:,.0f}")

# ================================================================
# Validation
# ================================================================
print("\n[5/5] Holdout validation...")

# Split
final_train = final_signal[df_train.index]
final_holdout = final_signal[df_holdout.index]

# Metrics
def calc_metrics(signal, prices):
    returns = prices.pct_change()
    strat_returns = returns * signal.shift(1)
    strat_returns = strat_returns.dropna()
    equity = (1 + strat_returns).cumprod()
    years = len(strat_returns) / 365.25
    cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(365) if strat_returns.std() > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    return {'cagr': cagr * 100, 'sharpe': sharpe, 'max_dd': max_dd * 100}

train_metrics = calc_metrics(final_train, df_train['close'])
holdout_metrics = calc_metrics(final_holdout, df_holdout['close'])

# ISP metrics
isp_train = calc_metrics(isp_positions_full[df_train.index], df_train['close'])
isp_holdout = calc_metrics(isp_positions_full[df_holdout.index], df_holdout['close'])

print(f"\n  HOLDOUT VALIDATION:")
print(f"  {'─'*60}")
print(f"  {'Metric':<20} {'ISP Train':<15} {'ISP Hold':<15} {'MTTD Train':<15} {'MTTD Hold':<15}")
print(f"  {'─'*60}")
print(f"  {'Sharpe':<20} {isp_train['sharpe']:<15.2f} {isp_holdout['sharpe']:<15.2f} {train_metrics['sharpe']:<15.2f} {holdout_metrics['sharpe']:<15.2f}")
print(f"  {'CAGR':<20} {isp_train['cagr']:<15.1f}% {isp_holdout['cagr']:<15.1f}% {train_metrics['cagr']:<15.1f}% {holdout_metrics['cagr']:<15.1f}%")
print(f"  {'MaxDD':<20} {isp_train['max_dd']:<15.1f}% {isp_holdout['max_dd']:<15.1f}% {train_metrics['max_dd']:<15.1f}% {holdout_metrics['max_dd']:<15.1f}%")

# Degradation
deg_isp = (isp_holdout['sharpe'] - isp_train['sharpe']) / abs(isp_train['sharpe']) * 100 if isp_train['sharpe'] != 0 else 0
deg_mttd = (holdout_metrics['sharpe'] - train_metrics['sharpe']) / abs(train_metrics['sharpe']) * 100 if train_metrics['sharpe'] != 0 else 0

print(f"\n  Degradation:")
print(f"    ISP:    {deg_isp:+.1f}%")
print(f"    MTTD:   {deg_mttd:+.1f}%")

# Final verdict
print(f"\n  VERDICT:")
if final_behavior['n_trades'] <= 30 and final_behavior['avg_hold'] >= 60:
    print(f"  ✅ ISP-STYLE BEHAVIOR ACHIEVED!")
    print(f"     • Trades: {final_behavior['n_trades']} (target: ≤30)")
    print(f"     • Avg Hold: {final_behavior['avg_hold']:.0f} days (target: ≥60)")
else:
    print(f"  ⚠️ Still too active — need more filtering")

print(f"\n  COMPARISON SUMMARY:")
print(f"  ─────────────────────────────────────────")
print(f"  ISP:     {isp_behavior['n_trades']} trades, {isp_behavior['avg_hold']:.0f} days avg hold")
print(f"  MTTD:    {final_behavior['n_trades']} trades, {final_behavior['avg_hold']:.0f} days avg hold")
print(f"  Target:  ≤30 trades, ≥60 days avg hold")
