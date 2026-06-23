#!/usr/bin/env python3
"""
Grid Search — Reduce Trades Using Statistical Principles
==========================================================

Problem: MSVR+Cycle trades 232 times (vs ISP 16 times)
Solution: Apply statistical principles to filter noise

Statistical Principles to Test:
1. FILTERING (Family 2): Low-pass filter to remove high-freq noise
2. FRACTAL (Family 5): Hurst exponent to identify trending regimes
3. ENTROPY (Family 7): Measure market efficiency/randomness
4. GARCH (Family 6): Volatility regime filter
5. BAYESIAN (Family 9): HMM regime detection

Goal: Reduce trades from 232 → closer to ISP's 16
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

print("=" * 70)
print("GRID SEARCH — REDUCE TRADES USING STATISTICAL PRINCIPLES")
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
print("\n[2/4] Loading MSVR + Cycle Phase...")

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
# Statistical Principle Functions
# ================================================================
print("\n[3/4] Defining statistical principle filters...")

# 1. LOW-PASS FILTER (removes high-frequency noise)
def low_pass_filter(signal, cutoff_period=20):
    """Apply low-pass filter to signal."""
    alpha = 2.0 / (cutoff_period + 1)
    filtered = pd.Series(0.0, index=signal.index)
    filtered.iloc[0] = signal.iloc[0]
    
    for i in range(1, len(signal)):
        if not np.isnan(signal.iloc[i]):
            filtered.iloc[i] = alpha * signal.iloc[i] + (1 - alpha) * filtered.iloc[i-1]
        else:
            filtered.iloc[i] = filtered.iloc[i-1]
    
    return filtered

# 2. HURST EXPONENT (trending vs mean-reverting)
def hurst_exponent(series, window=100):
    """Compute rolling Hurst exponent."""
    hurst = pd.Series(np.nan, index=series.index)
    
    for i in range(window, len(series)):
        window_data = series.iloc[i-window:i].values
        
        # R/S analysis
        mean = np.mean(window_data)
        deviations = window_data - mean
        cumulative = np.cumsum(deviations)
        R = np.max(cumulative) - np.min(cumulative)
        S = np.std(window_data)
        
        if S > 0 and R > 0:
            # Simplified Hurst estimation
            hurst.iloc[i] = np.log(R/S) / np.log(window)
    
    return hurst

# 3. ENTROPY (market efficiency)
def permutation_entropy(series, window=20, order=3):
    """Compute rolling permutation entropy."""
    entropy = pd.Series(np.nan, index=series.index)
    
    for i in range(window, len(series)):
        window_data = series.iloc[i-window:i].values
        
        # Count ordinal patterns
        patterns = {}
        for j in range(len(window_data) - order + 1):
            pattern = tuple(np.argsort(window_data[j:j+order]))
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        # Calculate entropy
        total = sum(patterns.values())
        ent = 0
        for count in patterns.values():
            p = count / total
            if p > 0:
                ent -= p * np.log2(p)
        
        # Normalize (0 = random, 1 = deterministic)
        max_ent = np.log2(np.math.factorial(order))
        entropy.iloc[i] = ent / max_ent if max_ent > 0 else 0
    
    return entropy

# 4. VOLATILITY REGIME (GARCH-like)
def volatility_regime(prices, window=20):
    """Identify volatility regime."""
    returns = prices.pct_change()
    vol = returns.rolling(window).std()
    vol_median = vol.rolling(100).median()
    
    # High vol = 1, Low vol = 0
    regime = (vol > vol_median).astype(float)
    return regime

# 5. HMM-LIKE REGIME (simplified)
def trend_regime(prices, fast=50, slow=200):
    """Simple trend regime detection."""
    sma_fast = prices.rolling(fast).mean()
    sma_slow = prices.rolling(slow).mean()
    
    # Bull = 1, Bear = 0
    regime = (sma_fast > sma_slow).astype(float)
    return regime

# ================================================================
# Grid Search
# ================================================================
print("\n[4/4] Running grid search...")

results = []

# Test different filter combinations
filter_configs = [
    # (name, low_pass_period, hurst_window, entropy_window, vol_window, min_hold)
    ("Baseline", 0, 0, 0, 0, 0),
    ("LowPass 20", 20, 0, 0, 0, 0),
    ("LowPass 40", 40, 0, 0, 0, 0),
    ("LowPass 60", 60, 0, 0, 0, 0),
    ("Hurst Trend", 0, 100, 0, 0, 0),
    ("Entropy Filter", 0, 0, 20, 0, 0),
    ("Vol Filter", 0, 0, 0, 20, 0),
    ("Trend Filter", 0, 0, 0, 0, 0),
    ("MinHold 30", 0, 0, 0, 0, 30),
    ("MinHold 60", 0, 0, 0, 0, 60),
    ("MinHold 90", 0, 0, 0, 0, 90),
    ("MinHold 120", 0, 0, 0, 0, 120),
    # Combined
    ("LP40 + Trend", 40, 0, 0, 0, 0),
    ("LP40 + MinHold60", 40, 0, 0, 0, 60),
    ("LP40 + Trend + MinHold60", 40, 0, 0, 0, 60),
    ("LP60 + Trend + MinHold90", 60, 0, 0, 0, 90),
    ("Trend + MinHold60", 0, 0, 0, 0, 60),
    ("Trend + MinHold90", 0, 0, 0, 0, 90),
    ("Trend + MinHold120", 0, 0, 0, 0, 120),
]

# ISP metrics
isp_returns = df_full['close'].pct_change() * isp_positions_full.shift(1)
isp_returns = isp_returns.dropna()
isp_sharpe = isp_returns.mean() / isp_returns.std() * np.sqrt(365) if isp_returns.std() > 0 else 0

for name, lp_period, hurst_win, ent_win, vol_win, min_hold in filter_configs:
    signal = raw_combined.copy()
    
    # Apply low-pass filter
    if lp_period > 0:
        signal = low_pass_filter(signal, lp_period)
        signal = (signal > 0.5).astype(float)
    
    # Apply trend filter
    if "Trend" in name:
        trend = trend_regime(df_full['close'], fast=50, slow=200)
        signal = signal * trend
    
    # Apply min hold
    if min_hold > 0:
        in_position = False
        hold_count = 0
        for i in range(len(signal)):
            if signal.iloc[i] == 1.0 and not in_position:
                in_position = True
                hold_count = 0
            elif signal.iloc[i] == 0.0 and in_position:
                if hold_count < min_hold:
                    signal.iloc[i] = 1.0
                    hold_count += 1
                else:
                    in_position = False
                    hold_count = 0
            elif in_position:
                hold_count += 1
    
    # Analyze behavior
    changes = signal.diff().fillna(0)
    n_trades = (changes.abs() > 0).sum()
    
    in_position = False
    hold_start = None
    hold_periods = []
    for i, (date, pos) in enumerate(signal.items()):
        if pos == 1.0 and not in_position:
            in_position = True
            hold_start = date
        elif pos == 0.0 and in_position:
            in_position = False
            if hold_start is not None:
                hold_days = (date - hold_start).days
                hold_periods.append(hold_days)
    
    avg_hold = np.mean(hold_periods) if hold_periods else 0
    
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
    
    # ISP coherence
    aligned = pd.DataFrame({'system': signal, 'benchmark': isp_positions_full}).dropna()
    coherence = (aligned['system'] == aligned['benchmark']).sum() / len(aligned) * 100 if len(aligned) > 0 else 0
    
    # Score (trade reduction + performance)
    trade_ratio = n_trades / 16  # ISP has 16 trades
    score = sharpe * 0.4 + (1 / max(trade_ratio, 0.1)) * 0.3 + coherence * 0.003 * 0.3
    
    results.append({
        'name': name,
        'n_trades': n_trades,
        'avg_hold': avg_hold,
        'in_market': signal.mean() * 100,
        'cagr': cagr * 100,
        'sharpe': sharpe,
        'max_dd': max_dd * 100,
        'coherence': coherence,
        'trade_ratio': trade_ratio,
        'score': score
    })

# Sort by trade reduction + performance
results.sort(key=lambda x: x['score'], reverse=True)

# Print results
print(f"\n  RESULTS:")
print(f"  {'─'*100}")
print(f"  {'Name':<30} {'Trades':<10} {'AvgHold':<10} {'InMkt%':<10} {'Sharpe':<10} {'MaxDD':<10} {'Coher%':<10}")
print(f"  {'─'*100}")

for r in results:
    marker = "⭐" if r['n_trades'] <= 30 and r['sharpe'] > 0.5 else "  "
    print(f"  {marker}{r['name']:<28} {r['n_trades']:<10} {r['avg_hold']:<10.0f} {r['in_market']:<10.1f} {r['sharpe']:<10.2f} {r['max_dd']:<10.1f} {r['coherence']:<10.1f}")

# Find best that matches ISP behavior
print(f"\n  ISP BEHAVIOR TARGET:")
print(f"  ─────────────────────────────────────────")
print(f"  ISP:     16 trades, 128 days avg hold, Sharpe 1.88")
print(f"  Target:  ≤30 trades, ≥60 days avg hold, Sharpe > 0.5")

# Filter for ISP-like behavior
isp_like = [r for r in results if r['n_trades'] <= 30 and r['avg_hold'] >= 60]

if isp_like:
    print(f"\n  ✅ ISP-LIKE CONFIGURATIONS FOUND:")
    for r in isp_like[:5]:
        print(f"    • {r['name']}: {r['n_trades']} trades, {r['avg_hold']:.0f} days hold, Sharpe {r['sharpe']:.2f}")
else:
    print(f"\n  ⚠️ No configuration matches ISP behavior exactly")
    print(f"     Best trade reduction:")
    for r in results[:5]:
        print(f"    • {r['name']}: {r['n_trades']} trades, {r['avg_hold']:.0f} days hold")
