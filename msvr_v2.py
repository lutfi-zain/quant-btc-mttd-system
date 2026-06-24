#!/usr/bin/env python3
"""
MSVR v2 — High-Quality Signal Generation
==========================================

Problem: Enhanced MSVR has 48 trades (4x Ichimoku) but metrics only +5%
Solution: Focus on QUALITY not QUANTITY

Principles to add:
1. Volume Confirmation (Family 8: Volume) — confirm moves with volume
2. Regime Detection (Family 5: Fractal/Hurst) — only trade in trending regime
3. Volatility Filter (Family 6: GARCH-like) — avoid high volatility
4. Adaptive Threshold — dynamic entry based on market conditions
"""

import numpy as np
import pandas as pd
import sys
import os
import json
import importlib.util
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/ubuntu/projects/quant-technical-indicator-bank')
from indicators_helper import *

print("=" * 70)
print("MSVR v2 — High-Quality Signal Generation")
print("=" * 70)

# ================================================================
# Load Data
# ================================================================
print("\n[1/4] Loading data...")

with open('data/btc_daily.json') as f:
    btc_data = json.load(f)

df_full = pd.DataFrame(btc_data['aligned_data'])
df_full['time'] = pd.to_datetime(df_full['time'])
df_full = df_full.set_index('time')
df_full = df_full[df_full.index >= '2018-01-01']

print(f"  Data: {len(df_full)} bars")

# ================================================================
# Layer 1: MSVR Base (Family 1: Smoothing)
# ================================================================
print("\n[2/4] Computing indicators...")

spec = importlib.util.spec_from_file_location('msvr', 
    '/home/ubuntu/projects/quant-technical-indicator-bank/perpetual/median_standard_deviation_viresearch.py')
msvr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msvr_module)

msvr_result = msvr_module.median_standard_deviation_viresearch(df_full)
df_full['msvr_signal'] = msvr_result['vii']
df_full['msvr_direction'] = (df_full['msvr_signal'] > 0).astype(float)

# ================================================================
# Layer 2: Cycle Phase (Family 4: Spectral)
# ================================================================
def compute_cycle_phase(df, lookback=40):
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
df_full['cycle_signal'] = -np.cos(phase)
df_full['cycle_direction'] = (df_full['cycle_signal'] > 0).astype(float)

# ================================================================
# Layer 3: Ehler SuperSmoother (Family 2: Filtering)
# ================================================================
def ehler_supersmoother(series, length=7):
    a1 = np.exp(-1.414 * np.pi / length)
    b1 = 2 * a1 * np.cos(np.radians(1.414 * 180.0 / length))
    c2 = b1
    c3 = -a1 * a1
    c1 = 1 - c2 - c3

    vals = series.ffill().fillna(0).values
    filt = np.zeros(len(vals))
    filt[0] = vals[0]
    if len(vals) > 1:
        filt[1] = vals[1]
    for i in range(2, len(vals)):
        filt[i] = c1 * (vals[i] + vals[i-1]) / 2 + c2 * filt[i-1] + c3 * filt[i-2]
    return pd.Series(filt, index=series.index)

df_full['msvr_smooth'] = ehler_supersmoother(df_full['msvr_signal'], length=7)
df_full['smooth_direction'] = (df_full['msvr_smooth'] > 0).astype(float)

# ================================================================
# Layer 4: Volume Confirmation (Family 8: Volume)
# ================================================================
# On-Balance Volume
obv = pd.Series(0.0, index=df_full.index)
for i in range(1, len(df_full)):
    if df_full['close'].iloc[i] > df_full['close'].iloc[i-1]:
        obv.iloc[i] = obv.iloc[i-1] + df_full['volume'].iloc[i]
    elif df_full['close'].iloc[i] < df_full['close'].iloc[i-1]:
        obv.iloc[i] = obv.iloc[i-1] - df_full['volume'].iloc[i]
    else:
        obv.iloc[i] = obv.iloc[i-1]

obv_smooth = ehler_supersmoother(obv, length=20)
df_full['obv_direction'] = (obv_smooth > obv_smooth.shift(1)).astype(float)

# Volume spike detection
vol_ma = df_full['volume'].rolling(20).mean()
df_full['volume_spike'] = (df_full['volume'] > vol_ma * 1.5).astype(float)

# ================================================================
# Layer 5: Regime Detection (Family 5: Fractal/Hurst)
# ================================================================
def hurst_exponent(series, window=100):
    """Simplified Hurst exponent."""
    hurst = pd.Series(0.5, index=series.index)
    
    for i in range(window, len(series)):
        window_data = series.iloc[i-window:i].values
        
        mean = np.mean(window_data)
        deviations = window_data - mean
        cumulative = np.cumsum(deviations)
        R = np.max(cumulative) - np.min(cumulative)
        S = np.std(window_data)
        
        if S > 0 and R > 0:
            hurst.iloc[i] = np.log(R/S) / np.log(window)
    
    return hurst

df_full['hurst'] = hurst_exponent(df_full['close'], window=100)
df_full['trending_regime'] = (df_full['hurst'] > 0.55).astype(float)  # Trending market

# ================================================================
# Layer 6: Volatility Filter (Family 6: GARCH-like)
# ================================================================
returns = df_full['close'].pct_change()
df_full['volatility'] = returns.rolling(20).std()
df_full['vol_median'] = df_full['volatility'].rolling(100).median()
df_full['low_volatility'] = (df_full['volatility'] < df_full['vol_median'] * 1.2).astype(float)

# ================================================================
# Layer 7: Efficiency Ratio (Family 5: Fractal)
# ================================================================
def efficiency_ratio(series, period=14):
    change = series.diff().abs()
    volatility = change.rolling(period).sum()
    direction = series.diff(period).abs()
    return direction / volatility

df_full['er'] = efficiency_ratio(df_full['close'], period=14)
df_full['er_gate'] = (df_full['er'] > 0.25).astype(float)

# ================================================================
# Layer 8: Shannon Entropy (Family 7: Entropy)
# ================================================================
def shannon_entropy(series, window=15, bins=6):
    def calc_shannon(x):
        if len(x) < window:
            return np.nan
        counts, _ = np.histogram(x, bins=bins)
        probs = counts / len(x)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    
    returns = series.pct_change().fillna(0)
    return returns.rolling(window=window).apply(calc_shannon, raw=True)

df_full['entropy'] = shannon_entropy(df_full['close'], window=15, bins=6)
df_full['entropy_gate'] = (df_full['entropy'] < 2.5).astype(float)

# ================================================================
# COMPOSITE SIGNAL — Quality over Quantity
# ================================================================
print("\n[3/4] Generating composite signal...")

# Core signals (must agree)
core_signal = df_full['msvr_direction'] * df_full['cycle_direction'] * df_full['smooth_direction']

# Confirmation signals (at least 2 of 3 must agree)
confirm_signal = df_full['obv_direction'] + df_full['er_gate'] + df_full['entropy_gate']
confirm_pass = (confirm_signal >= 2).astype(float)

# Regime filter (must be trending)
regime_pass = df_full['trending_regime']

# Volatility filter (prefer low volatility)
vol_pass = df_full['low_volatility']

# Final signal: Core + Confirm + Regime
df_full['final_signal'] = core_signal * confirm_pass * regime_pass

# Apply min hold (45 days)
def apply_min_hold(signal, min_hold=45):
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

df_full['position'] = apply_min_hold(df_full['final_signal'], min_hold=45)

# ================================================================
# Compute Metrics
# ================================================================
print("\n[4/4] Computing metrics...")

def compute_metrics(positions, prices):
    returns = prices.pct_change()
    strategy_returns = returns * positions.shift(1)
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) == 0:
        return {'cagr': 0, 'sharpe': 0, 'max_dd': 0, 'n_trades': 0, 'win_rate': 0}

    equity = (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25

    cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    
    changes = positions.diff().fillna(0)
    n_trades = (changes.abs() > 0).sum() // 2
    
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
        'max_dd': round(max_dd * 100, 2),
        'n_trades': n_trades,
        'win_rate': round(win_rate, 1)
    }

msvr_v2_metrics = compute_metrics(df_full['position'], df_full['close'])

# Ichimoku metrics
from ichimoku_quant import generate_ichimoku_features, generate_ichimoku_signals
df_ichimoku = generate_ichimoku_features(df_full)
df_ichimoku = generate_ichimoku_signals(df_ichimoku)

ich_returns = df_full['close'].pct_change() * df_ichimoku['Pos'].shift(1)
ich_returns = ich_returns.dropna()
ich_equity = (1 + ich_returns).cumprod()
ich_years = len(ich_returns) / 365.25
ich_cagr = (ich_equity.iloc[-1]) ** (1/ich_years) - 1 if ich_years > 0 else 0
ich_sharpe = ich_returns.mean() / ich_returns.std() * np.sqrt(365) if ich_returns.std() > 0 else 0
ich_peak = ich_equity.cummax()
ich_maxdd = ((ich_equity - ich_peak) / ich_peak).min()

ich_changes = df_ichimoku['Pos'].diff().fillna(0)
ich_trades = (ich_changes.abs() > 0).sum() // 2
in_pos = False
ich_trade_rets = []
for i, (date, pos) in enumerate(df_ichimoku['Pos'].items()):
    if pos == 1.0 and not in_pos:
        in_pos = True
        entry = df_full.loc[date, 'close']
    elif pos == 0.0 and in_pos:
        in_pos = False
        exit_p = df_full.loc[date, 'close']
        ich_trade_rets.append((exit_p - entry) / entry)
ich_winning = sum(1 for r in ich_trade_rets if r > 0)
ich_winrate = ich_winning / len(ich_trade_rets) * 100 if ich_trade_rets else 0

print(f"\n{'='*70}")
print("COMPARISON: MSVR v2 vs Enhanced MSVR vs Ichimoku")
print(f"{'='*70}")

print(f"\n{'Metric':<15} {'MSVR v2':<15} {'Enhanced':<15} {'Ichimoku':<15} {'Winner':<15}")
print(f"{'-'*75}")
print(f"{'Sharpe':<15} {msvr_v2_metrics['sharpe']:<15} {'1.35':<15} {ich_sharpe:<15.2f} ", end="")
if msvr_v2_metrics['sharpe'] > ich_sharpe and msvr_v2_metrics['sharpe'] > 1.35:
    print("MSVR v2 ⭐")
elif ich_sharpe > 1.35:
    print("Ichimoku ⭐")
else:
    print("Enhanced ⭐")

print(f"{'CAGR':<15} {msvr_v2_metrics['cagr']:<15} {'58.5%':<15} {ich_cagr*100:<15.1f}% ", end="")
if msvr_v2_metrics['cagr'] > ich_cagr*100 and msvr_v2_metrics['cagr'] > 58.5:
    print("MSVR v2 ⭐")
elif ich_cagr*100 > 58.5:
    print("Ichimoku ⭐")
else:
    print("Enhanced ⭐")

print(f"{'Win Rate':<15} {msvr_v2_metrics['win_rate']:<15} {'35.4%':<15} {ich_winrate:<15.1f}% ", end="")
if msvr_v2_metrics['win_rate'] > ich_winrate and msvr_v2_metrics['win_rate'] > 35.4:
    print("MSVR v2 ⭐")
elif ich_winrate > 35.4:
    print("Ichimoku ⭐")
else:
    print("Enhanced ⭐")

print(f"{'Trades':<15} {msvr_v2_metrics['n_trades']:<15} {'48':<15} {ich_trades:<15} ", end="")
if msvr_v2_metrics['n_trades'] < ich_trades and msvr_v2_metrics['n_trades'] < 48:
    print("MSVR v2 ⭐")
elif ich_trades < 48:
    print("Ichimoku ⭐")
else:
    print("Enhanced ⭐")
