#!/usr/bin/env python3
"""
MSVR v6 — Selective Entry + Extended Hold
===========================================

Key Insight from v5:
- MSVR v5: 11 trades, 72.7% win, but ONLY 8% time in market!
- Ichimoku: 11 trades, 63.6% win, 45% time in market

Problem: MSVR v5 exits TOO FAST → low CAGR

Solution:
1. Keep selective entry (11 trades, 72.7% win)
2. EXTEND max_hold (90 days instead of 60)
3. ONLY exit on entropy SPIKE (not normal entropy)
4. Allow longer holds to capture trends

This should increase time in market while maintaining quality.
"""

import numpy as np
import pandas as pd
import json
import importlib.util
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/ubuntu/projects/quant-technical-indicator-bank')
from indicators_helper import *

print("=" * 70)
print("MSVR v6 — Selective Entry + Extended Hold")
print("=" * 70)

# ================================================================
# Load Data
# ================================================================
print("\n[1/6] Loading data...")

with open('data/btc_daily.json') as f:
    btc_data = json.load(f)

df = pd.DataFrame(btc_data['aligned_data'])
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = df[df.index >= '2018-01-01']

print(f"  Data: {len(df)} bars ({df.index[0]} to {df.index[-1]})")

# ================================================================
# Layer 1: MSVR Base (Family 1: Smoothing)
# ================================================================
print("\n[2/6] Computing indicators...")

spec = importlib.util.spec_from_file_location('msvr', 
    '/home/ubuntu/projects/quant-technical-indicator-bank/perpetual/median_standard_deviation_viresearch.py')
msvr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msvr_module)

msvr_result = msvr_module.median_standard_deviation_viresearch(df)
df['msvr_signal'] = msvr_result['vii']
df['msvr_direction'] = (df['msvr_signal'] > 0).astype(float)

# ================================================================
# Layer 2: SuperSmoother (Family 2: Filtering)
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

df['msvr_smooth'] = ehler_supersmoother(df['msvr_signal'], length=7)
df['smooth_direction'] = (df['msvr_smooth'] > 0).astype(float)

# ================================================================
# Layer 3: LinearReg (Family 3: Regression)
# ================================================================
from indicators.linear_reg_trend import linear_reg_trend
lr_result = linear_reg_trend(df, length=50)
df['lr_direction'] = lr_result['direction'].clip(-1, 1)
df['lr_direction'] = (df['lr_direction'] > 0).astype(float)

# ================================================================
# Layer 4: Cycle Phase (Family 4: Spectral)
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

phase = compute_cycle_phase(df, lookback=40)
df['cycle_signal'] = -np.cos(phase)
df['cycle_direction'] = (df['cycle_signal'] > 0).astype(float)

# ================================================================
# Layer 5: Efficiency Ratio (Family 5: Fractal)
# ================================================================
def efficiency_ratio(series, period=14):
    change = series.diff().abs()
    volatility = change.rolling(period).sum()
    direction = series.diff(period).abs()
    return direction / volatility

df['er'] = efficiency_ratio(df['close'], period=14)
df['er_gate'] = (df['er'] > 0.30).astype(float)

# ================================================================
# Layer 6: Volatility Cluster (Family 6: GARCH)
# ================================================================
from indicators.volatility_cluster import volatility_cluster
vol_result = volatility_cluster(df, window=20, threshold=1.2)
df['vol_direction'] = vol_result['direction'].clip(-1, 1)
df['vol_direction'] = (df['vol_direction'] > 0).astype(float)

# ================================================================
# Layer 7: Shannon Entropy (Family 7: Entropy) — ENTRY + EXIT
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

df['entropy'] = shannon_entropy(df['close'], window=15, bins=6)

# Entry: Low entropy (structured market)
df['entropy_entry'] = (df['entropy'] < 2.3).astype(float)

# EXIT: EXTREME Entropy SPIKE (market becomes VERY random)
df['entropy_ma'] = df['entropy'].rolling(30).mean()
df['entropy_std'] = df['entropy'].rolling(30).std()
df['entropy_extreme'] = ((df['entropy'] > df['entropy_ma'] + 2 * df['entropy_std']) & 
                         df['entropy'].notna() & 
                         df['entropy_ma'].notna()).astype(float)

# ================================================================
# Layer 8: Volume Confirm (Family 8: Volume)
# ================================================================
from indicators.volume_confirm import volume_confirm
vol_confirm_result = volume_confirm(df, obv_short=10, obv_long=30, spike_mult=1.5)
df['volume_direction'] = vol_confirm_result['direction'].clip(-1, 1)
df['volume_direction'] = (df['volume_direction'] > 0).astype(float)

# ================================================================
# Layer 9: HMM Regime (Family 9: Bayesian)
# ================================================================
from indicators.hmm_regime import hmm_regime
hmm_result = hmm_regime(df, n_states=3, window=100)
df['regime_direction'] = hmm_result['direction'].clip(-1, 1)
df['regime_direction'] = (df['regime_direction'] > 0).astype(float)

# ================================================================
# COMPOSITE SIGNAL — SELECTIVE ENTRY
# ================================================================
print("\n[3/6] Generating selective entry signal...")

# Core: All 4 must agree (strict)
core_signal = (
    df['msvr_direction'] * 
    df['smooth_direction'] * 
    df['lr_direction'] * 
    df['cycle_direction']
)

# Gates: ALL 5 must pass (very strict!)
gate_signal = (
    df['er_gate'] * 
    df['vol_direction'] * 
    df['entropy_entry'] * 
    df['volume_direction'] * 
    df['regime_direction']
)

# Entry: Core × Gates (ALL must pass)
df['entry_signal'] = core_signal * gate_signal

# ================================================================
# EXIT SIGNAL — EXTREME Entropy Spike Only
# ================================================================
print("\n[4/6] Computing exit signal (Extreme Entropy Spike)...")

# Exit ONLY on EXTREME entropy spike (not normal)
df['exit_signal'] = df['entropy_extreme']

# ================================================================
# POSITION — Selective Entry + Extended Hold
# ================================================================
print("\n[5/6] Applying selective entry + extended hold...")

def apply_extended_hold(entry_signal, exit_signal, min_hold=25, max_hold=90):
    """
    Selective entry + extended hold.
    - Entry: ALL signals must agree
    - Exit: EXTREME entropy spike only
    - Min hold: 25 days
    - Max hold: 90 days (extended!)
    """
    result = pd.Series(0.0, index=entry_signal.index)
    in_position = False
    hold_count = 0
    
    for i in range(len(result)):
        if entry_signal.iloc[i] == 1.0 and not in_position:
            # Entry
            in_position = True
            hold_count = 0
            result.iloc[i] = 1.0
        elif in_position:
            hold_count += 1
            
            if hold_count >= min_hold and exit_signal.iloc[i] == 1.0:
                # Exit: min_hold satisfied + EXTREME entropy spike
                in_position = False
                hold_count = 0
                result.iloc[i] = 0.0
            elif hold_count >= max_hold:
                # Exit: max_hold reached
                in_position = False
                hold_count = 0
                result.iloc[i] = 0.0
            else:
                result.iloc[i] = 1.0
        else:
            result.iloc[i] = 0.0
    
    return result

# Test different max_hold values
max_hold_values = [60, 75, 90, 120]
results = []

for mh in max_hold_values:
    df['position'] = apply_extended_hold(
        df['entry_signal'], 
        df['exit_signal'], 
        min_hold=25, 
        max_hold=mh
    )
    
    # Compute metrics
    returns = df['close'].pct_change() * df['position'].shift(1)
    returns = returns.dropna()
    
    if len(returns) == 0:
        continue
    
    equity = (1 + returns).cumprod()
    years = len(returns) / 365.25
    
    cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    
    # Trades and win rate
    changes = df['position'].diff().fillna(0)
    n_trades = (changes.abs() > 0).sum() // 2
    
    in_pos = False
    trade_rets = []
    avg_hold = 0
    hold_days = []
    
    for i, (date, pos) in enumerate(df['position'].items()):
        if pos == 1.0 and not in_pos:
            in_pos = True
            entry_price = df.loc[date, 'close']
            entry_idx = i
        elif pos == 0.0 and in_pos:
            in_pos = False
            exit_price = df.loc[date, 'close']
            trade_ret = (exit_price - entry_price) / entry_price
            trade_rets.append(trade_ret)
            hold_days.append(i - entry_idx)
    
    winning = sum(1 for r in trade_rets if r > 0)
    win_rate = winning / len(trade_rets) * 100 if trade_rets else 0
    avg_hold = np.mean(hold_days) if hold_days else 0
    
    # Time in market
    in_market_pct = df['position'].mean() * 100
    
    results.append({
        'max_hold': mh,
        'trades': n_trades,
        'win_rate': round(win_rate, 1),
        'sharpe': round(sharpe, 2),
        'cagr': round(cagr * 100, 1),
        'max_dd': round(max_dd * 100, 1),
        'in_market': round(in_market_pct, 1),
        'avg_hold': round(avg_hold, 0)
    })

# ================================================================
# Print Results
# ================================================================
print("\n" + "=" * 70)
print("MSVR v6 RESULTS — Selective Entry + Extended Hold")
print("=" * 70)

print(f"\n{'MaxHold':<10} {'Trades':<10} {'WinRate':<10} {'Sharpe':<10} {'CAGR':<10} {'MaxDD':<10} {'InMkt%':<10} {'AvgHold':<10}")
print("-" * 80)

for r in results:
    print(f"{r['max_hold']:<10} {r['trades']:<10} {r['win_rate']:<10} {r['sharpe']:<10} {r['cagr']:<10} {r['max_dd']:<10} {r['in_market']:<10} {r['avg_hold']:<10}")

# Find best
best_sharpe = max(results, key=lambda x: x['sharpe'])
best_winrate = max(results, key=lambda x: x['win_rate'])
best_cagr = max(results, key=lambda x: x['cagr'])

print(f"\n{'='*70}")
print("BEST CONFIGS:")
print(f"{'='*70}")
print(f"Best Sharpe:    MaxHold={best_sharpe['max_hold']}, Sharpe={best_sharpe['sharpe']}, WinRate={best_sharpe['win_rate']}%, CAGR={best_sharpe['cagr']}%, AvgHold={best_sharpe['avg_hold']}d")
print(f"Best WinRate:   MaxHold={best_winrate['max_hold']}, Sharpe={best_winrate['sharpe']}, WinRate={best_winrate['win_rate']}%, CAGR={best_winrate['cagr']}%, AvgHold={best_winrate['avg_hold']}d")
print(f"Best CAGR:      MaxHold={best_cagr['max_hold']}, Sharpe={best_cagr['sharpe']}, WinRate={best_cagr['win_rate']}%, CAGR={best_cagr['cagr']}%, AvgHold={best_cagr['avg_hold']}d")

# Compare with Ichimoku
print(f"\n{'='*70}")
print("COMPARISON WITH ICHIMOKU:")
print(f"{'='*70}")
print(f"{'Config':<25} {'Trades':<10} {'WinRate':<10} {'Sharpe':<10} {'CAGR':<10} {'AvgHold':<10}")
print("-" * 75)
print(f"{'Ichimoku':<25} {'11':<10} {'63.6%':<10} {'1.31':<10} {'55.6%':<10} {'62d':<10}")
print(f"{'MSVR v3 (MH=45)':<25} {'15':<10} {'66.7%':<10} {'1.12':<10} {'39.5%':<10} {'47d':<10}")
print(f"{'MSVR v6 (Best)':<25} {best_sharpe['trades']:<10} {best_sharpe['win_rate']}%{'':<7} {best_sharpe['sharpe']:<10} {best_sharpe['cagr']}%{'':<7} {best_sharpe['avg_hold']}d{'':<7}")
