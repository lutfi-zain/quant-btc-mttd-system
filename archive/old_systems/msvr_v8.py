#!/usr/bin/env python3
"""
MSVR v8 — Optimized for Sharpe > 1.35
=======================================

Target: 25-35 trades, Sharpe > 1.35, Win Rate > 60%

Changes from v7:
1. Extended max_hold to 120 days (capture longer trends)
2. Relaxed exit conditions (only exit on EXTREME signals)
3. Added Regime Change exit (Family 9: Bayesian)
4. Optimized gate thresholds

Key Insight:
To get high Sharpe, we need to:
- Stay in market LONGER during trends
- Exit QUICKLY during reversals
- Avoid CHOPPY markets
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
print("MSVR v8 — Optimized for Sharpe > 1.35")
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
df['er_gate'] = (df['er'] > 0.20).astype(float)  # Even more relaxed

# ================================================================
# Layer 6: Volatility Cluster (Family 6: GARCH)
# ================================================================
from indicators.volatility_cluster import volatility_cluster
vol_result = volatility_cluster(df, window=20, threshold=1.3)  # Relaxed threshold
df['vol_direction'] = vol_result['direction'].clip(-1, 1)
df['vol_direction'] = (df['vol_direction'] > 0).astype(float)

# ================================================================
# Layer 7: Shannon Entropy (Family 7: Entropy)
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
df['entropy_gate'] = (df['entropy'] < 2.8).astype(float)  # More relaxed

# EXIT: EXTREME Entropy spike (2.5 std, not 2.0)
df['entropy_ma'] = df['entropy'].rolling(30).mean()
df['entropy_std'] = df['entropy'].rolling(30).std()
df['entropy_extreme'] = ((df['entropy'] > df['entropy_ma'] + 2.5 * df['entropy_std']) & 
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

# Track regime changes for EXIT
df['regime_change'] = (df['regime_direction'] != df['regime_direction'].shift(1)).astype(float)

# ================================================================
# COMPOSITE SIGNAL — OPTIMIZED ENTRY
# ================================================================
print("\n[3/6] Generating optimized entry signal...")

# Core: All 4 must agree (strict)
core_signal = (
    df['msvr_direction'] * 
    df['smooth_direction'] * 
    df['lr_direction'] * 
    df['cycle_direction']
)

# Gates: 3 of 5 must pass (very relaxed!)
gate_signal = (
    df['er_gate'] + 
    df['vol_direction'] + 
    df['entropy_gate'] + 
    df['volume_direction'] + 
    df['regime_direction']
)
gates_pass = (gate_signal >= 3).astype(float)  # 3 of 5 (very relaxed)

# Entry: Core × Gates
df['entry_signal'] = core_signal * gates_pass

# ================================================================
# EXIT SIGNAL — EXTREME Only
# ================================================================
print("\n[4/6] Computing exit signal (EXTREME only)...")

# Exit ONLY on EXTREME signals
# 1. EXTREME entropy spike (2.5 std)
# 2. MSVR STRONG reversal (not just weak)
df['msvr_strong_exit'] = (df['msvr_signal'] < -0.15).astype(float)  # Strong reversal

df['exit_signal'] = (
    (df['entropy_extreme'] == 1) | 
    (df['msvr_strong_exit'] == 1)
).astype(float)

# ================================================================
# POSITION — Optimized for Sharpe
# ================================================================
print("\n[5/6] Applying optimized position sizing...")

def apply_optimized_position(entry_signal, exit_signal, min_hold=30, max_hold=120):
    """
    Optimized position for high Sharpe.
    - Stay in market LONGER during trends
    - Exit QUICKLY on EXTREME signals only
    - Min hold: 30 days
    - Max hold: 120 days (extended!)
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
                # Exit: min_hold + EXTREME signal
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

# Test different configurations
configs = [
    {'min_hold': 25, 'max_hold': 90, 'gates_required': 3},
    {'min_hold': 30, 'max_hold': 120, 'gates_required': 3},
    {'min_hold': 35, 'max_hold': 120, 'gates_required': 3},
    {'min_hold': 30, 'max_hold': 150, 'gates_required': 3},  # Very extended
]

results = []

for cfg in configs:
    # Recompute entry with different gate requirement
    gates_pass_cfg = (gate_signal >= cfg['gates_required']).astype(float)
    entry_signal_cfg = core_signal * gates_pass_cfg
    
    df['position'] = apply_optimized_position(
        entry_signal_cfg, 
        df['exit_signal'], 
        min_hold=cfg['min_hold'], 
        max_hold=cfg['max_hold']
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
        'config': f"MH={cfg['min_hold']}/{cfg['max_hold']}_G{cfg['gates_required']}",
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
print("MSVR v8 RESULTS — Optimized for Sharpe > 1.35")
print("=" * 70)

print(f"\n{'Config':<25} {'Trades':<10} {'WinRate':<10} {'Sharpe':<10} {'CAGR':<10} {'MaxDD':<10} {'InMkt%':<10} {'AvgHold':<10}")
print("-" * 95)

for r in results:
    print(f"{r['config']:<25} {r['trades']:<10} {r['win_rate']:<10} {r['sharpe']:<10} {r['cagr']:<10} {r['max_dd']:<10} {r['in_market']:<10} {r['avg_hold']:<10}")

# Find best in target range (25-35 trades)
target_results = [r for r in results if 25 <= r['trades'] <= 35]

if target_results:
    best_sharpe = max(target_results, key=lambda x: x['sharpe'])
    best_winrate = max(target_results, key=lambda x: x['win_rate'])
    best_cagr = max(target_results, key=lambda x: x['cagr'])
    
    print(f"\n{'='*70}")
    print("BEST CONFIGS (25-35 trades):")
    print(f"{'='*70}")
    print(f"Best Sharpe:    {best_sharpe['config']}, Sharpe={best_sharpe['sharpe']}, WinRate={best_sharpe['win_rate']}%, CAGR={best_sharpe['cagr']}%, Trades={best_sharpe['trades']}")
    print(f"Best WinRate:   {best_winrate['config']}, Sharpe={best_winrate['sharpe']}, WinRate={best_winrate['win_rate']}%, CAGR={best_winrate['cagr']}%, Trades={best_winrate['trades']}")
    print(f"Best CAGR:      {best_cagr['config']}, Sharpe={best_cagr['sharpe']}, WinRate={best_cagr['win_rate']}%, CAGR={best_cagr['cagr']}%, Trades={best_cagr['trades']}")
else:
    print("\n⚠️  No configs in 25-35 trade range!")
    print("Best overall:")
    best_sharpe = max(results, key=lambda x: x['sharpe'])
    print(f"  {best_sharpe['config']}, Sharpe={best_sharpe['sharpe']}, Trades={best_sharpe['trades']}")

# Compare with Ichimoku
print(f"\n{'='*70}")
print("COMPARISON WITH ICHIMOKU:")
print(f"{'='*70}")
print(f"{'Config':<25} {'Trades':<10} {'WinRate':<10} {'Sharpe':<10} {'CAGR':<10} {'AvgHold':<10}")
print("-" * 75)
print(f"{'Ichimoku':<25} {'11':<10} {'63.6%':<10} {'1.31':<10} {'55.6%':<10} {'62d':<10}")

if target_results:
    best = best_sharpe
    print(f"{'MSVR v8 (Best)':<25} {best['trades']:<10} {best['win_rate']}%{'':<7} {best['sharpe']:<10} {best['cagr']}%{'':<7} {best['avg_hold']}d{'':<7}")
