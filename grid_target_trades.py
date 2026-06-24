#!/usr/bin/env python3
"""
Grid Search Targeting 25-35 Trades
====================================

Goal: Find configs with 25-35 trades while maintaining good metrics
"""

import numpy as np
import pandas as pd
import json
import sqlite3
import importlib.util
import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/ubuntu/projects/quant-technical-indicator-bank')
from indicators_helper import *

print("=" * 70)
print("GRID SEARCH — TARGET 25-35 TRADES")
print("=" * 70)

# ================================================================
# Load Data
# ================================================================
print("\n[1/4] Loading data...")

with open('data/btc_daily.json') as f:
    btc_data = json.load(f)

df = pd.DataFrame(btc_data['aligned_data'])
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = df[df.index >= '2018-01-01']

# Load regime data
regime_df = pd.read_csv('mttd/regime_data.csv')
regime_df['date'] = pd.to_datetime(regime_df['date'])
regime_df = regime_df.set_index('date')
regime_df = regime_df.reindex(df.index)
df['regime'] = regime_df['regime']
df['composite_score'] = regime_df['composite_score']

print(f"  Data: {len(df)} bars")
print(f"  Regime: {df['regime'].value_counts().to_dict()}")

# ================================================================
# Indicator Functions
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

def efficiency_ratio(series, period=14):
    change = series.diff().abs()
    volatility = change.rolling(period).sum()
    direction = series.diff(period).abs()
    return direction / volatility

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

# ================================================================
# Generate Base Signals
# ================================================================
print("\n[2/4] Generating base signals...")

# MSVR
spec = importlib.util.spec_from_file_location('msvr', 
    '/home/ubuntu/projects/quant-technical-indicator-bank/perpetual/median_standard_deviation_viresearch.py')
msvr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msvr_module)
msvr_result = msvr_module.median_standard_deviation_viresearch(df)
df['msvr_signal'] = msvr_result['vii']
df['msvr_direction'] = (df['msvr_signal'] > 0).astype(float)

# SuperSmoother
df['msvr_smooth'] = ehler_supersmoother(df['msvr_signal'], length=7)
df['smooth_direction'] = (df['msvr_smooth'] > 0).astype(float)

# LinearReg
from indicators.linear_reg_trend import linear_reg_trend
lr = linear_reg_trend(df, length=50)
df['lr_direction'] = (lr['direction'] > 0).astype(float)

# Cycle Phase
phase = compute_cycle_phase(df, lookback=40)
df['cycle_signal'] = -np.cos(phase)
df['cycle_direction'] = (df['cycle_signal'] > 0).astype(float)

# Efficiency Ratio
df['er'] = efficiency_ratio(df['close'], period=14)
df['er_gate'] = (df['er'] > 0.20).astype(float)

# Volatility Cluster
from indicators.volatility_cluster import volatility_cluster
vol = volatility_cluster(df, window=20, threshold=1.3)
df['vol_direction'] = (vol['direction'] > 0).astype(float)

# Shannon Entropy
df['entropy'] = shannon_entropy(df['close'], window=15, bins=6)
df['entropy_gate'] = (df['entropy'] < 2.8).astype(float)

# Volume Confirm
from indicators.volume_confirm import volume_confirm
vc = volume_confirm(df, obv_short=10, obv_long=30, spike_mult=1.5)
df['volume_direction'] = (vc['direction'] > 0).astype(float)

# HMM Regime
from indicators.hmm_regime import hmm_regime
hmm = hmm_regime(df, n_states=3, window=100)
df['hmm_direction'] = (hmm['direction'] > 0).astype(float)

# Ichimoku
from ichimoku_quant import generate_ichimoku_features, generate_ichimoku_signals
df_ich = generate_ichimoku_features(df.copy())
df_ich = generate_ichimoku_signals(df_ich)
df['ichimoku_signal'] = df_ich['Pos']

# Supertrend
spec_st = importlib.util.spec_from_file_location('supertrend', 
    '/home/ubuntu/projects/quant-technical-indicator-bank/perpetual/median_supertrend_viresearch.py')
st_module = importlib.util.module_from_spec(spec_st)
spec_st.loader.exec_module(st_module)
st_result = st_module.median_supertrend_viresearch(df)
df['supertrend_signal'] = st_result['vii']
df['supertrend_direction'] = (df['supertrend_signal'] > 0).astype(float)

# Keltner Channel
kc_mid = ema(df['close'], 20)
atr_kc = ema(df['high'] - df['low'], 20)
df['kc_upper'] = kc_mid + 1.5 * atr_kc
df['kc_lower'] = kc_mid - 1.5 * atr_kc
df['keltner_signal'] = 0.0
df.loc[df['close'] > df['kc_upper'], 'keltner_signal'] = 1.0
df.loc[df['close'] < df['kc_lower'], 'keltner_signal'] = -1.0
df['keltner_direction'] = (df['keltner_signal'] > 0).astype(float)

# ================================================================
# Grid Search
# ================================================================
print("\n[3/4] Running grid search...")

def apply_position(entry_signal, exit_signal, min_hold, max_hold):
    result = pd.Series(0.0, index=entry_signal.index)
    in_position = False
    hold_count = 0
    for i in range(len(result)):
        if entry_signal.iloc[i] == 1.0 and not in_position:
            in_position = True
            hold_count = 0
            result.iloc[i] = 1.0
        elif in_position:
            hold_count += 1
            if hold_count >= min_hold and exit_signal.iloc[i] == 1.0:
                in_position = False
                hold_count = 0
                result.iloc[i] = 0.0
            elif hold_count >= max_hold:
                in_position = False
                hold_count = 0
                result.iloc[i] = 0.0
            else:
                result.iloc[i] = 1.0
        else:
            result.iloc[i] = 0.0
    return result

def compute_metrics(positions, prices):
    returns = prices.pct_change() * positions.shift(1)
    returns = returns.dropna()
    if len(returns) == 0:
        return {'trades': 0, 'win_rate': 0, 'sharpe': 0, 'cagr': 0}
    equity = (1 + returns).cumprod()
    years = len(returns) / 365.25
    cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0
    changes = positions.diff().fillna(0)
    n_trades = (changes.abs() > 0).sum() // 2
    in_pos = False
    trade_rets = []
    for i, (date, pos) in enumerate(positions.items()):
        if pos == 1.0 and not in_pos:
            in_pos = True
            entry_price = prices.loc[date]
        elif pos == 0.0 and in_pos:
            in_pos = False
            exit_price = prices.loc[date]
            trade_rets.append((exit_price - entry_price) / entry_price)
    winning = sum(1 for r in trade_rets if r > 0)
    win_rate = winning / len(trade_rets) * 100 if trade_rets else 0
    return {'trades': n_trades, 'win_rate': round(win_rate, 1), 'sharpe': round(sharpe, 2), 'cagr': round(cagr * 100, 1)}

# Test configurations
configs = []

# Base signals
bases = {
    'MSVR': df['msvr_direction'],
    'Ichimoku': df['ichimoku_signal'].clip(0, 1),
    'Supertrend': df['supertrend_direction'],
    'Keltner': df['keltner_direction'],
}

# Filters
filters = {
    'smooth': df['smooth_direction'],
    'lr': df['lr_direction'],
    'cycle': df['cycle_direction'],
    'er': df['er_gate'],
    'vol': df['vol_direction'],
    'entropy': df['entropy_gate'],
    'volume': df['volume_direction'],
    'hmm': df['hmm_direction'],
}

# Regime modes
regime_modes = ['none', 'bull_only', 'bull_with_filters']

# Parameter ranges (focused on getting 25-35 trades)
min_hold_range = [15, 20, 25, 30]
max_hold_range = [45, 60, 75, 90]
regime_threshold_range = [0.0, 0.3, 0.5]

results = []

for base_name, base_signal in bases.items():
    for regime_mode in regime_modes:
        for mh in min_hold_range:
            for xh in max_hold_range:
                for rt in regime_threshold_range:
                    # Build entry signal
                    if regime_mode == 'none':
                        entry = base_signal.copy()
                    elif regime_mode == 'bull_only':
                        regime_filter = (df['composite_score'] > rt).astype(float)
                        entry = base_signal * regime_filter
                    else:  # bull_with_filters
                        regime_filter = (df['composite_score'] > rt).astype(float)
                        filter_sum = sum(f.values for f in filters.values())
                        filter_pass = (filter_sum >= 3).astype(float)
                        entry = base_signal * regime_filter * filter_pass
                    
                    # Exit signal
                    exit_signal = (df['entropy'] > df['entropy'].rolling(30).mean() + 2.5 * df['entropy'].rolling(30).std()).astype(float)
                    
                    # Apply position
                    df['position'] = apply_position(entry, exit_signal, mh, xh)
                    
                    # Compute metrics
                    metrics = compute_metrics(df['position'], df['close'])
                    
                    # Only keep configs with 20-40 trades
                    if 20 <= metrics['trades'] <= 40:
                        configs.append({
                            'system': f"{base_name}_{regime_mode}",
                            'min_hold': mh,
                            'max_hold': xh,
                            'regime_threshold': rt,
                            'trades': metrics['trades'],
                            'win_rate': metrics['win_rate'],
                            'sharpe': metrics['sharpe'],
                            'cagr': metrics['cagr']
                        })

# Sort by trades (closest to 30)
configs.sort(key=lambda x: abs(x['trades'] - 30))

# ================================================================
# Print Results
# ================================================================
print("\n" + "=" * 70)
print("CONFIGS WITH 20-40 TRADES (sorted by distance to 30 trades)")
print("=" * 70)

print(f"\n{'System':<30} {'MH':<5} {'XH':<5} {'T':<5} {'Trades':<8} {'WinRate':<10} {'Sharpe':<10} {'CAGR':<10}")
print("-" * 85)

for c in configs[:30]:  # Top 30
    print(f"{c['system']:<30} {c['min_hold']:<5} {c['max_hold']:<5} {c['regime_threshold']:<5} {c['trades']:<8} {c['win_rate']:<10} {c['sharpe']:<10} {c['cagr']:<10}")

# Find best in 25-35 range
target_configs = [c for c in configs if 25 <= c['trades'] <= 35]

if target_configs:
    best_sharpe = max(target_configs, key=lambda x: x['sharpe'])
    best_winrate = max(target_configs, key=lambda x: x['win_rate'])
    best_cagr = max(target_configs, key=lambda x: x['cagr'])
    
    print(f"\n{'='*70}")
    print("BEST CONFIGS (25-35 trades):")
    print(f"{'='*70}")
    print(f"Best Sharpe:    {best_sharpe['system']}, MH={best_sharpe['min_hold']}/{best_sharpe['max_hold']}, Sharpe={best_sharpe['sharpe']}, WinRate={best_sharpe['win_rate']}%, Trades={best_sharpe['trades']}")
    print(f"Best WinRate:   {best_winrate['system']}, MH={best_winrate['min_hold']}/{best_winrate['max_hold']}, Sharpe={best_winrate['sharpe']}, WinRate={best_winrate['win_rate']}%, Trades={best_winrate['trades']}")
    print(f"Best CAGR:      {best_cagr['system']}, MH={best_cagr['min_hold']}/{best_cagr['max_hold']}, Sharpe={best_cagr['sharpe']}, WinRate={best_cagr['win_rate']}%, Trades={best_cagr['trades']}")

# Save results
results_df = pd.DataFrame(configs)
results_df.to_csv('mttd/target_trades_grid_results.csv', index=False)
print(f"\nResults saved to mttd/target_trades_grid_results.csv ({len(configs)} configs)")
