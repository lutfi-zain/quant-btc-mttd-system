#!/usr/bin/env python3
"""
Test ALL Base Signals with MSVR v8 Filtering Framework
=======================================================
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
print("TEST ALL BASE SIGNALS WITH MSVR v8 FILTERING")
print("=" * 70)

# ================================================================
# Load Data
# ================================================================
print("\n[1/3] Loading data...")

with open('data/btc_daily.json') as f:
    btc_data = json.load(f)

df = pd.DataFrame(btc_data['aligned_data'])
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = df[df.index >= '2018-01-01']

print(f"  Data: {len(df)} bars")

# ================================================================
# Common Filtering Functions
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

def compute_filters(df):
    """Compute all MSVR v8 filters."""
    # SuperSmoother
    df['msvr_smooth'] = ehler_supersmoother(df.get('msvr_signal', df['close']), length=7)
    
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
    df['regime_direction'] = (hmm['direction'] > 0).astype(float)
    
    return df

def apply_position(entry_signal, exit_signal, min_hold=25, max_hold=90):
    """Apply position with min/max hold."""
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
    """Compute trading metrics."""
    returns = prices.pct_change() * positions.shift(1)
    returns = returns.dropna()
    
    if len(returns) == 0:
        return {'trades': 0, 'win_rate': 0, 'sharpe': 0, 'cagr': 0, 'avg_hold': 0}
    
    equity = (1 + returns).cumprod()
    years = len(returns) / 365.25
    cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    sharpe = returns.mean() / returns.std() * np.sqrt(365) if returns.std() > 0 else 0
    
    # Trades
    changes = positions.diff().fillna(0)
    n_trades = (changes.abs() > 0).sum() // 2
    
    in_pos = False
    trade_rets = []
    hold_days = []
    for i, (date, pos) in enumerate(positions.items()):
        if pos == 1.0 and not in_pos:
            in_pos = True
            entry_price = prices.loc[date]
            entry_idx = i
        elif pos == 0.0 and in_pos:
            in_pos = False
            exit_price = prices.loc[date]
            trade_rets.append((exit_price - entry_price) / entry_price)
            hold_days.append(i - entry_idx)
    
    winning = sum(1 for r in trade_rets if r > 0)
    win_rate = winning / len(trade_rets) * 100 if trade_rets else 0
    avg_hold = np.mean(hold_days) if hold_days else 0
    
    return {
        'trades': n_trades,
        'win_rate': round(win_rate, 1),
        'sharpe': round(sharpe, 2),
        'cagr': round(cagr * 100, 1),
        'avg_hold': round(avg_hold, 0)
    }

# ================================================================
# Base Signal Definitions
# ================================================================
print("\n[2/3] Testing base signals...")

results = []

# 1. MSVR (current base)
print("\n  Testing MSVR base...")
spec = importlib.util.spec_from_file_location('msvr', 
    '/home/ubuntu/projects/quant-technical-indicator-bank/perpetual/median_standard_deviation_viresearch.py')
msvr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msvr_module)
msvr_result = msvr_module.median_standard_deviation_viresearch(df)
df['msvr_signal'] = msvr_result['vii']
df['msvr_direction'] = (df['msvr_signal'] > 0).astype(float)

df_msvr = df.copy()
df_msvr = compute_filters(df_msvr)
core = df_msvr['msvr_direction'] * df_msvr['lr_direction'] * df_msvr['cycle_direction']
gates = df_msvr['er_gate'] + df_msvr['vol_direction'] + df_msvr['entropy_gate'] + df_msvr['volume_direction'] + df_msvr['regime_direction']
gates_pass = (gates >= 3).astype(float)
entry = core * gates_pass
exit_sig = ((df_msvr['entropy'] > df_msvr['entropy'].rolling(30).mean() + 2.5 * df_msvr['entropy'].rolling(30).std()) | (df_msvr['msvr_signal'] < -0.15)).astype(float)
df_msvr['position'] = apply_position(entry, exit_sig)
results.append(('MSVR', compute_metrics(df_msvr['position'], df_msvr['close'])))

# 2. Ichimoku
print("  Testing Ichimoku base...")
from ichimoku_quant import generate_ichimoku_features, generate_ichimoku_signals
df_ich = generate_ichimoku_features(df.copy())
df_ich = generate_ichimoku_signals(df_ich)
df_ich['base_signal'] = df_ich['Pos']
df_ich = compute_filters(df_ich)
core = df_ich['base_signal'] * df_ich['lr_direction'] * df_ich['cycle_direction']
gates = df_ich['er_gate'] + df_ich['vol_direction'] + df_ich['entropy_gate'] + df_ich['volume_direction'] + df_ich['regime_direction']
gates_pass = (gates >= 3).astype(float)
entry = core * gates_pass
exit_sig = ((df_ich['entropy'] > df_ich['entropy'].rolling(30).mean() + 2.5 * df_ich['entropy'].rolling(30).std()) | (df_ich.get('msvr_signal', pd.Series(0, index=df_ich.index)) < -0.15)).astype(float)
df_ich['position'] = apply_position(entry, exit_sig)
results.append(('Ichimoku', compute_metrics(df_ich['position'], df_ich['close'])))

# 3. Bollinger Breakout
print("  Testing Bollinger base...")
df_bb = df.copy()
bb_mid = df_bb['close'].rolling(25).mean()
bb_std = df_bb['close'].rolling(25).std()
df_bb['bb_upper'] = bb_mid + 2.0 * bb_std
df_bb['bb_lower'] = bb_mid - 2.0 * bb_std
df_bb['base_signal'] = 0.0
df_bb.loc[df_bb['close'] > df_bb['bb_upper'], 'base_signal'] = 1.0
df_bb.loc[df_bb['close'] < df_bb['bb_lower'], 'base_signal'] = -1.0
df_bb['base_direction'] = (df_bb['base_signal'] > 0).astype(float)
df_bb = compute_filters(df_bb)
core = df_bb['base_direction'] * df_bb['lr_direction'] * df_bb['cycle_direction']
gates = df_bb['er_gate'] + df_bb['vol_direction'] + df_bb['entropy_gate'] + df_bb['volume_direction'] + df_bb['regime_direction']
gates_pass = (gates >= 3).astype(float)
entry = core * gates_pass
exit_sig = ((df_bb['entropy'] > df_bb['entropy'].rolling(30).mean() + 2.5 * df_bb['entropy'].rolling(30).std())).astype(float)
df_bb['position'] = apply_position(entry, exit_sig)
results.append(('Bollinger', compute_metrics(df_bb['position'], df_bb['close'])))

# 4. ADX Trend
print("  Testing ADX base...")
df_adx = df.copy()
# Compute ADX
plus_dm = df_adx['high'].diff()
minus_dm = -df_adx['low'].diff()
plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
atr = ema(df_adx['high'] - df_adx['low'], 14)
plus_di = 100 * ema(plus_dm, 14) / atr
minus_di = 100 * ema(minus_dm, 14) / atr
dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
adx = ema(dx, 14)
df_adx['adx'] = adx
df_adx['plus_di'] = plus_di
df_adx['minus_di'] = minus_di
df_adx['base_signal'] = 0.0
df_adx.loc[(df_adx['adx'] > 25) & (df_adx['plus_di'] > df_adx['minus_di']), 'base_signal'] = 1.0
df_adx.loc[(df_adx['adx'] > 25) & (df_adx['minus_di'] > df_adx['plus_di']), 'base_signal'] = -1.0
df_adx['base_direction'] = (df_adx['base_signal'] > 0).astype(float)
df_adx = compute_filters(df_adx)
core = df_adx['base_direction'] * df_adx['lr_direction'] * df_adx['cycle_direction']
gates = df_adx['er_gate'] + df_adx['vol_direction'] + df_adx['entropy_gate'] + df_adx['volume_direction'] + df_adx['regime_direction']
gates_pass = (gates >= 3).astype(float)
entry = core * gates_pass
exit_sig = ((df_adx['entropy'] > df_adx['entropy'].rolling(30).mean() + 2.5 * df_adx['entropy'].rolling(30).std())).astype(float)
df_adx['position'] = apply_position(entry, exit_sig)
results.append(('ADX', compute_metrics(df_adx['position'], df_adx['close'])))

# 5. Supertrend
print("  Testing Supertrend base...")
df_st = df.copy()
spec_st = importlib.util.spec_from_file_location('supertrend', '/home/ubuntu/projects/quant-technical-indicator-bank/perpetual/median_supertrend_viresearch.py')
st_module = importlib.util.module_from_spec(spec_st)
spec_st.loader.exec_module(st_module)
st_result = st_module.median_supertrend_viresearch(df_st)
df_st['base_signal'] = st_result['vii']
df_st['base_direction'] = (df_st['base_signal'] > 0).astype(float)
df_st = compute_filters(df_st)
core = df_st['base_direction'] * df_st['lr_direction'] * df_st['cycle_direction']
gates = df_st['er_gate'] + df_st['vol_direction'] + df_st['entropy_gate'] + df_st['volume_direction'] + df_st['regime_direction']
gates_pass = (gates >= 3).astype(float)
entry = core * gates_pass
exit_sig = ((df_st['entropy'] > df_st['entropy'].rolling(30).mean() + 2.5 * df_st['entropy'].rolling(30).std())).astype(float)
df_st['position'] = apply_position(entry, exit_sig)
results.append(('Supertrend', compute_metrics(df_st['position'], df_st['close'])))

# 6. Keltner Channel
print("  Testing Keltner base...")
df_kc = df.copy()
kc_mid = ema(df_kc['close'], 20)
atr_kc = ema(df_kc['high'] - df_kc['low'], 20)
df_kc['kc_upper'] = kc_mid + 1.5 * atr_kc
df_kc['kc_lower'] = kc_mid - 1.5 * atr_kc
df_kc['base_signal'] = 0.0
df_kc.loc[df_kc['close'] > df_kc['kc_upper'], 'base_signal'] = 1.0
df_kc.loc[df_kc['close'] < df_kc['kc_lower'], 'base_signal'] = -1.0
df_kc['base_direction'] = (df_kc['base_signal'] > 0).astype(float)
df_kc = compute_filters(df_kc)
core = df_kc['base_direction'] * df_kc['lr_direction'] * df_kc['cycle_direction']
gates = df_kc['er_gate'] + df_kc['vol_direction'] + df_kc['entropy_gate'] + df_kc['volume_direction'] + df_kc['regime_direction']
gates_pass = (gates >= 3).astype(float)
entry = core * gates_pass
exit_sig = ((df_kc['entropy'] > df_kc['entropy'].rolling(30).mean() + 2.5 * df_kc['entropy'].rolling(30).std())).astype(float)
df_kc['position'] = apply_position(entry, exit_sig)
results.append(('Keltner', compute_metrics(df_kc['position'], df_kc['close'])))

# ================================================================
# Print Results
# ================================================================
print("\n" + "=" * 70)
print("RESULTS — ALL BASE SIGNALS + MSVR v8 FILTERING")
print("=" * 70)

print(f"\n{'Base Signal':<15} {'Trades':<10} {'WinRate':<10} {'Sharpe':<10} {'CAGR':<10} {'AvgHold':<10}")
print("-" * 65)

for name, m in results:
    print(f"{name:<15} {m['trades']:<10} {m['win_rate']:<10} {m['sharpe']:<10} {m['cagr']:<10} {m['avg_hold']:<10}")

# Find best
best_sharpe = max(results, key=lambda x: x[1]['sharpe'])
best_winrate = max(results, key=lambda x: x[1]['win_rate'])
best_cagr = max(results, key=lambda x: x[1]['cagr'])

print(f"\n{'='*70}")
print("WINNERS:")
print(f"{'='*70}")
print(f"Best Sharpe:    {best_sharpe[0]} ({best_sharpe[1]['sharpe']})")
print(f"Best WinRate:   {best_winrate[0]} ({best_winrate[1]['win_rate']}%)")
print(f"Best CAGR:      {best_cagr[0]} ({best_cagr[1]['cagr']}%)")
