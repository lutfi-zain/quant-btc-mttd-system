#!/usr/bin/env python3
"""
Generate CLEAN Trade Charts
============================

Only shows:
- Entry (▲ green)
- Exit (▼ red)
No repeated signals!
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import importlib.util
import sys
import os
sys.path.append('/home/ubuntu/projects/quant-technical-indicator-bank')
from indicators_helper import *

print("=" * 70)
print("GENERATE CLEAN TRADE CHARTS")
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
# Generate Signals
# ================================================================
print("\n[2/3] Generating signals...")

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
# Extract Trades (CLEAN - only entry/exit)
# ================================================================
print("\n[3/3] Extracting trades and generating charts...")

def extract_trades(positions):
    """Extract clean trades: only entry and exit points"""
    trades = []
    in_position = False
    entry_date = None
    entry_price = None
    
    for i, (date, pos) in enumerate(positions.items()):
        if pos == 1.0 and not in_position:
            # ENTRY
            in_position = True
            entry_date = date
            entry_price = df.loc[date, 'close']
        elif pos == 0.0 and in_position:
            # EXIT
            in_position = False
            exit_date = date
            exit_price = df.loc[date, 'close']
            pnl = (exit_price - entry_price) / entry_price
            trades.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'win': pnl > 0
            })
    
    return trades

def apply_position(entry_signal, min_hold=15, max_hold=60):
    """Apply position with min/max hold"""
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
            if hold_count >= max_hold:
                in_position = False
                hold_count = 0
                result.iloc[i] = 0.0
            else:
                result.iloc[i] = 1.0
        else:
            result.iloc[i] = 0.0
    
    return result

# Define systems
systems = {
    'MSVR': {
        'entry': df['msvr_direction'],
        'overlay': None,
        'min_hold': 15,
        'max_hold': 90
    },
    'Ichimoku': {
        'entry': df['ichimoku_signal'].clip(0, 1),
        'overlay': 'ichimoku',
        'min_hold': 15,
        'max_hold': 60
    },
    'Supertrend': {
        'entry': df['supertrend_direction'],
        'overlay': 'supertrend',
        'min_hold': 15,
        'max_hold': 90
    },
    'Keltner': {
        'entry': df['keltner_direction'],
        'overlay': 'keltner',
        'min_hold': 15,
        'max_hold': 60
    }
}

# Generate charts
for name, config in systems.items():
    print(f"  Generating {name} chart...")
    
    # Apply position
    positions = apply_position(config['entry'], config['min_hold'], config['max_hold'])
    
    # Extract trades
    trades = extract_trades(positions)
    
    # Calculate metrics
    wins = sum(1 for t in trades if t['win'])
    win_rate = wins / len(trades) * 100 if trades else 0
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.suptitle(f'{name} — {len(trades)} trades, {win_rate:.1f}% win rate', 
                 fontsize=14, fontweight='bold')
    
    # Price chart
    ax1 = axes[0]
    ax1.plot(df.index, df['close'], color='#333333', linewidth=0.8, alpha=0.8)
    ax1.set_ylabel('BTC Price', fontsize=10)
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add indicator overlay
    if config['overlay'] == 'keltner':
        ax1.plot(df.index, df['kc_upper'], color='blue', linewidth=0.5, alpha=0.5)
        ax1.plot(df.index, df['kc_lower'], color='blue', linewidth=0.5, alpha=0.5)
        ax1.fill_between(df.index, df['kc_lower'], df['kc_upper'], alpha=0.1, color='blue')
    elif config['overlay'] == 'ichimoku':
        ax1.plot(df.index, df_ich['SenkouSpanA'], color='green', linewidth=0.5, alpha=0.5)
        ax1.plot(df.index, df_ich['SenkouSpanB'], color='red', linewidth=0.5, alpha=0.5)
    elif config['overlay'] == 'supertrend':
        ax1.plot(df.index, df['supertrend_signal'], color='purple', linewidth=0.5, alpha=0.5)
    
    # Plot ONLY entry/exit points (CLEAN!)
    for trade in trades:
        if trade['win']:
            color = '#2ecc71'  # Green for win
            marker_entry = '^'
            marker_exit = 'v'
        else:
            color = '#e74c3c'  # Red for loss
            marker_entry = '^'
            marker_exit = 'v'
        
        # Entry point
        ax1.scatter(trade['entry_date'], trade['entry_price'], 
                   marker=marker_entry, color=color, s=100, zorder=5, edgecolors='black', linewidth=0.5)
        
        # Exit point
        ax1.scatter(trade['exit_date'], trade['exit_price'], 
                   marker=marker_exit, color=color, s=100, zorder=5, edgecolors='black', linewidth=0.5)
        
        # Draw line connecting entry to exit
        ax1.plot([trade['entry_date'], trade['exit_date']], 
                [trade['entry_price'], trade['exit_price']], 
                color=color, linewidth=1, alpha=0.5)
    
    # Win/Loss bar chart
    ax2 = axes[1]
    trade_pnls = [t['pnl'] * 100 for t in trades]
    trade_dates = [t['exit_date'] for t in trades]
    colors_bar = ['#2ecc71' if t['win'] else '#e74c3c' for t in trades]
    ax2.bar(trade_dates, trade_pnls, color=colors_bar, width=10, alpha=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.set_ylabel('Trade P&L %', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Equity curve
    ax3 = axes[2]
    equity = pd.Series(1.0, index=df.index)
    for i, trade in enumerate(trades):
        # Find dates
        mask = (df.index >= trade['entry_date']) & (df.index <= trade['exit_date'])
        # Apply return
        equity.loc[mask] = equity.loc[mask] * (1 + trade['pnl'] / len(trades))
    
    ax3.plot(df.index, equity, color='#3498db', linewidth=1)
    ax3.fill_between(df.index, 1, equity, where=equity >= 1, alpha=0.3, color='#2ecc71')
    ax3.fill_between(df.index, 1, equity, where=equity < 1, alpha=0.3, color='#e74c3c')
    ax3.set_ylabel('Equity', fontsize=10)
    ax3.set_xlabel('Date', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Add metrics box
    total_return = (equity.iloc[-1] - 1) * 100
    metrics_text = f"Trades: {len(trades)} | Win Rate: {win_rate:.1f}% | Return: {total_return:.1f}%"
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'mttd/charts/{name.lower()}_clean.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✅ {name}: {len(trades)} trades, {win_rate:.1f}% win, saved to mttd/charts/{name.lower()}_clean.png")

print("\n" + "=" * 70)
print("ALL CLEAN CHARTS GENERATED!")
print("=" * 70)
