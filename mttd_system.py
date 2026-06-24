#!/usr/bin/env python3
"""
MTTD System — Robust Configuration
====================================

Main trading system using:
- MSVR (Median Standard Deviation Viresearch) for direction
- Cycle Phase (FFT-based) for timing
- Trend Filter: SMA(75) > SMA(250)
- Bollinger Filter: 25-period, 2.0 std
- Min Hold: 45 days

Config: T75/250_BB25_2.0s_MH45
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path
from datetime import datetime

# Add indicator bank to path
project_root = os.path.dirname(os.path.abspath(__file__))
bank_root = '/home/ubuntu/projects/quant-technical-indicator-bank'
sys.path.append(project_root)
sys.path.append(bank_root)
from indicators_helper import sma

# ================================================================
# Configuration
# ================================================================
TREND_FAST = 75
TREND_SLOW = 250
BB_PERIOD = 25
BB_STD = 2.0
MIN_HOLD = 45
CYCLE_LOOKBACK = 40

# Output directory
OUTPUT_DIR = os.path.join(project_root, 'mttd')

print("=" * 70)
print("MTTD SYSTEM — ROBUST CONFIGURATION")
print(f"Config: T{TREND_FAST}/{TREND_SLOW}_BB{BB_PERIOD}_{BB_STD}s_MH{MIN_HOLD}")
print("=" * 70)

# ================================================================
# Load BTC Data
# ================================================================
print("\n[1/6] Loading BTC data...")

with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
    btc_data = json.load(f)

df = pd.DataFrame(btc_data['aligned_data'])
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = df[df.index >= '2018-01-01']

print(f"  Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")

# ================================================================
# Load MSVR Indicator
# ================================================================
print("\n[2/6] Loading MSVR indicator...")

spec = importlib.util.spec_from_file_location(
    'msvr', 
    os.path.join(bank_root, 'perpetual/median_standard_deviation_viresearch.py')
)
msvr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msvr_module)

msvr_result = msvr_module.median_standard_deviation_viresearch(df)
msvr_signal = msvr_result['vii']

print(f"  MSVR signal loaded: {(msvr_signal == 1).sum()} bullish, {(msvr_signal == -1).sum()} bearish")

# ================================================================
# Compute Cycle Phase
# ================================================================
print("\n[3/6] Computing cycle phase...")

def compute_cycle_phase(df, lookback):
    """
    Compute cycle phase using FFT.
    Returns phase (0 to 2π) for timing entries/exits.
    """
    src = (df['high'] + df['low'] + df['close']) / 3.0
    n = len(df)
    phase = pd.Series(np.nan, index=df.index)
    
    min_period = 5
    max_period = lookback // 2
    
    for i in range(lookback - 1, n):
        window = src.iloc[i - lookback + 1:i + 1].values
        
        if np.any(np.isnan(window)):
            continue
        
        # Detrend
        window_detrended = window - np.mean(window)
        
        # Hanning window
        hann = np.hanning(lookback)
        window窗ed = window_detrended * hann
        
        # FFT
        fft_vals = np.fft.rfft(window窗ed)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(lookback, d=1)
        
        # Find dominant frequency
        min_freq = 1.0 / max_period
        max_freq = 1.0 / min_period
        valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
        valid_power = power[valid_mask]
        valid_freqs = freqs[valid_mask]
        
        if len(valid_power) > 0 and np.sum(valid_power) > 0:
            dominant_idx = np.argmax(valid_power)
            dominant_freq = valid_freqs[dominant_idx]
            dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else lookback
            
            # Compute phase
            cycle_pos = i % int(dominant_period)
            phase.iloc[i] = 2 * np.pi * cycle_pos / dominant_period
    
    return phase

phase = compute_cycle_phase(df, lookback=CYCLE_LOOKBACK)
cycle_signal = -np.cos(phase)  # +1 at trough (buy), -1 at peak (sell)

print(f"  Cycle phase computed with lookback={CYCLE_LOOKBACK}")

# ================================================================
# Apply Filters
# ================================================================
print("\n[4/6] Applying filters...")

# Basic combined signal
msvr_binary = (msvr_signal > 0).astype(float)
cycle_binary = (cycle_signal > 0).astype(float)
raw_combined = msvr_binary * cycle_binary

# Trend Filter
trend_fast = sma(df['close'], TREND_FAST)
trend_slow = sma(df['close'], TREND_SLOW)
trend_filter = (trend_fast > trend_slow).astype(float)

# Bollinger Filter
bb_mid = sma(df['close'], BB_PERIOD)
bb_std = df['close'].rolling(BB_PERIOD).std()
bb_upper = bb_mid + BB_STD * bb_std
bb_lower = bb_mid - BB_STD * bb_std
bb_signal = ((df['close'] > bb_lower) & (df['close'] < bb_upper)).astype(float)

# Combined signal
combined = raw_combined * trend_filter * bb_signal

# Apply Min Hold
def apply_min_hold(signal, min_hold):
    """Force minimum holding period."""
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

final_signal = apply_min_hold(combined, MIN_HOLD)

print(f"  Trend filter: {trend_filter.mean()*100:.1f}% bullish")
print(f"  Bollinger filter: {bb_signal.mean()*100:.1f}% in band")
print(f"  Final signal: {final_signal.mean()*100:.1f}% in position")

# ================================================================
# Compute Metrics
# ================================================================
print("\n[5/6] Computing metrics...")

def compute_metrics(signal, prices, transaction_cost=0.001):
    """Compute comprehensive trading metrics."""
    returns = prices.pct_change()
    strategy_returns = returns * signal.shift(1)
    strategy_returns = strategy_returns.dropna()
    
    # Transaction costs
    transitions = signal.diff().fillna(0)
    strategy_returns = strategy_returns - transitions.loc[strategy_returns.index] * (transaction_cost / 2)

    if len(strategy_returns) == 0:
        return {
            'cagr': 0, 'sharpe': 0, 'sortino': 0, 'calmar': 0,
            'max_dd': 0, 'n_trades': 0, 'win_rate': 0, 'avg_hold': 0
        }

    equity = (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25

    cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
    downside = strategy_returns[strategy_returns < 0]
    sortino = strategy_returns.mean() / downside.std() * np.sqrt(365) if len(downside) > 0 and downside.std() > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
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
            entry_price = prices.loc[date]
        elif pos == 0.0 and in_position:
            in_position = False
            if hold_start is not None:
                hold_days = (date - hold_start).days
                hold_periods.append(hold_days)
                exit_price = prices.loc[date]
                trade_ret = (exit_price - entry_price) / entry_price
                trade_returns.append(trade_ret)
    
    winning = sum(1 for r in trade_returns if r > 0)
    total = len(trade_returns)
    win_rate = winning / total * 100 if total > 0 else 0
    avg_hold = np.mean(hold_periods) if hold_periods else 0

    return {
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'calmar': round(calmar, 2),
        'max_dd': round(max_dd * 100, 2),
        'n_trades': n_trades,
        'win_rate': round(win_rate, 1),
        'avg_hold': round(avg_hold, 0),
        'equity': equity
    }

metrics = compute_metrics(final_signal, df['close'])

print(f"\n  PERFORMANCE METRICS:")
print(f"  {'─'*40}")
print(f"  Sharpe Ratio:     {metrics['sharpe']:.2f}")
print(f"  CAGR:             {metrics['cagr']:.1f}%")
print(f"  Max Drawdown:     {metrics['max_dd']:.1f}%")
print(f"  Win Rate:         {metrics['win_rate']:.1f}%")
print(f"  Trades:           {metrics['n_trades']}")
print(f"  Avg Hold:         {metrics['avg_hold']:.0f} days")

# ================================================================
# Save Results
# ================================================================
print("\n[6/6] Saving results...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Signals CSV
signals_df = pd.DataFrame({
    'date': df.index,
    'position': final_signal.values,
    'btc_price': df['close'].values
})
signals_path = os.path.join(OUTPUT_DIR, 'signals.csv')
signals_df.to_csv(signals_path, index=False)
print(f"  Saved: {signals_path}")

# 2. Equity CSV
if len(metrics['equity']) > 0:
    equity_df = pd.DataFrame({
        'date': metrics['equity'].index,
        'equity': metrics['equity'].values
    })
    equity_df['drawdown'] = (metrics['equity'] / metrics['equity'].cummax() - 1).values
    equity_path = os.path.join(OUTPUT_DIR, 'equity.csv')
    equity_df.to_csv(equity_path, index=False)
    print(f"  Saved: {equity_path}")

# 3. Metrics JSON
metrics_output = {
    'config': {
        'trend_fast': int(TREND_FAST),
        'trend_slow': int(TREND_SLOW),
        'bb_period': int(BB_PERIOD),
        'bb_std': float(BB_STD),
        'min_hold': int(MIN_HOLD),
        'cycle_lookback': int(CYCLE_LOOKBACK)
    },
    'performance': {
        'sharpe': float(metrics['sharpe']),
        'cagr': float(metrics['cagr']),
        'sortino': float(metrics['sortino']),
        'calmar': float(metrics['calmar']),
        'max_dd': float(metrics['max_dd']),
        'n_trades': int(metrics['n_trades']),
        'win_rate': float(metrics['win_rate']),
        'avg_hold': float(metrics['avg_hold'])
    },
    'generated_at': datetime.now().isoformat()
}

metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics_output, f, indent=2)
print(f"  Saved: {metrics_path}")

print("\n" + "=" * 70)
print("MTTD SYSTEM COMPLETE")
print("=" * 70)
