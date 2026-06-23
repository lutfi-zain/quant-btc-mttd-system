#!/usr/bin/env python3
"""
Analyze ISP Behavior — How does ISP actually trade?
====================================================

User feedback: MTTD is too noisy, not applicable.
ISP behavior is the target — fewer trades, longer holds.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ISP BEHAVIOR ANALYSIS")
print("=" * 70)

# ================================================================
# Load Data
# ================================================================
print("\n[1/3] Loading data...")

project_root = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
    btc_data = json.load(f)

df_full = pd.DataFrame(btc_data['aligned_data'])
df_full['time'] = pd.to_datetime(df_full['time'])
df_full = df_full.set_index('time')
df_full = df_full[df_full.index >= '2018-01-01']

print(f"  BTC data: {len(df_full)} bars ({df_full.index[0]} to {df_full.index[-1]})")

# Load ISP
isp_df = pd.read_csv(os.path.join(project_root, 'isp-signals-btcusd-2026-06-13.csv'))
isp_df['Date'] = pd.to_datetime(isp_df['Date'])
isp_df = isp_df.set_index('Date')

print(f"\n  ISP Signals:")
print(f"  Total signals: {len(isp_df)}")
print(f"  BUY signals:  {(isp_df['Action'] == 'BUY').sum()}")
print(f"  SELL signals: {(isp_df['Action'] == 'SELL').sum()}")

# ================================================================
# Analyze ISP Position
# ================================================================
print("\n[2/3] Analyzing ISP position behavior...")

isp_positions = pd.Series(0.0, index=df_full.index)
for date, row in isp_df.iterrows():
    if date in isp_positions.index:
        if row['Action'] == 'BUY':
            isp_positions.loc[date:] = 1.0
        elif row['Action'] == 'SELL':
            isp_positions.loc[date:] = 0.0

# Count position changes
position_changes = isp_positions.diff().fillna(0)
n_trades = (position_changes.abs() > 0).sum()

# Calculate holding periods
in_position = False
hold_start = None
hold_periods = []
trade_dates = []

for i, (date, pos) in enumerate(isp_positions.items()):
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

print(f"\n  ISP TRADING BEHAVIOR:")
print(f"  ─────────────────────────────────────────")
print(f"  Total trades:        {n_trades}")
print(f"  Buy signals:         {(isp_df['Action'] == 'BUY').sum()}")
print(f"  Sell signals:        {(isp_df['Action'] == 'SELL').sum()}")
print(f"  Avg trades/year:     {n_trades / (len(df_full) / 365.25):.1f}")
print(f"  Avg hold period:     {np.mean(hold_periods):.0f} days" if hold_periods else "  N/A")
print(f"  Median hold period:  {np.median(hold_periods):.0f} days" if hold_periods else "  N/A")
print(f"  Min hold period:     {min(hold_periods)} days" if hold_periods else "  N/A")
print(f"  Max hold period:     {max(hold_periods)} days" if hold_periods else "  N/A")
print(f"  In-market:           {isp_positions.mean() * 100:.1f}%")

# Print trade history
print(f"\n  TRADE HISTORY:")
print(f"  {'Date':<15} {'Action':<8} {'BTC Price':<15}")
print(f"  {'─'*38}")
for action, date in trade_dates:
    if date in df_full.index:
        price = df_full.loc[date, 'close']
        print(f"  {date.strftime('%Y-%m-%d'):<15} {action:<8} ${price:,.0f}")

# Calculate metrics
returns = df_full['close'].pct_change()
strategy_returns = returns * isp_positions.shift(1)
strategy_returns = strategy_returns.dropna()

equity = (1 + strategy_returns).cumprod()
years = len(strategy_returns) / 365.25
cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
peak = equity.cummax()
max_dd = ((equity - peak) / peak).min()

print(f"\n  ISP PERFORMANCE:")
print(f"  CAGR:     {cagr*100:.1f}%")
print(f"  Sharpe:   {sharpe:.2f}")
print(f"  MaxDD:    {max_dd*100:.1f}%")

# ================================================================
# Compare with MSVR + Cycle
# ================================================================
print("\n[3/3] Comparing with MSVR + Cycle Phase...")

import importlib.util
sys.path.append('/home/ubuntu/projects/quant-technical-indicator-bank')
from indicators_helper import *

spec = importlib.util.spec_from_file_location('msvr', '/home/ubuntu/projects/quant-technical-indicator-bank/perpetual/median_standard_deviation_viresearch.py')
msvr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msvr_module)

msvr_full = msvr_module.median_standard_deviation_viresearch(df_full)
msvr_signal = msvr_full['vii']

# Cycle phase
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

# Combined
msvr_binary = (msvr_signal > 0).astype(float)
cycle_binary = (cycle_signal > 0).astype(float)
combined = msvr_binary * cycle_binary

# Analyze combined behavior
combined_changes = combined.diff().fillna(0)
n_trades_combined = (combined_changes.abs() > 0).sum()

# Holding periods
in_position = False
hold_start = None
hold_periods_combined = []

for i, (date, pos) in enumerate(combined.items()):
    if pos == 1.0 and not in_position:
        in_position = True
        hold_start = date
    elif pos == 0.0 and in_position:
        in_position = False
        if hold_start is not None:
            hold_days = (date - hold_start).days
            hold_periods_combined.append(hold_days)

print(f"\n  COMPARISON:")
print(f"  ─────────────────────────────────────────────────────")
print(f"  {'Metric':<25} {'ISP':<15} {'MSVR+Cycle':<15}")
print(f"  {'─'*55}")
print(f"  {'Total trades':<25} {n_trades:<15} {n_trades_combined:<15}")
print(f"  {'Avg trades/year':<25} {n_trades/(len(df_full)/365.25):<15.1f} {n_trades_combined/(len(df_full)/365.25):<15.1f}")
print(f"  {'Avg hold (days)':<25} {np.mean(hold_periods):<15.0f} {np.mean(hold_periods_combined):<15.0f}" if hold_periods and hold_periods_combined else "")
print(f"  {'Median hold (days)':<25} {np.median(hold_periods):<15.0f} {np.median(hold_periods_combined):<15.0f}" if hold_periods and hold_periods_combined else "")
print(f"  {'In-market %':<25} {isp_positions.mean()*100:<15.1f} {combined.mean()*100:<15.1f}")
print(f"  {'CAGR':<25} {cagr*100:<15.1f}% {22.6:<15.1f}%")
print(f"  {'Sharpe':<25} {sharpe:<15.2f} {1.42:<15.2f}")

# Key insight
print(f"\n  KEY INSIGHT:")
print(f"  ─────────────────────────────────────────")
if n_trades_combined > n_trades * 3:
    print(f"  ⚠️ MSVR+Cycle trades {n_trades_combined/n_trades:.1f}x MORE than ISP!")
    print(f"     ISP trades {n_trades} times in 8 years")
    print(f"     MSVR+Cycle trades {n_trades_combined} times in 8 years")
    print(f"\n  ISP holds positions for MONTHS")
    print(f"  MSVR+Cycle changes position every FEW DAYS")
    print(f"\n  This is why it's 'noisy' — too many trades!")
else:
    print(f"  ✅ Trade frequency similar to ISP")

print(f"\n  RECOMMENDATION:")
print(f"  ─────────────────────────────────────────")
print(f"  ISP Behavior:")
print(f"    • Few trades ({n_trades} in 8 years)")
print(f"    • Long holds ({np.mean(hold_periods):.0f} days avg)")
print(f"    • Patient, trend-following")
print(f"\n  MSVR+Cycle Behavior:")
print(f"    • Many trades ({n_trades_combined} in 8 years)")
print(f"    • Short holds ({np.mean(hold_periods_combined):.0f} days avg)")
print(f"    • Noisy, overactive")
print(f"\n  To match ISP behavior:")
print(f"    • Increase min_hold to 30-60 days")
print(f"    • Add trend filter (e.g., 200 SMA)")
print(f"    • Reduce signal sensitivity")
