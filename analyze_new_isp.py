#!/usr/bin/env python3
"""
Analyze NEW ISP Signals (2026-06-21)
=====================================

New data from user with more recent signals.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ANALYZING NEW ISP SIGNALS (2026-06-21)")
print("=" * 70)

# ================================================================
# Load NEW ISP Data
# ================================================================
print("\n[1/3] Loading new ISP data...")

project_root = os.path.dirname(os.path.abspath(__file__))

# New ISP data from user
new_isp_data = """Date,Action,Price,EquityPct,Cost,BTCHeld,TotalEquity
2019-03-26,BUY,3930.73,100,10000.00,2.54405670,10000.00
2019-07-03,SELL,11898.21,100,30269.72,0.00000000,30269.72
2020-01-06,BUY,7753.30,100,30269.72,3.90410804,30269.72
2020-02-23,SELL,9959.72,100,38883.82,0.00000000,38883.82
2020-04-25,BUY,7514.18,100,38883.82,5.17472605,38883.82
2020-08-18,SELL,12031.95,100,62262.05,0.00000000,62262.05
2020-10-12,BUY,11624.51,100,62262.05,5.35610061,62262.05
2021-01-14,SELL,38905.95,100,208384.18,0.00000000,208384.18
2021-02-02,BUY,35559.80,100,208384.18,5.86010558,208384.18
2021-04-20,SELL,56408.91,100,330562.17,0.00000000,330562.17
2021-07-23,BUY,33172.73,100,330562.17,9.96487682,330562.17
2021-09-08,SELL,46477.75,100,463145.05,0.00000000,463145.05
2021-10-01,BUY,47957.60,100,463145.05,9.65738597,463145.05
2021-11-17,SELL,60052.41,100,579949.30,0.00000000,579949.30
2023-01-11,BUY,17580.22,100,579949.30,32.98873973,579949.30
2023-02-23,SELL,23831.85,100,786182.70,0.00000000,786182.70
2023-03-13,BUY,24195.31,100,786182.70,32.49318554,786182.70
2023-04-22,SELL,27846.12,100,904809.14,0.00000000,904809.14
2023-06-18,BUY,26314.84,100,904809.14,34.38398803,904809.14
2023-07-20,SELL,29843.27,100,1026130.64,0.00000000,1026130.64
2023-09-17,BUY,26468.92,100,1026130.64,38.76737844,1026130.64
2024-03-26,SELL,69993.66,100,2713470.71,0.00000000,2713470.71
2024-05-17,BUY,66829.74,100,2713470.71,40.60274222,2713470.71
2024-06-12,SELL,68324.24,100,2774151.50,0.00000000,2774151.50
2024-07-14,BUY,60967.75,100,2774151.50,45.50194986,2774151.50
2024-08-01,SELL,65104.83,100,2962396.71,0.00000000,2962396.71
2024-10-14,BUY,66063.83,100,2962396.71,44.84143154,2962396.71
2024-12-21,SELL,96803.61,100,4340812.45,0.00000000,4340812.45
2025-04-20,BUY,85059.54,100,4340812.45,51.03263492,4340812.45
2025-06-01,SELL,105821.35,100,5400342.32,0.00000000,5400342.32
2025-07-07,BUY,108309.63,100,5400342.32,49.86022315,5400342.32
2025-07-31,SELL,116120.39,100,5789788.56,0.00000000,5789788.56
2026-04-08,BUY,70931.44,100,5789788.56,81.62513770,5789788.56
2026-05-15,SELL,79136.63,100,6459538.32,0.00000000,6459538.32"""

# Save to file
new_isp_path = os.path.join(project_root, 'isp-signals-btcusd-2026-06-21.csv')
with open(new_isp_path, 'w') as f:
    f.write(new_isp_data)

# Load
new_isp_df = pd.read_csv(new_isp_path)
new_isp_df['Date'] = pd.to_datetime(new_isp_df['Date'])
new_isp_df = new_isp_df.set_index('Date')

print(f"  New ISP signals: {len(new_isp_df)} entries")
print(f"  BUY signals:  {(new_isp_df['Action'] == 'BUY').sum()}")
print(f"  SELL signals: {(new_isp_df['Action'] == 'SELL').sum()}")

# ================================================================
# Analyze ISP Behavior
# ================================================================
print("\n[2/3] Analyzing ISP behavior...")

# Load BTC data
with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
    btc_data = json.load(f)

df_full = pd.DataFrame(btc_data['aligned_data'])
df_full['time'] = pd.to_datetime(df_full['time'])
df_full = df_full.set_index('time')
df_full = df_full[df_full.index >= '2018-01-01']

# Create position series
isp_positions = pd.Series(0.0, index=df_full.index)
for date, row in new_isp_df.iterrows():
    if date in isp_positions.index:
        if row['Action'] == 'BUY':
            isp_positions.loc[date:] = 1.0
        elif row['Action'] == 'SELL':
            isp_positions.loc[date:] = 0.0

# Count trades and holding periods
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

n_trades = len(trade_dates)

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

print(f"\n  ISP TRADING BEHAVIOR:")
print(f"  ─────────────────────────────────────────")
print(f"  Total trades:        {n_trades}")
print(f"  Buy signals:         {(new_isp_df['Action'] == 'BUY').sum()}")
print(f"  Sell signals:        {(new_isp_df['Action'] == 'SELL').sum()}")
print(f"  Avg trades/year:     {n_trades / (len(df_full) / 365.25):.1f}")
print(f"  Avg hold period:     {np.mean(hold_periods):.0f} days" if hold_periods else "  N/A")
print(f"  Median hold period:  {np.median(hold_periods):.0f} days" if hold_periods else "  N/A")
print(f"  Min hold period:     {min(hold_periods)} days" if hold_periods else "  N/A")
print(f"  Max hold period:     {max(hold_periods)} days" if hold_periods else "  N/A")
print(f"  In-market:           {isp_positions.mean() * 100:.1f}%")

# Print trade history
print(f"\n  TRADE HISTORY:")
print(f"  {'Date':<15} {'Action':<8} {'BTC Price':<15} {'Hold Days':<12} {'Return':<10}")
print(f"  {'─'*60}")

prev_price = None
prev_date = None
for action, date in trade_dates:
    if date in df_full.index:
        price = df_full.loc[date, 'close']
        if action == 'SELL' and prev_price is not None:
            ret = (price - prev_price) / prev_price * 100
            hold = (date - prev_date).days
            print(f"  {date.strftime('%Y-%m-%d'):<15} {action:<8} ${price:,.0f}      {hold:<12} {ret:+.1f}%")
        else:
            print(f"  {date.strftime('%Y-%m-%d'):<15} {action:<8} ${price:,.0f}")
        prev_price = price
        prev_date = date

# Final equity
final_equity = new_isp_df.iloc[-1]['TotalEquity']
initial_equity = new_isp_df.iloc[0]['TotalEquity']
total_return = (final_equity - initial_equity) / initial_equity * 100

print(f"\n  PERFORMANCE:")
print(f"  ─────────────────────────────────────────")
print(f"  Initial equity:   ${initial_equity:,.2f}")
print(f"  Final equity:     ${final_equity:,.2f}")
print(f"  Total return:     {total_return:,.1f}%")
print(f"  CAGR:             {cagr*100:.1f}%")
print(f"  Sharpe:           {sharpe:.2f}")
print(f"  Max Drawdown:     {max_dd*100:.1f}%")

# ================================================================
# Compare with old ISP
# ================================================================
print("\n[3/3] Comparing with old ISP...")

old_isp_path = os.path.join(project_root, 'isp-signals-btcusd-2026-06-13.csv')
old_isp_df = pd.read_csv(old_isp_path)
old_isp_df['Date'] = pd.to_datetime(old_isp_df['Date'])
old_isp_df = old_isp_df.set_index('Date')

print(f"\n  COMPARISON:")
print(f"  ─────────────────────────────────────────")
print(f"  {'Metric':<25} {'Old ISP':<15} {'New ISP':<15}")
print(f"  {'─'*55}")
print(f"  {'Total signals':<25} {len(old_isp_df):<15} {len(new_isp_df):<15}")
print(f"  {'Buy signals':<25} {(old_isp_df['Action'] == 'BUY').sum():<15} {(new_isp_df['Action'] == 'BUY').sum():<15}")
print(f"  {'Sell signals':<25} {(old_isp_df['Action'] == 'SELL').sum():<15} {(new_isp_df['Action'] == 'SELL').sum():<15}")
print(f"  {'Date range':<25} {'2019-2025':<15} {'2019-2026':<15}")
print(f"  {'Final equity':<25} {'N/A':<15} ${final_equity:,.0f}")
