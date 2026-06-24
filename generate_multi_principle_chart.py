#!/usr/bin/env python3
"""
Generate Clean Trade Chart for Multi-Principle MTTD System
=============================================================
Clean chart: only entry (^) and exit (v) markers, no clutter.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import os
import warnings
warnings.filterwarnings('ignore')

from multi_principle_strategy import multi_principle_strategy, backtest, generate_ichimoku_features

# Load data
print("Loading data...")
with open('data/btc_daily.json') as f:
    btc_data = json.load(f)
df = pd.DataFrame(btc_data['aligned_data'])
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = df[df.index >= '2018-01-01']

# Best config
config = {
    't_entry': 0.25,
    'er_entry': 0.20,
    'entropy_thresh': 2.3,
    'min_hold_days': 10,
    'max_hold_days': 60,
    'chikou_thresh': -0.30,
    'immunity_thresh': 0.50,
    'imo_min_limit': -0.30,
    'imo_exit_bull': -0.30,
    'roc_gate_limit': -0.20,
    'cooldown': 5,
    'confirm_entry': 2,
    'confirm_exit': 1
}

print("Running strategy...")
result = multi_principle_strategy(df.copy(), **config)
metrics = backtest(result, df['close'])

# Extract trades
pos = result['Pos']
trades = []
in_pos = False
entry_date = None
entry_price = None

for i, (date, p) in enumerate(pos.items()):
    if p == 1.0 and not in_pos:
        in_pos = True
        entry_date = date
        entry_price = df['close'].loc[date]
    elif p == 0.0 and in_pos:
        in_pos = False
        exit_price = df['close'].loc[date]
        ret = (exit_price / entry_price - 1) * 100
        trades.append({
            'entry_date': entry_date,
            'exit_date': date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return': ret,
            'hold_days': (date - entry_date).days,
            'is_win': ret > 0
        })

n_win = sum(1 for t in trades if t['is_win'])
n_loss = len(trades) - n_win

print(f"Trades: {len(trades)}, Win: {n_win} ({n_win/len(trades)*100:.0f}%), Loss: {n_loss}")
print(f"Sharpe: {metrics['sharpe']}, CAGR: {metrics['cagr']}%")

# ================================================================
# GENERATE CLEAN CHART
# ================================================================
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#333355',
    'axes.labelcolor': '#888899',
    'text.color': '#ccccdd',
    'grid.color': '#222244',
    'grid.alpha': 0.3,
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10),
                                 gridspec_kw={'height_ratios': [3, 1]},
                                 sharex=True)

fig.suptitle(f'MTTD Multi-Principle — {len(trades)} trades, {metrics["win_rate"]:.0f}% win • Sharpe {metrics["sharpe"]:.2f} • CAGR {metrics["cagr"]:.0f}%',
             fontsize=14, fontweight='bold', color='#ccccdd', y=0.98)

# TOP PANEL: BTC Price
ax1.plot(df.index, df['close'], color='#4488ff', linewidth=0.8, alpha=0.7)

# Entry markers
entry_dates = [t['entry_date'] for t in trades]
entry_prices = [t['entry_price'] for t in trades]
ax1.scatter(entry_dates, entry_prices, marker='^', color='#fbbf24', s=80, zorder=10,
            edgecolors='#92400e', linewidth=0.5, label=f'Entry ({len(trades)})')

# Exit markers (neutral — single color)
exit_dates = [t['exit_date'] for t in trades]
exit_prices = [t['exit_price'] for t in trades]
ax1.scatter(exit_dates, exit_prices, marker='v', color='#ff6b6b', s=80, zorder=10,
            edgecolors='#cc4444', linewidth=0.5, label=f'Exit ({len(trades)})')

ax1.set_ylabel('BTC Price (USD)', fontsize=10, color='#888899')
ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
ax1.legend(loc='upper left', fontsize=9, framealpha=0.8, facecolor='#222244', edgecolor='#333355')

# Holdout boundary
holdout = pd.Timestamp('2025-01-01')
ax1.axvline(x=holdout, color='#ef4444', linestyle='--', linewidth=1, alpha=0.4)
ax1.text(holdout, ax1.get_ylim()[0]*1.05, 'HOLDOUT', rotation=90,
         fontsize=8, color='#ef4444', alpha=0.5, va='bottom')

# BOTTOM PANEL: Trade returns
trade_dates = [t['exit_date'] for t in trades]
trade_returns = [t['return'] for t in trades]
colors = ['#22c55e' if r > 0 else '#ef4444' for r in trade_returns]

bars = ax2.bar(trade_dates, trade_returns, color=colors, alpha=0.7, width=15)
ax2.axhline(y=0, color='#555577', linewidth=0.5)
ax2.set_ylabel('Return (%)', fontsize=10, color='#888899')

# Label big returns only
for bar, ret in zip(bars, trade_returns):
    if abs(ret) > 15:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., h, f'{ret:.0f}%',
                ha='center', va='bottom' if h > 0 else 'top',
                fontsize=7, color='#aaaabb')

ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_major_locator(mdates.YearLocator())

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save
os.makedirs('mttd/multi_principle', exist_ok=True)
path = 'mttd/multi_principle/trade_chart.png'
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Chart saved to {path}")

# Also save trade list
print("\nTrade List:")
for t in trades:
    emoji = '✅' if t['is_win'] else '❌'
    print(f"  {t['entry_date'].date()} → {t['exit_date'].date()} : {t['return']:+.1f}% ({t['hold_days']}d) {emoji}")
