#!/usr/bin/env python3
"""
MTTD Chart Generator
Generates a comprehensive chart with:
1. BTC Price + Buy/Sell markers
2. Ensemble signal
3. Individual indicators (10 subplots)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

def load_data():
    """Load BTC data and MTTD signals."""
    # Load BTC data
    with open('data/btc_daily.json') as f:
        btc_data = json.load(f)
    
    btc = pd.DataFrame(btc_data['aligned_data'])
    btc['time'] = pd.to_datetime(btc['time'])
    btc = btc.set_index('time')
    btc = btc[btc.index >= '2018-01-01']
    
    # Load MTTD data
    with open('mttd_data.json') as f:
        mttd_data = json.load(f)
    
    return btc, mttd_data

def plot_chart():
    """Generate comprehensive MTTD chart."""
    print("Loading data...")
    btc, mttd_data = load_data()
    
    print("Creating chart...")
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(12, 1, height_ratios=[3, 1] + [1]*10, hspace=0.3)
    
    # 1. BTC Price + Ensemble Signals
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(btc.index, btc['close'], color='#2196F3', linewidth=1.5, label='BTC Price')
    ax1.set_ylabel('BTC Price ($)', fontsize=10)
    ax1.set_title('MTTD Ensemble Trading System - BTC/USD', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add buy/sell markers
    markers = mttd_data.get('aggregate', {}).get('markers', [])
    buy_dates = [pd.to_datetime(m['time']) for m in markers if m['text'] == 'BUY']
    buy_prices = [btc.loc[d, 'close'] if d in btc.index else np.nan for d in buy_dates]
    sell_dates = [pd.to_datetime(m['time']) for m in markers if m['text'] == 'SELL']
    sell_prices = [btc.loc[d, 'close'] if d in btc.index else np.nan for d in sell_dates]
    
    ax1.scatter(buy_dates, buy_prices, color='#10b981', marker='^', s=100, zorder=5, label='BUY')
    ax1.scatter(sell_dates, sell_prices, color='#f43f5e', marker='v', s=100, zorder=5, label='SELL')
    ax1.legend(loc='upper left')
    
    # 2. Ensemble Signal
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    signals = mttd_data.get('aggregate', {}).get('signals', [])
    if signals:
        sig_dates = [pd.to_datetime(s['time']) for s in signals]
        sig_values = [s['value'] for s in signals]
        ax2.fill_between(sig_dates, sig_values, alpha=0.5, color='#9C27B0')
        ax2.plot(sig_dates, sig_values, color='#9C27B0', linewidth=1)
    ax2.set_ylabel('Ensemble', fontsize=10)
    ax2.set_ylim(-0.1, 1.1)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # 3-12. Individual Indicators
    indicators = mttd_data.get('indicators', {})
    indicator_names = list(indicators.keys())[:10]
    
    for i, ind_name in enumerate(indicator_names):
        ax = fig.add_subplot(gs[i+2], sharex=ax1)
        ind_data = indicators[ind_name]
        
        # Plot indicator values
        values = ind_data.get('values', [])
        if values:
            val_dates = [pd.to_datetime(v['time']) for v in values]
            val_values = [v['value'] for v in values]
            colors = [v.get('color', '#2196F3') for v in values]
            
            # Use color based on signal
            for j in range(len(val_dates)-1):
                ax.plot(val_dates[j:j+2], val_values[j:j+2], color=colors[j], linewidth=1)
        
        # Shorten name for display
        short_name = ind_name[:30] + '...' if len(ind_name) > 30 else ind_name
        ax.set_ylabel(short_name, fontsize=8)
        ax.grid(True, alpha=0.3)
    
    ax1.set_xlabel('Date', fontsize=10)
    
    # Save chart
    output_path = 'mttd_chart.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    chart_path = plot_chart()
    print(f"\nDone! Chart saved to: {chart_path}")
