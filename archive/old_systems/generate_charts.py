#!/usr/bin/env python3
"""
Generate MTTD System Performance Charts
========================================

Creates a 4-panel performance chart for the MTTD system.
Loads mttd/signals.csv and mttd/equity.csv, or generates sample data if missing.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Project root
project_root = os.path.dirname(os.path.abspath(__file__))
mttd_dir = os.path.join(project_root, 'mttd')

# Ensure mttd directory exists
os.makedirs(mttd_dir, exist_ok=True)

def load_or_generate_data():
    """Load signals.csv and equity.csv, or generate from mttd_data.json."""
    signals_path = os.path.join(mttd_dir, 'signals.csv')
    equity_path = os.path.join(mttd_dir, 'equity.csv')
    metrics_path = os.path.join(mttd_dir, 'metrics.json')
    
    # Check if files exist
    if os.path.exists(signals_path) and os.path.exists(equity_path):
        print(f"Loading existing data from {mttd_dir}")
        signals_df = pd.read_csv(signals_path, index_col='date', parse_dates=True)
        equity_df = pd.read_csv(equity_path, index_col='date', parse_dates=True)
    else:
        print("Generating sample data from mttd_data.json...")
        # Load mttd_data.json
        mttd_data_path = os.path.join(mttd_dir, 'mttd_data.json')
        if not os.path.exists(mttd_data_path):
            print(f"Error: {mttd_data_path} not found")
            sys.exit(1)
        
        with open(mttd_data_path, 'r') as f:
            mttd_data = json.load(f)
        
        # Convert candles to DataFrame
        candles = pd.DataFrame(mttd_data['candles'])
        candles['time'] = pd.to_datetime(candles['time'])
        candles = candles.set_index('time')
        
        # Generate sample signals (simple moving average crossover)
        signals = pd.Series(0.0, index=candles.index)
        sma_short = candles['close'].rolling(20).mean()
        sma_long = candles['close'].rolling(50).mean()
        
        # Buy when short > long, sell when short < long
        in_position = False
        for i in range(50, len(candles)):
            if sma_short.iloc[i] > sma_long.iloc[i] and not in_position:
                signals.iloc[i] = 1.0  # Buy
                in_position = True
            elif sma_short.iloc[i] < sma_long.iloc[i] and in_position:
                signals.iloc[i] = 0.0  # Sell
                in_position = False
            else:
                signals.iloc[i] = 1.0 if in_position else 0.0
        
        # Forward fill signals
        signals = signals.replace(0, np.nan).ffill().fillna(0)
        
        # Create signals DataFrame
        signals_df = pd.DataFrame({
            'signal': signals,
            'price': candles['close']
        }, index=candles.index)
        
        # Calculate equity curve
        returns = candles['close'].pct_change()
        strategy_returns = returns * signals.shift(1)
        equity_curve = (1 + strategy_returns).cumprod()
        equity_curve.iloc[0] = 1.0  # Start at 1.0
        
        equity_df = pd.DataFrame({
            'equity': equity_curve,
            'drawdown': equity_curve / equity_curve.cummax() - 1
        }, index=candles.index)
        
        # Save to CSV
        signals_df.to_csv(signals_path)
        equity_df.to_csv(equity_path)
        print(f"Generated and saved sample data to {mttd_dir}")
        
        # Generate metrics.json
        metrics = {
            'total_return': float((equity_curve.iloc[-1] - 1) * 100),
            'cagr': float(((equity_curve.iloc[-1]) ** (365.25 / len(candles)) - 1) * 100),
            'sharpe': float(strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0),
            'max_drawdown': float(equity_df['drawdown'].min() * 100),
            'win_rate': float((strategy_returns > 0).sum() / (strategy_returns != 0).sum() * 100 if (strategy_returns != 0).sum() > 0 else 0),
            'total_trades': int((signals.diff().abs() > 0).sum()),
            'avg_trade_duration': float((signals.diff().abs() > 0).sum() / 2),
            'start_date': str(candles.index[0].date()),
            'end_date': str(candles.index[-1].date())
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Generated metrics.json")
    
    return signals_df, equity_df

def generate_chart(signals_df, equity_df):
    """Generate 4-panel performance chart."""
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    
    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle('MTTD System Performance', fontsize=16, fontweight='bold')
    
    # Panel 1: BTC Price with Buy/Sell markers
    ax1 = axes[0]
    ax1.plot(signals_df.index, signals_df['btc_price'], color='#2196F3', linewidth=1.5, label='BTC Price', alpha=0.9)
    
    # Mark buy/sell signals
    buy_signals = signals_df[signals_df['position'].diff() > 0]
    sell_signals = signals_df[signals_df['position'].diff() < 0]
    
    ax1.scatter(buy_signals.index, buy_signals['btc_price'], marker='^', color='#4CAF50', s=100, label='Buy', zorder=5)
    ax1.scatter(sell_signals.index, sell_signals['btc_price'], marker='v', color='#F44336', s=100, label='Sell', zorder=5)
    
    ax1.set_ylabel('BTC Price (USD)')
    ax1.legend(loc='upper left')
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add vertical dashed line for holdout boundary
    holdout_date = pd.Timestamp('2025-01-01')
    if holdout_date in signals_df.index:
        ax1.axvline(x=holdout_date, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.text(holdout_date, ax1.get_ylim()[1]*0.95, '← Training | Holdout →', 
                 ha='center', fontsize=10, color='red', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))
    
    # Panel 2: Equity curve
    ax2 = axes[1]
    ax2.plot(equity_df.index, equity_df['equity'], color='#4CAF50', linewidth=2, label='Equity Curve')
    ax2.axhline(y=1.0, color='black', linewidth=0.5, linestyle=':')
    ax2.set_ylabel('Equity (1.0 = start)')
    ax2.legend(loc='upper left')
    
    # Panel 3: Drawdown
    ax3 = axes[2]
    ax3.fill_between(equity_df.index, equity_df['drawdown'], 0, color='#F44336', alpha=0.5)
    ax3.plot(equity_df.index, equity_df['drawdown'], color='#F44336', linewidth=1)
    ax3.set_ylabel('Drawdown')
    ax3.set_ylim(-1, 0)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
    
    # Panel 4: Monthly returns
    ax4 = axes[3]
    monthly_returns = equity_df['equity'].resample('ME').last().pct_change().dropna()
    colors = ['#4CAF50' if x > 0 else '#F44336' for x in monthly_returns]
    ax4.bar(monthly_returns.index, monthly_returns, color=colors, alpha=0.7, width=20)
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_ylabel('Monthly Returns')
    ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
    
    # Format x-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save chart
    chart_path = os.path.join(mttd_dir, 'system_performance.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Chart saved to {chart_path}")
    
    return chart_path

def print_summary(metrics_path):
    """Print summary metrics."""
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        
        print("\n" + "="*50)
        print("MTTD SYSTEM PERFORMANCE SUMMARY")
        print("="*50)
        
        # Handle different metrics structures
        if 'performance' in metrics:
            perf = metrics['performance']
            print(f"Sharpe Ratio: {perf.get('sharpe', 'N/A'):.2f}")
            print(f"CAGR: {perf.get('cagr', 'N/A'):.2f}%")
            print(f"Sortino Ratio: {perf.get('sortino', 'N/A'):.2f}")
            print(f"Calmar Ratio: {perf.get('calmar', 'N/A'):.2f}")
            print(f"Max Drawdown: {perf.get('max_dd', 'N/A'):.2f}%")
            print(f"Win Rate: {perf.get('win_rate', 'N/A'):.2f}%")
            print(f"Total Trades: {perf.get('n_trades', 'N/A')}")
            print(f"Average Hold: {perf.get('avg_hold', 'N/A'):.0f} days")
        else:
            # Fallback to flat structure
            print(f"Total Return: {metrics.get('total_return', 'N/A'):.2f}%")
            print(f"CAGR: {metrics.get('cagr', 'N/A'):.2f}%")
            print(f"Sharpe Ratio: {metrics.get('sharpe', 'N/A'):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.2f}%")
            print(f"Win Rate: {metrics.get('win_rate', 'N/A'):.2f}%")
            print(f"Total Trades: {metrics.get('total_trades', 'N/A')}")
        
        if 'config' in metrics:
            print(f"\nConfiguration:")
            config = metrics['config']
            for key, value in config.items():
                print(f"  {key}: {value}")
        
        print("="*50)
    else:
        print("Metrics file not found")

if __name__ == "__main__":
    print("Generating MTTD System Performance Charts...")
    
    # Load or generate data
    signals_df, equity_df = load_or_generate_data()
    
    # Generate chart
    chart_path = generate_chart(signals_df, equity_df)
    
    # Print summary
    metrics_path = os.path.join(mttd_dir, 'metrics.json')
    print_summary(metrics_path)
    
    print(f"\nChart generated successfully: {chart_path}")