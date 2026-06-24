#!/usr/bin/env python3
"""
Generate Comparison Equity Curve Chart
=======================================
Publication-quality chart showing MSVR Hybrid equity curve alongside Ichimoku,
with trade entry/exit markers.

Output: mttd/msvr_hybrid_comparison.png (1200x800)
"""

import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# Import modules
import sys
sys.path.append('/home/ubuntu/projects/quant-technical-indicator-bank')
sys.path.append('/home/ubuntu/projects/quant-btc-mttd-system')

from msvr_hybrid import (
    load_btc_data, generate_composite_signal, enforce_min_hold,
    compute_trade_list, compute_metrics
)
from ichimoku_quant import (
    generate_ichimoku_features, generate_ichimoku_signals,
    compute_ichimoku_metrics
)


def compute_equity_curve(positions: pd.Series, prices: pd.Series) -> pd.Series:
    """Compute equity curve from position series and prices."""
    daily_returns = prices.pct_change().fillna(0)
    strategy_returns = daily_returns * positions.shift(1).fillna(0)
    equity = (1 + strategy_returns).cumprod()
    return equity


def plot_comparison_chart(hybrid_equity, ichimoku_equity, hybrid_trades, 
                          btc_prices, hybrid_positions, ichimoku_positions,
                          output_path='mttd/msvr_hybrid_comparison.png'):
    """
    Generate publication-quality comparison chart.
    """
    # Style configuration
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Color palette - professional, high contrast
    COLORS = {
        'hybrid': '#2196F3',        # Blue for hybrid
        'ichimoku': '#FF9800',      # Orange for ichimoku
        'btc': '#9E9E9E',           # Gray for BTC buy-and-hold
        'entry': '#4CAF50',         # Green for entry
        'exit': '#F44336',          # Red for exit
        'bg': '#FAFAFA',            # Light background
        'grid': '#E0E0E0',          # Grid color
        'text': '#212121',          # Dark text
        'subtitle': '#757575'       # Subtitle color
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor(COLORS['bg'])
    
    # Main equity curve axes
    ax1 = axes[0]
    ax1.set_facecolor('#FFFFFF')
    
    # Performance comparison axes
    ax2 = axes[1]
    ax2.set_facecolor('#FFFFFF')
    
    # --- Plot 1: Equity Curves ---
    
    # Normalize all curves to start at 1
    hybrid_eq_norm = hybrid_equity / hybrid_equity.iloc[0]
    ichimoku_eq_norm = ichimoku_equity / ichimoku_equity.iloc[0]
    btc_eq_norm = btc_prices / btc_prices.iloc[0]
    
    # Plot equity curves with gradient fill
    ax1.fill_between(hybrid_eq_norm.index, 1, hybrid_eq_norm, 
                      alpha=0.15, color=COLORS['hybrid'], label='_nolegend_')
    ax1.plot(hybrid_eq_norm.index, hybrid_eq_norm, 
             color=COLORS['hybrid'], linewidth=2.5, label='MSVR Hybrid', zorder=5)
    
    ax1.plot(ichimoku_eq_norm.index, ichimoku_eq_norm, 
             color=COLORS['ichimoku'], linewidth=2.0, label='Ichimoku', 
             linestyle='--', alpha=0.85, zorder=4)
    
    ax1.plot(btc_eq_norm.index, btc_eq_norm, 
             color=COLORS['btc'], linewidth=1.5, label='BTC Buy & Hold', 
             linestyle=':', alpha=0.6, zorder=3)
    
    # --- Add Trade Markers ---
    
    # Track entry/exit positions for hybrid
    in_position = False
    entry_date = None
    
    for i, (date, pos) in enumerate(hybrid_positions.items()):
        if pos == 1 and not in_position:
            in_position = True
            entry_date = date
            # Entry marker (triangle up)
            eq_val = hybrid_eq_norm.loc[date] if date in hybrid_eq_norm.index else None
            if eq_val is not None:
                ax1.scatter(date, eq_val, marker='^', s=120, 
                           color=COLORS['entry'], edgecolors='white', linewidth=1.5,
                           zorder=10, label='Entry' if i == 0 else '_nolegend_')
        elif pos == 0 and in_position:
            in_position = False
            # Exit marker (triangle down)
            eq_val = hybrid_eq_norm.loc[date] if date in hybrid_eq_norm.index else None
            if eq_val is not None:
                ax1.scatter(date, eq_val, marker='v', s=120, 
                           color=COLORS['exit'], edgecolors='white', linewidth=1.5,
                           zorder=10, label='Exit' if i < 10 else '_nolegend_')
    
    # Close any open position
    if in_position:
        last_date = hybrid_positions.index[-1]
        eq_val = hybrid_eq_norm.iloc[-1]
        ax1.scatter(last_date, eq_val, marker='v', s=120, 
                   color=COLORS['exit'], edgecolors='white', linewidth=1.5,
                   zorder=10)
    
    # --- Styling ---
    
    ax1.set_title('MSVR Hybrid vs Ichimoku Equity Curve Comparison', 
                  fontsize=18, fontweight='bold', color=COLORS['text'], pad=20)
    ax1.set_ylabel('Normalized Equity (1x = Start)', fontsize=13, color=COLORS['text'])
    
    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.tick_params(axis='x', labelsize=11)
    ax1.tick_params(axis='y', labelsize=11)
    
    # Grid
    ax1.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='-')
    ax1.set_axisbelow(True)
    
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color(COLORS['grid'])
    ax1.spines['bottom'].set_color(COLORS['grid'])
    
    # Add legend with metrics
    hybrid_metrics = compute_metrics(hybrid_trades, 
                                      positions=hybrid_positions, 
                                      df_prices=btc_prices)
    
    ichimoku_metrics = compute_ichimoku_metrics(
        pd.DataFrame({'Pos': ichimoku_positions}), btc_prices
    )
    
    legend_text = [
        f'MSVR Hybrid: {hybrid_metrics["n_trades"]} trades, '
        f'{hybrid_metrics["win_rate"]}% win, '
        f'Sharpe {hybrid_metrics["sharpe"]:.2f}',
        f'Ichimoku: {ichimoku_metrics["n_trades"]} trades, '
        f'{ichimoku_metrics["win_rate"]}% win, '
        f'Sharpe {ichimoku_metrics["sharpe"]:.2f}'
    ]
    
    # Custom legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Line2D([0], [0], color=COLORS['hybrid'], linewidth=2.5, label='MSVR Hybrid'),
        Line2D([0], [0], color=COLORS['ichimoku'], linewidth=2.0, 
               linestyle='--', label='Ichimoku'),
        Line2D([0], [0], color=COLORS['btc'], linewidth=1.5, 
               linestyle=':', label='BTC Buy & Hold'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=COLORS['entry'],
               markersize=10, label='Entry', markeredgecolor='white'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor=COLORS['exit'],
               markersize=10, label='Exit', markeredgecolor='white'),
    ]
    
    legend1 = ax1.legend(handles=legend_elements, loc='upper left', fontsize=10,
                         framealpha=0.95, edgecolor=COLORS['grid'])
    legend1.get_frame().set_facecolor('#FFFFFF')
    
    # Add text box with metrics
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.95, 
                 edgecolor=COLORS['grid'])
    textstr = '\n'.join(legend_text)
    ax1.text(0.98, 0.35, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right', 
             bbox=props, family='monospace', color=COLORS['text'])
    
    # --- Plot 2: Drawdown Comparison ---
    
    # Compute drawdowns
    def compute_drawdown(equity):
        peak = equity.cummax()
        dd = (equity - peak) / peak
        return dd
    
    hybrid_dd = compute_drawdown(hybrid_eq_norm)
    ichimoku_dd = compute_drawdown(ichimoku_eq_norm)
    
    ax2.fill_between(hybrid_dd.index, 0, hybrid_dd * 100, 
                      alpha=0.4, color=COLORS['hybrid'], label='Hybrid DD')
    ax2.fill_between(ichimoku_dd.index, 0, ichimoku_dd * 100, 
                      alpha=0.4, color=COLORS['ichimoku'], label='Ichimoku DD')
    
    ax2.set_title('Drawdown Comparison', fontsize=12, fontweight='bold', 
                  color=COLORS['text'], pad=10)
    ax2.set_ylabel('Drawdown (%)', fontsize=11, color=COLORS['text'])
    ax2.set_xlabel('Date', fontsize=11, color=COLORS['text'])
    
    # Format x-axis
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.tick_params(axis='x', labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)
    
    # Grid
    ax2.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='-')
    ax2.set_axisbelow(True)
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color(COLORS['grid'])
    ax2.spines['bottom'].set_color(COLORS['grid'])
    
    # Legend for drawdown
    legend2 = ax2.legend(loc='lower left', fontsize=9, framealpha=0.95, 
                         edgecolor=COLORS['grid'])
    legend2.get_frame().set_facecolor('#FFFFFF')
    
    # Add footer
    fig.text(0.5, 0.01, 
             'MSVR Hybrid: Combines MSVR Direction + Cycle Phase + SuperSmoother + Entropy + Efficiency Ratio | '
             'Min Hold: 10 days | Transaction Cost: 0.1%',
             ha='center', fontsize=8, color=COLORS['subtitle'], style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    
    print(f"Chart saved to: {output_path}")
    return output_path


def main():
    print("=" * 70)
    print("GENERATING COMPARISON EQUITY CURVE CHART")
    print("=" * 70)
    
    # 1. Load data
    print("\n1. Loading BTC data...")
    df = load_btc_data()
    print(f"   Data: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
    
    # 2. Generate MSVR Hybrid signals
    print("\n2. Generating MSVR Hybrid signals...")
    df_hybrid = generate_composite_signal(df.copy())
    df_hybrid['position'] = enforce_min_hold(df_hybrid['composite_signal'], min_hold=10)
    hybrid_trades = compute_trade_list(df_hybrid, df_hybrid['close'], transaction_cost=0.001)
    hybrid_metrics = compute_metrics(hybrid_trades, 
                                      positions=df_hybrid['position'], 
                                      df_prices=df_hybrid['close'])
    print(f"   Trades: {hybrid_metrics['n_trades']}, Win Rate: {hybrid_metrics['win_rate']}%, "
          f"Sharpe: {hybrid_metrics['sharpe']}")
    
    # 3. Generate Ichimoku signals
    print("\n3. Generating Ichimoku signals...")
    df_ichimoku = generate_ichimoku_features(df.copy())
    df_ichimoku = generate_ichimoku_signals(df_ichimoku)
    ichimoku_metrics = compute_ichimoku_metrics(df_ichimoku, df_ichimoku['close'])
    print(f"   Trades: {ichimoku_metrics['n_trades']}, Win Rate: {ichimoku_metrics['win_rate']}%, "
          f"Sharpe: {ichimoku_metrics['sharpe']}")
    
    # 4. Compute equity curves
    print("\n4. Computing equity curves...")
    hybrid_equity = compute_equity_curve(df_hybrid['position'], df_hybrid['close'])
    ichimoku_equity = compute_equity_curve(df_ichimoku['Pos'], df_ichimoku['close'])
    print(f"   Hybrid final equity: {hybrid_equity.iloc[-1]:.2f}x")
    print(f"   Ichimoku final equity: {ichimoku_equity.iloc[-1]:.2f}x")
    
    # 5. Generate chart
    print("\n5. Generating chart...")
    output_path = plot_comparison_chart(
        hybrid_equity=hybrid_equity,
        ichimoku_equity=ichimoku_equity,
        hybrid_trades=hybrid_trades,
        btc_prices=df_hybrid['close'],
        hybrid_positions=df_hybrid['position'],
        ichimoku_positions=df_ichimoku['Pos'],
        output_path='mttd/msvr_hybrid_comparison.png'
    )
    
    # 6. Verify
    import os
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"\n✓ Chart saved successfully!")
        print(f"  Path: {output_path}")
        print(f"  Size: {file_size / 1024:.1f} KB")
    else:
        print(f"\n✗ Failed to save chart!")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
