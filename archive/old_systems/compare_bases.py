#!/usr/bin/env python3
"""
Compare Base Signals
====================

Loads all test results from base_signal_results.json, prints a formatted
comparison table, generates mttd/base_comparison.png, and identifies the
BEST base signal meeting success criteria.

Success Criteria:
- Sharpe > 1.35
- Win Rate > 60%
- 25-35 trades
- CAGR > 50%
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Project root
project_root = os.path.dirname(os.path.abspath(__file__))
mttd_dir = os.path.join(project_root, 'mttd')

# Ensure mttd directory exists
os.makedirs(mttd_dir, exist_ok=True)

# Success criteria thresholds
SUCCESS_CRITERIA = {
    'min_sharpe': 1.35,
    'min_win_rate': 60.0,
    'min_trades': 25,
    'max_trades': 35,
    'min_cagr': 50.0
}


def load_results(results_path):
    """Load base signal results from JSON file."""
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found")
        sys.exit(1)
    
    with open(results_path, 'r') as f:
        data = json.load(f)
    
    return data


def validate_success_criteria(metrics):
    """Check if a base signal meets all success criteria."""
    checks = {
        'sharpe': metrics.get('sharpe', 0) > SUCCESS_CRITERIA['min_sharpe'],
        'win_rate': metrics.get('win_rate', 0) > SUCCESS_CRITERIA['min_win_rate'],
        'trades': SUCCESS_CRITERIA['min_trades'] <= metrics.get('trades', 0) <= SUCCESS_CRITERIA['max_trades'],
        'cagr': metrics.get('cagr', 0) > SUCCESS_CRITERIA['min_cagr']
    }
    
    return all(checks.values()), checks


def print_comparison_table(results):
    """Print a formatted comparison table to stdout."""
    print("\n" + "="*95)
    print("BASE SIGNAL COMPARISON TABLE")
    print("="*95)
    
    # Header
    header = f"{'Base Signal':<35} {'Trades':>8} {'WinRate':>10} {'Sharpe':>8} {'CAGR':>8} {'AvgHold':>8} {'MaxDD':>8}"
    print(header)
    print("-"*95)
    
    # Data rows
    rows = []
    for key, metrics in results['results'].items():
        row = {
            'key': key,
            'signal_name': metrics['signal_name'],
            'trades': metrics['trades'],
            'win_rate': metrics['win_rate'],
            'sharpe': metrics['sharpe'],
            'cagr': metrics['cagr'],
            'avg_hold': metrics['avg_hold'],
            'max_dd': metrics['max_dd']
        }
        rows.append(row)
        
        # Print row
        print(f"{metrics['signal_name']:<35} {metrics['trades']:>8} {metrics['win_rate']:>9.1f}% "
              f"{metrics['sharpe']:>8.2f} {metrics['cagr']:>7.1f}% "
              f"{metrics['avg_hold']:>6.0f}d {metrics['max_dd']:>7.1f}%")
    
    print("-"*95)
    print("="*95)
    
    return rows


def find_best_signal(rows):
    """Find the best base signal based on Sharpe ratio and trade count."""
    # Filter for signals with trade count in target range (25-35)
    target_range = [r for r in rows if SUCCESS_CRITERIA['min_trades'] <= r['trades'] <= SUCCESS_CRITERIA['max_trades']]
    
    if not target_range:
        # Fall back to all signals if none in target range
        print("\nWarning: No signals in target trade range (25-35). Considering all signals.")
        target_range = rows
    
    # Sort by Sharpe ratio (primary) then by CAGR (secondary)
    best = max(target_range, key=lambda x: (x['sharpe'], x['cagr']))
    
    return best


def generate_comparison_chart(rows, best_signal, results):
    """Generate comparison chart at mttd/base_comparison.png."""
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'
    
    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Base Signal Comparison Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Extract data
    signal_names = [r['signal_name'].split('(')[0].strip()[:20] for r in rows]
    trades = [r['trades'] for r in rows]
    win_rates = [r['win_rate'] for r in rows]
    sharpes = [r['sharpe'] for r in rows]
    cagrs = [r['cagr'] for r in rows]
    avg_holds = [r['avg_hold'] for r in rows]
    max_dds = [r['max_dd'] for r in rows]
    
    # Colors: highlight best signal
    colors = ['#4CAF50' if r['key'] == best_signal['key'] else '#2196F3' for r in rows]
    
    # Panel 1: Sharpe Ratio Comparison (Bar chart)
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(signal_names, sharpes, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax1.axhline(y=SUCCESS_CRITERIA['min_sharpe'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f"Target: {SUCCESS_CRITERIA['min_sharpe']}")
    ax1.set_ylabel('Sharpe Ratio', fontsize=11)
    ax1.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, max(sharpes) * 1.3)
    
    # Add value labels on bars
    for bar, val in zip(bars1, sharpes):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03, 
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Panel 2: CAGR Comparison (Bar chart)
    ax2 = fig.add_subplot(gs[0, 1])
    bars2 = ax2.bar(signal_names, cagrs, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    ax2.axhline(y=SUCCESS_CRITERIA['min_cagr'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f"Target: {SUCCESS_CRITERIA['min_cagr']}%")
    ax2.set_ylabel('CAGR (%)', fontsize=11)
    ax2.set_title('CAGR Comparison', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.set_ylim(0, max(cagrs) * 1.3)
    
    # Add value labels on bars
    for bar, val in zip(bars2, cagrs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Panel 3: Win Rate vs Trade Count (Scatter plot)
    ax3 = fig.add_subplot(gs[1, 0])
    scatter_colors = ['#4CAF50' if r['key'] == best_signal['key'] else '#2196F3' for r in rows]
    sizes = [200 if r['key'] == best_signal['key'] else 120 for r in rows]
    
    scatter = ax3.scatter(trades, win_rates, c=scatter_colors, s=sizes, alpha=0.8, 
                         edgecolors='white', linewidths=2, zorder=5)
    
    # Add signal name labels
    for i, r in enumerate(rows):
        offset_y = 1.5 if i % 2 == 0 else -2.5
        ax3.annotate(r['signal_name'].split('(')[0].strip()[:15], 
                    (trades[i], win_rates[i]),
                    textcoords="offset points", xytext=(0, offset_y),
                    ha='center', fontsize=9, fontweight='bold')
    
    # Add success criteria regions
    ax3.axvspan(SUCCESS_CRITERIA['min_trades'], SUCCESS_CRITERIA['max_trades'], 
               alpha=0.15, color='green', label=f"Target: {SUCCESS_CRITERIA['min_trades']}-{SUCCESS_CRITERIA['max_trades']} trades")
    ax3.axhline(y=SUCCESS_CRITERIA['min_win_rate'], color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
               label=f"Target: {SUCCESS_CRITERIA['min_win_rate']}%")
    
    ax3.set_xlabel('Number of Trades', fontsize=11)
    ax3.set_ylabel('Win Rate (%)', fontsize=11)
    ax3.set_title('Win Rate vs Trade Count', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right')
    
    # Panel 4: Risk-Return Scatter (Sharpe vs Max Drawdown)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Convert max drawdown to positive for visualization
    abs_max_dds = [abs(d) for d in max_dds]
    
    scatter2 = ax4.scatter(abs_max_dds, sharpes, c=scatter_colors, s=sizes, alpha=0.8,
                          edgecolors='white', linewidths=2, zorder=5)
    
    # Add signal name labels
    for i, r in enumerate(rows):
        offset_y = 0.03 if i % 2 == 0 else -0.05
        ax4.annotate(r['signal_name'].split('(')[0].strip()[:15],
                    (abs_max_dds[i], sharpes[i]),
                    textcoords="offset points", xytext=(5, offset_y * 100),
                    ha='left', fontsize=9, fontweight='bold')
    
    ax4.set_xlabel('Max Drawdown (%)', fontsize=11)
    ax4.set_ylabel('Sharpe Ratio', fontsize=11)
    ax4.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
    
    # Add quadrant labels
    ax4.axhline(y=np.mean(sharpes), color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax4.axvline(x=np.mean(abs_max_dds), color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # Add legend for best signal
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50', markersize=12, label='Best Signal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=12, label='Other Signals')
    ]
    ax4.legend(handles=legend_elements, loc='upper left')
    
    # Save chart
    chart_path = os.path.join(mttd_dir, 'base_comparison.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return chart_path


def print_best_signal_analysis(best_signal, results):
    """Print detailed analysis of the best signal."""
    print("\n" + "="*95)
    print("BEST BASE SIGNAL IDENTIFICATION")
    print("="*95)
    
    print(f"\n🏆 BEST SIGNAL: {best_signal['signal_name']}")
    print(f"   Key: {best_signal['key']}")
    
    print(f"\n   Performance Metrics:")
    print(f"   ─────────────────────────────────────────────────")
    print(f"   Trades:        {best_signal['trades']}")
    print(f"   Win Rate:      {best_signal['win_rate']:.1f}%")
    print(f"   Sharpe Ratio:  {best_signal['sharpe']:.2f}")
    print(f"   CAGR:          {best_signal['cagr']:.1f}%")
    print(f"   Avg Hold:      {best_signal['avg_hold']} days")
    print(f"   Max Drawdown:  {best_signal['max_dd']:.1f}%")
    
    # Additional metrics from original data
    key = best_signal['key']
    if key in results['results']:
        extra = results['results'][key]
        print(f"   Sortino:       {extra.get('sortino', 'N/A')}")
        print(f"   Calmar:        {extra.get('calmar', 'N/A')}")
        print(f"   Position %:    {extra.get('position_pct', 'N/A')}%")
    
    # Validate against success criteria
    meets_criteria, checks = validate_success_criteria(results['results'][key])
    
    print(f"\n   Success Criteria Validation:")
    print(f"   ─────────────────────────────────────────────────")
    print(f"   {'Sharpe > 1.35':<25} {'✅ PASS' if checks['sharpe'] else '❌ FAIL'} ({best_signal['sharpe']:.2f})")
    print(f"   {'Win Rate > 60%':<25} {'✅ PASS' if checks['win_rate'] else '❌ FAIL'} ({best_signal['win_rate']:.1f}%)")
    print(f"   {'Trades 25-35':<25} {'✅ PASS' if checks['trades'] else '❌ FAIL'} ({best_signal['trades']})")
    print(f"   {'CAGR > 50%':<25} {'✅ PASS' if checks['cagr'] else '❌ FAIL'} ({best_signal['cagr']:.1f}%)")
    
    print(f"\n   {'─'*50}")
    if meets_criteria:
        print(f"   🎉 RESULT: MEETS ALL SUCCESS CRITERIA!")
    else:
        print(f"   ⚠️  RESULT: Does NOT meet all success criteria")
        # Identify what's missing
        missing = [k for k, v in checks.items() if not v]
        print(f"   Missing: {', '.join(missing)}")
    
    print("="*95)
    
    return meets_criteria


def print_rankings(rows):
    """Print ranked list by Sharpe ratio."""
    print("\n" + "="*95)
    print("RANKINGS BY SHARPE RATIO")
    print("="*95)
    
    # Sort by Sharpe ratio
    ranked = sorted(rows, key=lambda x: x['sharpe'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Base Signal':<35} {'Sharpe':>8} {'CAGR':>8} {'WinRate':>10} {'Trades':>8}")
    print("-"*95)
    
    for i, r in enumerate(ranked, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal}{i:<4} {r['signal_name']:<35} {r['sharpe']:>8.2f} {r['cagr']:>7.1f}% "
              f"{r['win_rate']:>9.1f}% {r['trades']:>8}")
    
    print("-"*95)
    print("="*95)


def main():
    """Main entry point."""
    print("\n" + "="*95)
    print("MTTD BASE SIGNAL COMPARISON")
    print("="*95)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load results
    results_path = os.path.join(project_root, 'base_signal_results.json')
    results = load_results(results_path)
    
    # Print metadata
    meta = results.get('metadata', {})
    print(f"Data Range: {meta.get('data_range', 'N/A')}")
    print(f"Bars: {meta.get('bars', 'N/A')}")
    print(f"Gate Threshold: {meta.get('gate_threshold', 'N/A')}")
    print(f"Transaction Cost: {meta.get('transaction_cost', 'N/A')}")
    
    # Print comparison table
    rows = print_comparison_table(results)
    
    # Print rankings
    print_rankings(rows)
    
    # Find best signal
    best_signal = find_best_signal(rows)
    
    # Generate comparison chart
    chart_path = generate_comparison_chart(rows, best_signal, results)
    print(f"\n📊 Chart generated: {chart_path}")
    
    # Print best signal analysis
    meets_criteria = print_best_signal_analysis(best_signal, results)
    
    # Summary
    print("\n" + "="*95)
    print("SUMMARY")
    print("="*95)
    print(f"Total signals compared: {len(rows)}")
    print(f"Best signal: {best_signal['signal_name']}")
    print(f"Best Sharpe: {best_signal['sharpe']:.2f}")
    print(f"Meets success criteria: {'Yes' if meets_criteria else 'No'}")
    print("="*95 + "\n")
    
    return 0 if meets_criteria else 1


if __name__ == "__main__":
    sys.exit(main())
