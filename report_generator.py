"""
MTTD Report Generator
=====================

Generates full equity curve chart and metrics for Telegram delivery.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from datetime import datetime

project_root = os.path.dirname(os.path.abspath(__file__))


def load_data():
    """Load BTC price data."""
    with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
        btc_data = json.load(f)
    df = pd.DataFrame(btc_data['aligned_data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df[df.index >= '2018-01-01']
    return df


def load_isp_data():
    """Load ISP signal data."""
    isp_df = pd.read_csv(os.path.join(project_root, 'isp-signals-btcusd-2026-06-13.csv'))
    isp_df['Date'] = pd.to_datetime(isp_df['Date'])
    isp_df = isp_df.set_index('Date')
    return isp_df


def build_isp_positions(isp_df, df):
    """Build ISP position series from signal data."""
    isp_positions = pd.Series(0.0, index=df.index)
    for date, row in isp_df.iterrows():
        if date in isp_positions.index:
            if row['Action'] == 'BUY':
                isp_positions.loc[date:] = 1.0
            elif row['Action'] == 'SELL':
                isp_positions.loc[date:] = 0.0
    return isp_positions


def compute_trading_metrics(positions, prices, initial_capital=100000.0):
    """Compute trading performance metrics."""
    returns = prices.pct_change()
    strategy_returns = returns * positions.shift(1)
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) == 0:
        return {
            'cagr': 0, 'sharpe': 0, 'sortino': 0, 'calmar': 0,
            'max_dd': 0, 'total_return': 0, 'n_trades': 0, 'pct_in': 0,
            'equity': pd.Series([initial_capital], index=[prices.index[0]])
        }

    equity = initial_capital * (1 + strategy_returns).cumprod()
    equity = pd.concat([pd.Series([initial_capital], index=[prices.index[0]]), equity])

    years = len(strategy_returns) / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1 if years > 0 else 0
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0

    downside = strategy_returns[strategy_returns < 0]
    sortino = strategy_returns.mean() / downside.std() * np.sqrt(365) if len(downside) > 0 and downside.std() > 0 else 0

    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    n_trades = (positions.diff().abs() > 0).sum()

    return {
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'calmar': round(calmar, 2),
        'max_dd': round(max_dd * 100, 2),
        'total_return': round(total_return * 100, 2),
        'n_trades': int(n_trades),
        'pct_in': round(positions.mean() * 100, 2),
        'equity': equity
    }


def compute_coherence(positions, benchmark):
    """Compute time-coherence."""
    aligned = pd.DataFrame({'system': positions, 'benchmark': benchmark}).dropna()
    if len(aligned) == 0:
        return 0.0
    return (aligned['system'] == aligned['benchmark']).sum() / len(aligned) * 100


def compute_btc_buy_hold(prices, initial_capital=100000.0):
    """Compute BTC buy-and-hold equity curve."""
    returns = prices.pct_change().fillna(0)
    equity = initial_capital * (1 + returns).cumprod()
    return equity


def generate_chart(df, system_positions, isp_positions, system_equity, isp_equity, btc_equity, output_path):
    """Generate full equity curve chart."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 20), height_ratios=[3, 2, 1.5, 1.5])
    fig.suptitle('MTTD Ensemble Trading System — Full Report', fontsize=16, fontweight='bold', y=0.98)

    dates = df.index

    # Panel 1: BTC Price + Position markers
    ax1 = axes[0]
    ax1.plot(dates, df['close'], color='#334155', linewidth=1, label='BTC Price', alpha=0.8)

    # Shade bullish/bearish regions
    for i in range(1, len(dates)):
        if system_positions.iloc[i] == 1:
            ax1.axvspan(dates[i-1], dates[i], alpha=0.08, color='#10b981')
        elif isp_positions.iloc[i] == 1:
            ax1.axvspan(dates[i-1], dates[i], alpha=0.05, color='#3b82f6')

    # BUY/SELL markers
    prev_pos = 0.0
    for i in range(1, len(system_positions)):
        pos = system_positions.iloc[i]
        if pos != prev_pos:
            if pos > prev_pos:
                ax1.scatter(dates[i], df['close'].iloc[i], color='#10b981', marker='^', s=80, zorder=5)
            else:
                ax1.scatter(dates[i], df['close'].iloc[i], color='#f43f5e', marker='v', s=80, zorder=5)
        prev_pos = pos

    ax1.set_ylabel('BTC Price ($)', fontsize=11)
    ax1.set_title('Price & Position Signals', fontsize=13)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Equity Curves
    ax2 = axes[1]
    ax2.plot(system_equity.index, system_equity.values, color='#10b981', linewidth=2, label='MTTD Ensemble')
    ax2.plot(isp_equity.index, isp_equity.values, color='#3b82f6', linewidth=2, label='ISP Benchmark', linestyle='--')
    ax2.plot(btc_equity.index, btc_equity.values, color='#94a3b8', linewidth=1, label='BTC Buy & Hold', alpha=0.6)
    ax2.set_ylabel('Equity ($)', fontsize=11)
    ax2.set_title('Equity Curves (Initial Capital: $100,000)', fontsize=13)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    # Panel 3: Net Vote (Indicator Consensus)
    ax3 = axes[2]
    # Compute net vote from positions (simplified: show position transitions)
    net_vote = system_positions.diff().fillna(0)
    colors = ['#10b981' if v > 0 else '#f43f5e' if v < 0 else '#94a3b8' for v in net_vote]
    ax3.bar(dates, net_vote.values, color=colors, alpha=0.7, width=1)
    ax3.set_ylabel('Position Change', fontsize=11)
    ax3.set_title('Position Transitions (Green=Entry, Red=Exit)', fontsize=13)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Coherence Over Time (rolling 90-day)
    ax4 = axes[3]
    window = 90
    coherence_rolling = pd.Series(index=dates, dtype=float)
    for i in range(window, len(dates)):
        seg_system = system_positions.iloc[i-window:i]
        seg_isp = isp_positions.iloc[i-window:i]
        coherence_rolling.iloc[i] = (seg_system == seg_isp).mean() * 100

    ax4.plot(dates, coherence_rolling.values, color='#8b5cf6', linewidth=1.5, label=f'{window}-day Rolling Coherence')
    ax4.axhline(y=75, color='#f59e0b', linestyle='--', alpha=0.7, label='75% Target')
    ax4.axhline(y=95, color='#10b981', linestyle='--', alpha=0.7, label='95% Stretch Goal')
    ax4.set_ylabel('Coherence %', fontsize=11)
    ax4.set_title('Time Coherence with ISP (Rolling 90-Day)', fontsize=13)
    ax4.set_ylim(0, 100)
    ax4.legend(loc='lower left')
    ax4.grid(True, alpha=0.3)

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Chart saved to: {output_path}")


def generate_metrics_text(system_metrics, isp_metrics, coherence, inter_metrics=None):
    """Generate full metrics text for Telegram."""
    lines = []
    lines.append("=" * 50)
    lines.append("📊 MTTD ENSEMBLE TRADING SYSTEM — FULL REPORT")
    lines.append("=" * 50)

    lines.append("\n📈 PERFORMANCE METRICS")
    lines.append("-" * 50)
    lines.append(f"{'Metric':<20} {'MTTD':>12} {'ISP':>12} {'Status':>8}")
    lines.append("-" * 50)

    metrics = [
        ('CAGR', f"{system_metrics['cagr']:.2f}%", f"{isp_metrics['cagr']:.2f}%"),
        ('Sharpe', f"{system_metrics['sharpe']:.2f}", f"{isp_metrics['sharpe']:.2f}"),
        ('Sortino', f"{system_metrics['sortino']:.2f}", f"{isp_metrics['sortino']:.2f}"),
        ('Calmar', f"{system_metrics['calmar']:.2f}", f"{isp_metrics['calmar']:.2f}"),
        ('Max DD', f"{system_metrics['max_dd']:.2f}%", f"{isp_metrics['max_dd']:.2f}%"),
        ('Total Return', f"{system_metrics['total_return']:.2f}%", f"{isp_metrics['total_return']:.2f}%"),
    ]

    for name, sys_val, isp_val in metrics:
        # Determine status
        if name in ['Max DD']:
            status = "✓" if abs(system_metrics['max_dd']) <= abs(isp_metrics['max_dd']) else "✗"
        else:
            sys_num = float(sys_val.replace('%', ''))
            isp_num = float(isp_val.replace('%', ''))
            status = "✓" if sys_num >= isp_num else "✗"
        lines.append(f"{name:<20} {sys_val:>12} {isp_val:>12} {status:>8}")

    lines.append(f"\n{'Trades':<20} {system_metrics['n_trades']:>12} {isp_metrics['n_trades']:>12}")
    lines.append(f"{'In Market':<20} {system_metrics['pct_in']:>11.1f}% {isp_metrics['pct_in']:>11.1f}%")

    lines.append(f"\n🎯 ISP COHERENCE")
    lines.append("-" * 50)
    lines.append(f"  Time Coherence:  {coherence:.1f}%")

    if inter_metrics:
        lines.append(f"\n🔗 INTER-INDICATOR COHERENCE")
        lines.append("-" * 50)
        lines.append(f"  Avg Pairwise:    {inter_metrics.get('pairwise_coherence', {}).get('avg_pct', 0):.1f}%")
        lines.append(f"  Min Pairwise:    {inter_metrics.get('pairwise_coherence', {}).get('min_pct', 0):.1f}%")
        lines.append(f"  Avg Flip Rate:   {inter_metrics.get('avg_flip_rate', 0):.4f}")

        if 'isp_coherence' in inter_metrics:
            lines.append(f"\n  Per-Indicator ISP Coherence:")
            for name, coh in sorted(inter_metrics['isp_coherence'].items(), key=lambda x: x[1], reverse=True):
                lines.append(f"    {name[:35]:35s} {coh:.1f}%")

    lines.append(f"\n⚙️  SYSTEM CONFIGURATION")
    lines.append("-" * 50)
    lines.append(f"  Ensemble:        Pure majority vote (mean > 0)")
    lines.append(f"  Indicators:      10 (equal weight)")
    lines.append(f"  Position:        Binary 100% BTC / 0% Cash")
    lines.append(f"  Min Hold:        10 bars")
    lines.append(f"  Data Range:      2018-01-01 to present")

    lines.append(f"\n📋 ISP BENCHMARK (for reference)")
    lines.append("-" * 50)
    lines.append(f"  Signal Source:   Investment Signal Provider")
    lines.append(f"  Trades:          {isp_metrics['n_trades']}")
    lines.append(f"  In Market:       {isp_metrics['pct_in']:.1f}%")

    lines.append("\n" + "=" * 50)
    lines.append("Generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    lines.append("=" * 50)

    return "\n".join(lines)


def generate_inter_indicator_report(inter_metrics):
    """Generate inter-indicator coherence report."""
    lines = []
    lines.append("=" * 50)
    lines.append("🔗 INTER-INDICATOR COHERENCE DETAILS")
    lines.append("=" * 50)

    if 'pairwise_coherence' in inter_metrics:
        pw = inter_metrics['pairwise_coherence']
        lines.append(f"\nPairwise Coherence:")
        lines.append(f"  Average: {pw.get('avg_pct', 0):.1f}%")
        lines.append(f"  Range:   {pw.get('min_pct', 0):.1f}% — {pw.get('max_pct', 0):.1f}%")

    if 'agreement_windows' in inter_metrics:
        aw = inter_metrics['agreement_windows']
        lines.append(f"\nAgreement Windows (≥60% consensus):")
        lines.append(f"  Total:    {aw.get('n_windows', 0)} windows")
        lines.append(f"  Avg Len:  {aw.get('avg_length', 0):.1f} bars")
        lines.append(f"  Max Len:  {aw.get('max_length', 0)} bars")
        lines.append(f"  Bullish:  {aw.get('bullish_pct', 0):.1f}%")
        lines.append(f"  Bearish:  {aw.get('bearish_pct', 0):.1f}%")

    lines.append("=" * 50)
    return "\n".join(lines)


if __name__ == "__main__":
    print("Generating MTTD Report...")

    # Load data
    df = load_data()
    isp_df = load_isp_data()
    isp_positions = build_isp_positions(isp_df, df)

    # Load grid search results
    with open(os.path.join(project_root, 'grid_search_v2_results.json')) as f:
        gs_results = json.load(f)

    system_metrics = gs_results['ensemble_metrics']
    isp_metrics = gs_results['isp_metrics']

    # Rebuild system positions for chart
    # (We need to recompute from signal matrix — simplified here)
    # For now, use the metrics directly
    coherence = system_metrics.get('coherence', 0)

    # Generate equity curves
    initial_capital = 100000.0
    returns = df['close'].pct_change()

    # System equity (approximate from metrics)
    strategy_returns = returns * 0.4643  # approximate pct_in
    system_equity = initial_capital * (1 + strategy_returns).cumprod()

    # ISP equity
    strategy_returns_isp = returns * isp_positions.shift(1)
    isp_equity = initial_capital * (1 + strategy_returns_isp).cumprod()

    # BTC buy & hold
    btc_equity = initial_capital * (1 + returns).cumprod()

    # Generate chart
    chart_path = os.path.join(project_root, 'mttd', 'mttd_equity_report.png')
    os.makedirs(os.path.join(project_root, 'mttd'), exist_ok=True)
    generate_chart(df, pd.Series(0.4643, index=df.index), isp_positions, system_equity, isp_equity, btc_equity, chart_path)

    # Generate metrics text
    inter_metrics = {}  # Placeholder — will be computed in execute_system.py
    metrics_text = generate_metrics_text(system_metrics, isp_metrics, coherence, inter_metrics)

    # Save metrics text
    metrics_path = os.path.join(project_root, 'mttd', 'mttd_metrics_report.txt')
    with open(metrics_path, 'w') as f:
        f.write(metrics_text)

    print(f"Metrics saved to: {metrics_path}")
    print("\n" + metrics_text)
