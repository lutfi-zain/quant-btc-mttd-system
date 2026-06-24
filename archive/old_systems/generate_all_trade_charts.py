#!/usr/bin/env python3
"""
Generate Trade Charts for Each Indicator
==========================================

Creates individual trade visualization charts for each of the 4 indicators
(MSVR, Ichimoku, Supertrend, Keltner), showing:
- Price data with OHLC
- Entry signals (green arrows)
- Exit signals (red arrows)
- Win/loss highlighting (green/red for profitable/losing trades)
- Indicator behavior (support/resistance lines where applicable)

Output: mttd/charts/{indicator}_trade_chart.png
"""

import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BANK_ROOT = '/home/ubuntu/projects/quant-technical-indicator-bank'
sys.path.append(PROJECT_ROOT)
sys.path.append(BANK_ROOT)

from indicators_helper import sma, ema, atr, linreg


# ================================================================
# Helper Functions (from holdout_best_configs.py)
# ================================================================

def ehler_supersmoother(series, length=7):
    """Ehler's SuperSmoother Filter (Family 2: Filtering)."""
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
    """FFT-based cycle phase timing (Family 4: Spectral)."""
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
        windowed = window_detrended * hann
        fft_vals = np.fft.rfft(windowed)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(lookback, d=1)
        min_freq = 1.0 / max_period
        max_freq = 1.0 / min_period
        valid_mask = (freqs >= min_freq) & (freqs <= max_period)
        valid_power = power[valid_mask]
        valid_freqs = freqs[valid_mask]
        if len(valid_power) > 0 and np.sum(valid_power) > 0:
            dominant_idx = np.argmax(valid_power)
            dominant_freq = valid_freqs[dominant_idx]
            dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else lookback
            cycle_pos = i % int(dominant_period)
            phase.iloc[i] = 2 * np.pi * cycle_pos / dominant_period
    return phase


# ================================================================
# Signal Generators (from holdout_best_configs.py)
# ================================================================

def apply_position_state_machine(entry_signal, exit_signal, min_hold, max_hold, cooldown=5):
    """
    Clean state machine: entry → hold → exit → cooldown → entry → ...
    
    Prevents immediate re-entry after exit (cooldown period).
    """
    position = pd.Series(0.0, index=entry_signal.index)
    in_position = False
    hold_count = 0
    cooldown_count = 0

    for i in range(len(entry_signal)):
        if cooldown_count > 0:
            cooldown_count -= 1

        if entry_signal.iloc[i] == 1.0 and not in_position and cooldown_count == 0:
            # ENTRY — only if not in position AND cooldown done
            in_position = True
            hold_count = 0
            position.iloc[i] = 1.0
        elif in_position:
            hold_count += 1
            if (hold_count >= min_hold and exit_signal.iloc[i] == 1.0) or hold_count >= max_hold:
                # EXIT — min_hold + exit signal, OR max_hold forced
                in_position = False
                hold_count = 0
                cooldown_count = cooldown
                position.iloc[i] = 0.0
            else:
                position.iloc[i] = 1.0
        else:
            position.iloc[i] = 0.0

    return position


def generate_keltner_signal(df, use_filters=False, min_hold=15, max_hold=60, cooldown=5):
    """Generate Keltner Channel trading signal."""
    result = df.copy()
    kc_mid = ema(result['close'], 20)
    kc_atr = ema(result['high'] - result['low'], 20)
    result['kc_upper'] = kc_mid + 1.5 * kc_atr
    result['kc_lower'] = kc_mid - 1.5 * kc_atr

    result['kc_buy'] = (result['close'] > result['kc_upper']).astype(float)
    result['kc_sell'] = (result['close'] < result['kc_lower']).astype(float)

    entry_signal = result['kc_buy']
    exit_signal = result['kc_sell']

    if use_filters:
        spec = importlib.util.spec_from_file_location(
            'msvr', BANK_ROOT + '/perpetual/median_standard_deviation_viresearch.py'
        )
        msvr_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(msvr_module)
        msvr_result = msvr_module.median_standard_deviation_viresearch(result)
        result['msvr_vii'] = msvr_result['vii']

        momentum = result['close'].pct_change(periods=10)
        smooth = ehler_supersmoother(momentum, length=5)
        result['smooth_direction'] = (smooth > 0).astype(float)
        phase = compute_cycle_phase(result, lookback=40)
        cycle_signal = -np.cos(phase)
        result['cycle_direction'] = (cycle_signal > 0).astype(float)
        result['msvr_direction'] = (result['msvr_vii'] > 0).astype(float)

        filter_pass = (result['msvr_direction'] * result['smooth_direction'] * result['cycle_direction']).astype(float)
        entry_signal = result['kc_buy'] * filter_pass

    position = apply_position_state_machine(entry_signal, exit_signal, min_hold, max_hold, cooldown)
    return position, result


def generate_ichimoku_signal(df, min_hold=15, max_hold=60, cooldown=5):
    """Generate Ichimoku trading signal with cooldown."""
    sys.path.insert(0, PROJECT_ROOT)
    from ichimoku_quant import generate_ichimoku_features, generate_ichimoku_signals

    df_ich = generate_ichimoku_features(df.copy())
    df_ich = generate_ichimoku_signals(
        df_ich,
        confirm_entry=2, confirm_exit=1, min_hold_days=min_hold,
        er_entry=0.25, t_entry=0.40, chikou_thresh=-0.30,
        immunity_thresh=0.50, entropy_thresh=2.271,
        imo_min_limit=-0.30, imo_exit_bull=-0.30, roc_gate_limit=-0.20
    )

    raw_pos = df_ich['Pos'].copy()

    # Apply max_hold + cooldown on top of Ichimoku's position
    in_position = False
    hold_count = 0
    cooldown_count = 0
    position = pd.Series(0.0, index=raw_pos.index)

    for i in range(len(raw_pos)):
        if cooldown_count > 0:
            cooldown_count -= 1

        if raw_pos.iloc[i] == 1.0 and not in_position and cooldown_count == 0:
            in_position = True
            hold_count = 0
            position.iloc[i] = 1.0
        elif in_position:
            hold_count += 1
            if hold_count >= max_hold or (hold_count >= min_hold and raw_pos.iloc[i] == 0.0):
                in_position = False
                hold_count = 0
                cooldown_count = cooldown
                position.iloc[i] = 0.0
            else:
                position.iloc[i] = 1.0
        else:
            position.iloc[i] = 0.0

    return position, df_ich


def generate_supertrend_signal(df, min_hold=15, max_hold=90, cooldown=5):
    """Generate Supertrend trading signal with cooldown."""
    spec_st = importlib.util.spec_from_file_location(
        'supertrend', BANK_ROOT + '/perpetual/median_supertrend_viresearch.py'
    )
    st_module = importlib.util.module_from_spec(spec_st)
    spec_st.loader.exec_module(st_module)
    st_result = st_module.median_supertrend_viresearch(df)

    st_vii = st_result['vii']
    entry_signal = (st_vii > 0).astype(float)
    exit_signal = (st_vii < 0).astype(float)

    position = apply_position_state_machine(entry_signal, exit_signal, min_hold, max_hold, cooldown)
    return position, st_result


def generate_msvr_signal(df, min_hold=15, max_hold=90, cooldown=5):
    """Generate MSVR trading signal with cooldown."""
    spec = importlib.util.spec_from_file_location(
        'msvr', BANK_ROOT + '/perpetual/median_standard_deviation_viresearch.py'
    )
    msvr_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(msvr_module)
    msvr_result = msvr_module.median_standard_deviation_viresearch(df)

    msvr_vii = msvr_result['vii']
    entry_signal = (msvr_vii > 0).astype(float)
    exit_signal = (msvr_vii < 0).astype(float)

    position = apply_position_state_machine(entry_signal, exit_signal, min_hold, max_hold, cooldown)
    return position, msvr_result


# ================================================================
# Trade Extraction
# ================================================================

def extract_trades(position, prices):
    """
    Extract individual trades from position series.
    
    Returns list of dicts:
    [
        {
            'entry_date': pd.Timestamp,
            'exit_date': pd.Timestamp,
            'entry_price': float,
            'exit_price': float,
            'return_pct': float,
            'hold_days': int,
            'is_win': bool
        },
        ...
    ]
    """
    trades = []
    in_position = False
    entry_date = None
    entry_price = None
    
    for i, (date, pos) in enumerate(position.items()):
        if pos == 1.0 and not in_position:
            in_position = True
            entry_date = date
            entry_price = prices.loc[date]
        elif pos == 0.0 and in_position:
            in_position = False
            exit_price = prices.loc[date]
            return_pct = (exit_price - entry_price) / entry_price * 100
            hold_days = (date - entry_date).days
            
            trades.append({
                'entry_date': entry_date,
                'exit_date': date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'return_pct': return_pct,
                'hold_days': hold_days,
                'is_win': return_pct > 0
            })
    
    return trades


def compute_trade_metrics(trades):
    """Compute summary metrics for a list of trades."""
    if len(trades) == 0:
        return {
            'n_trades': 0,
            'win_rate': 0,
            'avg_return': 0,
            'total_return': 0,
            'sharpe': 0,
            'avg_hold': 0
        }
    
    returns = [t['return_pct'] for t in trades]
    wins = sum(1 for t in trades if t['is_win'])
    n_trades = len(trades)
    win_rate = wins / n_trades * 100
    avg_return = np.mean(returns)
    total_return = np.sum(returns)
    avg_hold = np.mean([t['hold_days'] for t in trades])
    
    # Sharpe-like ratio (annualized)
    if np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(n_trades / 365 * 365)
    else:
        sharpe = 0
    
    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'total_return': total_return,
        'sharpe': sharpe,
        'avg_hold': avg_hold
    }


# ================================================================
# Chart Generation
# ================================================================

def generate_trade_chart(df, position, indicator_data, trades, indicator_name,
                         output_dir, extra_info=None):
    """
    Generate a trade visualization chart for an indicator.
    
    CLEAN version: only entry/exit arrows, no connecting lines, no shaded areas.
    """
    # Use minimal style
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#cccccc',
        'axes.grid': True,
        'grid.alpha': 0.2,
        'grid.color': '#cccccc',
    })
    
    # Create figure with 2 subplots (price + returns)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                     gridspec_kw={'height_ratios': [3, 1]},
                                     sharex=True)
    
    # Compute metrics
    metrics = compute_trade_metrics(trades)
    n_win = sum(1 for t in trades if t['is_win'])
    n_loss = len(trades) - n_win
    
    # Build title
    title = f"{indicator_name} | {metrics['n_trades']} trades | {metrics['win_rate']:.0f}% win | Sharpe {metrics['sharpe']:.2f} | CAGR {metrics['total_return']/len(df)*365:.0f}%"
    if extra_info:
        title += f" | {extra_info}"
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    
    # ---- TOP PANEL: Price - simple clean line ----
    ax1.plot(df.index, df['close'], color='#333333', linewidth=0.8, label='BTC')
    
    # Entry markers (▲ green) - one per trade, period
    for trade in trades:
        ax1.scatter(trade['entry_date'], trade['entry_price'],
                   marker='^', color='#22c55e', s=60, zorder=10,
                   edgecolors='#166534', linewidth=0.5, label='_' if trade != trades[0] else 'Entry')
    
    # Exit markers (▼ red for loss, ▼ green for win)
    for trade in trades:
        color = '#22c55e' if trade['is_win'] else '#ef4444'
        edge = '#166534' if trade['is_win'] else '#991b1b'
        ax1.scatter(trade['exit_date'], trade['exit_price'],
                   marker='v', color=color, s=60, zorder=10,
                   edgecolors=edge, linewidth=0.5, label='_' if trade != trades[0] else 'Exit')
    
    ax1.set_ylabel('Price (USD)', fontsize=10)
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Holdout boundary
    holdout_date = pd.Timestamp('2025-01-01')
    ax1.axvline(x=holdout_date, color='#ef4444', linestyle='--', linewidth=1, alpha=0.5)
    ax1.text(holdout_date, ax1.get_ylim()[0]*1.1, 'HOLDOUT',
             rotation=90, fontsize=8, color='#ef4444', alpha=0.7, va='bottom')
    
    # Simple legend
    legend_elements = [
        plt.Line2D([0], [0], color='#333333', linewidth=0.8, label='BTC'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#22c55e', markersize=8, label=f'Entry ({len(trades)})'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#22c55e', markersize=8, label=f'Win ({n_win})'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='#ef4444', markersize=8, label=f'Loss ({n_loss})'),
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=8, framealpha=0.9)
    
    # ---- BOTTOM PANEL: Trade returns - simple bars ----
    trade_dates = [t['exit_date'] for t in trades]
    trade_returns = [t['return_pct'] for t in trades]
    trade_colors = ['#22c55e' if r > 0 else '#ef4444' for r in trade_returns]
    
    bars = ax2.bar(trade_dates, trade_returns, color=trade_colors, alpha=0.8, width=10)
    ax2.axhline(y=0, color='#333333', linewidth=0.5)
    ax2.set_ylabel('Return (%)', fontsize=10)
    
    # Add value labels only for larger bars (avoid clutter)
    for bar, ret in zip(bars, trade_returns):
        if abs(ret) > 10:  # Only label big moves
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h,
                    f'{ret:.0f}%', ha='center', va='bottom' if h > 0 else 'top',
                    fontsize=6, color='#666666')
    
    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save chart
    chart_filename = f"{indicator_name.lower()}_trade_chart.png"
    chart_path = os.path.join(output_dir, chart_filename)
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return chart_path


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 70)
    print("GENERATING TRADE CHARTS FOR ALL INDICATORS")
    print("=" * 70)
    print()
    
    # ================================================================
    # Step 1: Load BTC Data
    # ================================================================
    print("[1/6] Loading BTC data...")
    
    with open(os.path.join(PROJECT_ROOT, 'data', 'btc_daily.json')) as f:
        btc_data = json.load(f)
    
    df = pd.DataFrame(btc_data['aligned_data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df[df.index >= '2018-01-01']
    
    print(f"  Total data: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
    
    # ================================================================
    # Step 2: Create output directory
    # ================================================================
    print("\n[2/6] Creating output directory...")
    
    output_dir = os.path.join(PROJECT_ROOT, 'mttd', 'charts')
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output: {output_dir}")
    
    # ================================================================
    # Step 3: Generate MSVR trades and chart
    # ================================================================
    print("\n[3/6] Generating MSVR trade chart...")
    
    position_msvr, msvr_data = generate_msvr_signal(df, min_hold=15, max_hold=90)
    trades_msvr = extract_trades(position_msvr, df['close'])
    metrics_msvr = compute_trade_metrics(trades_msvr)
    
    chart_path = generate_trade_chart(
        df, position_msvr, msvr_data, trades_msvr, 'MSVR',
        output_dir,
        extra_info='MH=15/90 | Median Standard Deviation'
    )
    print(f"  Trades: {metrics_msvr['n_trades']}, Win Rate: {metrics_msvr['win_rate']:.1f}%")
    print(f"  Saved: {chart_path}")
    
    # ================================================================
    # Step 4: Generate Ichimoku trades and chart
    # ================================================================
    print("\n[4/6] Generating Ichimoku trade chart...")
    
    position_ichimoku, ichimoku_data = generate_ichimoku_signal(df, min_hold=15, max_hold=60)
    trades_ichimoku = extract_trades(position_ichimoku, df['close'])
    metrics_ichimoku = compute_trade_metrics(trades_ichimoku)
    
    chart_path = generate_trade_chart(
        df, position_ichimoku, ichimoku_data, trades_ichimoku, 'Ichimoku',
        output_dir,
        extra_info='MH=15/60 | Ichimoku Momentum Oscillator'
    )
    print(f"  Trades: {metrics_ichimoku['n_trades']}, Win Rate: {metrics_ichimoku['win_rate']:.1f}%")
    print(f"  Saved: {chart_path}")
    
    # ================================================================
    # Step 5: Generate Supertrend trades and chart
    # ================================================================
    print("\n[5/6] Generating Supertrend trade chart...")
    
    position_supertrend, supertrend_data = generate_supertrend_signal(df, min_hold=15, max_hold=90)
    trades_supertrend = extract_trades(position_supertrend, df['close'])
    metrics_supertrend = compute_trade_metrics(trades_supertrend)
    
    chart_path = generate_trade_chart(
        df, position_supertrend, supertrend_data, trades_supertrend, 'Supertrend',
        output_dir,
        extra_info='MH=15/90 | Median Supertrend'
    )
    print(f"  Trades: {metrics_supertrend['n_trades']}, Win Rate: {metrics_supertrend['win_rate']:.1f}%")
    print(f"  Saved: {chart_path}")
    
    # ================================================================
    # Step 6: Generate Keltner trades and chart
    # ================================================================
    print("\n[6/6] Generating Keltner trade chart...")
    
    position_keltner, keltner_data = generate_keltner_signal(
        df, use_filters=True, min_hold=15, max_hold=60
    )
    trades_keltner = extract_trades(position_keltner, df['close'])
    metrics_keltner = compute_trade_metrics(trades_keltner)
    
    chart_path = generate_trade_chart(
        df, position_keltner, keltner_data, trades_keltner, 'Keltner',
        output_dir,
        extra_info='MH=15/60 | Bull With Filters (MSVR/Smooth/Cycle)'
    )
    print(f"  Trades: {metrics_keltner['n_trades']}, Win Rate: {metrics_keltner['win_rate']:.1f}%")
    print(f"  Saved: {chart_path}")
    
    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("TRADE CHARTS GENERATED SUCCESSFULLY")
    print("=" * 70)
    
    print("\nChart Files:")
    for filename in ['msvr_trade_chart.png', 'ichimoku_trade_chart.png', 
                     'supertrend_trade_chart.png', 'keltner_trade_chart.png']:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            print(f"  ✅ {filename}")
        else:
            print(f"  ❌ {filename} (NOT FOUND)")
    
    print("\nTrade Summary:")
    print(f"  {'Indicator':<15} {'Trades':>8} {'Win%':>8} {'Avg Ret':>10} {'Avg Hold':>10}")
    print("  " + "-" * 60)
    print(f"  {'MSVR':<15} {metrics_msvr['n_trades']:>8} {metrics_msvr['win_rate']:>7.1f}% {metrics_msvr['avg_return']:>9.2f}% {metrics_msvr['avg_hold']:>8.0f}d")
    print(f"  {'Ichimoku':<15} {metrics_ichimoku['n_trades']:>8} {metrics_ichimoku['win_rate']:>7.1f}% {metrics_ichimoku['avg_return']:>9.2f}% {metrics_ichimoku['avg_hold']:>8.0f}d")
    print(f"  {'Supertrend':<15} {metrics_supertrend['n_trades']:>8} {metrics_supertrend['win_rate']:>7.1f}% {metrics_supertrend['avg_return']:>9.2f}% {metrics_supertrend['avg_hold']:>8.0f}d")
    print(f"  {'Keltner':<15} {metrics_keltner['n_trades']:>8} {metrics_keltner['win_rate']:>7.1f}% {metrics_keltner['avg_return']:>9.2f}% {metrics_keltner['avg_hold']:>8.0f}d")
    
    print("\n" + "=" * 70)
    print("ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
