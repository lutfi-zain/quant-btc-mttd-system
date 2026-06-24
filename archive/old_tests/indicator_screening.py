#!/usr/bin/env python3
"""
Indicator Screening Report
==========================

Tests each candidate indicator individually against BTC and identifies those with:
- Low correlation to Ichimoku signals (complementary information)
- High standalone Sharpe ratio (independent alpha)

Outputs ranked results to `mttd/indicator_screening_report.csv`.

Walk-forward: Train 2018-2023, Test 2024-2026
Transaction costs: 0.1% round-trip
"""

import os
import sys
import json
import importlib.util
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ================================================================
# Configuration
# ================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BANK_ROOT = '/home/ubuntu/projects/quant-technical-indicator-bank'
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'mttd')
TRANSACTION_COST = 0.001  # 0.1% round-trip

# Add paths
sys.path.append(PROJECT_ROOT)
sys.path.append(BANK_ROOT)

print("=" * 70)
print("INDICATOR SCREENING REPORT")
print("=" * 70)
print(f"Transaction costs: {TRANSACTION_COST*100:.1f}% round-trip")
print(f"Walk-forward: Train 2018-2023, Test 2024-2026")

# ================================================================
# Load BTC Data
# ================================================================
print("\n[1/5] Loading BTC data...")

with open(os.path.join(PROJECT_ROOT, 'data', 'btc_daily.json')) as f:
    btc_data = json.load(f)

df = pd.DataFrame(btc_data['aligned_data'])
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = df[df.index >= '2018-01-01']
df = df[df['close'] > 0]  # Remove zero-price rows

print(f"  Loaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

# ================================================================
# Generate Ichimoku Baseline Signal
# ================================================================
print("\n[2/5] Generating Ichimoku baseline signal...")

from ichimoku_quant import generate_ichimoku_features, generate_ichimoku_signals

df_ichimoku = generate_ichimoku_features(df.copy())
df_ichimoku = generate_ichimoku_signals(
    df_ichimoku,
    confirm_entry=2,
    confirm_exit=1,
    min_hold_days=10,
    er_entry=0.25,
    t_entry=0.40,
    chikou_thresh=-0.30,
    immunity_thresh=0.50,
    entropy_thresh=2.271
)

ichimoku_signal = df_ichimoku['Pos'].fillna(0)
ichimoku_return = df_ichimoku['close'].pct_change()
ichimoku_strategy_return = ichimoku_return * ichimoku_signal.shift(1)

print(f"  Ichimoku signal: {(ichimoku_signal == 1).sum()} bars in position")

# ================================================================
# Define Indicator Catalog
# ================================================================
print("\n[3/5] Loading and testing indicators...")

# Each entry: (module_name, function_name, signal_column, signal_type, default_kwargs)
# signal_type: 'vii', 'qb', 'dir', 'sig', 'trend', 'back_quant', 'long_short', 'long_short_c'

INDICATOR_CATALOG = [
    # Family 1: Smoothing/Moving Average
    ('alma_lag_viresearch', 'alma_lag_viresearch', 'vii', 'vii', {}),
    ('ewma_viresearch', 'ewma_viresearch', 'vii', 'vii', {}),
    ('dema_sma_standard_deviation_viresearch', 'dema_sma_standard_deviation_viresearch', 'vii', 'vii', {}),
    ('dsma_viresearch', 'dsma_viresearch', 'vii', 'vii', {}),
    ('double_src_sma_standard_deviation_viresearch', 'double_src_sma_standard_deviation_viresearch', 'vii', 'vii', {}),
    ('lsma_viresearch', 'lsma_viresearch', 'vii', 'vii', {}),
    ('lsma_atr_viresearch', 'lsma_atr_viresearch', 'vii', 'vii', {}),
    ('median_standard_deviation_viresearch', 'median_standard_deviation_viresearch', 'vii', 'vii', {}),
    ('median_deviation_suite_investorunknown', 'median_deviation_suite_investorunknown', 'sig', 'sig', {}),
    ('adaptive_volatility_controlled_lsma_quantalgo', 'adaptive_volatility_controlled_lsma_quantalgo', 'trend_direction', 'dir', {}),

    # Family 2: Trend Following
    ('dema_supertrend_viresearch', 'dema_supertrend_viresearch', 'vii', 'vii', {}),
    ('median_supertrend_viresearch', 'median_supertrend_viresearch', 'vii', 'vii', {}),
    ('dema_vstop_viresearch', 'dema_vstop_viresearch', 'vii', 'vii', {}),
    ('vii_stop', 'vii_stop', 'vii', 'vii', {}),
    ('linear_st_quantedgeb', 'linear_st_quantedgeb', 'qb', 'qb', {}),
    ('quantile_dema_trend_quantedgeb', 'quantile_dema_trend_quantedgeb', 'qb', 'qb', {}),
    ('dega_rma_quantedgeb', 'dega_rma_quantedgeb', 'qb', 'qb', {}),
    ('gaussian_smooth_trend_quantedgeb', 'gaussian_smooth_trend_quantedgeb', 'qb', 'qb', {}),

    # Family 3: Volatility Band
    ('dema_rsi_overlay', 'dema_rsi_overlay', 'back_quant', 'back_quant', {}),
    ('inverted_sd_dema_rsi_viresearch', 'inverted_sd_dema_rsi_viresearch', 'vii', 'vii', {}),
    ('polynomial_deviation_bands', 'polynomial_deviation_bands', 'trend', 'trend', {}),
    ('hilo_interpolation_quantedgeb', 'hilo_interpolation_quantedgeb', 'qb', 'qb', {}),

    # Family 4: Momentum/RSI
    ('dema_ema_crossover_viresearch', 'dema_ema_crossover_viresearch', 'vii', 'vii', {}),
    ('dema_dmi_viresearch', 'dema_dmi_viresearch', 'vii', 'vii', {}),
    ('irs_elder_force_volume_index', 'irs_elder_force_volume_index', 'vii', 'vii', {}),

    # Family 5: Volume-Based
    ('volume_trend_swing_points_viresearch', 'volume_trend_swing_points_viresearch', 'vii', 'vii', {}),
    ('p_motion_trend_quantedgeb', 'p_motion_trend_quantedgeb', 'qb', 'qb', {}),

    # Family 6: Multi-Timeframe
    ('ts_aggregated_median_absolute_deviation_tobbysimard', 'ts_aggregated_median_absolute_deviation_tobbysimard', 'long_signal', 'long_short', {}),

    # Family 7: Ichimoku-like
    ('enhanced_kijun_sen_base', 'enhanced_kijun_sen_base', 'long_c', 'long_short_c', {'smf': 'ATR'}),

    # Family 8: Regime Detection
    ('adaptive_regime_cloud', 'adaptive_regime_cloud', 'long_signal', 'long_short', {}),
    ('madtrend_investorunknown', 'madtrend_investorunknown', 'dir', 'dir', {}),
    ('root_mean_square_deviation_trend', 'root_mean_square_deviation_trend', 'direction', 'dir', {}),
]


def load_indicator_module(module_name):
    """Dynamically load an indicator module from the bank."""
    bank_path = os.path.join(BANK_ROOT, 'perpetual', f'{module_name}.py')
    if not os.path.exists(bank_path):
        return None
    spec = importlib.util.spec_from_file_location(module_name, bank_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extract_signal(result_df, signal_col, signal_type):
    """Extract and normalize a binary signal from indicator output."""
    if signal_col not in result_df.columns:
        return None

    raw = result_df[signal_col].copy()

    if signal_type == 'vii':
        # Already 1/-1/0
        return raw.fillna(0).astype(float)
    elif signal_type == 'qb':
        # Already 1/-1
        return raw.fillna(0).astype(float)
    elif signal_type == 'dir':
        # 1/-1/0
        return raw.fillna(0).astype(float)
    elif signal_type == 'sig':
        # 1/-1/0
        return raw.fillna(0).astype(float)
    elif signal_type == 'trend':
        # Positive = bull, Negative = bear, 0 = neutral
        return raw.fillna(0).apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)).astype(float)
    elif signal_type == 'back_quant':
        # 1/-1/0
        return raw.fillna(0).astype(float)
    elif signal_type == 'long_short':
        # True/False long_signal -> position state
        long_sig = result_df.get('long_signal', pd.Series(False, index=result_df.index))
        short_sig = result_df.get('short_signal', pd.Series(False, index=result_df.index))

        # Convert to position state
        pos = np.zeros(len(result_df))
        in_pos = False
        long_vals = long_sig.values if hasattr(long_sig, 'values') else np.array(long_sig)
        short_vals = short_sig.values if hasattr(short_sig, 'values') else np.array(short_sig)

        for i in range(len(pos)):
            if long_vals[i]:
                in_pos = True
            elif short_vals[i]:
                in_pos = False
            pos[i] = 1.0 if in_pos else 0.0

        return pd.Series(pos, index=result_df.index)
    elif signal_type == 'long_short_c':
        # True/False long_c/short_c -> position state
        long_sig = result_df.get('long_c', pd.Series(False, index=result_df.index))
        short_sig = result_df.get('short_c', pd.Series(False, index=result_df.index))

        pos = np.zeros(len(result_df))
        in_pos = False
        long_vals = long_sig.values if hasattr(long_sig, 'values') else np.array(long_sig)
        short_vals = short_sig.values if hasattr(short_sig, 'values') else np.array(short_sig)

        for i in range(len(pos)):
            if long_vals[i] and not in_pos:
                in_pos = True
            elif short_vals[i] and in_pos:
                in_pos = False
            pos[i] = 1.0 if in_pos else 0.0

        return pd.Series(pos, index=result_df.index)

    return None


def compute_metrics(signal, prices, transaction_cost=TRANSACTION_COST):
    """Compute trading metrics for a signal."""
    # Convert directional signal to position state
    # signal: 1/-1/0 directional, or 1.0/0.0 position state
    # We need to detect transitions: 0->1 (entry), 1->0 (exit)
    # For directional signals (1/-1), convert to position: 1 when >0, 0 when <=0
    
    # Check if signal is already position-based (only 0 and 1)
    unique_vals = set(signal.unique())
    if unique_vals.issubset({0.0, 1.0}):
        pos_signal = signal.copy()
    else:
        # Convert directional (1/-1/0) to position (1/0)
        pos_signal = signal.apply(lambda x: 1.0 if x > 0 else 0.0)
    
    returns = prices.pct_change()
    strategy_returns = returns * pos_signal.shift(1)
    strategy_returns = strategy_returns.dropna()

    # Transaction costs
    transitions = pos_signal.diff().fillna(0)
    strategy_returns = strategy_returns - transitions.loc[strategy_returns.index] * (transaction_cost / 2)

    if len(strategy_returns) == 0 or strategy_returns.std() == 0:
        return {
            'sharpe': 0, 'cagr': 0, 'win_rate': 0, 'n_trades': 0,
            'max_dd': 0, 'sortino': 0, 'avg_hold': 0
        }

    equity = (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25

    cagr = (equity.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365)
    downside = strategy_returns[strategy_returns < 0]
    sortino = strategy_returns.mean() / downside.std() * np.sqrt(365) if len(downside) > 0 and downside.std() > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()

    # Count trades and win rate using position state transitions
    in_position = False
    hold_start = None
    hold_periods = []
    trade_returns = []

    for i, (date, pos) in enumerate(pos_signal.items()):
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
        'sharpe': round(sharpe, 2),
        'cagr': round(cagr * 100, 2),
        'win_rate': round(win_rate, 1),
        'n_trades': total,
        'max_dd': round(max_dd * 100, 2),
        'sortino': round(sortino, 2),
        'avg_hold': round(avg_hold, 0)
    }


# ================================================================
# Test Each Indicator
# ================================================================
print("\n  Testing indicators individually...")

results = []

for idx, (module_name, func_name, signal_col, signal_type, kwargs) in enumerate(INDICATOR_CATALOG):
    display_name = module_name.replace('_', ' ').title()

    try:
        mod = load_indicator_module(module_name)
        if mod is None:
            print(f"  [{idx+1:2d}/{len(INDICATOR_CATALOG)}] {display_name}: FILE NOT FOUND")
            continue

        func = getattr(mod, func_name)
        result_df = func(df.copy(), **kwargs)
        signal = extract_signal(result_df, signal_col, signal_type)

        if signal is None:
            print(f"  [{idx+1:2d}/{len(INDICATOR_CATALOG)}] {display_name}: SIGNAL '{signal_col}' NOT FOUND")
            continue

        # Ensure signal is aligned with BTC data
        signal = signal.reindex(df.index).fillna(0)

        # Skip if signal has no variation
        if signal.nunique() <= 1:
            print(f"  [{idx+1:2d}/{len(INDICATOR_CATALOG)}] {display_name}: NO SIGNAL VARIATION")
            continue

        # Compute full-period metrics
        full_metrics = compute_metrics(signal, df['close'])

        # Compute train metrics (2018-2023)
        train_mask = df.index < '2024-01-01'
        train_signal = signal[train_mask]
        train_prices = df['close'][train_mask]
        train_metrics = compute_metrics(train_signal, train_prices)

        # Compute test metrics (2024-2026)
        test_mask = df.index >= '2024-01-01'
        test_signal = signal[test_mask]
        test_prices = df['close'][test_mask]
        test_metrics = compute_metrics(test_signal, test_prices)

        # Compute correlation with Ichimoku
        # For position-based signals (long_short, long_short_c), compare position overlap
        if signal_type in ('long_short', 'long_short_c'):
            # These are already position states (1.0/0.0)
            ich_pos = ichimoku_signal
            ind_pos = signal
        else:
            # For directional signals, convert to position for comparison
            ind_pos = signal.apply(lambda x: 1.0 if x > 0 else 0.0)
            ich_pos = ichimoku_signal

        # Correlation of position states
        valid_mask = (train_mask) & (ind_pos.index >= df.index[0])
        if valid_mask.sum() > 30:
            corr_with_ichimoku = ind_pos[valid_mask].corr(ich_pos[valid_mask])
        else:
            corr_with_ichimoku = 0.0

        if np.isnan(corr_with_ichimoku):
            corr_with_ichimoku = 0.0

        # Signal strength (time in position)
        signal_coverage = (signal != 0).mean() * 100

        results.append({
            'indicator': module_name,
            'display_name': display_name,
            'signal_col': signal_col,
            'signal_type': signal_type,
            'family': '',  # Will fill later
            'full_sharpe': full_metrics['sharpe'],
            'full_cagr': full_metrics['cagr'],
            'full_win_rate': full_metrics['win_rate'],
            'full_trades': full_metrics['n_trades'],
            'full_max_dd': full_metrics['max_dd'],
            'train_sharpe': train_metrics['sharpe'],
            'train_cagr': train_metrics['cagr'],
            'train_win_rate': train_metrics['win_rate'],
            'train_trades': train_metrics['n_trades'],
            'test_sharpe': test_metrics['sharpe'],
            'test_cagr': test_metrics['cagr'],
            'test_win_rate': test_metrics['win_rate'],
            'test_trades': test_metrics['n_trades'],
            'corr_with_ichimoku': round(corr_with_ichimoku, 3),
            'signal_coverage': round(signal_coverage, 1),
            'avg_hold': full_metrics['avg_hold'],
        })

        status = f"Sharpe={full_metrics['sharpe']:.2f}, Trades={full_metrics['n_trades']}, Corr={corr_with_ichimoku:.3f}"
        print(f"  [{idx+1:2d}/{len(INDICATOR_CATALOG)}] {display_name}: {status}")

    except Exception as e:
        print(f"  [{idx+1:2d}/{len(INDICATOR_CATALOG)}] {display_name}: ERROR - {str(e)[:60]}")

# ================================================================
# Rank Indicators
# ================================================================
print("\n[4/5] Ranking indicators...")

results_df = pd.DataFrame(results)

if len(results_df) == 0:
    print("  ERROR: No indicators produced valid signals!")
    sys.exit(1)

# Multi-criteria ranking:
# 1. High standalone Sharpe (weight: 0.4)
# 2. Low absolute correlation with Ichimoku (weight: 0.3)
# 3. Reasonable trade count 10-50 (weight: 0.2)
# 4. High win rate (weight: 0.1)

# Normalize Sharpe to [0, 1] range
sharpe_min = results_df['full_sharpe'].min()
sharpe_max = results_df['full_sharpe'].max()
sharpe_range = sharpe_max - sharpe_min if sharpe_max > sharpe_min else 1.0
results_df['sharpe_score'] = (results_df['full_sharpe'] - sharpe_min) / sharpe_range

# Correlation score: lower is better (0 = no correlation, 1 = perfect correlation)
# Use absolute correlation
results_df['abs_corr'] = results_df['corr_with_ichimoku'].abs()
corr_max = results_df['abs_corr'].max() if results_df['abs_corr'].max() > 0 else 1.0
results_df['corr_score'] = 1.0 - (results_df['abs_corr'] / corr_max)

# Trade count score: penalize if too few (<10) or too many (>50)
results_df['trade_score'] = results_df['full_trades'].apply(
    lambda x: 1.0 if 10 <= x <= 50 else (0.5 if 5 <= x < 10 or 50 < x <= 80 else 0.2)
)

# Win rate score
wr_min = results_df['full_win_rate'].min()
wr_max = results_df['full_win_rate'].max()
wr_range = wr_max - wr_min if wr_max > wr_min else 1.0
results_df['wr_score'] = (results_df['full_win_rate'] - wr_min) / wr_range

# Composite score
results_df['composite_score'] = (
    0.4 * results_df['sharpe_score'] +
    0.3 * results_df['corr_score'] +
    0.2 * results_df['trade_score'] +
    0.1 * results_df['wr_score']
)

# Sort by composite score
results_df = results_df.sort_values('composite_score', ascending=False).reset_index(drop=True)
results_df['rank'] = range(1, len(results_df) + 1)

# ================================================================
# Output Report
# ================================================================
print("\n[5/5] Generating report...")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Select key columns for output
output_cols = [
    'rank', 'indicator', 'display_name', 'signal_type', 'family',
    'full_sharpe', 'full_cagr', 'full_win_rate', 'full_trades', 'full_max_dd',
    'train_sharpe', 'train_win_rate', 'train_trades',
    'test_sharpe', 'test_win_rate', 'test_trades',
    'corr_with_ichimoku', 'signal_coverage', 'composite_score'
]

report_df = results_df[output_cols].copy()
report_path = os.path.join(OUTPUT_DIR, 'indicator_screening_report.csv')
report_df.to_csv(report_path, index=False)

print(f"\n  Report saved: {report_path}")
print(f"  Total indicators tested: {len(results_df)}")

# ================================================================
# Display Top 10
# ================================================================
print("\n" + "=" * 70)
print("TOP 10 INDICATOR CANDIDATES")
print("(Ranked by: High Sharpe + Low Ichimoku Correlation)")
print("=" * 70)

top10 = results_df.head(10)
for _, row in top10.iterrows():
    print(f"\n  #{int(row['rank']):2d} | {row['display_name']}")
    print(f"       Signal: {row['signal_col']} ({row['signal_type']})")
    print(f"       Full Sharpe: {row['full_sharpe']:.2f} | Win Rate: {row['full_win_rate']:.1f}% | Trades: {int(row['full_trades'])}")
    print(f"       Train Sharpe: {row['train_sharpe']:.2f} | Test Sharpe: {row['test_sharpe']:.2f}")
    print(f"       Correlation w/ Ichimoku: {row['corr_with_ichimoku']:.3f}")
    print(f"       CAGR: {row['full_cagr']:.1f}% | Max DD: {row['full_max_dd']:.1f}% | Coverage: {row['signal_coverage']:.1f}%")
    print(f"       Composite Score: {row['composite_score']:.3f}")

# ================================================================
# Summary Statistics
# ================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

top5 = results_df.head(5)
print(f"\n  Top 5 Average Metrics:")
print(f"    Sharpe:     {top5['full_sharpe'].mean():.2f}")
print(f"    Win Rate:   {top5['full_win_rate'].mean():.1f}%")
print(f"    Trades:     {top5['full_trades'].mean():.0f}")
print(f"    CAGR:       {top5['full_cagr'].mean():.1f}%")
print(f"    Max DD:     {top5['full_max_dd'].mean():.1f}%")
print(f"    Ichimoku Corr: {top5['corr_with_ichimoku'].mean():.3f}")

# Correlation analysis
low_corr = results_df[results_df['abs_corr'] < 0.3]
high_sharpe = results_df[results_df['full_sharpe'] > 0.4]
both = results_df[(results_df['abs_corr'] < 0.3) & (results_df['full_sharpe'] > 0.4)]

print(f"\n  Filter Analysis:")
print(f"    Total indicators:       {len(results_df)}")
print(f"    Low Ichimoku corr (<0.3): {len(low_corr)}")
print(f"    High Sharpe (>0.4):      {len(high_sharpe)}")
print(f"    Both (low corr + high Sharpe): {len(both)}")

if len(both) > 0:
    print(f"\n  Best candidates for combination (low corr + high Sharpe):")
    for _, row in both.iterrows():
        print(f"    - {row['display_name']}: Sharpe={row['full_sharpe']:.2f}, Corr={row['corr_with_ichimoku']:.3f}")

# ================================================================
# Save Full Results (JSON)
# ================================================================
full_results_path = os.path.join(OUTPUT_DIR, 'indicator_screening_full.json')
results_df.to_json(full_results_path, orient='records', indent=2)
print(f"\n  Full results: {full_results_path}")

print("\n" + "=" * 70)
print("SCREENING COMPLETE")
print("=" * 70)
