#!/usr/bin/env python3
"""
Holdout Validation — Combination Strategy
==========================================

Validates the best configuration from optimize_combination.py by splitting
data into training and holdout sets, running backtests on both, and
calculating performance degradation.

Target Metrics:
- Sharpe > 1.20
- Win Rate > 55%
- Trades: 25-40

Degradation Threshold: < 20%
"""

import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
bank_root = '/home/ubuntu/projects/quant-technical-indicator-bank'
sys.path.append(project_root)
sys.path.append(bank_root)

from indicators_helper import sma, ema, atr, linreg

# ================================================================
# Best Configuration from optimize_combination.py
# ================================================================
BEST_CONFIG = {
    'min_hold': 30,
    'max_hold': 60,
    'gate_threshold': 3,
    'approach': 'OR',  # OR combination of Supertrend + Keltner
}

TRANSACTION_COST = 0.001  # 0.1% round-trip
DEGRADATION_THRESHOLD = 0.20  # 20% threshold

# Holdout period boundary
HOLDOUT_START = '2025-01-01'

print("=" * 70)
print("HOLDOUT VALIDATION — COMBINATION STRATEGY")
print("=" * 70)
print(f"\nBest Configuration from optimize_combination.py:")
for k, v in BEST_CONFIG.items():
    print(f"  {k}: {v}")
print(f"\nDegradation Threshold: {DEGRADATION_THRESHOLD*100:.0f}%")
print(f"Holdout Start: {HOLDOUT_START}")

# ================================================================
# Load BTC Data (2018-2026)
# ================================================================
print("\n[1/9] Loading BTC data...")

with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
    btc_data = json.load(f)

df_full = pd.DataFrame(btc_data['aligned_data'])
df_full['time'] = pd.to_datetime(df_full['time'])
df_full = df_full.set_index('time')
df_full = df_full[df_full.index >= '2018-01-01']

print(f"  Full Data: {len(df_full)} bars ({df_full.index[0]} to {df_full.index[-1]})")

# Split into training and holdout
df_train = df_full[df_full.index < HOLDOUT_START].copy()
df_holdout = df_full[df_full.index >= HOLDOUT_START].copy()

print(f"  Training:  {len(df_train)} bars ({df_train.index[0]} to {df_train.index[-1]})")
print(f"  Holdout:   {len(df_holdout)} bars ({df_holdout.index[0]} to {df_holdout.index[-1]})")

# ================================================================
# Common Filtering Framework
# ================================================================
print("\n[2/9] Computing common filters on full data...")

# --- Filter 1: MSVR Direction ---
spec_msvr = importlib.util.spec_from_file_location(
    'msvr',
    os.path.join(bank_root, 'perpetual/median_standard_deviation_viresearch.py')
)
msvr_module = importlib.util.module_from_spec(spec_msvr)
spec_msvr.loader.exec_module(msvr_module)
msvr_result = msvr_module.median_standard_deviation_viresearch(df_full)
df_full['msvr_vii'] = msvr_result['vii']
df_full['msvr_direction'] = (df_full['msvr_vii'] > 0).astype(float)
print(f"  MSVR: {(df_full['msvr_direction']==1).sum()} bars bullish")

# --- Filter 2: SuperSmoother Momentum ---
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

df_full['momentum'] = df_full['close'].pct_change(periods=10)
df_full['momentum_smooth'] = ehler_supersmoother(df_full['momentum'], length=5)
df_full['smooth_direction'] = (df_full['momentum_smooth'] > 0).astype(float)
print(f"  SuperSmoother: {(df_full['smooth_direction']==1).sum()} bars bullish")

# --- Filter 3: Cycle Phase ---
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

phase = compute_cycle_phase(df_full, lookback=40)
df_full['cycle_signal'] = -np.cos(phase)
df_full['cycle_direction'] = (df_full['cycle_signal'] > 0).astype(float)
print(f"  Cycle Phase: {(df_full['cycle_direction']==1).sum()} bars bullish")

# --- Filter 4: Shannon Entropy ---
def shannon_entropy(series, window=15, bins=6):
    """Shannon Entropy of rolling returns (Family 7: Entropy)."""
    def calc_shannon(x):
        if len(x) < window:
            return np.nan
        counts, _ = np.histogram(x, bins=bins)
        probs = counts / len(x)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    returns = series.pct_change().fillna(0)
    return returns.rolling(window=window).apply(calc_shannon, raw=True)

df_full['entropy'] = shannon_entropy(df_full['close'], window=15, bins=6)
df_full['entropy_gate'] = (df_full['entropy'] < 2.8).astype(float)
print(f"  Shannon Entropy: {(df_full['entropy_gate']==1).sum()} bars low entropy")

# Gate: 4 filters
filters_full = pd.DataFrame({
    'msvr': df_full['msvr_direction'],
    'smooth': df_full['smooth_direction'],
    'cycle': df_full['cycle_direction'],
    'entropy': df_full['entropy_gate'],
})
filters_pass_full = filters_full.sum(axis=1)
print(f"\n  Filters (sum): {filters_pass_full.sum():.0f} total bullish bars")

# ================================================================
# Layer 1: Supertrend Base Signal
# ================================================================
print("\n[3/9] Computing Supertrend base signal...")

spec_st = importlib.util.spec_from_file_location(
    'supertrend',
    os.path.join(bank_root, 'perpetual/median_supertrend_viresearch.py')
)
st_module = importlib.util.module_from_spec(spec_st)
spec_st.loader.exec_module(st_module)
st_result = st_module.median_supertrend_viresearch(df_full)

df_full['st_vii'] = st_result['vii']
df_full['st_buy'] = (df_full['st_vii'] > 0).astype(float)
df_full['st_sell'] = (df_full['st_vii'] < 0).astype(float)

print(f"  Supertrend vii mean: {df_full['st_vii'].mean():.3f}")
print(f"  Raw Supertrend buy bars: {int(df_full['st_buy'].sum())}")

# ================================================================
# Layer 2: Keltner Channel Base Signal
# ================================================================
print("\n[4/9] Computing Keltner Channel base signal...")

KC_PERIOD = 20
KC_ATR_MULT = 1.5

df_full['kc_mid'] = ema(df_full['close'], KC_PERIOD)
df_full['kc_atr'] = ema(df_full['high'] - df_full['low'], KC_PERIOD)
df_full['kc_upper'] = df_full['kc_mid'] + KC_ATR_MULT * df_full['kc_atr']
df_full['kc_lower'] = df_full['kc_mid'] - KC_ATR_MULT * df_full['kc_atr']

df_full['kc_buy'] = (df_full['close'] > df_full['kc_upper']).astype(float)
df_full['kc_sell'] = (df_full['close'] < df_full['kc_lower']).astype(float)

print(f"  KC mid: {df_full['kc_mid'].mean():.2f}")
print(f"  Raw KC buy bars: {int(df_full['kc_buy'].sum())}")

# ================================================================
# Layer 3: Combine Signals (OR approach)
# ================================================================
print(f"\n[5/9] Combining signals using {BEST_CONFIG['approach']} approach...")

# --- OR: Either signal can trigger ---
df_full['or_buy'] = ((df_full['st_buy'] == 1.0) | (df_full['kc_buy'] == 1.0)).astype(float)
df_full['or_sell'] = ((df_full['st_sell'] == 1.0) | (df_full['kc_sell'] == 1.0)).astype(float)
print(f"  OR buy bars: {int(df_full['or_buy'].sum())}")
print(f"  OR sell bars: {int(df_full['or_sell'].sum())}")

# ================================================================
# Trade Constraints Function
# ================================================================
def apply_trade_constraints(entry_signal, exit_signal, min_hold=25, max_hold=60):
    """Apply position with min/max hold constraints."""
    result = pd.Series(0.0, index=entry_signal.index)
    in_position = False
    hold_count = 0

    for i in range(len(result)):
        if entry_signal.iloc[i] == 1.0 and not in_position:
            in_position = True
            hold_count = 0
            result.iloc[i] = 1.0
        elif in_position:
            hold_count += 1
            if hold_count >= min_hold and exit_signal.iloc[i] == 1.0:
                in_position = False
                hold_count = 0
                result.iloc[i] = 0.0
            elif hold_count >= max_hold:
                in_position = False
                hold_count = 0
                result.iloc[i] = 0.0
            else:
                result.iloc[i] = 1.0
        else:
            result.iloc[i] = 0.0
    return result


# ================================================================
# Metrics Computation Function
# ================================================================
def compute_metrics(signal, prices, transaction_cost=TRANSACTION_COST):
    """Compute comprehensive trading metrics."""
    returns = prices.pct_change()
    strategy_returns = returns * signal.shift(1)
    strategy_returns = strategy_returns.dropna()

    # Transaction costs
    transitions = signal.diff().fillna(0)
    strategy_returns = strategy_returns - transitions.loc[strategy_returns.index] * (transaction_cost / 2)

    if len(strategy_returns) == 0:
        return {
            'cagr': 0, 'sharpe': 0, 'sortino': 0, 'calmar': 0,
            'max_dd': 0, 'n_trades': 0, 'win_rate': 0, 'avg_hold': 0
        }

    equity = (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25

    cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
    downside = strategy_returns[strategy_returns < 0]
    sortino = strategy_returns.mean() / downside.std() * np.sqrt(365) if len(downside) > 0 and downside.std() > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Count trades and win rate
    in_position = False
    hold_start = None
    hold_periods = []
    trade_returns = []

    for date, pos in signal.items():
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
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'calmar': round(calmar, 2),
        'max_dd': round(max_dd * 100, 2),
        'n_trades': total,
        'win_rate': round(win_rate, 1),
        'avg_hold': round(avg_hold, 0),
    }


# ================================================================
# Run Backtest on Training Data
# ================================================================
print(f"\n[6/9] Running backtest on TRAINING data (2018-2024)...")

# Split full data columns into training and holdout
df_train = df_full[df_full.index < HOLDOUT_START].copy()
df_holdout = df_full[df_full.index >= HOLDOUT_START].copy()

# Compute gate for training
base_buy_train = df_train['or_buy']
base_sell_train = df_train['or_sell']
filters_pass_train = filters_pass_full[df_train.index]

# Gate: sum of 4 filters + base signal
gate_sum_train = filters_pass_train + base_buy_train
gate_pass_train = (gate_sum_train >= BEST_CONFIG['gate_threshold']).astype(float)

# Entry: gate pass AND base signal buy
entry_signal_train = base_buy_train * gate_pass_train
exit_signal_train = base_sell_train

# Apply trade constraints
position_train = apply_trade_constraints(
    entry_signal_train, exit_signal_train,
    BEST_CONFIG['min_hold'], BEST_CONFIG['max_hold']
)

# Compute metrics for training
metrics_train = compute_metrics(position_train, df_train['close'])

print(f"  Training Period: {df_train.index[0].date()} to {df_train.index[-1].date()}")
print(f"  Trades: {metrics_train['n_trades']}")
print(f"  Win Rate: {metrics_train['win_rate']:.1f}%")
print(f"  Sharpe: {metrics_train['sharpe']:.2f}")
print(f"  CAGR: {metrics_train['cagr']:.1f}%")

# ================================================================
# Run Backtest on Holdout Data
# ================================================================
print(f"\n[7/9] Running backtest on HOLDOUT data (2025-2026)...")

# Compute gate for holdout
base_buy_holdout = df_holdout['or_buy']
base_sell_holdout = df_holdout['or_sell']
filters_pass_holdout = filters_pass_full[df_holdout.index]

# Gate: sum of 4 filters + base signal
gate_sum_holdout = filters_pass_holdout + base_buy_holdout
gate_pass_holdout = (gate_sum_holdout >= BEST_CONFIG['gate_threshold']).astype(float)

# Entry: gate pass AND base signal buy
entry_signal_holdout = base_buy_holdout * gate_pass_holdout
exit_signal_holdout = base_sell_holdout

# Apply trade constraints
position_holdout = apply_trade_constraints(
    entry_signal_holdout, exit_signal_holdout,
    BEST_CONFIG['min_hold'], BEST_CONFIG['max_hold']
)

# Compute metrics for holdout
metrics_holdout = compute_metrics(position_holdout, df_holdout['close'])

print(f"  Holdout Period: {df_holdout.index[0].date()} to {df_holdout.index[-1].date()}")
print(f"  Trades: {metrics_holdout['n_trades']}")
print(f"  Win Rate: {metrics_holdout['win_rate']:.1f}%")
print(f"  Sharpe: {metrics_holdout['sharpe']:.2f}")
print(f"  CAGR: {metrics_holdout['cagr']:.1f}%")

# ================================================================
# Calculate Degradation
# ================================================================
print(f"\n[8/9] Calculating performance degradation...")

def calc_degradation(train_val, holdout_val):
    """Calculate degradation: (train - holdout) / train"""
    if train_val == 0:
        return 0
    return (train_val - holdout_val) / train_val

# Degradation metrics
degradations = {}

# Sharpe degradation
degradations['Sharpe'] = {
    'train': metrics_train['sharpe'],
    'holdout': metrics_holdout['sharpe'],
    'deg': calc_degradation(metrics_train['sharpe'], metrics_holdout['sharpe']),
}

# Win Rate degradation
degradations['Win Rate'] = {
    'train': metrics_train['win_rate'],
    'holdout': metrics_holdout['win_rate'],
    'deg': calc_degradation(metrics_train['win_rate'], metrics_holdout['win_rate']),
}

# CAGR degradation
degradations['CAGR'] = {
    'train': metrics_train['cagr'],
    'holdout': metrics_holdout['cagr'],
    'deg': calc_degradation(metrics_train['cagr'], metrics_holdout['cagr']),
}

# Trades count (not degradation, just comparison)
degradations['Trades'] = {
    'train': metrics_train['n_trades'],
    'holdout': metrics_holdout['n_trades'],
    'deg': None,  # Not applicable
}

print("  Degradation calculated for Sharpe, Win Rate, and CAGR")

# ================================================================
# Print Comparison Table
# ================================================================
print(f"\n[9/9] Generating comparison table...")

print("\n" + "=" * 70)
print("HOLDOUT VALIDATION RESULTS")
print("=" * 70)

print(f"\n  Configuration: {BEST_CONFIG['approach']} (min_hold={BEST_CONFIG['min_hold']}, "
      f"max_hold={BEST_CONFIG['max_hold']}, gate={BEST_CONFIG['gate_threshold']})")
print(f"  Training:  {df_train.index[0].date()} to {df_train.index[-1].date()} ({len(df_train)} bars)")
print(f"  Holdout:   {df_holdout.index[0].date()} to {df_holdout.index[-1].date()} ({len(df_holdout)} bars)")
print(f"  Threshold: {DEGRADATION_THRESHOLD*100:.0f}% degradation")

print(f"\n{'─'*70}")
print(f"  {'Metric':<12} {'Training':>12} {'Holdout':>12} {'Degradation':>14} {'Status':>10}")
print(f"{'─'*70}")

# Print each metric
all_pass = True

for metric_name, values in degradations.items():
    train_val = values['train']
    holdout_val = values['holdout']
    deg = values['deg']
    
    if deg is not None:
        # Check if degradation is below threshold
        is_pass = abs(deg) <= DEGRADATION_THRESHOLD
        if not is_pass:
            all_pass = False
        status = "PASS" if is_pass else "FAIL"
        deg_str = f"{deg*100:>+.1f}%"
    else:
        # For trades, just compare
        is_pass = True
        status = "---"
        deg_str = "---"
    
    if metric_name == 'Trades':
        print(f"  {metric_name:<12} {train_val:>12} {holdout_val:>12} {deg_str:>14} {status:>10}")
    elif metric_name == 'Win Rate':
        print(f"  {metric_name:<12} {train_val:>11.1f}% {holdout_val:>11.1f}% {deg_str:>14} {status:>10}")
    elif metric_name == 'CAGR':
        print(f"  {metric_name:<12} {train_val:>11.1f}% {holdout_val:>11.1f}% {deg_str:>14} {status:>10}")
    else:
        print(f"  {metric_name:<12} {train_val:>12.2f} {holdout_val:>12.2f} {deg_str:>14} {status:>10}")

print(f"{'─'*70}")

# Overall status
print(f"\n  Overall Degradation Check: ", end="")
if all_pass:
    print("✅ PASS — All metrics within 20% degradation threshold")
else:
    print("⚠️  FAIL — One or more metrics exceed 20% degradation threshold")

# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n  Best Configuration:")
print(f"    Approach:      {BEST_CONFIG['approach']}")
print(f"    min_hold:      {BEST_CONFIG['min_hold']}")
print(f"    max_hold:      {BEST_CONFIG['max_hold']}")
print(f"    gate_threshold: {BEST_CONFIG['gate_threshold']}")

print(f"\n  Training Performance:")
print(f"    Sharpe:   {metrics_train['sharpe']:.2f}")
print(f"    Win Rate: {metrics_train['win_rate']:.1f}%")
print(f"    CAGR:     {metrics_train['cagr']:.1f}%")
print(f"    Trades:   {metrics_train['n_trades']}")

print(f"\n  Holdout Performance:")
print(f"    Sharpe:   {metrics_holdout['sharpe']:.2f}")
print(f"    Win Rate: {metrics_holdout['win_rate']:.1f}%")
print(f"    CAGR:     {metrics_holdout['cagr']:.1f}%")
print(f"    Trades:   {metrics_holdout['n_trades']}")

print(f"\n  Degradation:")
for metric_name in ['Sharpe', 'Win Rate', 'CAGR']:
    deg = degradations[metric_name]['deg']
    if deg is not None:
        print(f"    {metric_name}: {deg*100:>+.1f}%")

print(f"\n{'='*70}")
print("HOLDOUT VALIDATION — COMPLETE")
print(f"{'='*70}")
