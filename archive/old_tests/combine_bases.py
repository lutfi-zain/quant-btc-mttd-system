#!/usr/bin/env python3
"""
Combine Base Signals — Supertrend + Keltner
============================================

Combines two base signals (Supertrend and Keltner Channel) using four approaches:
1. AND: Both signals must agree
2. OR: Either signal can trigger
3. Voting: Majority vote (2 of 3: both bases + combined direction)
4. Weighted: 50/50 average of signals

Common Filters (gate_threshold=3, min_hold=25, max_hold=60):
1. MSVR direction (median_standard_deviation_viresearch)
2. SuperSmoother momentum confirmation
3. Cycle Phase timing (FFT-based)
4. Shannon Entropy uncertainty filter (< 2.8)

Gate: 3 of 5 signals must pass (4 filters + base signal)

Performance Metrics: trades, win rate, Sharpe, CAGR
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

print("=" * 70)
print("COMBINE BASES — SUPERTREND + KELTNER")
print("=" * 70)

# ================================================================
# Configuration
# ================================================================
GATE_THRESHOLD = 3  # Out of 5 signals (4 filters + base)
MIN_HOLD = 25
MAX_HOLD = 60
TRANSACTION_COST = 0.001  # 0.1% round-trip

# ================================================================
# Load BTC Data (2018-2026)
# ================================================================
print("\n[1/8] Loading BTC data...")

with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
    btc_data = json.load(f)

df = pd.DataFrame(btc_data['aligned_data'])
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = df[df.index >= '2018-01-01']

print(f"  Data: {len(df)} bars ({df.index[0]} to {df.index[-1]})")

# ================================================================
# Common Filtering Framework
# ================================================================
print("\n[2/8] Computing common filters...")

# --- Filter 1: MSVR Direction ---
spec_msvr = importlib.util.spec_from_file_location(
    'msvr',
    os.path.join(bank_root, 'perpetual/median_standard_deviation_viresearch.py')
)
msvr_module = importlib.util.module_from_spec(spec_msvr)
spec_msvr.loader.exec_module(msvr_module)
msvr_result = msvr_module.median_standard_deviation_viresearch(df)
df['msvr_vii'] = msvr_result['vii']
df['msvr_direction'] = (df['msvr_vii'] > 0).astype(float)
print(f"  MSVR: {(df['msvr_direction']==1).sum()} bars bullish")


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

df['momentum'] = df['close'].pct_change(periods=10)
df['momentum_smooth'] = ehler_supersmoother(df['momentum'], length=5)
df['smooth_direction'] = (df['momentum_smooth'] > 0).astype(float)
print(f"  SuperSmoother: {(df['smooth_direction']==1).sum()} bars bullish")


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

phase = compute_cycle_phase(df, lookback=40)
df['cycle_signal'] = -np.cos(phase)
df['cycle_direction'] = (df['cycle_signal'] > 0).astype(float)
print(f"  Cycle Phase: {(df['cycle_direction']==1).sum()} bars bullish")


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

df['entropy'] = shannon_entropy(df['close'], window=15, bins=6)
df['entropy_gate'] = (df['entropy'] < 2.8).astype(float)
print(f"  Shannon Entropy: {(df['entropy_gate']==1).sum()} bars low entropy")


# Gate: 4 filters
filters = pd.DataFrame({
    'msvr': df['msvr_direction'],
    'smooth': df['smooth_direction'],
    'cycle': df['cycle_direction'],
    'entropy': df['entropy_gate'],
})
filters_pass = filters.sum(axis=1)
print(f"\n  Filters (sum): {filters_pass.sum():.0f} total bullish bars")


# ================================================================
# Layer 1: Supertrend Base Signal
# ================================================================
print("\n[3/8] Computing Supertrend base signal...")

spec_st = importlib.util.spec_from_file_location(
    'supertrend',
    os.path.join(bank_root, 'perpetual/median_supertrend_viresearch.py')
)
st_module = importlib.util.module_from_spec(spec_st)
spec_st.loader.exec_module(st_module)
st_result = st_module.median_supertrend_viresearch(df)

df['st_vii'] = st_result['vii']
df['st_buy'] = (df['st_vii'] > 0).astype(float)
df['st_sell'] = (df['st_vii'] < 0).astype(float)

print(f"  Supertrend vii mean: {df['st_vii'].mean():.3f}")
print(f"  Raw Supertrend buy bars: {int(df['st_buy'].sum())}")
print(f"  Raw Supertrend sell bars: {int(df['st_sell'].sum())}")


# ================================================================
# Layer 2: Keltner Channel Base Signal
# ================================================================
print("\n[4/8] Computing Keltner Channel base signal...")

KC_PERIOD = 20
KC_ATR_MULT = 1.5

df['kc_mid'] = ema(df['close'], KC_PERIOD)
df['kc_atr'] = ema(df['high'] - df['low'], KC_PERIOD)
df['kc_upper'] = df['kc_mid'] + KC_ATR_MULT * df['kc_atr']
df['kc_lower'] = df['kc_mid'] - KC_ATR_MULT * df['kc_atr']

df['kc_buy'] = (df['close'] > df['kc_upper']).astype(float)
df['kc_sell'] = (df['close'] < df['kc_lower']).astype(float)

print(f"  KC mid: {df['kc_mid'].mean():.2f}")
print(f"  Raw KC buy bars: {int(df['kc_buy'].sum())}")
print(f"  Raw KC sell bars: {int(df['kc_sell'].sum())}")


# ================================================================
# Layer 3: Combine Signals (4 Approaches)
# ================================================================
print("\n[5/8] Combining signals (4 approaches)...")

# --- AND: Both signals must agree ---
df['and_buy'] = ((df['st_buy'] == 1.0) & (df['kc_buy'] == 1.0)).astype(float)
df['and_sell'] = ((df['st_sell'] == 1.0) & (df['kc_sell'] == 1.0)).astype(float)
print(f"  AND buy bars: {int(df['and_buy'].sum())}")

# --- OR: Either signal can trigger ---
df['or_buy'] = ((df['st_buy'] == 1.0) | (df['kc_buy'] == 1.0)).astype(float)
df['or_sell'] = ((df['st_sell'] == 1.0) | (df['kc_sell'] == 1.0)).astype(float)
print(f"  OR buy bars: {int(df['or_buy'].sum())}")

# --- Voting: Majority vote (2 of 3: ST, KC, and combined direction) ---
# Combined direction: average of ST and KC direction signals
df['combined_direction'] = (df['st_buy'] + df['kc_buy']) / 2.0
df['vote_buy'] = ((df['st_buy'] + df['kc_buy'] + df['combined_direction'] >= 2.0)).astype(float)
df['vote_sell'] = ((df['st_sell'] + df['kc_sell'] + (1 - df['combined_direction']) >= 2.0)).astype(float)
print(f"  Voting buy bars: {int(df['vote_buy'].sum())}")

# --- Weighted: 50/50 average of signals ---
df['weighted_buy'] = ((df['st_buy'] * 0.5 + df['kc_buy'] * 0.5) > 0.5).astype(float)
df['weighted_sell'] = ((df['st_sell'] * 0.5 + df['kc_sell'] * 0.5) > 0.5).astype(float)
print(f"  Weighted buy bars: {int(df['weighted_buy'].sum())}")


# ================================================================
# Trade Constraints Function
# ================================================================
def apply_trade_constraints(entry_signal, exit_signal, min_hold=MIN_HOLD, max_hold=MAX_HOLD):
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
        'equity': equity
    }


# ================================================================
# Generate Composite Signals with Gate
# ================================================================
print("\n[6/8] Applying gate filter (threshold={})...".format(GATE_THRESHOLD))

# For each combination approach, compute: gate_pass = (filters + base_signal) >= GATE_THRESHOLD
# where filters is the sum of 4 filters (0-4) and base_signal is 0 or 1

systems = {
    'Supertrend': {'buy': df['st_buy'], 'sell': df['st_sell']},
    'Keltner': {'buy': df['kc_buy'], 'sell': df['kc_sell']},
    'AND': {'buy': df['and_buy'], 'sell': df['and_sell']},
    'OR': {'buy': df['or_buy'], 'sell': df['or_sell']},
    'Voting': {'buy': df['vote_buy'], 'sell': df['vote_sell']},
    'Weighted': {'buy': df['weighted_buy'], 'sell': df['weighted_sell']},
}

results = {}

for name, sigs in systems.items():
    # Gate: sum of 4 filters + base signal (0 or 1)
    base_signal = sigs['buy']
    gate_sum = filters_pass + base_signal
    gate_pass = (gate_sum >= GATE_THRESHOLD).astype(float)
    
    # Entry: gate pass AND base signal buy
    entry_signal = base_signal * gate_pass
    exit_signal = sigs['sell']
    
    # Apply trade constraints
    df[f'{name.lower()}_position'] = apply_trade_constraints(entry_signal, exit_signal)
    
    # Compute metrics
    metrics = compute_metrics(df[f'{name.lower()}_position'], df['close'])
    results[name] = metrics
    
    print(f"  {name}: gate_pass={int(gate_pass.sum())}, position={df[f'{name.lower()}_position'].mean()*100:.1f}%")


# ================================================================
# Print Comparison Table
# ================================================================
print("\n[7/8] Performance metrics...")

print("\n" + "=" * 70)
print("COMBINATION COMPARISON TABLE")
print("=" * 70)

print(f"\n{'─'*70}")
print(f"  Gate: {GATE_THRESHOLD} of 5 (4 filters + base)")
print(f"  Constraints: min_hold={MIN_HOLD}, max_hold={MAX_HOLD}")
print(f"  Costs: {TRANSACTION_COST*100:.1f}% round-trip")
print(f"{'─'*70}")

# Header
header = f"{'System':<20} {'Trades':>8} {'WinRate':>10} {'Sharpe':>8} {'CAGR':>10}"
print(header)
print("-" * 70)

# Data rows
for name, metrics in results.items():
    print(f"{name:<20} {metrics['n_trades']:>8} {metrics['win_rate']:>9.1f}% "
          f"{metrics['sharpe']:>8.2f} {metrics['cagr']:>9.1f}%")

print("-" * 70)
print("=" * 70)


# ================================================================
# Summary and Analysis
# ================================================================
print("\n[8/8] Summary...")

# Find best by Sharpe
best_sharpe_name = max(results, key=lambda x: results[x]['sharpe'])
best_sharpe = results[best_sharpe_name]['sharpe']

# Find best by Win Rate
best_winrate_name = max(results, key=lambda x: results[x]['win_rate'])
best_winrate = results[best_winrate_name]['win_rate']

# Find best by CAGR
best_cagr_name = max(results, key=lambda x: results[x]['cagr'])
best_cagr = results[best_cagr_name]['cagr']

print("\n" + "=" * 70)
print("BEST SYSTEMS")
print("=" * 70)
print(f"  Best Sharpe:    {best_sharpe_name} ({best_sharpe:.2f})")
print(f"  Best Win Rate:  {best_winrate_name} ({best_winrate:.1f}%)")
print(f"  Best CAGR:      {best_cagr_name} ({best_cagr:.1f}%)")
print("=" * 70)

# Overall best
overall_best_name = max(results, key=lambda x: results[x]['sharpe'])
overall_best = results[overall_best_name]
print(f"\n🏆 OVERALL BEST: {overall_best_name}")
print(f"   Trades: {overall_best['n_trades']}, Win Rate: {overall_best['win_rate']:.1f}%, "
      f"Sharpe: {overall_best['sharpe']:.2f}, CAGR: {overall_best['cagr']:.1f}%")

print("\n" + "=" * 70)
print("COMBINE BASES — COMPLETE")
print("=" * 70)
