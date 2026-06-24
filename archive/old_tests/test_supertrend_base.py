#!/usr/bin/env python3
"""
Test Base Supertrend Signal with Common Filtering Framework
============================================================

Base Signal: median_supertrend_viresearch vii > 0 / vii < 0
- Buy: vii > 0 (bullish Supertrend)
- Sell: vii < 0 (bearish Supertrend)

Common Filters (gate_threshold=3, min_hold=25, max_hold=60):
1. MSVR direction (median_standard_deviation_viresearch)
2. SuperSmoother momentum confirmation
3. Cycle Phase timing (FFT-based)
4. Shannon Entropy uncertainty filter
5. Efficiency Ratio trend strength gate
6. Linear regression direction

Performance Metrics: trades, win rate, Sharpe, CAGR, avg hold, max DD
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
print("TEST BASE SUPERTREND SIGNAL — COMMON FILTERING FRAMEWORK")
print("=" * 70)

# ================================================================
# Load BTC Data (2018-2026)
# ================================================================
print("\n[1/6] Loading BTC data...")

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

# --- Filter 1: MSVR Direction ---
print("\n[2/6] Loading MSVR indicator...")

spec = importlib.util.spec_from_file_location(
    'msvr',
    os.path.join(bank_root, 'perpetual/median_standard_deviation_viresearch.py')
)
msvr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msvr_module)
msvr_result = msvr_module.median_standard_deviation_viresearch(df)
df['msvr_vii'] = msvr_result['vii']
df['msvr_direction'] = (df['msvr_vii'] > 0).astype(float)

print(f"  MSVR: {(df['msvr_direction']==1).sum()} bars bullish")


# --- Filter 2: SuperSmoother ---
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


# --- Filter 5: Efficiency Ratio ---
def efficiency_ratio(series, period=14):
    """Kaufman Efficiency Ratio (Family 5: Fractal)."""
    change = series.diff().abs()
    volatility = change.rolling(period).sum()
    direction = series.diff(period).abs()
    return direction / volatility

df['er'] = efficiency_ratio(df['close'], period=14)
df['er_gate'] = (df['er'] > 0.20).astype(float)

print(f"  Efficiency Ratio: {(df['er_gate']==1).sum()} bars trending")


# --- Filter 6: Linear Regression Direction ---
df['lr_value'] = linreg(df['close'], length=50, offset=0)
df['lr_direction'] = (df['close'] > df['lr_value']).astype(float)

print(f"  Linear Regression: {(df['lr_direction']==1).sum()} bars bullish")

# ================================================================
# Gate: at least 3 of 6 filters must pass
# ================================================================
gates = pd.DataFrame({
    'msvr': df['msvr_direction'],
    'smooth': df['smooth_direction'],
    'cycle': df['cycle_direction'],
    'entropy': df['entropy_gate'],
    'er': df['er_gate'],
    'lr': df['lr_direction'],
})
gates_pass = (gates.sum(axis=1) >= 3).astype(float)
print(f"\n  Gate (>=3 of 6): {gates_pass.sum()} bars pass")


# ================================================================
# Layer 1: Supertrend Base Signal
# ================================================================
print("\n[3/6] Computing Supertrend base signal...")

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
# Composite Signal: Base × Gates
# ================================================================
print("\n[4/6] Generating composite signal...")

# Entry: Supertrend buy AND gates pass
df['entry_signal'] = df['st_buy'] * gates_pass

# Exit: Supertrend sell
df['exit_signal'] = df['st_sell']

# ================================================================
# Apply Trade Constraints (min_hold=25, max_hold=60)
# ================================================================
print("\n[5/6] Applying trade constraints...")

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

df['position'] = apply_trade_constraints(
    df['entry_signal'],
    df['exit_signal'],
    min_hold=25,
    max_hold=60
)

position_pct = df['position'].mean() * 100
print(f"  Position: {position_pct:.1f}% of bars in position")


# ================================================================
# Compute Performance Metrics
# ================================================================
print("\n[6/6] Computing performance metrics...")

def compute_metrics(signal, prices, transaction_cost=0.001):
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

metrics = compute_metrics(df['position'], df['close'])

# ================================================================
# Print Results
# ================================================================
print("\n" + "=" * 70)
print("PERFORMANCE METRICS — SUPERTREND BASE")
print("=" * 70)

print(f"\n{'─'*50}")
print(f"  Signal:       Supertrend (vii > 0 / vii < 0)")
print(f"  Filters:      MSVR, SuperSmoother, Cycle Phase, Shannon Entropy, ER, LR")
print(f"  Gate:         >= 3 of 6 filters")
print(f"  Constraints:  min_hold=25, max_hold=60")
print(f"  Costs:        0.1% round-trip")
print(f"{'─'*50}")
print(f"  Trades:       {metrics['n_trades']}")
print(f"  Win Rate:     {metrics['win_rate']:.1f}%")
print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
print(f"  Sortino Ratio:{metrics['sortino']:.2f}")
print(f"  Calmar Ratio: {metrics['calmar']:.2f}")
print(f"  CAGR:         {metrics['cagr']:.1f}%")
print(f"  Max Drawdown: {metrics['max_dd']:.1f}%")
print(f"  Avg Hold:     {metrics['avg_hold']:.0f} days")
print(f"{'─'*50}")

print("\n" + "=" * 70)
print("TEST COMPLETE — SUPERTREND BASE")
print("=" * 70)
