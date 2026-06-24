#!/usr/bin/env python3
"""
Optimize Combination — Grid Search over Parameters
====================================================

Uses the best combination approach from combine_bases.py (Supertrend + Keltner)
and performs grid search over:
1. min_hold: [20, 25, 30, 35]
2. max_hold: [50, 60, 75, 90]
3. gate_threshold: [2, 3, 4]

Total combinations: 4 × 4 × 3 = 48

Ranks combinations by:
1. Sharpe ratio
2. Win rate
3. CAGR
4. Composite score (weighted combination)

Identifies the best balanced configuration based on target metrics:
- Sharpe > 1.20
- Win Rate > 55%
- Trades: 25-40
- CAGR > 45%
"""

import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import warnings
from itertools import product
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
bank_root = '/home/ubuntu/projects/quant-technical-indicator-bank'
sys.path.append(project_root)
sys.path.append(bank_root)

from indicators_helper import sma, ema, atr, linreg

# ================================================================
# Configuration
# ================================================================
TRANSACTION_COST = 0.001  # 0.1% round-trip

# Parameter Grid (4 × 4 × 3 = 48 combinations)
PARAM_GRID = {
    'min_hold': [20, 25, 30, 35],
    'max_hold': [50, 60, 75, 90],
    'gate_threshold': [2, 3, 4],
}

# Combination approaches to test
COMBINATION_APPROACHES = ['AND', 'OR', 'Voting', 'Weighted']

print("=" * 70)
print("OPTIMIZE COMBINATION — GRID SEARCH")
print("=" * 70)
print(f"\nCombination: Supertrend + Keltner")
print(f"Parameters:")
for name, values in PARAM_GRID.items():
    print(f"  {name}: {values}")
print(f"Total combinations: {len(PARAM_GRID['min_hold'])} × {len(PARAM_GRID['max_hold'])} × {len(PARAM_GRID['gate_threshold'])} = {len(PARAM_GRID['min_hold']) * len(PARAM_GRID['max_hold']) * len(PARAM_GRID['gate_threshold'])}")
print(f"Transaction Cost: {TRANSACTION_COST*100:.1f}% round-trip")

# ================================================================
# Load BTC Data (2018-2026)
# ================================================================
print("\n[1/10] Loading BTC data...")

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
print("\n[2/10] Computing common filters...")

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
print("\n[3/10] Computing Supertrend base signal...")

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
print("\n[4/10] Computing Keltner Channel base signal...")

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
print("\n[5/10] Combining signals (4 approaches)...")

# --- AND: Both signals must agree ---
df['and_buy'] = ((df['st_buy'] == 1.0) & (df['kc_buy'] == 1.0)).astype(float)
df['and_sell'] = ((df['st_sell'] == 1.0) & (df['kc_sell'] == 1.0)).astype(float)
print(f"  AND buy bars: {int(df['and_buy'].sum())}")

# --- OR: Either signal can trigger ---
df['or_buy'] = ((df['st_buy'] == 1.0) | (df['kc_buy'] == 1.0)).astype(float)
df['or_sell'] = ((df['st_sell'] == 1.0) | (df['kc_sell'] == 1.0)).astype(float)
print(f"  OR buy bars: {int(df['or_buy'].sum())}")

# --- Voting: Majority vote (2 of 3: ST, KC, and combined direction) ---
df['combined_direction'] = (df['st_buy'] + df['kc_buy']) / 2.0
df['voting_buy'] = ((df['st_buy'] + df['kc_buy'] + df['combined_direction'] >= 2.0)).astype(float)
df['voting_sell'] = ((df['st_sell'] + df['kc_sell'] + (1 - df['combined_direction']) >= 2.0)).astype(float)
print(f"  Voting buy bars: {int(df['voting_buy'].sum())}")

# --- Weighted: 50/50 average of signals ---
df['weighted_buy'] = ((df['st_buy'] * 0.5 + df['kc_buy'] * 0.5) > 0.5).astype(float)
df['weighted_sell'] = ((df['st_sell'] * 0.5 + df['kc_sell'] * 0.5) > 0.5).astype(float)
print(f"  Weighted buy bars: {int(df['weighted_buy'].sum())}")


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
        'equity': equity
    }


# ================================================================
# Step 6: Test Each Combination Approach with Default Parameters
# ================================================================
print("\n[6/10] Testing each combination approach with default parameters...")

default_min_hold = 25
default_max_hold = 60
default_gate_threshold = 3

approach_results = {}

for approach in COMBINATION_APPROACHES:
    base_buy = f"{approach.lower()}_buy"
    base_sell = f"{approach.lower()}_sell"
    
    # Gate: sum of 4 filters + base signal (0 or 1)
    base_signal = df[base_buy]
    gate_sum = filters_pass + base_signal
    gate_pass = (gate_sum >= default_gate_threshold).astype(float)
    
    # Entry: gate pass AND base signal buy
    entry_signal = base_signal * gate_pass
    exit_signal = df[base_sell]
    
    # Apply trade constraints
    position = apply_trade_constraints(entry_signal, exit_signal, default_min_hold, default_max_hold)
    
    # Compute metrics
    metrics = compute_metrics(position, df['close'])
    approach_results[approach] = metrics
    
    print(f"  {approach}: Trades={metrics['n_trades']}, WinRate={metrics['win_rate']:.1f}%, "
          f"Sharpe={metrics['sharpe']:.2f}, CAGR={metrics['cagr']:.1f}%")


# ================================================================
# Step 7: Identify Best Combination Approach
# ================================================================
print("\n[7/10] Identifying best combination approach...")

# Calculate composite score for each approach
# Score = 0.4 * Sharpe + 0.3 * (WinRate/100) + 0.3 * (CAGR/100)
for approach, metrics in approach_results.items():
    composite = (0.4 * metrics['sharpe'] + 
                 0.3 * (metrics['win_rate'] / 100) + 
                 0.3 * (metrics['cagr'] / 100))
    approach_results[approach]['composite'] = composite

# Sort by composite score
sorted_approaches = sorted(approach_results.items(), key=lambda x: x[1]['composite'], reverse=True)
best_approach = sorted_approaches[0][0]

print(f"\n  Approach Ranking:")
print(f"  {'Approach':<12} {'Sharpe':>8} {'WinRate':>10} {'CAGR':>10} {'Composite':>10}")
print(f"  {'-'*52}")

for approach, metrics in sorted_approaches:
    print(f"  {approach:<12} {metrics['sharpe']:>8.2f} {metrics['win_rate']:>9.1f}% "
          f"{metrics['cagr']:>9.1f}% {metrics['composite']:>10.3f}")

print(f"\n  ✅ Best approach: {best_approach}")
print(f"     Sharpe={approach_results[best_approach]['sharpe']:.2f}, "
      f"WinRate={approach_results[best_approach]['win_rate']:.1f}%, "
      f"CAGR={approach_results[best_approach]['cagr']:.1f}%")


# ================================================================
# Step 8: Grid Search Over Parameters
# ================================================================
print("\n[8/10] Grid search over parameters...")

# Generate all parameter combinations
param_names = list(PARAM_GRID.keys())
param_values = list(PARAM_GRID.values())
all_combinations = list(product(*param_values))

print(f"  Total combinations: {len(all_combinations)}")
print(f"  Testing approach: {best_approach}")

# Run grid search
results = []

for idx, params in enumerate(all_combinations):
    param_dict = dict(zip(param_names, params))
    
    try:
        # Get base signals for best approach
        base_buy = f"{best_approach.lower()}_buy"
        base_sell = f"{best_approach.lower()}_sell"
        
        # Gate: sum of 4 filters + base signal
        base_signal = df[base_buy]
        gate_sum = filters_pass + base_signal
        gate_pass = (gate_sum >= param_dict['gate_threshold']).astype(float)
        
        # Entry: gate pass AND base signal buy
        entry_signal = base_signal * gate_pass
        exit_signal = df[base_sell]
        
        # Apply trade constraints
        position = apply_trade_constraints(entry_signal, exit_signal, 
                                          param_dict['min_hold'], 
                                          param_dict['max_hold'])
        
        # Compute metrics
        metrics = compute_metrics(position, df['close'])
        
        # Store result
        result = {
            'min_hold': param_dict['min_hold'],
            'max_hold': param_dict['max_hold'],
            'gate_threshold': param_dict['gate_threshold'],
            'approach': best_approach,
            'sharpe': metrics['sharpe'],
            'win_rate': metrics['win_rate'],
            'cagr': metrics['cagr'],
            'n_trades': metrics['n_trades'],
            'avg_hold': metrics['avg_hold'],
            'max_dd': metrics['max_dd'],
            'sortino': metrics['sortino'],
            'calmar': metrics['calmar'],
        }
        
        # Calculate composite score
        result['composite'] = (0.4 * metrics['sharpe'] + 
                               0.3 * (metrics['win_rate'] / 100) + 
                               0.3 * (metrics['cagr'] / 100))
        
        results.append(result)
        
    except Exception as e:
        print(f"  Error with params {param_dict}: {e}")
    
    # Print progress every 12 combinations
    if (idx + 1) % 12 == 0:
        print(f"  Completed {idx + 1}/{len(all_combinations)} combinations...")

print(f"  Total combinations tested: {len(results)}")


# ================================================================
# Step 9: Rank and Analyze Results
# ================================================================
print("\n[9/10] Ranking and analyzing results...")

results_df = pd.DataFrame(results)

# Sort by composite score (best first)
results_df = results_df.sort_values('composite', ascending=False)

# Print all 48 combinations ranked
print("\n" + "=" * 70)
print("ALL 48 COMBINATIONS — RANKED BY COMPOSITE SCORE")
print("=" * 70)

print(f"\n{'Rank':<6} {'MinH':<6} {'MaxH':<6} {'Gate':<6} {'Approach':<12} {'Trades':>8} {'Win%':>8} {'Sharpe':>8} {'CAGR':>8} {'Score':>8}")
print("-" * 90)

for idx, (_, row) in enumerate(results_df.iterrows()):
    marker = " <--" if idx == 0 else ""
    print(f"{idx+1:<6} {row['min_hold']:<6} {row['max_hold']:<6} {row['gate_threshold']:<6} "
          f"{row['approach']:<12} {row['n_trades']:>8} {row['win_rate']:>7.1f}% "
          f"{row['sharpe']:>8.2f} {row['cagr']:>7.1f}% {row['composite']:>8.3f}{marker}")


# ================================================================
# Step 10: Identify Best Balanced Configuration
# ================================================================
print("\n[10/10] Identifying best balanced configuration...")

# Filter for target metrics
# Target: Sharpe > 1.20, Win Rate > 55%, Trades 25-40, CAGR > 45%
target_results = results_df[
    (results_df['sharpe'] >= 1.20) &
    (results_df['win_rate'] >= 55) &
    (results_df['n_trades'] >= 25) &
    (results_df['n_trades'] <= 40) &
    (results_df['cagr'] >= 45)
]

print(f"\n  Configs meeting ALL targets:")
print(f"    Sharpe > 1.20, Win Rate > 55%, Trades 25-40, CAGR > 45%")
print(f"    Found: {len(target_results)} configs")

if len(target_results) > 0:
    print(f"\n  {'Rank':<6} {'MinH':<6} {'MaxH':<6} {'Gate':<6} {'Trades':>8} {'Win%':>8} {'Sharpe':>8} {'CAGR':>8} {'Score':>8}")
    print(f"  {'-'*74}")
    
    for idx, (_, row) in enumerate(target_results.head(10).iterrows()):
        print(f"  {idx+1:<6} {row['min_hold']:<6} {row['max_hold']:<6} {row['gate_threshold']:<6} "
              f"{row['n_trades']:>8} {row['win_rate']:>7.1f}% "
              f"{row['sharpe']:>8.2f} {row['cagr']:>7.1f}% {row['composite']:>8.3f}")

# Best by Sharpe
best_sharpe_row = results_df.iloc[0]  # Already sorted by composite

# Best by Win Rate
best_winrate = results_df[
    (results_df['n_trades'] >= 20) &
    (results_df['n_trades'] <= 40)
].sort_values('win_rate', ascending=False).iloc[0] if len(results_df[
    (results_df['n_trades'] >= 20) &
    (results_df['n_trades'] <= 40)
]) > 0 else None

# Best by CAGR
best_cagr = results_df.iloc[0]  # Already sorted by composite

print("\n" + "=" * 70)
print("BEST CONFIGURATIONS")
print("=" * 70)

print(f"\n📊 Best by Composite Score (Overall Balanced):")
print(f"   min_hold={best_sharpe_row['min_hold']}, max_hold={best_sharpe_row['max_hold']}, "
      f"gate_threshold={best_sharpe_row['gate_threshold']}")
print(f"   Approach: {best_sharpe_row['approach']}")
print(f"   Trades: {best_sharpe_row['n_trades']}, Win Rate: {best_sharpe_row['win_rate']:.1f}%")
print(f"   Sharpe: {best_sharpe_row['sharpe']:.2f}, CAGR: {best_sharpe_row['cagr']:.1f}%")
print(f"   Composite Score: {best_sharpe_row['composite']:.3f}")

if best_winrate is not None:
    print(f"\n📈 Best Win Rate (Trades 20-40):")
    print(f"   min_hold={best_winrate['min_hold']}, max_hold={best_winrate['max_hold']}, "
          f"gate_threshold={best_winrate['gate_threshold']}")
    print(f"   Approach: {best_winrate['approach']}")
    print(f"   Trades: {best_winrate['n_trades']}, Win Rate: {best_winrate['win_rate']:.1f}%")
    print(f"   Sharpe: {best_winrate['sharpe']:.2f}, CAGR: {best_winrate['cagr']:.1f}%")

# Check if we found any config meeting all targets
if len(target_results) > 0:
    best_target = target_results.iloc[0]
    print(f"\n🏆 OPTIMAL CONFIG (Meeting ALL targets):")
    print(f"   min_hold={best_target['min_hold']}, max_hold={best_target['max_hold']}, "
          f"gate_threshold={best_target['gate_threshold']}")
    print(f"   Approach: {best_target['approach']}")
    print(f"   Trades: {best_target['n_trades']}, Win Rate: {best_target['win_rate']:.1f}%")
    print(f"   Sharpe: {best_target['sharpe']:.2f}, CAGR: {best_target['cagr']:.1f}%")
    print(f"   Composite Score: {best_target['composite']:.3f}")
else:
    print(f"\n⚠️  No configs met ALL targets.")
    print(f"   Best Sharpe: {results_df.iloc[0]['sharpe']:.2f}")
    print(f"   Best Win Rate: {results_df.sort_values('win_rate', ascending=False).iloc[0]['win_rate']:.1f}%")
    print(f"   Best CAGR: {results_df.iloc[0]['cagr']:.1f}%")

print("\n" + "=" * 70)
print("OPTIMIZE COMBINATION — COMPLETE")
print("=" * 70)
