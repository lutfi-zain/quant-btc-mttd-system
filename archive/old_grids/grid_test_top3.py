#!/usr/bin/env python3
"""
Grid Test Top 3 Regime-Filtered Systems
========================================

Exhaustive parameter grid search on the 3 best regime-filtered systems:
  1. Ichimoku  + bull_only
  2. Keltner   + bull_with_filters
  3. Supertrend + bull_with_filters

Parameter matrix (120 combos per system, 360 total):
  min_hold:          [15, 20, 25, 30, 35, 40]
  max_hold:          [45, 60, 75, 90, 120]
  regime_threshold:  [0.0, 0.3, 0.5, 0.7]

Holdout split:
  Train: 2018-01-01 to 2024-12-31
  Test:  2025-01-01 to 2026-06-30

Output: mttd/top3_grid_results.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
import importlib.util
from itertools import product
from datetime import datetime

warnings.filterwarnings('ignore')

# ================================================================
# Paths
# ================================================================
project_root = os.path.dirname(os.path.abspath(__file__))
bank_root = '/home/ubuntu/projects/quant-technical-indicator-bank'
sys.path.append(project_root)
sys.path.append(bank_root)
sys.path.append(os.path.join(project_root, 'indicators'))

OUTPUT_DIR = os.path.join(project_root, 'mttd')
REGIME_DATA_PATH = os.path.join(OUTPUT_DIR, 'regime_data.csv')
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'top3_grid_results.csv')

# ================================================================
# Constants
# ================================================================
TRANSACTION_COST = 0.001  # 0.1% round-trip

# Holdout splits
TRAIN_START = '2018-01-01'
TRAIN_END   = '2024-12-31'
TEST_START  = '2025-01-01'
TEST_END    = '2026-06-30'

# Grid search parameters
MIN_HOLD_GRID = [15, 20, 25, 30, 35, 40]
MAX_HOLD_GRID = [45, 60, 75, 90, 120]
REGIME_THRESHOLD_GRID = [0.0, 0.3, 0.5, 0.7]

# The 3 regime-filtered systems to test
TOP3_SYSTEMS = [
    {'name': 'Ichimoku_bull_only',      'base': 'Ichimoku',  'regime_mode': 'bull_only',         'extra_filters': False},
    {'name': 'Keltner_bull_with_filters', 'base': 'Keltner',   'regime_mode': 'bull_with_filters', 'extra_filters': True},
    {'name': 'Supertrend_bull_with_filters', 'base': 'Supertrend', 'regime_mode': 'bull_with_filters', 'extra_filters': True},
]

print("=" * 70)
print("GRID TEST TOP 3 REGIME-FILTERED SYSTEMS")
print("=" * 70)
n_combos_per = len(MIN_HOLD_GRID) * len(MAX_HOLD_GRID) * len(REGIME_THRESHOLD_GRID)
print(f"  Parameter combos per system: {n_combos_per}")
print(f"  Total combos: {n_combos_per * len(TOP3_SYSTEMS)}")
print(f"  min_hold:         {MIN_HOLD_GRID}")
print(f"  max_hold:         {MAX_HOLD_GRID}")
print(f"  regime_threshold: {REGIME_THRESHOLD_GRID}")

# ================================================================
# Helper Functions (spectral / filtering)
# ================================================================

def ehler_supersmoother(series: pd.Series, length: int = 7) -> pd.Series:
    """Ehler's SuperSmoother Filter — spectral noise reduction."""
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


def shannon_entropy(series: pd.Series, window: int = 15, bins: int = 6) -> pd.Series:
    """Shannon Entropy of rolling returns — randomness measure."""
    def calc_shannon(x):
        if len(x) < window:
            return np.nan
        counts, _ = np.histogram(x, bins=bins)
        probs = counts / len(x)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    returns = series.pct_change().fillna(0)
    return returns.rolling(window=window).apply(calc_shannon, raw=True)


def efficiency_ratio(series: pd.Series, period: int = 14) -> pd.Series:
    """Kaufman Efficiency Ratio — trend vs noise."""
    change = series.diff().abs()
    volatility = change.rolling(period).sum()
    direction = series.diff(period).abs()
    return direction / volatility


def compute_cycle_phase(df: pd.DataFrame, lookback: int = 40) -> pd.Series:
    """FFT-based cycle phase computation."""
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
        valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
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
# Shared Filters
# ================================================================

def compute_shared_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Compute filters shared across all base signals."""
    df = df.copy()

    # Cycle Phase (FFT-based)
    phase = compute_cycle_phase(df, lookback=40)
    df['cycle_signal'] = -np.cos(phase)
    df['cycle_direction'] = (df['cycle_signal'] > 0).astype(float)

    # Efficiency Ratio
    df['er'] = efficiency_ratio(df['close'], period=14)
    df['er_gate'] = (df['er'] > 0.20).astype(float)

    # Shannon Entropy
    df['entropy'] = shannon_entropy(df['close'], window=15, bins=6)
    df['entropy_gate'] = (df['entropy'] < 2.8).astype(float)

    # Trend filter: SMA(75) > SMA(250)
    df['trend_fast'] = df['close'].rolling(75, min_periods=1).mean()
    df['trend_slow'] = df['close'].rolling(250, min_periods=1).mean()
    df['trend_filter'] = (df['trend_fast'] > df['trend_slow']).astype(float)

    # Bollinger filter
    bb_mid = df['close'].rolling(25, min_periods=1).mean()
    bb_std = df['close'].rolling(25, min_periods=1).std()
    df['bb_lower'] = bb_mid - 2.0 * bb_std
    df['bb_upper'] = bb_mid + 2.0 * bb_std
    df['bb_filter'] = (
        (df['close'] > df['bb_lower']) & (df['close'] < df['bb_upper'])
    ).astype(float)

    return df


# ================================================================
# Base Signal Generators
# ================================================================

def generate_ichimoku_signal(df: pd.DataFrame) -> pd.Series:
    """Ichimoku signal from ichimoku_quant.py."""
    from ichimoku_quant import generate_ichimoku_features, generate_ichimoku_signals
    df_ich = generate_ichimoku_features(df.copy())
    df_ich = generate_ichimoku_signals(df_ich)
    return df_ich['Pos'].astype(float)


def generate_supertrend_signal(df: pd.DataFrame) -> pd.Series:
    """Supertrend (Median Supertrend Viresearch) signal."""
    spec = importlib.util.spec_from_file_location(
        'supertrend',
        os.path.join(bank_root, 'perpetual/median_supertrend_viresearch.py')
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.median_supertrend_viresearch(df.copy())
    raw = result['vii']
    return (raw > 0).astype(float)


def generate_keltner_signal(df: pd.DataFrame) -> pd.Series:
    """Keltner Channel breakout signal."""
    kc_mid = df['close'].ewm(span=20, adjust=False).mean()
    atr_val = (df['high'] - df['low']).ewm(span=20, adjust=False).mean()
    kc_upper = kc_mid + 1.5 * atr_val
    kc_lower = kc_mid - 1.5 * atr_val
    signal = pd.Series(0.0, index=df.index)
    signal[df['close'] > kc_upper] = 1.0
    return signal


# ================================================================
# Signal Registry (top 3 only)
# ================================================================

BASE_SIGNAL_GENERATORS = {
    'Ichimoku':   generate_ichimoku_signal,
    'Keltner':    generate_keltner_signal,
    'Supertrend': generate_supertrend_signal,
}

# ================================================================
# Position Management
# ================================================================

def apply_position(entry_signal: pd.Series, min_hold: int, max_hold: int) -> pd.Series:
    """
    Apply min/max hold constraints to a raw entry signal.

    Entry triggers when entry_signal == 1 and we are flat.
    Exit triggers when min_hold is reached and entry_signal drops to 0,
    or when max_hold is reached (forced exit).
    """
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
            if hold_count >= min_hold and entry_signal.iloc[i] == 0.0:
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
# Regime Filters
# ================================================================

def apply_regime_filter_bull_only(position: pd.Series, regime_df: pd.DataFrame,
                                   threshold: float) -> pd.Series:
    """Only keep positions where regime composite_score > threshold."""
    aligned = pd.DataFrame({
        'position': position,
        'composite_score': regime_df['composite_score'].reindex(position.index, method='ffill').fillna(0)
    }, index=position.index)
    return aligned['position'] * (aligned['composite_score'] > threshold).astype(float)


def apply_regime_filter_bull_with_filters(position: pd.Series, regime_df: pd.DataFrame,
                                           extra_filters: pd.DataFrame,
                                           threshold: float) -> pd.Series:
    """Regime + trend filter + BB filter combined."""
    aligned = pd.DataFrame({
        'position': position,
        'composite_score': regime_df['composite_score'].reindex(position.index, method='ffill').fillna(0),
        'trend_filter': extra_filters['trend_filter'].reindex(position.index, method='ffill').fillna(0),
        'bb_filter': extra_filters['bb_filter'].reindex(position.index, method='ffill').fillna(0),
    }, index=position.index)

    regime_pass = (aligned['composite_score'] > threshold).astype(float)
    combined_filter = regime_pass * aligned['trend_filter'] * aligned['bb_filter']
    return position * combined_filter


# ================================================================
# Metrics
# ================================================================

def compute_metrics(positions: pd.Series, prices: pd.Series) -> dict:
    """Compute comprehensive trading metrics for a given period."""
    returns = prices.pct_change()
    strategy_returns = returns * positions.shift(1)
    strategy_returns = strategy_returns.dropna()

    # Transaction costs
    transitions = positions.diff().fillna(0)
    strategy_returns = strategy_returns - transitions.loc[strategy_returns.index] * (TRANSACTION_COST / 2)

    if len(strategy_returns) == 0 or strategy_returns.std() == 0:
        return {
            'trades': 0, 'win_rate': 0.0, 'sharpe': 0.0,
            'cagr': 0.0, 'avg_hold': 0.0, 'max_dd': 0.0,
            'total_return': 0.0,
        }

    equity = (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25

    cagr = (equity.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365)

    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    total_return = equity.iloc[-1] - 1.0

    # Count trades and win rate
    in_position = False
    hold_start = None
    trade_returns = []
    hold_periods = []

    for date, pos in positions.items():
        if pos == 1.0 and not in_position:
            in_position = True
            hold_start = date
            entry_price = prices.loc[date]
        elif pos == 0.0 and in_position:
            in_position = False
            if hold_start is not None:
                exit_price = prices.loc[date]
                trade_ret = (exit_price - entry_price) / entry_price
                trade_returns.append(trade_ret)
                hold_periods.append((date - hold_start).days)

    n_trades = len(trade_returns)
    winning = sum(1 for r in trade_returns if r > 0)
    win_rate = winning / n_trades * 100 if n_trades > 0 else 0.0
    avg_hold = np.mean(hold_periods) if hold_periods else 0.0

    return {
        'trades': n_trades,
        'win_rate': round(win_rate, 1),
        'sharpe': round(sharpe, 2),
        'cagr': round(cagr * 100, 2),
        'avg_hold': round(avg_hold, 0),
        'max_dd': round(max_dd * 100, 2),
        'total_return': round(total_return * 100, 2),
    }


# ================================================================
# Main Grid Search
# ================================================================

def main():
    print("\n" + "=" * 70)
    print("STEP 1 / 4 — Load BTC data")
    print("=" * 70)

    with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
        btc_data = json.load(f)

    df = pd.DataFrame(btc_data['aligned_data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df[df.index >= TRAIN_START]
    print(f"  Loaded {len(df)} bars  {df.index[0].date()} → {df.index[-1].date()}")

    print("\n" + "=" * 70)
    print("STEP 2 / 4 — Load regime data")
    print("=" * 70)

    regime_df = pd.read_csv(REGIME_DATA_PATH)
    regime_df['date'] = pd.to_datetime(regime_df['date'])
    regime_df = regime_df.set_index('date')
    regime_df = regime_df[regime_df.index >= TRAIN_START]
    print(f"  Regime rows: {len(regime_df)}")
    for regime in ['Bull', 'Neutral', 'Bear']:
        count = (regime_df['regime'] == regime).sum()
        print(f"    {regime:10s}: {count:5d} ({count / len(regime_df) * 100:.1f}%)")

    print("\n" + "=" * 70)
    print("STEP 3 / 4 — Compute shared filters & generate base signals")
    print("=" * 70)

    df_filters = compute_shared_filters(df)

    base_signals = {}
    for name, gen_func in BASE_SIGNAL_GENERATORS.items():
        try:
            sig = gen_func(df)
            n_active = (sig == 1.0).sum()
            print(f"  {name:12s}: {n_active:5d} active bars ({n_active / len(df) * 100:.1f}%)")
            base_signals[name] = sig
        except Exception as e:
            print(f"  {name:12s}: FAILED — {e}")
            base_signals[name] = pd.Series(0.0, index=df.index)

    print("\n" + "=" * 70)
    print("STEP 4 / 4 — Grid search")
    print("=" * 70)

    results = []
    total_combos = len(TOP3_SYSTEMS) * n_combos_per
    combo_idx = 0

    for system_cfg in TOP3_SYSTEMS:
        system_name = system_cfg['name']
        base_name   = system_cfg['base']
        regime_mode = system_cfg['regime_mode']
        extra       = system_cfg['extra_filters']

        base_signal = base_signals[base_name]
        print(f"\n  ── {system_name} ──")

        for min_hold, max_hold, reg_thresh in product(
            MIN_HOLD_GRID, MAX_HOLD_GRID, REGIME_THRESHOLD_GRID
        ):
            combo_idx += 1

            # Skip invalid: min_hold must be < max_hold
            if min_hold >= max_hold:
                continue

            # Apply min/max hold to base signal
            base_position = apply_position(base_signal, min_hold, max_hold)

            # Apply regime filter
            if regime_mode == 'bull_only':
                final_position = apply_regime_filter_bull_only(
                    base_position, regime_df, reg_thresh
                )
            elif regime_mode == 'bull_with_filters':
                final_position = apply_regime_filter_bull_with_filters(
                    base_position, regime_df, df_filters, reg_thresh
                )
            else:
                final_position = base_position.copy()

            # ── Train/test split ──
            train_mask = (final_position.index >= TRAIN_START) & (final_position.index <= TRAIN_END)
            test_mask  = (final_position.index >= TEST_START)  & (final_position.index <= TEST_END)

            train_pos    = final_position[train_mask]
            test_pos     = final_position[test_mask]
            train_prices = df['close'][train_mask]
            test_prices  = df['close'][test_mask]

            # ── Metrics ──
            train_m = compute_metrics(train_pos, train_prices)
            test_m  = compute_metrics(test_pos, test_prices)

            # ── Degradation ──
            if train_m['sharpe'] != 0:
                degradation = (test_m['sharpe'] - train_m['sharpe']) / abs(train_m['sharpe']) * 100
            else:
                degradation = 0.0 if test_m['sharpe'] == 0 else -100.0

            # ── Win rate change ──
            if train_m['win_rate'] != 0:
                wr_change = (test_m['win_rate'] - train_m['win_rate']) / train_m['win_rate'] * 100
            else:
                wr_change = 0.0

            row = {
                'system':           system_name,
                'min_hold':         min_hold,
                'max_hold':         max_hold,
                'regime_threshold': reg_thresh,
                # Training
                'train_win_rate':   train_m['win_rate'],
                'train_sharpe':     train_m['sharpe'],
                'train_cagr':       train_m['cagr'],
                'train_trades':     train_m['trades'],
                'train_avg_hold':   train_m['avg_hold'],
                'train_max_dd':     train_m['max_dd'],
                'train_total_ret':  train_m['total_return'],
                # Test
                'test_win_rate':    test_m['win_rate'],
                'test_sharpe':      test_m['sharpe'],
                'test_cagr':        test_m['cagr'],
                'test_trades':      test_m['trades'],
                'test_avg_hold':    test_m['avg_hold'],
                'test_max_dd':      test_m['max_dd'],
                'test_total_ret':   test_m['total_return'],
                # Comparison
                'degradation':      round(degradation, 1),
                'win_rate_change':  round(wr_change, 1),
            }
            results.append(row)

            # Progress
            if combo_idx % 60 == 0 or combo_idx == total_combos:
                print(f"    [{combo_idx:3d}/{total_combos}] mh={min_hold:2d} MH={max_hold:2d} "
                      f"rT={reg_thresh:.1f} | T_Sh={train_m['sharpe']:>6.2f} "
                      f"Te_Sh={test_m['sharpe']:>6.2f} Degr={degradation:>+7.1f}%")

    # ── Save results ──
    print("\n" + "=" * 70)
    print("Saving results")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved: {OUTPUT_CSV}")
    print(f"  Total rows: {len(results_df)}")

    # ── Per-system breakdown ──
    print("\n" + "=" * 70)
    print("PER-SYSTEM SUMMARY")
    print("=" * 70)

    for sys_name in results_df['system'].unique():
        sub = results_df[results_df['system'] == sys_name]
        valid = sub[(sub['train_trades'] > 0) & (sub['test_trades'] > 0)]
        print(f"\n  {sys_name}")
        print(f"    Total combos tested:        {len(sub)}")
        print(f"    Combos with trades (both):   {len(valid)}")
        if len(valid) > 0:
            print(f"    Avg test Sharpe:             {valid['test_sharpe'].mean():.2f}")
            print(f"    Avg test WinRate:            {valid['test_win_rate'].mean():.1f}%")
            print(f"    Avg test CAGR:               {valid['test_cagr'].mean():.1f}%")
            print(f"    Avg degradation:             {valid['degradation'].mean():+.1f}%")
            best = valid.nlargest(1, 'test_sharpe').iloc[0]
            print(f"    Best by Sharpe:  mh={int(best['min_hold'])} MH={int(best['max_hold'])} "
                  f"rT={best['regime_threshold']:.1f}  →  Te_Sh={best['test_sharpe']:.2f}  "
                  f"WR={best['test_win_rate']:.1f}%  Degr={best['degradation']:+.1f}%")

    # ── Global top-10 ──
    valid_all = results_df[(results_df['train_trades'] > 0) & (results_df['test_trades'] > 0)]
    print(f"\n  GLOBAL — Top 10 by Test Sharpe (trades>0 both):")
    if len(valid_all) > 0:
        top10 = valid_all.nlargest(10, 'test_sharpe')
        print(f"  {'System':<38s} {'mh':>3s} {'MH':>3s} {'rT':>4s} | {'T_Sh':>6s} {'Te_Sh':>6s} {'Te_WR':>6s} {'CAGR':>6s} {'Degr':>8s}")
        print(f"  {'-'*96}")
        for _, r in top10.iterrows():
            print(f"  {r['system']:<38s} {int(r['min_hold']):>3d} {int(r['max_hold']):>3d} "
                  f"{r['regime_threshold']:>4.1f} | {r['train_sharpe']:>6.2f} "
                  f"{r['test_sharpe']:>6.2f} {r['test_win_rate']:>5.1f}% "
                  f"{r['test_cagr']:>5.1f}% {r['degradation']:>+7.1f}%")

    print("\n" + "=" * 70)
    print("GRID TEST TOP 3 — COMPLETE")
    print(f"Output: {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == '__main__':
    main()
