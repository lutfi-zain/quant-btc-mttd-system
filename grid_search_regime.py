#!/usr/bin/env python3
"""
Grid Search with Regime Filtering
===================================

Tests all base signal combinations (MSVR, Ichimoku, Supertrend, Keltner,
Bollinger, ADX) with and without regime filter options. Performs grid search
over min_hold, max_hold, regime_threshold parameters. Runs holdout validation
(2018-2024 train, 2025-2026 test) and outputs results to regime_grid_results.csv.

Regime Data Source: mttd/regime_data.csv (from regime_detector.py)
"""

import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import warnings
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
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'regime_grid_results.csv')

# ================================================================
# Constants
# ================================================================
TRANSACTION_COST = 0.001  # 0.1% round-trip

# Holdout splits
TRAIN_START = '2018-01-01'
TRAIN_END = '2024-12-31'
TEST_START = '2025-01-01'
TEST_END = '2026-06-30'

# Grid search parameters
MIN_HOLD_GRID = [20, 25, 30, 35]
MAX_HOLD_GRID = [50, 60, 75, 90]
REGIME_THRESHOLD_GRID = [0.0, 0.3, 0.5]

# Regime filter modes
REGIME_FILTER_MODES = ['none', 'bull_only', 'bull_with_filters']

print("=" * 70)
print("GRID SEARCH WITH REGIME FILTERING")
print("=" * 70)

# ================================================================
# Helper Functions
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
    df['cycle_signal'] = -np.cos(phase)  # +1 at trough (buy), -1 at peak (sell)
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

def generate_msvr_signal(df: pd.DataFrame) -> pd.Series:
    """MSVR (Median Standard Deviation Viresearch) signal."""
    spec = importlib.util.spec_from_file_location(
        'msvr',
        os.path.join(bank_root, 'perpetual/median_standard_deviation_viresearch.py')
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.median_standard_deviation_viresearch(df)
    raw = result['vii']  # +1 bullish, -1 bearish
    return (raw > 0).astype(float)


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
    raw = result['vii']  # +1 bullish, -1 bearish
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


def generate_bollinger_signal(df: pd.DataFrame) -> pd.Series:
    """Bollinger Band breakout signal."""
    bb_mid = df['close'].rolling(25, min_periods=1).mean()
    bb_std = df['close'].rolling(25, min_periods=1).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    signal = pd.Series(0.0, index=df.index)
    signal[df['close'] > bb_upper] = 1.0
    signal[df['close'] < bb_lower] = -1.0
    return (signal > 0).astype(float)


def generate_adx_signal(df: pd.DataFrame) -> pd.Series:
    """ADX trend-following signal."""
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    atr_val = (df['high'] - df['low']).ewm(span=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_val
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(span=14, adjust=False).mean()
    signal = pd.Series(0.0, index=df.index)
    signal[(adx > 25) & (plus_di > minus_di)] = 1.0
    return signal


# ================================================================
# Signal Registry
# ================================================================

BASE_SIGNALS = {
    'MSVR': generate_msvr_signal,
    'Ichimoku': generate_ichimoku_signal,
    'Supertrend': generate_supertrend_signal,
    'Keltner': generate_keltner_signal,
    'Bollinger': generate_bollinger_signal,
    'ADX': generate_adx_signal,
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
                # Exit: min_hold reached AND signal turned off
                in_position = False
                hold_count = 0
                result.iloc[i] = 0.0
            elif hold_count >= max_hold:
                # Forced exit at max hold
                in_position = False
                hold_count = 0
                result.iloc[i] = 0.0
            else:
                result.iloc[i] = 1.0
        else:
            result.iloc[i] = 0.0

    return result


# ================================================================
# Regime Filter
# ================================================================

def apply_regime_filter(base_position: pd.Series, regime_df: pd.DataFrame,
                        mode: str, threshold: float) -> pd.Series:
    """
    Apply regime filter to position series.

    Modes:
    - 'none': no filtering
    - 'bull_only': only stay in position when composite_score > threshold
    - 'bull_with_filters': regime + trend + BB filters combined
    """
    if mode == 'none':
        return base_position

    # Align regime data with positions
    aligned = pd.DataFrame({
        'position': base_position,
        'composite_score': regime_df['composite_score'].reindex(base_position.index, method='ffill').fillna(0)
    }, index=base_position.index)

    if mode == 'bull_only':
        # Only keep positions where regime is Bull (composite_score > threshold)
        aligned['position'] = aligned['position'] * (aligned['composite_score'] > threshold).astype(float)
    elif mode == 'bull_with_filters':
        # Regime + trend filter + BB filter combined
        # These filters need to be computed on the same df, so we pass them via the caller
        pass  # Handled in the main loop

    return aligned['position']


def apply_regime_filter_with_extra(position: pd.Series, regime_df: pd.DataFrame,
                                    extra_filters: pd.DataFrame,
                                    threshold: float) -> pd.Series:
    """Apply regime + trend + BB filters combined."""
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
            'total_return': 0.0
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
        'total_return': round(total_return * 100, 2)
    }


# ================================================================
# Main Grid Search
# ================================================================

def main():
    # ------------------------------------------------------------
    # Step 1: Load BTC Data
    # ------------------------------------------------------------
    print("\n[1/4] Loading BTC data...")
    with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
        btc_data = json.load(f)

    df = pd.DataFrame(btc_data['aligned_data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df[df.index >= TRAIN_START]
    print(f"  Loaded {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")

    # ------------------------------------------------------------
    # Step 2: Load Regime Data
    # ------------------------------------------------------------
    print("\n[2/4] Loading regime data...")
    regime_df = pd.read_csv(REGIME_DATA_PATH)
    regime_df['date'] = pd.to_datetime(regime_df['date'])
    regime_df = regime_df.set_index('date')
    regime_df = regime_df[regime_df.index >= TRAIN_START]
    print(f"  Regime data: {len(regime_df)} dates")
    print(f"  Regime distribution:")
    for regime in ['Bull', 'Neutral', 'Bear']:
        count = (regime_df['regime'] == regime).sum()
        pct = count / len(regime_df) * 100
        print(f"    {regime:10s}: {count:5d} ({pct:5.1f}%)")

    # ------------------------------------------------------------
    # Step 3: Compute Shared Filters
    # ------------------------------------------------------------
    print("\n[3/4] Computing shared filters...")
    df_filters = compute_shared_filters(df)

    # ------------------------------------------------------------
    # Step 4: Generate Base Signals
    # ------------------------------------------------------------
    print("\n  Generating base signals...")
    base_signals = {}
    for name, gen_func in BASE_SIGNALS.items():
        try:
            base_signals[name] = gen_func(df)
            n_active = (base_signals[name] == 1.0).sum()
            print(f"    {name:12s}: {n_active:5d} active bars ({n_active/len(df)*100:.1f}%)")
        except Exception as e:
            print(f"    {name:12s}: FAILED — {e}")
            base_signals[name] = pd.Series(0.0, index=df.index)

    # ------------------------------------------------------------
    # Step 5: Grid Search
    # ------------------------------------------------------------
    print(f"\n[4/4] Running grid search...")
    print(f"  Base signals:  {len(BASE_SIGNALS)}")
    print(f"  Regime modes:  {REGIME_FILTER_MODES}")
    print(f"  min_hold:      {MIN_HOLD_GRID}")
    print(f"  max_hold:      {MAX_HOLD_GRID}")
    print(f"  regime_thresh: {REGIME_THRESHOLD_GRID}")

    # Calculate total combinations
    total = (len(BASE_SIGNALS) * len(REGIME_FILTER_MODES) *
             len(MIN_HOLD_GRID) * len(MAX_HOLD_GRID) * len(REGIME_THRESHOLD_GRID))
    print(f"  Total combos:  {total}")
    print()

    results = []
    combo_idx = 0

    for base_name, base_signal in base_signals.items():
        for regime_mode in REGIME_FILTER_MODES:
            for min_hold, max_hold, reg_thresh in product(
                MIN_HOLD_GRID, MAX_HOLD_GRID, REGIME_THRESHOLD_GRID
            ):
                combo_idx += 1

                # Skip invalid combos
                if min_hold >= max_hold:
                    continue

                # Apply min/max hold to base signal
                base_position = apply_position(base_signal, min_hold, max_hold)

                # Apply regime filter
                if regime_mode == 'none':
                    final_position = base_position.copy()
                elif regime_mode == 'bull_only':
                    final_position = apply_regime_filter(
                        base_position, regime_df, 'bull_only', reg_thresh
                    )
                elif regime_mode == 'bull_with_filters':
                    final_position = apply_regime_filter_with_extra(
                        base_position, regime_df, df_filters, reg_thresh
                    )
                else:
                    final_position = base_position.copy()

                # Split into train/test
                train_mask = (final_position.index >= TRAIN_START) & (final_position.index <= TRAIN_END)
                test_mask = (final_position.index >= TEST_START) & (final_position.index <= TEST_END)

                train_pos = final_position[train_mask]
                test_pos = final_position[test_mask]
                train_prices = df['close'][train_mask]
                test_prices = df['close'][test_mask]

                # Compute metrics
                train_metrics = compute_metrics(train_pos, train_prices)
                test_metrics = compute_metrics(test_pos, test_prices)

                # Compute degradation
                if train_metrics['sharpe'] != 0:
                    degradation = (test_metrics['sharpe'] - train_metrics['sharpe']) / abs(train_metrics['sharpe']) * 100
                else:
                    degradation = 0.0 if test_metrics['sharpe'] == 0 else -100.0

                # Win rate change
                if train_metrics['win_rate'] != 0:
                    wr_change = (test_metrics['win_rate'] - train_metrics['win_rate']) / train_metrics['win_rate'] * 100
                else:
                    wr_change = 0.0

                row = {
                    'base_signal': base_name,
                    'regime_filter': regime_mode,
                    'min_hold': min_hold,
                    'max_hold': max_hold,
                    'regime_threshold': reg_thresh,
                    # Training metrics
                    'train_trades': train_metrics['trades'],
                    'train_win_rate': train_metrics['win_rate'],
                    'train_sharpe': train_metrics['sharpe'],
                    'train_cagr': train_metrics['cagr'],
                    'train_avg_hold': train_metrics['avg_hold'],
                    'train_max_dd': train_metrics['max_dd'],
                    'train_total_return': train_metrics['total_return'],
                    # Test metrics
                    'test_trades': test_metrics['trades'],
                    'test_win_rate': test_metrics['win_rate'],
                    'test_sharpe': test_metrics['sharpe'],
                    'test_cagr': test_metrics['cagr'],
                    'test_avg_hold': test_metrics['avg_hold'],
                    'test_max_dd': test_metrics['max_dd'],
                    'test_total_return': test_metrics['total_return'],
                    # Comparison
                    'sharpe_degradation': round(degradation, 1),
                    'win_rate_change': round(wr_change, 1),
                }
                results.append(row)

                # Progress update
                if combo_idx % 50 == 0 or combo_idx == total:
                    print(f"  [{combo_idx:4d}/{total}] {base_name:12s} | {regime_mode:20s} | "
                          f"mh={min_hold:2d} MH={max_hold:2d} | rT={reg_thresh:.1f} | "
                          f"Train Sharpe={train_metrics['sharpe']:.2f} | Test Sharpe={test_metrics['sharpe']:.2f}")

    # ------------------------------------------------------------
    # Save Results
    # ------------------------------------------------------------
    print(f"\n  Total results: {len(results)}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  Saved: {OUTPUT_CSV}")

    # ------------------------------------------------------------
    # Summary Statistics
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("GRID SEARCH SUMMARY")
    print("=" * 70)

    # Filter to valid results (trades > 0 in both train and test)
    valid = results_df[(results_df['train_trades'] > 0) & (results_df['test_trades'] > 0)]
    print(f"\n  Valid combos (trades > 0 in both): {len(valid)} / {len(results_df)}")

    if len(valid) > 0:
        # Top 10 by test Sharpe
        top_sharpe = valid.nlargest(10, 'test_sharpe')
        print(f"\n  Top 10 by Test Sharpe:")
        print(f"  {'Signal':<12s} {'Regime':<20s} {'mh':>3s} {'MH':>3s} {'rT':>4s} | "
              f"{'T_Sharpe':>8s} {'Wr%':>5s} | {'Te_Sharpe':>9s} {'Wr%':>5s} | {'Degr%':>6s}")
        print(f"  {'-'*90}")
        for _, r in top_sharpe.iterrows():
            print(f"  {r['base_signal']:<12s} {r['regime_filter']:<20s} {r['min_hold']:>3.0f} {r['max_hold']:>3.0f} "
                  f"{r['regime_threshold']:>4.1f} | {r['train_sharpe']:>8.2f} {r['train_win_rate']:>5.1f} | "
                  f"{r['test_sharpe']:>9.2f} {r['test_win_rate']:>5.1f} | {r['sharpe_degradation']:>6.1f}")

        # Top 10 by test Sharpe with degradation < 30%
        robust = valid[valid['sharpe_degradation'].abs() < 30]
        if len(robust) > 0:
            top_robust = robust.nlargest(10, 'test_sharpe')
            print(f"\n  Top 10 Robust (|degradation| < 30%, by Test Sharpe):")
            print(f"  {'Signal':<12s} {'Regime':<20s} {'mh':>3s} {'MH':>3s} {'rT':>4s} | "
                  f"{'T_Sharpe':>8s} {'Wr%':>5s} | {'Te_Sharpe':>9s} {'Wr%':>5s} | {'Degr%':>6s}")
            print(f"  {'-'*90}")
            for _, r in top_robust.iterrows():
                print(f"  {r['base_signal']:<12s} {r['regime_filter']:<20s} {r['min_hold']:>3.0f} {r['max_hold']:>3.0f} "
                      f"{r['regime_threshold']:>4.1f} | {r['train_sharpe']:>8.2f} {r['train_win_rate']:>5.1f} | "
                      f"{r['test_sharpe']:>9.2f} {r['test_win_rate']:>5.1f} | {r['sharpe_degradation']:>6.1f}")

        # Success criteria: Sharpe > 1.20, Win Rate > 60%, Degradation < 30%
        success = valid[
            (valid['test_sharpe'] > 1.20) &
            (valid['test_win_rate'] > 60) &
            (valid['sharpe_degradation'].abs() < 30)
        ]
        print(f"\n  Success Criteria (Sharpe>1.20, WR>60%, |Degr|<30%): {len(success)} combos")
        if len(success) > 0:
            print(f"  {'Signal':<12s} {'Regime':<20s} {'mh':>3s} {'MH':>3s} {'rT':>4s} | "
                  f"{'T_Sharpe':>8s} {'Wr%':>5s} | {'Te_Sharpe':>9s} {'Wr%':>5s} | {'Degr%':>6s}")
            print(f"  {'-'*90}")
            for _, r in success.iterrows():
                print(f"  {r['base_signal']:<12s} {r['regime_filter']:<20s} {r['min_hold']:>3.0f} {r['max_hold']:>3.0f} "
                      f"{r['regime_threshold']:>4.1f} | {r['train_sharpe']:>8.2f} {r['train_win_rate']:>5.1f} | "
                      f"{r['test_sharpe']:>9.2f} {r['test_win_rate']:>5.1f} | {r['sharpe_degradation']:>6.1f}")

    # Regime filter comparison
    print(f"\n  Average metrics by regime filter mode:")
    for mode in REGIME_FILTER_MODES:
        subset = valid[valid['regime_filter'] == mode] if len(valid) > 0 else results_df[results_df['regime_filter'] == mode]
        if len(subset) > 0:
            avg_t_sharpe = subset['train_sharpe'].mean()
            avg_te_sharpe = subset['test_sharpe'].mean()
            avg_wr = subset['test_win_rate'].mean()
            avg_degr = subset['sharpe_degradation'].mean()
            print(f"    {mode:<20s}: T_Sharpe={avg_t_sharpe:.2f}, Te_Sharpe={avg_te_sharpe:.2f}, "
                  f"WR={avg_wr:.1f}%, AvgDegr={avg_degr:.1f}%")

    # Base signal comparison
    print(f"\n  Average metrics by base signal:")
    for sig_name in BASE_SIGNALS:
        subset = valid[valid['base_signal'] == sig_name] if len(valid) > 0 else results_df[results_df['base_signal'] == sig_name]
        if len(subset) > 0:
            avg_t_sharpe = subset['train_sharpe'].mean()
            avg_te_sharpe = subset['test_sharpe'].mean()
            avg_wr = subset['test_win_rate'].mean()
            avg_degr = subset['sharpe_degradation'].mean()
            print(f"    {sig_name:<12s}: T_Sharpe={avg_t_sharpe:.2f}, Te_Sharpe={avg_te_sharpe:.2f}, "
                  f"WR={avg_wr:.1f}%, AvgDegr={avg_degr:.1f}%")

    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print(f"Results saved to: {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == '__main__':
    main()
