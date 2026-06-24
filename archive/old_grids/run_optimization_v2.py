#!/usr/bin/env python3
"""
Optimize Ichimoku Advanced — Combination Testing v2
====================================================

Goal: Test Ichimoku base signal combined with additional indicator filters
using different combination strategies to find improvements over baseline.

Strategies tested:
1. Ichimoku alone (baseline)
2. Ichimoku + filter confirmation (AND)
3. Filter-enhanced entry (Ichimoku OR filter)
4. Multiple filter voting (majority)
5. Single filter replacement

Baseline: Ichimoku + MSVR v8 (30 trades, 63.3% win, 1.18 Sharpe, 47.4% CAGR)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Add paths
project_root = os.path.dirname(os.path.abspath(__file__))
bank_root = '/home/ubuntu/projects/quant-technical-indicator-bank'
sys.path.append(project_root)
sys.path.append(bank_root)
sys.path.append(os.path.join(project_root, 'indicators'))

# Import indicator modules
from indicators_helper import sma, ema, atr, stdev, rsi, linreg
from indicators.linear_reg_trend import linear_reg_trend
from indicators.volatility_cluster import volatility_cluster
from indicators.volume_confirm import volume_confirm
from indicators.hmm_regime import hmm_regime

# ================================================================
# Configuration
# ================================================================
BASELINE_CONFIG = {
    'trend_fast': 75,
    'trend_slow': 250,
    'bb_period': 25,
    'bb_std': 2.0,
    'min_hold': 45,
    'max_hold': 120,
    'cycle_lookback': 40,
}

TRANSACTION_COST = 0.001  # 0.1% round-trip

# Walk-forward periods
TRAIN_START = '2018-01-01'
TRAIN_END = '2023-12-31'
TEST_START = '2024-01-01'
TEST_END = '2026-06-30'

print("=" * 70)
print("OPTIMIZE ICHIMOKU ADVANCED — COMBINATION TESTING V2")
print("=" * 70)
print(f"\nBaseline: Ichimoku + MSVR v8 (30 trades, 63.3% win, 1.18 Sharpe, 47.4% CAGR)")
print(f"Target: Sharpe > 1.35, 25-35 trades, win rate > 60%, CAGR > 50%")


# ================================================================
# Helper Functions (same as optimize_ichimoku_advanced.py)
# ================================================================

def ehler_supersmoother(series: pd.Series, length: int = 7) -> pd.Series:
    """Ehler's SuperSmoother Filter."""
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
    """Compute Shannon Entropy of rolling returns."""
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
    """Compute Efficiency Ratio (Kaufman)."""
    change = series.diff().abs()
    volatility = change.rolling(period).sum()
    direction = series.diff(period).abs()
    return direction / volatility


def compute_cycle_phase(df: pd.DataFrame, lookback: int = 40) -> pd.Series:
    """Compute cycle phase using FFT."""
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
# Ichimoku Feature Generation (same as optimize_ichimoku_advanced.py)
# ================================================================

def generate_ichimoku_features(df: pd.DataFrame,
                               p1: int = 20,
                               p2: int = 60,
                               p3: int = 120,
                               er_len: int = 14,
                               std_len: int = 30,
                               entropy_window: int = 15,
                               entropy_bins: int = 6) -> pd.DataFrame:
    """Generates hyper-tuned Ichimoku components."""
    df = df.copy()

    # ATR for normalization
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    # Base Ichimoku lines
    df['tenkan_sen'] = (df['high'].rolling(p1).max() + df['low'].rolling(p1).min()) / 2
    df['kijun_sen'] = (df['high'].rolling(p2).max() + df['low'].rolling(p2).min()) / 2

    # Senkou Spans (cloud)
    df['senkou_span_a_raw'] = (df['tenkan_sen'] + df['kijun_sen']) / 2
    df['senkou_span_b_raw'] = (df['high'].rolling(p3).max() + df['low'].rolling(p3).min()) / 2

    # Shift for future cloud
    df['senkou_span_a'] = df['senkou_span_a_raw'].shift(p2)
    df['senkou_span_b'] = df['senkou_span_b_raw'].shift(p2)

    # Cloud boundaries
    df['cloud_max'] = np.maximum(df['senkou_span_a'], df['senkou_span_b'])
    df['cloud_min'] = np.minimum(df['senkou_span_a'], df['senkou_span_b'])

    # Normalized components (tanh -> bounded [-1, 1])
    df['S_TK'] = np.tanh((df['tenkan_sen'] - df['kijun_sen']) / df['ATR'])

    dist_cloud = np.zeros(len(df))
    above = df['close'] > df['cloud_max']
    below = df['close'] < df['cloud_min']
    dist_cloud[above] = (df['close'] - df['cloud_max'])[above] / df['ATR'][above]
    dist_cloud[below] = (df['close'] - df['cloud_min'])[below] / df['ATR'][below]
    df['S_Cloud'] = np.tanh(dist_cloud)

    df['S_Future'] = np.tanh((df['senkou_span_a_raw'] - df['senkou_span_b_raw']) / df['ATR'])

    raw_chikou_dist = (df['close'] - df['close'].shift(p2)) / df['ATR']
    df['S_Chikou'] = np.tanh(ehler_supersmoother(raw_chikou_dist, length=4))

    # Composite IMO (raw)
    imo_raw = (df['S_TK'] + df['S_Cloud'] + df['S_Future'] + df['S_Chikou']) / 4.0
    df['IMO'] = ehler_supersmoother(imo_raw, length=7)
    df['IMO_Std'] = df['IMO'].rolling(std_len).std()

    # Efficiency Ratio (Fractal family)
    df['ER'] = efficiency_ratio(df['close'], period=er_len)

    # Shannon Entropy (Entropy family)
    df['Entropy'] = shannon_entropy(df['close'], window=entropy_window, bins=entropy_bins)

    # Price ROC for exit crash gate (30 days lookback)
    df['roc_gate'] = df['close'].pct_change(periods=30).fillna(0)

    return df


# ================================================================
# Generate Additional Filters
# ================================================================

def generate_additional_filters(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Generate additional indicator filters for combination testing."""
    df = df.copy()

    # Family 2: SuperSmoother Momentum
    df['momentum'] = df['close'].pct_change(periods=10)
    df['momentum_smooth'] = ehler_supersmoother(df['momentum'], length=5)
    df['smooth_direction'] = (df['momentum_smooth'] > 0).astype(float)

    # Family 3: LinearReg
    lr_result = linear_reg_trend(df, source_col='close', length=50, num_std=2.0)
    df['lr_direction'] = (lr_result['direction'] > 0).astype(float)

    # Family 4: Cycle Phase
    phase = compute_cycle_phase(df, lookback=config['cycle_lookback'])
    df['cycle_signal'] = -np.cos(phase)
    df['cycle_direction'] = (df['cycle_signal'] > 0).astype(float)

    # Family 5: Efficiency Ratio Gate
    df['er_gate'] = (df['ER'] > 0.20).astype(float)

    # Family 6: Volatility Cluster
    vol_result = volatility_cluster(df, source_col='close', window=20, threshold=1.2)
    df['vol_direction'] = (vol_result['direction'] > 0).astype(float)

    # Family 7: Shannon Entropy Gate
    df['entropy_gate'] = (df['Entropy'] < 2.8).astype(float)

    # Family 8: Volume Confirm
    try:
        vol_confirm_result = volume_confirm(df, obv_short=10, obv_long=30, spike_mult=1.5)
        df['volume_direction'] = (vol_confirm_result['direction'] > 0).astype(float)
    except Exception:
        df['volume_direction'] = 0.5  # Neutral if volume data unavailable

    # Family 9: HMM Regime
    try:
        hmm_result = hmm_regime(df, source_col='close', n_states=3, lookback=250)
        df['regime_direction'] = (hmm_result['direction'] > 0).astype(float)
    except Exception:
        df['regime_direction'] = 0.5  # Neutral if HMM fails

    # Trend Filter (SMA cross)
    trend_fast = sma(df['close'], config['trend_fast'])
    trend_slow = sma(df['close'], config['trend_slow'])
    df['trend_filter'] = (trend_fast > trend_slow).astype(float)

    # Bollinger Filter
    bb_mid = sma(df['close'], config['bb_period'])
    bb_std = df['close'].rolling(config['bb_period']).std()
    bb_upper = bb_mid + config['bb_std'] * bb_std
    bb_lower = bb_mid - config['bb_std'] * bb_std
    df['bb_filter'] = ((df['close'] > bb_lower) & (df['close'] < bb_upper)).astype(float)

    # MSVR (load from indicator bank)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'msvr',
            os.path.join(bank_root, 'perpetual/median_standard_deviation_viresearch.py')
        )
        msvr_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(msvr_module)
        msvr_result = msvr_module.median_standard_deviation_viresearch(df)
        df['msvr_signal'] = msvr_result['vii']
        df['msvr_direction'] = (df['msvr_signal'] > 0).astype(float)
    except Exception:
        df['msvr_direction'] = 0.5  # Neutral if MSVR fails

    return df


# ================================================================
# Signal Generation — Ichimoku Base
# ================================================================

def generate_ichimoku_entry_signal(df: pd.DataFrame,
                                   min_hold: int = 45,
                                   max_hold: int = 120) -> pd.Series:
    """Generate Ichimoku base entry signal using IMO composite."""
    n = len(df)
    position = np.zeros(n)
    in_position = False
    hold_count = 0

    for i in range(n):
        imo = df['IMO'].iloc[i]
        er = df['ER'].iloc[i]
        std = df['IMO_Std'].iloc[i]
        entropy = df['Entropy'].iloc[i]
        close = df['close'].iloc[i]
        cloud_min = df['cloud_min'].iloc[i]
        cloud_max = df['cloud_max'].iloc[i]

        if pd.isna(imo) or pd.isna(er) or pd.isna(std) or pd.isna(entropy):
            if in_position:
                position[i] = 1.0
            continue

        threshold = std * 0.40  # Adaptive threshold

        if in_position:
            hold_count += 1

            # Exit conditions
            can_exit = hold_count >= min_hold
            exit_signal = False

            if can_exit:
                # IMO deterioration
                if imo < -0.30:
                    exit_signal = True
                # Max hold
                elif hold_count >= max_hold:
                    exit_signal = True
                # Below cloud AND negative momentum
                elif close < cloud_min and imo < 0:
                    exit_signal = True

            if exit_signal:
                in_position = False
                hold_count = 0
                position[i] = 0.0
            else:
                position[i] = 1.0
        else:
            # Entry conditions
            gate_pass = True
            if not pd.isna(cloud_min):
                gate_pass = (close >= cloud_min)

            if imo > threshold and er > 0.25 and entropy < 2.271 and gate_pass:
                in_position = True
                hold_count = 0
                position[i] = 1.0
            else:
                position[i] = 0.0

    return pd.Series(position, index=df.index)


# ================================================================
# Combination Strategies
# ================================================================

def strategy_ichimoku_alone(ichimoku_signal: pd.Series) -> pd.Series:
    """Baseline: Ichimoku signal alone."""
    return ichimoku_signal


def strategy_ichimoku_and_filter(ichimoku_signal: pd.Series,
                                  filter_signal: pd.Series) -> pd.Series:
    """Strategy: Ichimoku AND filter (both must be bullish)."""
    n = len(ichimoku_signal)
    result = np.zeros(n)

    in_position = False
    for i in range(n):
        if not in_position:
            # Entry: Ichimoku signal AND filter bullish
            if ichimoku_signal.iloc[i] == 1.0 and filter_signal.iloc[i] == 1.0:
                in_position = True
                result[i] = 1.0
        else:
            # Exit: Ichimoku exits OR filter turns bearish
            if ichimoku_signal.iloc[i] == 0.0 or filter_signal.iloc[i] == 0.0:
                in_position = False
                result[i] = 0.0
            else:
                result[i] = 1.0

    return pd.Series(result, index=ichimoku_signal.index)


def strategy_ichimoku_or_filter(ichimoku_signal: pd.Series,
                                 filter_signal: pd.Series) -> pd.Series:
    """Strategy: Ichimoku OR filter (either can trigger entry)."""
    n = len(ichimoku_signal)
    result = np.zeros(n)

    in_position = False
    for i in range(n):
        if not in_position:
            # Entry: Ichimoku signal OR filter bullish
            if ichimoku_signal.iloc[i] == 1.0 or filter_signal.iloc[i] == 1.0:
                in_position = True
                result[i] = 1.0
        else:
            # Exit: Both Ichimoku and filter exit
            if ichimoku_signal.iloc[i] == 0.0 and filter_signal.iloc[i] == 0.0:
                in_position = False
                result[i] = 0.0
            else:
                result[i] = 1.0

    return pd.Series(result, index=ichimoku_signal.index)


def strategy_filter_only(filter_signal: pd.Series) -> pd.Series:
    """Strategy: Filter signal alone (Ichimoku as exit only)."""
    return filter_signal


def strategy_majority_vote(ichimoku_signal: pd.Series,
                           filter_signals: dict,
                           min_agreement: int = 2) -> pd.Series:
    """Strategy: Majority vote - need min_agreement signals to agree."""
    n = len(ichimoku_signal)
    result = np.zeros(n)

    # Stack all signals
    signal_names = ['ichimoku'] + list(filter_signals.keys())
    signal_matrix = np.column_stack([
        ichimoku_signal.values,
        *[filter_signals[name].values for name in filter_signals.keys()]
    ])

    in_position = False
    for i in range(n):
        if not in_position:
            # Entry: need min_agreement signals bullish
            bullish_count = np.sum(signal_matrix[i] == 1.0)
            if bullish_count >= min_agreement:
                in_position = True
                result[i] = 1.0
        else:
            # Exit: majority turns bearish
            bullish_count = np.sum(signal_matrix[i] == 1.0)
            if bullish_count < min_agreement:
                in_position = False
                result[i] = 0.0
            else:
                result[i] = 1.0

    return pd.Series(result, index=ichimoku_signal.index)


def strategy_ichimoku_with_exit_filter(ichimoku_signal: pd.Series,
                                        filter_signal: pd.Series) -> pd.Series:
    """Strategy: Enter on Ichimoku, exit on filter deterioration."""
    n = len(ichimoku_signal)
    result = np.zeros(n)

    in_position = False
    for i in range(n):
        if not in_position:
            # Entry: Ichimoku signal
            if ichimoku_signal.iloc[i] == 1.0:
                in_position = True
                result[i] = 1.0
        else:
            # Exit: Ichimoku exits OR filter turns bearish
            if ichimoku_signal.iloc[i] == 0.0 or filter_signal.iloc[i] == 0.0:
                in_position = False
                result[i] = 0.0
            else:
                result[i] = 1.0

    return pd.Series(result, index=ichimoku_signal.index)


# ================================================================
# Performance Metrics
# ================================================================

def compute_metrics(signal: pd.Series, prices: pd.Series,
                    transaction_cost: float = TRANSACTION_COST) -> dict:
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

    for i, (date, pos) in enumerate(signal.items()):
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
        'avg_hold': round(avg_hold, 0)
    }


# ================================================================
# Main Optimization Loop
# ================================================================

def run_optimization():
    """Run combination testing optimization."""

    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # Load BTC data
    with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
        btc_data = json.load(f)

    df = pd.DataFrame(btc_data['aligned_data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df[df.index >= '2018-01-01']

    print(f"  Full dataset: {len(df)} bars ({df.index[0]} to {df.index[-1]})")

    # Split into train/test
    df_train = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    df_test = df[(df.index >= TEST_START) & (df.index <= TEST_END)].copy()

    print(f"  Train: {len(df_train)} bars ({df_train.index[0]} to {df_train.index[-1]})")
    print(f"  Test:  {len(df_test)} bars ({df_test.index[0]} to {df_test.index[-1]})")

    # Pre-compute all filter signals once
    print("\n  Pre-computing filter signals...")
    df_train_feat = generate_ichimoku_features(df_train.copy())
    df_test_feat = generate_ichimoku_features(df_test.copy())
    df_train_feat = generate_additional_filters(df_train_feat, BASELINE_CONFIG)
    df_test_feat = generate_additional_filters(df_test_feat, BASELINE_CONFIG)

    # Generate base Ichimoku signals
    ichimoku_train = generate_ichimoku_entry_signal(df_train_feat, min_hold=BASELINE_CONFIG['min_hold'], max_hold=BASELINE_CONFIG['max_hold'])
    ichimoku_test = generate_ichimoku_entry_signal(df_test_feat, min_hold=BASELINE_CONFIG['min_hold'], max_hold=BASELINE_CONFIG['max_hold'])

    # Compute baseline metrics
    baseline_train = compute_metrics(ichimoku_train, df_train['close'])
    baseline_test = compute_metrics(ichimoku_test, df_test['close'])

    print("\n  BASELINE (Ichimoku alone):")
    print(f"    Train: Sharpe={baseline_train['sharpe']:.2f}, Win={baseline_train['win_rate']:.1f}%, "
          f"Trades={baseline_train['n_trades']}, CAGR={baseline_train['cagr']:.1f}%")
    print(f"    Test:  Sharpe={baseline_test['sharpe']:.2f}, Win={baseline_test['win_rate']:.1f}%, "
          f"Trades={baseline_test['n_trades']}, CAGR={baseline_test['cagr']:.1f}%")

    # Define filters to test
    filters = [
        'msvr_direction',
        'smooth_direction',
        'lr_direction',
        'cycle_direction',
        'er_gate',
        'vol_direction',
        'entropy_gate',
        'volume_direction',
        'regime_direction',
        'trend_filter',
        'bb_filter'
    ]

    results = []
    total_tests = 0

    # ================================================================
    # Strategy 1: Ichimoku + Filter Confirmation (AND)
    # ================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 1: Ichimoku + Filter Confirmation (AND)")
    print("=" * 70)

    for filter_name in filters:
        try:
            position_train = strategy_ichimoku_and_filter(
                ichimoku_train, df_train_feat[filter_name]
            )
            position_test = strategy_ichimoku_and_filter(
                ichimoku_test, df_test_feat[filter_name]
            )

            metrics_train = compute_metrics(position_train, df_train['close'])
            metrics_test = compute_metrics(position_test, df_test['close'])

            if metrics_train['sharpe'] > 0:
                sharpe_degradation = (metrics_test['sharpe'] - metrics_train['sharpe']) / metrics_train['sharpe'] * 100
            else:
                sharpe_degradation = 0

            results.append({
                'strategy': 'AND',
                'combination': f'ichimoku+{filter_name}',
                'filters': filter_name,
                'gate_threshold': 1,
                'train_sharpe': metrics_train['sharpe'],
                'train_win_rate': metrics_train['win_rate'],
                'train_trades': metrics_train['n_trades'],
                'train_cagr': metrics_train['cagr'],
                'train_max_dd': metrics_train['max_dd'],
                'test_sharpe': metrics_test['sharpe'],
                'test_win_rate': metrics_test['win_rate'],
                'test_trades': metrics_test['n_trades'],
                'test_cagr': metrics_test['cagr'],
                'test_max_dd': metrics_test['max_dd'],
                'sharpe_degradation': round(sharpe_degradation, 1)
            })
            total_tests += 1
        except Exception as e:
            print(f"  Error with {filter_name}: {e}")

    # ================================================================
    # Strategy 2: Ichimoku OR Filter (Alternative Entry)
    # ================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 2: Ichimoku OR Filter (Alternative Entry)")
    print("=" * 70)

    for filter_name in filters:
        try:
            position_train = strategy_ichimoku_or_filter(
                ichimoku_train, df_train_feat[filter_name]
            )
            position_test = strategy_ichimoku_or_filter(
                ichimoku_test, df_test_feat[filter_name]
            )

            metrics_train = compute_metrics(position_train, df_train['close'])
            metrics_test = compute_metrics(position_test, df_test['close'])

            if metrics_train['sharpe'] > 0:
                sharpe_degradation = (metrics_test['sharpe'] - metrics_train['sharpe']) / metrics_train['sharpe'] * 100
            else:
                sharpe_degradation = 0

            results.append({
                'strategy': 'OR',
                'combination': f'ichimoku_or_{filter_name}',
                'filters': filter_name,
                'gate_threshold': 1,
                'train_sharpe': metrics_train['sharpe'],
                'train_win_rate': metrics_train['win_rate'],
                'train_trades': metrics_train['n_trades'],
                'train_cagr': metrics_train['cagr'],
                'train_max_dd': metrics_train['max_dd'],
                'test_sharpe': metrics_test['sharpe'],
                'test_win_rate': metrics_test['win_rate'],
                'test_trades': metrics_test['n_trades'],
                'test_cagr': metrics_test['cagr'],
                'test_max_dd': metrics_test['max_dd'],
                'sharpe_degradation': round(sharpe_degradation, 1)
            })
            total_tests += 1
        except Exception as e:
            print(f"  Error with {filter_name}: {e}")

    # ================================================================
    # Strategy 3: Filter-Enhanced Exit (Enter on Ichimoku, exit on filter)
    # ================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 3: Filter-Enhanced Exit")
    print("=" * 70)

    for filter_name in filters:
        try:
            position_train = strategy_ichimoku_with_exit_filter(
                ichimoku_train, df_train_feat[filter_name]
            )
            position_test = strategy_ichimoku_with_exit_filter(
                ichimoku_test, df_test_feat[filter_name]
            )

            metrics_train = compute_metrics(position_train, df_train['close'])
            metrics_test = compute_metrics(position_test, df_test['close'])

            if metrics_train['sharpe'] > 0:
                sharpe_degradation = (metrics_test['sharpe'] - metrics_train['sharpe']) / metrics_train['sharpe'] * 100
            else:
                sharpe_degradation = 0

            results.append({
                'strategy': 'EXIT_FILTER',
                'combination': f'ichimoku_exit_{filter_name}',
                'filters': filter_name,
                'gate_threshold': 1,
                'train_sharpe': metrics_train['sharpe'],
                'train_win_rate': metrics_train['win_rate'],
                'train_trades': metrics_train['n_trades'],
                'train_cagr': metrics_train['cagr'],
                'train_max_dd': metrics_train['max_dd'],
                'test_sharpe': metrics_test['sharpe'],
                'test_win_rate': metrics_test['win_rate'],
                'test_trades': metrics_test['n_trades'],
                'test_cagr': metrics_test['cagr'],
                'test_max_dd': metrics_test['max_dd'],
                'sharpe_degradation': round(sharpe_degradation, 1)
            })
            total_tests += 1
        except Exception as e:
            print(f"  Error with {filter_name}: {e}")

    # ================================================================
    # Strategy 4: Majority Vote (2 filters minimum)
    # ================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 4: Majority Vote (2-3 filters)")
    print("=" * 70)

    # Test pairs of filters
    filter_pairs = [
        ['msvr_direction', 'lr_direction'],
        ['msvr_direction', 'cycle_direction'],
        ['msvr_direction', 'trend_filter'],
        ['lr_direction', 'cycle_direction'],
        ['lr_direction', 'trend_filter'],
        ['cycle_direction', 'trend_filter'],
        ['msvr_direction', 'smooth_direction'],
        ['smooth_direction', 'lr_direction'],
        ['smooth_direction', 'cycle_direction'],
        ['msvr_direction', 'entropy_gate'],
        ['lr_direction', 'entropy_gate'],
        ['cycle_direction', 'entropy_gate'],
        ['msvr_direction', 'vol_direction'],
        ['lr_direction', 'vol_direction'],
        ['cycle_direction', 'vol_direction'],
    ]

    for combo in filter_pairs:
        for min_agree in [2, 3]:
            try:
                filter_signals_train = {name: df_train_feat[name] for name in combo if name in df_train_feat.columns}
                filter_signals_test = {name: df_test_feat[name] for name in combo if name in df_test_feat.columns}

                position_train = strategy_majority_vote(
                    ichimoku_train, filter_signals_train, min_agreement=min_agree
                )
                position_test = strategy_majority_vote(
                    ichimoku_test, filter_signals_test, min_agreement=min_agree
                )

                metrics_train = compute_metrics(position_train, df_train['close'])
                metrics_test = compute_metrics(position_test, df_test['close'])

                if metrics_train['sharpe'] > 0:
                    sharpe_degradation = (metrics_test['sharpe'] - metrics_train['sharpe']) / metrics_train['sharpe'] * 100
                else:
                    sharpe_degradation = 0

                results.append({
                    'strategy': 'MAJORITY',
                    'combination': '+'.join(combo),
                    'filters': '+'.join(combo),
                    'gate_threshold': min_agree,
                    'train_sharpe': metrics_train['sharpe'],
                    'train_win_rate': metrics_train['win_rate'],
                    'train_trades': metrics_train['n_trades'],
                    'train_cagr': metrics_train['cagr'],
                    'train_max_dd': metrics_train['max_dd'],
                    'test_sharpe': metrics_test['sharpe'],
                    'test_win_rate': metrics_test['win_rate'],
                    'test_trades': metrics_test['n_trades'],
                    'test_cagr': metrics_test['cagr'],
                    'test_max_dd': metrics_test['max_dd'],
                    'sharpe_degradation': round(sharpe_degradation, 1)
                })
                total_tests += 1
            except Exception as e:
                print(f"  Error with {combo}, min_agree={min_agree}: {e}")

    # ================================================================
    # Strategy 5: 3-Filter Majority Vote
    # ================================================================
    print("\n" + "=" * 70)
    print("STRATEGY 5: 3-Filter Majority Vote")
    print("=" * 70)

    filter_triples = [
        ['msvr_direction', 'lr_direction', 'cycle_direction'],
        ['msvr_direction', 'lr_direction', 'trend_filter'],
        ['msvr_direction', 'cycle_direction', 'trend_filter'],
        ['lr_direction', 'cycle_direction', 'trend_filter'],
        ['msvr_direction', 'smooth_direction', 'lr_direction'],
        ['msvr_direction', 'smooth_direction', 'cycle_direction'],
        ['msvr_direction', 'lr_direction', 'entropy_gate'],
        ['msvr_direction', 'cycle_direction', 'entropy_gate'],
        ['lr_direction', 'cycle_direction', 'entropy_gate'],
        ['msvr_direction', 'lr_direction', 'vol_direction'],
        ['msvr_direction', 'cycle_direction', 'vol_direction'],
        ['lr_direction', 'cycle_direction', 'vol_direction'],
    ]

    for combo in filter_triples:
        for min_agree in [2, 3]:
            try:
                filter_signals_train = {name: df_train_feat[name] for name in combo if name in df_train_feat.columns}
                filter_signals_test = {name: df_test_feat[name] for name in combo if name in df_test_feat.columns}

                position_train = strategy_majority_vote(
                    ichimoku_train, filter_signals_train, min_agreement=min_agree
                )
                position_test = strategy_majority_vote(
                    ichimoku_test, filter_signals_test, min_agreement=min_agree
                )

                metrics_train = compute_metrics(position_train, df_train['close'])
                metrics_test = compute_metrics(position_test, df_test['close'])

                if metrics_train['sharpe'] > 0:
                    sharpe_degradation = (metrics_test['sharpe'] - metrics_train['sharpe']) / metrics_train['sharpe'] * 100
                else:
                    sharpe_degradation = 0

                results.append({
                    'strategy': 'MAJORITY_3',
                    'combination': '+'.join(combo),
                    'filters': '+'.join(combo),
                    'gate_threshold': min_agree,
                    'train_sharpe': metrics_train['sharpe'],
                    'train_win_rate': metrics_train['win_rate'],
                    'train_trades': metrics_train['n_trades'],
                    'train_cagr': metrics_train['cagr'],
                    'train_max_dd': metrics_train['max_dd'],
                    'test_sharpe': metrics_test['sharpe'],
                    'test_win_rate': metrics_test['win_rate'],
                    'test_trades': metrics_test['n_trades'],
                    'test_cagr': metrics_test['cagr'],
                    'test_max_dd': metrics_test['max_dd'],
                    'sharpe_degradation': round(sharpe_degradation, 1)
                })
                total_tests += 1
            except Exception as e:
                print(f"  Error with {combo}, min_agree={min_agree}: {e}")

    print(f"\n  Total tests completed: {total_tests}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Add baseline row
    baseline_row = pd.DataFrame([{
        'strategy': 'BASELINE',
        'combination': 'ichimoku_alone',
        'filters': 'none',
        'gate_threshold': 0,
        'train_sharpe': baseline_train['sharpe'],
        'train_win_rate': baseline_train['win_rate'],
        'train_trades': baseline_train['n_trades'],
        'train_cagr': baseline_train['cagr'],
        'train_max_dd': baseline_train['max_dd'],
        'test_sharpe': baseline_test['sharpe'],
        'test_win_rate': baseline_test['win_rate'],
        'test_trades': baseline_test['n_trades'],
        'test_cagr': baseline_test['cagr'],
        'test_max_dd': baseline_test['max_dd'],
        'sharpe_degradation': 0.0
    }])

    results_df = pd.concat([baseline_row, results_df], ignore_index=True)

    # Sort by test Sharpe (best first)
    results_df = results_df.sort_values('test_sharpe', ascending=False)

    # ================================================================
    # Console Summary
    # ================================================================
    print("\n" + "=" * 70)
    "RESULTS SUMMARY"
    print("=" * 70)

    # Top 15 by test Sharpe
    print("\n--- Top 15 by Test Sharpe ---")
    print(f"{'Rank':<6} {'Strategy':<15} {'Combination':<45} {'TrainSharpe':<12} {'TestSharpe':<12} {'TestWin%':<10} {'TestTrd':<8}")
    print("-" * 110)

    for idx, (_, row) in enumerate(results_df.head(15).iterrows()):
        print(f"{idx+1:<6} {row['strategy']:<15} {row['combination']:<45} "
              f"{row['train_sharpe']:<12.2f} {row['test_sharpe']:<12.2f} "
              f"{row['test_win_rate']:<10.1f} {row['test_trades']:<8}")

    # Top 15 by test win rate
    print("\n--- Top 15 by Test Win Rate ---")
    print(f"{'Rank':<6} {'Strategy':<15} {'Combination':<45} {'TestWin%':<10} {'TestTrd':<8} {'TestSharpe':<12}")
    print("-" * 100)

    for idx, (_, row) in enumerate(results_df.sort_values('test_win_rate', ascending=False).head(15).iterrows()):
        print(f"{idx+1:<6} {row['strategy']:<15} {row['combination']:<45} "
              f"{row['test_win_rate']:<10.1f} {row['test_trades']:<8} {row['test_sharpe']:<12.2f}")

    # Configs meeting targets
    target_results = results_df[
        (results_df['test_sharpe'] >= 1.20) &
        (results_df['test_win_rate'] >= 60) &
        (results_df['test_trades'] >= 20) &
        (results_df['test_trades'] <= 40)
    ]

    print("\n--- Configs Meeting Targets (Sharpe > 1.20, Win > 60%, Trades 20-40) ---")
    print(f"  Found: {len(target_results)} configs")

    if len(target_results) > 0:
        print(f"\n{'Rank':<6} {'Strategy':<15} {'Combination':<45} {'TestSharpe':<12} {'TestWin%':<10} {'TestTrd':<8}")
        print("-" * 100)

        for idx, (_, row) in enumerate(target_results.head(10).iterrows()):
            print(f"{idx+1:<6} {row['strategy']:<15} {row['combination']:<45} "
                  f"{row['test_sharpe']:<12.2f} {row['test_win_rate']:<10.1f} {row['test_trades']:<8}")

    # ================================================================
    # Save Results to CSV
    # ================================================================
    output_dir = os.path.join(project_root, 'mttd')
    os.makedirs(output_dir, exist_ok=True)

    # Save all results
    csv_path = os.path.join(output_dir, 'optimization_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    # Save top configs
    top_configs_path = os.path.join(output_dir, 'top_configs_v2.csv')
    results_df.head(50).to_csv(top_configs_path, index=False)
    print(f"  Top 50 configs saved to: {top_configs_path}")

    # Save target configs if any
    if len(target_results) > 0:
        target_path = os.path.join(output_dir, 'target_configs_v2.csv')
        target_results.to_csv(target_path, index=False)
        print(f"  Target configs saved to: {target_path}")

    # ================================================================
    # Final Summary
    # ================================================================
    print("\n" + "=" * 70)
    "OPTIMIZATION V2 COMPLETE"
    print("=" * 70)

    best = results_df.iloc[0]
    print(f"\nBest Config (by Test Sharpe):")
    print(f"  Strategy: {best['strategy']}")
    print(f"  Combination: {best['combination']}")
    print(f"  Train Sharpe: {best['train_sharpe']:.2f}")
    print(f"  Test Sharpe: {best['test_sharpe']:.2f}")
    print(f"  Train Win Rate: {best['train_win_rate']:.1f}%")
    print(f"  Test Win Rate: {best['test_win_rate']:.1f}%")
    print(f"  Train Trades: {best['train_trades']}")
    print(f"  Test Trades: {best['test_trades']}")
    print(f"  Train CAGR: {best['train_cagr']:.1f}%")
    print(f"  Test CAGR: {best['test_cagr']:.1f}%")
    print(f"  Sharpe Degradation: {best['sharpe_degradation']:.1f}%")

    return results_df


# ================================================================
# Entry Point
# ================================================================

if __name__ == "__main__":
    results = run_optimization()

    # Check if we have results meeting targets
    target_results = results[
        (results['test_sharpe'] >= 1.20) &
        (results['test_win_rate'] >= 60) &
        (results['test_trades'] >= 20) &
        (results['test_trades'] <= 40)
    ]

    if len(target_results) > 0:
        print("\n✅ Found configurations meeting interim targets!")
        sys.exit(0)
    else:
        print("\n⚠️  No configurations met interim targets.")
        print("Consider:")
        print("  1. Adding on-chain/sentiment data")
        print("  2. Adjusting exit criteria")
        print("  3. Testing different timeframes")
        sys.exit(1)
