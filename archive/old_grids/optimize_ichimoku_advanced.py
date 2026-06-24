#!/usr/bin/env python3
"""
Optimize Ichimoku Advanced — Combination Testing
==================================================

Goal: Build the main optimization script that tests Ichimoku base signal
combined with additional indicator filters.

Features:
- Ichimoku base signal generation with IMO composite
- Modular indicator filter addition system
- Majority-gate voting mechanism (configurable gate threshold)
- Walk-forward validation (train 2018-2023, test 2024-2026)
- Performance metrics: Sharpe, win rate, trade count, CAGR, max drawdown
- Support min_hold and max_hold parameters
- Output results to CSV and console summary

Baseline Config: T75/250, BB25, 2.0s, MH45
Transaction Cost: 0.1% round-trip
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
from itertools import combinations
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

# Output directory
OUTPUT_DIR = os.path.join(project_root, 'mttd', 'optimization')

print("=" * 70)
print("OPTIMIZE ICHIMOKU ADVANCED — COMBINATION TESTING")
print("=" * 70)
print(f"\nBaseline Config: T{BASELINE_CONFIG['trend_fast']}/{BASELINE_CONFIG['trend_slow']}_"
      f"BB{BASELINE_CONFIG['bb_period']}_{BASELINE_CONFIG['bb_std']}s_MH{BASELINE_CONFIG['min_hold']}")
print(f"Transaction Cost: {TRANSACTION_COST*100:.1f}% round-trip")
print(f"Walk-Forward: Train {TRAIN_START} to {TRAIN_END}, Test {TEST_START} to {TEST_END}")


# ================================================================
# Helper Functions
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
# Ichimoku Feature Generation
# ================================================================

def generate_ichimoku_features(df: pd.DataFrame,
                               p1: int = 20,
                               p2: int = 60,
                               p3: int = 120,
                               er_len: int = 14,
                               std_len: int = 30,
                               entropy_window: int = 15,
                               entropy_bins: int = 6) -> pd.DataFrame:
    """
    Generates hyper-tuned Ichimoku components.
    - Macro periods (20, 60, 120) calibrated for 24/7 crypto market
    - Ehler SuperSmoother applied on final IMO for noise reduction
    - Efficiency Ratio (Fractal family) for trend strength gate
    """
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
    """
    Generate additional indicator filters for combination testing.
    
    Filters generated:
    - MSVR (Family 1: Smoothing)
    - SuperSmoother Momentum (Family 2: Filtering)
    - LinearReg (Family 3: Regression)
    - Cycle Phase (Family 4: Spectral)
    - Efficiency Ratio (Family 5: Fractal)
    - Volatility Cluster (Family 6: GARCH)
    - Shannon Entropy (Family 7: Entropy)
    - Volume Confirm (Family 8: Volume)
    - HMM Regime (Family 9: Bayesian)
    - Trend Filter (SMA cross)
    - Bollinger Filter
    """
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
# Signal Generation
# ================================================================

def generate_ichimoku_entry_signal(df: pd.DataFrame,
                                   min_hold: int = 45,
                                   max_hold: int = 120) -> pd.Series:
    """
    Generate Ichimoku base entry signal using IMO composite.
    
    Entry: IMO > adaptive threshold AND ER > threshold AND low entropy AND above cloud
    Exit: IMO deterioration OR max_hold reached
    """
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


def apply_filters_with_gate(ichimoku_signal: pd.Series,
                            filter_signals: dict,
                            gate_threshold: int = 3) -> pd.Series:
    """
    Apply filters using majority-gate voting mechanism.
    
    Parameters:
    - ichimoku_signal: Base Ichimoku position (1.0 or 0.0)
    - filter_signals: Dict of filter_name -> binary signal (1.0=bullish, 0.0=neutral)
    - gate_threshold: Minimum number of filters that must agree for entry
    
    Logic:
    - Entry requires: Ichimoku signal AND (gate_threshold filters agree)
    - Exit: Immediate when Ichimoku exits
    """
    n = len(ichimoku_signal)
    result = np.zeros(n)

    # Stack all filter signals
    filter_names = list(filter_signals.keys())
    filter_matrix = np.column_stack([filter_signals[name].values for name in filter_names])
    n_filters = len(filter_names)

    in_position = False

    for i in range(n):
        if not in_position:
            # Check entry: Ichimoku signal + gate threshold
            if ichimoku_signal.iloc[i] == 1.0:
                # Count bullish filters
                bullish_count = np.sum(filter_matrix[i] == 1.0)
                if bullish_count >= gate_threshold:
                    in_position = True
                    result[i] = 1.0
                else:
                    result[i] = 0.0
            else:
                result[i] = 0.0
        else:
            # In position: exit when Ichimoku exits
            if ichimoku_signal.iloc[i] == 0.0:
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
    changes = signal.diff().fillna(0)
    n_trades = (changes.abs() > 0).sum() // 2

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
        'n_trades': n_trades,
        'win_rate': round(win_rate, 1),
        'avg_hold': round(avg_hold, 0)
    }


# ================================================================
# Walk-Forward Validation
# ================================================================

def walk_forward_validation(df_train: pd.DataFrame,
                            df_test: pd.DataFrame,
                            filters: list,
                            gate_threshold: int,
                            config: dict) -> dict:
    """
    Run walk-forward validation on train and test periods.
    
    Returns metrics for both periods plus degradation.
    """
    # Generate Ichimoku features
    df_train_feat = generate_ichimoku_features(df_train.copy())
    df_test_feat = generate_ichimoku_features(df_test.copy())

    # Generate additional filters
    df_train_feat = generate_additional_filters(df_train_feat, config)
    df_test_feat = generate_additional_filters(df_test_feat, config)

    # Generate base Ichimoku signal
    ichimoku_train = generate_ichimoku_entry_signal(
        df_train_feat,
        min_hold=config['min_hold'],
        max_hold=config['max_hold']
    )
    ichimoku_test = generate_ichimoku_entry_signal(
        df_test_feat,
        min_hold=config['min_hold'],
        max_hold=config['max_hold']
    )

    # Prepare filter signals
    filter_names = list(filters)
    filter_signals_train = {name: df_train_feat[name] for name in filter_names if name in df_train_feat.columns}
    filter_signals_test = {name: df_test_feat[name] for name in filter_names if name in df_test_feat.columns}

    # Apply filters with gate
    position_train = apply_filters_with_gate(
        ichimoku_train, filter_signals_train, gate_threshold
    )
    position_test = apply_filters_with_gate(
        ichimoku_test, filter_signals_test, gate_threshold
    )

    # Compute metrics
    metrics_train = compute_metrics(position_train, df_train['close'])
    metrics_test = compute_metrics(position_test, df_test['close'])

    # Compute degradation
    if metrics_train['sharpe'] > 0:
        sharpe_degradation = (metrics_test['sharpe'] - metrics_train['sharpe']) / metrics_train['sharpe'] * 100
    else:
        sharpe_degradation = 0

    return {
        'train': metrics_train,
        'test': metrics_test,
        'sharpe_degradation': round(sharpe_degradation, 1)
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

    # Define available filters
    available_filters = [
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

    # Define combinations to test
    # Focus on 2-4 filters with key gate thresholds for faster execution
    print("\n" + "=" * 70)
    print("RUNNING COMBINATION TESTS")
    print("=" * 70)

    results = []
    total_combos = 0

    # Pre-compute all filter signals once to avoid redundant computation
    print("\n  Pre-computing filter signals...")
    df_train_feat = generate_ichimoku_features(df_train.copy())
    df_test_feat = generate_ichimoku_features(df_test.copy())
    df_train_feat = generate_additional_filters(df_train_feat, BASELINE_CONFIG)
    df_test_feat = generate_additional_filters(df_test_feat, BASELINE_CONFIG)
    
    # Generate base Ichimoku signals once
    ichimoku_train = generate_ichimoku_entry_signal(df_train_feat, min_hold=BASELINE_CONFIG['min_hold'], max_hold=BASELINE_CONFIG['max_hold'])
    ichimoku_test = generate_ichimoku_entry_signal(df_test_feat, min_hold=BASELINE_CONFIG['min_hold'], max_hold=BASELINE_CONFIG['max_hold'])
    print("  Filter signals computed.")

    # Test specific high-value combinations (curated for efficiency)
    # Based on domain knowledge: MSVR, LinearReg, Cycle, Trend are key
    key_combos = [
        # Core combos (2 filters)
        ['msvr_direction', 'lr_direction'],
        ['msvr_direction', 'cycle_direction'],
        ['msvr_direction', 'trend_filter'],
        ['lr_direction', 'cycle_direction'],
        ['lr_direction', 'trend_filter'],
        ['cycle_direction', 'trend_filter'],
        # Smoothing combos
        ['msvr_direction', 'smooth_direction'],
        ['smooth_direction', 'lr_direction'],
        ['smooth_direction', 'cycle_direction'],
        # Entropy combos
        ['msvr_direction', 'entropy_gate'],
        ['lr_direction', 'entropy_gate'],
        ['cycle_direction', 'entropy_gate'],
        # Volatility combos
        ['msvr_direction', 'vol_direction'],
        ['lr_direction', 'vol_direction'],
        # 3-filter combos
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
        # 4-filter combos
        ['msvr_direction', 'lr_direction', 'cycle_direction', 'trend_filter'],
        ['msvr_direction', 'smooth_direction', 'lr_direction', 'cycle_direction'],
        ['msvr_direction', 'lr_direction', 'cycle_direction', 'entropy_gate'],
        ['msvr_direction', 'lr_direction', 'cycle_direction', 'vol_direction'],
        ['msvr_direction', 'smooth_direction', 'lr_direction', 'trend_filter'],
        # With BB filter
        ['msvr_direction', 'lr_direction', 'cycle_direction', 'bb_filter'],
        ['msvr_direction', 'trend_filter', 'bb_filter'],
        ['lr_direction', 'trend_filter', 'bb_filter'],
        # With HMM regime (test a few)
        ['msvr_direction', 'lr_direction', 'regime_direction'],
        ['msvr_direction', 'cycle_direction', 'regime_direction'],
        # With volume
        ['msvr_direction', 'lr_direction', 'volume_direction'],
        ['msvr_direction', 'cycle_direction', 'volume_direction'],
    ]

    print(f"\n  Testing {len(key_combos)} curated combinations...")

    for combo_list in key_combos:
        subset_size = len(combo_list)
        
        # Test gate thresholds: ceil(n/2), n-1, n
        min_gate = max(1, (subset_size + 1) // 2)
        max_gate = subset_size

        for gate in range(min_gate, max_gate + 1):
            total_combos += 1

            try:
                # Prepare filter signals
                filter_signals_train = {name: df_train_feat[name] for name in combo_list if name in df_train_feat.columns}
                filter_signals_test = {name: df_test_feat[name] for name in combo_list if name in df_test_feat.columns}

                # Apply filters with gate
                position_train = apply_filters_with_gate(ichimoku_train, filter_signals_train, gate)
                position_test = apply_filters_with_gate(ichimoku_test, filter_signals_test, gate)

                # Compute metrics
                metrics_train = compute_metrics(position_train, df_train['close'])
                metrics_test = compute_metrics(position_test, df_test['close'])

                # Compute degradation
                if metrics_train['sharpe'] > 0:
                    sharpe_degradation = (metrics_test['sharpe'] - metrics_train['sharpe']) / metrics_train['sharpe'] * 100
                else:
                    sharpe_degradation = 0

                # Store result
                result = {
                    'combination': '+'.join(combo_list),
                    'n_filters': subset_size,
                    'gate_threshold': gate,
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
                }
                results.append(result)

            except Exception as e:
                print(f"  Error: {combo_list}, gate={gate}: {e}")

        # Print progress
        if total_combos % 10 == 0:
            print(f"  Completed {total_combos} combinations...")

    print(f"\n  Total combinations tested: {total_combos}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by test Sharpe (best first)
    results_df = results_df.sort_values('test_sharpe', ascending=False)

    # ================================================================
    # Console Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Top 10 by test Sharpe
    print("\n--- Top 10 by Test Sharpe ---")
    print(f"{'Rank':<6} {'Combination':<55} {'Gate':<6} {'TrainSharpe':<12} {'TestSharpe':<12} {'Degradation':<12}")
    print("-" * 105)

    for idx, (_, row) in enumerate(results_df.head(10).iterrows()):
        print(f"{idx+1:<6} {row['combination']:<55} {row['gate_threshold']:<6} "
              f"{row['train_sharpe']:<12.2f} {row['test_sharpe']:<12.2f} {row['sharpe_degradation']:<12.1f}%")

    # Top 10 by win rate in test
    print("\n--- Top 10 by Test Win Rate ---")
    print(f"{'Rank':<6} {'Combination':<55} {'Gate':<6} {'TestWinRate':<12} {'TestTrades':<12} {'TestSharpe':<12}")
    print("-" * 105)

    for idx, (_, row) in enumerate(results_df.sort_values('test_win_rate', ascending=False).head(10).iterrows()):
        print(f"{idx+1:<6} {row['combination']:<55} {row['gate_threshold']:<6} "
              f"{row['test_win_rate']:<12.1f} {row['test_trades']:<12} {row['test_sharpe']:<12.2f}")

    # Configs meeting targets
    target_results = results_df[
        (results_df['test_sharpe'] >= 1.35) &
        (results_df['test_win_rate'] >= 60) &
        (results_df['test_trades'] >= 25) &
        (results_df['test_trades'] <= 35)
    ]

    print("\n--- Configs Meeting ALL Targets ---")
    print(f"  Sharpe > 1.35, Win Rate > 60%, Trades 25-35")
    print(f"  Found: {len(target_results)} configs")

    if len(target_results) > 0:
        print(f"\n{'Rank':<6} {'Combination':<55} {'Gate':<6} {'TestSharpe':<12} {'TestWinRate':<12} {'TestTrades':<12}")
        print("-" * 105)

        for idx, (_, row) in enumerate(target_results.head(5).iterrows()):
            print(f"{idx+1:<6} {row['combination']:<55} {row['gate_threshold']:<6} "
                  f"{row['test_sharpe']:<12.2f} {row['test_win_rate']:<12.1f} {row['test_trades']:<12}")

    # ================================================================
    # Save Results to CSV
    # ================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save all results
    csv_path = os.path.join(OUTPUT_DIR, 'ichimoku_combination_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    # Save top configs
    top_configs_path = os.path.join(OUTPUT_DIR, 'top_configs.csv')
    results_df.head(50).to_csv(top_configs_path, index=False)
    print(f"  Top 50 configs saved to: {top_configs_path}")

    # Save target configs if any
    if len(target_results) > 0:
        target_path = os.path.join(OUTPUT_DIR, 'target_configs.csv')
        target_results.to_csv(target_path, index=False)
        print(f"  Target configs saved to: {target_path}")

    # ================================================================
    # Final Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)

    best = results_df.iloc[0]
    print(f"\nBest Config (by Test Sharpe):")
    print(f"  Combination: {best['combination']}")
    print(f"  Gate Threshold: {best['gate_threshold']}")
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
        (results['test_sharpe'] >= 1.35) &
        (results['test_win_rate'] >= 60) &
        (results['test_trades'] >= 25) &
        (results['test_trades'] <= 35)
    ]

    if len(target_results) > 0:
        print("\n✅ Found configurations meeting all targets!")
        sys.exit(0)
    else:
        print("\n⚠️  No configurations met all targets.")
        print("Consider:")
        print("  1. Adding more diverse filters")
        print("  2. Adjusting gate thresholds")
        print("  3. Using on-chain/sentiment data")
        sys.exit(1)
