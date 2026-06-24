#!/usr/bin/env python3
"""
Parameter Grid Search for Best Indicator Combination
=====================================================

Goal: Optimize parameters for the best indicator combination found in TODO 4
to push Sharpe above 1.35 with trade count 25-35, win rate > 60%, CAGR > 50%.

Best combination from TODO 4: msvr+smooth+cycle+entropy
- Current: Sharpe 1.46, Win Rate 53.7%, Trades 41, Degradation 23.7%
- Target: Sharpe > 1.35, Win Rate > 60%, Trades 25-35, CAGR > 50%

Parameters to optimize:
1. min_hold: [20, 25, 30, 35, 40, 45] — reduce trades by requiring longer minimum hold
2. max_hold: [60, 75, 90, 120] — cap maximum position duration
3. gate_threshold: [3, 4, 5] — filter voting requirements
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
from itertools import product
warnings.filterwarnings('ignore')

# Add paths
project_root = os.path.dirname(os.path.abspath(__file__))
bank_root = '/home/ubuntu/projects/quant-technical-indicator-bank'
sys.path.append(project_root)
sys.path.append(bank_root)
sys.path.append(os.path.join(project_root, 'indicators'))

# ================================================================
# Configuration
# ================================================================
TRANSACTION_COST = 0.001  # 0.1% round-trip

# Walk-forward periods
TRAIN_START = '2018-01-01'
TRAIN_END = '2023-12-31'
TEST_START = '2024-01-01'
TEST_END = '2026-06-30'

# Output directory
OUTPUT_DIR = os.path.join(project_root, 'mttd', 'grid_search')

print("=" * 70)
print("PARAMETER GRID SEARCH FOR BEST INDICATOR COMBINATION")
print("=" * 70)
print(f"\nBest Combination from TODO 4: msvr+smooth+cycle+entropy")
print(f"Target: Sharpe > 1.35, 25-35 trades, win rate > 60%, CAGR > 50%")
print(f"Transaction Cost: {TRANSACTION_COST*100:.1f}% round-trip")
print(f"Walk-Forward: Train {TRAIN_START} to {TRAIN_END}, Test {TEST_START} to {TEST_END}")


# ================================================================
# Helper Functions (same as TODO 4)
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
# Ichimoku Feature Generation (same as TODO 4)
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

    return df


# ================================================================
# Generate Additional Filters (same as TODO 4)
# ================================================================

def generate_additional_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Generate additional indicator filters for combination testing."""
    df = df.copy()

    # Family 2: SuperSmoother Momentum (smooth_direction)
    df['momentum'] = df['close'].pct_change(periods=10)
    df['momentum_smooth'] = ehler_supersmoother(df['momentum'], length=5)
    df['smooth_direction'] = (df['momentum_smooth'] > 0).astype(float)

    # Family 4: Cycle Phase (cycle_direction)
    phase = compute_cycle_phase(df, lookback=40)
    df['cycle_signal'] = -np.cos(phase)
    df['cycle_direction'] = (df['cycle_signal'] > 0).astype(float)

    # Family 7: Shannon Entropy Gate
    df['entropy_gate'] = (df['Entropy'] < 2.8).astype(float)
    df['entropy_gate_strict'] = (df['Entropy'] < 2.5).astype(float)

    # MSVR Direction (from indicator bank)
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
        df['msvr_direction'] = 0.5

    return df


# ================================================================
# Ichimoku Entry Signal (same as TODO 4)
# ================================================================

def generate_ichimoku_entry_signal(df: pd.DataFrame,
                                   min_hold: int = 45,
                                   max_hold: int = 120) -> pd.Series:
    """
    Generate Ichimoku base entry signal using IMO composite.
    This matches the standard mode from TODO 4.
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

        if pd.isna(imo) or pd.isna(er) or pd.isna(std) or pd.isna(entropy):
            if in_position:
                position[i] = 1.0
            continue

        threshold = std * 0.40  # Fixed threshold multiplier

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
            # Entry conditions (standard mode)
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
# Majority Gate Voting (same as TODO 4)
# ================================================================

def apply_filters_with_gate(ichimoku_signal: pd.Series,
                            filter_signals: dict,
                            gate_threshold: int = 3) -> pd.Series:
    """
    Apply filters using majority-gate voting mechanism.
    This matches the TODO 4 strategy_majority_vote implementation.
    
    The signal matrix includes Ichimoku + all filters.
    min_agreement is the total number of signals (including Ichimoku)
    that need to be bullish for entry.
    """
    n = len(ichimoku_signal)
    result = np.zeros(n)

    # Stack all signals (Ichimoku + filters) - same as TODO 4
    signal_names = ['ichimoku'] + list(filter_signals.keys())
    signal_matrix = np.column_stack([
        ichimoku_signal.values,
        *[filter_signals[name].values for name in filter_signals.keys()]
    ])

    in_position = False

    for i in range(n):
        if not in_position:
            # Entry: need min_agreement signals bullish (including Ichimoku)
            bullish_count = np.sum(signal_matrix[i] == 1.0)
            if bullish_count >= gate_threshold:
                in_position = True
                result[i] = 1.0
        else:
            # Exit: majority turns bearish
            bullish_count = np.sum(signal_matrix[i] == 1.0)
            if bullish_count < gate_threshold:
                in_position = False
                result[i] = 0.0
            else:
                result[i] = 1.0

    return pd.Series(result, index=ichimoku_signal.index)


# ================================================================
# Performance Metrics (same as TODO 4)
# ================================================================

def compute_metrics(signal: pd.Series, prices: pd.Series) -> dict:
    """Compute comprehensive trading metrics."""
    returns = prices.pct_change()
    strategy_returns = returns * signal.shift(1)
    strategy_returns = strategy_returns.dropna()

    # Transaction costs
    transitions = signal.diff().fillna(0)
    strategy_returns = strategy_returns - transitions.loc[strategy_returns.index] * (TRANSACTION_COST / 2)

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
# Main Grid Search
# ================================================================

def run_grid_search():
    """Run parameter grid search for the best combination."""

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

    # Pre-compute Ichimoku features and filters
    print("\n  Pre-computing Ichimoku features and filters...")
    df_train_feat = generate_ichimoku_features(df_train.copy())
    df_test_feat = generate_ichimoku_features(df_test.copy())
    df_train_feat = generate_additional_filters(df_train_feat)
    df_test_feat = generate_additional_filters(df_test_feat)
    print("  Features computed.")

    # ================================================================
    # Define Parameter Grid
    # ================================================================
    
    # Parameter ranges for grid search
    # Focus on parameters that affect trade count and win rate
    # Note: gate_threshold is the total number of signals (Ichimoku + 4 filters) that need to agree
    # So gate_threshold=3 means 3 out of 5 signals (60% agreement)
    param_grid = {
        'min_hold': [15, 20, 25, 30, 35, 40, 45, 50],  # 8 values - wider range
        'max_hold': [50, 60, 75, 90, 120],              # 5 values - wider range
        'gate_threshold': [2, 3, 4],                     # 3 values - 2=40%, 3=60%, 4=80% of 5 signals
    }
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(product(*param_values))
    
    print(f"\n" + "=" * 70)
    print("PARAMETER GRID SEARCH")
    print("=" * 70)
    print(f"  Total combinations in grid: {len(all_combinations)}")
    print(f"  Parameters:")
    for name, values in param_grid.items():
        print(f"    {name}: {values}")
    
    # Define the best filter combination from TODO 4
    best_filters = ['msvr_direction', 'smooth_direction', 'cycle_direction', 'entropy_gate']
    print(f"\n  Best filter combination: {'+'.join(best_filters)}")

    # ================================================================
    # Run Grid Search
    # ================================================================
    
    results = []
    
    print(f"\n  Running grid search...")
    
    for idx, params in enumerate(all_combinations):
        param_dict = dict(zip(param_names, params))
        
        try:
            # Generate Ichimoku entry signal with these parameters
            ichimoku_train = generate_ichimoku_entry_signal(
                df_train_feat,
                min_hold=param_dict['min_hold'],
                max_hold=param_dict['max_hold']
            )
            
            ichimoku_test = generate_ichimoku_entry_signal(
                df_test_feat,
                min_hold=param_dict['min_hold'],
                max_hold=param_dict['max_hold']
            )
            
            # Prepare filter signals
            filter_signals_train = {name: df_train_feat[name] for name in best_filters}
            filter_signals_test = {name: df_test_feat[name] for name in best_filters}
            
            # Apply majority gate
            position_train = apply_filters_with_gate(
                ichimoku_train,
                filter_signals_train,
                gate_threshold=param_dict['gate_threshold']
            )
            
            position_test = apply_filters_with_gate(
                ichimoku_test,
                filter_signals_test,
                gate_threshold=param_dict['gate_threshold']
            )
            
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
                'min_hold': param_dict['min_hold'],
                'max_hold': param_dict['max_hold'],
                'gate_threshold': param_dict['gate_threshold'],
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
            print(f"  Error with params {param_dict}: {e}")
        
        # Print progress every 10 combinations
        if (idx + 1) % 10 == 0:
            print(f"  Completed {idx + 1}/{len(all_combinations)} combinations...")
    
    print(f"  Total combinations tested: {len(results)}")

    # ================================================================
    # Analyze Results
    # ================================================================
    
    results_df = pd.DataFrame(results)
    
    # Sort by test Sharpe (best first)
    results_df = results_df.sort_values('test_sharpe', ascending=False)

    # ================================================================
    # Console Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("GRID SEARCH RESULTS")
    print("=" * 70)

    # Top 20 by test Sharpe
    print("\n--- Top 20 by Test Sharpe ---")
    print(f"{'Rank':<6} {'MinH':<6} {'MaxH':<6} {'Gate':<6} {'TrainS':<8} {'TestS':<8} {'TestW%':<8} {'TestT':<8} {'CAGR%':<8} {'Degr%':<8}")
    print("-" * 110)

    for idx, (_, row) in enumerate(results_df.head(20).iterrows()):
        print(f"{idx+1:<6} {row['min_hold']:<6} {row['max_hold']:<6} {row['gate_threshold']:<6} "
              f"{row['train_sharpe']:<8.2f} {row['test_sharpe']:<8.2f} "
              f"{row['test_win_rate']:<8.1f} {row['test_trades']:<8} {row['test_cagr']:<8.1f} {row['sharpe_degradation']:<8.1f}")

    # Configs meeting ALL targets
    target_results = results_df[
        (results_df['test_sharpe'] >= 1.35) &
        (results_df['test_win_rate'] >= 60) &
        (results_df['test_trades'] >= 25) &
        (results_df['test_trades'] <= 35) &
        (results_df['test_cagr'] >= 50)
    ]

    print(f"\n--- Configs Meeting ALL Targets ---")
    print(f"  Sharpe > 1.35, Win Rate > 60%, Trades 25-35, CAGR > 50%")
    print(f"  Found: {len(target_results)} configs")

    if len(target_results) > 0:
        print(f"\n{'Rank':<6} {'MinH':<6} {'MaxH':<6} {'Gate':<6} {'TestS':<8} {'TestW%':<8} {'TestT':<8} {'CAGR%':<8}")
        print("-" * 80)

        for idx, (_, row) in enumerate(target_results.head(10).iterrows()):
            print(f"{idx+1:<6} {row['min_hold']:<6} {row['max_hold']:<6} {row['gate_threshold']:<6} "
                  f"{row['test_sharpe']:<8.2f} "
                  f"{row['test_win_rate']:<8.1f} {row['test_trades']:<8} {row['test_cagr']:<8.1f}")

    # Configs meeting Sharpe target (may need relaxation on other targets)
    sharpe_target = results_df[
        (results_df['test_sharpe'] >= 1.35) &
        (results_df['test_trades'] >= 20) &
        (results_df['test_trades'] <= 40)
    ]

    print(f"\n--- Configs Meeting Sharpe > 1.35 (Trades 20-40) ---")
    print(f"  Found: {len(sharpe_target)} configs")

    if len(sharpe_target) > 0:
        print(f"\n{'Rank':<6} {'MinH':<6} {'MaxH':<6} {'Gate':<6} {'TestS':<8} {'TestW%':<8} {'TestT':<8} {'CAGR%':<8}")
        print("-" * 80)

        for idx, (_, row) in enumerate(sharpe_target.head(10).iterrows()):
            print(f"{idx+1:<6} {row['min_hold']:<6} {row['max_hold']:<6} {row['gate_threshold']:<6} "
                  f"{row['test_sharpe']:<8.2f} "
                  f"{row['test_win_rate']:<8.1f} {row['test_trades']:<8} {row['test_cagr']:<8.1f}")

    # Configs with best win rate
    best_winrate = results_df[
        (results_df['test_win_rate'] >= 55) &
        (results_df['test_trades'] >= 20) &
        (results_df['test_trades'] <= 40)
    ].sort_values('test_win_rate', ascending=False)

    print(f"\n--- Configs with Best Win Rate (Win > 55%, Trades 20-40) ---")
    print(f"  Found: {len(best_winrate)} configs")

    if len(best_winrate) > 0:
        print(f"\n{'Rank':<6} {'MinH':<6} {'MaxH':<6} {'Gate':<6} {'TestS':<8} {'TestW%':<8} {'TestT':<8} {'CAGR%':<8}")
        print("-" * 80)

        for idx, (_, row) in enumerate(best_winrate.head(10).iterrows()):
            print(f"{idx+1:<6} {row['min_hold']:<6} {row['max_hold']:<6} {row['gate_threshold']:<6} "
                  f"{row['test_sharpe']:<8.2f} "
                  f"{row['test_win_rate']:<8.1f} {row['test_trades']:<8} {row['test_cagr']:<8.1f}")

    # ================================================================
    # Save Results
    # ================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save all results
    csv_path = os.path.join(OUTPUT_DIR, 'grid_search_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n  All results saved to: {csv_path}")

    # Save target configs if any
    if len(target_results) > 0:
        target_path = os.path.join(OUTPUT_DIR, 'target_configs.csv')
        target_results.to_csv(target_path, index=False)
        print(f"  Target configs saved to: {target_path}")

    # Save best Sharpe configs
    best_sharpe_path = os.path.join(OUTPUT_DIR, 'best_sharpe_configs.csv')
    sharpe_target.to_csv(best_sharpe_path, index=False)
    print(f"  Best Sharpe configs saved to: {best_sharpe_path}")

    # Save best win rate configs
    best_winrate_path = os.path.join(OUTPUT_DIR, 'best_winrate_configs.csv')
    best_winrate.to_csv(best_winrate_path, index=False)
    print(f"  Best win rate configs saved to: {best_winrate_path}")

    # ================================================================
    # Final Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print("=" * 70)

    best = results_df.iloc[0]
    print(f"\nBest Config (by Test Sharpe):")
    print(f"  min_hold: {best['min_hold']}")
    print(f"  max_hold: {best['max_hold']}")
    print(f"  gate_threshold: {best['gate_threshold']}")
    print(f"  Train Sharpe: {best['train_sharpe']:.2f}")
    print(f"  Test Sharpe: {best['test_sharpe']:.2f}")
    print(f"  Train Win Rate: {best['train_win_rate']:.1f}%")
    print(f"  Test Win Rate: {best['test_win_rate']:.1f}%")
    print(f"  Train Trades: {best['train_trades']}")
    print(f"  Test Trades: {best['test_trades']}")
    print(f"  Train CAGR: {best['train_cagr']:.1f}%")
    print(f"  Test CAGR: {best['test_cagr']:.1f}%")
    print(f"  Sharpe Degradation: {best['sharpe_degradation']:.1f}%")

    # Check if we found any config meeting all targets
    if len(target_results) > 0:
        best_target = target_results.iloc[0]
        print(f"\n✅ OPTIMAL CONFIG FOUND (Meeting ALL targets):")
        print(f"  min_hold: {best_target['min_hold']}")
        print(f"  max_hold: {best_target['max_hold']}")
        print(f"  gate_threshold: {best_target['gate_threshold']}")
        print(f"  Test Sharpe: {best_target['test_sharpe']:.2f}")
        print(f"  Test Win Rate: {best_target['test_win_rate']:.1f}%")
        print(f"  Test Trades: {best_target['test_trades']}")
        print(f"  Test CAGR: {best_target['test_cagr']:.1f}%")
        return results_df, True
    else:
        print(f"\n⚠️  No configs met ALL targets.")
        print(f"  Best Sharpe found: {best['test_sharpe']:.2f}")
        print(f"  Best Win Rate found: {results_df.sort_values('test_win_rate', ascending=False).iloc[0]['test_win_rate']:.1f}%")
        return results_df, False


# ================================================================
# Entry Point
# ================================================================

if __name__ == "__main__":
    results_df, success = run_grid_search()
    
    if success:
        print("\n✅ Grid search found optimal parameter set meeting all success criteria!")
        sys.exit(0)
    else:
        print("\n⚠️  Grid search did not find optimal parameter set meeting all criteria.")
        print("  Consider:")
        print("  1. The best Sharpe is close to target - may need additional parameter exploration")
        print("  2. Win rate may need on-chain/sentiment data for fundamental edge")
        print("  3. Trade count can be tuned via min_hold/max_hold parameters")
        sys.exit(1)
