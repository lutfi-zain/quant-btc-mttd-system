#!/usr/bin/env python3
"""
Holdout Validation for Best Configurations
============================================

Runs walk-forward holdout validation on all 5 best configurations from grid search,
computing performance metrics separately for training (2018-2024) and test (2025-2026)
periods.

Best Configs to Validate:
| System     | Filter          | MH     | Expected Sharpe |
|------------|-----------------|--------|-----------------|
| Keltner    | bull_with_filters | 15/60 | 0.96           |
| Keltner    | bull_with_filters | 25/60 | 0.95           |
| Ichimoku   | none            | 15/60  | 0.83           |
| Supertrend | none            | 15/90  | 0.72           |
| MSVR       | none            | 15/90  | 0.55           |

Output: mttd/holdout_best_results.csv
"""

import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BANK_ROOT = '/home/ubuntu/projects/quant-technical-indicator-bank'
sys.path.append(PROJECT_ROOT)
sys.path.append(BANK_ROOT)

from indicators_helper import sma, ema, atr, linreg


# ================================================================
# Helper Functions
# ================================================================

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


def efficiency_ratio(series, period=14):
    """Kaufman Efficiency Ratio (Family 5: Fractal)."""
    change = series.diff().abs()
    volatility = change.rolling(period).sum()
    direction = series.diff(period).abs()
    return direction / volatility


# ================================================================
# Signal Generators
# ================================================================

def generate_keltner_signal(df, use_filters=False, min_hold=15, max_hold=60):
    """
    Generate Keltner Channel trading signal.
    
    Base Signal: 20 EMA, 1.5x ATR breakout
    - Buy: Price closes above upper Keltner channel
    - Sell: Price closes below lower Keltner channel
    
    If use_filters=True (bull_with_filters): Apply MSVR-based filters
    """
    result = df.copy()
    
    # Keltner Channel computation
    kc_mid = ema(result['close'], 20)
    kc_atr = ema(result['high'] - result['low'], 20)
    result['kc_upper'] = kc_mid + 1.5 * kc_atr
    result['kc_lower'] = kc_mid - 1.5 * kc_atr
    
    # Base signals
    result['kc_buy'] = (result['close'] > result['kc_upper']).astype(float)
    result['kc_sell'] = (result['close'] < result['kc_lower']).astype(float)
    
    # Entry signal
    entry_signal = result['kc_buy']
    
    # Exit signal
    exit_signal = result['kc_sell']
    
    # Apply filters if requested
    if use_filters:
        # Load MSVR
        spec = importlib.util.spec_from_file_location(
            'msvr',
            os.path.join(BANK_ROOT, 'perpetual/median_standard_deviation_viresearch.py')
        )
        msvr_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(msvr_module)
        msvr_result = msvr_module.median_standard_deviation_viresearch(result)
        result['msvr_vii'] = msvr_result['vii']
        
        # SuperSmoother direction
        momentum = result['close'].pct_change(periods=10)
        smooth = ehler_supersmoother(momentum, length=5)
        result['smooth_direction'] = (smooth > 0).astype(float)
        
        # Cycle Phase
        phase = compute_cycle_phase(result, lookback=40)
        cycle_signal = -np.cos(phase)
        result['cycle_direction'] = (cycle_signal > 0).astype(float)
        
        # MSVR direction
        result['msvr_direction'] = (result['msvr_vii'] > 0).astype(float)
        
        # Require all 3 to be bullish for entry
        filter_pass = (result['msvr_direction'] * result['smooth_direction'] * result['cycle_direction']).astype(float)
        entry_signal = result['kc_buy'] * filter_pass
    
    # Apply trade constraints
    position = pd.Series(0.0, index=result.index)
    in_position = False
    hold_count = 0
    
    for i in range(len(result)):
        if entry_signal.iloc[i] == 1.0 and not in_position:
            in_position = True
            hold_count = 0
            position.iloc[i] = 1.0
        elif in_position:
            hold_count += 1
            if hold_count >= min_hold and exit_signal.iloc[i] == 1.0:
                in_position = False
                hold_count = 0
                position.iloc[i] = 0.0
            elif hold_count >= max_hold:
                in_position = False
                hold_count = 0
                position.iloc[i] = 0.0
            else:
                position.iloc[i] = 1.0
        else:
            position.iloc[i] = 0.0
    
    return position


def generate_ichimoku_signal(df, min_hold=15, max_hold=60):
    """
    Generate Ichimoku trading signal.
    
    Uses IMO (Ichimoku Momentum Oscillator) from ichimoku_quant.py
    with dynamic entry/exit thresholds.
    """
    # Load Ichimoku module
    sys.path.insert(0, PROJECT_ROOT)
    from ichimoku_quant import generate_ichimoku_features, generate_ichimoku_signals
    
    df_ich = generate_ichimoku_features(df.copy())
    df_ich = generate_ichimoku_signals(
        df_ich,
        confirm_entry=2,
        confirm_exit=1,
        min_hold_days=min_hold,
        er_entry=0.25,
        t_entry=0.40,
        chikou_thresh=-0.30,
        immunity_thresh=0.50,
        entropy_thresh=2.271,
        imo_min_limit=-0.30,
        imo_exit_bull=-0.30,
        roc_gate_limit=-0.20
    )
    
    position = df_ich['Pos'].copy()
    
    # Enforce max_hold constraint
    in_position = False
    hold_count = 0
    
    for i in range(len(position)):
        if position.iloc[i] == 1.0 and not in_position:
            in_position = True
            hold_count = 0
        elif in_position:
            hold_count += 1
            if hold_count >= max_hold:
                position.iloc[i] = 0.0
                in_position = False
                hold_count = 0
        else:
            hold_count = 0
    
    return position


def generate_supertrend_signal(df, min_hold=15, max_hold=90):
    """
    Generate Supertrend trading signal.
    
    Uses median_supertrend_viresearch indicator.
    """
    # Load Supertrend module
    spec_st = importlib.util.spec_from_file_location(
        'supertrend',
        os.path.join(BANK_ROOT, 'perpetual/median_supertrend_viresearch.py')
    )
    st_module = importlib.util.module_from_spec(spec_st)
    spec_st.loader.exec_module(st_module)
    st_result = st_module.median_supertrend_viresearch(df)
    
    st_vii = st_result['vii']
    st_buy = (st_vii > 0).astype(float)
    st_sell = (st_vii < 0).astype(float)
    
    # Apply trade constraints
    position = pd.Series(0.0, index=df.index)
    in_position = False
    hold_count = 0
    
    for i in range(len(df)):
        if st_buy.iloc[i] == 1.0 and not in_position:
            in_position = True
            hold_count = 0
            position.iloc[i] = 1.0
        elif in_position:
            hold_count += 1
            if hold_count >= min_hold and st_sell.iloc[i] == 1.0:
                in_position = False
                hold_count = 0
                position.iloc[i] = 0.0
            elif hold_count >= max_hold:
                in_position = False
                hold_count = 0
                position.iloc[i] = 0.0
            else:
                position.iloc[i] = 1.0
        else:
            position.iloc[i] = 0.0
    
    return position


def generate_msvr_signal(df, min_hold=15, max_hold=90):
    """
    Generate MSVR trading signal.
    
    Uses median_standard_deviation_viresearch indicator.
    """
    # Load MSVR module
    spec = importlib.util.spec_from_file_location(
        'msvr',
        os.path.join(BANK_ROOT, 'perpetual/median_standard_deviation_viresearch.py')
    )
    msvr_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(msvr_module)
    msvr_result = msvr_module.median_standard_deviation_viresearch(df)
    
    msvr_vii = msvr_result['vii']
    msvr_buy = (msvr_vii > 0).astype(float)
    msvr_sell = (msvr_vii < 0).astype(float)
    
    # Apply trade constraints
    position = pd.Series(0.0, index=df.index)
    in_position = False
    hold_count = 0
    
    for i in range(len(df)):
        if msvr_buy.iloc[i] == 1.0 and not in_position:
            in_position = True
            hold_count = 0
            position.iloc[i] = 1.0
        elif in_position:
            hold_count += 1
            if hold_count >= min_hold and msvr_sell.iloc[i] == 1.0:
                in_position = False
                hold_count = 0
                position.iloc[i] = 0.0
            elif hold_count >= max_hold:
                in_position = False
                hold_count = 0
                position.iloc[i] = 0.0
            else:
                position.iloc[i] = 1.0
        else:
            position.iloc[i] = 0.0
    
    return position


# ================================================================
# Metrics Computation
# ================================================================

def compute_metrics(position, prices, transaction_cost=0.001):
    """
    Compute comprehensive trading metrics for a given position series.
    
    Returns dict with:
    - sharpe: Annualized Sharpe ratio
    - win_rate: Percentage of winning trades
    - cagr: Compound Annual Growth Rate
    - n_trades: Total number of trades
    - avg_hold: Average holding period in days
    - max_dd: Maximum drawdown percentage
    """
    returns = prices.pct_change()
    strategy_returns = returns * position.shift(1)
    strategy_returns = strategy_returns.dropna()
    
    # Transaction costs
    transitions = position.diff().fillna(0)
    strategy_returns = strategy_returns - transitions.loc[strategy_returns.index] * (transaction_cost / 2)
    
    if len(strategy_returns) == 0:
        return {
            'sharpe': 0, 'win_rate': 0, 'cagr': 0,
            'n_trades': 0, 'avg_hold': 0, 'max_dd': 0
        }
    
    # Equity curve
    equity = (1 + strategy_returns).cumprod()
    
    # Sharpe ratio (annualized)
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
    
    # CAGR
    years = len(strategy_returns) / 365.25
    cagr = (equity.iloc[-1] ** (1/years) - 1) * 100 if years > 0 else 0
    
    # Max drawdown
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min() * 100
    
    # Trade analysis
    in_position = False
    hold_start = None
    trade_returns = []
    hold_periods = []
    
    for date, pos in position.items():
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
    
    # Win rate
    winning = sum(1 for r in trade_returns if r > 0)
    total = len(trade_returns)
    win_rate = winning / total * 100 if total > 0 else 0
    
    # Average hold
    avg_hold = np.mean(hold_periods) if hold_periods else 0
    
    return {
        'sharpe': round(sharpe, 2),
        'win_rate': round(win_rate, 1),
        'cagr': round(cagr, 2),
        'n_trades': total,
        'avg_hold': round(avg_hold, 0),
        'max_dd': round(max_dd, 2)
    }


# ================================================================
# Main Holdout Validation
# ================================================================

def main():
    print("=" * 70)
    print("HOLDOUT VALIDATION FOR BEST CONFIGURATIONS")
    print("=" * 70)
    print()
    print("Training Period: 2018-01-01 to 2024-12-31")
    print("Test Period:     2025-01-01 to 2026-06-24")
    print()
    
    # ================================================================
    # Step 1: Load BTC Data
    # ================================================================
    print("[1/4] Loading BTC data...")
    
    with open(os.path.join(PROJECT_ROOT, 'data', 'btc_daily.json')) as f:
        btc_data = json.load(f)
    
    df = pd.DataFrame(btc_data['aligned_data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df[df.index >= '2018-01-01']
    
    print(f"  Total data: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
    
    # Split into train/test
    train_end = '2024-12-31'
    test_start = '2025-01-01'
    
    df_train = df[df.index <= train_end].copy()
    df_test = df[df.index >= test_start].copy()
    
    print(f"  Training: {len(df_train)} bars ({df_train.index[0].date()} to {df_train.index[-1].date()})")
    print(f"  Test:     {len(df_test)} bars ({df_test.index[0].date()} to {df_test.index[-1].date()})")
    
    # ================================================================
    # Step 2: Define Best Configurations
    # ================================================================
    print("\n[2/4] Defining best configurations...")
    
    configs = [
        {
            'system': 'Keltner',
            'filter': 'bull_with_filters',
            'min_hold': 15,
            'max_hold': 60,
            'description': 'Keltner + MSVR/Smooth/Cycle filters (MH=15/60)'
        },
        {
            'system': 'Keltner',
            'filter': 'bull_with_filters',
            'min_hold': 25,
            'max_hold': 60,
            'description': 'Keltner + MSVR/Smooth/Cycle filters (MH=25/60)'
        },
        {
            'system': 'Ichimoku',
            'filter': 'none',
            'min_hold': 15,
            'max_hold': 60,
            'description': 'Ichimoku IMO (MH=15/60)'
        },
        {
            'system': 'Supertrend',
            'filter': 'none',
            'min_hold': 15,
            'max_hold': 90,
            'description': 'Supertrend (MH=15/90)'
        },
        {
            'system': 'MSVR',
            'filter': 'none',
            'min_hold': 15,
            'max_hold': 90,
            'description': 'MSVR (MH=15/90)'
        }
    ]
    
    for i, cfg in enumerate(configs):
        print(f"  {i+1}. {cfg['description']}")
    
    # ================================================================
    # Step 3: Run Holdout Validation for Each Config
    # ================================================================
    print("\n[3/4] Running holdout validation...")
    
    results = []
    
    for i, cfg in enumerate(configs):
        print(f"\n  [{i+1}/5] {cfg['description']}")
        
        # Generate signals on full data first
        if cfg['system'] == 'Keltner':
            use_filters = (cfg['filter'] == 'bull_with_filters')
            position_full = generate_keltner_signal(
                df, 
                use_filters=use_filters,
                min_hold=cfg['min_hold'],
                max_hold=cfg['max_hold']
            )
        elif cfg['system'] == 'Ichimoku':
            position_full = generate_ichimoku_signal(
                df,
                min_hold=cfg['min_hold'],
                max_hold=cfg['max_hold']
            )
        elif cfg['system'] == 'Supertrend':
            position_full = generate_supertrend_signal(
                df,
                min_hold=cfg['min_hold'],
                max_hold=cfg['max_hold']
            )
        elif cfg['system'] == 'MSVR':
            position_full = generate_msvr_signal(
                df,
                min_hold=cfg['min_hold'],
                max_hold=cfg['max_hold']
            )
        else:
            raise ValueError(f"Unknown system: {cfg['system']}")
        
        # Split position into train/test periods
        position_train = position_full[df_train.index[0]:train_end]
        position_test = position_full[test_start:]
        
        # Compute metrics for training period
        train_metrics = compute_metrics(position_train, df_train['close'])
        
        # Compute metrics for test period
        test_metrics = compute_metrics(position_test, df_test['close'])
        
        # Compute degradation
        if train_metrics['sharpe'] != 0:
            degradation = ((test_metrics['sharpe'] - train_metrics['sharpe']) / abs(train_metrics['sharpe'])) * 100
        else:
            degradation = 0
        
        print(f"    Train: Sharpe={train_metrics['sharpe']:.2f}, WinRate={train_metrics['win_rate']:.1f}%, Trades={train_metrics['n_trades']}")
        print(f"    Test:  Sharpe={test_metrics['sharpe']:.2f}, WinRate={test_metrics['win_rate']:.1f}%, Trades={test_metrics['n_trades']}")
        print(f"    Degradation: {degradation:.1f}%")
        
        results.append({
            'System': cfg['system'],
            'Filter': cfg['filter'],
            'MH': f"{cfg['min_hold']}/{cfg['max_hold']}",
            'Description': cfg['description'],
            'Train_Sharpe': train_metrics['sharpe'],
            'Test_Sharpe': test_metrics['sharpe'],
            'Train_WinRate': train_metrics['win_rate'],
            'Test_WinRate': test_metrics['win_rate'],
            'Train_CAGR': train_metrics['cagr'],
            'Test_CAGR': test_metrics['cagr'],
            'Train_Trades': train_metrics['n_trades'],
            'Test_Trades': test_metrics['n_trades'],
            'Train_MaxDD': train_metrics['max_dd'],
            'Test_MaxDD': test_metrics['max_dd'],
            'Degradation': round(degradation, 2)
        })
    
    # ================================================================
    # Step 4: Save Results and Print Summary
    # ================================================================
    print("\n[4/4] Saving results...")
    
    # Create output directory
    output_dir = os.path.join(PROJECT_ROOT, 'mttd')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'holdout_best_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    
    # Print formatted table
    print("\n" + "=" * 70)
    print("HOLDOUT VALIDATION RESULTS")
    print("=" * 70)
    
    print(f"\n{'System':<12} {'Filter':<20} {'MH':<8} {'Train':>8} {'Test':>8} {'Degrad%':>8}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['System']:<12} {r['Filter']:<20} {r['MH']:<8} "
              f"{r['Train_Sharpe']:>8.2f} {r['Test_Sharpe']:>8.2f} {r['Degradation']:>7.1f}%")
    
    print("-" * 70)
    
    # Find best config (lowest degradation)
    best_idx = results_df['Degradation'].abs().idxmin()
    best = results_df.iloc[best_idx]
    
    print(f"\nBest Config (Lowest Degradation): {best['Description']}")
    print(f"  Train Sharpe: {best['Train_Sharpe']:.2f}, Test Sharpe: {best['Test_Sharpe']:.2f}")
    print(f"  Degradation: {best['Degradation']:.1f}%")
    
    # Find highest test Sharpe
    highest_test_idx = results_df['Test_Sharpe'].idxmax()
    highest_test = results_df.iloc[highest_test_idx]
    
    print(f"\nHighest Test Sharpe: {highest_test['Description']}")
    print(f"  Test Sharpe: {highest_test['Test_Sharpe']:.2f}, Test WinRate: {highest_test['Test_WinRate']:.1f}%")
    
    print("\n" + "=" * 70)
    print("HOLDOUT VALIDATION COMPLETE")
    print("=" * 70)
    
    return results_df


if __name__ == "__main__":
    main()
