#!/usr/bin/env python3
"""
MSVR v3 — 9-Family Composite Signal Engine (v5 — Optimized)
==========================================================

Architecture: Builds on proven MSVR Hybrid approach (Sharpe 1.09)
and enhances with additional statistical families for higher Sharpe.

Key Insight from previous attempts:
- Requiring ALL 4 gates simultaneously is too restrictive (1.5% pass rate)
- Need more time in market (current 7.4% is too low for high Sharpe)
- Exit logic needs improvement (hold winning trades longer)

MSVR v3 Architecture (Optimized):
  Direction Core (2 layers — both required):
    Layer 1: MSVR Base (Family 1: Smoothing) — bullish flag
    Layer 2: SuperSmoother (Family 2: Filtering) — smooth MSVR

  Timing Core (2 layers — OR condition):
    Layer 3: Cycle Phase (Family 4: Spectral) — entry timing
    Layer 4: Efficiency Ratio (Family 5: Fractal) — trending gate

  Confirmation Layer (1 layer — voting, not required):
    Layer 5: LinearReg (Family 3: Regression) — direction confirmation

  Gate Layers (4 layers — voting system, 2 of 4 must pass):
    Layer 6: Volatility (Family 6: GARCH-like) — low vol gate
    Layer 7: Shannon Entropy (Family 7: Entropy) — predictability gate
    Layer 8: Volume Confirmation (Family 8: Volume) — volume gate
    Layer 9: HMM Regime (Family 9: Bayesian) — regime gate

Entry: direction_core AND timing AND (2_of_4_gates OR linear_reg_confirm)
Exit: Trailing stop + momentum loss

Target: Sharpe >1.35, Win Rate >65%, Trades <15.
"""

import numpy as np
import pandas as pd
import sys
import os
import json
import importlib.util
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# Paths
# ================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INDICATOR_BANK = '/home/ubuntu/projects/quant-technical-indicator-bank'
sys.path.append(PROJECT_ROOT)
sys.path.append(INDICATOR_BANK)


# ================================================================
# Data Loading
# ================================================================
def load_btc_data():
    """Load BTC daily OHLCV data from 2018-01-01."""
    data_path = os.path.join(PROJECT_ROOT, 'data', 'btc_daily.json')
    with open(data_path) as f:
        btc_data = json.load(f)
    df = pd.DataFrame(btc_data['aligned_data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df[df.index >= '2018-01-01']
    return df


# ================================================================
# Layer Implementations
# ================================================================

def layer1_msvr_base(df):
    """Layer 1: MSVR Base (Family 1: Smoothing) — direction signal."""
    spec = importlib.util.spec_from_file_location(
        'msvr',
        os.path.join(INDICATOR_BANK, 'perpetual/median_standard_deviation_viresearch.py')
    )
    msvr_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(msvr_module)
    result = msvr_module.median_standard_deviation_viresearch(df)
    return pd.Series(result['vii'].values, index=df.index)


def layer2_supersmoother(series, length=10):
    """Layer 2: SuperSmoother (Family 2: Filtering) — smooth MSVR signal."""
    from indicators.ehler_supersmoother import ehler_supersmoother
    temp_df = pd.DataFrame({'close': series})
    result = ehler_supersmoother(temp_df, source_col='close', length=length)
    return result['smooth']


def layer3_linear_reg(df, length=50):
    """Layer 3: LinearReg (Family 3: Regression) — slope confirmation."""
    from indicators.linear_reg_trend import linear_reg_trend
    result = linear_reg_trend(df, source_col='close', length=length, num_std=2.0)
    return result['direction']


def layer4_cycle_phase(df, lookback=40):
    """Layer 4: Cycle Phase (Family 4: Spectral) — entry timing."""
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

    direction = pd.Series(np.where(np.cos(phase) < 0, 1, -1), index=df.index)
    return direction.fillna(0).astype(int)


def layer5_efficiency_ratio(df, period=14, threshold=0.25):
    """Layer 5: Efficiency Ratio (Family 5: Fractal) — trending gate."""
    from indicators.efficiency_ratio import efficiency_ratio
    result = efficiency_ratio(df, source_col='close', period=period, threshold=threshold)
    return result['direction']


def layer6_volatility_gate(df, window=20, threshold=1.2):
    """Layer 6: Volatility (Family 6: GARCH-like) — low vol gate."""
    from indicators.volatility_cluster import volatility_cluster
    result = volatility_cluster(df, source_col='close', window=window,
                                median_window=100, threshold=threshold)
    return result['direction']


def layer7_entropy_gate(df, window=15, threshold=2.5):
    """Layer 7: Shannon Entropy (Family 7: Entropy) — predictability gate."""
    from indicators.shannon_entropy import shannon_entropy
    result = shannon_entropy(df, source_col='close', window=window,
                             bins=6, threshold=threshold)
    return result['direction']


def layer8_volume_confirm(df, obv_short=10, obv_long=30):
    """Layer 8: Volume Confirmation (Family 8: Volume) — volume gate."""
    from indicators.volume_confirm import volume_confirm
    result = volume_confirm(df, obv_short=obv_short, obv_long=obv_long,
                            spike_mult=1.5, spike_lookback=20, fi_smooth=13)
    return result['direction']


def layer9_hmm_regime(df, n_states=3, lookback=250):
    """Layer 9: HMM Regime (Family 9: Bayesian) — regime gate."""
    from indicators.hmm_regime import hmm_regime
    result = hmm_regime(df, source_col='close', n_states=n_states, lookback=lookback)
    return result['direction']


# ================================================================
# Composite Signal Builder (Optimized Architecture)
# ================================================================
def build_composite_signal(df, verbose=True, **params):
    """
    Build composite signal using optimized architecture.

    Architecture:
    - Direction Core: MSVR + SuperSmoother (both required)
    - Timing: Cycle Phase OR Efficiency Ratio (OR condition)
    - Gates: Voting system (2 of 4 must pass) OR LinearReg confirmation
    - Exit: Trailing stop + momentum

    Returns DataFrame with all layer values and composite_signal column.
    """
    # Extract parameters with defaults
    smooth_length = params.get('smooth_length', 7)
    lr_length = params.get('lr_length', 50)
    cycle_lookback = params.get('cycle_lookback', 40)
    er_period = params.get('er_period', 14)
    er_threshold = params.get('er_threshold', 0.25)
    vol_window = params.get('vol_window', 20)
    vol_threshold = params.get('vol_threshold', 1.5)
    entropy_window = params.get('entropy_window', 15)
    entropy_threshold = params.get('entropy_threshold', 2.0)
    vol_obv_short = params.get('vol_obv_short', 10)
    vol_obv_long = params.get('vol_obv_long', 30)
    hmm_states = params.get('hmm_states', 3)
    hmm_lookback = params.get('hmm_lookback', 250)
    gate_threshold = params.get('gate_threshold', 2)  # 2 of 4 gates must pass
    trailing_stop_pct = params.get('trailing_stop_pct', 0.15)  # 15% trailing stop
    momentum_exit_days = params.get('momentum_exit_days', 10)  # exit after 10 days below 0

    if verbose:
        print("  Computing all 9 layers...")

    # Compute all layers
    msvr_raw = layer1_msvr_base(df)
    msvr_smooth = layer2_supersmoother(msvr_raw, length=smooth_length)
    lr_dir = layer3_linear_reg(df, length=lr_length)
    cycle_dir = layer4_cycle_phase(df, lookback=cycle_lookback)
    er_dir = layer5_efficiency_ratio(df, period=er_period, threshold=er_threshold)
    vol_dir = layer6_volatility_gate(df, window=vol_window, threshold=vol_threshold)
    entropy_dir = layer7_entropy_gate(df, window=entropy_window, threshold=entropy_threshold)
    vol_confirm_dir = layer8_volume_confirm(df, obv_short=vol_obv_short, obv_long=vol_obv_long)
    hmm_dir = layer9_hmm_regime(df, n_states=hmm_states, lookback=hmm_lookback)

    # Convert MSVR to binary for core signal
    msvr_bullish = (msvr_raw > 0).astype(int)
    smooth_bullish = (msvr_smooth > 0).astype(int)

    # --- Store all layers ---
    result = pd.DataFrame(index=df.index)
    result['layer_msvr'] = msvr_raw.values
    result['layer_msvr_bullish'] = msvr_bullish.values
    result['layer_msvr_smooth'] = msvr_smooth.values
    result['layer_smooth_bullish'] = smooth_bullish.values
    result['layer_linear_reg'] = lr_dir.values
    result['layer_cycle'] = cycle_dir.values
    result['layer_er'] = er_dir.values
    result['layer_volatility'] = vol_dir.values
    result['layer_entropy'] = entropy_dir.values
    result['layer_volume'] = vol_confirm_dir.values
    result['layer_hmm'] = hmm_dir.values

    # --- Core signal: MSVR + SuperSmoother (proven from MSVR Hybrid) ---
    core_signal = msvr_bullish * smooth_bullish

    # --- Timing: Cycle Phase OR Efficiency Ratio (proven from MSVR Hybrid) ---
    timing_pass = ((cycle_dir == 1) | (er_dir == 1)).astype(int)

    # --- Gate voting: 2 of 4 gates must pass ---
    gate_votes = ((vol_dir == 1).astype(int) +
                  (entropy_dir == 1).astype(int) +
                  (vol_confirm_dir == 1).astype(int) +
                  (hmm_dir == 1).astype(int))
    gates_pass = (gate_votes >= gate_threshold).astype(int)

    # --- LinearReg confirmation (optional boost) ---
    lr_confirm = (lr_dir == 1).astype(int)

    # --- Entry signal ---
    # Entry requires: core AND timing AND (gates OR lr_confirm)
    entry_signal = core_signal * timing_pass * (gates_pass | lr_confirm)

    result['core_signal'] = core_signal.values
    result['timing_pass'] = timing_pass.values
    result['gate_votes'] = gate_votes.values
    result['gates_pass'] = gates_pass.values
    result['lr_confirm'] = lr_confirm.values
    result['entry_signal'] = entry_signal.values

    if verbose:
        print(f"  Core signal (MSVR + Smooth): {core_signal.sum()} bars")
        print(f"  Timing pass (Cycle OR ER): {timing_pass.sum()} bars")
        print(f"  Gate votes distribution:")
        for v in range(5):
            c = (gate_votes == v).sum()
            print(f"    {v} gates: {c} bars ({c/len(df)*100:.1f}%)")
        print(f"  Gates pass (≥{gate_threshold}): {gates_pass.sum()} bars ({gates_pass.mean()*100:.1f}%)")
        print(f"  LR confirm: {lr_confirm.sum()} bars ({lr_confirm.mean()*100:.1f}%)")
        print(f"  Entry signal: {entry_signal.sum()} bars ({entry_signal.mean()*100:.1f}%)")
        print(f"\n  Layer bullish %:")
        for col in ['layer_msvr_bullish', 'layer_smooth_bullish', 'layer_linear_reg',
                     'layer_cycle', 'layer_er', 'layer_volatility', 'layer_entropy',
                     'layer_volume', 'layer_hmm']:
            b = (result[col] == 1).mean() * 100
            print(f"    {col:25s}: {b:.1f}%")

    return result


# ================================================================
# Position Builder with Trailing Stop (Optimized)
# ================================================================
def build_position(df, result, min_hold=45, trailing_stop_pct=0.15,
                   momentum_exit_days=10, confirm_entry=2, verbose=True):
    """
    Build position series from entry/exit signals with optimized exit logic.

    Exit Logic (Two conditions, either triggers exit after min_hold):
    1. Trailing Stop: Price drops X% from peak since entry
    2. Momentum Loss: MSVR smooth stays below 0 for N consecutive days

    This allows winning trades to run while cutting losers.
    """
    pos = pd.Series(0, index=df.index)
    in_position = False
    hold_days = 0
    confirm_count = 0
    entry_price = 0
    peak_price = 0
    momentum_below_zero_days = 0

    entry_signal = result['entry_signal']
    msvr_smooth = result['layer_msvr_smooth']
    prices = df['close']

    for i in range(len(df)):
        entry = entry_signal.iloc[i]
        price = prices.iloc[i]
        smooth_val = msvr_smooth.iloc[i]

        if not in_position:
            # Looking for entry
            if entry == 1:
                confirm_count += 1
                if confirm_count >= confirm_entry:
                    in_position = True
                    hold_days = 0
                    entry_price = price
                    peak_price = price
                    momentum_below_zero_days = 0
                    pos.iloc[i] = 1
            else:
                confirm_count = 0
        else:
            # In position
            hold_days += 1
            pos.iloc[i] = 1

            # Update peak price
            if price > peak_price:
                peak_price = price

            # Track momentum
            if smooth_val < 0:
                momentum_below_zero_days += 1
            else:
                momentum_below_zero_days = 0

            # Check exit conditions (only after min_hold)
            if hold_days >= min_hold:
                # Condition 1: Trailing stop
                price_drop = (peak_price - price) / peak_price
                if price_drop >= trailing_stop_pct:
                    in_position = False
                    hold_days = 0
                    pos.iloc[i] = 0
                    continue

                # Condition 2: Momentum loss
                if momentum_below_zero_days >= momentum_exit_days:
                    in_position = False
                    hold_days = 0
                    pos.iloc[i] = 0
                    continue

    if verbose:
        in_market = (pos == 1).sum()
        print(f"  In-market: {in_market} / {len(pos)} days ({in_market/len(pos)*100:.1f}%)")

    return pos


# ================================================================
# Backtest Engine
# ================================================================
def backtest(df, position, transaction_cost=0.001):
    """Compute backtest metrics from position series."""
    prices = df['close']
    daily_returns = prices.pct_change().fillna(0)

    strategy_returns = daily_returns * position.shift(1).fillna(0)
    transitions = position.diff().fillna(0)
    strategy_returns = strategy_returns - transitions.abs() * (transaction_cost / 2)

    equity = (1 + strategy_returns).cumprod()

    if strategy_returns.std() > 0:
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365)
    else:
        sharpe = 0.0

    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()

    years = len(strategy_returns) / 365.25
    cagr = (equity.iloc[-1] ** (1 / years) - 1) if years > 0 else 0

    trades = []
    in_pos = False
    entry_date = None
    entry_price = None

    for i, (date, p) in enumerate(position.items()):
        if p == 1 and not in_pos:
            in_pos = True
            entry_date = date
            entry_price = prices.loc[date]
        elif p == 0 and in_pos:
            in_pos = False
            exit_price = prices.loc[date]
            ret = (exit_price - entry_price) / entry_price - transaction_cost
            trades.append({
                'entry': entry_date,
                'exit': date,
                'days': (date - entry_date).days,
                'return': ret
            })

    if in_pos:
        exit_price = prices.iloc[-1]
        ret = (exit_price - entry_price) / entry_price - transaction_cost
        trades.append({
            'entry': entry_date,
            'exit': prices.index[-1],
            'days': (prices.index[-1] - entry_date).days,
            'return': ret
        })

    trades_df = pd.DataFrame(trades)

    if len(trades_df) > 0:
        wins = (trades_df['return'] > 0).sum()
        win_rate = wins / len(trades_df) * 100
        avg_hold = trades_df['days'].mean()
    else:
        win_rate = 0
        avg_hold = 0

    time_in_market = position.mean() * 100

    return {
        'n_trades': len(trades_df),
        'win_rate': round(win_rate, 1),
        'sharpe': round(sharpe, 2),
        'cagr': round(cagr * 100, 2),
        'max_dd': round(max_dd * 100, 2),
        'avg_hold': round(avg_hold, 0),
        'time_in_market': round(time_in_market, 1),
        'total_return': round((equity.iloc[-1] - 1) * 100, 2),
        'trades_df': trades_df,
        'equity': equity,
        'position': position
    }


# ================================================================
# Parameter Sweep
# ================================================================
def sweep_parameters(df):
    """Sweep key parameters to find optimal configuration."""
    best_score = 0  # combined score
    best_config = {}
    best_metrics = {}

    configs = [
        # (smooth_length, cycle_lookback, er_threshold, entropy_threshold,
        #  vol_threshold, gate_threshold, trailing_stop_pct, momentum_exit_days, min_hold)
        # Base configs with different gate thresholds
        (7, 40, 0.25, 2.0, 1.5, 2, 0.15, 10, 45),
        (7, 40, 0.25, 2.0, 1.5, 1, 0.15, 10, 45),  # 1 gate only
        (7, 40, 0.25, 2.0, 1.5, 3, 0.15, 10, 45),  # 3 gates
        # Adjusted entropy thresholds
        (7, 40, 0.25, 1.8, 1.5, 2, 0.15, 10, 45),
        (7, 40, 0.25, 2.2, 1.5, 2, 0.15, 10, 45),
        (7, 40, 0.25, 2.5, 1.5, 2, 0.15, 10, 45),
        # Adjusted volatility thresholds
        (7, 40, 0.25, 2.0, 1.2, 2, 0.15, 10, 45),
        (7, 40, 0.25, 2.0, 1.8, 2, 0.15, 10, 45),
        # Adjusted ER thresholds
        (7, 40, 0.20, 2.0, 1.5, 2, 0.15, 10, 45),
        (7, 40, 0.30, 2.0, 1.5, 2, 0.15, 10, 45),
        # Adjusted trailing stop
        (7, 40, 0.25, 2.0, 1.5, 2, 0.10, 10, 45),
        (7, 40, 0.25, 2.0, 1.5, 2, 0.20, 10, 45),
        (7, 40, 0.25, 2.0, 1.5, 2, 0.25, 10, 45),
        # Adjusted momentum exit
        (7, 40, 0.25, 2.0, 1.5, 2, 0.15, 5, 45),
        (7, 40, 0.25, 2.0, 1.5, 2, 0.15, 15, 45),
        (7, 40, 0.25, 2.0, 1.5, 2, 0.15, 20, 45),
        # Combined aggressive (1 gate, tight stop)
        (7, 40, 0.25, 2.0, 1.5, 1, 0.10, 5, 45),
        (7, 40, 0.25, 2.5, 1.5, 1, 0.10, 5, 45),
        # Combined conservative (2 gates, wide stop)
        (7, 40, 0.25, 2.0, 1.5, 2, 0.25, 15, 45),
        (7, 40, 0.25, 2.5, 1.5, 2, 0.25, 15, 45),
        # More smooth length options
        (10, 40, 0.25, 2.0, 1.5, 2, 0.15, 10, 45),
        (14, 40, 0.25, 2.0, 1.5, 2, 0.15, 10, 45),
        # Cycle lookback variations
        (7, 30, 0.25, 2.0, 1.5, 2, 0.15, 10, 45),
        (7, 50, 0.25, 2.0, 1.5, 2, 0.15, 10, 45),
        # Very aggressive (1 gate, no LR requirement)
        (7, 40, 0.20, 1.8, 1.5, 1, 0.12, 7, 45),
        (7, 40, 0.20, 2.0, 1.2, 1, 0.12, 7, 45),
        # Very conservative (2 gates, wide stop)
        (7, 40, 0.30, 2.2, 1.8, 2, 0.20, 20, 45),
        (10, 40, 0.25, 2.0, 1.5, 2, 0.20, 15, 45),
    ]

    print(f"\n  Sweeping {len(configs)} configurations...")

    for idx, (sl, cl, et, entt, vt, gt, ts, me, mh) in enumerate(configs):
        config = {
            'smooth_length': sl,
            'cycle_lookback': cl,
            'er_threshold': et,
            'entropy_threshold': entt,
            'vol_threshold': vt,
            'gate_threshold': gt,
            'trailing_stop_pct': ts,
            'momentum_exit_days': me,
        }

        result = build_composite_signal(df, verbose=False, **config)
        pos = build_position(df, result, min_hold=mh, verbose=False,
                            trailing_stop_pct=ts, momentum_exit_days=me)
        metrics = backtest(df, pos)

        # Score: prioritize meeting constraints
        score = 0
        if metrics['n_trades'] < 15:
            score += 30
            score += (15 - metrics['n_trades']) * 2  # bonus for fewer trades
        if metrics['win_rate'] > 60:
            score += 30
            score += (metrics['win_rate'] - 60) * 1  # bonus for higher win rate
        if metrics['sharpe'] > 1.0:
            score += 40
            score += min(metrics['sharpe'], 2.0) * 10  # bonus for higher Sharpe
        # Penalty for too little time in market
        if metrics['time_in_market'] < 10:
            score -= 10

        if score > best_score:
            best_score = score
            best_config = config.copy()
            best_config['min_hold'] = mh
            best_metrics = metrics.copy()
            print(f"    [{idx+1}/{len(configs)}] NEW BEST (score={score:.0f}): "
                  f"Sharpe={metrics['sharpe']:.2f}, Trades={metrics['n_trades']}, "
                  f"WinRate={metrics['win_rate']}%, TimeInMkt={metrics['time_in_market']}%")
        elif idx % 5 == 0:
            print(f"    [{idx+1}/{len(configs)}] Current: Sharpe={metrics['sharpe']:.2f}, "
                  f"Trades={metrics['n_trades']}, WinRate={metrics['win_rate']}%")

    return best_config, best_metrics


# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("MSVR v3 — 9-Family Composite Signal Engine (Optimized)")
    print("=" * 70)

    print("\n[1] Loading BTC daily data...")
    df = load_btc_data()
    print(f"  Data: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")

    import sys
    run_sweep = '--sweep' in sys.argv

    if run_sweep:
        print("\n[2] Running parameter sweep...")
        best_config, best_metrics = sweep_parameters(df)

        if best_config:
            print(f"\n  Best config: {best_config}")
            print(f"  Best metrics:")
            for k, v in best_metrics.items():
                if k not in ['trades_df', 'equity', 'position']:
                    print(f"    {k}: {v}")

            result = build_composite_signal(df, verbose=True, **best_config)
            position = build_position(df, result,
                                      min_hold=best_config.get('min_hold', 45),
                                      trailing_stop_pct=best_config.get('trailing_stop_pct', 0.15),
                                      momentum_exit_days=best_config.get('momentum_exit_days', 10),
                                      verbose=True)
            metrics = backtest(df, position, transaction_cost=0.001)
        else:
            print("\n  No valid configuration found!")
            sys.exit(1)
    else:
        # Run with optimized config
        config = {
            'smooth_length': 7,
            'lr_length': 50,
            'cycle_lookback': 40,
            'er_period': 14,
            'er_threshold': 0.25,
            'vol_window': 20,
            'vol_threshold': 1.5,
            'entropy_window': 15,
            'entropy_threshold': 2.0,
            'vol_obv_short': 10,
            'vol_obv_long': 30,
            'hmm_states': 3,
            'hmm_lookback': 250,
            'gate_threshold': 2,
            'trailing_stop_pct': 0.15,
            'momentum_exit_days': 10,
        }

        print(f"\n[2] Building composite signal...")
        result = build_composite_signal(df, verbose=True, **config)

        print(f"\n[3] Building position (min_hold=45, trailing_stop=15%, momentum_exit=10d)...")
        position = build_position(df, result, min_hold=45,
                                  trailing_stop_pct=config['trailing_stop_pct'],
                                  momentum_exit_days=config['momentum_exit_days'],
                                  verbose=True)

        print(f"\n[4] Running backtest (transaction_cost=0.1%)...")
        metrics = backtest(df, position, transaction_cost=0.001)

    # Print results
    print(f"\n{'=' * 70}")
    print(f"RESULTS")
    print(f"{'=' * 70}")
    print(f"  Trades:           {metrics['n_trades']}")
    print(f"  Win Rate:         {metrics['win_rate']}%")
    print(f"  Sharpe Ratio:     {metrics['sharpe']}")
    print(f"  CAGR:             {metrics['cagr']}%")
    print(f"  Max Drawdown:     {metrics['max_dd']}%")
    print(f"  Avg Hold:         {metrics['avg_hold']:.0f} days")
    print(f"  Time in Market:   {metrics['time_in_market']}%")
    print(f"  Total Return:     {metrics['total_return']}%")

    if not metrics['trades_df'].empty:
        print(f"\n  Trade List:")
        print(metrics['trades_df'][['entry', 'exit', 'days', 'return']].to_string(index=False))

    print(f"\n{'=' * 70}")
    print(f"CONSTRAINT CHECK")
    print(f"{'=' * 70}")
    print(f"  Trades < 15:     {'✓' if metrics['n_trades'] < 15 else '✗'} ({metrics['n_trades']})")
    print(f"  Win Rate > 60%:  {'✓' if metrics['win_rate'] > 60 else '✗'} ({metrics['win_rate']}%)")
    print(f"  Sharpe > 1.35:   {'✓' if metrics['sharpe'] > 1.35 else '✗'} ({metrics['sharpe']})")
    print(f"  {'=' * 50}")

    # Save results
    output_dir = os.path.join(PROJECT_ROOT, 'mttd')
    os.makedirs(output_dir, exist_ok=True)

    signals_path = os.path.join(output_dir, 'msvr_v3_signals.csv')
    save_df = result.copy()
    save_df['position'] = position.values
    save_df['btc_price'] = df['close'].values
    save_df.to_csv(signals_path)
    print(f"\n  Saved signals: {signals_path}")

    equity_df = pd.DataFrame({
        'date': metrics['equity'].index,
        'equity': metrics['equity'].values
    })
    equity_path = os.path.join(output_dir, 'msvr_v3_equity.csv')
    equity_df.to_csv(equity_path, index=False)
    print(f"  Saved equity: {equity_path}")

    print(f"\n{'=' * 70}")
    print(f"MSVR v3 COMPLETE")
    print(f"{'=' * 70}")
