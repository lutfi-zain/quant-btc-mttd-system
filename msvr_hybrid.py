#!/usr/bin/env python3
"""
MSVR Hybrid — Combines MSVR direction with Ichimoku filtering principles
==========================================================================

Goal: Beat Ichimoku's selectivity while maintaining Sharpe > 1.35.

Filters:
1. MSVR Direction (Family 1: Smoothing)
2. Cycle Phase Timing (Family 4: Spectral) - FFT lookback=40
3. SuperSmoother (Family 2: Filtering) - smooth MSVR signal
4. Shannon Entropy Gate (Family 7: Entropy) - entropy < 2.5
5. Efficiency Ratio Gate (Family 5: Fractal) - ER > 0.25

Composite signal: product of all bullish flags (all must be 1)
Position: long-only, min hold 45 days, transaction cost 0.1% round-trip.
"""

import numpy as np
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add paths
sys.path.append('/home/ubuntu/projects/quant-technical-indicator-bank')
sys.path.append('/home/ubuntu/projects/quant-btc-mttd-system')

def load_btc_data():
    """Load BTC daily data from 2018-01-01."""
    import json
    data_path = '/home/ubuntu/projects/quant-btc-mttd-system/data/btc_daily.json'
    with open(data_path) as f:
        btc_data = json.load(f)
    df = pd.DataFrame(btc_data['aligned_data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df[df.index >= '2018-01-01']
    return df

def load_msvr_signal(df):
    """Load MSVR signal from indicator bank."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('msvr', 
        '/home/ubuntu/projects/quant-technical-indicator-bank/perpetual/median_standard_deviation_viresearch.py')
    msvr_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(msvr_module)
    
    result = msvr_module.median_standard_deviation_viresearch(df)
    # result is a DataFrame with 'vii' column (positive = bullish)
    return result['vii']

def compute_cycle_phase(df, lookback=40, min_period=5, max_period=20):
    """
    FFT Cycle Phase (Family 4: Spectral).
    Detects dominant cycle period and phase.
    Returns direction column: +1 at trough (buy), -1 at peak (sell).
    """
    src = (df['high'] + df['low'] + df['close']) / 3.0
    n = len(df)
    phase = pd.Series(np.nan, index=df.index)
    period = pd.Series(np.nan, index=df.index)
    
    for i in range(lookback - 1, n):
        window = src.iloc[i - lookback + 1:i + 1].values
        if np.any(np.isnan(window)):
            continue
        
        window_detrended = window - np.mean(window)
        hann = np.hanning(lookback)
        window窗ed = window_detrended * hann
        
        fft_vals = np.fft.rfft(window窗ed)
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
            period.iloc[i] = dominant_period
    
    # Direction: +1 at cycle trough (cos = -1), -1 at peak (cos = +1)
    direction = pd.Series(np.where(np.cos(phase) < 0, 1, -1), index=df.index)
    # Where phase is NaN, direction is 0 (no signal)
    direction = direction.fillna(0).astype(int)
    return direction

def ehler_supersmoother(series: pd.Series, length: int = 7) -> pd.Series:
    """Ehler's SuperSmoother filter (Family 2: Filtering)."""
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

def shannon_entropy_gate(series: pd.Series, window: int = 15, bins: int = 6, threshold: float = 2.5):
    """
    Shannon Entropy gate (Family 7: Entropy).
    Returns binary gate: 1 if entropy < threshold (tradeable), 0 otherwise.
    """
    def calc_shannon(x):
        if len(x) < window:
            return np.nan
        counts, _ = np.histogram(x, bins=bins)
        probs = counts / len(x)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    
    returns = series.pct_change().fillna(0)
    entropy = returns.rolling(window=window).apply(calc_shannon, raw=True)
    gate = (entropy < threshold).astype(int)
    # Fill NaN with 0 (no trade)
    gate = gate.fillna(0)
    return gate

def efficiency_ratio_gate(series: pd.Series, period: int = 14, threshold: float = 0.25):
    """
    Efficiency Ratio gate (Family 5: Fractal).
    Returns binary gate: 1 if ER > threshold (trending), 0 otherwise.
    """
    change = series.diff().abs()
    volatility = change.rolling(period).sum()
    direction = series.diff(period).abs()
    er = direction / volatility
    gate = (er > threshold).astype(int)
    gate = gate.fillna(0)
    return gate

def generate_composite_signal(df, 
                               cycle_lookback=40,
                               smooth_length=7,
                               entropy_window=15,
                               entropy_threshold=2.5,
                               er_period=14,
                               er_threshold=0.25):
    """
    Generate composite signal combining all filters.
    Returns binary position column: 1 = long, 0 = flat.
    """
    df = df.copy()
    
    # 1. MSVR Direction
    msvr_series = load_msvr_signal(df)
    df['msvr_signal'] = msvr_series
    df['msvr_bullish'] = (msvr_series > 0).astype(int)
    
    # 2. Cycle Phase Direction
    df['cycle_direction'] = compute_cycle_phase(df, lookback=cycle_lookback)
    df['cycle_bullish'] = (df['cycle_direction'] > 0).astype(int)
    
    # 3. SuperSmoother Direction (applied to MSVR signal)
    msvr_smooth = ehler_supersmoother(df['msvr_signal'], length=smooth_length)
    df['msvr_smooth'] = msvr_smooth
    df['smooth_bullish'] = (msvr_smooth > 0).astype(int)
    
    # 4. Shannon Entropy Gate
    df['entropy_gate'] = shannon_entropy_gate(df['close'], window=entropy_window, threshold=entropy_threshold)
    
    # 5. Efficiency Ratio Gate
    df['er_gate'] = efficiency_ratio_gate(df['close'], period=er_period, threshold=er_threshold)
    
    # Composite signal: product of all bullish flags (all must be 1)
    df['composite_signal'] = (df['msvr_bullish'] * 
                              df['cycle_bullish'] * 
                              df['smooth_bullish'] * 
                              df['entropy_gate'] * 
                              df['er_gate'])
    
    return df

def enforce_min_hold(positions, min_hold=45):
    """
    Enforce minimum hold period of `min_hold` days.
    Once in position, must stay for at least min_hold days.
    """
    pos = positions.copy()
    in_position = False
    hold_days = 0
    
    for i in range(len(pos)):
        if pos.iloc[i] == 1 and not in_position:
            # Entry
            in_position = True
            hold_days = 0
        elif pos.iloc[i] == 1 and in_position:
            # Still in position
            hold_days += 1
            if hold_days < min_hold:
                # Force stay in position even if signal says exit
                pos.iloc[i] = 1
        elif pos.iloc[i] == 0 and in_position:
            # Exit signal
            if hold_days < min_hold:
                # Not enough hold days, force stay
                pos.iloc[i] = 1
                hold_days += 1
            else:
                # Exit allowed
                in_position = False
                hold_days = 0
    return pos

def compute_trade_list(df, prices, transaction_cost=0.001):
    """
    Compute trade list with entry/exit dates and returns.
    Transaction cost applied as 0.1% round-trip.
    """
    positions = df['position']  # after min hold enforcement
    
    trades = []
    in_position = False
    entry_date = None
    entry_price = None
    
    for i, (date, pos) in enumerate(positions.items()):
        if pos == 1 and not in_position:
            # Entry
            in_position = True
            entry_date = date
            entry_price = prices.loc[date]
        elif pos == 0 and in_position:
            # Exit
            in_position = False
            exit_price = prices.loc[date]
            # Compute return with transaction cost
            gross_return = (exit_price - entry_price) / entry_price
            net_return = gross_return - transaction_cost
            trades.append({
                'entry_date': entry_date,
                'exit_date': date,
                'return': net_return
            })
    
    # If still in position at end, close at last price
    if in_position:
        exit_price = prices.iloc[-1]
        gross_return = (exit_price - entry_price) / entry_price
        net_return = gross_return - transaction_cost
        trades.append({
            'entry_date': entry_date,
            'exit_date': prices.index[-1],
            'return': net_return
        })
    
    return pd.DataFrame(trades)

def compute_metrics(trades_df, df_prices=None):
    """Compute summary statistics from trade list."""
    if trades_df.empty:
        return {'n_trades': 0, 'win_rate': 0, 'sharpe': 0, 'total_return': 0}
    
    n_trades = len(trades_df)
    returns = trades_df['return']
    wins = (returns > 0).sum()
    win_rate = wins / n_trades * 100
    
    # Sharpe calculation using daily returns from price series
    if df_prices is not None and not trades_df.empty:
        # Build equity curve from trade list
        equity = pd.Series(1.0, index=df_prices.index)
        in_position = False
        entry_price = None
        entry_idx = None
        for i, (date, pos) in enumerate(df_prices.items()):
            # Find if date is in trades_df entry or exit
            entry_mask = trades_df['entry_date'] == date
            exit_mask = trades_df['exit_date'] == date
            if entry_mask.any():
                in_position = True
                entry_price = df_prices.loc[date]
                entry_idx = i
            elif exit_mask.any():
                in_position = False
                # compute return from entry to exit
                exit_price = df_prices.loc[date]
                # apply to equity from entry_idx to i
                if entry_idx is not None:
                    equity.iloc[entry_idx:i+1] = equity.iloc[entry_idx] * (exit_price / entry_price)
        # Compute daily returns
        daily_returns = equity.pct_change().fillna(0)
        # Annualize
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365) if daily_returns.std() > 0 else 0
    else:
        # fallback approximation
        sharpe = returns.mean() / returns.std() * np.sqrt(365 / 60) if returns.std() > 0 else 0
    
    total_return = (1 + returns).prod() - 1
    
    return {
        'n_trades': n_trades,
        'win_rate': round(win_rate, 1),
        'sharpe': round(sharpe, 2),
        'total_return': round(total_return * 100, 2)
    }

if __name__ == "__main__":
    print("=" * 70)
    print("MSVR HYBRID — Combines MSVR + Ichimoku Filtering Principles")
    print("=" * 70)
    
    # Load data
    df = load_btc_data()
    print(f"\nData: {len(df)} bars ({df.index[0]} to {df.index[-1]})")
    
    # Generate composite signal
    df = generate_composite_signal(df)
    
    # Enforce minimum hold period
    df['position'] = enforce_min_hold(df['composite_signal'], min_hold=45)
    
    # Compute trade list
    trades_df = compute_trade_list(df, df['close'], transaction_cost=0.001)
    
    # Compute metrics
    metrics = compute_metrics(trades_df, df_prices=df['close'])
    
    print(f"\nPerformance Summary:")
    print(f"  Trades:      {metrics['n_trades']}")
    print(f"  Win Rate:    {metrics['win_rate']}%")
    print(f"  Sharpe:      {metrics['sharpe']}")
    print(f"  Total Return: {metrics['total_return']}%")
    
    # Show trade list
    if not trades_df.empty:
        print(f"\nTrade List:")
        print(trades_df.to_string(index=False))
    else:
        print("\nNo trades generated.")
    
    # Verify constraints
    print(f"\nConstraint Check:")
    print(f"  Trades < 20:    {'✓' if metrics['n_trades'] < 20 else '✗'} ({metrics['n_trades']})")
    print(f"  Win Rate > 60%: {'✓' if metrics['win_rate'] > 60 else '✗'} ({metrics['win_rate']}%)")
    print(f"  Sharpe > 1.35:  {'✓' if metrics['sharpe'] > 1.35 else '✗'} ({metrics['sharpe']})")