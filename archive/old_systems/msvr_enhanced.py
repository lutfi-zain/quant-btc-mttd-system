#!/usr/bin/env python3
"""
Enhanced MSVR — Multi-Principle Indicator
==========================================

Like Ichimoku uses 4 families, we enhance MSVR with:
- Family 1 (Smoothing): MSVR base (DEMA + Median + ATR)
- Family 2 (Filtering): Ehler SuperSmoother
- Family 4 (Spectral): FFT Cycle Phase
- Family 5 (Fractal): Efficiency Ratio
- Family 7 (Entropy): Shannon Entropy

Goal: Beat Ichimoku's Sharpe 1.31, CAGR 55.6%, Win Rate 63.6%
"""

import numpy as np
import pandas as pd
import sys
import os

# Add paths
sys.path.append('/home/ubuntu/projects/quant-technical-indicator-bank')
from indicators_helper import *

def ehler_supersmoother(series: pd.Series, length: int = 7) -> pd.Series:
    """
    Ehler's SuperSmoother Filter (Family 2: Filtering).
    Removes high-frequency noise without lag penalty.
    """
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
    """
    Shannon Entropy (Family 7: Entropy).
    Measures randomness/complexity of returns.
    """
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
    """
    Efficiency Ratio (Family 5: Fractal).
    Measures trend strength vs noise.
    ER = |direction| / |volatility|
    """
    change = series.diff().abs()
    volatility = change.rolling(period).sum()
    direction = series.diff(period).abs()
    return direction / volatility

def compute_cycle_phase(df, lookback=40):
    """
    FFT Cycle Phase (Family 4: Spectral).
    Detects dominant cycle period and phase.
    """
    src = (df['high'] + df['low'] + df['close']) / 3.0
    n = len(df)
    phase = pd.Series(np.nan, index=df.index)
    period = pd.Series(np.nan, index=df.index)
    
    min_period = 5
    max_period = lookback // 2
    
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
    
    return phase, period

def msvr_enhanced(df: pd.DataFrame, 
                  cycle_lookback: int = 40,
                  smooth_length: int = 7,
                  entropy_window: int = 15,
                  er_period: int = 14,
                  er_threshold: float = 0.25,
                  entropy_threshold: float = 2.5) -> pd.DataFrame:
    """
    Enhanced MSVR with multi-principle filtering.
    
    Layers:
    1. MSVR Base (Family 1: Smoothing) - Direction
    2. Cycle Phase (Family 4: Spectral) - Timing
    3. SuperSmoother (Family 2: Filtering) - Noise reduction
    4. Efficiency Ratio (Family 5: Fractal) - Trend strength gate
    5. Shannon Entropy (Family 7: Entropy) - Randomness filter
    """
    df = df.copy()
    
    # ================================================================
    # Layer 1: MSVR Base (Family 1: Smoothing)
    # ================================================================
    # Load MSVR indicator
    import importlib.util
    spec = importlib.util.spec_from_file_location('msvr', 
        '/home/ubuntu/projects/quant-technical-indicator-bank/perpetual/median_standard_deviation_viresearch.py')
    msvr_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(msvr_module)
    
    msvr_result = msvr_module.median_standard_deviation_viresearch(df)
    df['msvr_signal'] = msvr_result['vii']
    df['msvr_direction'] = (df['msvr_signal'] > 0).astype(float)
    
    # ================================================================
    # Layer 2: Cycle Phase (Family 4: Spectral)
    # ================================================================
    phase, dominant_period = compute_cycle_phase(df, lookback=cycle_lookback)
    df['cycle_phase'] = phase
    df['cycle_signal'] = -np.cos(phase)  # +1 at trough, -1 at peak
    df['cycle_direction'] = (df['cycle_signal'] > 0).astype(float)
    
    # ================================================================
    # Layer 3: SuperSmoother (Family 2: Filtering)
    # ================================================================
    # Apply SuperSmoother to MSVR signal for noise reduction
    msvr_smooth = ehler_supersmoother(df['msvr_signal'], length=smooth_length)
    df['msvr_smooth'] = msvr_smooth
    df['smooth_direction'] = (msvr_smooth > 0).astype(float)
    
    # ================================================================
    # Layer 4: Efficiency Ratio (Family 5: Fractal)
    # ================================================================
    er = efficiency_ratio(df['close'], period=er_period)
    df['efficiency_ratio'] = er
    df['er_gate'] = (er > er_threshold).astype(float)
    
    # ================================================================
    # Layer 5: Shannon Entropy (Family 7: Entropy)
    # ================================================================
    entropy = shannon_entropy(df['close'], window=entropy_window, bins=6)
    df['entropy'] = entropy
    df['entropy_gate'] = (entropy < entropy_threshold).astype(float)
    
    # ================================================================
    # Composite Signal (Multi-Principle)
    # ================================================================
    # Combine all principles
    df['composite_raw'] = (
        df['msvr_direction'] * 0.3 +      # MSVR base
        df['cycle_direction'] * 0.25 +     # Cycle timing
        df['smooth_direction'] * 0.25 +    # Smoothed MSVR
        df['er_gate'] * 0.1 +              # Trend strength
        df['entropy_gate'] * 0.1           # Low entropy
    )
    
    # Binary signal (threshold = 0.5)
    df['composite_signal'] = (df['composite_raw'] > 0.5).astype(float)
    
    return df

def compute_metrics(df, prices):
    """Compute trading metrics."""
    positions = df['composite_signal']
    returns = prices.pct_change()
    strategy_returns = returns * positions.shift(1)
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) == 0:
        return {'cagr': 0, 'sharpe': 0, 'sortino': 0, 'calmar': 0, 'max_dd': 0, 'n_trades': 0, 'win_rate': 0}

    equity = (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25

    cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    
    # Win rate
    changes = positions.diff().fillna(0)
    n_trades = (changes.abs() > 0).sum() // 2
    
    in_position = False
    trade_returns = []
    for i, (date, pos) in enumerate(positions.items()):
        if pos == 1.0 and not in_position:
            in_position = True
            entry_price = prices.loc[date]
        elif pos == 0.0 and in_position:
            in_position = False
            exit_price = prices.loc[date]
            trade_ret = (exit_price - entry_price) / entry_price
            trade_returns.append(trade_ret)
    
    winning = sum(1 for r in trade_returns if r > 0)
    total = len(trade_returns)
    win_rate = winning / total * 100 if total > 0 else 0

    return {
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd * 100, 2),
        'n_trades': n_trades,
        'win_rate': round(win_rate, 1)
    }

if __name__ == "__main__":
    import importlib.util
    import json
    import warnings
    warnings.filterwarnings('ignore')
    
    print("=" * 70)
    print("ENHANCED MSVR — Multi-Principle Indicator")
    print("=" * 70)
    
    # Load data
    with open('/home/ubuntu/projects/quant-btc-mttd-system/data/btc_daily.json') as f:
        btc_data = json.load(f)
    
    df_full = pd.DataFrame(btc_data['aligned_data'])
    df_full['time'] = pd.to_datetime(df_full['time'])
    df_full = df_full.set_index('time')
    df_full = df_full[df_full.index >= '2018-01-01']
    
    print(f"\nData: {len(df_full)} bars ({df_full.index[0]} to {df_full.index[-1]})")
    
    # Run Enhanced MSVR
    df_enhanced = msvr_enhanced(df_full)
    
    # Compute metrics
    metrics = compute_metrics(df_enhanced, df_full['close'])
    
    print(f"\nEnhanced MSVR Performance:")
    print(f"  Sharpe:     {metrics['sharpe']}")
    print(f"  CAGR:       {metrics['cagr']}%")
    print(f"  MaxDD:      {metrics['max_dd']}%")
    print(f"  Win Rate:   {metrics['win_rate']}%")
    print(f"  Trades:     {metrics['n_trades']}")
    
    # Compare with Ichimoku
    from ichimoku_quant import generate_ichimoku_features, generate_ichimoku_signals
    
    df_ichimoku = generate_ichimoku_features(df_full)
    df_ichimoku = generate_ichimoku_signals(df_ichimoku)
    
    # Ichimoku metrics
    ich_positions = df_ichimoku['Pos']
    ich_returns = df_full['close'].pct_change() * ich_positions.shift(1)
    ich_returns = ich_returns.dropna()
    ich_equity = (1 + ich_returns).cumprod()
    ich_years = len(ich_returns) / 365.25
    ich_cagr = (ich_equity.iloc[-1]) ** (1/ich_years) - 1 if ich_years > 0 else 0
    ich_sharpe = ich_returns.mean() / ich_returns.std() * np.sqrt(365) if ich_returns.std() > 0 else 0
    ich_peak = ich_equity.cummax()
    ich_maxdd = ((ich_equity - ich_peak) / ich_peak).min()
    
    ich_changes = ich_positions.diff().fillna(0)
    ich_trades = (ich_changes.abs() > 0).sum() // 2
    in_pos = False
    ich_trade_rets = []
    for i, (date, pos) in enumerate(ich_positions.items()):
        if pos == 1.0 and not in_pos:
            in_pos = True
            entry = df_full.loc[date, 'close']
        elif pos == 0.0 and in_pos:
            in_pos = False
            exit_p = df_full.loc[date, 'close']
            ich_trade_rets.append((exit_p - entry) / entry)
    ich_winning = sum(1 for r in ich_trade_rets if r > 0)
    ich_winrate = ich_winning / len(ich_trade_rets) * 100 if ich_trade_rets else 0
    
    print(f"\nComparison with Ichimoku:")
    print(f"  {'Metric':<15} {'Enhanced MSVR':<15} {'Ichimoku':<15} {'Winner':<15}")
    print(f"  {'-'*60}")
    print(f"  {'Sharpe':<15} {metrics['sharpe']:<15} {ich_sharpe:<15} {'MSVR' if metrics['sharpe'] > ich_sharpe else 'Ichimoku'}")
    print(f"  {'CAGR':<15} {metrics['cagr']:<15} {ich_cagr*100:<15.1f} {'MSVR' if metrics['cagr'] > ich_cagr*100 else 'Ichimoku'}")
    print(f"  {'Win Rate':<15} {metrics['win_rate']:<15} {ich_winrate:<15.1f} {'MSVR' if metrics['win_rate'] > ich_winrate else 'Ichimoku'}")
    print(f"  {'Trades':<15} {metrics['n_trades']:<15} {ich_trades:<15} {'MSVR' if metrics['n_trades'] < ich_trades else 'Ichimoku'}")
