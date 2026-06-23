#!/usr/bin/env python3
"""
MTTD Grid Search — Optimize parameters to match ISP performance metrics.
"""
import os
import sys
import json
import importlib.util
import re
import numpy as np
import pandas as pd
from itertools import product

project_root = "/home/ubuntu/projects/quant-technical-indicator-bank"
sys.path.append(project_root)
from indicators_helper import *

# ================================================================
# Load Data
# ================================================================
def load_data():
    cache_path = os.path.join(project_root, "mttd", "data", "btc_daily.json")
    with open(cache_path, 'r') as f:
        raw = json.load(f)
    
    if 'candles' in raw:
        candles = raw['candles']
    elif isinstance(raw, list):
        candles = raw
    else:
        for key in raw.keys():
            if isinstance(raw[key], list) and len(raw[key]) > 100:
                candles = raw[key]
                break
    
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df = df.sort_index()
    return df[df.index >= '2018-01-01'].copy()

# ================================================================
# ISP Position Series
# ================================================================
def build_isp_positions(df):
    csv_path = os.path.join(project_root, "isp-signals-btcusd-2026-06-13.csv")
    isp = pd.read_csv(csv_path)
    isp['Date'] = pd.to_datetime(isp['Date'])
    
    isp_position = pd.Series(0.0, index=df.index)
    for _, row in isp.iterrows():
        date = row['Date']
        regime = row['Regime']
        if date in isp_position.index:
            if regime in ['Strong Bull', 'Weak Bull']:
                isp_position.loc[date:] = 1.0
            else:
                isp_position.loc[date:] = 0.0
        else:
            nearest_idx = df.index.get_indexer([date], method='nearest')[0]
            nearest_date = df.index[nearest_idx]
            if regime in ['Strong Bull', 'Weak Bull']:
                isp_position.loc[nearest_date:] = 1.0
            else:
                isp_position.loc[nearest_date:] = 0.0
    
    return isp_position

# ================================================================
# Indicator Loading
# ================================================================
def normalize_name(name):
    n = name.replace("(", "").replace(")", "")
    n = n.replace("%", "")
    n = re.sub(r"[|:\-`]", " ", n)
    n = n.lower().strip()
    n = re.sub(r"\s+", "_", n)
    n = re.sub(r"_+", "_", n)
    return n

def detect_direction_series(res_df):
    for col in ['dir', 'sig', 'direction', 'vii', 'qb', 'st_direction', 'trend_direction', 'trend']:
        if col in res_df.columns:
            return res_df[col]
    if 'long_signal' in res_df.columns and 'short_signal' in res_df.columns:
        direction = pd.Series(0.0, index=res_df.index)
        curr = 0.0
        for i in range(len(res_df)):
            l = res_df['long_signal'].iloc[i]
            s = res_df['short_signal'].iloc[i]
            l_val = bool(l) if not pd.isna(l) else False
            s_val = bool(s) if not pd.isna(s) else False
            if l_val and not s_val:
                curr = 1.0
            elif s_val and not l_val:
                curr = -1.0
            direction.iloc[i] = curr
        return direction
    if 'in_long_position' in res_df.columns and 'in_short_position' in res_df.columns:
        direction = pd.Series(0.0, index=res_df.index)
        direction[res_df['in_long_position'] == 1] = 1.0
        direction[res_df['in_short_position'] == 1] = -1.0
        return direction
    for col in res_df.columns:
        col_lower = col.lower()
        if 'direction' in col_lower or 'signal' in col_lower or 'trend' in col_lower:
            unique_vals = res_df[col].dropna().unique()
            if len(unique_vals) <= 10:
                return res_df[col]
    return None

def load_indicators(df):
    """Load all 15 indicators and return their direction signals."""
    SELECTED_INDICATORS = [
        {"name": "Adaptive Regime Cloud", "category": "perpetual"},
        {"name": "Adaptive Volatility Controlled LSMA | QuantAlgo", "category": "perpetual"},
        {"name": "Polynomial Deviation Bands", "category": "perpetual"},
        {"name": "alma lag | viResearch", "category": "perpetual"},
        {"name": "lsma | viResearch", "category": "perpetual"},
        {"name": "DSMA | viResearch", "category": "perpetual"},
        {"name": "IRS`Elder Force Volume Index", "category": "perpetual"},
        {"name": "Gaussian Smooth Trend | QuantEdgeB", "category": "perpetual"},
        {"name": "DEGA RMA | QuantEdgeB", "category": "perpetual"},
        {"name": "Linear % ST | QuantEdgeB", "category": "perpetual"},
        {"name": "Quantile DEMA Trend | QuantEdgeB", "category": "perpetual"},
        {"name": "HILO Interpolation | QuantEdgeB", "category": "perpetual"},
        {"name": "MadTrend | InvestorUnknown", "category": "perpetual"},
        {"name": "Median Deviation Suite | InvestorUnknown", "category": "perpetual"},
        {"name": "Root Mean Square Deviation Trend", "category": "perpetual"}
    ]
    
    signals = {}
    for ind in SELECTED_INDICATORS:
        name = ind['name']
        cat = ind['category']
        normalized = normalize_name(name)
        py_file = os.path.join(project_root, cat, f"{normalized}.py")
        
        try:
            spec = importlib.util.spec_from_file_location(normalized, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            func = getattr(module, normalized)
            res_df = func(df)
            dir_series = detect_direction_series(res_df)
            if dir_series is not None:
                dir_series = dir_series.reindex(df.index).fillna(0.0)
                signals[normalized] = (dir_series > 0).astype(float) * 2 - 1  # Convert to -1/+1
        except Exception as e:
            print(f"  Warning: Failed to load {name}: {e}")
    
    return signals

# ================================================================
# Metrics Calculation
# ================================================================
def calculate_metrics(returns):
    returns = returns.dropna()
    if len(returns) < 10:
        return {'sharpe': 0, 'sortino': 0, 'omega': 0, 'max_dd': 0}
    
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    sharpe = (mean_ret / std_ret) * np.sqrt(365) if std_ret > 0 else 0
    
    downside_returns = returns[returns < 0]
    downside_std = downside_returns.std()
    sortino = (mean_ret / downside_std) * np.sqrt(365) if downside_std > 0 else 0
    
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    omega = gains / losses if losses > 0 else 1.0
    
    equity = (1 + returns).cumprod()
    max_dd = ((equity - equity.cummax()) / equity.cummax()).min() * 100
    
    return {
        'sharpe': sharpe,
        'sortino': sortino,
        'omega': omega,
        'max_dd': max_dd,
        'total_return': ((1 + returns).prod() - 1) * 100
    }

# ================================================================
# Ensemble Computation
# ================================================================
def compute_ensemble(signals_dict, threshold, ema_length, use_sma_filter=True):
    """Compute ensemble position from indicator signals."""
    # Stack signals into matrix
    signal_names = list(signals_dict.keys())
    signal_matrix = pd.DataFrame(signals_dict)
    
    # Average signal
    avg_signal = signal_matrix.mean(axis=1)
    
    # EMA smoothing
    if ema_length > 1:
        avg_signal = avg_signal.ewm(span=ema_length, adjust=False).mean()
    
    # Threshold
    position = (avg_signal > threshold).astype(float)
    
    return position

def apply_sma_filter(position, close_prices, sma_period=200):
    """Apply 200-day SMA filter."""
    sma = close_prices.rolling(window=sma_period, min_periods=sma_period).mean()
    regime_filter = (close_prices > sma).astype(float)
    return position * regime_filter

def backtest(position, df, exclude_2018=True):
    """Run backtest and return metrics."""
    returns = df['close'].pct_change()
    
    # Position shifted by 1 day to avoid look-ahead
    position_shifted = position.shift(1).fillna(0)
    strategy_returns = returns * position_shifted
    
    if exclude_2018:
        mask = df.index.year != 2018
        strategy_returns = strategy_returns[mask]
    
    return calculate_metrics(strategy_returns)

# ================================================================
# Grid Search
# ================================================================
def grid_search(df, signals_dict, isp_positions):
    """Grid search over parameters to find best match to ISP metrics."""
    
    # ISP target metrics (No 2018)
    target = {
        'sharpe': 1.7538,
        'sortino': 1.6845,
        'omega': 1.5387
    }
    
    # Parameter grid
    thresholds = np.arange(-0.2, 0.8, 0.05)
    ema_lengths = [1, 2, 3, 5, 7, 10]
    use_sma = [True, False]
    
    results = []
    total_combos = len(thresholds) * len(ema_lengths) * len(use_sma)
    
    print(f"Grid search: {total_combos} combinations")
    print(f"Target: Sharpe={target['sharpe']:.4f}, Sortino={target['sortino']:.4f}, Omega={target['omega']:.4f}")
    print()
    
    for i, (thresh, ema, sma) in enumerate(product(thresholds, ema_lengths, use_sma)):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{total_combos}")
        
        # Compute ensemble
        position = compute_ensemble(signals_dict, thresh, ema)
        
        # Apply SMA filter if enabled
        if sma:
            position = apply_sma_filter(position, df['close'], sma_period=200)
        
        # Backtest
        metrics = backtest(position, df, exclude_2018=True)
        
        # Calculate distance to target
        dist = np.sqrt(
            (metrics['sharpe'] - target['sharpe'])**2 +
            (metrics['sortino'] - target['sortino'])**2 +
            (metrics['omega'] - target['omega'])**2
        )
        
        # Also check 2018 trades
        position_2018 = position[df.index.year == 2018]
        trades_2018 = (position_2018.diff().abs() > 0).sum()
        
        results.append({
            'threshold': thresh,
            'ema_length': ema,
            'use_sma_filter': sma,
            'sharpe': metrics['sharpe'],
            'sortino': metrics['sortino'],
            'omega': metrics['omega'],
            'max_dd': metrics['max_dd'],
            'total_return': metrics['total_return'],
            'distance': dist,
            'trades_2018': trades_2018
        })
    
    # Sort by distance
    results.sort(key=lambda x: x['distance'])
    
    return results

# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("MTTD GRID SEARCH — Matching ISP Performance")
    print("=" * 80)
    print()
    
    # Load data
    print("[1] Loading BTC data...")
    df = load_data()
    print(f"    Loaded {len(df)} days")
    
    # Load indicators
    print("[2] Loading indicators...")
    signals_dict = load_indicators(df)
    print(f"    Loaded {len(signals_dict)} indicators")
    
    # Build ISP positions
    print("[3] Building ISP position series...")
    isp_positions = build_isp_positions(df)
    
    # Grid search
    print("[4] Running grid search...")
    results = grid_search(df, signals_dict, isp_positions)
    
    # Print top 10 results
    print()
    print("=" * 80)
    print("TOP 10 RESULTS")
    print("=" * 80)
    print()
    header = f"{'Rank':<6} {'Thresh':<8} {'EMA':<6} {'SMA':<6} {'Sharpe':<10} {'Sortino':<10} {'Omega':<10} {'MaxDD%':<10} {'2018':<6} {'Dist':<10}"
    print(header)
    print("-" * 86)
    
    for i, r in enumerate(results[:10]):
        row = f"{i+1:<6} {r['threshold']:<8.2f} {r['ema_length']:<6} {str(r['use_sma_filter']):<6} {r['sharpe']:<10.4f} {r['sortino']:<10.4f} {r['omega']:<10.4f} {r['max_dd']:<10.2f} {r['trades_2018']:<6} {r['distance']:<10.4f}"
        print(row)
    
    # Best result details
    best = results[0]
    print()
    print("=" * 80)
    print("BEST RESULT")
    print("=" * 80)
    print(f"  Threshold:      {best['threshold']:.2f}")
    print(f"  EMA Length:     {best['ema_length']}")
    print(f"  SMA Filter:     {best['use_sma_filter']}")
    print(f"  Sharpe:         {best['sharpe']:.4f} (target: 1.7538)")
    print(f"  Sortino:        {best['sortino']:.4f} (target: 1.6845)")
    print(f"  Omega:          {best['omega']:.4f} (target: 1.5387)")
    print(f"  Max Drawdown:   {best['max_dd']:.2f}%")
    print(f"  Total Return:   {best['total_return']:.2f}%")
    print(f"  2018 Trades:    {best['trades_2018']}")
    print(f"  Distance:       {best['distance']:.4f}")
    
    # Save results
    output_path = os.path.join(project_root, "mttd", "grid_search_results.json")
    with open(output_path, 'w') as f:
        json.dump(results[:50], f, indent=2, default=str)
    print(f"\n  Results saved to: {output_path}")
