"""
Compute Best 5-Indicator Ensemble Combinations for ISP Coherence
================================================================

Tests all C(N,5) combinations of top performers with BINARY 100% EQUITY sizing.
For each combination:
- Equal-weighted ensemble signal (average of 5 indicator signals)
- Binary position: 100% if average > 0, else 0%
- Time-coherence with ISP (binary: in/out match)
- Expected Sharpe, Sortino, Max DD from historical backtest
"""

import os
import sys
import json
import csv
import importlib.util
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

project_root = "/home/ubuntu/projects/quant-technical-indicator-bank"
sys.path.append(project_root)
from indicators_helper import *

# Import system modules
from mttd.ensemble_engine import compute_ensemble_signal
from mttd.coherence_metrics import (
    load_isp_positions,
    compute_time_coherence,
    measure_coherence
)
from mttd.audit_indicators import (
    detect_direction_series,
    indicator_to_position,
    normalize_name as audit_normalize_name
)
from mttd.execute_system import load_data


def load_indicator_signals(indicator_info, df_eval):
    """Load and compute indicator signals."""
    name = indicator_info['indicator']
    category = indicator_info['category']
    normalized = indicator_info['normalized']
    
    py_file = os.path.join(project_root, category, f"{normalized}.py")
    
    if not os.path.exists(py_file):
        return None
    
    try:
        spec = importlib.util.spec_from_file_location(normalized, py_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, normalized)
        
        res_df = func(df_eval)
        
        if not isinstance(res_df, pd.DataFrame):
            return None
        
        direction = detect_direction_series(res_df)
        
        if direction is None:
            return None
        
        direction_aligned = direction.reindex(df_eval.index).fillna(0.0)
        indicator_position = indicator_to_position(direction_aligned)
        
        return indicator_position
        
    except Exception as e:
        return None


def compute_binary_ensemble_backtest(positions_dict, price_series, isp_positions):
    """
    Compute binary ensemble for a combination of indicators.
    
    BINARY SIZING: 100% BTC or 0% cash. No 50% positions.
    """
    indicator_names = list(positions_dict.keys())
    n_indicators = len(indicator_names)
    
    # Stack positions into matrix
    pos_matrix = pd.DataFrame(positions_dict)
    
    # Equal-weighted average signal
    avg_signal = pos_matrix.mean(axis=1)
    
    # BINARY POSITION: 100% if average > 0, else 0%
    binary_position = (avg_signal > 0).astype(float)
    
    # Compute time-coherence with ISP
    tc = compute_time_coherence(binary_position, isp_positions)
    
    # Compute strategy returns
    daily_returns = price_series.pct_change().fillna(0.0)
    strategy_returns = binary_position.shift(1).fillna(0.0) * daily_returns
    strategy_returns = strategy_returns.iloc[1:]
    
    # Performance metrics
    if len(strategy_returns) > 0:
        total_return = (1 + strategy_returns).prod() - 1
        years = len(strategy_returns) / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Sharpe (annualized)
        if strategy_returns.std() > 0:
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(365)
        else:
            sharpe = 0.0
        
        # Sortino (annualized)
        downside_returns = strategy_returns[strategy_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = (strategy_returns.mean() / downside_returns.std()) * np.sqrt(365)
        else:
            sortino = 0.0
        
        # Max Drawdown
        equity_curve = (1 + strategy_returns).cumprod()
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_dd = drawdown.min()
    else:
        total_return = 0
        annualized_return = 0
        sharpe = 0
        sortino = 0
        max_dd = 0
    
    # Count trades
    position_diff = binary_position.diff()
    n_entries = (position_diff == 1).sum()
    n_exits = (position_diff == -1).sum()
    n_trades = min(n_entries, n_exits)
    
    return {
        'indicators': indicator_names,
        'coherence': round(tc['coherence_pct'], 2),
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'max_dd': round(float(max_dd), 4),
        'total_return_pct': round(float(total_return * 100), 2),
        'annualized_return_pct': round(float(annualized_return * 100), 2),
        'n_trades': int(n_trades),
        'n_in_position': int(binary_position.sum()),
        'pct_in_position': round(float(binary_position.mean() * 100), 2)
    }


def main():
    print("=" * 80)
    print("5-INDICATOR ENSEMBLE COMBINATION ANALYSIS")
    print("BINARY 100% EQUITY SIZING")
    print("=" * 80)
    
    # Load price data
    print("\n[1] Loading price data...")
    df = load_data()
    print(f"    Loaded {len(df)} daily bars ({df.index[0]} to {df.index[-1]})")
    
    # Load ISP positions
    print("\n[2] Loading ISP positions...")
    csv_path = os.path.join(project_root, "isp-signals-btcusd-2026-06-13.csv")
    isp_positions = load_isp_positions(csv_path)
    print(f"    ISP positions loaded: {len(isp_positions)} bars")
    
    # Get ISP date range
    df_csv = pd.read_csv(csv_path)
    first_date = str(df_csv['Date'].iloc[0])
    last_date = str(df_csv['Date'].iloc[-1])
    
    df_eval = df.loc[first_date:last_date].copy()
    isp_eval = isp_positions.loc[first_date:last_date]
    price_eval = df_eval['close']
    
    print(f"    Evaluation range: {first_date} to {last_date} ({len(df_eval)} days)")
    print(f"    ISP in-position days: {int(isp_eval.sum())}")
    
    # Load coherence and performance data
    print("\n[3] Loading indicator data...")
    
    # Load coherence results
    with open(os.path.join(project_root, "mttd/all_indicator_coherence_results.json")) as f:
        coherence_data = json.load(f)
    
    # Load test results
    with open(os.path.join(project_root, "mttd/indicator_test_results.csv")) as f:
        reader = csv.DictReader(f)
        test1 = {row['normalized']: row for row in reader}
    
    with open(os.path.join(project_root, "mttd/all_indicator_test_results.csv")) as f:
        reader = csv.DictReader(f)
        test2 = {row['normalized']: row for row in reader}
    
    # Merge test results (test1 preferred)
    all_test = {**test2, **test1}
    
    # Build unified dataset: indicators with both coherence AND performance data
    unified = []
    for ind in coherence_data:
        name = ind['indicator']
        if name in all_test:
            t = all_test[name]
            unified.append({
                'indicator': name,
                'category': ind['category'],
                'normalized': name,
                'coherence': ind['coherence'],
                'sharpe': float(t['sharpe_ratio']),
                'max_dd': float(t['max_drawdown_pct']),
                'total_return': float(t['total_return_pct']),
                'trades': int(t['trades']),
                'avg_hold': float(t['avg_hold_days'])
            })
    
    # Sort by coherence and take top 20
    unified.sort(key=lambda x: x['coherence'], reverse=True)
    top20 = unified[:20]
    
    print(f"    Indicators with both coherence and performance: {len(unified)}")
    print(f"    Using top 20 by coherence")
    
    for i, ind in enumerate(top20):
        print(f"    {i+1:2d}. {ind['indicator']:50s} coh={ind['coherence']:.2f}%  sharpe={ind['sharpe']:.3f}")
    
    # Compute indicator signals
    print("\n[4] Computing indicator signals...")
    indicator_positions = {}
    for ind in top20:
        print(f"    Computing {ind['indicator']}...", end="", flush=True)
        pos = load_indicator_signals(ind, df_eval)
        if pos is not None:
            indicator_positions[ind['indicator']] = pos
            print(f" ✓ ({int(pos.sum())} days in position)")
        else:
            print(f" ✗ (failed)")
    
    print(f"\n    Successfully loaded: {len(indicator_positions)}/{len(top20)} indicators")
    
    if len(indicator_positions) < 5:
        print("ERROR: Need at least 5 indicators to form combinations")
        return
    
    # Test all C(N,5) combinations
    print(f"\n[5] Testing all C({len(indicator_positions)},5) combinations...")
    indicator_names = list(indicator_positions.keys())
    all_combinations = list(itertools.combinations(indicator_names, 5))
    print(f"    Total combinations to test: {len(all_combinations)}")
    
    results = []
    for idx, combo in enumerate(all_combinations, 1):
        if idx % 1000 == 0:
            print(f"    Processing {idx}/{len(all_combinations)}...", flush=True)
        
        # Get positions for this combination
        combo_positions = {name: indicator_positions[name] for name in combo}
        
        # Compute ensemble metrics
        result = compute_binary_ensemble_backtest(combo_positions, price_eval, isp_eval)
        results.append(result)
    
    # Sort by coherence (primary) and Sharpe (secondary)
    results.sort(key=lambda x: (-x['coherence'], -x['sharpe']))
    
    # Get top 10
    top10 = results[:10]
    
    print(f"\n[6] Top 10 combinations by coherence:")
    print("=" * 100)
    for i, r in enumerate(top10):
        print(f"\n  #{i+1}")
        print(f"    Indicators: {', '.join(r['indicators'][:3])}...")
        print(f"    Coherence:  {r['coherence']:.2f}%")
        print(f"    Sharpe:     {r['sharpe']:.4f}")
        print(f"    Sortino:    {r['sortino']:.4f}")
        print(f"    Max DD:     {r['max_dd']:.4f}")
        print(f"    Trades:     {r['n_trades']}")
        print(f"    Return:     {r['total_return_pct']:.2f}%")
        print(f"    In-Pos %:   {r['pct_in_position']:.1f}%")
    
    # Save results
    output_path = os.path.join(project_root, "ensemble_combinations_5ind.json")
    with open(output_path, 'w') as f:
        json.dump(top10, f, indent=2)
    
    print(f"\n[7] Results saved to: {output_path}")
    print("=" * 80)
    
    return top10


if __name__ == "__main__":
    results = main()
