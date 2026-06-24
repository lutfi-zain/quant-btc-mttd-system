#!/usr/bin/env python3
"""
Test Hybrid Performance vs Baselines
=====================================

Compares:
1. MSVR Hybrid (new) - MSVR + Ichimoku filtering principles
2. MSVR Enhanced (baseline) - Multi-principle MSVR
3. Ichimoku (baseline) - Advanced Ichimoku system

Target metrics: < 20 trades, > 60% win rate, Sharpe > 1.35
"""

import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

def load_btc_data():
    """Load BTC daily data from 2018-01-01."""
    data_path = '/home/ubuntu/projects/quant-btc-mttd-system/data/btc_daily.json'
    with open(data_path) as f:
        btc_data = json.load(f)
    df = pd.DataFrame(btc_data['aligned_data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df[df.index >= '2018-01-01']
    return df

def run_hybrid(df):
    """Run MSVR Hybrid strategy."""
    from msvr_hybrid import (
        generate_composite_signal, 
        enforce_min_hold, 
        compute_trade_list, 
        compute_metrics
    )
    
    df = df.copy()
    df = generate_composite_signal(df)
    df['position'] = enforce_min_hold(df['composite_signal'], min_hold=5)
    trades_df = compute_trade_list(df, df['close'], transaction_cost=0.001)
    metrics = compute_metrics(trades_df, positions=df['position'], df_prices=df['close'])
    
    return metrics, trades_df

def run_msvr_enhanced(df):
    """Run MSVR Enhanced strategy."""
    from msvr_enhanced import msvr_enhanced, compute_metrics as msvr_compute_metrics
    
    df = df.copy()
    df = msvr_enhanced(df)
    metrics = msvr_compute_metrics(df, df['close'])
    
    # Also compute trade list for comparison
    positions = df['composite_signal']
    trades = []
    in_position = False
    entry_date = None
    entry_price = None
    
    for i, (date, pos) in enumerate(positions.items()):
        if pos == 1.0 and not in_position:
            in_position = True
            entry_date = date
            entry_price = df.loc[date, 'close']
        elif pos == 0.0 and in_position:
            in_position = False
            exit_price = df.loc[date, 'close']
            trade_ret = (exit_price - entry_price) / entry_price
            trades.append({
                'entry_date': entry_date,
                'exit_date': date,
                'return': trade_ret
            })
    
    trades_df = pd.DataFrame(trades)
    return metrics, trades_df

def run_ichimoku(df):
    """Run Ichimoku strategy."""
    from ichimoku_quant import (
        generate_ichimoku_features,
        generate_ichimoku_signals,
        compute_ichimoku_metrics
    )
    
    df = df.copy()
    df = generate_ichimoku_features(df)
    df = generate_ichimoku_signals(df)
    metrics = compute_ichimoku_metrics(df, df['close'])
    
    # Build trade list
    positions = df['Pos']
    trades = []
    in_position = False
    entry_date = None
    entry_price = None
    
    for i, (date, pos) in enumerate(positions.items()):
        if pos == 1.0 and not in_position:
            in_position = True
            entry_date = date
            entry_price = df.loc[date, 'close']
        elif pos == 0.0 and in_position:
            in_position = False
            exit_price = df.loc[date, 'close']
            trade_ret = (exit_price - entry_price) / entry_price
            trades.append({
                'entry_date': entry_date,
                'exit_date': date,
                'return': trade_ret
            })
    
    trades_df = pd.DataFrame(trades)
    return metrics, trades_df

def print_comparison(hybrid_metrics, msvr_metrics, ichimoku_metrics):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON: MSVR Hybrid vs Baselines")
    print("=" * 80)
    
    # Header
    print(f"\n{'Metric':<20} {'MSVR Hybrid':<15} {'MSVR Enhanced':<15} {'Ichimoku':<15} {'Target':<15}")
    print("-" * 80)
    
    # Trades
    h_trades = hybrid_metrics.get('n_trades', 0)
    m_trades = msvr_metrics.get('n_trades', 0)
    i_trades = ichimoku_metrics.get('n_trades', 0)
    print(f"{'Trades':<20} {h_trades:<15} {m_trades:<15} {i_trades:<15} {'< 20':<15}")
    
    # Win Rate
    h_wr = hybrid_metrics.get('win_rate', 0)
    m_wr = msvr_metrics.get('win_rate', 0)
    i_wr = ichimoku_metrics.get('win_rate', 0)
    print(f"{'Win Rate':<20} {h_wr:<15.1f} {m_wr:<15.1f} {i_wr:<15.1f} {'> 60%':<15}")
    
    # Sharpe
    h_sharpe = hybrid_metrics.get('sharpe', 0)
    m_sharpe = msvr_metrics.get('sharpe', 0)
    i_sharpe = ichimoku_metrics.get('sharpe', 0)
    print(f"{'Sharpe':<20} {h_sharpe:<15.2f} {m_sharpe:<15.2f} {i_sharpe:<15.2f} {'> 1.35':<15}")
    
    # Total Return
    h_ret = hybrid_metrics.get('total_return', 0)
    m_ret = msvr_metrics.get('total_return', 0)
    i_ret = ichimoku_metrics.get('total_return', 0)
    print(f"{'Total Return':<20} {h_ret:<15.2f} {m_ret:<15.2f} {i_ret:<15.2f} {'N/A':<15}")
    
    # CAGR (if available)
    h_cagr = hybrid_metrics.get('cagr', 0)
    m_cagr = msvr_metrics.get('cagr', 0)
    i_cagr = ichimoku_metrics.get('cagr', 0)
    print(f"{'CAGR':<20} {h_cagr:<15.2f} {m_cagr:<15.2f} {i_cagr:<15.2f} {'N/A':<15}")
    
    # Max Drawdown (if available)
    h_dd = hybrid_metrics.get('max_dd', 0)
    m_dd = msvr_metrics.get('max_dd', 0)
    i_dd = ichimoku_metrics.get('max_dd', 0)
    print(f"{'Max Drawdown':<20} {h_dd:<15.2f} {m_dd:<15.2f} {i_dd:<15.2f} {'N/A':<15}")
    
    print("\n" + "=" * 80)
    print("CONSTRAINT VERIFICATION")
    print("=" * 80)
    
    # Verify constraints
    constraints = [
        ("Trades < 20", h_trades < 20, f"{h_trades} trades"),
        ("Win Rate > 60%", h_wr > 60, f"{h_wr:.1f}%"),
        ("Sharpe > 1.35", h_sharpe > 1.35, f"{h_sharpe:.2f}")
    ]
    
    for name, passed, value in constraints:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status} ({value})")
    
    # Comparison with Ichimoku
    print("\n" + "=" * 80)
    print("COMPARISON vs ICHIMOKU")
    print("=" * 80)
    
    comparisons = [
        ("Selectivity (fewer trades)", h_trades, i_trades, "less"),
        ("Win Rate", h_wr, i_wr, "higher"),
        ("Sharpe", h_sharpe, i_sharpe, "higher"),
        ("Total Return", h_ret, i_ret, "higher")
    ]
    
    for name, hybrid_val, ichimoku_val, better in comparisons:
        if better == "less":
            winner = "Hybrid" if hybrid_val < ichimoku_val else "Ichimoku"
        else:
            winner = "Hybrid" if hybrid_val > ichimoku_val else "Ichimoku"
        
        diff = abs(hybrid_val - ichimoku_val)
        if ichimoku_val != 0:
            pct_diff = diff / abs(ichimoku_val) * 100
            print(f"  {name}: {winner} (Hybrid: {hybrid_val:.2f}, Ichimoku: {ichimoku_val:.2f}, diff: {pct_diff:.1f}%)")
        else:
            print(f"  {name}: {winner} (Hybrid: {hybrid_val:.2f}, Ichimoku: {ichimoku_val:.2f})")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Overall assessment
    meets_trades = h_trades < 20
    meets_wr = h_wr > 60
    meets_sharpe = h_sharpe > 1.35
    
    if meets_trades and meets_wr and meets_sharpe:
        print("\n✓ HYBRID MEETS ALL TARGET METRICS")
    else:
        print("\n✗ HYBRID DOES NOT MEET ALL TARGET METRICS")
        if not meets_trades:
            print(f"  - Trades: {h_trades} (target: < 20)")
        if not meets_wr:
            print(f"  - Win Rate: {h_wr:.1f}% (target: > 60%)")
        if not meets_sharpe:
            print(f"  - Sharpe: {h_sharpe:.2f} (target: > 1.35)")
    
    # Comparison assessment
    beats_ichimoku_selectivity = h_trades < i_trades
    beats_ichimoku_winrate = h_wr > i_wr
    beats_ichimoku_sharpe = h_sharpe > i_sharpe
    
    print(f"\nvs Ichimoku:")
    print(f"  - Selectivity: {'✓' if beats_ichimoku_selectivity else '✗'} ({h_trades} vs {i_trades} trades)")
    print(f"  - Win Rate: {'✓' if beats_ichimoku_winrate else '✗'} ({h_wr:.1f}% vs {i_wr:.1f}%)")
    print(f"  - Sharpe: {'✓' if beats_ichimoku_sharpe else '✗'} ({h_sharpe:.2f} vs {i_sharpe:.2f})")

if __name__ == "__main__":
    print("=" * 80)
    print("MSVR HYBRID PERFORMANCE TEST")
    print("=" * 80)
    
    # Load data
    df = load_btc_data()
    print(f"\nData: {len(df)} bars ({df.index[0]} to {df.index[-1]})")
    
    # Run strategies
    print("\nRunning MSVR Hybrid...")
    hybrid_metrics, hybrid_trades = run_hybrid(df)
    
    print("Running MSVR Enhanced...")
    msvr_metrics, msvr_trades = run_msvr_enhanced(df)
    
    print("Running Ichimoku...")
    ichimoku_metrics, ichimoku_trades = run_ichimoku(df)
    
    # Print comparison
    print_comparison(hybrid_metrics, msvr_metrics, ichimoku_metrics)
    
    # Print trade details
    print("\n" + "=" * 80)
    print("HYBRID TRADE LIST (Top 10 by return)")
    print("=" * 80)
    
    if not hybrid_trades.empty:
        top_trades = hybrid_trades.nlargest(10, 'return')
        print(top_trades.to_string(index=False))
    else:
        print("No trades generated.")
