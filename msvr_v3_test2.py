#!/usr/bin/env python3
"""Quick test script for MSVR v3 - precompute layers."""

import numpy as np
import pandas as pd
import sys
import os
import json
import importlib.util
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INDICATOR_BANK = '/home/ubuntu/projects/quant-technical-indicator-bank'
sys.path.append(PROJECT_ROOT)
sys.path.append(INDICATOR_BANK)

from msvr_v3 import (load_btc_data, layer1_msvr_base, layer2_supersmoother,
                      layer3_linear_reg, layer4_cycle_phase, layer5_efficiency_ratio,
                      layer6_volatility_gate, layer7_entropy_gate, layer8_volume_confirm,
                      layer9_hmm_regime, backtest)


def precompute_layers(df):
    """Precompute all layers once."""
    print("Precomputing layers...")
    msvr_raw = layer1_msvr_base(df)
    msvr_smooth7 = layer2_supersmoother(msvr_raw, length=7)
    msvr_smooth10 = layer2_supersmoother(msvr_raw, length=10)
    lr_dir = layer3_linear_reg(df, length=50)
    cycle_dir40 = layer4_cycle_phase(df, lookback=40)
    er_dir20 = layer5_efficiency_ratio(df, period=14, threshold=0.20)
    er_dir25 = layer5_efficiency_ratio(df, period=14, threshold=0.25)
    er_dir30 = layer5_efficiency_ratio(df, period=14, threshold=0.30)
    vol_dir12 = layer6_volatility_gate(df, window=20, threshold=1.2)
    vol_dir15 = layer6_volatility_gate(df, window=20, threshold=1.5)
    vol_dir18 = layer6_volatility_gate(df, window=20, threshold=1.8)
    entropy_dir18 = layer7_entropy_gate(df, window=15, threshold=1.8)
    entropy_dir20 = layer7_entropy_gate(df, window=15, threshold=2.0)
    entropy_dir22 = layer7_entropy_gate(df, window=15, threshold=2.2)
    entropy_dir25 = layer7_entropy_gate(df, window=15, threshold=2.5)
    vol_confirm_dir = layer8_volume_confirm(df, obv_short=10, obv_long=30)
    hmm_dir = layer9_hmm_regime(df, n_states=3, lookback=250)

    # Convert MSVR to binary
    msvr_bullish = (msvr_raw > 0).astype(int)
    smooth7_bullish = (msvr_smooth7 > 0).astype(int)
    smooth10_bullish = (msvr_smooth10 > 0).astype(int)
    lr_confirm = (lr_dir == 1).astype(int)

    layers = {
        'msvr_raw': msvr_raw,
        'msvr_bullish': msvr_bullish,
        'smooth7': msvr_smooth7,
        'smooth7_bullish': smooth7_bullish,
        'smooth10': msvr_smooth10,
        'smooth10_bullish': smooth10_bullish,
        'lr_confirm': lr_confirm,
        'cycle40': cycle_dir40,
        'er20': er_dir20,
        'er25': er_dir25,
        'er30': er_dir30,
        'vol12': vol_dir12,
        'vol15': vol_dir15,
        'vol18': vol_dir18,
        'ent18': entropy_dir18,
        'ent20': entropy_dir20,
        'ent22': entropy_dir22,
        'ent25': entropy_dir25,
        'vol_confirm': vol_confirm_dir,
        'hmm': hmm_dir,
    }
    print("Done precomputing.")
    return layers


def test_config(df, layers, config, min_hold=45):
    """Test a specific configuration using precomputed layers."""
    smooth_key = config.get('smooth', 'smooth7')
    cycle_key = config.get('cycle', 'cycle40')
    er_key = config.get('er', 'er25')
    vol_key = config.get('vol', 'vol15')
    entropy_key = config.get('entropy', 'ent20')
    gate_threshold = config.get('gate_threshold', 2)
    trailing_stop_pct = config.get('trailing_stop_pct', 0.15)
    momentum_exit_days = config.get('momentum_exit_days', 10)
    require_lr = config.get('require_lr', False)

    # Core signal
    core_signal = layers['msvr_bullish'] * layers[smooth_key + '_bullish']

    # Timing
    timing_pass = ((layers[cycle_key] == 1) | (layers[er_key] == 1)).astype(int)

    # Gates
    gate_votes = ((layers[vol_key] == 1).astype(int) +
                  (layers[entropy_key] == 1).astype(int) +
                  (layers['vol_confirm'] == 1).astype(int) +
                  (layers['hmm'] == 1).astype(int))
    gates_pass = (gate_votes >= gate_threshold).astype(int)

    # Entry
    if require_lr:
        entry_signal = core_signal * timing_pass * gates_pass * layers['lr_confirm']
    else:
        entry_signal = core_signal * timing_pass * gates_pass

    # Build position
    pos = pd.Series(0, index=df.index)
    in_position = False
    hold_days = 0
    confirm_count = 0
    peak_price = 0
    momentum_below_zero_days = 0

    prices = df['close']
    smooth = layers[smooth_key]
    for i in range(len(df)):
        entry = entry_signal.iloc[i]
        price = prices.iloc[i]
        smooth_val = smooth.iloc[i]

        if not in_position:
            if entry == 1:
                confirm_count += 1
                if confirm_count >= 2:
                    in_position = True
                    hold_days = 0
                    peak_price = price
                    momentum_below_zero_days = 0
                    pos.iloc[i] = 1
            else:
                confirm_count = 0
        else:
            hold_days += 1
            pos.iloc[i] = 1
            if price > peak_price:
                peak_price = price
            if smooth_val < 0:
                momentum_below_zero_days += 1
            else:
                momentum_below_zero_days = 0

            if hold_days >= min_hold:
                price_drop = (peak_price - price) / peak_price
                if price_drop >= trailing_stop_pct:
                    in_position = False
                    hold_days = 0
                    pos.iloc[i] = 0
                    continue
                if momentum_below_zero_days >= momentum_exit_days:
                    in_position = False
                    hold_days = 0
                    pos.iloc[i] = 0
                    continue

    # Backtest
    metrics = backtest(df, pos, transaction_cost=0.001)
    return metrics


if __name__ == "__main__":
    print("Loading data...")
    df = load_btc_data()
    print(f"Data: {len(df)} bars\n")

    layers = precompute_layers(df)

    # Test configurations - focused on finding the sweet spot
    configs = [
        # Name, params, min_hold
        ("2 gates, 20% stop, 10d mom", {'gate_threshold': 2, 'trailing_stop_pct': 0.20, 'momentum_exit_days': 10}, 45),
        ("2 gates, 25% stop, 10d mom", {'gate_threshold': 2, 'trailing_stop_pct': 0.25, 'momentum_exit_days': 10}, 45),
        ("2 gates, 30% stop, 10d mom", {'gate_threshold': 2, 'trailing_stop_pct': 0.30, 'momentum_exit_days': 10}, 45),
        ("3 gates, 20% stop, 10d mom", {'gate_threshold': 3, 'trailing_stop_pct': 0.20, 'momentum_exit_days': 10}, 45),
        ("3 gates, 25% stop, 10d mom", {'gate_threshold': 3, 'trailing_stop_pct': 0.25, 'momentum_exit_days': 10}, 45),
        ("3 gates, 30% stop, 10d mom", {'gate_threshold': 3, 'trailing_stop_pct': 0.30, 'momentum_exit_days': 10}, 45),
        ("3 gates, 25% stop, 15d mom", {'gate_threshold': 3, 'trailing_stop_pct': 0.25, 'momentum_exit_days': 15}, 45),
        ("3 gates, 30% stop, 15d mom", {'gate_threshold': 3, 'trailing_stop_pct': 0.30, 'momentum_exit_days': 15}, 45),
        # With LR confirmation
        ("2g + LR, 20% stop", {'gate_threshold': 2, 'trailing_stop_pct': 0.20, 'momentum_exit_days': 10, 'require_lr': True}, 45),
        ("2g + LR, 25% stop", {'gate_threshold': 2, 'trailing_stop_pct': 0.25, 'momentum_exit_days': 10, 'require_lr': True}, 45),
        ("3g + LR, 25% stop", {'gate_threshold': 3, 'trailing_stop_pct': 0.25, 'momentum_exit_days': 10, 'require_lr': True}, 45),
        ("3g + LR, 30% stop", {'gate_threshold': 3, 'trailing_stop_pct': 0.30, 'momentum_exit_days': 10, 'require_lr': True}, 45),
        # Different entropy thresholds
        ("3g, ent=2.2, 25% stop", {'gate_threshold': 3, 'entropy': 'ent22', 'trailing_stop_pct': 0.25, 'momentum_exit_days': 10}, 45),
        ("3g, ent=2.5, 25% stop", {'gate_threshold': 3, 'entropy': 'ent25', 'trailing_stop_pct': 0.25, 'momentum_exit_days': 10}, 45),
        ("3g + LR, ent=2.2, 30% stop", {'gate_threshold': 3, 'entropy': 'ent22', 'trailing_stop_pct': 0.30, 'momentum_exit_days': 15, 'require_lr': True}, 45),
        # Different ER thresholds
        ("3g, er=0.20, 25% stop", {'gate_threshold': 3, 'er': 'er20', 'trailing_stop_pct': 0.25, 'momentum_exit_days': 10}, 45),
        ("3g, er=0.30, 25% stop", {'gate_threshold': 3, 'er': 'er30', 'trailing_stop_pct': 0.25, 'momentum_exit_days': 10}, 45),
        # Different smooth lengths
        ("3g, smooth10, 25% stop", {'gate_threshold': 3, 'smooth': 'smooth10', 'trailing_stop_pct': 0.25, 'momentum_exit_days': 10}, 45),
        ("3g + LR, smooth10, 30% stop", {'gate_threshold': 3, 'smooth': 'smooth10', 'trailing_stop_pct': 0.30, 'momentum_exit_days': 15, 'require_lr': True}, 45),
        # Best combinations
        ("3g + LR, ent=2.2, er=0.20, 30% stop", {'gate_threshold': 3, 'entropy': 'ent22', 'er': 'er20', 'trailing_stop_pct': 0.30, 'momentum_exit_days': 15, 'require_lr': True}, 45),
        ("3g + LR, ent=2.2, smooth10, 30% stop", {'gate_threshold': 3, 'entropy': 'ent22', 'smooth': 'smooth10', 'trailing_stop_pct': 0.30, 'momentum_exit_days': 15, 'require_lr': True}, 45),
    ]

    best_sharpe = 0
    best_name = ""
    best_metrics = None

    print(f"\n{'Config':<45} {'Trades':>6} {'WinRate':>8} {'Sharpe':>7} {'CAGR':>7} {'TimeInMkt':>10}")
    print("-" * 90)

    for name, params, mh in configs:
        metrics = test_config(df, layers, params, min_hold=mh)
        print(f"{name:<45} {metrics['n_trades']:>6} {metrics['win_rate']:>7.1f}% {metrics['sharpe']:>7.2f} {metrics['cagr']:>6.1f}% {metrics['time_in_market']:>9.1f}%")

        if metrics['sharpe'] > best_sharpe:
            best_sharpe = metrics['sharpe']
            best_name = name
            best_metrics = metrics

    print(f"\n{'=' * 90}")
    print(f"BEST: {best_name}")
    print(f"  Trades: {best_metrics['n_trades']}, Win Rate: {best_metrics['win_rate']}%, Sharpe: {best_metrics['sharpe']}")
    print(f"  CAGR: {best_metrics['cagr']}%, Max DD: {best_metrics['max_dd']}%, Time In Market: {best_metrics['time_in_market']}%")
    print(f"\nConstraint Check:")
    print(f"  Trades < 15: {'✓' if best_metrics['n_trades'] < 15 else '✗'} ({best_metrics['n_trades']})")
    print(f"  Win Rate > 60%: {'✓' if best_metrics['win_rate'] > 60 else '✗'} ({best_metrics['win_rate']}%)")
    print(f"  Sharpe > 1.35: {'✓' if best_metrics['sharpe'] > 1.35 else '✗'} ({best_metrics['sharpe']})")

    # Show trade list for best
    if not best_metrics['trades_df'].empty:
        print(f"\nTrade List:")
        print(best_metrics['trades_df'][['entry', 'exit', 'days', 'return']].to_string(index=False))
