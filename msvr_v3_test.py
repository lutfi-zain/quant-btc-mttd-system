#!/usr/bin/env python3
"""Quick test script for MSVR v3 parameter optimization."""

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

# Import from msvr_v3
from msvr_v3 import (load_btc_data, layer1_msvr_base, layer2_supersmoother,
                      layer3_linear_reg, layer4_cycle_phase, layer5_efficiency_ratio,
                      layer6_volatility_gate, layer7_entropy_gate, layer8_volume_confirm,
                      layer9_hmm_regime, backtest)


def test_config(df, config, min_hold=45):
    """Test a specific configuration."""
    smooth_length = config.get('smooth_length', 7)
    cycle_lookback = config.get('cycle_lookback', 40)
    er_threshold = config.get('er_threshold', 0.25)
    entropy_threshold = config.get('entropy_threshold', 2.0)
    vol_threshold = config.get('vol_threshold', 1.5)
    gate_threshold = config.get('gate_threshold', 2)
    trailing_stop_pct = config.get('trailing_stop_pct', 0.15)
    momentum_exit_days = config.get('momentum_exit_days', 10)
    require_lr = config.get('require_lr', False)

    # Compute layers
    msvr_raw = layer1_msvr_base(df)
    msvr_smooth = layer2_supersmoother(msvr_raw, length=smooth_length)
    lr_dir = layer3_linear_reg(df, length=50)
    cycle_dir = layer4_cycle_phase(df, lookback=cycle_lookback)
    er_dir = layer5_efficiency_ratio(df, period=14, threshold=er_threshold)
    vol_dir = layer6_volatility_gate(df, window=20, threshold=vol_threshold)
    entropy_dir = layer7_entropy_gate(df, window=15, threshold=entropy_threshold)
    vol_confirm_dir = layer8_volume_confirm(df, obv_short=10, obv_long=30)
    hmm_dir = layer9_hmm_regime(df, n_states=3, lookback=250)

    # Core signal
    msvr_bullish = (msvr_raw > 0).astype(int)
    smooth_bullish = (msvr_smooth > 0).astype(int)
    core_signal = msvr_bullish * smooth_bullish

    # Timing
    timing_pass = ((cycle_dir == 1) | (er_dir == 1)).astype(int)

    # Gates
    gate_votes = ((vol_dir == 1).astype(int) +
                  (entropy_dir == 1).astype(int) +
                  (vol_confirm_dir == 1).astype(int) +
                  (hmm_dir == 1).astype(int))
    gates_pass = (gate_votes >= gate_threshold).astype(int)

    # Entry
    if require_lr:
        lr_confirm = (lr_dir == 1).astype(int)
        entry_signal = core_signal * timing_pass * gates_pass * lr_confirm
    else:
        entry_signal = core_signal * timing_pass * gates_pass

    # Build position
    pos = pd.Series(0, index=df.index)
    in_position = False
    hold_days = 0
    confirm_count = 0
    entry_price = 0
    peak_price = 0
    momentum_below_zero_days = 0

    prices = df['close']
    for i in range(len(df)):
        entry = entry_signal.iloc[i]
        price = prices.iloc[i]
        smooth_val = msvr_smooth.iloc[i]

        if not in_position:
            if entry == 1:
                confirm_count += 1
                if confirm_count >= 2:
                    in_position = True
                    hold_days = 0
                    entry_price = price
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

    # Test configurations
    configs = [
        # Config name, params, min_hold
        ("Base (2 gates, 15% stop)", {'gate_threshold': 2, 'trailing_stop_pct': 0.15, 'momentum_exit_days': 10}, 45),
        ("Strict (3 gates, 15% stop)", {'gate_threshold': 3, 'trailing_stop_pct': 0.15, 'momentum_exit_days': 10}, 45),
        ("Very Strict (4 gates, 15% stop)", {'gate_threshold': 4, 'trailing_stop_pct': 0.15, 'momentum_exit_days': 10}, 45),
        ("1 gate, 15% stop", {'gate_threshold': 1, 'trailing_stop_pct': 0.15, 'momentum_exit_days': 10}, 45),
        ("2 gates, 10% stop", {'gate_threshold': 2, 'trailing_stop_pct': 0.10, 'momentum_exit_days': 10}, 45),
        ("2 gates, 20% stop", {'gate_threshold': 2, 'trailing_stop_pct': 0.20, 'momentum_exit_days': 10}, 45),
        ("2 gates, 25% stop", {'gate_threshold': 2, 'trailing_stop_pct': 0.25, 'momentum_exit_days': 10}, 45),
        ("2 gates, 30% stop", {'gate_threshold': 2, 'trailing_stop_pct': 0.30, 'momentum_exit_days': 10}, 45),
        ("3 gates, 20% stop", {'gate_threshold': 3, 'trailing_stop_pct': 0.20, 'momentum_exit_days': 10}, 45),
        ("3 gates, 25% stop", {'gate_threshold': 3, 'trailing_stop_pct': 0.25, 'momentum_exit_days': 10}, 45),
        ("2 gates, 15% stop, 5d mom", {'gate_threshold': 2, 'trailing_stop_pct': 0.15, 'momentum_exit_days': 5}, 45),
        ("2 gates, 15% stop, 15d mom", {'gate_threshold': 2, 'trailing_stop_pct': 0.15, 'momentum_exit_days': 15}, 45),
        ("2 gates, 15% stop, 20d mom", {'gate_threshold': 2, 'trailing_stop_pct': 0.15, 'momentum_exit_days': 20}, 45),
        ("3 gates, 20% stop, 15d mom", {'gate_threshold': 3, 'trailing_stop_pct': 0.20, 'momentum_exit_days': 15}, 45),
        ("3 gates, 25% stop, 15d mom", {'gate_threshold': 3, 'trailing_stop_pct': 0.25, 'momentum_exit_days': 15}, 45),
        ("2 gates, 20% stop, 20d mom", {'gate_threshold': 2, 'trailing_stop_pct': 0.20, 'momentum_exit_days': 20}, 45),
        # With LR confirmation
        ("2 gates + LR, 20% stop", {'gate_threshold': 2, 'trailing_stop_pct': 0.20, 'momentum_exit_days': 10, 'require_lr': True}, 45),
        ("3 gates + LR, 20% stop", {'gate_threshold': 3, 'trailing_stop_pct': 0.20, 'momentum_exit_days': 10, 'require_lr': True}, 45),
        # Different entropy thresholds
        ("2 gates, ent=1.8, 20% stop", {'gate_threshold': 2, 'entropy_threshold': 1.8, 'trailing_stop_pct': 0.20, 'momentum_exit_days': 10}, 45),
        ("2 gates, ent=2.2, 20% stop", {'gate_threshold': 2, 'entropy_threshold': 2.2, 'trailing_stop_pct': 0.20, 'momentum_exit_days': 10}, 45),
        ("3 gates, ent=2.2, 25% stop", {'gate_threshold': 3, 'entropy_threshold': 2.2, 'trailing_stop_pct': 0.25, 'momentum_exit_days': 15}, 45),
        # Different ER thresholds
        ("2 gates, er=0.20, 20% stop", {'gate_threshold': 2, 'er_threshold': 0.20, 'trailing_stop_pct': 0.20, 'momentum_exit_days': 10}, 45),
        ("2 gates, er=0.30, 20% stop", {'gate_threshold': 2, 'er_threshold': 0.30, 'trailing_stop_pct': 0.20, 'momentum_exit_days': 10}, 45),
        # Combined best guesses
        ("3 gates, ent=2.2, er=0.20, 25% stop", {'gate_threshold': 3, 'entropy_threshold': 2.2, 'er_threshold': 0.20, 'trailing_stop_pct': 0.25, 'momentum_exit_days': 15}, 45),
        ("3 gates, ent=2.0, er=0.25, 25% stop, 15d", {'gate_threshold': 3, 'entropy_threshold': 2.0, 'er_threshold': 0.25, 'trailing_stop_pct': 0.25, 'momentum_exit_days': 15}, 45),
    ]

    best_sharpe = 0
    best_name = ""
    best_metrics = None

    print(f"{'Config':<45} {'Trades':>6} {'WinRate':>8} {'Sharpe':>7} {'CAGR':>7} {'TimeInMkt':>10}")
    print("-" * 90)

    for name, params, mh in configs:
        metrics = test_config(df, params, min_hold=mh)
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
