#!/usr/bin/env python3
"""MSVR v3 Combined Test - MSVR Hybrid entry + trailing stop exit."""

import numpy as np
import pandas as pd
import sys
import os
import json
import importlib.util
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/home/ubuntu/projects/quant-technical-indicator-bank')
sys.path.append('/home/ubuntu/projects/quant-btc-mttd-system')


def load_btc_data():
    data_path = '/home/ubuntu/projects/quant-btc-mttd-system/data/btc_daily.json'
    with open(data_path) as f:
        btc_data = json.load(f)
    df = pd.DataFrame(btc_data['aligned_data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df[df.index >= '2018-01-01']
    return df


def load_msvr_signal(df):
    spec = importlib.util.spec_from_file_location('msvr',
        '/home/ubuntu/projects/quant-technical-indicator-bank/perpetual/median_standard_deviation_viresearch.py')
    msvr_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(msvr_module)
    result = msvr_module.median_standard_deviation_viresearch(df)
    return result['vii']


def ehler_supersmoother(series, length=7):
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


def shannon_entropy_gate(series, window=15, bins=6, threshold=2.5):
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
    gate = gate.fillna(0)
    return gate


def efficiency_ratio_gate(series, period=14, threshold=0.25):
    change = series.diff().abs()
    volatility = change.rolling(period).sum()
    direction = series.diff(period).abs()
    er = direction / volatility
    gate = (er > threshold).astype(int)
    gate = gate.fillna(0)
    return gate


def backtest(df, position, transaction_cost=0.001):
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
            trades.append({'entry': entry_date, 'exit': date, 'days': (date - entry_date).days, 'return': ret})
    if in_pos:
        exit_price = prices.iloc[-1]
        ret = (exit_price - entry_price) / entry_price - transaction_cost
        trades.append({'entry': entry_date, 'exit': prices.index[-1], 'days': (prices.index[-1] - entry_date).days, 'return': ret})

    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        wins = (trades_df['return'] > 0).sum()
        win_rate = wins / len(trades_df) * 100
        avg_hold = trades_df['days'].mean()
    else:
        win_rate = 0
        avg_hold = 0

    return {
        'n_trades': len(trades_df),
        'win_rate': round(win_rate, 1),
        'sharpe': round(sharpe, 2),
        'cagr': round(cagr * 100, 2),
        'max_dd': round(max_dd * 100, 2),
        'avg_hold': round(avg_hold, 0),
        'time_in_market': round(position.mean() * 100, 1),
        'total_return': round((equity.iloc[-1] - 1) * 100, 2),
        'trades_df': trades_df,
        'equity': equity,
        'position': position
    }


def test_combined(df, config):
    """Test combined approach."""
    msvr_raw = load_msvr_signal(df)
    msvr_smooth = ehler_supersmoother(msvr_raw, length=config.get('smooth_length', 7))
    cycle_dir = compute_cycle_phase(df, lookback=config.get('cycle_lookback', 40))
    er_gate = efficiency_ratio_gate(df['close'], period=14, threshold=config.get('er_threshold', 0.30))
    entropy_gate = shannon_entropy_gate(df['close'], window=15, threshold=config.get('entropy_threshold', 1.9))

    # Entry signal (MSVR Hybrid)
    msvr_bullish = (msvr_raw > 0).astype(int)
    smooth_bullish = (msvr_smooth > 0).astype(int)
    core_signal = msvr_bullish * smooth_bullish
    timing_pass = ((cycle_dir == 1) | (er_gate == 1)).astype(int)
    entry_signal = core_signal * timing_pass * entropy_gate

    # Build position with trailing stop
    min_hold = config.get('min_hold', 45)
    trailing_stop = config.get('trailing_stop', 0.15)
    msvr_exit_threshold = config.get('msvr_exit_threshold', -0.3)
    confirm_entry = config.get('confirm_entry', 2)

    pos = pd.Series(0, index=df.index)
    in_position = False
    hold_days = 0
    confirm_count = 0
    entry_price = 0
    peak_price = 0
    prices = df['close']

    for i in range(len(df)):
        entry = entry_signal.iloc[i]
        price = prices.iloc[i]
        smooth_val = msvr_smooth.iloc[i]

        if not in_position:
            if entry == 1:
                confirm_count += 1
                if confirm_count >= confirm_entry:
                    in_position = True
                    hold_days = 0
                    entry_price = price
                    peak_price = price
                    pos.iloc[i] = 1
            else:
                confirm_count = 0
        else:
            hold_days += 1
            pos.iloc[i] = 1
            if price > peak_price:
                peak_price = price

            # Exit conditions (only after min_hold)
            if hold_days >= min_hold:
                # Condition 1: Trailing stop
                price_drop = (peak_price - price) / peak_price
                if price_drop >= trailing_stop:
                    in_position = False
                    hold_days = 0
                    pos.iloc[i] = 0
                    continue

                # Condition 2: MSVR momentum exit
                if smooth_val < msvr_exit_threshold:
                    in_position = False
                    hold_days = 0
                    pos.iloc[i] = 0
                    continue

    return backtest(df, pos)


if __name__ == "__main__":
    print("Loading data...")
    df = load_btc_data()
    print(f"Data: {len(df)} bars\n")

    configs = [
        # Trailing stop with min_hold=45
        {"name": "mh=45, ts=10%, exit=-0.3", "min_hold": 45, "trailing_stop": 0.10, "msvr_exit_threshold": -0.3},
        {"name": "mh=45, ts=15%, exit=-0.3", "min_hold": 45, "trailing_stop": 0.15, "msvr_exit_threshold": -0.3},
        {"name": "mh=45, ts=20%, exit=-0.3", "min_hold": 45, "trailing_stop": 0.20, "msvr_exit_threshold": -0.3},
        {"name": "mh=45, ts=25%, exit=-0.3", "min_hold": 45, "trailing_stop": 0.25, "msvr_exit_threshold": -0.3},
        {"name": "mh=45, ts=30%, exit=-0.3", "min_hold": 45, "trailing_stop": 0.30, "msvr_exit_threshold": -0.3},
        # Different exit thresholds
        {"name": "mh=45, ts=15%, exit=-0.2", "min_hold": 45, "trailing_stop": 0.15, "msvr_exit_threshold": -0.2},
        {"name": "mh=45, ts=20%, exit=-0.2", "min_hold": 45, "trailing_stop": 0.20, "msvr_exit_threshold": -0.2},
        {"name": "mh=45, ts=25%, exit=-0.2", "min_hold": 45, "trailing_stop": 0.25, "msvr_exit_threshold": -0.2},
        {"name": "mh=45, ts=15%, exit=-0.4", "min_hold": 45, "trailing_stop": 0.15, "msvr_exit_threshold": -0.4},
        {"name": "mh=45, ts=20%, exit=-0.4", "min_hold": 45, "trailing_stop": 0.20, "msvr_exit_threshold": -0.4},
        {"name": "mh=45, ts=25%, exit=-0.4", "min_hold": 45, "trailing_stop": 0.25, "msvr_exit_threshold": -0.4},
        # Different entropy thresholds
        {"name": "mh=45, ts=20%, exit=-0.3, ent=2.2", "min_hold": 45, "trailing_stop": 0.20, "msvr_exit_threshold": -0.3, "entropy_threshold": 2.2},
        {"name": "mh=45, ts=25%, exit=-0.3, ent=2.2", "min_hold": 45, "trailing_stop": 0.25, "msvr_exit_threshold": -0.3, "entropy_threshold": 2.2},
        # Different ER thresholds
        {"name": "mh=45, ts=20%, exit=-0.3, er=0.35", "min_hold": 45, "trailing_stop": 0.20, "msvr_exit_threshold": -0.3, "er_threshold": 0.35},
        {"name": "mh=45, ts=25%, exit=-0.3, er=0.35", "min_hold": 45, "trailing_stop": 0.25, "msvr_exit_threshold": -0.3, "er_threshold": 0.35},
        # Trailing stop ONLY (no momentum exit)
        {"name": "mh=45, ts=15%, NO exit", "min_hold": 45, "trailing_stop": 0.15, "msvr_exit_threshold": 999},
        {"name": "mh=45, ts=20%, NO exit", "min_hold": 45, "trailing_stop": 0.20, "msvr_exit_threshold": 999},
        {"name": "mh=45, ts=25%, NO exit", "min_hold": 45, "trailing_stop": 0.25, "msvr_exit_threshold": 999},
        {"name": "mh=45, ts=30%, NO exit", "min_hold": 45, "trailing_stop": 0.30, "msvr_exit_threshold": 999},
        # Combined best guesses
        {"name": "mh=45, ts=20%, exit=-0.3, ent=2.2, er=0.35", "min_hold": 45, "trailing_stop": 0.20, "msvr_exit_threshold": -0.3, "entropy_threshold": 2.2, "er_threshold": 0.35},
        {"name": "mh=45, ts=25%, exit=-0.3, ent=2.2, er=0.35", "min_hold": 45, "trailing_stop": 0.25, "msvr_exit_threshold": -0.3, "entropy_threshold": 2.2, "er_threshold": 0.35},
    ]

    best_sharpe = 0
    best_name = ""
    best_metrics = None

    print(f"{'Config':<55} {'Trades':>6} {'WinRate':>8} {'Sharpe':>7} {'CAGR':>7} {'TimeInMkt':>10}")
    print("-" * 100)

    for config in configs:
        name = config.pop('name')
        metrics = test_combined(df, config)
        print(f"{name:<55} {metrics['n_trades']:>6} {metrics['win_rate']:>7.1f}% {metrics['sharpe']:>7.2f} {metrics['cagr']:>6.1f}% {metrics['time_in_market']:>9.1f}%")
        config['name'] = name

        if metrics['sharpe'] > best_sharpe:
            best_sharpe = metrics['sharpe']
            best_name = name
            best_metrics = metrics

    print(f"\n{'=' * 100}")
    print(f"BEST: {best_name}")
    print(f"  Trades: {best_metrics['n_trades']}, Win Rate: {best_metrics['win_rate']}%, Sharpe: {best_metrics['sharpe']}")
    print(f"  CAGR: {best_metrics['cagr']}%, Max DD: {best_metrics['max_dd']}%, Time In Market: {best_metrics['time_in_market']}%")
    print(f"\nConstraint Check:")
    print(f"  Trades < 15: {'✓' if best_metrics['n_trades'] < 15 else '✗'} ({best_metrics['n_trades']})")
    print(f"  Win Rate > 60%: {'✓' if best_metrics['win_rate'] > 60 else '✗'} ({best_metrics['win_rate']}%)")
    print(f"  Sharpe > 1.35: {'✓' if best_metrics['sharpe'] > 1.35 else '✗'} ({best_metrics['sharpe']})")

    if not best_metrics['trades_df'].empty:
        print(f"\nTrade List:")
        print(best_metrics['trades_df'][['entry', 'exit', 'days', 'return']].to_string(index=False))
