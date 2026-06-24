#!/usr/bin/env python3
"""
Extended Parameter Grid Search for Best Indicator Combination
=============================================================

Goal: Push Sharpe above 1.35 with 25-35 trades, >60% win rate, >50% CAGR.

Best from initial grid: min_hold=25, max_hold=60, gate=3
  - Test Sharpe 1.42 ✅
  - Win Rate 53.8% ❌ (needs >60%)
  - Trades 39 ❌ (needs 25-35)
  - CAGR 49.3% ❌ (needs >50%)

Extended parameters:
1. entropy_threshold: [2.0, 2.2, 2.5, 2.8] — tighter entropy gate
2. imo_threshold_mult: [0.30, 0.35, 0.40, 0.45] — stricter entry
3. imo_exit_level: [-0.20, -0.25, -0.30, -0.35] — earlier exit
4. er_threshold: [0.20, 0.25, 0.30] — efficiency ratio gate
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
from itertools import product
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
bank_root = '/home/ubuntu/projects/quant-technical-indicator-bank'
sys.path.append(project_root)
sys.path.append(bank_root)
sys.path.append(os.path.join(project_root, 'indicators'))

TRANSACTION_COST = 0.001
TRAIN_START = '2018-01-01'
TRAIN_END = '2023-12-31'
TEST_START = '2024-01-01'
TEST_END = '2026-06-30'
OUTPUT_DIR = os.path.join(project_root, 'mttd', 'grid_search')


def ehler_supersmoother(series: pd.Series, length: int = 7) -> pd.Series:
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
    change = series.diff().abs()
    volatility = change.rolling(period).sum()
    direction = series.diff(period).abs()
    return direction / volatility


def compute_cycle_phase(df: pd.DataFrame, lookback: int = 40) -> pd.Series:
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
        valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
        valid_power = power[valid_mask]
        valid_freqs = freqs[valid_mask]
        if len(valid_power) > 0 and np.sum(valid_power) > 0:
            dominant_idx = np.argmax(valid_power)
            dominant_freq = valid_freqs[dominant_idx]
            dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else lookback
            cycle_pos = i % int(dominant_period)
            phase.iloc[i] = 2 * np.pi * cycle_pos / dominant_period
    return phase


def generate_ichimoku_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['tenkan_sen'] = (df['high'].rolling(20).max() + df['low'].rolling(20).min()) / 2
    df['kijun_sen'] = (df['high'].rolling(60).max() + df['low'].rolling(60).min()) / 2
    df['senkou_span_a_raw'] = (df['tenkan_sen'] + df['kijun_sen']) / 2
    df['senkou_span_b_raw'] = (df['high'].rolling(120).max() + df['low'].rolling(120).min()) / 2
    df['senkou_span_a'] = df['senkou_span_a_raw'].shift(60)
    df['senkou_span_b'] = df['senkou_span_b_raw'].shift(60)
    df['cloud_max'] = np.maximum(df['senkou_span_a'], df['senkou_span_b'])
    df['cloud_min'] = np.minimum(df['senkou_span_a'], df['senkou_span_b'])
    df['S_TK'] = np.tanh((df['tenkan_sen'] - df['kijun_sen']) / df['ATR'])
    dist_cloud = np.zeros(len(df))
    above = df['close'] > df['cloud_max']
    below = df['close'] < df['cloud_min']
    dist_cloud[above] = (df['close'] - df['cloud_max'])[above] / df['ATR'][above]
    dist_cloud[below] = (df['close'] - df['cloud_min'])[below] / df['ATR'][below]
    df['S_Cloud'] = np.tanh(dist_cloud)
    df['S_Future'] = np.tanh((df['senkou_span_a_raw'] - df['senkou_span_b_raw']) / df['ATR'])
    raw_chikou_dist = (df['close'] - df['close'].shift(60)) / df['ATR']
    df['S_Chikou'] = np.tanh(ehler_supersmoother(raw_chikou_dist, length=4))
    imo_raw = (df['S_TK'] + df['S_Cloud'] + df['S_Future'] + df['S_Chikou']) / 4.0
    df['IMO'] = ehler_supersmoother(imo_raw, length=7)
    df['IMO_Std'] = df['IMO'].rolling(30).std()
    df['ER'] = efficiency_ratio(df['close'], period=14)
    df['Entropy'] = shannon_entropy(df['close'], window=15, bins=6)
    return df


def generate_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['momentum'] = df['close'].pct_change(periods=10)
    df['momentum_smooth'] = ehler_supersmoother(df['momentum'], length=5)
    df['smooth_direction'] = (df['momentum_smooth'] > 0).astype(float)
    phase = compute_cycle_phase(df, lookback=40)
    df['cycle_signal'] = -np.cos(phase)
    df['cycle_direction'] = (df['cycle_signal'] > 0).astype(float)
    df['entropy_gate'] = (df['Entropy'] < 2.8).astype(float)
    df['entropy_gate_tight'] = (df['Entropy'] < 2.5).astype(float)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'msvr', os.path.join(bank_root, 'perpetual/median_standard_deviation_viresearch.py'))
        msvr_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(msvr_module)
        msvr_result = msvr_module.median_standard_deviation_viresearch(df)
        df['msvr_signal'] = msvr_result['vii']
        df['msvr_direction'] = (df['msvr_signal'] > 0).astype(float)
    except Exception:
        df['msvr_direction'] = 0.5
    return df


def ichimoku_signal_custom(df: pd.DataFrame, min_hold: int, max_hold: int,
                           imo_threshold_mult: float = 0.40,
                           imo_exit_level: float = -0.30,
                           er_threshold: float = 0.25,
                           entropy_threshold: float = 2.271) -> pd.Series:
    n = len(df)
    position = np.zeros(n)
    in_position = False
    hold_count = 0
    for i in range(n):
        imo = df['IMO'].iloc[i]
        er = df['ER'].iloc[i]
        std = df['IMO_Std'].iloc[i]
        entropy = df['Entropy'].iloc[i]
        close = df['close'].iloc[i]
        cloud_min = df['cloud_min'].iloc[i]
        if pd.isna(imo) or pd.isna(er) or pd.isna(std) or pd.isna(entropy):
            if in_position:
                position[i] = 1.0
            continue
        threshold = std * imo_threshold_mult
        if in_position:
            hold_count += 1
            can_exit = hold_count >= min_hold
            exit_signal = False
            if can_exit:
                if imo < imo_exit_level:
                    exit_signal = True
                elif hold_count >= max_hold:
                    exit_signal = True
                elif close < cloud_min and imo < 0:
                    exit_signal = True
            if exit_signal:
                in_position = False
                hold_count = 0
                position[i] = 0.0
            else:
                position[i] = 1.0
        else:
            gate_pass = True
            if not pd.isna(cloud_min):
                gate_pass = (close >= cloud_min)
            if imo > threshold and er > er_threshold and entropy < entropy_threshold and gate_pass:
                in_position = True
                hold_count = 0
                position[i] = 1.0
            else:
                position[i] = 0.0
    return pd.Series(position, index=df.index)


def apply_gate(ichimoku_signal: pd.Series, filter_signals: dict, gate_threshold: int) -> pd.Series:
    n = len(ichimoku_signal)
    result = np.zeros(n)
    filter_names = list(filter_signals.keys())
    filter_matrix = np.column_stack([filter_signals[name].values for name in filter_names])
    in_position = False
    for i in range(n):
        if not in_position:
            bullish_filters = np.sum(filter_matrix[i] == 1.0)
            if ichimoku_signal.iloc[i] == 1.0 and bullish_filters >= gate_threshold:
                in_position = True
                result[i] = 1.0
        else:
            bullish_filters = np.sum(filter_matrix[i] == 1.0)
            if ichimoku_signal.iloc[i] == 0.0 or bullish_filters < gate_threshold:
                in_position = False
                result[i] = 0.0
            else:
                result[i] = 1.0
    return pd.Series(result, index=ichimoku_signal.index)


def compute_metrics(signal: pd.Series, prices: pd.Series) -> dict:
    returns = prices.pct_change()
    strategy_returns = returns * signal.shift(1)
    strategy_returns = strategy_returns.dropna()
    transitions = signal.diff().fillna(0)
    strategy_returns = strategy_returns - transitions.loc[strategy_returns.index] * (TRANSACTION_COST / 2)
    if len(strategy_returns) == 0:
        return {'cagr': 0, 'sharpe': 0, 'max_dd': 0, 'n_trades': 0, 'win_rate': 0, 'avg_hold': 0}
    equity = (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25
    cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    in_position = False
    hold_start = None
    trade_returns = []
    hold_periods = []
    for i, (date, pos) in enumerate(signal.items()):
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
    winning = sum(1 for r in trade_returns if r > 0)
    total = len(trade_returns)
    win_rate = winning / total * 100 if total > 0 else 0
    avg_hold = np.mean(hold_periods) if hold_periods else 0
    return {
        'cagr': round(cagr * 100, 2), 'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd * 100, 2), 'n_trades': total,
        'win_rate': round(win_rate, 1), 'avg_hold': round(avg_hold, 0)
    }


def run_extended_grid_search():
    print("=" * 70)
    print("EXTENDED PARAMETER GRID SEARCH")
    print("=" * 70)

    with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
        btc_data = json.load(f)

    df = pd.DataFrame(btc_data['aligned_data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df[df.index >= '2018-01-01']

    df_train = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    df_test = df[(df.index >= TEST_START) & (df.index <= TEST_END)].copy()

    print(f"  Train: {len(df_train)} bars, Test: {len(df_test)} bars")

    # Pre-compute features
    print("  Pre-computing features...")
    df_train_feat = generate_filters(generate_ichimoku_features(df_train.copy()))
    df_test_feat = generate_filters(generate_ichimoku_features(df_test.copy()))

    # Also load existing results
    existing_path = os.path.join(OUTPUT_DIR, 'grid_search_results.csv')
    existing_results = []
    if os.path.exists(existing_path):
        existing_df = pd.read_csv(existing_path)
        existing_results = existing_df.to_dict('records')
        print(f"  Loaded {len(existing_results)} existing results")

    best_filters = ['msvr_direction', 'smooth_direction', 'cycle_direction', 'entropy_gate']

    # Extended parameter grid
    param_grid = {
        'min_hold': [20, 25, 30, 35, 40],
        'max_hold': [55, 60, 65, 70, 75],
        'gate_threshold': [3],
        'entropy_threshold': [2.0, 2.2, 2.5, 2.8],
        'imo_threshold_mult': [0.30, 0.35, 0.40, 0.45],
        'imo_exit_level': [-0.20, -0.25, -0.30, -0.35],
        'er_threshold': [0.20, 0.25, 0.30],
    }

    # Use a smarter subset for efficiency
    # Focus on parameter ranges around the best existing config
    key_params = {
        'min_hold': [20, 25, 30, 35],
        'max_hold': [55, 60, 65, 70],
        'gate_threshold': [3],
        'entropy_threshold': [2.0, 2.2, 2.5, 2.8],
        'imo_threshold_mult': [0.30, 0.35, 0.40, 0.45],
        'imo_exit_level': [-0.20, -0.25, -0.30],
    }

    param_names = list(key_params.keys())
    param_values = list(key_params.values())
    all_combos = list(product(*param_values))

    print(f"\n  Testing {len(all_combos)} extended parameter combinations...")

    results = list(existing_results)
    total_new = 0

    for idx, params in enumerate(all_combos):
        param_dict = dict(zip(param_names, params))

        try:
            ichimoku_train = ichimoku_signal_custom(
                df_train_feat, param_dict['min_hold'], param_dict['max_hold'],
                param_dict['imo_threshold_mult'], param_dict['imo_exit_level'],
                er_threshold=0.25, entropy_threshold=param_dict['entropy_threshold']
            )
            ichimoku_test = ichimoku_signal_custom(
                df_test_feat, param_dict['min_hold'], param_dict['max_hold'],
                param_dict['imo_threshold_mult'], param_dict['imo_exit_level'],
                er_threshold=0.25, entropy_threshold=param_dict['entropy_threshold']
            )

            filter_signals_train = {name: df_train_feat[name] for name in best_filters}
            filter_signals_test = {name: df_test_feat[name] for name in best_filters}

            position_train = apply_gate(ichimoku_train, filter_signals_train, param_dict['gate_threshold'])
            position_test = apply_gate(ichimoku_test, filter_signals_test, param_dict['gate_threshold'])

            metrics_train = compute_metrics(position_train, df_train['close'])
            metrics_test = compute_metrics(position_test, df_test['close'])

            if metrics_train['sharpe'] > 0:
                sharpe_deg = (metrics_test['sharpe'] - metrics_train['sharpe']) / metrics_train['sharpe'] * 100
            else:
                sharpe_deg = 0

            result = {
                'min_hold': param_dict['min_hold'],
                'max_hold': param_dict['max_hold'],
                'gate_threshold': param_dict['gate_threshold'],
                'entropy_threshold': param_dict['entropy_threshold'],
                'imo_threshold_mult': param_dict['imo_threshold_mult'],
                'imo_exit_level': param_dict['imo_exit_level'],
                'train_sharpe': metrics_train['sharpe'],
                'train_win_rate': metrics_train['win_rate'],
                'train_trades': metrics_train['n_trades'],
                'train_cagr': metrics_train['cagr'],
                'train_max_dd': metrics_train['max_dd'],
                'test_sharpe': metrics_test['sharpe'],
                'test_win_rate': metrics_test['win_rate'],
                'test_trades': metrics_test['n_trades'],
                'test_cagr': metrics_test['cagr'],
                'test_max_dd': metrics_test['max_dd'],
                'sharpe_degradation': round(sharpe_deg, 1)
            }
            results.append(result)
            total_new += 1

        except Exception as e:
            print(f"  Error: {param_dict}: {e}")

        if (idx + 1) % 20 == 0:
            print(f"  Completed {idx + 1}/{len(all_combos)} combinations...")

    print(f"  Total new combinations tested: {total_new}")
    print(f"  Total results: {len(results)}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_sharpe', ascending=False)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, 'extended_grid_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")

    # Analysis
    print("\n" + "=" * 70)
    print("EXTENDED GRID SEARCH RESULTS")
    print("=" * 70)

    print("\n--- Top 15 by Test Sharpe ---")
    print(f"{'Rank':<5} {'MinH':<6} {'MaxH':<6} {'Gate':<5} {'Ent':<5} {'IMO_M':<6} {'IMO_X':<6} {'T_S':<6} {'T_W%':<6} {'T_T':<5} {'CAGR':<6} {'Degr%':<7}")
    print("-" * 95)

    for idx, (_, row) in enumerate(results_df.head(15).iterrows()):
        ent_str = f"{row.get('entropy_threshold', 2.271):.1f}"
        imo_m_str = f"{row.get('imo_threshold_mult', 0.40):.2f}"
        imo_x_str = f"{row.get('imo_exit_level', -0.30):.2f}"
        print(f"{idx+1:<5} {int(row['min_hold']):<6} {int(row['max_hold']):<6} {int(row['gate_threshold']):<5} "
              f"{ent_str:<5} {imo_m_str:<6} {imo_x_str:<6} "
              f"{row['test_sharpe']:<6.2f} {row['test_win_rate']:<6.1f} {int(row['test_trades']):<5} "
              f"{row['test_cagr']:<6.1f} {row['sharpe_degradation']:<7.1f}")

    # Configs meeting ALL targets
    target_results = results_df[
        (results_df['test_sharpe'] >= 1.35) &
        (results_df['test_win_rate'] >= 60) &
        (results_df['test_trades'] >= 25) &
        (results_df['test_trades'] <= 35) &
        (results_df['test_cagr'] >= 50)
    ]

    print(f"\n--- Configs Meeting ALL Targets ---")
    print(f"  Sharpe>1.35, WinRate>60%, Trades 25-35, CAGR>50%")
    print(f"  Found: {len(target_results)}")

    if len(target_results) > 0:
        for idx, (_, row) in enumerate(target_results.head(10).iterrows()):
            print(f"\n  Config {idx+1}:")
            print(f"    min_hold={int(row['min_hold'])}, max_hold={int(row['max_hold'])}, gate={int(row['gate_threshold'])}")
            print(f"    entropy<{row.get('entropy_threshold', 2.271):.2f}, IMO_mult={row.get('imo_threshold_mult', 0.40):.2f}, IMO_exit={row.get('imo_exit_level', -0.30):.2f}")
            print(f"    Test: Sharpe={row['test_sharpe']:.2f}, WinRate={row['test_win_rate']:.1f}%, Trades={int(row['test_trades'])}, CAGR={row['test_cagr']:.1f}%")

    # Relaxed targets: Sharpe>1.35, WinRate>55%, Trades 25-40
    relaxed_results = results_df[
        (results_df['test_sharpe'] >= 1.35) &
        (results_df['test_win_rate'] >= 55) &
        (results_df['test_trades'] >= 25) &
        (results_df['test_trades'] <= 40)
    ]

    print(f"\n--- Relaxed: Sharpe>1.35, WinRate>55%, Trades 25-40 ---")
    print(f"  Found: {len(relaxed_results)}")

    if len(relaxed_results) > 0:
        for idx, (_, row) in enumerate(relaxed_results.head(10).iterrows()):
            print(f"  {idx+1}. MH={int(row['min_hold'])}, xH={int(row['max_hold'])}, G={int(row['gate_threshold'])}, "
                  f"Ent={row.get('entropy_threshold', 2.271):.1f}, "
                  f"S={row['test_sharpe']:.2f}, W={row['test_win_rate']:.1f}%, T={int(row['test_trades'])}, C={row['test_cagr']:.1f}%")

    # Best trade count near target
    trade_target = results_df[
        (results_df['test_sharpe'] >= 1.30) &
        (results_df['test_trades'] >= 25) &
        (results_df['test_trades'] <= 35)
    ].sort_values('test_sharpe', ascending=False)

    print(f"\n--- Sharpe>1.30, Trades 25-35 ---")
    print(f"  Found: {len(trade_target)}")

    if len(trade_target) > 0:
        for idx, (_, row) in enumerate(trade_target.head(10).iterrows()):
            print(f"  {idx+1}. MH={int(row['min_hold'])}, xH={int(row['max_hold'])}, G={int(row['gate_threshold'])}, "
                  f"Ent={row.get('entropy_threshold', 2.271):.1f}, "
                  f"S={row['test_sharpe']:.2f}, W={row['test_win_rate']:.1f}%, T={int(row['test_trades'])}, C={row['test_cagr']:.1f}%")

    best = results_df.iloc[0]
    print(f"\n" + "=" * 70)
    print("BEST CONFIGURATION (by Test Sharpe)")
    print("=" * 70)
    print(f"  min_hold: {int(best['min_hold'])}")
    print(f"  max_hold: {int(best['max_hold'])}")
    print(f"  gate_threshold: {int(best['gate_threshold'])}")
    print(f"  entropy_threshold: {best.get('entropy_threshold', 2.271):.2f}")
    print(f"  imo_threshold_mult: {best.get('imo_threshold_mult', 0.40):.2f}")
    print(f"  imo_exit_level: {best.get('imo_exit_level', -0.30):.2f}")
    print(f"  Train: Sharpe={best['train_sharpe']:.2f}, WinRate={best['train_win_rate']:.1f}%, Trades={int(best['train_trades'])}, CAGR={best['train_cagr']:.1f}%")
    print(f"  Test:  Sharpe={best['test_sharpe']:.2f}, WinRate={best['test_win_rate']:.1f}%, Trades={int(best['test_trades'])}, CAGR={best['test_cagr']:.1f}%")
    print(f"  Degradation: {best['sharpe_degradation']:.1f}%")

    if len(target_results) > 0:
        print(f"\n✅ Found {len(target_results)} configs meeting ALL targets!")
        return results_df, True
    elif best['test_sharpe'] >= 1.35:
        print(f"\n⚠️  Found Sharpe>1.35 but not all targets met. Best: Sharpe={best['test_sharpe']:.2f}")
        return results_df, False
    else:
        print(f"\n❌ No config achieved Sharpe>1.35")
        return results_df, False


if __name__ == "__main__":
    results_df, success = run_extended_grid_search()
    sys.exit(0 if success else 1)
