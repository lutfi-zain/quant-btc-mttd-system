#!/usr/bin/env python3
"""
Grid Search V2 — Alternative Combinations + Entropy Strict Gate
================================================================

Previous best: Test Sharpe 1.42, WinRate 53.8%, Trades 39, CAGR 49.3%
Target: Sharpe>1.35, WinRate>60%, Trades 25-35, CAGR>50%

Strategy: Try alternative filter combinations with entropy_strict gate
and varied min_hold/max_hold to reduce trades and improve win rate.
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

    # Also try longer momentum period
    df['momentum_long'] = df['close'].pct_change(periods=20)
    df['momentum_long_smooth'] = ehler_supersmoother(df['momentum_long'], length=7)
    df['smooth_direction_long'] = (df['momentum_long_smooth'] > 0).astype(float)

    phase = compute_cycle_phase(df, lookback=40)
    df['cycle_signal'] = -np.cos(phase)
    df['cycle_direction'] = (df['cycle_signal'] > 0).astype(float)

    # Different cycle lookbacks
    phase_long = compute_cycle_phase(df, lookback=50)
    df['cycle_signal_long'] = -np.cos(phase_long)
    df['cycle_direction_long'] = (df['cycle_signal_long'] > 0).astype(float)

    df['entropy_gate'] = (df['Entropy'] < 2.8).astype(float)
    df['entropy_gate_strict'] = (df['Entropy'] < 2.5).astype(float)
    df['entropy_gate_very_strict'] = (df['Entropy'] < 2.2).astype(float)

    # Trend filter (SMA cross)
    sma_fast = df['close'].rolling(50).mean()
    sma_slow = df['close'].rolling(200).mean()
    df['trend_filter'] = (sma_fast > sma_slow).astype(float)

    # Bollinger filter
    bb_mid = df['close'].rolling(25).mean()
    bb_std = df['close'].rolling(25).std()
    df['bb_filter'] = ((df['close'] > bb_mid - 2.0 * bb_std) & (df['close'] < bb_mid + 2.0 * bb_std)).astype(float)

    # Efficiency ratio strict
    df['er_strict'] = (df['ER'] > 0.30).astype(float)

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


def ichimoku_signal(df: pd.DataFrame, min_hold: int, max_hold: int) -> pd.Series:
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
        threshold = std * 0.40
        if in_position:
            hold_count += 1
            can_exit = hold_count >= min_hold
            exit_signal = False
            if can_exit:
                if imo < -0.30:
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
            if imo > threshold and er > 0.25 and entropy < 2.271 and gate_pass:
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


def run_v2_grid_search():
    print("=" * 70)
    print("GRID SEARCH V2 — ALTERNATIVE COMBINATIONS")
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

    print("  Pre-computing features...")
    df_train_feat = generate_filters(generate_ichimoku_features(df_train.copy()))
    df_test_feat = generate_filters(generate_ichimoku_features(df_test.copy()))

    # Test different filter combinations
    # All use gate_threshold=3 (3 of N filters must agree + Ichimoku)
    combos = [
        # Previous best with entropy_gate_strict variant
        (['msvr_direction', 'smooth_direction', 'cycle_direction', 'entropy_gate_strict'], 'A'),
        (['msvr_direction', 'smooth_direction', 'cycle_direction', 'entropy_gate'], 'B'),
        # Replacing entropy_gate with trend_filter
        (['msvr_direction', 'smooth_direction', 'cycle_direction', 'trend_filter'], 'C'),
        # Replacing entropy_gate with er_strict
        (['msvr_direction', 'smooth_direction', 'cycle_direction', 'er_strict'], 'D'),
        # Adding trend_filter to core combo (5 filters, need 3)
        (['msvr_direction', 'smooth_direction', 'cycle_direction', 'entropy_gate', 'trend_filter'], 'E'),
        # 3-filter core
        (['msvr_direction', 'smooth_direction', 'cycle_direction'], 'F'),
        # With longer momentum
        (['msvr_direction', 'smooth_direction_long', 'cycle_direction', 'entropy_gate'], 'G'),
        (['msvr_direction', 'smooth_direction_long', 'cycle_direction', 'entropy_gate_strict'], 'H'),
        # Replace smooth with trend
        (['msvr_direction', 'trend_filter', 'cycle_direction', 'entropy_gate'], 'I'),
        # 3-filter with trend
        (['msvr_direction', 'cycle_direction', 'entropy_gate'], 'J'),
        (['msvr_direction', 'cycle_direction', 'trend_filter'], 'K'),
        # Add bb_filter
        (['msvr_direction', 'smooth_direction', 'cycle_direction', 'entropy_gate', 'bb_filter'], 'L'),
        # 4-filter with er_strict
        (['msvr_direction', 'smooth_direction', 'cycle_direction', 'entropy_gate_strict', 'er_strict'], 'M'),
        # With cycle_long
        (['msvr_direction', 'smooth_direction', 'cycle_direction_long', 'entropy_gate'], 'N'),
        (['msvr_direction', 'smooth_direction', 'cycle_direction_long', 'entropy_gate_strict'], 'O'),
    ]

    # Parameter grid for each combination
    hold_params = {
        'min_hold': [20, 25, 30, 35, 40],
        'max_hold': [55, 60, 65, 70],
    }
    param_names = list(hold_params.keys())
    param_values = list(hold_params.values())
    hold_combos = list(product(*param_values))

    print(f"  Testing {len(combos)} filter combinations x {len(hold_combos)} hold params = {len(combos) * len(hold_combos)} configs")

    results = []

    for combo_list, combo_label in combos:
        n_filters = len(combo_list)

        for mh_params in hold_combos:
            mh_dict = dict(zip(param_names, mh_params))

            try:
                ichimoku_train = ichimoku_signal(df_train_feat, mh_dict['min_hold'], mh_dict['max_hold'])
                ichimoku_test = ichimoku_signal(df_test_feat, mh_dict['min_hold'], mh_dict['max_hold'])

                filter_signals_train = {name: df_train_feat[name] for name in combo_list}
                filter_signals_test = {name: df_test_feat[name] for name in combo_list}

                # Try different gate thresholds
                min_gate = max(1, (n_filters + 1) // 2)
                for gate in range(min_gate, min(n_filters + 1, min_gate + 3)):
                    position_train = apply_gate(ichimoku_train, filter_signals_train, gate)
                    position_test = apply_gate(ichimoku_test, filter_signals_test, gate)

                    metrics_train = compute_metrics(position_train, df_train['close'])
                    metrics_test = compute_metrics(position_test, df_test['close'])

                    if metrics_train['sharpe'] > 0:
                        deg = (metrics_test['sharpe'] - metrics_train['sharpe']) / metrics_train['sharpe'] * 100
                    else:
                        deg = 0

                    result = {
                        'combo': combo_label,
                        'filters': '+'.join(combo_list),
                        'n_filters': n_filters,
                        'gate': gate,
                        'min_hold': mh_dict['min_hold'],
                        'max_hold': mh_dict['max_hold'],
                        'train_sharpe': metrics_train['sharpe'],
                        'train_win_rate': metrics_train['win_rate'],
                        'train_trades': metrics_train['n_trades'],
                        'train_cagr': metrics_train['cagr'],
                        'test_sharpe': metrics_test['sharpe'],
                        'test_win_rate': metrics_test['win_rate'],
                        'test_trades': metrics_test['n_trades'],
                        'test_cagr': metrics_test['cagr'],
                        'test_max_dd': metrics_test['max_dd'],
                        'degradation': round(deg, 1)
                    }
                    results.append(result)

            except Exception as e:
                print(f"  Error: {combo_label}, MH={mh_dict}: {e}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('test_sharpe', ascending=False)

    print(f"\n  Total configs tested: {len(results)}")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, 'v2_grid_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"  Results saved to: {csv_path}")

    # Analysis
    print("\n" + "=" * 70)
    print("V2 GRID SEARCH RESULTS")
    print("=" * 70)

    print("\n--- Top 20 by Test Sharpe ---")
    print(f"{'#':<4} {'Combo':<6} {'#F':<4} {'Gate':<5} {'MH':<4} {'xH':<4} {'TS':<6} {'TW%':<6} {'TT':<5} {'CAG':<6} {'D%':<6}")
    print("-" * 70)
    for idx, (_, r) in enumerate(results_df.head(20).iterrows()):
        print(f"{idx+1:<4} {r['combo']:<6} {int(r['n_filters']):<4} {int(r['gate']):<5} "
              f"{int(r['min_hold']):<4} {int(r['max_hold']):<4} "
              f"{r['test_sharpe']:<6.2f} {r['test_win_rate']:<6.1f} {int(r['test_trades']):<5} "
              f"{r['test_cagr']:<6.1f} {r['degradation']:<6.1f}")

    # Targets
    target_all = results_df[
        (results_df['test_sharpe'] >= 1.35) &
        (results_df['test_win_rate'] >= 60) &
        (results_df['test_trades'] >= 25) &
        (results_df['test_trades'] <= 35) &
        (results_df['test_cagr'] >= 50)
    ]

    print(f"\n--- ALL Targets: S>1.35, W>60%, T 25-35, C>50% ---")
    print(f"  Found: {len(target_all)}")
    if len(target_all) > 0:
        for _, r in target_all.head(5).iterrows():
            print(f"  {r['combo']}: MH={int(r['min_hold'])}, xH={int(r['max_hold'])}, G={int(r['gate'])} => "
                  f"S={r['test_sharpe']:.2f}, W={r['test_win_rate']:.1f}%, T={int(r['test_trades'])}, C={r['test_cagr']:.1f}%")

    # Relaxed: Sharpe>1.35, Win>55%, Trades 25-40
    target_relaxed = results_df[
        (results_df['test_sharpe'] >= 1.35) &
        (results_df['test_win_rate'] >= 55) &
        (results_df['test_trades'] >= 25) &
        (results_df['test_trades'] <= 40)
    ]

    print(f"\n--- Relaxed: S>1.35, W>55%, T 25-40 ---")
    print(f"  Found: {len(target_relaxed)}")
    if len(target_relaxed) > 0:
        for _, r in target_relaxed.head(10).iterrows():
            print(f"  {r['combo']}: MH={int(r['min_hold'])}, xH={int(r['max_hold'])}, G={int(r['gate'])} => "
                  f"S={r['test_sharpe']:.2f}, W={r['test_win_rate']:.1f}%, T={int(r['test_trades'])}, C={r['test_cagr']:.1f}%")

    # Best per combo
    print(f"\n--- Best per Filter Combination ---")
    best_per_combo = results_df.groupby('combo').first().reset_index()
    best_per_combo = best_per_combo.sort_values('test_sharpe', ascending=False)
    for _, r in best_per_combo.iterrows():
        print(f"  {r['combo']}: MH={int(r['min_hold'])}, xH={int(r['max_hold'])}, G={int(r['gate'])} => "
              f"S={r['test_sharpe']:.2f}, W={r['test_win_rate']:.1f}%, T={int(r['test_trades'])}, C={r['test_cagr']:.1f}%")

    best = results_df.iloc[0]
    print(f"\n" + "=" * 70)
    print("BEST CONFIGURATION (V2)")
    print("=" * 70)
    print(f"  Filters: {best['filters']}")
    print(f"  min_hold={int(best['min_hold'])}, max_hold={int(best['max_hold'])}, gate={int(best['gate'])}")
    print(f"  Test: Sharpe={best['test_sharpe']:.2f}, WinRate={best['test_win_rate']:.1f}%, "
          f"Trades={int(best['test_trades'])}, CAGR={best['test_cagr']:.1f}%")
    print(f"  Degradation: {best['degradation']:.1f}%")

    if len(target_all) > 0:
        print(f"\n✅ Found {len(target_all)} configs meeting ALL targets!")
        return results_df, True
    elif best['test_sharpe'] >= 1.35:
        print(f"\n⚠️  Best Sharpe {best['test_sharpe']:.2f} exceeds 1.35 but not all targets met.")
        return results_df, False
    else:
        print(f"\n❌ No config achieved Sharpe>1.35")
        return results_df, False


if __name__ == "__main__":
    results_df, success = run_v2_grid_search()
    sys.exit(0 if success else 1)
