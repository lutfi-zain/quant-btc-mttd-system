#!/usr/bin/env python3
"""
Final Grid Search V3 — Ultra Fast
==================================

Uses simplified cycle phase (zero-crossing) instead of FFT for speed.
Tests profit-taking + trailing stop combinations.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

project_root = '/home/ubuntu/projects/quant-btc-mttd-system'
bank_root = '/home/ubuntu/projects/quant-technical-indicator-bank'
sys.path.append(project_root)
sys.path.append(bank_root)

TRANSACTION_COST = 0.001
OUTPUT_DIR = os.path.join(project_root, 'mttd', 'grid_search')


def ehler_supersmoother(series, length=7):
    a1 = np.exp(-1.414 * np.pi / length)
    b1 = 2 * a1 * np.cos(np.radians(1.414 * 180.0 / length))
    c2, c3 = b1, -a1 * a1
    c1 = 1 - c2 - c3
    vals = series.ffill().fillna(0).values
    filt = np.zeros(len(vals))
    filt[0] = vals[0]
    if len(vals) > 1:
        filt[1] = vals[1]
    for i in range(2, len(vals)):
        filt[i] = c1 * (vals[i] + vals[i-1]) / 2 + c2 * filt[i-1] + c3 * filt[i-2]
    return pd.Series(filt, index=series.index)


def shannon_entropy(series, window=15, bins=6):
    def calc_shannon(x):
        if len(x) < window:
            return np.nan
        counts, _ = np.histogram(x, bins=bins)
        probs = counts / len(x)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    returns = series.pct_change().fillna(0)
    return returns.rolling(window=window).apply(calc_shannon, raw=True)


def efficiency_ratio(series, period=14):
    change = series.diff().abs()
    volatility = change.rolling(period).sum()
    direction = series.diff(period).abs()
    return direction / volatility


def compute_cycle_fast(series, lookback=40):
    """Fast cycle using zero-crossing of detrended price."""
    # Simple detrend using rolling mean
    trend = series.rolling(lookback).mean()
    detrended = series - trend
    # Smooth to reduce noise
    smooth = ehler_supersmoother(detrended, length=10)
    return smooth


def generate_all_features(df):
    df = df.copy()
    
    # ATR
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # Ichimoku
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
    
    # Filters
    df['momentum'] = df['close'].pct_change(periods=10)
    df['momentum_smooth'] = ehler_supersmoother(df['momentum'], length=5)
    df['smooth_direction'] = (df['momentum_smooth'] > 0).astype(float)
    
    # Fast cycle using zero-crossing of detrended smooth
    cycle_raw = compute_cycle_fast(df['close'], lookback=40)
    df['cycle_direction'] = (cycle_raw > 0).astype(float)
    
    df['entropy_gate'] = (df['Entropy'] < 2.8).astype(float)
    df['entropy_gate_strict'] = (df['Entropy'] < 2.5).astype(float)
    
    sma_fast = df['close'].rolling(50).mean()
    sma_slow = df['close'].rolling(200).mean()
    df['trend_filter'] = (sma_fast > sma_slow).astype(float)
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


def ichimoku_signal_fast(df_arr, idx_start, idx_end, min_hold, max_hold,
                          profit_take_pct=0.10, trailing_stop_pct=0.05):
    """Fast signal using pre-extracted arrays."""
    n = idx_end - idx_start
    position = np.zeros(n)
    in_position = False
    hold_count = 0
    entry_price = 0.0
    peak_price = 0.0
    
    for j in range(n):
        i = idx_start + j
        imo = df_arr['IMO'][i]
        er = df_arr['ER'][i]
        std = df_arr['IMO_Std'][i]
        entropy = df_arr['Entropy'][i]
        close = df_arr['close'][i]
        cloud_min = df_arr['cloud_min'][i]
        
        if np.isnan(imo) or np.isnan(er) or np.isnan(std) or np.isnan(entropy):
            if in_position:
                position[j] = 1.0
            continue
        
        threshold = std * 0.40
        
        if in_position:
            hold_count += 1
            peak_price = max(peak_price, close)
            can_exit = hold_count >= min_hold
            exit_signal = False
            
            if can_exit and entry_price > 0:
                trade_return = (close - entry_price) / entry_price
                if trade_return >= profit_take_pct:
                    exit_signal = True
            
            if can_exit and not exit_signal and peak_price > 0:
                dd_from_peak = (peak_price - close) / peak_price
                if dd_from_peak >= trailing_stop_pct:
                    exit_signal = True
            
            if not exit_signal and can_exit:
                if imo < -0.30:
                    exit_signal = True
                elif hold_count >= max_hold:
                    exit_signal = True
                elif close < cloud_min and imo < 0:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                hold_count = 0
            else:
                position[j] = 1.0
        else:
            gate_pass = True
            if not np.isnan(cloud_min):
                gate_pass = (close >= cloud_min)
            
            if imo > threshold and er > 0.25 and entropy < 2.271 and gate_pass:
                in_position = True
                hold_count = 0
                entry_price = close
                peak_price = close
                position[j] = 1.0
    
    return position


def apply_gate_fast(ichimoku_pos, filter_signals, gate_threshold):
    """Fast gate using pre-extracted filter arrays."""
    n = len(ichimoku_pos)
    result = np.zeros(n)
    filter_matrix = np.column_stack(filter_signals)
    in_position = False
    
    for i in range(n):
        if not in_position:
            bullish = np.sum(filter_matrix[i] == 1.0)
            if ichimoku_pos[i] == 1.0 and bullish >= gate_threshold:
                in_position = True
                result[i] = 1.0
        else:
            bullish = np.sum(filter_matrix[i] == 1.0)
            if ichimoku_pos[i] == 0.0 or bullish < gate_threshold:
                in_position = False
            else:
                result[i] = 1.0
    
    return result


def compute_metrics_fast(signal, prices):
    """Fast metrics from numpy arrays."""
    n = len(prices)
    if n < 2:
        return {'cagr': 0, 'sharpe': 0, 'max_dd': 0, 'n_trades': 0, 'win_rate': 0}
    
    returns = np.diff(prices) / prices[:-1]
    
    # Signal shifted by 1
    strat_returns = np.zeros(n - 1)
    strat_returns[1:] = returns * signal[:-1]
    
    # Transaction costs
    transitions = np.zeros(n)
    transitions[1:] = np.abs(np.diff(signal))
    strat_returns -= transitions[1:] * (TRANSACTION_COST / 2)
    
    # Filter out initial zeros
    active = np.abs(strat_returns) > 0
    if not np.any(active):
        return {'cagr': 0, 'sharpe': 0, 'max_dd': 0, 'n_trades': 0, 'win_rate': 0}
    
    equity = np.cumprod(1 + strat_returns)
    years = len(strat_returns) / 365.25
    cagr = (equity[-1]) ** (1/years) - 1 if years > 0 else 0
    
    std_r = np.std(strat_returns)
    sharpe = np.mean(strat_returns) / std_r * np.sqrt(365) if std_r > 0 else 0
    
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = np.min(dd)
    
    # Trades
    in_pos = False
    entry_price = 0.0
    n_trades = 0
    wins = 0
    
    for i in range(n):
        if signal[i] == 1.0 and not in_pos:
            in_pos = True
            entry_price = prices[i]
        elif signal[i] == 0.0 and in_pos:
            in_pos = False
            n_trades += 1
            if prices[i] > entry_price:
                wins += 1
    
    win_rate = wins / n_trades * 100 if n_trades > 0 else 0
    
    return {
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'max_dd': round(max_dd * 100, 2),
        'n_trades': n_trades,
        'win_rate': round(win_rate, 1)
    }


def run_grid_search():
    print("=" * 70)
    print("FINAL GRID SEARCH V3 — ULTRA FAST")
    print("=" * 70)
    
    t0 = time.time()
    
    # Load data
    with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
        btc_data = json.load(f)
    
    df = pd.DataFrame(btc_data['aligned_data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df[df.index >= '2018-01-01']
    
    # Pre-compute ALL features once
    print("  Pre-computing features...")
    df_feat = generate_all_features(df.copy())
    print(f"  Features done in {time.time()-t0:.1f}s")
    
    # Find boundaries
    train_mask = (df_feat.index >= '2018-01-01') & (df_feat.index <= '2023-12-31')
    test_mask = (df_feat.index >= '2024-01-01') & (df_feat.index <= '2026-06-30')
    train_start_idx = int(np.where(train_mask)[0][0])
    train_end_idx = int(np.where(train_mask)[0][-1]) + 1
    test_start_idx = int(np.where(test_mask)[0][0])
    test_end_idx = int(np.where(test_mask)[0][-1]) + 1
    
    prices_all = df['close'].values
    prices_train = prices_all[train_start_idx:train_end_idx]
    prices_test = prices_all[test_start_idx:test_end_idx]
    
    print(f"  Train: {train_end_idx - train_start_idx} bars, Test: {test_end_idx - test_start_idx} bars")
    
    # Extract arrays for speed
    df_arrays = {
        'IMO': df_feat['IMO'].values,
        'ER': df_feat['ER'].values,
        'IMO_Std': df_feat['IMO_Std'].values,
        'Entropy': df_feat['Entropy'].values,
        'close': df_feat['close'].values,
        'cloud_min': df_feat['cloud_min'].values,
    }
    
    filter_names = ['msvr_direction', 'smooth_direction', 'cycle_direction', 'entropy_gate']
    filter_arrays_train = [df_feat[col].values[train_start_idx:train_end_idx] for col in filter_names]
    filter_arrays_test = [df_feat[col].values[test_start_idx:test_end_idx] for col in filter_names]
    
    # ================================================================
    # GRID SEARCH
    # ================================================================
    print("\n  Running grid search...")
    
    # Parameter grid - focused on most impactful ranges
    min_holds = [15, 20, 25, 30, 35, 40, 45]
    max_holds = [50, 55, 60, 70, 75, 90, 120]
    gates = [2, 3]
    profit_takes = [0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]
    trailing_stops = [0.0, 0.03, 0.05, 0.07, 0.10]
    
    total = len(min_holds) * len(max_holds) * len(gates) * len(profit_takes) * len(trailing_stops)
    print(f"  Total: {total} combinations")
    
    results = []
    count = 0
    t1 = time.time()
    
    for mh in min_holds:
        for xh in max_holds:
            for gate in gates:
                for pt in profit_takes:
                    for ts in trailing_stops:
                        count += 1
                        
                        try:
                            ichimoku_train = ichimoku_signal_fast(
                                df_arrays, train_start_idx, train_end_idx, mh, xh, pt, ts)
                            ichimoku_test = ichimoku_signal_fast(
                                df_arrays, test_start_idx, test_end_idx, mh, xh, pt, ts)
                            
                            # Verify lengths match
                            assert len(ichimoku_train) == len(filter_arrays_train[0]), \
                                f'Train signal {len(ichimoku_train)} != filter {len(filter_arrays_train[0])}'
                            assert len(ichimoku_test) == len(filter_arrays_test[0]), \
                                f'Test signal {len(ichimoku_test)} != filter {len(filter_arrays_test[0])}'
                            
                            pos_train = apply_gate_fast(ichimoku_train, filter_arrays_train, gate)
                            pos_test = apply_gate_fast(ichimoku_test, filter_arrays_test, gate)
                            
                            m_train = compute_metrics_fast(pos_train, prices_train)
                            m_test = compute_metrics_fast(pos_test, prices_test)
                            
                            if m_train['sharpe'] > 0:
                                deg = (m_test['sharpe'] - m_train['sharpe']) / m_train['sharpe'] * 100
                            else:
                                deg = 0
                            
                            results.append({
                                'min_hold': mh, 'max_hold': xh, 'gate': gate,
                                'profit_take': pt, 'trailing_stop': ts,
                                'train_sharpe': m_train['sharpe'],
                                'train_winrate': m_train['win_rate'],
                                'train_trades': m_train['n_trades'],
                                'train_cagr': m_train['cagr'],
                                'test_sharpe': m_test['sharpe'],
                                'test_winrate': m_test['win_rate'],
                                'test_trades': m_test['n_trades'],
                                'test_cagr': m_test['cagr'],
                                'test_maxdd': m_test['max_dd'],
                                'degradation': round(deg, 1)
                            })
                        except Exception as e:
                            if count <= 3:
                                print(f"  Error: {e}")
                        
                        if count % 500 == 0:
                            elapsed = time.time() - t1
                            rate = count / elapsed if elapsed > 0 else 0
                            eta = (total - count) / rate if rate > 0 else 0
                            print(f"  {count}/{total} ({elapsed:.0f}s, {rate:.0f}/s, ETA {eta:.0f}s)")
    
    elapsed = time.time() - t1
    print(f"  Grid search done in {elapsed:.1f}s ({count/elapsed:.0f} configs/s)")
    print(f"  Total configs: {len(results)}")
    
    if len(results) == 0:
        print("  ERROR: No results collected!")
        return None, 0
    
    # ================================================================
    # ANALYSIS
    # ================================================================
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('test_sharpe', ascending=False)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, 'final_grid_results.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\n  Results saved to: {csv_path}")
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Top 20 by Sharpe
    print("\n--- Top 20 by Test Sharpe ---")
    print(f"{'#':<4} {'MH':<4} {'xH':<4} {'G':<3} {'PT%':<6} {'TS%':<6} {'TS':<6} {'TW%':<6} {'TT':<4} {'CAG':<6} {'D%':<6}")
    print("-" * 70)
    for idx, (_, r) in enumerate(df_results.head(20).iterrows()):
        print(f"{idx+1:<4} {int(r['min_hold']):<4} {int(r['max_hold']):<4} {int(r['gate']):<3} "
              f"{r['profit_take']*100:<6.1f} {r['trailing_stop']*100:<6.1f} "
              f"{r['test_sharpe']:<6.2f} {r['test_winrate']:<6.1f} {int(r['test_trades']):<4} "
              f"{r['test_cagr']:<6.1f} {r['degradation']:<6.1f}")
    
    # ALL targets
    target_all = df_results[
        (df_results['test_sharpe'] >= 1.35) &
        (df_results['test_winrate'] >= 60) &
        (df_results['test_trades'] >= 25) &
        (df_results['test_trades'] <= 35) &
        (df_results['test_cagr'] >= 50)
    ]
    print(f"\n--- ALL Targets: S>1.35, W>60%, T 25-35, C>50% ---")
    print(f"  Found: {len(target_all)}")
    if len(target_all) > 0:
        for idx, (_, r) in enumerate(target_all.head(10).iterrows()):
            print(f"  {idx+1}. MH={int(r['min_hold'])}, xH={int(r['max_hold'])}, G={int(r['gate'])}, "
                  f"PT={r['profit_take']*100:.1f}%, TS={r['trailing_stop']*100:.1f}%")
            print(f"     S={r['test_sharpe']:.2f}, W={r['test_winrate']:.1f}%, T={int(r['test_trades'])}, C={r['test_cagr']:.1f}%")
    
    # Sharpe > 1.35
    sharpe_above = df_results[df_results['test_sharpe'] >= 1.35]
    print(f"\n--- Sharpe > 1.35 (any metrics) ---")
    print(f"  Found: {len(sharpe_above)}")
    if len(sharpe_above) > 0:
        for idx, (_, r) in enumerate(sharpe_above.head(15).iterrows()):
            print(f"  {idx+1}. MH={int(r['min_hold'])}, xH={int(r['max_hold'])}, G={int(r['gate'])}, "
                  f"PT={r['profit_take']*100:.1f}%, TS={r['trailing_stop']*100:.1f}%")
            print(f"     S={r['test_sharpe']:.2f}, W={r['test_winrate']:.1f}%, T={int(r['test_trades'])}, C={r['test_cagr']:.1f}%")
    
    # Best win rate
    best_wr = df_results.sort_values('test_winrate', ascending=False).head(10)
    print(f"\n--- Top 10 by Win Rate ---")
    for idx, (_, r) in enumerate(best_wr.iterrows()):
        print(f"  {idx+1}. MH={int(r['min_hold'])}, xH={int(r['max_hold'])}, PT={r['profit_take']*100:.1f}%, TS={r['trailing_stop']*100:.1f}%")
        print(f"     S={r['test_sharpe']:.2f}, W={r['test_winrate']:.1f}%, T={int(r['test_trades'])}, C={r['test_cagr']:.1f}%")
    
    # Balanced score
    def score_row(r):
        s = min(r['test_sharpe'] / 1.5, 1.0)
        w = min(r['test_winrate'] / 70.0, 1.0)
        if 25 <= r['test_trades'] <= 35:
            t = 1.0
        elif r['test_trades'] < 25:
            t = max(0, 1.0 - (25 - r['test_trades']) / 20.0)
        else:
            t = max(0, 1.0 - (r['test_trades'] - 35) / 20.0)
        c = min(r['test_cagr'] / 60.0, 1.0)
        return 0.4 * s + 0.2 * w + 0.15 * t + 0.25 * c
    
    df_results['score'] = df_results.apply(score_row, axis=1)
    df_scored = df_results.sort_values('score', ascending=False)
    
    print(f"\n--- Top 15 by Balanced Score ---")
    print(f"{'#':<4} {'Score':<7} {'MH':<4} {'xH':<4} {'PT%':<6} {'TS%':<6} {'TS':<6} {'TW%':<6} {'TT':<4} {'CAG':<6}")
    print("-" * 65)
    for idx, (_, r) in enumerate(df_scored.head(15).iterrows()):
        print(f"{idx+1:<4} {r['score']:<7.3f} {int(r['min_hold']):<4} {int(r['max_hold']):<4} "
              f"{r['profit_take']*100:<6.1f} {r['trailing_stop']*100:<6.1f} "
              f"{r['test_sharpe']:<6.2f} {r['test_winrate']:<6.1f} {int(r['test_trades']):<4} "
              f"{r['test_cagr']:<6.1f}")
    
    # ================================================================
    # OPTIMAL CONFIG
    # ================================================================
    print("\n" + "=" * 70)
    print("OPTIMAL CONFIGURATION")
    print("=" * 70)
    
    if len(target_all) > 0:
        best = target_all.iloc[0]
        print(f"\n✅ ALL TARGETS MET!")
    elif len(sharpe_above) > 0:
        best = sharpe_above.iloc[0]
        print(f"\n⚠️  Sharpe target met (1.35+), other targets partially met:")
    else:
        best = df_results.iloc[0]
        print(f"\n❌ Best config found:")
    
    print(f"\n  MIN_HOLD: {int(best['min_hold'])}")
    print(f"  MAX_HOLD: {int(best['max_hold'])}")
    print(f"  GATE: {int(best['gate'])}")
    print(f"  PROFIT_TAKE: {best['profit_take']*100:.1f}%")
    print(f"  TRAILING_STOP: {best['trailing_stop']*100:.1f}%")
    print(f"\n  TEST METRICS:")
    print(f"    Sharpe:   {best['test_sharpe']:.2f} {'✅' if best['test_sharpe'] >= 1.35 else '❌'} (>1.35)")
    print(f"    Win Rate: {best['test_winrate']:.1f}% {'✅' if best['test_winrate'] >= 60 else '❌'} (>60%)")
    print(f"    Trades:   {int(best['test_trades'])} {'✅' if 25 <= best['test_trades'] <= 35 else '❌'} (25-35)")
    print(f"    CAGR:     {best['test_cagr']:.1f}% {'✅' if best['test_cagr'] >= 50 else '❌'} (>50%)")
    print(f"    Max DD:   {best['test_maxdd']:.1f}%")
    print(f"    Degrad:   {best['degradation']:.1f}%")
    
    targets_met = sum([
        best['test_sharpe'] >= 1.35,
        best['test_winrate'] >= 60,
        25 <= best['test_trades'] <= 35,
        best['test_cagr'] >= 50
    ])
    print(f"\n  TARGETS MET: {targets_met}/4")
    
    total_time = time.time() - t0
    print(f"\n  Total time: {total_time:.1f}s")
    
    return df_results, targets_met


if __name__ == "__main__":
    results_df, targets_met = run_grid_search()
    sys.exit(0 if targets_met == 4 else 1)
