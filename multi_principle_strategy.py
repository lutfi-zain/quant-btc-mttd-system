#!/usr/bin/env python3
"""
Multi-Principle MTTD Strategy v2
=================================
Direct port of proven ichimoku_quant logic + additional family gates.

Strategy Logic (copied from ichimoku_quant's 6-family approach):
  Entry (ALL must pass):
    1. IMO > IMO_STD * threshold       (Adaptive threshold)
    2. ER > ER_ENTRY                   (Fractal trend gate)
    3. Entropy < ENTROPY_THRESH        (Noise gate)
    4. Close >= Cloud_Min              (Cloud trend filter)
    5. 2-bar confirmation
    
  Exit (ANY can trigger):
    1. Chikou < CHIKOU_THRESH (not immune)
    2. IMO < IMO_EXIT_BULL (above cloud) or IMO < 0 (below cloud)
    
  Immunity (hold through bull):
    1. IMO >= IMMUNITY_THRESH (strong bull)
    2. OR (Close >= Cloud_Max AND ROC >= CRASH_GATE AND IMO >= IMO_MIN_LIMIT)
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# Feature Generators (from ichimoku_quant)
# ================================================================

def compute_atr(df, window=60):
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

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

# ================================================================
# Ichimoku-Quant Features (direct port)
# ================================================================

def generate_ichimoku_features(df, p1=20, p2=60, p3=120, er_len=14, std_len=30, 
                                entropy_window=15, entropy_bins=6):
    """Direct port of ichimoku_quant features.py"""
    result = df.copy()
    
    # ATR
    result['ATR'] = compute_atr(result, p2)
    
    # Base Ichimoku lines
    result['tenkan_sen'] = (result['high'].rolling(p1).max() + result['low'].rolling(p1).min()) / 2
    result['kijun_sen'] = (result['high'].rolling(p2).max() + result['low'].rolling(p2).min()) / 2
    
    result['senkou_span_a_raw'] = (result['tenkan_sen'] + result['kijun_sen']) / 2
    result['senkou_span_b_raw'] = (result['high'].rolling(p3).max() + result['low'].rolling(p3).min()) / 2
    
    result['senkou_span_a'] = result['senkou_span_a_raw'].shift(p2)
    result['senkou_span_b'] = result['senkou_span_b_raw'].shift(p2)
    
    # Normalized components
    result['S_TK'] = np.tanh((result['tenkan_sen'] - result['kijun_sen']) / result['ATR'])
    
    cloud_max = np.maximum(result['senkou_span_a'], result['senkou_span_b'])
    cloud_min = np.minimum(result['senkou_span_a'], result['senkou_span_b'])
    dist_cloud = np.zeros(len(result))
    above = result['close'] > cloud_max
    below = result['close'] < cloud_min
    dist_cloud[above] = (result['close'] - cloud_max)[above] / result['ATR'][above]
    dist_cloud[below] = (result['close'] - cloud_min)[below] / result['ATR'][below]
    result['S_Cloud'] = np.tanh(dist_cloud)
    
    result['S_Future'] = np.tanh((result['senkou_span_a_raw'] - result['senkou_span_b_raw']) / result['ATR'])
    
    raw_chikou_dist = (result['close'] - result['close'].shift(p2)) / result['ATR']
    smoothed_chikou_dist = ehler_supersmoother(raw_chikou_dist, length=4)
    result['S_Chikou'] = np.tanh(smoothed_chikou_dist)
    
    # Composite IMO
    imo_raw = (result['S_TK'] + result['S_Cloud'] + result['S_Future'] + result['S_Chikou']) / 4.0
    result['IMO'] = ehler_supersmoother(imo_raw, length=7)
    result['IMO_Std'] = result['IMO'].rolling(std_len).std()
    
    # Efficiency Ratio
    result['ER'] = efficiency_ratio(result['close'], period=er_len)
    
    # Shannon Entropy
    result['Entropy'] = shannon_entropy(result['close'], window=entropy_window, bins=entropy_bins)
    
    # ROC for crash gate
    result['roc_gate'] = result['close'].pct_change(periods=30).fillna(0)
    
    # Cloud max/min for immunity
    result['cloud_max'] = cloud_max
    result['cloud_min'] = cloud_min
    
    return result

# ================================================================
# Strategy (direct port of ichimoku_quant strategy.py)
# ================================================================

def multi_principle_strategy(df, 
                            confirm_entry=2,
                            confirm_exit=1,
                            min_hold_days=10,
                            max_hold_days=120,
                            er_entry=0.25,
                            t_entry=0.40,
                            chikou_thresh=-0.30,
                            immunity_thresh=0.50,
                            entropy_thresh=2.271,
                            imo_min_limit=-0.30,
                            imo_exit_bull=-0.30,
                            roc_gate_limit=-0.20,
                            cooldown=5):
    """
    Multi-principle strategy directly ported from ichimoku_quant.
    
    Architecture (6+ families):
    - Smoothing: Ichimoku lines (tenkan/kijun/senkou)
    - Filtering: Ehler SuperSmoother on IMO
    - Fractal: Efficiency Ratio gate
    - Entropy: Shannon Entropy gate
    - Spectral: Cycle detection via normalized components
    - Momentum: Chikou span for exit
    """
    result = generate_ichimoku_features(df)
    
    pos = 0.0
    positions = []
    confirm_count = 0
    hold_days = 0
    intent = None
    cooldown_count = 0
    
    for i, (idx, row) in enumerate(result.iterrows()):
        imo = row['IMO']
        er = row['ER']
        std = row['IMO_Std']
        chikou = row['S_Chikou']
        entropy = row['Entropy']
        close = row['close']
        cloud_a = row['senkou_span_a']
        cloud_b = row['senkou_span_b']
        cloud_min = row['cloud_min']
        cloud_max = row['cloud_max']
        roc = row['roc_gate']
        
        # Cooldown
        if cooldown_count > 0:
            cooldown_count -= 1
        
        # Skip if NaN
        if pd.isna(imo) or pd.isna(er) or pd.isna(std) or pd.isna(entropy):
            positions.append(pos)
            continue
        
        threshold = std * t_entry
        
        if pos > 0:
            hold_days += 1
        else:
            hold_days = 0
        
        can_exit = hold_days >= min_hold_days
        
        if pos == 0.0:
            # === ENTRY GATES ===
            gate_pass = True
            if not pd.isna(cloud_min):
                gate_pass = (close >= cloud_min)
            
            if (imo > threshold and 
                er > er_entry and 
                entropy < entropy_thresh and 
                gate_pass and
                cooldown_count == 0):
                
                if intent != 1.0:
                    intent = 1.0
                    confirm_count = 1
                else:
                    confirm_count += 1
                if confirm_count >= confirm_entry:
                    pos = 1.0
                    confirm_count = 0
                    hold_days = 0
                    intent = None
            else:
                intent = None
                confirm_count = 0
        
        else:  # pos == 1.0
            # === EXIT LOGIC ===
            exit_signal = False
            
            if can_exit:
                is_above_cloud = (not pd.isna(cloud_max) and close >= cloud_max)
                is_not_crashing = (roc >= roc_gate_limit)
                
                # Dynamic immunity
                is_immune = (imo >= immunity_thresh)
                if is_above_cloud and is_not_crashing:
                    is_immune = is_immune or (imo >= imo_min_limit)
                
                # Dynamic macro exit threshold
                current_macro_exit_th = 0.0
                if is_above_cloud and is_not_crashing:
                    current_macro_exit_th = imo_exit_bull
                
                if chikou < chikou_thresh and not is_immune:
                    exit_signal = True
                elif imo < current_macro_exit_th:
                    exit_signal = True
            
            # Max hold forced exit
            if hold_days >= max_hold_days:
                exit_signal = True
            
            if exit_signal:
                if intent != 0.0:
                    intent = 0.0
                    confirm_count = 1
                else:
                    confirm_count += 1
                if confirm_count >= confirm_exit:
                    pos = 0.0
                    confirm_count = 0
                    hold_days = 0
                    intent = None
                    cooldown_count = cooldown
            else:
                intent = None
                confirm_count = 0
        
        positions.append(pos)
    
    result['Pos'] = positions
    return result

# ================================================================
# Backtest
# ================================================================

def backtest(result_df, prices, tc=0.001):
    """Compute trade metrics with transaction costs."""
    pos = result_df['Pos']
    
    # Extract trades
    trades = []
    in_pos = False
    entry_date = None
    entry_price = None
    
    for i, (date, p) in enumerate(pos.items()):
        if p == 1.0 and not in_pos:
            in_pos = True
            entry_date = date
            entry_price = prices.loc[date]
        elif p == 0.0 and in_pos:
            in_pos = False
            exit_price = prices.loc[date]
            gross_ret = (exit_price / entry_price) - 1
            net_ret = gross_ret - tc  # 0.1% round trip
            trades.append({
                'entry_date': entry_date,
                'exit_date': date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'gross_return': gross_ret,
                'net_return': net_ret,
                'hold_days': (date - entry_date).days,
                'is_win': net_ret > 0
            })
    
    if not trades:
        return {'trades': 0, 'win_rate': 0, 'sharpe': 0, 'cagr': 0, 'max_dd': 0}
    
    # Compute returns series
    daily_returns = pos.shift(1) * prices.pct_change() - tc * pos.diff().abs() / 2
    daily_returns = daily_returns.dropna()
    
    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(365) if daily_returns.std() > 0 else 0
    equity = (1 + daily_returns).cumprod()
    years = len(daily_returns) / 365.25
    cagr = equity.iloc[-1] ** (1/years) - 1 if years > 0 else 0
    
    # Max drawdown
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    max_dd = dd.min()
    
    n_win = sum(1 for t in trades if t['is_win'])
    win_rate = n_win / len(trades) * 100
    avg_hold = np.mean([t['hold_days'] for t in trades])
    avg_return = np.mean([t['net_return'] for t in trades]) * 100
    
    return {
        'trades': len(trades),
        'win_rate': round(win_rate, 1),
        'sharpe': round(sharpe, 2),
        'cagr': round(cagr * 100, 1),
        'max_dd': round(max_dd * 100, 1),
        'avg_hold': int(avg_hold),
        'avg_return': round(avg_return, 2)
    }

# ================================================================
# Main
# ================================================================

def main():
    import json
    import time
    
    print("=" * 70)
    print("MULTI-PRINCIPLE MTTD STRATEGY v2")
    print("=" * 70)
    
    # Load data
    print("\n[1/5] Loading data...")
    with open('data/btc_daily.json') as f:
        btc_data = json.load(f)
    df = pd.DataFrame(btc_data['aligned_data'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df[df.index >= '2018-01-01']
    
    train = df[df.index < '2025-01-01']
    test = df[df.index >= '2025-01-01']
    
    print(f"  Total: {len(df)} bars")
    print(f"  Train: {len(train)} bars ({train.index[0].date()} to {train.index[-1].date()})")
    print(f"  Test:  {len(test)} bars ({test.index[0].date()} to {test.index[-1].date()})")
    
    # Default config (matching ichimoku_quant)
    config = {
        'min_hold_days': 10,
        'max_hold_days': 120,
        'er_entry': 0.25,
        't_entry': 0.40,
        'chikou_thresh': -0.30,
        'immunity_thresh': 0.50,
        'entropy_thresh': 2.271,
        'imo_min_limit': -0.30,
        'imo_exit_bull': -0.30,
        'roc_gate_limit': -0.20,
        'cooldown': 5
    }
    
    print(f"\n[2/5] Running strategy (ichimoku_quant default params)...")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    start = time.time()
    result = multi_principle_strategy(train.copy(), **config)
    train_metrics = backtest(result, train['close'])
    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s")
    print(f"  Train: {train_metrics['trades']} trades, {train_metrics['win_rate']}% win, Sharpe {train_metrics['sharpe']}, CAGR {train_metrics['cagr']}%")
    
    start = time.time()
    result_test = multi_principle_strategy(test.copy(), **config)
    test_metrics = backtest(result_test, test['close'])
    elapsed = time.time() - start
    print(f"  Test:  {test_metrics['trades']} trades, {test_metrics['win_rate']}% win, Sharpe {test_metrics['sharpe']}, CAGR {test_metrics['cagr']}%")
    
    # Degradation
    if abs(train_metrics['sharpe']) > 0.01:
        degrad = (test_metrics['sharpe'] - train_metrics['sharpe']) / abs(train_metrics['sharpe']) * 100
    else:
        degrad = 0
    print(f"  Degradation: {degrad:.1f}%")
    
    print(f"\n[3/5] Summary:")
    print(f"  {'':20} {'Trades':>8} {'Win%':>8} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'AvgHold':>8}")
    print(f"  {'Train':20} {train_metrics['trades']:>8} {train_metrics['win_rate']:>7.1f}% {train_metrics['sharpe']:>8.2f} {train_metrics['cagr']:>7.1f}% {train_metrics['max_dd']:>7.1f}% {train_metrics['avg_hold']:>8}")
    print(f"  {'Test':20} {test_metrics['trades']:>8} {test_metrics['win_rate']:>7.1f}% {test_metrics['sharpe']:>8.2f} {test_metrics['cagr']:>7.1f}% {test_metrics['max_dd']:>7.1f}% {test_metrics['avg_hold']:>8}")
    
    print(f"\n[4/5] Trade list:")
    trades = []
    in_pos = False
    entry_date = None
    entry_price = None
    for i, (date, p) in enumerate(result['Pos'].items()):
        if p == 1.0 and not in_pos:
            in_pos = True
            entry_date = date
            entry_price = train['close'].loc[date]
        elif p == 0.0 and in_pos:
            in_pos = False
            exit_price = train['close'].loc[date]
            ret = (exit_price / entry_price - 1) * 100
            trades.append({'entry': entry_date.date(), 'exit': date.date(), 'return': ret, 'win': ret > 0})
    
    print(f"  Train trades ({len(trades)}):")
    for t in trades[:10]:
        print(f"    {t['entry']} → {t['exit']} : {t['return']:+.1f}% {'✅' if t['win'] else '❌'}")
    if len(trades) > 10:
        print(f"    ... and {len(trades)-10} more")
    
    print(f"\n[5/5] Saving results...")
    os.makedirs('mttd/multi_principle', exist_ok=True)
    
    # Save combined result
    full_result = multi_principle_strategy(df.copy(), **config)
    full_metrics = backtest(full_result, df['close'])
    print(f"  Full period: {full_metrics['trades']} trades, {full_metrics['win_rate']}% win, Sharpe {full_metrics['sharpe']}, CAGR {full_metrics['cagr']}%")
    
    full_result.to_csv('mttd/multi_principle/signals.csv')
    
    # Save metrics
    results = {
        'config': config,
        'train': train_metrics,
        'test': test_metrics,
        'full': full_metrics,
        'degradation': round(degrad, 1)
    }
    with open('mttd/multi_principle/results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"  Saved to mttd/multi_principle/")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

if __name__ == '__main__':
    main()
