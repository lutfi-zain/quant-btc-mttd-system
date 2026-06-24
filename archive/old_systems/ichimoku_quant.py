#!/usr/bin/env python3
"""
Ichimoku Quant System — Ported from quant-lttd-ichimoku
=========================================================

Advanced Ichimoku implementation with:
- Ehler SuperSmoother (Family 2: Filtering)
- Shannon Entropy (Family 7: Entropy)
- Efficiency Ratio (Family 5: Fractal)
- Multiple confirmation bars
- Dynamic immunity based on IMO level

Ported from: /home/ubuntu/projects/quant-lttd-ichimoku/
"""

import numpy as np
import pandas as pd

def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Compute Average True Range."""
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift(1)).abs()
    tr3 = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()

def ehler_supersmoother(series: pd.Series, length: int = 7) -> pd.Series:
    """
    Ehler's SuperSmoother Filter (Spectral / Filtering family).
    Removes high-frequency noise below 'length' cycle period without lag penalty.
    """
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
    """
    Shannon Entropy of rolling returns (Entropy & Information family).
    Measures the randomness/complexity of the price return distribution.
    """
    def calc_shannon(x):
        if len(x) < window:
            return np.nan
        counts, _ = np.histogram(x, bins=bins)
        probs = counts / len(x)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))
    
    returns = series.pct_change().fillna(0)
    return returns.rolling(window=window).apply(calc_shannon, raw=True)

def generate_ichimoku_features(df: pd.DataFrame, 
                                p1: int = 20, 
                                p2: int = 60, 
                                p3: int = 120, 
                                er_len: int = 14, 
                                std_len: int = 30, 
                                entropy_window: int = 15, 
                                entropy_bins: int = 6) -> pd.DataFrame:
    """
    Generates hyper-tuned Ichimoku components.
    - Macro periods (20, 60, 120) calibrated for 24/7 crypto market
    - Ehler SuperSmoother applied on final IMO for noise reduction
    - Efficiency Ratio (Fractal family) for trend strength gate
    """
    df = df.copy()
    df['ATR'] = compute_atr(df, p2)

    # Base Ichimoku lines
    df['tenkan_sen'] = (df['high'].rolling(p1).max() + df['low'].rolling(p1).min()) / 2
    df['kijun_sen'] = (df['high'].rolling(p2).max() + df['low'].rolling(p2).min()) / 2

    df['senkou_span_a_raw'] = (df['tenkan_sen'] + df['kijun_sen']) / 2
    df['senkou_span_b_raw'] = (df['high'].rolling(p3).max() + df['low'].rolling(p3).min()) / 2

    df['senkou_span_a'] = df['senkou_span_a_raw'].shift(p2)
    df['senkou_span_b'] = df['senkou_span_b_raw'].shift(p2)

    # Normalized components (tanh → bounded [-1, 1])
    df['S_TK'] = np.tanh((df['tenkan_sen'] - df['kijun_sen']) / df['ATR'])

    cloud_max = np.maximum(df['senkou_span_a'], df['senkou_span_b'])
    cloud_min = np.minimum(df['senkou_span_a'], df['senkou_span_b'])
    dist_cloud = np.zeros(len(df))
    above = df['close'] > cloud_max
    below = df['close'] < cloud_min
    dist_cloud[above] = (df['close'] - cloud_max)[above] / df['ATR'][above]
    dist_cloud[below] = (df['close'] - cloud_min)[below] / df['ATR'][below]
    df['S_Cloud'] = np.tanh(dist_cloud)

    df['S_Future'] = np.tanh((df['senkou_span_a_raw'] - df['senkou_span_b_raw']) / df['ATR'])
    raw_chikou_dist = (df['close'] - df['close'].shift(p2)) / df['ATR']
    smoothed_chikou_dist = ehler_supersmoother(raw_chikou_dist, length=4)
    df['S_Chikou'] = np.tanh(smoothed_chikou_dist)

    # Composite IMO (raw)
    imo_raw = (df['S_TK'] + df['S_Cloud'] + df['S_Future'] + df['S_Chikou']) / 4.0

    # Apply Ehler SuperSmoother (noise reduction without lag)
    df['IMO'] = ehler_supersmoother(imo_raw, length=7)
    df['IMO_Std'] = df['IMO'].rolling(std_len).std()

    # Efficiency Ratio (Fractal family — trend vs noise measure)
    change = df['close'].diff().abs()
    volatility = change.rolling(er_len).sum()
    direction = df['close'].diff(er_len).abs()
    df['ER'] = direction / volatility

    # Shannon Entropy (Entropy family — randomness filter)
    df['Entropy'] = shannon_entropy(df['close'], window=entropy_window, bins=entropy_bins)

    # Price ROC for exit crash gate (30 days lookback)
    df['roc_gate'] = df['close'].pct_change(periods=30).fillna(0)

    return df

def generate_ichimoku_signals(df: pd.DataFrame,
                               confirm_entry: int = 2,
                               confirm_exit: int = 1,
                               min_hold_days: int = 10,
                               er_entry: float = 0.25,
                               t_entry: float = 0.40,
                               chikou_thresh: float = -0.30,
                               immunity_thresh: float = 0.50,
                               entropy_thresh: float = 2.271,
                               imo_min_limit: float = -0.30,
                               imo_exit_bull: float = -0.30,
                               roc_gate_limit: float = -0.20) -> pd.DataFrame:
    """
    Clean binary denoised signal generator.
    
    Architecture:
    - Layer 1: Ehler SuperSmoother (spectral/filtering)
    - Layer 2: Efficiency Ratio gate (fractal family) — entry only
    - Layer 3: Adaptive Volatility Threshold — entry level
    - Layer 4: S_Chikou Momentum Drop — exit level
    - Layer 5: Signal Persistence / Confirmation bars
    - Layer 6: Minimum Hold Period
    """
    if 'IMO' not in df.columns or 'ER' not in df.columns or 'IMO_Std' not in df.columns or 'Entropy' not in df.columns or 'senkou_span_a' not in df.columns or 'senkou_span_b' not in df.columns:
        raise ValueError("Required columns not found. Run generate_ichimoku_features first.")

    df = df.copy()

    pos = 0.0
    signals = []
    regimes = []
    confirm_count = 0
    hold_days = 0
    intent = None

    for _, row in df.iterrows():
        imo = row['IMO']
        er = row['ER']
        std = row['IMO_Std']
        chikou = row.get('S_Chikou', 0.0)
        entropy = row.get('Entropy', 0.0)
        close = row['close']
        cloud_a = row['senkou_span_a']
        cloud_b = row['senkou_span_b']

        if pd.isna(imo) or pd.isna(er) or pd.isna(std) or pd.isna(entropy):
            signals.append(pos)
            regimes.append('Neutral' if pos == 0 else 'Positioned')
            continue

        threshold = std * t_entry

        if pos > 0:
            hold_days += 1
        else:
            hold_days = 0

        can_exit = hold_days >= min_hold_days

        if pos == 0.0:
            # ENTRY: requires IMO above adaptive threshold AND sufficient ER AND low entropy
            cloud_min = np.minimum(cloud_a, cloud_b) if (not pd.isna(cloud_a) and not pd.isna(cloud_b)) else (cloud_a if not pd.isna(cloud_a) else (cloud_b if not pd.isna(cloud_b) else np.nan))
            gate_pass = True
            if not pd.isna(cloud_min):
                gate_pass = (close >= cloud_min)

            if imo > threshold and er > er_entry and entropy < entropy_thresh and gate_pass:
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
            # EXIT: Early exit if momentum drops (S_Chikou) OR macro trend dies
            exit_signal = False
            if can_exit:
                cloud_max = np.maximum(cloud_a, cloud_b) if (not pd.isna(cloud_a) and not pd.isna(cloud_b)) else np.nan
                is_above_cloud = (not pd.isna(cloud_max) and close >= cloud_max)
                
                roc_gate = row.get('roc_gate', 0.0)
                is_not_crashing = (roc_gate >= roc_gate_limit)
                
                is_immune = (imo >= immunity_thresh)
                if is_above_cloud and is_not_crashing:
                    is_immune = is_immune or (imo >= imo_min_limit)
                
                current_macro_exit_th = 0.0
                if is_above_cloud and is_not_crashing:
                    current_macro_exit_th = imo_exit_bull
                
                if chikou < chikou_thresh and not is_immune:
                    exit_signal = True
                elif imo < current_macro_exit_th:
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
            else:
                intent = None
                confirm_count = 0

        signals.append(pos)

        if pos == 1.0:
            regime = 'Strong Bull' if imo > threshold else 'Weak Bull'
        else:
            regime = 'Neutral'
        regimes.append(regime)

    df['Pos'] = signals
    df['Regime'] = regimes
    return df

def compute_ichimoku_metrics(df: pd.DataFrame, prices: pd.Series) -> dict:
    """Compute trading metrics for Ichimoku signals."""
    positions = df['Pos']
    returns = prices.pct_change()
    strategy_returns = returns * positions.shift(1)
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) == 0:
        return {'cagr': 0, 'sharpe': 0, 'sortino': 0, 'calmar': 0, 'max_dd': 0, 'n_trades': 0, 'win_rate': 0}

    equity = (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25

    cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
    downside = strategy_returns[strategy_returns < 0]
    sortino = strategy_returns.mean() / downside.std() * np.sqrt(365) if len(downside) > 0 and downside.std() > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # Count trades and win rate
    changes = positions.diff().fillna(0)
    n_trades = (changes.abs() > 0).sum() // 2
    
    in_position = False
    hold_start = None
    trade_returns = []
    
    for i, (date, pos) in enumerate(positions.items()):
        if pos == 1.0 and not in_position:
            in_position = True
            hold_start = date
            entry_price = prices.loc[date]
        elif pos == 0.0 and in_position:
            in_position = False
            if hold_start is not None:
                exit_price = prices.loc[date]
                trade_ret = (exit_price - entry_price) / entry_price
                trade_returns.append(trade_ret)
    
    winning = sum(1 for r in trade_returns if r > 0)
    total = len(trade_returns)
    win_rate = winning / total * 100 if total > 0 else 0

    return {
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'calmar': round(calmar, 2),
        'max_dd': round(max_dd * 100, 2),
        'n_trades': n_trades,
        'win_rate': round(win_rate, 1),
        'equity': equity
    }
