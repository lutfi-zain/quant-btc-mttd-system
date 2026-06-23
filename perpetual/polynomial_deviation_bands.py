import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path to import indicators_helper
sys.path.append(str(Path(__file__).resolve().parents[1]))
from indicators_helper import *

def frama_calc(src: pd.Series, period: int) -> pd.Series:
    n = period if period % 2 == 0 else period + 1
    n2_half = int(n / 2)
    
    hh = src.rolling(window=n, min_periods=1).max()
    ll = src.rolling(window=n, min_periods=1).min()
    
    hh1 = src.rolling(window=n2_half, min_periods=1).max()
    ll1 = src.rolling(window=n2_half, min_periods=1).min()
    
    hh2 = src.shift(n2_half).rolling(window=n2_half, min_periods=1).max()
    ll2 = src.shift(n2_half).rolling(window=n2_half, min_periods=1).min()
    
    n1 = (hh1 - ll1) / n2_half
    n2 = (hh2 - ll2) / n2_half
    n3 = (hh - ll) / n
    
    frama_val = pd.Series(index=src.index, dtype=float)
    n1_val = n1.values
    n2_val = n2.values
    n3_val = n3.values
    src_val = src.values
    
    curr_frama = np.nan
    for i in range(len(src)):
        if i < n:
            curr_frama = src_val[i]
            frama_val.iloc[i] = curr_frama
            continue
            
        v1 = n1_val[i]
        v2 = n2_val[i]
        v3 = n3_val[i]
        
        if v3 > 0 and (v1 + v2) > 0:
            d = (np.log(v1 + v2) - np.log(v3)) / np.log(2.0)
        else:
            d = 1.0
            
        a = np.exp(-4.6 * (d - 1.0))
        a = np.max([0.01, np.min([1.0, a])])
        
        if np.isnan(curr_frama):
            curr_frama = src_val[i]
        else:
            curr_frama = a * src_val[i] + (1.0 - a) * curr_frama
        frama_val.iloc[i] = curr_frama
    return frama_val

def quantile_dev_calc(src: pd.Series, period: int) -> pd.Series:
    res = pd.Series(index=src.index, dtype=float)
    src_vals = src.values
    q90_idx = int(np.round(period * 0.75))
    q10_idx = int(np.round(period * 0.25))
    q90_idx = np.max([0, np.min([period - 1, q90_idx])])
    q10_idx = np.max([0, np.min([period - 1, q10_idx])])
    
    for i in range(len(src)):
        if i < period - 1:
            res.iloc[i] = np.nan
            continue
        window = np.sort(src_vals[i - period + 1 : i + 1])
        res.iloc[i] = window[q90_idx] - window[q10_idx]
    return res

def calc_kaufman_dev(src: pd.Series, period: int) -> pd.Series:
    change = (src - src.shift(period)).abs()
    diff = src.diff().abs()
    volatility = diff.rolling(window=period, min_periods=1).sum()
    
    er = np.where(volatility == 0, 0.0, change / volatility)
    fastSC = 2.0 / 3.0
    slowSC = 2.0 / 31.0
    sc = (er * (fastSC - slowSC) + slowSC) ** 2
    
    ema_src = ema(src, period)
    dev = (src - ema_src).abs()
    
    kama_dev = np.zeros(len(src))
    dev_vals = dev.values
    
    curr = np.nan
    for i in range(len(src)):
        d_val = dev_vals[i]
        if np.isnan(curr):
            curr = d_val
        else:
            curr = curr + sc[i] * (d_val - curr)
        kama_dev[i] = curr
    return pd.Series(kama_dev, index=src.index)

def calc_gaussian_dev(src: pd.Series, period: int) -> pd.Series:
    weights = np.zeros(period)
    sumw = 0.0
    for i in range(period):
        w = np.exp(-((i / period) / 2.0) ** 2)
        weights[i] = w
        sumw += w
        
    res = pd.Series(index=src.index, dtype=float)
    src_vals = src.values
    for i in range(len(src)):
        if i < period - 1:
            res.iloc[i] = np.nan
            continue
        window = src_vals[i - period + 1 : i + 1][::-1]
        wmean = np.sum(window * weights) / sumw
        wvar = np.sum(((window - wmean) ** 2) * weights) / sumw
        res.iloc[i] = np.sqrt(wvar)
    return res

def get_poly_fit_val(y_vals, degree):
    x = np.arange(len(y_vals))
    try:
        # np.polyfit returns coefficients in descending order of power.
        # Evaluating at x = 0 is equivalent to taking the constant term (coeffs[-1]).
        coeffs = np.polyfit(x, y_vals, degree)
        return coeffs[-1]
    except Exception:
        return np.nan

def polynomial_deviation_bands(df: pd.DataFrame,
                               deg: str = "2nd",
                               source_col: str = "close",
                               regressions_length: int = 14,
                               dev_type: str = "Standard Deviation",
                               dev_lookback: int = 20,
                               multiplier: float = 1.5) -> pd.DataFrame:
    
    src = df[source_col]
    n = len(df)
    
    # Map degree string to int
    deg_map = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
    degree = deg_map.get(deg, 2)
    
    # Calculate Polynomial regression value
    reg_vals = np.zeros(n)
    src_vals = src.values
    
    for i in range(n):
        if i < regressions_length - 1:
            reg_vals[i] = np.nan
            continue
        # Window of src from oldest to newest
        y_vals = src_vals[i - regressions_length + 1 : i + 1]
        reg_vals[i] = get_poly_fit_val(y_vals, degree)
        
    reg_val_series = pd.Series(reg_vals, index=df.index)
    
    # Calculate Deviations
    if dev_type == "Standard Deviation":
        dev_val = stdev(reg_val_series, dev_lookback)
    elif dev_type == "Mean Absolute Deviation":
        mean = sma(reg_val_series, dev_lookback)
        dev_val = sma((reg_val_series - mean).abs(), dev_lookback)
    elif dev_type == "Median Absolute Deviation":
        dev_val = pd.Series(index=df.index, dtype=float)
        for i in range(n):
            if i < dev_lookback - 1:
                dev_val.iloc[i] = np.nan
                continue
            window = reg_vals[i - dev_lookback + 1 : i + 1]
            med = np.median(window)
            dev_val.iloc[i] = np.median(np.abs(window - med))
    elif dev_type == "Exponential Deviation":
        ema_val = ema(reg_val_series, dev_lookback)
        dev_val = ema((reg_val_series - ema_val).abs(), dev_lookback)
    elif dev_type == "True Range Deviation":
        tr_val = tr(df['high'], df['low'], df['close'])
        dev_val = stdev(tr_val, dev_lookback)
    elif dev_type == "Hull Deviation":
        hma_val = hma(reg_val_series, dev_lookback)
        dev_val = (reg_val_series - hma_val).abs()
    elif dev_type == "Frama Deviation":
        frama_val = frama_calc(reg_val_series, dev_lookback)
        dev_val = (reg_val_series - frama_val).abs()
    elif dev_type == "Kauffman Adaptive Deviation":
        dev_val = calc_kaufman_dev(reg_val_series, dev_lookback)
    elif dev_type == "Gaussian Deviation":
        dev_val = calc_gaussian_dev(reg_val_series, dev_lookback)
    elif dev_type == "Quantile Deviation":
        dev_val = quantile_dev_calc(reg_val_series, dev_lookback)
    else:
        dev_val = stdev(reg_val_series, dev_lookback)
        
    upper_band = reg_val_series + (dev_val * multiplier)
    lower_band = reg_val_series - (dev_val * multiplier)
    
    # Stateful trending logic
    trend = np.zeros(n)
    curr_trend = 0
    close_vals = df['close'].values
    
    for i in range(n):
        u_b = upper_band.iloc[i]
        l_b = lower_band.iloc[i]
        if np.isnan(u_b) or np.isnan(l_b):
            trend[i] = 0.0
            continue
        if close_vals[i] > u_b:
            curr_trend = 1
        elif close_vals[i] < l_b:
            curr_trend = -1
        trend[i] = curr_trend
        
    results = pd.DataFrame(index=df.index)
    results['reg_val'] = reg_val_series
    results['upper_band'] = upper_band
    results['lower_band'] = lower_band
    results['trend'] = trend
    
    return results
