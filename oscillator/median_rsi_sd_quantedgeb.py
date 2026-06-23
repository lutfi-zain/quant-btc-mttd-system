import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from indicators_helper import *

def percentile_nearest_rank(source: pd.Series, length: int, percentage: float) -> pd.Series:
    rank = int(np.ceil((percentage / 100.0) * length))
    idx = rank - 1
    
    def calc_percentile(window):
        if len(window) < length or np.any(np.isnan(window)):
            return np.nan
        sorted_window = np.sort(window)
        return sorted_window[idx]
        
    return source.rolling(window=length, min_periods=length).apply(calc_percentile, raw=True)

def median_rsi_sd_quantedgeb(
    df: pd.DataFrame,
    len_rsi: int = 21,
    len_median: int = 10,
    lu: float = 65.0,
    su: float = 45.0
) -> pd.DataFrame:
    close = df['close']
    median_val = percentile_nearest_rank(close, len_median, 50.0)
    rsi_val = rsi(median_val, len_rsi)
    
    sd_val = stdev(median_val, len_median)
    up_sd = median_val + sd_val
    dn_sd = median_val - sd_val
    
    median_s = close < up_sd
    
    long_c = (rsi_val > lu) & (~median_s)
    short_c = rsi_val < su
    
    qb = 0
    qb_vals = np.zeros(len(df))
    for i in range(len(df)):
        if long_c.iloc[i] and not short_c.iloc[i]:
            qb = 1
        elif short_c.iloc[i]:
            qb = -1
        qb_vals[i] = qb
        
    return pd.DataFrame({
        'rsi': rsi_val,
        'qb': qb_vals
    }, index=df.index)
