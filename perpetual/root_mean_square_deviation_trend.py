import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path to import indicators_helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indicators_helper import *

def tema(source: pd.Series, length: int) -> pd.Series:
    ema1 = ema(source, length)
    ema2 = ema(ema1, length)
    ema3 = ema(ema2, length)
    return 3.0 * ema1 - 3.0 * ema2 + ema3

def calculate_frama(df: pd.DataFrame, source: pd.Series, length: int) -> pd.Series:
    if length % 2 != 0:
        length = length + 1
    half_len = length // 2
    
    high = df['high']
    low = df['low']
    
    hh1 = high.rolling(window=half_len, min_periods=half_len).max()
    ll1 = low.rolling(window=half_len, min_periods=half_len).min()
    n1 = (hh1 - ll1) / half_len
    
    hh2 = high.shift(half_len).rolling(window=half_len, min_periods=half_len).max()
    ll2 = low.shift(half_len).rolling(window=half_len, min_periods=half_len).min()
    n2 = (hh2 - ll2) / half_len
    
    hh3 = high.rolling(window=length, min_periods=length).max()
    ll3 = low.rolling(window=length, min_periods=length).min()
    n3 = (hh3 - ll3) / length
    
    frama_val = pd.Series(index=source.index, dtype=float)
    
    current_frama = np.nan
    for i in range(len(source)):
        n1_v = n1.iloc[i]
        n2_v = n2.iloc[i]
        n3_v = n3.iloc[i]
        src_v = source.iloc[i]
        
        if np.isnan(n1_v) or np.isnan(n2_v) or np.isnan(n3_v):
            frama_val.iloc[i] = src_v
            current_frama = src_v
            continue
            
        dim = 0.0
        if (n1_v + n2_v) > 0 and n3_v > 0:
            dim = (np.log(n1_v + n2_v) - np.log(n3_v)) / np.log(2.0)
            
        alpha = np.exp(-4.6 * (dim - 1.0))
        alpha = max(0.01, min(1.0, alpha))
        
        if np.isnan(current_frama):
            current_frama = src_v
        else:
            current_frama = alpha * src_v + (1.0 - alpha) * current_frama
            
        frama_val.iloc[i] = current_frama
        
    return frama_val

def get_source_series(df: pd.DataFrame, src_type: str) -> pd.Series:
    if src_type == "open":
        return df['open']
    elif src_type == "high":
        return df['high']
    elif src_type == "low":
        return df['low']
    elif src_type == "close":
        return df['close']
    elif src_type == "oc2":
        return (df['open'] + df['close']) / 2.0
    elif src_type == "hl2":
        return (df['high'] + df['low']) / 2.0
    elif src_type == "occ3":
        return (df['open'] + df['close'] + df['close']) / 3.0
    elif src_type == "hlc3":
        return (df['high'] + df['low'] + df['close']) / 3.0
    elif src_type == "ohlc4":
        return (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    elif src_type == "hlcc4":
        return (df['high'] + df['low'] + df['close'] + df['close']) / 4.0
    else:
        return df['close']

def root_mean_square_deviation_trend(
    df: pd.DataFrame,
    input_src: str = "close",
    avg_type: str = "SMA",
    length: int = 28,
    mult: float = 1.0,
    intrabar: bool = True
) -> pd.DataFrame:
    """
    Python translation of RMSD Trend [InvestorUnknown] indicator.
    """
    src = get_source_series(df, input_src)
    
    # Calculate avg based on avg_type
    if avg_type == "SMA":
        avg = sma(src, length)
    elif avg_type == "EMA":
        avg = ema(src, length)
    elif avg_type == "HMA":
        avg = hma(src, length)
    elif avg_type == "DEMA":
        avg = dema(src, length)
    elif avg_type == "TEMA":
        avg = tema(src, length)
    elif avg_type == "RMA":
        avg = rma(src, length)
    elif avg_type == "FRAMA":
        avg = calculate_frama(df, src, length)
    else:
        avg = sma(src, length)
        
    # Calculate custom RMSD
    # rmsd = np.sqrt( ((src - avg)**2).rolling(window=length, min_periods=length).mean() )
    rmsd_series = np.sqrt(((src - avg) ** 2).rolling(window=length, min_periods=1).mean())
    
    avg_p = avg + (rmsd_series * mult)
    avg_m = avg - (rmsd_series * mult)
    
    # Bar-by-bar loop for direction
    direction_arr = np.zeros(len(df), dtype=int)
    direction = 0
    
    for i in range(len(df)):
        src_v = src.iloc[i]
        avg_p_v = avg_p.iloc[i]
        avg_m_v = avg_m.iloc[i]
        
        src_prev = src.iloc[i-1] if i > 0 else np.nan
        avg_p_prev = avg_p.iloc[i-1] if i > 0 else np.nan
        avg_m_prev = avg_m.iloc[i-1] if i > 0 else np.nan
        
        # Check crossover(src, avg_p)
        crossover_val = (src_v > avg_p_v) and (src_prev <= avg_p_prev) if (not np.isnan(src_prev) and not np.isnan(avg_p_prev)) else False
        # Check crossunder(src, avg_m)
        crossunder_val = (src_v < avg_m_v) and (src_prev >= avg_m_prev) if (not np.isnan(src_prev) and not np.isnan(avg_m_prev)) else False
        
        if crossover_val:
            direction = 1
        elif crossunder_val:
            direction = -1
            
        direction_arr[i] = direction
        
    dir_series = pd.Series(direction_arr, index=df.index)
    
    # Calculate plotted versions (shifting if not intrabar)
    if intrabar:
        avg_plot = avg
        avg_p_plot = avg_p
        avg_m_plot = avg_m
        dir_plot = dir_series
    else:
        avg_plot = avg.shift(1)
        avg_p_plot = avg_p.shift(1)
        avg_m_plot = avg_m.shift(1)
        dir_plot = dir_series.shift(1)
        
    # candle_h_l calculation
    high = df['high']
    low = df['low']
    open_s = df['open']
    close_s = df['close']
    
    candle_h_l = np.where(src > avg_p_plot, high, np.where(src < avg_m_plot, low, (open_s + close_s) / 2.0))
    
    return pd.DataFrame({
        'direction': dir_series,
        'avg': avg,
        'avg_p': avg_p,
        'avg_m': avg_m,
        'direction_plot': dir_plot,
        'avg_plot': avg_plot,
        'avg_p_plot': avg_p_plot,
        'avg_m_plot': avg_m_plot,
        'candle_h_l': candle_h_l
    }, index=df.index)
