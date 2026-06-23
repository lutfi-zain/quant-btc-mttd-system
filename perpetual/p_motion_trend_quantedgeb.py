import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path to import indicators_helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indicators_helper import *

def p_motion_trend_quantedgeb(
    df: pd.DataFrame,
    ema_len: int = 21,
    sd_length: int = 30,
    mult_sdup: float = 1.5,
    mult_sddn: float = 1.5,
    dema_len: int = 7,
    prc_len: int = 2
) -> pd.DataFrame:
    """
    Python translation of P-Motion Trend | QuantEdgeB indicator.
    """
    close = df['close']
    
    # Calculate DEMA
    dema_val = dema(close, dema_len)
    
    # Calculate prc_smooth (rolling median of DEMA)
    prc_smooth = dema_val.rolling(window=prc_len, min_periods=1).median()
    
    # Calculate filter_SD
    filter_sd = stdev(dema_val, sd_length)
    
    # Calculate EMA of prc_smooth
    ema_val = ema(prc_smooth, ema_len)
    
    # Upper and Lower envelopes
    long_e = ema_val + filter_sd * mult_sdup
    short_e = ema_val - filter_sd * mult_sddn
    
    # Signals
    long_c = close > long_e
    short_c = close < short_e
    
    # Bar-by-bar state loop
    qb_arr = np.zeros(len(df))
    qb = 0
    for i in range(len(df)):
        l_c = long_c.iloc[i]
        s_c = short_c.iloc[i]
        
        if pd.isna(l_c):
            l_c = False
        if pd.isna(s_c):
            s_c = False
            
        if l_c and not s_c:
            qb = 1
        elif s_c:
            qb = -1
        qb_arr[i] = qb
        
    pl_arr = np.where(qb_arr > 0, 1, np.where(qb_arr < 0, -1, 0))
    
    # Extra plots
    plotline1 = ema(close, 3)
    plotline2 = ema(prc_smooth, ema_len)
    plotline3 = ema(prc_smooth, ema_len * 2)
    plotline4 = ema(prc_smooth, 10)
    
    return pd.DataFrame({
        'dema': dema_val,
        'prc_smooth': prc_smooth,
        'filter_sd': filter_sd,
        'ema': ema_val,
        'long_e': long_e,
        'short_e': short_e,
        'qb': qb_arr,
        'pl': pl_arr,
        'plotline1': plotline1,
        'plotline2': plotline2,
        'plotline3': plotline3,
        'plotline4': plotline4
    }, index=df.index)
