import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path to import indicators_helper
sys.path.append(str(Path(__file__).resolve().parents[1]))
from indicators_helper import *

def gaussian_smooth_trend_quantedgeb(df: pd.DataFrame,
                                     col_mode: str = "Strategy",
                                     label_enabled: bool = False,
                                     source_col: str = 'close',
                                     len_dema: int = 7,
                                     len_fg: int = 4,
                                     sigma_fg: float = 2.0,
                                     len_s: int = 12,
                                     len_sd: int = 30,
                                     mult_sdup: float = 2.5,
                                     mult_sddn: float = 1.8) -> pd.DataFrame:
    
    src = df[source_col]
    n = len(df)
    
    # 1. DEMA
    dema_val = dema(src, len_dema)
    
    # 2. Gaussian Filter
    # F_Gaussian weights
    weights = np.zeros(len_fg)
    sum_w = 0.0
    for i in range(len_fg):
        w = np.exp(-0.5 * (((i - (len_fg - 1) / 2.0) / sigma_fg) ** 2))
        weights[i] = w
        sum_w += w
        
    filter_gaussian = np.zeros(n)
    dema_vals = dema_val.values
    
    for t in range(n):
        if t < len_fg - 1:
            filter_gaussian[t] = dema_vals[t]
            continue
            
        # weighted sum over last len_fg bars
        w_sum = 0.0
        for i in range(len_fg):
            val = dema_vals[t - i]
            if np.isnan(val):
                val = 0.0
            w_sum += val * weights[i]
            
        filter_gaussian[t] = w_sum / sum_w
        
    filter_gaussian_series = pd.Series(filter_gaussian, index=df.index)
    
    # 3. SMMA Calculation
    # F_SMMA(filter_Gaussian, len_S)
    # dema_s = F_DEMA(src_s, len_s)
    dema_s = dema(filter_gaussian_series, len_s)
    dema_s_vals = dema_s.values
    
    smma_vals = np.zeros(n)
    
    for i in range(n):
        if i < len_s - 1:
            smma_vals[i] = filter_gaussian[i]
        elif i == len_s - 1:
            if not np.isnan(dema_s_vals[i]):
                smma_vals[i] = dema_s_vals[i]
            else:
                smma_vals[i] = filter_gaussian[i]
        else:
            alpha = 1.0 / len_s
            prev_smma = smma_vals[i-1]
            smma_vals[i] = (filter_gaussian[i] - prev_smma) * alpha + prev_smma
            
    smma_series = pd.Series(smma_vals, index=df.index)
    
    # 4. SD Filter
    filter_sd = stdev(smma_series, len_sd)
    
    long_v = smma_series + filter_sd * mult_sdup
    short_v = smma_series - filter_sd * mult_sddn
    
    # 5. Final Signal QB
    qb_vals = np.zeros(n)
    curr_qb = 0.0
    close_vals = df['close'].values
    long_v_vals = long_v.values
    short_v_vals = short_v.values
    
    for i in range(n):
        l_v = long_v_vals[i]
        s_v = short_v_vals[i]
        
        if np.isnan(l_v) or np.isnan(s_v):
            qb_vals[i] = 0.0
            continue
            
        long_c = close_vals[i] > l_v
        short_c = close_vals[i] < s_v
        
        if long_c and not short_c:
            curr_qb = 1.0
        elif short_c:
            curr_qb = -1.0
            
        qb_vals[i] = curr_qb
        
    results = pd.DataFrame(index=df.index)
    results['dema'] = dema_val
    results['filter_gaussian'] = filter_gaussian_series
    results['smma'] = smma_series
    results['long_v'] = long_v
    results['short_v'] = short_v
    results['qb'] = pd.Series(qb_vals, index=df.index)
    
    return results
