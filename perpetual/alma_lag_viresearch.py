import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path to import indicators_helper
sys.path.append(str(Path(__file__).resolve().parents[1]))
from indicators_helper import *

def alma_calc(src: pd.Series, window_size: int, offset_ratio: float = 0.85, sigma_divisor: float = 6.0) -> pd.Series:
    offset = int(np.floor((window_size - 1) * offset_ratio))
    sigma = window_size / sigma_divisor
    
    weights = np.zeros(window_size)
    sum_w = 0.0
    for i in range(window_size):
        w = np.exp(-((i - offset) ** 2) / (2.0 * (sigma ** 2)))
        weights[i] = w
        sum_w += w
        
    def calc_alma(y):
        return np.dot(y, weights) / sum_w
        
    return src.rolling(window=window_size, min_periods=window_size).apply(calc_alma, raw=True)

def alma_lag_viresearch(df: pd.DataFrame,
                        len_subject: int = 78) -> pd.DataFrame:
    
    n = len(df)
    
    # alma calculation
    alma_val = alma_calc(df['close'], len_subject, 0.85, 6.0)
    
    # Stateful trend vii
    vii = np.zeros(n)
    curr_vii = 0.0
    
    close_vals = df['close'].values
    alma_vals = alma_val.values
    
    for i in range(n):
        if i == 0 or np.isnan(alma_vals[i]) or np.isnan(alma_vals[i-1]):
            vii[i] = 0.0
            continue
            
        almal = (alma_vals[i] > alma_vals[i-1]) and (close_vals[i] > alma_vals[i])
        almas = (alma_vals[i] < alma_vals[i-1]) and (close_vals[i] < alma_vals[i])
        
        L = almal
        S = almas
        
        if L and not S:
            curr_vii = 1.0
        elif S:
            curr_vii = -1.0
            
        vii[i] = curr_vii
        
    results = pd.DataFrame(index=df.index)
    results['alma'] = alma_val
    results['vii'] = vii
    
    return results
