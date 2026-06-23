import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from indicators_helper import *

def dft(x, y, Nx, direction=1):
    """Discrete Fourier Transform matching PineScript implementation."""
    x_arr = x.copy()
    y_arr = y.copy()
    x_out = np.zeros(Nx)
    y_out = np.zeros(Nx)
    
    for i in range(Nx):
        x_i = 0.0
        y_i = 0.0
        kx = i / Nx
        arg = -direction * 2 * np.pi * kx
        for k in range(Nx):
            cos_val = np.cos(k * arg)
            sin_val = np.sin(k * arg)
            x_i += x[k] * cos_val - y[k] * sin_val
            y_i += x[k] * sin_val + y[k] * cos_val
        x_out[i] = x_i
        y_out[i] = y_i
    
    if direction == 1:
        x_out = x_out / Nx
        y_out = y_out / Nx
    
    return x_out, y_out

def fourier_for_loop(
    df: pd.DataFrame,
    n: int = 1,
    start: int = 1,
    end: int = 45,
    upper: int = 40,
    lower: int = -10
) -> pd.DataFrame:
    xval = (df['high'] + df['low'] + df['close']) / 3.0
    
    subject_vals = np.empty(len(df))
    subject_vals[:] = np.nan
    
    for idx in range(len(df)):
        if idx < n - 1:
            continue
        # Build input arrays for DFT
        x_in = np.zeros(n)
        y_in = np.zeros(n)
        for i in range(n):
            x_in[i] = xval.iloc[idx - n + 1 + i]
            y_in[i] = 0.0
        # Apply DFT
        x_out, y_out = dft(x_in, y_in, n, direction=1)
        # Compute magnitude of DC component (index 0)
        mag_0 = np.sqrt(x_out[0]**2 + y_out[0]**2)
        subject_vals[idx] = mag_0
        
    subject_series = pd.Series(subject_vals, index=df.index)
    
    score = np.zeros(len(df))
    for idx in range(len(df)):
        total = 0.0
        val = subject_series.iloc[idx]
        if pd.isna(val):
            score[idx] = np.nan
            continue
            
        has_nans = False
        for i in range(start, end + 1):
            if idx - i < 0:
                has_nans = True
                break
            prev_val = subject_series.iloc[idx - i]
            if pd.isna(prev_val):
                has_nans = True
                break
            total += 1.0 if val > prev_val else -1.0
            
        if has_nans:
            score[idx] = np.nan
        else:
            score[idx] = total
            
    score_series = pd.Series(score, index=df.index)
    
    l_cond = score_series > upper
    s_cond = crossunder(score_series, pd.Series(lower, index=df.index))
    
    out_vals = np.zeros(len(df))
    out_val = 0
    for i in range(len(df)):
        if l_cond.iloc[i] and not s_cond.iloc[i]:
            out_val = 1
        elif s_cond.iloc[i]:
            out_val = -1
        out_vals[i] = out_val
        
    return pd.DataFrame({
        'score': score_series,
        'out': out_vals
    }, index=df.index)
