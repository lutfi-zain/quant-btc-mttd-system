import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from indicators_helper import *

def z_score_adaptive_oscillator_suite(
    df: pd.DataFrame,
    use_rsi: bool = True,
    use_cci: bool = True,
    use_cmo: bool = True,
    use_mfi: bool = False,
    rsi_len: int = 35,
    rsi_long_th: float = 55.0,
    rsi_short_th: float = 45.0,
    cci_len: int = 35,
    cci_long_th: float = 100.0,
    cci_short_th: float = -100.0,
    cmo_len: int = 30,
    cmo_long_th: float = 30.0,
    cmo_short_th: float = -30.0,
    mfi_len: int = 30,
    mfi_long_th: float = 55.0,
    mfi_short_th: float = 45.0,
    z_score_len: int = 14,
    threshold: float = 1.5
) -> pd.DataFrame:
    src = df['close']
    
    rsi_vals = rsi(src, rsi_len)
    cci_vals = cci(df['high'], df['low'], src, cci_len)
    cmo_vals = cmo(src, cmo_len)
    mfi_vals = mfi(df['high'], df['low'], src, df['volume'], mfi_len)
    
    rsi_mean = sma(rsi_vals, z_score_len)
    rsi_std = stdev(rsi_vals, z_score_len)
    z_rsi = (rsi_vals - rsi_mean) / rsi_std
    is_high_z_rsi = z_rsi.abs() > threshold
    
    cci_mean = sma(cci_vals, z_score_len)
    cci_std = stdev(cci_vals, z_score_len)
    z_cci = (cci_vals - cci_mean) / cci_std
    is_high_z_cci = z_cci.abs() > threshold
    
    cmo_mean = sma(cmo_vals, z_score_len)
    cmo_std = stdev(cmo_vals, z_score_len)
    z_cmo = (cmo_vals - cmo_mean) / cmo_std
    is_high_z_cmo = z_cmo.abs() > threshold
    
    mfi_mean = sma(mfi_vals, z_score_len)
    mfi_std = stdev(mfi_vals, z_score_len)
    z_mfi = (mfi_vals - mfi_mean) / mfi_std
    is_high_z_mfi = z_mfi.abs() > threshold
    
    rsi_sig = 0
    cci_sig = 0
    cmo_sig = 0
    mfi_sig = 0
    
    trend_sig = 0
    
    trend_sig_vals = np.empty(len(df))
    trend_sig_vals[:] = np.nan
    
    osc_count = (1 if use_rsi else 0) + (1 if use_cci else 0) + (1 if use_cmo else 0) + (1 if use_mfi else 0)
    consensus_strength = np.empty(len(df))
    consensus_strength[:] = np.nan
    
    for idx in range(len(df)):
        if use_rsi:
            r_v = rsi_vals.iloc[idx]
            hz = is_high_z_rsi.iloc[idx]
            if not pd.isna(r_v) and not pd.isna(hz):
                if hz:
                    rsi_sig = 1 if r_v > 50 else -1 if r_v < 50 else rsi_sig
                else:
                    rsi_sig = 1 if r_v > rsi_long_th else -1 if r_v < rsi_short_th else rsi_sig
            
        if use_cci:
            cc_v = cci_vals.iloc[idx]
            hz = is_high_z_cci.iloc[idx]
            if not pd.isna(cc_v) and not pd.isna(hz):
                if hz:
                    cci_sig = 1 if cc_v > 0 else -1 if cc_v < 0 else cci_sig
                else:
                    cci_sig = 1 if cc_v > cci_long_th else -1 if cc_v < cci_short_th else cci_sig
            
        if use_cmo:
            cm_v = cmo_vals.iloc[idx]
            hz = is_high_z_cmo.iloc[idx]
            if not pd.isna(cm_v) and not pd.isna(hz):
                if hz:
                    cmo_sig = 1 if cm_v > 0 else -1 if cm_v < 0 else cmo_sig
                else:
                    cmo_sig = 1 if cm_v > cmo_long_th else -1 if cm_v < cmo_short_th else cmo_sig
            
        if use_mfi:
            mf_v = mfi_vals.iloc[idx]
            hz = is_high_z_mfi.iloc[idx]
            if not pd.isna(mf_v) and not pd.isna(hz):
                if hz:
                    mfi_sig = 1 if mf_v > 50 else -1 if mf_v < 50 else mfi_sig
                else:
                    mfi_sig = 1 if mf_v > mfi_long_th else -1 if mf_v < mfi_short_th else mfi_sig
                    
        sum_sig = 0
        sum_sig += cci_sig if use_cci else 0
        sum_sig += cmo_sig if use_cmo else 0
        sum_sig += mfi_sig if use_mfi else 0
        sum_sig += rsi_sig if use_rsi else 0
        
        if osc_count % 2 == 1:
            trend_sig = 1 if sum_sig > 0 else -1 if sum_sig < 0 else trend_sig
        else:
            trend_sig = trend_sig if sum_sig == 0 else (1 if sum_sig > 0 else -1 if sum_sig < 0 else trend_sig)
            
        trend_sig_vals[idx] = trend_sig
        if osc_count > 0:
            consensus_strength[idx] = sum_sig / osc_count
            
    return pd.DataFrame({
        'consensus_strength': pd.Series(consensus_strength, index=df.index),
        'trend_signal': pd.Series(trend_sig_vals, index=df.index)
    }, index=df.index)
