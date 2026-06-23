import os
import sys
import pandas as pd
import numpy as np

# Add parent directory to path to import indicators_helper
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from indicators_helper import *

def ehma(source: pd.Series, length: int) -> pd.Series:
    return ema(2 * ema(source, int(length / 2)) - ema(source, length), int(np.round(np.sqrt(length))))

def thma(source: pd.Series, length: int) -> pd.Series:
    return wma(wma(source, int(length / 3)) * 3 - wma(source, int(length / 2)) - wma(source, length), length)

def mode_smoothing(mode_switch: str, src: pd.Series, length: int, volume: pd.Series = None) -> pd.Series:
    if mode_switch == 'Hma':
        return hma(src, length)
    elif mode_switch == 'Ehma':
        return ehma(src, length)
    elif mode_switch == 'Thma':
        return thma(src, int(length / 2))
    elif mode_switch == 'SMA':
        return sma(src, length)
    elif mode_switch == 'Ema':
        return ema(src, length)
    elif mode_switch == 'Wma':
        return wma(src, length)
    elif mode_switch == 'Tema':
        return tema(src, length)
    elif mode_switch == 'Vwma':
        return vwma(src, volume, length) if volume is not None else sma(src, length)
    else:
        return pd.Series(np.nan, index=src.index)

def kalman_filtered_rsi_oscillator(
    df: pd.DataFrame, 
    process_noise: float = 0.01, 
    measurement_noise: float = 3.0, 
    n: int = 5, 
    rsi_period: int = 14, 
    smooth: bool = False, 
    mode_switch: str = 'Ema', 
    smoothlen: int = 7
) -> pd.DataFrame:
    
    pricesource = df['close']
    state_estimate = np.full(n, np.nan)
    error_covariance = np.full(n, 100.0)
    
    kalman_filtered_price = np.empty(len(df))
    
    for idx in range(len(df)):
        price = pricesource.iloc[idx]
        if pd.isna(price):
            kalman_filtered_price[idx] = np.nan
            continue
        
        # f_init
        if np.isnan(state_estimate[0]):
            state_estimate[:] = price
            error_covariance[:] = 1.0
            
        # Prediction
        predicted_state_estimate = state_estimate.copy()
        predicted_error_covariance = error_covariance + process_noise
        
        # Update
        for i in range(n):
            kg = predicted_error_covariance[i] / (predicted_error_covariance[i] + measurement_noise)
            state_estimate[i] = predicted_state_estimate[i] + kg * (price - predicted_state_estimate[i])
            error_covariance[i] = (1 - kg) * predicted_error_covariance[i]
            
        kalman_filtered_price[idx] = state_estimate[0]
        
    kalman_filtered_price_series = pd.Series(kalman_filtered_price, index=df.index)
    
    # Calculate RSI
    rsi_val = rsi(kalman_filtered_price_series, rsi_period)
    
    # Optional Smoothing
    if smooth:
        rsi_val = mode_smoothing(mode_switch, rsi_val, smoothlen, df.get('volume'))
        
    # Normalize
    lowest_rsi = lowest(rsi_val, 100)
    highest_rsi = highest(rsi_val, 100)
    normalized_rsi = (rsi_val - lowest_rsi) / (highest_rsi - lowest_rsi) - 0.5
    
    # Direction: 1 (bullish) if normalized RSI > 0 (RSI > 50%), -1 (bearish) if <= 0
    direction = normalized_rsi.apply(lambda x: 1.0 if x > 0 else -1.0 if not pd.isna(x) else 0.0)

    return pd.DataFrame({
        'kalman_rsi': normalized_rsi,
        'direction': direction
    }, index=df.index)
