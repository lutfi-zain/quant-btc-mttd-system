import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path to import indicators_helper
sys.path.append(str(Path(__file__).resolve().parents[1]))
from indicators_helper import *

def adaptive_regime_cloud(df: pd.DataFrame,
                          lookback: int = 50,
                          adaptive_period: int = 30,
                          volatility_period: int = 10,
                          cloud_expansion: float = 1.6,
                          regime_threshold: float = 0.65,
                          fast_response: bool = True) -> pd.DataFrame:
    
    n = len(df)
    close_vals = df['close'].values
    
    # Calculate log returns
    log_returns = np.zeros(n)
    for i in range(1, n):
        if close_vals[i-1] > 0 and close_vals[i] > 0:
            log_returns[i] = np.log(close_vals[i] / close_vals[i-1])
        else:
            log_returns[i] = 0.0

    # Hurst exponent calculation pre-allocation
    hurst = np.full(n, 0.5)
    
    for i in range(n):
        if i < lookback:
            hurst[i] = 0.5
            continue
        
        # logReturns window from i - lookback + 1 to i
        window = log_returns[i - lookback + 1 : i + 1]
        mean_ret = np.mean(window)
        
        # Cumulative deviations
        cum_dev = 0.0
        cum_devs = []
        for val in window:
            cum_dev += (val - mean_ret)
            cum_devs.append(cum_dev)
        
        R = np.max(cum_devs) - np.min(cum_devs)
        
        # Standard deviation (S)
        S = np.std(window, ddof=0)
        
        if S > 0 and R > 0:
            H = np.log(R / S) / np.log(lookback)
        else:
            H = 0.5
            
        hurst[i] = np.max([0.3, np.min([0.7, H])])

    # Volatility pre-calculation using rolling window standard deviation of log returns
    # volatility = ta.stdev(logReturns, volatilityPeriod)
    log_returns_series = pd.Series(log_returns)
    volatility = stdev(log_returns_series, volatility_period).values

    # Output arrays
    midline = np.zeros(n)
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    adaptive_alphas = np.zeros(n)
    
    in_long_position = np.zeros(n, dtype=bool)
    in_short_position = np.zeros(n, dtype=bool)
    long_signals = np.zeros(n, dtype=bool)
    short_signals = np.zeros(n, dtype=bool)

    # Initialize position variables
    curr_long = False
    curr_short = False
    curr_midline = np.nan

    base_alpha = 2.0 / (adaptive_period + 1.0)

    for i in range(n):
        h_val = hurst[i]
        is_trending = h_val > regime_threshold
        is_mean_reverting = h_val < (1.0 - regime_threshold)
        
        # Calculate alpha
        if is_trending:
            regime_multiplier = 4.0 if fast_response else 1.5
        elif is_mean_reverting:
            regime_multiplier = 0.2 if fast_response else 0.7
        else:
            regime_multiplier = 1.5 if fast_response else 1.0
            
        adaptive_alpha = np.min([0.9, base_alpha * regime_multiplier])
        adaptive_alphas[i] = adaptive_alpha
        
        # Midline (Adaptive EMA)
        c_val = close_vals[i]
        if np.isnan(curr_midline) or i == 0:
            curr_midline = c_val
        else:
            curr_midline = curr_midline * (1.0 - adaptive_alpha) + c_val * adaptive_alpha
        midline[i] = curr_midline
        
        # Cloud width and bands
        vol = volatility[i]
        if np.isnan(vol):
            vol = 0.0
            
        base_width = vol * cloud_expansion
        
        if is_trending:
            width_multiplier = 1.5 if fast_response else 1.2
        elif is_mean_reverting:
            width_multiplier = 0.5 if fast_response else 0.7
        else:
            width_multiplier = 1.0
            
        cloud_width = base_width * width_multiplier * c_val
        upper_band[i] = curr_midline + cloud_width
        lower_band[i] = curr_midline - cloud_width
        
        # Signals logic (needs current and previous values)
        long_sig = False
        short_sig = False
        
        if i > 0:
            close_prev = close_vals[i-1]
            upper_prev = upper_band[i-1]
            lower_prev = lower_band[i-1]
            
            if is_trending:
                long_sig = (c_val > upper_band[i]) and (close_prev <= upper_prev)
                short_sig = (c_val < lower_band[i]) and (close_prev >= lower_prev)
            elif is_mean_reverting:
                long_sig = (c_val < lower_band[i]) and (close_prev >= lower_prev)
                short_sig = (c_val > upper_band[i]) and (close_prev <= upper_prev)
            else:
                long_sig = (c_val > upper_band[i]) and (close_prev <= upper_prev)
                short_sig = (c_val < lower_band[i]) and (close_prev >= lower_prev)

        long_signals[i] = long_sig
        short_signals[i] = short_sig
        
        # Update position states
        if long_sig:
            curr_long = True
            curr_short = False
        elif short_sig:
            curr_long = False
            curr_short = True
            
        in_long_position[i] = curr_long
        in_short_position[i] = curr_short

    results = pd.DataFrame(index=df.index)
    results['midline'] = midline
    results['upper_band'] = upper_band
    results['lower_band'] = lower_band
    results['hurst'] = hurst
    results['volatility'] = volatility
    results['adaptive_alpha'] = adaptive_alphas
    results['in_long_position'] = in_long_position
    results['in_short_position'] = in_short_position
    results['long_signal'] = long_signals
    results['short_signal'] = short_signals

    return results
