import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path to import indicators_helper
sys.path.append(str(Path(__file__).resolve().parents[1]))
from indicators_helper import *

def dema_adjusted_average_true_range(df: pd.DataFrame, 
                                     show_atr: bool = True, 
                                     ha_candles: bool = False, 
                                     period_dema: int = 7, 
                                     source_dema: str = 'close', 
                                     period_atr: int = 14, 
                                     factor_atr: float = 1.7, 
                                     paint_candles: bool = False, 
                                     show_ma: bool = False, 
                                     moving_average_type: str = "Ema", 
                                     moving_average_period: int = 50) -> pd.DataFrame:
    
    # Calculate Heikin-Ashi Close if selected
    if ha_candles:
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
        ha_open = pd.Series(index=df.index, dtype=float)
        if len(df) > 0:
            ha_open.iloc[0] = (df['open'].iloc[0] + df['close'].iloc[0]) / 2.0
            for i in range(1, len(df)):
                ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2.0
        source = ha_close
    else:
        source = df[source_dema]

    # Calculate DEMA
    ema1 = ema(source, period_dema)
    ema2 = ema(ema1, period_dema)
    dema_out = 2 * ema1 - ema2

    # Calculate ATR based on standard chart high/low/close
    atr_val = atr(df['high'], df['low'], df['close'], period_atr)
    true_range = atr_val * factor_atr

    # Stateful bar-by-bar calculation for DemaAtr
    dema_atr_vals = np.zeros(len(df))
    dema_out_vals = dema_out.values
    true_range_vals = true_range.values

    for i in range(len(df)):
        if i == 0:
            dema_atr_vals[i] = dema_out_vals[i]
        else:
            prev_dema = dema_atr_vals[i-1]
            if np.isnan(prev_dema):
                prev_dema = dema_out_vals[i]
            
            d_out = dema_out_vals[i]
            t_range = true_range_vals[i]
            
            if np.isnan(d_out) or np.isnan(t_range):
                dema_atr_vals[i] = prev_dema
                continue
                
            true_range_upper = d_out + t_range
            true_range_lower = d_out - t_range
            
            current_dema = prev_dema
            if true_range_lower > current_dema:
                current_dema = true_range_lower
            if true_range_upper < current_dema:
                current_dema = true_range_upper
            
            dema_atr_vals[i] = current_dema

    dema_atr = pd.Series(dema_atr_vals, index=df.index)

    # Calculate Moving Average switch
    def moving_average(src: pd.Series, length: int, ma_type: str) -> pd.Series:
        ma_type = ma_type.upper()
        if ma_type == "SMA":
            return sma(src, length)
        elif ma_type == "HULL":
            return hma(src, length)
        elif ma_type == "EMA":
            return ema(src, length)
        elif ma_type == "WMA":
            return wma(src, length)
        elif ma_type == "DEMA":
            return dema(src, length)
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")

    ma_out = moving_average(dema_atr, moving_average_period, moving_average_type)

    results = pd.DataFrame(index=df.index)
    results['dema_atr'] = dema_atr
    results['moving_average'] = ma_out

    return results
