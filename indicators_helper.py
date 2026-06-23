import numpy as np
import pandas as pd

def nz(series: pd.Series, replacement: float = 0.0) -> pd.Series:
    """Replaces NaN values with a replacement value (similar to nz in Pine Script)."""
    return series.fillna(replacement)

def sma(source: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average (SMA)."""
    return source.rolling(window=length, min_periods=1).mean()

def ema(source: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average (EMA)."""
    return source.ewm(span=length, adjust=False, min_periods=1).mean()

def wma(source: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average (WMA)."""
    weights = np.arange(1, length + 1)
    return source.rolling(window=length, min_periods=length).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )

def hma(source: pd.Series, length: int) -> pd.Series:
    """Hull Moving Average (HMA)."""
    half_length = int(length / 2)
    sqrt_length = int(np.sqrt(length))
    wma_half = wma(source, half_length)
    wma_full = wma(source, length)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_length)

def dema(source: pd.Series, length: int) -> pd.Series:
    """Double Exponential Moving Average (DEMA)."""
    ema1 = ema(source, length)
    ema2 = ema(ema1, length)
    return 2 * ema1 - ema2

def rma(source: pd.Series, length: int) -> pd.Series:
    """Running Moving Average (RMA) used in RSI/ATR."""
    alpha = 1.0 / length
    # Initialize with SMA
    sma_val = sma(source, length)
    rma_vals = pd.Series(index=source.index, dtype=float)
    if len(source) > 0:
        rma_vals.iloc[0] = source.iloc[0]
        for i in range(1, len(source)):
            if pd.isna(sma_val.iloc[i-1]):
                rma_vals.iloc[i] = sma_val.iloc[i]
            else:
                rma_vals.iloc[i] = alpha * source.iloc[i] + (1 - alpha) * rma_vals.iloc[i-1]
    return rma_vals

def tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range (TR)."""
    hl = high - low
    hpc = (high - close.shift(1)).abs()
    lpc = (low - close.shift(1)).abs()
    return pd.concat([hl, hpc, lpc], axis=1).max(axis=1)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Average True Range (ATR)."""
    return rma(tr(high, low, close), length)

def stdev(source: pd.Series, length: int) -> pd.Series:
    """Standard Deviation."""
    return source.rolling(window=length, min_periods=1).std(ddof=0)

def highest(source: pd.Series, length: int) -> pd.Series:
    """Highest value in the last length bars."""
    return source.rolling(window=length, min_periods=1).max()

def lowest(source: pd.Series, length: int) -> pd.Series:
    """Lowest value in the last length bars."""
    return source.rolling(window=length, min_periods=1).min()

def crossover(s1: pd.Series, s2) -> pd.Series:
    """Returns True if s1 crossed over s2 on the current bar."""
    s2_series = s2 if isinstance(s2, pd.Series) else pd.Series(s2, index=s1.index)
    return (s1 > s2_series) & (s1.shift(1) <= s2_series.shift(1))

def crossunder(s1: pd.Series, s2) -> pd.Series:
    """Returns True if s1 crossed under s2 on the current bar."""
    s2_series = s2 if isinstance(s2, pd.Series) else pd.Series(s2, index=s1.index)
    return (s1 < s2_series) & (s1.shift(1) >= s2_series.shift(1))

def linreg(source: pd.Series, length: int, offset: int = 0) -> pd.Series:
    """Linear regression value (similar to ta.linreg in Pine Script)."""
    x = np.arange(length)
    x_mean = x.mean()
    x_dev = x - x_mean
    x_var = (x_dev ** 2).sum()
    
    def calc_reg(y_window):
        if len(y_window) < length or np.any(np.isnan(y_window)):
            return np.nan
        y_mean = y_window.mean()
        y_dev = y_window - y_mean
        slope = np.dot(y_dev, x_dev) / x_var
        intercept = y_mean - slope * x_mean
        return intercept + slope * (length - 1 - offset)
    
    return source.rolling(window=length, min_periods=length).apply(calc_reg, raw=True)

def pivotlow(source: pd.Series, left: int, right: int) -> pd.Series:
    """Pivot Low."""
    pivots = pd.Series(np.nan, index=source.index)
    for i in range(left, len(source) - right):
        val = source.iloc[i]
        if pd.isna(val):
            continue
        is_pivot = True
        for l in range(1, left + 1):
            if pd.isna(source.iloc[i - l]) or source.iloc[i - l] < val:
                is_pivot = False
                break
        if not is_pivot:
            continue
        for r in range(1, right + 1):
            if pd.isna(source.iloc[i + r]) or source.iloc[i + r] <= val:
                is_pivot = False
                break
        if is_pivot:
            pivots.iloc[i + right] = val
    return pivots

def pivothigh(source: pd.Series, left: int, right: int) -> pd.Series:
    """Pivot High."""
    pivots = pd.Series(np.nan, index=source.index)
    for i in range(left, len(source) - right):
        val = source.iloc[i]
        if pd.isna(val):
            continue
        is_pivot = True
        for l in range(1, left + 1):
            if pd.isna(source.iloc[i - l]) or source.iloc[i - l] > val:
                is_pivot = False
                break
        if not is_pivot:
            continue
        for r in range(1, right + 1):
            if pd.isna(source.iloc[i + r]) or source.iloc[i + r] >= val:
                is_pivot = False
                break
        if is_pivot:
            pivots.iloc[i + right] = val
    return pivots

def valuewhen(cond: pd.Series, source: pd.Series, occurrence: int = 0) -> pd.Series:
    """Returns the value of source when cond was True occurrence times ago."""
    result = pd.Series(np.nan, index=source.index)
    true_vals = []
    for i in range(len(source)):
        if cond.iloc[i] == True or cond.iloc[i] == 1:
            true_vals.append(source.iloc[i])
        if len(true_vals) > occurrence:
            result.iloc[i] = true_vals[-(occurrence + 1)]
    return result

def barssince(cond: pd.Series) -> pd.Series:
    """Returns the number of bars since cond was last True."""
    result = pd.Series(np.nan, index=cond.index)
    last_true = -1
    for i in range(len(cond)):
        if cond.iloc[i] == True or cond.iloc[i] == 1:
            last_true = i
        if last_true != -1:
            result.iloc[i] = i - last_true
    return result

def vwma(source: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """Volume Weighted Moving Average."""
    pv = source * volume
    pv_sum = pv.rolling(window=length, min_periods=1).sum()
    v_sum = volume.rolling(window=length, min_periods=1).sum()
    return pv_sum / v_sum

def tema(source: pd.Series, length: int) -> pd.Series:
    """Triple Exponential Moving Average (TEMA)."""
    ema1 = ema(source, length)
    ema2 = ema(ema1, length)
    ema3 = ema(ema2, length)
    return 3 * ema1 - 3 * ema2 + ema3

def t3(source: pd.Series, length: int, volume_factor: float = 0.7) -> pd.Series:
    """T3 Moving Average."""
    a = volume_factor
    c1 = - (a ** 3)
    c2 = 3 * (a ** 2) + 3 * (a ** 3)
    c3 = -6 * (a ** 2) - 3 * a - 3 * (a ** 3)
    c4 = 1 + 3 * a + 3 * (a ** 2) + (a ** 3)
    
    e1 = ema(source, length)
    e2 = ema(e1, length)
    e3 = ema(e2, length)
    e4 = ema(e3, length)
    e5 = ema(e4, length)
    e6 = ema(e5, length)
    
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

def alma(source: pd.Series, length: int, offset: float, sigma: float) -> pd.Series:
    """Arnaud Legoux Moving Average (ALMA)."""
    m = offset * (length - 1)
    s = length / sigma
    weights = np.zeros(length)
    for i in range(length):
        weights[i] = np.exp(-((i - m) ** 2) / (2 * (s ** 2)))
    weights_sum = weights.sum()
    if weights_sum != 0:
        weights /= weights_sum
    else:
        weights = np.ones(length) / length
    
    return source.rolling(window=length, min_periods=length).apply(
        lambda x: np.dot(x, weights[::-1]), raw=True
    )

def rsi(source: pd.Series, length: int) -> pd.Series:
    """Relative Strength Index (RSI) using RMA."""
    change = source.diff()
    up = change.clip(lower=0)
    down = (-change).clip(lower=0)
    rma_up = rma(up, length)
    rma_down = rma(down, length)
    rs = rma_up / rma_down
    return 100 - (100 / (1 + rs))

def sar(high: pd.Series, low: pd.Series, close: pd.Series, start: float = 0.02, inc: float = 0.02, max_val: float = 0.2) -> pd.Series:
    """Parabolic SAR (Stop and Reverse)."""
    sar_vals = np.empty(len(close))
    sar_vals[:] = np.nan
    if len(close) == 0:
        return pd.Series(sar_vals, index=close.index)
        
    is_up = True
    sar_vals[0] = low.iloc[0]
    ep = high.iloc[0]
    af = start
    
    for i in range(1, len(close)):
        prev_sar = sar_vals[i-1]
        
        if is_up:
            sar_vals[i] = prev_sar + af * (ep - prev_sar)
            l1 = low.iloc[i-1]
            l2 = low.iloc[i-2] if i > 1 else l1
            sar_vals[i] = min(sar_vals[i], l1, l2)
            
            if low.iloc[i] < sar_vals[i]:
                is_up = False
                sar_vals[i] = ep
                ep = low.iloc[i]
                af = start
            else:
                if high.iloc[i] > ep:
                    ep = high.iloc[i]
                    af = min(af + inc, max_val)
        else:
            sar_vals[i] = prev_sar - af * (prev_sar - ep)
            h1 = high.iloc[i-1]
            h2 = high.iloc[i-2] if i > 1 else h1
            sar_vals[i] = max(sar_vals[i], h1, h2)
            
            if high.iloc[i] > sar_vals[i]:
                is_up = True
                sar_vals[i] = ep
                ep = high.iloc[i]
                af = start
            else:
                if low.iloc[i] < ep:
                    ep = low.iloc[i]
                    af = min(af + inc, max_val)
                    
    return pd.Series(sar_vals, index=close.index)

def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Commodity Channel Index (CCI)."""
    tp = (high + low + close) / 3
    tp_sma = sma(tp, length)
    
    def calc_md(window):
        if len(window) < length or np.any(np.isnan(window)):
            return np.nan
        mean = window.mean()
        return np.mean(np.abs(window - mean))
        
    md = tp.rolling(window=length, min_periods=length).apply(calc_md, raw=True)
    return (tp - tp_sma) / (0.015 * md)

def cmo(source: pd.Series, length: int) -> pd.Series:
    """Chande Momentum Oscillator (CMO)."""
    change = source.diff()
    up = change.clip(lower=0)
    down = (-change).clip(lower=0)
    sum_up = up.rolling(window=length, min_periods=length).sum()
    sum_down = down.rolling(window=length, min_periods=length).sum()
    return (sum_up - sum_down) / (sum_up + sum_down) * 100

def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """Money Flow Index (MFI)."""
    tp = (high + low + close) / 3
    rmf = tp * volume
    change = tp.diff()
    
    pos_mf = pd.Series(np.where(change > 0, rmf, 0.0), index=tp.index)
    neg_mf = pd.Series(np.where(change < 0, rmf, 0.0), index=tp.index)
    
    pos_mf_sum = pos_mf.rolling(window=length, min_periods=length).sum()
    neg_mf_sum = neg_mf.rolling(window=length, min_periods=length).sum()
    
    mfr = pos_mf_sum / neg_mf_sum
    return 100 - (100 / (1 + mfr))

def trima(source: pd.Series, length: int) -> pd.Series:
    """Triangular Moving Average (TRIMA)."""
    if length % 2 == 1:
        m = int((length + 1) / 2)
        return sma(sma(source, m), m)
    else:
        m1 = int(length / 2)
        m2 = m1 + 1
        return sma(sma(source, m1), m2)

def frama(source: pd.Series, length: int) -> pd.Series:
    """Fractal Adaptive Moving Average (FRAMA)."""
    n = length if length % 2 == 0 else length + 1
    half_n = n // 2
    
    hh1 = source.rolling(window=half_n).max()
    ll1 = source.rolling(window=half_n).min()
    
    hh2 = source.shift(half_n).rolling(window=half_n).max()
    ll2 = source.shift(half_n).rolling(window=half_n).min()
    
    hh = source.rolling(window=n).max()
    ll = source.rolling(window=n).min()
    
    n1 = (hh1 - ll1) / half_n
    n2 = (hh2 - ll2) / half_n
    n3 = (hh - ll) / n
    
    d = np.log((n1 + n2).clip(lower=1e-10) / n3.clip(lower=1e-10)) / np.log(2.0)
    d = d.clip(1.0, 2.0)
    
    alpha = np.exp(-4.6 * (d - 1.0)).clip(0.01, 1.0)
    
    frama_vals = np.empty(len(source))
    frama_vals[:] = np.nan
    
    start_idx = 0
    for idx in range(len(source)):
        if not pd.isna(alpha.iloc[idx]) and not pd.isna(source.iloc[idx]):
            start_idx = idx
            frama_vals[idx] = source.iloc[idx]
            break
            
    for idx in range(start_idx + 1, len(source)):
        a = alpha.iloc[idx]
        src_val = source.iloc[idx]
        prev_f = frama_vals[idx - 1]
        if pd.isna(a) or pd.isna(src_val):
            frama_vals[idx] = prev_f
        else:
            frama_vals[idx] = a * src_val + (1.0 - a) * prev_f
            
    return pd.Series(frama_vals, index=source.index)

def mom(source: pd.Series, length: int) -> pd.Series:
    """Momentum (difference between current value and value length bars ago)."""
    return source - source.shift(length)

def mode(source: pd.Series, length: int) -> pd.Series:
    """Mode (most common value in a sliding window)."""
    def calc_mode(window):
        if len(window) < length or np.any(np.isnan(window)):
            return np.nan
        vals, counts = np.unique(window, return_counts=True)
        max_count = counts.max()
        candidates = vals[counts == max_count]
        if len(candidates) == 1:
            return candidates[0]
        # Tie-breaker: return the candidate that appears most recently in the window
        for val in window[::-1]:
            if val in candidates:
                return val
        return candidates[0]
        
    return source.rolling(window=length, min_periods=length).apply(calc_mode, raw=True)







