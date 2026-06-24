"""
Ehler's SuperSmoother Filter (Family 2: Filtering)
====================================================

A two-pole recursive filter designed by John Ehler that removes high-frequency
noise while preserving signal phase. Unlike simple moving averages, it has
minimal lag because it uses a resonant filter structure.

Key properties:
- Two-pole (biquad) filter with cutoff frequency controlled by `length`
- Phase-accurate: minimal lag compared to SMA/EMA of equivalent length
- Removes noise without distorting trend direction

Mathematical basis:
    a1 = exp(-sqrt(2) * pi / length)
    b1 = 2 * a1 * cos(sqrt(2) * 180 / length)
    c1 = 1 - b1 + a1^2
    c2 = b1
    c3 = -a1^2

    filter[i] = c1 * (src[i] + src[i-1]) / 2 + c2 * filter[i-1] + c3 * filter[i-2]

Output:
    - smooth: The filtered (smoothed) price series
    - direction: +1 if smoothed is rising, -1 if smoothed is falling
"""

import numpy as np
import pandas as pd


def ehler_supersmoother(df: pd.DataFrame,
                        source_col: str = 'close',
                        length: int = 7) -> pd.DataFrame:
    """
    Apply Ehler's 2-pole SuperSmoother filter to a price series.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with OHLCV data. Must contain `source_col`.
    source_col : str, default 'close'
        Column name to smooth.
    length : int, default 7
        Filter period (controls cutoff frequency). Typical range: 5-50.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - smooth: Filtered value
        - direction: +1 (rising) or -1 (falling)
    """
    src = df[source_col].astype(float).copy()

    # Filter coefficients
    a1 = np.exp(-np.sqrt(2) * np.pi / length)
    b1 = 2.0 * a1 * np.cos(np.radians(np.sqrt(2) * 180.0 / length))
    c1 = 1.0 - b1 + a1 * a1
    c2 = b1
    c3 = -a1 * a1

    vals = src.ffill().fillna(0).values
    n = len(vals)
    filt = np.zeros(n, dtype=float)

    # Initialize first two bars
    filt[0] = vals[0]
    if n > 1:
        filt[1] = vals[1]

    # Recursive filter
    for i in range(2, n):
        filt[i] = (c1 * (vals[i] + vals[i - 1]) / 2.0
                   + c2 * filt[i - 1]
                   + c3 * filt[i - 2])

    smooth = pd.Series(filt, index=df.index)

    # Direction: +1 if smoothed is rising, -1 if falling
    direction = pd.Series(np.where(smooth.diff() > 0, 1, -1),
                          index=df.index)

    # First bar has no diff; use +1 as default
    direction.iloc[0] = 1

    result = pd.DataFrame(index=df.index)
    result['smooth'] = smooth
    result['direction'] = direction.astype(int)

    return result


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import json, pathlib

    data_path = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'btc_daily.json'
    with open(data_path) as f:
        raw = json.load(f)

    df = pd.DataFrame(raw['aligned_data'])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    out = ehler_supersmoother(df, source_col='close', length=7)
    print(f"Shape: {out.shape}")
    print(f"Columns: {list(out.columns)}")
    print(f"Direction values: {out['direction'].value_counts().to_dict()}")
    print(out.tail())
