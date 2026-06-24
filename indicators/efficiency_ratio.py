"""
Efficiency Ratio (Family 5: Fractal / Kaufman's Adaptive)
===========================================================

Also known as Kaufman's Efficiency Ratio (KER). Measures the ratio of
directional move to total volatility over a window:

    ER = |close - close_n| / sum(|close_i - close_{i-1}|, i=1..n)

Interpretation:
- ER close to 1  => price moves almost monotonically => strong **trend**
- ER close to 0  => price whipsaws back and forth   => **ranging / noise**

Our threshold: ER > 0.25 => trending (direction = +1), else ranging (-1).

This indicator is useful as a regime filter: only take trend-following signals
when the market is actually trending.

Output:
    - er: Efficiency Ratio value (0 to 1)
    - direction: +1 if ER > 0.25 (trending), -1 otherwise
"""

import numpy as np
import pandas as pd


def efficiency_ratio(df: pd.DataFrame,
                     source_col: str = 'close',
                     period: int = 14,
                     threshold: float = 0.25) -> pd.DataFrame:
    """
    Compute Kaufman's Efficiency Ratio.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with OHLCV data. Must contain `source_col`.
    source_col : str, default 'close'
        Column used to compute the ratio.
    period : int, default 14
        Lookback window (bars).
    threshold : float, default 0.25
        ER above this value is considered trending (direction = +1).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - er: Efficiency Ratio (0..1)
        - direction: +1 (trending) or -1 (ranging)
    """
    src = df[source_col].astype(float).copy()

    # Directional move: |close - close_n|
    direction = src.diff(period).abs()

    # Volatility: sum of absolute bar-to-bar changes
    volatility = src.diff().abs().rolling(window=period, min_periods=period).sum()

    # Efficiency Ratio = direction / volatility
    er = (direction / volatility).fillna(0)

    # Direction: +1 if ER > threshold (trending), -1 otherwise
    dir_series = pd.Series(
        np.where(er > threshold, 1, -1),
        index=df.index
    )

    result = pd.DataFrame(index=df.index)
    result['er'] = er
    result['direction'] = dir_series.astype(int)

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

    out = efficiency_ratio(df, source_col='close', period=14, threshold=0.25)
    print(f"Shape: {out.shape}")
    print(f"Columns: {list(out.columns)}")
    print(f"Direction values: {out['direction'].value_counts().to_dict()}")
    print(f"ER range: [{out['er'].min():.4f}, {out['er'].max():.4f}]")
    print(out.tail())
