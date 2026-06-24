"""
Linear Regression Trend Indicator (Family 3: Regression)
=========================================================

Fits an Ordinary Least Squares (OLS) line to prices over a rolling window
and creates upper/lower channel bands based on the standard error of the fit.

Signal Logic:
- Price < lower band  → oversold   → direction = +1 (buy)
- Price > upper band  → overbought → direction = -1 (sell)
- Price inside bands  → follow the slope:
    - Positive slope → direction = +1 (uptrend)
    - Negative slope → direction = -1 (downtrend)

The channel width adapts to how well the line fits the data: tight fits
produce narrow channels (easy to break out), noisy fits produce wider
channels (harder to trigger).

Output:
    - lr_line: Linear regression value (midpoint of channel)
    - upper_band: Upper channel band
    - lower_band: Lower channel band
    - slope: Normalized slope of the regression line
    - direction: +1 (bullish) or -1 (bearish)
"""

import numpy as np
import pandas as pd


def linear_reg_trend(df: pd.DataFrame,
                     source_col: str = 'close',
                     length: int = 50,
                     num_std: float = 2.0) -> pd.DataFrame:
    """
    Compute Linear Regression Trend with channel bands.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with OHLCV data. Must contain `source_col`.
    source_col : str, default 'close'
        Column used for the regression.
    length : int, default 50
        Lookback window (bars) for the linear regression fit.
    num_std : float, default 2.0
        Number of standard errors for the channel bands.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - lr_line: Linear regression value
        - upper_band: Upper channel band
        - lower_band: Lower channel band
        - slope: Normalized slope
        - direction: +1 (bullish) or -1 (bearish)
    """
    src = df[source_col].astype(float).copy()

    # x-axis for regression: 0, 1, 2, ..., length-1
    x = np.arange(length, dtype=float)
    x_mean = x.mean()
    x_dev = x - x_mean
    x_var = (x_dev ** 2).sum()

    # Pre-allocate output arrays
    n = len(src)
    lr_line = np.full(n, np.nan)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    slope = np.full(n, np.nan)

    src_vals = src.values

    for i in range(length - 1, n):
        y_window = src_vals[i - length + 1: i + 1]

        # Skip if any NaN in window
        if np.any(np.isnan(y_window)):
            continue

        y_mean = y_window.mean()
        y_dev = y_window - y_mean

        # OLS: slope = Cov(x,y) / Var(x)
        b1 = np.dot(x_dev, y_dev) / x_var
        b0 = y_mean - b1 * x_mean

        # Regression line value at the current bar (last point in window)
        y_hat = b0 + b1 * (length - 1)
        lr_line[i] = y_hat

        # Standard error of the estimate
        residuals = y_window - (b0 + b1 * x)
        se = np.sqrt((residuals ** 2).sum() / (length - 2)) if length > 2 else 0.0

        # Channel bands
        upper_band[i] = y_hat + num_std * se
        lower_band[i] = y_hat - num_std * se

        # Normalized slope: slope / price * 10000 for scale invariance
        slope[i] = b1 / y_hat * 10000.0 if y_hat != 0 else 0.0

    # Direction logic
    direction = np.full(n, -1, dtype=int)

    for i in range(n):
        if np.isnan(lr_line[i]):
            continue

        price = src_vals[i]
        ub = upper_band[i]
        lb = lower_band[i]

        if np.isnan(ub) or np.isnan(lb):
            continue

        if price < lb:
            # Oversold → bullish
            direction[i] = 1
        elif price > ub:
            # Overbought → bearish
            direction[i] = -1
        else:
            # Inside band → follow slope
            direction[i] = 1 if slope[i] > 0 else -1

    # Build result DataFrame
    result = pd.DataFrame(index=df.index)
    result['lr_line'] = lr_line
    result['upper_band'] = upper_band
    result['lower_band'] = lower_band
    result['slope'] = slope
    result['direction'] = direction

    return result


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import json
    import pathlib

    data_path = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'btc_daily.json'
    with open(data_path) as f:
        raw = json.load(f)

    df = pd.DataFrame(raw['aligned_data'])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    out = linear_reg_trend(df, source_col='close', length=50, num_std=2.0)
    print(f"Shape: {out.shape}")
    print(f"Columns: {list(out.columns)}")
    print(f"Direction values: {out['direction'].value_counts().to_dict()}")
    print(f"Direction unique: {sorted(out['direction'].unique())}")
    print(f"Slope range: [{out['slope'].min():.4f}, {out['slope'].max():.4f}]")
    print(f"LR Line range: [{out['lr_line'].min():.2f}, {out['lr_line'].max():.2f}]")
    print(f"Upper band range: [{out['upper_band'].min():.2f}, {out['upper_band'].max():.2f}]")
    print(f"Lower band range: [{out['lower_band'].min():.2f}, {out['lower_band'].max():.2f}]")
    print()
    print("Last 10 rows:")
    print(out.tail(10))
    print()

    # Sanity: direction should only contain -1 and 1
    unique_dirs = set(out['direction'].unique())
    assert unique_dirs.issubset({-1, 1}), f"Invalid direction values: {unique_dirs}"
    print("✅ Direction column contains only -1 and 1")
    print(f"✅ Total rows with valid direction: {(out['direction'].isin([-1, 1])).sum()} / {len(out)}")
