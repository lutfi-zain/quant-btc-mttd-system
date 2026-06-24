"""
Volatility Cluster Indicator (Family 6: GARCH-like)
====================================================

Detects high/low volatility regimes using rolling standard deviation
as a GARCH proxy. The core idea is to compare current volatility
against a longer-term median volatility to classify the current regime.

Signal Logic:
- Compute rolling standard deviation of log returns (short-term vol)
- Compute rolling median of that volatility (long-term baseline)
- If current vol > threshold × median_vol → HIGH volatility regime
  → direction = -1 (avoid trading, risk of whipsaw)
- If current vol <= threshold × median_vol → LOW volatility regime
  → direction = +1 (trade, trend likely to persist)

Rationale:
- High volatility environments are characterized by:
  * Sharp reversals
  * Increased noise
  * Lower signal-to-noise ratio
  * Wider stop-losses needed → worse risk/reward
- Low volatility environments tend to have:
  * Smoother trends
  * Better signal quality
  * Tighter stops possible
  * Higher win rates for trend-following

The threshold parameter controls sensitivity:
- threshold = 1.0 → current vol must be below median (very selective)
- threshold = 1.2 → current vol must be below 1.2× median (default)
- threshold = 1.5 → more lenient, more trading time

Output:
    - rolling_vol: Rolling volatility (annualized)
    - median_vol: Rolling median of volatility
    - vol_ratio: rolling_vol / median_vol
    - direction: +1 (low vol, trade) or -1 (high vol, avoid)
"""

import numpy as np
import pandas as pd


def volatility_cluster(df: pd.DataFrame,
                       source_col: str = 'close',
                       window: int = 20,
                       median_window: int = 100,
                       threshold: float = 1.2) -> pd.DataFrame:
    """
    Compute volatility cluster regime indicator.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with OHLCV data. Must contain `source_col`.
    source_col : str, default 'close'
        Column used to compute volatility.
    window : int, default 20
        Lookback window for rolling volatility (short-term).
    median_window : int, default 100
        Lookback window for median volatility (long-term baseline).
    threshold : float, default 1.2
        Volatility ratio threshold. High vol when vol_ratio > threshold.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - rolling_vol: Rolling volatility (annualized)
        - median_vol: Rolling median of volatility
        - vol_ratio: rolling_vol / median_vol
        - direction: +1 (low vol, trade) or -1 (high vol, avoid)
    """
    src = df[source_col].astype(float).copy()

    # Step 1: Compute log returns
    log_returns = np.log(src / src.shift(1))

    # Step 2: Rolling standard deviation (short-term volatility)
    rolling_vol = log_returns.rolling(window=window, min_periods=window).std()

    # Step 3: Rolling median of volatility (long-term baseline)
    median_vol = rolling_vol.rolling(window=median_window, min_periods=median_window).median()

    # Step 4: Volatility ratio
    vol_ratio = rolling_vol / median_vol

    # Step 5: Direction logic
    # High volatility → direction = -1 (avoid trading)
    # Low volatility  → direction = +1 (trade)
    direction = np.where(vol_ratio > threshold, -1, 1)

    # Handle NaN: default to -1 (avoid) when vol is unknown
    direction[np.isnan(vol_ratio)] = -1

    # Build result DataFrame
    result = pd.DataFrame(index=df.index)
    result['rolling_vol'] = rolling_vol
    result['median_vol'] = median_vol
    result['vol_ratio'] = vol_ratio
    result['direction'] = direction.astype(int)

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

    out = volatility_cluster(df, source_col='close', window=20, median_window=100, threshold=1.2)
    print(f"Shape: {out.shape}")
    print(f"Columns: {list(out.columns)}")
    print(f"Direction values: {out['direction'].value_counts().to_dict()}")
    print(f"Direction unique: {sorted(out['direction'].unique())}")
    print(f"Vol ratio range: [{out['vol_ratio'].min():.4f}, {out['vol_ratio'].max():.4f}]")
    print(f"Rolling vol range: [{out['rolling_vol'].min():.6f}, {out['rolling_vol'].max():.6f}]")
    print()
    print("Last 10 rows:")
    print(out.tail(10))
    print()

    # Sanity: direction should only contain -1 and 1
    unique_dirs = set(out['direction'].unique())
    assert unique_dirs.issubset({-1, 1}), f"Invalid direction values: {unique_dirs}"
    print("✅ Direction column contains only -1 and 1")
    print(f"✅ Total rows with valid direction: {(out['direction'].isin([-1, 1])).sum()} / {len(out)}")

    # Check March 2020 crash period (high vol should be -1)
    print()
    print("Checking March 2020 crash period (high vol = direction -1):")
    mar2020 = out.loc['2020-03-01':'2020-03-31']
    if len(mar2020) > 0:
        avg_vol_ratio = mar2020['vol_ratio'].mean()
        dir_counts = mar2020['direction'].value_counts().to_dict()
        print(f"  Average vol ratio: {avg_vol_ratio:.2f}")
        print(f"  Direction distribution: {dir_counts}")
        high_vol_ratio = avg_vol_ratio > 1.0
        has_negative = -1 in dir_counts
        print(f"  Vol ratio > 1.0: {high_vol_ratio} (expected: True)")
        print(f"  Has direction=-1: {has_negative} (expected: True)")

    # Check a calm period (2019 mid-year)
    print()
    print("Checking 2019 calm period (low vol = direction +1):")
    calm2019 = out.loc['2019-06-01':'2019-09-30']
    if len(calm2019) > 0:
        avg_vol_ratio_calm = calm2019['vol_ratio'].mean()
        dir_counts_calm = calm2019['direction'].value_counts().to_dict()
        print(f"  Average vol ratio: {avg_vol_ratio_calm:.2f}")
        print(f"  Direction distribution: {dir_counts_calm}")
        low_vol_ratio = avg_vol_ratio_calm < 1.2
        has_positive = 1 in dir_counts_calm
        print(f"  Vol ratio < 1.2: {low_vol_ratio} (expected: True for calm period)")
        print(f"  Has direction=1: {has_positive} (expected: True)")
