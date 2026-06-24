"""
Volume Confirmation Indicator (Family 8: Volume)
=================================================

Confirms price trends using volume analysis through three components:
1. OBV (On-Balance Volume) - cumulative volume flow
2. Volume Spike Detection - unusual volume activity
3. Force Index - price-volume momentum

Signal Logic:
- OBV Trend: Compare short-term OBV SMA to long-term OBV SMA
  * OBV rising (short > long) => accumulation => bullish volume
  * OBV falling (short < long) => distribution => bearish volume

- Volume Spike: Detect when current volume > spike_mult × recent average
  * Spikes during price rise => bullish confirmation
  * Spikes during price fall => bearish confirmation

- Force Index: close-to-close change × volume
  * Positive FI => buyers in control
  * Negative FI => sellers in control

Final Direction:
- +1 (bullish): OBV rising AND (no spike OR spike during rally) AND FI positive
- -1 (bearish): OBV falling AND (no spike OR spike during decline) AND FI negative
- Neutral defaults to -1 (conservative, avoid trading without confirmation)

Rationale:
- Volume precedes price movement in efficient markets
- Price moves without volume are likely to reverse
- Volume spikes during breakouts confirm the move is real
- OBV divergence from price signals potential reversals

Output:
    - obv: On-Balance Volume
    - obv_short_sma: Short-term OBV smoothing
    - obv_long_sma: Long-term OBV smoothing
    - obv_trend: +1 if OBV rising, -1 if falling
    - volume_spike: Boolean, True when volume spike detected
    - spike_direction: +1 spike with rally, -1 spike with decline, 0 no spike
    - force_index: Force Index value
    - fi_direction: +1 if FI positive, -1 if negative
    - direction: Final confirmed direction (+1 or -1)
"""

import numpy as np
import pandas as pd


def volume_confirm(df: pd.DataFrame,
                   obv_short: int = 10,
                   obv_long: int = 30,
                   spike_mult: float = 1.5,
                   spike_lookback: int = 20,
                   fi_smooth: int = 13) -> pd.DataFrame:
    """
    Compute Volume Confirmation indicator.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with OHLCV data. Must contain 'close' and 'volume'.
    obv_short : int, default 10
        Short-term SMA period for OBV trend.
    obv_long : int, default 30
        Long-term SMA period for OBV trend.
    spike_mult : float, default 1.5
        Multiplier for volume spike detection. Spike when vol > mult × avg.
    spike_lookback : int, default 20
        Lookback period for average volume baseline.
    fi_smooth : int, default 13
        Smoothing period for Force Index.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - obv: On-Balance Volume
        - obv_short_sma: Short-term OBV SMA
        - obv_long_sma: Long-term OBV SMA
        - obv_trend: +1 (OBV rising) or -1 (OBV falling)
        - volume_spike: True when volume spike detected
        - spike_direction: +1 (spike+rally), -1 (spike+decline), 0 (no spike)
        - force_index: Force Index value
        - fi_direction: +1 (positive) or -1 (negative)
        - direction: Final confirmed direction (+1 or -1)
    """
    close = df['close'].astype(float)
    volume = df['volume'].astype(float).fillna(0)

    # === Component 1: OBV (On-Balance Volume) ===
    price_change = close.diff()
    obv_direction = np.where(price_change > 0, volume,
                             np.where(price_change < 0, -volume, 0))
    obv = pd.Series(obv_direction, index=df.index).cumsum()

    # OBV trend via SMA crossover
    obv_short_sma = obv.rolling(window=obv_short, min_periods=obv_short).mean()
    obv_long_sma = obv.rolling(window=obv_long, min_periods=obv_long).mean()

    # OBV trend: +1 if short > long (accumulation), -1 if short < long (distribution)
    obv_trend = pd.Series(
        np.where(obv_short_sma > obv_long_sma, 1,
                 np.where(obv_short_sma < obv_long_sma, -1, 0)),
        index=df.index
    )

    # === Component 2: Volume Spike Detection ===
    avg_volume = volume.rolling(window=spike_lookback, min_periods=spike_lookback).mean()
    volume_spike = volume > (spike_mult * avg_volume)

    # Spike direction based on price action during spike
    # If price rose on spike => bullish (+1)
    # If price fell on spike => bearish (-1)
    spike_direction = pd.Series(0, index=df.index, dtype=float)
    spike_direction[volume_spike & (price_change > 0)] = 1
    spike_direction[volume_spike & (price_change < 0)] = -1

    # === Component 3: Force Index ===
    # Force Index = price change × volume
    raw_fi = price_change * volume
    # Smoothed Force Index
    force_index = raw_fi.rolling(window=fi_smooth, min_periods=1).mean()

    # Force Index direction
    fi_direction = pd.Series(
        np.where(force_index > 0, 1,
                 np.where(force_index < 0, -1, 0)),
        index=df.index
    )

    # === Final Direction: Volume Confirmation ===
    # Rules:
    # 1. OBV trend provides primary signal
    # 2. Volume spikes during breakouts should confirm the trend
    # 3. Force Index should agree with OBV trend

    # Component scores
    obv_score = obv_trend.copy()
    fi_score = fi_direction.copy()

    # Combined score
    # Weight: OBV trend (40%), Force Index (30%), Spike confirmation (30%)
    # If no spike, rely on OBV + FI
    # If spike exists, it must confirm or at least not contradict

    combined_score = 0.4 * obv_score + 0.3 * fi_score

    # Add spike confirmation if present (30% weight when spike exists)
    spike_present = spike_direction != 0
    combined_score[spike_present] += 0.3 * spike_direction[spike_present]
    # When no spike, redistribute weight to OBV
    combined_score[~spike_present] += 0.3 * obv_score[~spike_present]

    # Final direction: +1 if combined > 0, -1 otherwise
    direction = pd.Series(
        np.where(combined_score > 0, 1, -1),
        index=df.index
    )

    # Handle NaN: default to -1 (conservative)
    direction = direction.fillna(-1).astype(int)

    # Build result DataFrame
    result = pd.DataFrame(index=df.index)
    result['obv'] = obv
    result['obv_short_sma'] = obv_short_sma
    result['obv_long_sma'] = obv_long_sma
    result['obv_trend'] = obv_trend.astype(int)
    result['volume_spike'] = volume_spike
    result['spike_direction'] = spike_direction.astype(int)
    result['force_index'] = force_index
    result['fi_direction'] = fi_direction.astype(int)
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

    out = volume_confirm(df, obv_short=10, obv_long=30, spike_mult=1.5,
                         spike_lookback=20, fi_smooth=13)
    print(f"Shape: {out.shape}")
    print(f"Columns: {list(out.columns)}")
    print(f"Direction values: {out['direction'].value_counts().to_dict()}")
    print(f"Direction unique: {sorted(out['direction'].unique())}")
    print(f"OBV range: [{out['obv'].min():.0f}, {out['obv'].max():.0f}]")
    print(f"Volume spikes: {out['volume_spike'].sum()} / {len(out)}")
    print()
    print("Last 10 rows:")
    print(out.tail(10))
    print()

    # Sanity: direction should only contain -1 and 1
    unique_dirs = set(out['direction'].unique())
    assert unique_dirs.issubset({-1, 1}), f"Invalid direction values: {unique_dirs}"
    print("✅ Direction column contains only -1 and 1")
    print(f"✅ Total rows with valid direction: {(out['direction'].isin([-1, 1])).sum()} / {len(out)}")

    # Check that OBV trend and direction are correlated
    obv_bullish = (out['obv_trend'] == 1).sum()
    obv_bearish = (out['obv_trend'] == -1).sum()
    dir_bullish = (out['direction'] == 1).sum()
    dir_bearish = (out['direction'] == -1).sum()
    print()
    print(f"OBV trend distribution: +1={obv_bullish}, -1={obv_bearish}")
    print(f"Direction distribution: +1={dir_bullish}, -1={dir_bearish}")

    # Check during known high-volume periods
    print()
    print("Checking 2021 bull run volume:")
    bull2021 = out.loc['2021-01-01':'2021-04-30']
    if len(bull2021) > 0:
        avg_obv = bull2021['obv'].mean()
        dir_counts = bull2021['direction'].value_counts().to_dict()
        spikes = bull2021['volume_spike'].sum()
        print(f"  Average OBV: {avg_obv:.0f}")
        print(f"  Direction distribution: {dir_counts}")
        print(f"  Volume spikes: {spikes}")

    print()
    print("Checking 2022 bear market:")
    bear2022 = out.loc['2022-01-01':'2022-06-30']
    if len(bear2022) > 0:
        avg_obv = bear2022['obv'].mean()
        dir_counts = bear2022['direction'].value_counts().to_dict()
        spikes = bear2022['volume_spike'].sum()
        print(f"  Average OBV: {avg_obv:.0f}")
        print(f"  Direction distribution: {dir_counts}")
        print(f"  Volume spikes: {spikes}")
