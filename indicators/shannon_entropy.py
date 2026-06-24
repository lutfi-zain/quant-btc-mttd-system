"""
Shannon Entropy (Family 7: Entropy)
====================================

Measures the randomness or information content of a price series over a rolling
window. In efficient-market terms:

- Low entropy (< threshold) => returns are concentrated in a few bins =>
  market is trending or directional => **tradeable**.
- High entropy (>= threshold) => returns are spread uniformly => market is
  random/noisy => **not tradeable**.

Mathematical basis:
    1. Compute log-returns: r = close.pct_change()
    2. For each window, histogram the returns into `bins` bins
    3. Shannon entropy: H = -sum(p_i * log2(p_i))  where p_i = count_i / N
    4. If H < threshold (default 2.5): direction = +1 (tradeable)
       else: direction = -1 (not tradeable)

Output:
    - entropy: Rolling Shannon entropy value
    - direction: +1 if entropy < threshold, -1 otherwise
"""

import numpy as np
import pandas as pd


def shannon_entropy(df: pd.DataFrame,
                    source_col: str = 'close',
                    window: int = 15,
                    bins: int = 6,
                    threshold: float = 2.5) -> pd.DataFrame:
    """
    Compute rolling Shannon Entropy of returns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with OHLCV data. Must contain `source_col`.
    source_col : str, default 'close'
        Column used to compute log-returns.
    window : int, default 15
        Rolling window length (bars).
    bins : int, default 6
        Number of histogram bins for discretising returns.
    threshold : float, default 2.5
        Entropy below this value is considered tradeable (direction = +1).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - entropy: Shannon entropy value (bits)
        - direction: +1 (tradeable / low entropy) or -1 (not tradeable)
    """
    src = df[source_col].astype(float).copy()

    # Compute returns
    returns = src.pct_change().fillna(0)

    def _shannon(window_vals):
        """Compute Shannon entropy (in bits) for a window of returns."""
        if len(window_vals) < 2:
            return np.nan
        counts, _ = np.histogram(window_vals, bins=bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]  # drop zero bins
        if len(probs) == 0:
            return np.nan
        return -np.sum(probs * np.log2(probs))

    entropy_vals = returns.rolling(window=window, min_periods=window).apply(
        _shannon, raw=True
    )

    # Direction: +1 if entropy < threshold (tradeable), -1 otherwise
    direction = pd.Series(
        np.where(entropy_vals < threshold, 1, -1),
        index=df.index
    )

    result = pd.DataFrame(index=df.index)
    result['entropy'] = entropy_vals
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

    out = shannon_entropy(df, source_col='close', window=15, bins=6, threshold=2.5)
    print(f"Shape: {out.shape}")
    print(f"Columns: {list(out.columns)}")
    print(f"Direction values: {out['direction'].value_counts().to_dict()}")
    print(f"Entropy range: [{out['entropy'].min():.3f}, {out['entropy'].max():.3f}]")
    print(out.tail())
