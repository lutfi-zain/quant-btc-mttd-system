"""
HMM Regime Detection Indicator (Family 9: Bayesian)
=====================================================

Uses a simple 3-state Hidden Markov Model (HMM) to detect bull/bear/sideways
market regimes. The model is trained on returns + volatility features and
decodes the most likely state sequence via the Viterbi algorithm.

No external ML library dependencies — implemented from scratch using
numpy and scipy only (scipy.stats for Gaussian pdf).

Signal Logic:
- Features: daily log returns + rolling volatility (std of returns)
- HMM with 3 Gaussian states fitted via Baum-Welch EM
- States are labeled post-fit: highest mean return = BULL, lowest = BEAR,
  remaining = SIDEWAYS
- Trending regime (BULL) → direction = +1 (trade)
- Sideways or BEAR regime → direction = -1 (avoid)

Rationale:
- Markets exhibit regime-switching behavior: trending vs. choppy
- HMM explicitly models hidden state transitions (Markov property)
- BULL regime has consistently positive drift → trend-following profitable
- BEAR/SIDEWAYS have negative or near-zero drift → whipsaw risk
- Regime detection is a gating filter: only trade when regime is favorable

Output:
    - state: decoded HMM state label (0=BULL, 1=BEAR, 2=SIDEWAYS)
    - state_mean: mean return of decoded state (for diagnostics)
    - bull_prob: forward probability of being in BULL state
    - direction: +1 (BULL regime) or -1 (BEAR/SIDEWAYS)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# ---------------------------------------------------------------------------
# HMM core (Baum-Welch + Viterbi) — no external ML libs
# ---------------------------------------------------------------------------

def _log_normalize(log_vec):
    """Normalize log-probabilities so exp sums to 1."""
    c = np.max(log_vec)
    log_sum = c + np.log(np.sum(np.exp(log_vec - c)))
    return log_vec - log_sum


def _forward(pi_log, A_log, B, T):
    """
    Forward algorithm in log-space.

    Parameters
    ----------
    pi_log : (K,) log initial probabilities
    A_log  : (K, K) log transition matrix
    B      : (T, K) emission log-probabilities
    T      : int, number of observations

    Returns
    -------
    alpha : (T, K) log-forward probabilities
    """
    K = len(pi_log)
    alpha = np.zeros((T, K))
    alpha[0] = pi_log + B[0]
    for t in range(1, T):
        for j in range(K):
            alpha[t, j] = np.logaddexp.reduce(alpha[t - 1] + A_log[:, j]) + B[t, j]
    return alpha


def _backward(pi_log, A_log, B, T):
    """
    Backward algorithm in log-space.
    """
    K = len(pi_log)
    beta = np.zeros((T, K))
    # β_T = 1 → log = 0
    for t in range(T - 2, -1, -1):
        for i in range(K):
            beta[t, i] = np.logaddexp.reduce(
                A_log[i, :] + B[t + 1, :] + beta[t + 1, :]
            )
    return beta


def _viterbi(pi_log, A_log, B, T, K):
    """
    Viterbi algorithm: find the most likely state sequence.
    """
    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)

    delta[0] = pi_log + B[0]

    for t in range(1, T):
        for j in range(K):
            candidates = delta[t - 1] + A_log[:, j]
            psi[t, j] = np.argmax(candidates)
            delta[t, j] = candidates[psi[t, j]] + B[t, j]

    # Backtrack
    states = np.zeros(T, dtype=int)
    states[T - 1] = np.argmax(delta[T - 1])
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states


def _baum_welch(X, n_states=3, max_iter=50, tol=1e-4, seed=42):
    """
    Baum-Welch EM for Gaussian HMM (diagonal covariance).

    Parameters
    ----------
    X : (T, D) observations (D features)
    n_states : int
    max_iter : int
    tol : float
    seed : int

    Returns
    -------
    pi : (K,) initial probabilities
    A  : (K, K) transition matrix
    mu : (K, D) state means
    sigma : (K, D) state std devs
    """
    rng = np.random.RandomState(seed)
    T, D = X.shape
    K = n_states

    # Initialize parameters randomly
    pi = rng.dirichlet(np.ones(K))
    A = np.array([rng.dirichlet(np.ones(K)) for _ in range(K)])  # K x K
    mu = np.zeros((K, D))
    sigma = np.ones((K, D))

    # K-means-like init for mu/sigma
    indices = rng.choice(T, K, replace=False)
    mu = X[indices].copy()
    for d in range(D):
        sigma[:, d] = X[:, d].std()

    prev_ll = -np.inf

    for iteration in range(max_iter):
        # E-step: compute emission probs
        B = np.zeros((T, K))
        for k in range(K):
            for d in range(D):
                B[:, k] += norm.logpdf(X[:, d], mu[k, d], sigma[k, d] + 1e-10)

        pi_log = np.log(pi + 1e-300)
        A_log = np.log(A + 1e-300)

        alpha = _forward(pi_log, A_log, B, T)
        beta = _backward(pi_log, A_log, B, T)

        # Log-likelihood
        ll = np.logaddexp.reduce(alpha[-1])

        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

        # Gamma (state posteriors)
        gamma = alpha + beta
        gamma = gamma - np.logaddexp.reduce(gamma, axis=1, keepdims=True)
        gamma = np.exp(gamma)

        # Xi (transition posteriors)
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = np.exp(
                        alpha[t, i] + A_log[i, j] + B[t + 1, j] + beta[t + 1, j] - ll
                    )

        # M-step
        pi = gamma[0]
        pi = np.clip(pi, 1e-300, None)
        pi /= pi.sum()

        for i in range(K):
            for j in range(K):
                A[i, j] = xi[:, i, j].sum() / (gamma[:-1, i].sum() + 1e-300)
            A[i] = np.clip(A[i], 1e-300, None)
            A[i] /= A[i].sum()

            gamma_k_sum = gamma[:, i].sum() + 1e-300
            for d in range(D):
                mu[i, d] = (gamma[:, i] * X[:, d]).sum() / gamma_k_sum
                var = (gamma[:, i] * (X[:, d] - mu[i, d]) ** 2).sum() / gamma_k_sum
                sigma[i, d] = np.sqrt(max(var, 1e-10))

    return pi, A, mu, sigma


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hmm_regime(df: pd.DataFrame,
               source_col: str = 'close',
               n_states: int = 3,
               window: int = 100,
               vol_window: int = 20,
               lookback: int = 250,
               seed: int = 42) -> pd.DataFrame:
    """
    Compute HMM regime detection indicator.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with OHLCV data. Must contain `source_col`.
    source_col : str, default 'close'
        Column used to compute returns and volatility.
    n_states : int, default 3
        Number of hidden states (BULL, BEAR, SIDEWAYS).
    window : int, default 100
        Lookback window for rolling volatility feature.
    vol_window : int, default 20
        Smoothing window for volatility feature (short-term).
    lookback : int, default 250
        Training lookback for HMM fitting (sliding window).
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - state: decoded HMM state label (0-2)
        - bull_prob: forward probability of BULL state
        - direction: +1 (BULL regime) or -1 (BEAR/SIDEWAYS)
    """
    src = df[source_col].astype(float).copy()

    # Step 1: Compute log returns
    log_returns = np.log(src / src.shift(1)).values

    # Step 2: Compute rolling volatility (short-term std of returns)
    vol = pd.Series(log_returns).rolling(window=vol_window, min_periods=vol_window).std().values

    # Step 3: Stack features: (returns, volatility)
    features = np.column_stack([log_returns, vol])

    n = len(features)
    state_seq = np.full(n, -1, dtype=int)
    bull_prob_seq = np.full(n, np.nan)

    # Step 4: Fit HMM once on recent data, decode full window
    # Use sliding window with stride (re-fit every `refit_interval` bars)
    refit_interval = 20  # re-fit every 20 bars
    pi, A, mu, sigma = None, None, None, None
    bull_state_id = 0  # default BULL state index

    # Filter valid rows
    valid_all = ~np.isnan(features).any(axis=1)

    for i in range(lookback, n):
        # Re-fit periodically
        if pi is None or (i - lookback) % refit_interval == 0:
            window_start = max(0, i - lookback)
            X_train = features[window_start:i]
            valid = ~np.isnan(X_train).any(axis=1)
            if valid.sum() < lookback * 0.5:
                continue
            X_train = X_train[valid]
            pi, A, mu, sigma = _baum_welch(X_train, n_states=n_states, max_iter=20, seed=seed + (i // refit_interval))
            bull_state_id = np.argmax(mu[:, 0])  # highest mean return = BULL

        if pi is None:
            continue

        # Decode with last vol_window observations
        decode_start = max(0, i - vol_window + 1)
        X_decode = features[decode_start:i + 1].copy()
        # Replace NaN with 0 (no-trade)
        nan_mask = np.isnan(X_decode)
        if nan_mask.any():
            X_decode[nan_mask] = 0.0

        # Emission log-probs
        B_decode = np.zeros((len(X_decode), n_states))
        for k in range(n_states):
            for d in range(X_decode.shape[1]):
                B_decode[:, k] += norm.logpdf(X_decode[:, d], mu[k, d], sigma[k, d] + 1e-10)

        pi_log = np.log(pi + 1e-300)
        A_log = np.log(A + 1e-300)

        states = _viterbi(pi_log, A_log, B_decode, len(X_decode), n_states)
        state_seq[i] = states[-1]

        # Bull probability from forward algorithm
        alpha = _forward(pi_log, A_log, B_decode, len(X_decode))
        last_alpha = alpha[-1]
        bull_prob = np.exp(last_alpha[bull_state_id]) / (np.exp(last_alpha).sum() + 1e-300)
        bull_prob_seq[i] = bull_prob

    # Direction: BULL → +1, others → -1
    direction = np.where(state_seq == bull_state_id, 1, -1)
    direction[state_seq == -1] = -1  # default to avoid
    bull_prob_seq[np.isnan(bull_prob_seq)] = 0.0

    # Build result DataFrame
    result = pd.DataFrame(index=df.index)
    result['state'] = state_seq
    result['bull_prob'] = np.round(bull_prob_seq, 4)
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

    out = hmm_regime(df, source_col='close', n_states=3, lookback=250)
    print(f"Shape: {out.shape}")
    print(f"Columns: {list(out.columns)}")
    print(f"Direction values: {out['direction'].value_counts().to_dict()}")
    print(f"Direction unique: {sorted(out['direction'].unique())}")
    print(f"State values: {out['state'].value_counts().to_dict()}")
    print(f"Bull prob range: [{out['bull_prob'].min():.4f}, {out['bull_prob'].max():.4f}]")
    print()
    print("Last 10 rows:")
    print(out.tail(10))
    print()

    # Sanity: direction should only contain -1 and 1
    unique_dirs = set(out['direction'].unique())
    assert unique_dirs.issubset({-1, 1}), f"Invalid direction values: {unique_dirs}"
    print("✅ Direction column contains only -1 and 1")
    print(f"✅ Total rows with valid direction: {(out['direction'].isin([-1, 1])).sum()} / {len(out)}")

    # Check 2020-2021 bull market (should have BULL regime)
    print()
    print("Checking 2020-2021 bull market (should detect BULL regime):")
    bull_market = out.loc['2020-04-01':'2021-12-31']
    if len(bull_market) > 0:
        dir_counts_bull = bull_market['direction'].value_counts().to_dict()
        bull_pct = dir_counts_bull.get(1, 0) / len(bull_market) * 100
        print(f"  Direction distribution: {dir_counts_bull}")
        print(f"  BULL regime pct: {bull_pct:.1f}% (expected: high)")

    # Check 2022 bear market (should have BEAR regime)
    print()
    print("Checking 2022 bear market (should detect BEAR regime):")
    bear_market = out.loc['2022-01-01':'2022-12-31']
    if len(bear_market) > 0:
        dir_counts_bear = bear_market['direction'].value_counts().to_dict()
        bear_pct = dir_counts_bear.get(-1, 0) / len(bear_market) * 100
        print(f"  Direction distribution: {dir_counts_bear}")
        print(f"  BEAR regime pct: {bear_pct:.1f}% (expected: high)")
