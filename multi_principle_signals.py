#!/usr/bin/env python3
"""
Multi-Principle Signal Generators — 10 Statistical Families
============================================================

Implements indicators from all 10 statistical families for the MTTD system:

  1. Smoothing   — KAMA, MAMA (adaptive moving averages)
  2. Filtering   — Ehler SuperSmoother, Bandpass filter
  3. Regression  — LinearReg channel, LASSO trend
  4. Spectral    — FFT cycle phase, Wavelet denoising
  5. Fractal     — Efficiency Ratio, Hurst exponent
  6. GARCH       — Volatility cluster detection
  7. Entropy     — Shannon entropy, Permutation entropy
  8. Chaos       — Phase space reconstruction
  9. Bayesian    — HMM regime detection
 10. ML Hybrid   — Composite adaptive scoring

Each family produces numeric time series compatible with the MTTD system.
"""

import math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Lasso

# ---------------------------------------------------------------------------
# Project imports (existing indicators)
# ---------------------------------------------------------------------------
from indicators.efficiency_ratio import efficiency_ratio
from indicators.ehler_supersmoother import ehler_supersmoother
from indicators.shannon_entropy import shannon_entropy
from indicators.hmm_regime import hmm_regime
from indicators.linear_reg_trend import linear_reg_trend
from indicators.volatility_cluster import volatility_cluster
from indicators.volume_confirm import volume_confirm


# ===================================================================
# Family 1 — Smoothing: KAMA & MAMA
# ===================================================================

def kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """
    Kaufman's Adaptive Moving Average (KAMA).
    
    Uses Efficiency Ratio to dynamically adjust the smoothing constant.
    Fast alpha in trends, slow alpha in noise.
    """
    src = series.astype(float).values
    n = len(src)
    kama_vals = np.full(n, np.nan)

    # Efficiency Ratio lookback
    er_period = period

    # Smoothing constants
    fast_alpha = 2.0 / (fast + 1.0)
    slow_alpha = 2.0 / (slow + 1.0)

    for i in range(n):
        if i < er_period:
            kama_vals[i] = src[i]
            continue

        # Efficiency Ratio
        change = abs(src[i] - src[i - er_period])
        volatility = sum(abs(src[j] - src[j - 1]) for j in range(i - er_period + 1, i + 1))
        er = change / volatility if volatility > 1e-10 else 0.0

        # Smoothing constant
        ssc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2

        if i == er_period:
            kama_vals[i] = src[i]
        else:
            kama_vals[i] = kama_vals[i - 1] + ssc * (src[i] - kama_vals[i - 1])

    return pd.Series(kama_vals, index=series.index)


def mama(series: pd.Series, fast_limit: float = 0.5, slow_limit: float = 0.05) -> pd.Series:
    """
    MESA Adaptive Moving Average (MAMA) — simplified phase-based adaptation.
    
    Uses Hilbert Transform phase rate-of-change to adjust smoothing.
    Faster in trending phases, slower in cycling phases.
    
    Returns the MAMA line only (simplified).
    """
    src = series.astype(float).values
    n = len(src)
    mama_vals = np.full(n, np.nan)

    # Initialize
    smooth = np.zeros(n)
    detrender = np.zeros(n)
    q1 = np.zeros(n)
    i1 = np.zeros(n)
    ji = np.zeros(n)
    jq = np.zeros(n)
    phase = np.zeros(n)
    period_out = np.zeros(n)
    delta_phase = np.zeros(n)
    alpha = np.full(n, fast_limit)

    for i in range(6, n):
        # Smooth price with 4-bar WMA
        smooth[i] = (src[i] + 2 * src[i - 1] + 2 * src[i - 2] + src[i - 3]) / 6.0

        # Detrender
        detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i - 2]
                        - 0.5769 * smooth[i - 4] - 0.0962 * smooth[i - 6])

        # Quadrature components
        q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i - 2]
                 - 0.5769 * detrender[i - 4] - 0.0962 * detrender[i - 6])
        i1[i] = detrender[i - 3]

        # Smooth I1 and Q1
        if i > 8:
            ji[i] = (0.0962 * i1[i] + 0.5769 * i1[i - 2]
                     - 0.5769 * i1[i - 4] - 0.0962 * i1[i - 6])
            jq[i] = (0.0962 * q1[i] + 0.5769 * q1[i - 2]
                     - 0.5769 * q1[i - 4] - 0.0962 * q1[i - 6])

        # Phase
        if abs(i1[i]) > 1e-10:
            phase[i] = np.arctan2(abs((jq[i] + q1[i]) / (ji[i] + i1[i])), 1.0)
        else:
            phase[i] = 0.0

        # Delta phase
        delta_phase[i] = phase[i] - phase[i - 1] if i > 6 else 0.0
        if delta_phase[i] < 0.0:
            delta_phase[i] += np.pi  # Unwrap

        # Period (simplified)
        period_out[i] = period_out[i - 1] if i > 6 else 10.0
        if abs(delta_phase[i]) > 0.01:
            period_out[i] = 2 * np.pi / abs(delta_phase[i])
            period_out[i] = np.clip(period_out[i], 6, 50)

        # Adaptive alpha
        alpha[i] = fast_limit / period_out[i] if period_out[i] > 0 else fast_limit
        alpha[i] = np.clip(alpha[i], slow_limit, fast_limit)

    # Apply MAMA with adaptive alpha
    mama_vals[0] = src[0]
    for i in range(1, n):
        mama_vals[i] = mama_vals[i - 1] + alpha[i] * (src[i] - mama_vals[i - 1])

    return pd.Series(mama_vals, index=series.index)


def compute_family1_smoothing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Smoothing Family: KAMA and MAMA adaptive moving averages.
    
    Returns columns:
      - KAMA: Kaufman's Adaptive Moving Average
      - KAMA_direction: +1 if close > KAMA, -1 otherwise
      - MAMA: MESA Adaptive Moving Average
      - MAMA_direction: +1 if close > MAMA, -1 otherwise
    """
    close = df['close'].astype(float)

    kama_series = kama(close, period=10, fast=2, slow=30)
    mama_series = mama(close, fast_limit=0.5, slow_limit=0.05)

    result = pd.DataFrame(index=df.index)
    result['KAMA'] = kama_series
    result['KAMA_direction'] = np.where(close > kama_series, 1, -1)
    result['MAMA'] = mama_series
    result['MAMA_direction'] = np.where(close > mama_series, 1, -1)

    # Combined smoothing direction: average of both
    result['SMOOTH_direction'] = np.where(
        (result['KAMA_direction'] + result['MAMA_direction']) > 0, 1, -1
    )
    return result


# ===================================================================
# Family 2 — Filtering: Ehler SuperSmoother & Bandpass Filter
# ===================================================================

def bandpass_filter(series: pd.Series, period: int = 20, bandwidth: float = 0.5) -> pd.Series:
    """
    Ehler's Two-Pole Bandpass Filter.
    
    Isolates a specific cycle component (frequency band) from the price series.
    Useful for identifying cycle turning points.
    """
    src = series.values.astype(float)
    n = len(src)

    # Filter coefficients
    beta = np.cos(2.0 * np.pi / period)
    gamma = 1.0 / np.cos(4.0 * np.pi * bandwidth / period)
    alpha = gamma - np.sqrt(gamma * gamma - 1.0)

    filt = np.zeros(n)
    for i in range(2, n):
        filt[i] = (0.5 * (1.0 - alpha) * (src[i] - src[i - 2])
                   + beta * (1.0 + alpha) * filt[i - 1]
                   - alpha * filt[i - 2])

    return pd.Series(filt, index=series.index)


def compute_family2_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtering Family: Ehler SuperSmoother and Bandpass filter.
    
    Returns columns:
      - SS_smooth: SuperSmoother filtered price
      - SS_direction: +1 rising, -1 falling
      - BP_value: Bandpass filter output
      - BP_direction: +1 if BP positive (upward cycle phase), -1 otherwise
      - FILTER_direction: combined filtering direction
    """
    close = df['close'].astype(float)

    # Import existing SuperSmoother
    ss_result = ehler_supersmoother(df, source_col='close', length=7)
    bp_series = bandpass_filter(close, period=20, bandwidth=0.5)

    result = pd.DataFrame(index=df.index)
    result['SS_smooth'] = ss_result['smooth']
    result['SS_direction'] = ss_result['direction']

    result['BP_value'] = bp_series
    result['BP_direction'] = np.where(bp_series > 0, 1, -1)

    # Combined: average of both directions
    result['FILTER_direction'] = np.where(
        (result['SS_direction'] + result['BP_direction']) > 0, 1, -1
    )
    return result


# ===================================================================
# Family 3 — Regression: LinearReg Channel & LASSO Trend
# ===================================================================

def lasso_trend(series: pd.Series, length: int = 50, alpha_l1: float = 0.01) -> pd.Series:
    """
    LASSO (L1-regularized) trend estimation.
    
    Uses L1 regularization to produce sparse trend coefficients,
    naturally suppressing small wiggles and focusing on significant trends.
    """
    src = series.astype(float).values
    n = len(src)
    trend = np.full(n, np.nan)

    x = np.arange(length).reshape(-1, 1)
    # Add polynomial features: linear + quadratic
    x_poly = np.column_stack([x, x ** 2])

    for i in range(length - 1, n):
        y = src[i - length + 1: i + 1]

        if np.any(np.isnan(y)):
            continue

        # Standardize
        y_mean = y.mean()
        y_std = y.std() + 1e-10
        y_scaled = (y - y_mean) / y_std

        # Fit LASSO with small alpha
        model = Lasso(alpha=alpha_l1, fit_intercept=True, max_iter=1000)
        model.fit(x_poly, y_scaled)

        # Predict at the current bar (last point)
        x_last = np.array([[length - 1, (length - 1) ** 2]])
        y_pred_scaled = model.predict(x_last)[0]
        trend[i] = y_pred_scaled * y_std + y_mean

    return pd.Series(trend, index=series.index)


def compute_family3_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Regression Family: LinearReg channel and LASSO trend.
    
    Returns columns:
      - LR_line: Linear regression value
      - LR_direction: +1 bullish (or price above lower band), -1 bearish
      - LASSO_trend: LASSO trend estimate
      - LASSO_direction: +1 if close > LASSO trend, -1 otherwise
      - REGRESSION_direction: combined regression direction
    """
    close = df['close'].astype(float)

    lr_result = linear_reg_trend(df, source_col='close', length=50, num_std=2.0)
    lasso_series = lasso_trend(close, length=50, alpha_l1=0.01)

    result = pd.DataFrame(index=df.index)
    result['LR_line'] = lr_result['lr_line']
    result['LR_slope'] = lr_result['slope']
    result['LR_upper'] = lr_result['upper_band']
    result['LR_lower'] = lr_result['lower_band']
    result['LR_direction'] = lr_result['direction']

    result['LASSO_trend'] = lasso_series
    lasso_dir = np.full(len(close), -1, dtype=int)
    valid = ~lasso_series.isna()
    lasso_dir[valid] = np.where(close[valid] > lasso_series[valid], 1, -1)
    result['LASSO_direction'] = lasso_dir

    # Combined
    result['REGRESSION_direction'] = np.where(
        (result['LR_direction'] + result['LASSO_direction']) > 0, 1, -1
    )
    return result


# ===================================================================
# Family 4 — Spectral: FFT Cycle Phase & Wavelet Denoising
# ===================================================================

def compute_fft_cycle_phase(df: pd.DataFrame, lookback: int = 40) -> pd.Series:
    """
    Compute cycle phase using FFT.
    Returns phase (0 to 2π) for timing entries/exits.
    Ported from mttd_system.py.
    """
    src = (df['high'] + df['low'] + df['close']) / 3.0
    n = len(df)
    phase = pd.Series(np.nan, index=df.index)

    min_period = 5
    max_period = lookback // 2

    for i in range(lookback - 1, n):
        window = src.iloc[i - lookback + 1:i + 1].values

        if np.any(np.isnan(window)):
            continue

        # Detrend
        window_detrended = window - np.mean(window)

        # Hanning window
        hann = np.hanning(lookback)
        window_w = window_detrended * hann

        # FFT
        fft_vals = np.fft.rfft(window_w)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(lookback, d=1)

        # Find dominant frequency in cycle range
        valid_mask = (freqs >= 1.0 / max_period) & (freqs <= 1.0 / min_period)
        valid_power = power[valid_mask]
        valid_freqs = freqs[valid_mask]

        if len(valid_power) > 0 and np.sum(valid_power) > 0:
            dominant_idx = np.argmax(valid_power)
            dominant_freq = valid_freqs[dominant_idx]
            dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else lookback

            cycle_pos = i % int(dominant_period) if int(dominant_period) > 0 else 0
            phase.iloc[i] = 2 * np.pi * cycle_pos / dominant_period

    return phase


def haar_dwt(signal: np.ndarray) -> tuple:
    """
    Haar Discrete Wavelet Transform (single level).
    Returns (approximation coefficients, detail coefficients).
    """
    n = len(signal)
    if n % 2 != 0:
        signal = signal[:-1]
        n -= 1

    half = n // 2
    approx = np.zeros(half)
    detail = np.zeros(half)

    for i in range(half):
        approx[i] = (signal[2 * i] + signal[2 * i + 1]) / np.sqrt(2)
        detail[i] = (signal[2 * i] - signal[2 * i + 1]) / np.sqrt(2)

    return approx, detail


def haar_idwt(approx: np.ndarray, detail: np.ndarray) -> np.ndarray:
    """Inverse Haar DWT."""
    n = len(approx)
    signal = np.zeros(2 * n)
    for i in range(n):
        signal[2 * i] = (approx[i] + detail[i]) / np.sqrt(2)
        signal[2 * i + 1] = (approx[i] - detail[i]) / np.sqrt(2)
    return signal


def wavelet_denoise(series: pd.Series, levels: int = 2, threshold: float = 0.1) -> pd.Series:
    """
    Simple wavelet denoising using Haar DWT with soft thresholding.
    """
    src = series.values.astype(float)
    n = len(src)

    # Pad to power of 2 for full decomposition
    max_pow2 = int(2 ** np.ceil(np.log2(n)))
    padded = np.pad(src, (0, max_pow2 - n), 'edge')

    # Multi-level decomposition: collect approximations and details
    details = []
    approximation = padded.copy()
    for _ in range(levels):
        if len(approximation) < 2:
            break
        a, d = haar_dwt(approximation)
        details.append(d)
        approximation = a

    # Soft threshold detail coefficients
    for i in range(len(details)):
        d = details[i]
        d_sign = np.sign(d)
        d_abs = np.abs(d)
        d_std = np.std(d_abs) + 1e-10
        d_thresh = np.maximum(d_abs - threshold * d_std, 0)
        details[i] = d_sign * d_thresh

    # Reconstruct from bottom up
    reconstructed = approximation
    for d in reversed(details):
        min_len = min(len(reconstructed), len(d))
        reconstructed = haar_idwt(reconstructed[:min_len], d[:min_len])

    # Trim back to original length
    return pd.Series(reconstructed[:n], index=series.index)


def compute_family4_spectral(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spectral Family: FFT cycle phase and wavelet denoising.
    
    Returns columns:
      - FFT_phase: Cycle phase in radians [0, 2π]
      - FFT_signal: -cos(phase) → +1 at trough (buy), -1 at peak (sell)
      - Wavelet_denoised: Wavelet-denoised price
      - Wavelet_direction: +1 if rising, -1 if falling
      - SPECTRAL_direction: combined spectral direction
    """
    close = df['close'].astype(float)

    # FFT cycle phase
    phase = compute_fft_cycle_phase(df, lookback=40)
    fft_signal = -np.cos(phase)  # +1 at trough, -1 at peak
    fft_dir = np.where(fft_signal > 0, 1, -1)

    # Wavelet denoising
    wv = wavelet_denoise(close, levels=2, threshold=0.1)
    wv_diff = pd.Series(wv).diff()
    wv_dir = np.where(wv_diff > 0, 1, -1)
    wv_dir[0] = 1

    result = pd.DataFrame(index=df.index)
    result['FFT_phase'] = phase
    result['FFT_signal'] = fft_signal
    result['FFT_direction'] = fft_dir

    result['Wavelet_denoised'] = wv
    result['Wavelet_direction'] = wv_dir

    # Combined
    result['SPECTRAL_direction'] = np.where(
        (result['FFT_direction'] + result['Wavelet_direction']) > 0, 1, -1
    )
    return result


# ===================================================================
# Family 5 — Fractal: Efficiency Ratio & Hurst Exponent
# ===================================================================

def hurst_exponent(series: pd.Series, max_lag: int = 100) -> pd.Series:
    """
    Hurst Exponent via R/S analysis.
    
    H ≈ 0.5 → random walk
    H > 0.5 → trending (positive long-range correlation)
    H < 0.5 → mean-reverting (negative correlation)
    
    Returns rolling Hurst estimate.
    """
    src = series.astype(float).values
    n = len(src)
    hurst_vals = np.full(n, np.nan)

    # Use log-returns
    log_ret = np.full(n, 0.0)
    for i in range(1, n):
        if src[i] > 0 and src[i - 1] > 0:
            log_ret[i] = np.log(src[i] / src[i - 1])

    for i in range(max_lag * 2, n):
        window = log_ret[i - max_lag * 2 + 1: i + 1]

        # R/S analysis over multiple scales
        lags = np.unique(np.logspace(0, np.log10(len(window) // 2), 10).astype(int))
        lags = lags[lags >= 2]

        if len(lags) < 3:
            continue

        rs_values = []
        for lag in lags:
            # Split into blocks of size lag
            n_blocks = len(window) // lag
            if n_blocks < 1:
                continue

            blocks = window[:n_blocks * lag].reshape(n_blocks, lag)

            # Mean-adjusted cumulative deviations
            block_means = blocks.mean(axis=1, keepdims=True)
            deviations = blocks - block_means
            cum_dev = np.cumsum(deviations, axis=1)

            # Range
            r = cum_dev.max(axis=1) - cum_dev.min(axis=1)

            # Standard deviation
            s = blocks.std(axis=1, ddof=1) + 1e-10

            # R/S ratio (mean across blocks)
            rs = (r / s).mean()
            rs_values.append(rs)

        if len(rs_values) >= 3:
            lags_use = lags[:len(rs_values)]
            log_rs = np.log(rs_values)
            log_lags = np.log(lags_use)

            # OLS fit
            slope, _, _, _, _ = stats.linregress(log_lags, log_rs)
            hurst_vals[i] = slope

    return pd.Series(hurst_vals, index=series.index)


def compute_family5_fractal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fractal Family: Efficiency Ratio and Hurst exponent.
    
    Returns columns:
      - ER: Efficiency Ratio [0, 1]
      - ER_direction: +1 if trending (ER > 0.25), -1 otherwise
      - Hurst: Hurst exponent
      - Hurst_direction: +1 if trending (H > 0.55), -1 otherwise
      - FRACTAL_direction: combined fractal direction
    """
    close = df['close'].astype(float)

    er_result = efficiency_ratio(df, source_col='close', period=14, threshold=0.25)
    hurst_series = hurst_exponent(close, max_lag=100)

    result = pd.DataFrame(index=df.index)
    result['ER'] = er_result['er']
    result['ER_direction'] = er_result['direction']

    result['Hurst'] = hurst_series
    # Trending if H > 0.55, mean-reverting or random if H <= 0.55
    hurst_dir = np.full(len(close), -1, dtype=int)
    valid = ~hurst_series.isna()
    hurst_dir[valid] = np.where(hurst_series[valid] > 0.55, 1, -1)
    result['Hurst_direction'] = hurst_dir

    # Combined
    result['FRACTAL_direction'] = np.where(
        (result['ER_direction'] + result['Hurst_direction']) > 0, 1, -1
    )
    return result


# ===================================================================
# Family 6 — GARCH: Volatility Cluster Detection
# ===================================================================

def compute_family6_garch(df: pd.DataFrame) -> pd.DataFrame:
    """
    GARCH Family: Volatility cluster detection.
    
    Returns columns:
      - rolling_vol: Short-term volatility
      - median_vol: Long-term median volatility
      - vol_ratio: Current vol / median vol
      - GARCH_direction: +1 (low vol, trade) or -1 (high vol, avoid)
    """
    vc_result = volatility_cluster(df, source_col='close', window=20,
                                   median_window=100, threshold=1.2)

    result = pd.DataFrame(index=df.index)
    result['rolling_vol'] = vc_result['rolling_vol']
    result['median_vol'] = vc_result['median_vol']
    result['vol_ratio'] = vc_result['vol_ratio']
    result['GARCH_direction'] = vc_result['direction']
    return result


# ===================================================================
# Family 7 — Entropy: Shannon Entropy & Permutation Entropy
# ===================================================================

def permutation_entropy(series: pd.Series, m: int = 4, delay: int = 1, window: int = 60) -> pd.Series:
    """
    Permutation Entropy (Bandt & Pompe, 2002) using sliding window.
    
    Measures complexity based on ordinal patterns (permutations) of length m.
    - Low PE → deterministic/trending dynamics
    - High PE → random/complex dynamics
    
    Uses a sliding window of `window` delay vectors to compute the
    pattern distribution, normalized to [0, 1].
    """
    src = series.values.astype(float)
    n = len(src)
    pe_vals = np.full(n, np.nan)

    max_entropy = np.log(int(math.factorial(m)))

    def ordinal_pattern_tuple(vec):
        """Get ordinal pattern as a tuple of rank positions."""
        # For equal values, stable sort ensures first occurrence has lower rank
        indices = np.argsort(vec, kind='mergesort')
        # Map: each original position -> its rank
        ranks = [0] * m
        for rank, idx in enumerate(indices):
            ranks[idx] = rank
        return tuple(ranks)

    # Need at least (m-1)*delay + 1 + window samples
    min_samples = (m - 1) * delay + window
    if n < min_samples:
        return pd.Series(pe_vals, index=series.index)

    for i in range(min_samples, n):
        # Get the last `window` delay vectors
        v_start = i - window + 1 - (m - 1) * delay
        if v_start < 0:
            continue

        # Count patterns in the window
        pattern_counts = {}
        for v in range(window):
            vec = [src[v_start + v + k * delay] for k in range(m)]
            pat = ordinal_pattern_tuple(vec)
            pattern_counts[pat] = pattern_counts.get(pat, 0) + 1

        total = sum(pattern_counts.values())
        if total < 2:
            continue

        # Shannon entropy
        pe = 0.0
        for count in pattern_counts.values():
            p = count / total
            if p > 0:
                pe -= p * np.log(p)

        pe_vals[i] = pe / max_entropy if max_entropy > 0 else 0.0

    return pd.Series(pe_vals, index=series.index)


def compute_family7_entropy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entropy Family: Shannon entropy and Permutation entropy.
    
    Returns columns:
      - Shannon_entropy: Rolling Shannon entropy (bits)
      - Shannon_direction: +1 if low entropy (trending), -1 if high entropy
      - Perm_entropy: Permutation entropy [0, 1]
      - Perm_direction: +1 if low PE (deterministic), -1 if high PE
      - ENTROPY_direction: combined entropy direction
    """
    close = df['close'].astype(float)

    se_result = shannon_entropy(df, source_col='close', window=15, bins=6, threshold=2.5)
    pe_series = permutation_entropy(close, m=4, delay=1)

    result = pd.DataFrame(index=df.index)
    result['Shannon_entropy'] = se_result['entropy']
    result['Shannon_direction'] = se_result['direction']

    result['Perm_entropy'] = pe_series
    pe_dir = np.full(len(close), -1, dtype=int)
    valid = ~pe_series.isna()
    pe_dir[valid] = np.where(pe_series[valid] < 0.88, 1, -1)
    result['Perm_direction'] = pe_dir

    # Combined
    result['ENTROPY_direction'] = np.where(
        (result['Shannon_direction'] + result['Perm_direction']) > 0, 1, -1
    )
    return result


# ===================================================================
# Family 8 — Chaos: Phase Space Reconstruction
# ===================================================================

def phase_space_reconstruction(series: pd.Series, embed_dim: int = 5, delay: int = 2) -> pd.Series:
    """
    Phase Space Reconstruction using Takens' embedding theorem.
    
    Reconstructs the attractor dynamics from a 1D time series:
      X(t) = [s(t), s(t-τ), s(t-2τ), ..., s(t-(m-1)τ)]
    
    Returns a "chaos index" based on the recurrence properties:
    - Low chaos index → regular/periodic dynamics (trend)
    - High chaos index → chaotic dynamics (noise, avoid trading)
    
    The index is the average pairwise divergence rate in reconstructed space.
    """
    src = series.astype(float).values
    n = len(src)
    chaos_idx = np.full(n, np.nan)

    # Use first differences (log returns) for embedding
    ret = np.full(n, 0.0)
    for i in range(1, n):
        if src[i] > 0 and src[i - 1] > 0:
            ret[i] = np.log(src[i] / src[i - 1])

    min_samples = embed_dim * delay + 10

    for i in range(min_samples, n):
        # Build delay vectors from recent ret window (last 100 bars)
        window_size = min(100, i - min_samples + 10)
        window = ret[i - window_size + 1:i + 1]

        if len(window) < embed_dim * delay:
            continue

        # Reconstruct phase space vectors
        vectors = []
        for j in range(len(window) - embed_dim * delay + 1):
            vec = np.array([window[j + k * delay] for k in range(embed_dim)])
            # Skip zero vectors (no movement)
            if np.std(vec) > 1e-10:
                vectors.append(vec)

        if len(vectors) < 5:
            continue

        # Compute pairwise Euclidean distances
        vectors = np.array(vectors)
        n_vectors = len(vectors)

        # Sample a subset of pairs for efficiency
        max_pairs = min(200, n_vectors * (n_vectors - 1) // 2)
        distances = []

        for _ in range(max_pairs):
            a = np.random.randint(0, n_vectors)
            b = np.random.randint(0, n_vectors)
            if a != b:
                d = np.linalg.norm(vectors[a] - vectors[b])
                distances.append(d)

        if len(distances) < 10:
            continue

        # Chaos index: std / mean of distances
        # Regular dynamics → distances cluster around certain values
        # Chaotic dynamics → wide spread of distances
        dist_mean = np.mean(distances)
        dist_std = np.std(distances)
        chaos_idx[i] = dist_std / dist_mean if dist_mean > 1e-10 else 0.0

    return pd.Series(chaos_idx, index=series.index)


def compute_family8_chaos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chaos Family: Phase Space Reconstruction.
    
    Returns columns:
      - Chaos_index: Divergence rate in reconstructed phase space
      - Chaos_direction: +1 if low chaos (regular/trending), -1 if high chaos
    """
    close = df['close'].astype(float)

    chaos_series = phase_space_reconstruction(close, embed_dim=5, delay=2)

    result = pd.DataFrame(index=df.index)
    result['Chaos_index'] = chaos_series
    chaos_dir = np.full(len(close), -1, dtype=int)
    valid = ~chaos_series.isna()
    # Low chaos (std/mean < 1.0) → regular/trending → +1
    # High chaos → avoid
    chaos_dir[valid] = np.where(chaos_series[valid] < 1.0, 1, -1)
    result['Chaos_direction'] = chaos_dir

    # Normalized chaos index (0 = deterministic, 1 = chaotic)
    norm_chaos = chaos_series.copy()
    valid_vals = chaos_series[valid]
    if len(valid_vals) > 0:
        vmin, vmax = valid_vals.min(), valid_vals.max()
        if vmax > vmin:
            norm_chaos[valid] = (valid_vals - vmin) / (vmax - vmin)
    result['Chaos_normalized'] = norm_chaos
    return result


# ===================================================================
# Family 9 — Bayesian: HMM Regime Detection
# ===================================================================

def compute_family9_bayesian(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bayesian Family: HMM regime detection.
    
    Returns columns:
      - HMM_state: Decoded HMM state (0=BULL, 1=BEAR, 2=SIDEWAYS)
      - HMM_bull_prob: Probability of being in BULL state
      - HMM_direction: +1 (BULL) or -1 (BEAR/SIDEWAYS)
    """
    hmm_result = hmm_regime(df, source_col='close', n_states=3,
                            window=100, lookback=250, seed=42)

    result = pd.DataFrame(index=df.index)
    result['HMM_state'] = hmm_result['state']
    result['HMM_bull_prob'] = hmm_result['bull_prob']
    result['HMM_direction'] = hmm_result['direction']
    return result


# ===================================================================
# Family 10 — ML Hybrid: Composite Adaptive Scoring
# ===================================================================

def compute_family10_ml_hybrid(
    df: pd.DataFrame,
    family_signals: dict,
    lookback: int = 60
) -> pd.DataFrame:
    """
    ML Hybrid Family: Adaptive composite scoring.
    
    Combines all 9 family direction signals into a single meta-signal
    with adaptive weights based on recent (lookback) Sharpe contribution.
    
    If a family has been profitable recently, it gets higher weight.
    """
    n = len(df)

    # Collect all family direction columns
    family_dirs = {}
    for fname, fdf in family_signals.items():
        dir_col = None
        for col in fdf.columns:
            if '_direction' in col and fname.upper() in col.upper():
                dir_col = col
                break
            if col == f'{fname.upper()}_direction' or col == f'{fname}_direction':
                dir_col = col
                break
        if dir_col is None:
            # Try to find direction column with family prefix
            for col in fdf.columns:
                if col.endswith('_direction'):
                    dir_col = col
                    break
        if dir_col is not None:
            family_dirs[fname] = fdf[dir_col].astype(float).values

    if len(family_dirs) == 0:
        result = pd.DataFrame(index=df.index)
        result['ML_composite'] = 0.0
        result['ML_direction'] = -1
        return result

    # Direction matrix: (n, n_families) with values in {-1, 1}
    family_names = list(family_dirs.keys())
    n_families = len(family_names)
    dir_matrix = np.column_stack([family_dirs[name] for name in family_names])

    # Returns for adaptive weighting
    ret = df['close'].pct_change().fillna(0).values

    # Adaptive weights: recent Sharpe per family
    weights = np.ones(n_families) / n_families  # equal weight default

    # For each bar, recalculate weights if enough history
    composite = np.full(n, np.nan)

    for i in range(n):
        if np.any(np.isnan(dir_matrix[i])):
            continue

        # Rebalance weights periodically
        if i >= lookback and i % 10 == 0:
            # Recent strategy returns for each family
            start = i - lookback
            family_rets = dir_matrix[start:i] * ret[start:i, np.newaxis]

            # Sharpe per family
            sharpes = np.full(n_families, 0.0)
            for f in range(n_families):
                fr = family_rets[:, f]
                if fr.std() > 1e-10:
                    sharpes[f] = fr.mean() / fr.std() * np.sqrt(365)

            # Softmax weighting (positive Sharpe → higher weight)
            sharpes = np.clip(sharpes, -10, 10)  # clip extreme values
            exp_sharpe = np.exp(sharpes)
            weights = exp_sharpe / exp_sharpe.sum()

        # Weighted vote
        valid_mask = ~np.isnan(dir_matrix[i])
        if valid_mask.sum() > 0:
            w = weights[valid_mask]
            w = w / w.sum()
            composite[i] = np.dot(dir_matrix[i, valid_mask], w)

    result = pd.DataFrame(index=df.index)
    result['ML_composite'] = composite
    result['ML_direction'] = np.where(composite > 0, 1, -1)
    return result


# ===================================================================
# Master Feature Generator
# ===================================================================

def generate_multi_principle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate all multi-principle features from the 10 statistical families.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV data (columns: open, high, low, close, volume).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all feature columns from each family.
    """
    df = df.copy()

    print("[Multi-Principle] Generating features from 10 statistical families...")

    # Family 1: Smoothing
    print("  Family 1/10: Smoothing (KAMA, MAMA)...")
    f1 = compute_family1_smoothing(df)

    # Family 2: Filtering
    print("  Family 2/10: Filtering (SuperSmoother, Bandpass)...")
    f2 = compute_family2_filtering(df)

    # Family 3: Regression
    print("  Family 3/10: Regression (LinearReg, LASSO)...")
    f3 = compute_family3_regression(df)

    # Family 4: Spectral
    print("  Family 4/10: Spectral (FFT Cycle, Wavelet)...")
    f4 = compute_family4_spectral(df)

    # Family 5: Fractal
    print("  Family 5/10: Fractal (Efficiency Ratio, Hurst)...")
    f5 = compute_family5_fractal(df)

    # Family 6: GARCH
    print("  Family 6/10: GARCH (Volatility Cluster)...")
    f6 = compute_family6_garch(df)

    # Family 7: Entropy
    print("  Family 7/10: Entropy (Shannon, Permutation)...")
    f7 = compute_family7_entropy(df)

    # Family 8: Chaos
    print("  Family 8/10: Chaos (Phase Space)...")
    f8 = compute_family8_chaos(df)

    # Family 9: Bayesian
    print("  Family 9/10: Bayesian (HMM Regime)...")
    f9 = compute_family9_bayesian(df)

    # Combine all features
    all_features = pd.concat([f1, f2, f3, f4, f5, f6, f7, f8, f9], axis=1)

    # Family 10: ML Hybrid (requires all family signals)
    print("  Family 10/10: ML Hybrid (Composite Scoring)...")
    family_signals = {
        'SMOOTH': f1, 'FILTER': f2, 'REGRESSION': f3,
        'SPECTRAL': f4, 'FRACTAL': f5, 'GARCH': f6,
        'ENTROPY': f7, 'CHAOS': f8, 'BAYESIAN': f9
    }
    f10 = compute_family10_ml_hybrid(df, family_signals, lookback=60)
    all_features = pd.concat([all_features, f10], axis=1)

    # Add ATR for normalization
    from indicators_helper import atr
    all_features['ATR'] = atr(df['high'], df['low'], df['close'], 14)

    print(f"  Done. Total feature columns: {len(all_features.columns)}")
    return all_features


def generate_multi_principle_signals(
    df: pd.DataFrame,
    features: pd.DataFrame = None,
    min_hold: int = 10,
    ml_threshold: float = 0.0,
    require_agreement: int = 4
) -> pd.DataFrame:
    """
    Generate binary position signals from multi-principle features.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data.
    features : pd.DataFrame, optional
        Pre-computed features from generate_multi_principle_features().
        If None, computes them.
    min_hold : int
        Minimum holding period (days).
    ml_threshold : float
        ML composite score threshold for entry.
    require_agreement : int
        Minimum number of family directions that must agree.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with position signals.
    """
    if features is None:
        features = generate_multi_principle_features(df)

    df = df.copy()

    # Collect all family direction columns
    dir_cols = [c for c in features.columns if c.endswith('_direction')]

    if len(dir_cols) < 2:
        raise ValueError(f"Only {len(dir_cols)} direction columns found. Need at least 2.")

    # Full signal matrix (all families)
    signal_matrix = features[dir_cols].copy()

    # Count how many families agree on bullish (+1)
    bullish_count = (signal_matrix > 0).sum(axis=1).values
    n_families = len(dir_cols)

    # Majority signal: +1 if >50% families bullish, -1 if >50% bearish, 0 else
    majority = np.where(bullish_count > n_families / 2, 1,
                        np.where(bullish_count < n_families / 2, -1, 0))

    # ML hybrid refines the signal
    ml_composite = features['ML_composite'].values

    # Final entry: majority bullish AND ML score > threshold
    # Final exit: majority bearish OR ML score < -threshold
    n = len(df)
    pos = 0.0
    positions = np.zeros(n)
    hold_days = 0

    for i in range(n):
        if np.isnan(majority[i]) or np.isnan(ml_composite[i]):
            positions[i] = pos
            continue

        if pos > 0:
            hold_days += 1
        else:
            hold_days = 0

        can_exit = hold_days >= min_hold

        if pos == 0.0:
            # ENTRY: majority bullish AND ML positive AND sufficient agreement
            agreement = bullish_count[i] if not np.isnan(bullish_count[i]) else 0
            if (majority[i] == 1 and ml_composite[i] > ml_threshold
                    and agreement >= require_agreement):
                pos = 1.0
                hold_days = 0
        else:
            # EXIT: majority bearish OR ML negative (with min_hold)
            if can_exit and (majority[i] == -1 or ml_composite[i] < -ml_threshold):
                pos = 0.0
                hold_days = 0

        positions[i] = pos

    result = df[['open', 'high', 'low', 'close', 'volume']].copy()
    result['Position'] = positions
    result['Majority_vote'] = majority
    result['ML_composite'] = ml_composite
    result['Agreement'] = bullish_count

    return result


# ===================================================================
# Standalone test
# ===================================================================
if __name__ == '__main__':
    import json
    import pathlib
    import os

    data_path = pathlib.Path(__file__).resolve().parent / 'data' / 'btc_daily.json'
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        sys.exit(1)

    with open(data_path) as f:
        raw = json.load(f)

    df = pd.DataFrame(raw['aligned_data'])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df = df[df.index >= '2018-01-01']

    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    print()

    # Generate features
    features = generate_multi_principle_features(df)

    print()
    print("=" * 70)
    print("FEATURE SUMMARY")
    print("=" * 70)
    print(f"Total feature columns: {len(features.columns)}")
    print(f"Date range: {features.index[0]} to {features.index[-1]}")

    # Check direction columns
    dir_cols = [c for c in features.columns if c.endswith('_direction')]
    print(f"\nDirection columns ({len(dir_cols)}):")
    for col in dir_cols:
        non_null = features[col].notna().sum()
        vals = sorted(features[col].unique())
        print(f"  {col:30s}  non-null: {non_null:5d}  values: {vals}")

    # Check all numeric columns for NaN coverage
    print("\nNumeric columns range:")
    for col in features.columns:
        if col.endswith('_direction'):
            continue
        non_null = features[col].notna().sum()
        if non_null > 0:
            vmin = features[col].min()
            vmax = features[col].max()
            print(f"  {col:30s}  non-null: {non_null:5d}/{len(features)}  range: [{vmin:.4f}, {vmax:.4f}]")
        else:
            print(f"  {col:30s}  non-null: {non_null:5d}/{len(features)}  (ALL NaN)")

    # Generate signals
    print("\n" + "=" * 70)
    print("SIGNAL GENERATION")
    print("=" * 70)
    signals = generate_multi_principle_signals(df, features=features, min_hold=10)

    in_pos = (signals['Position'] > 0).sum()
    total = len(signals)
    print(f"Position: {in_pos}/{total} bars ({in_pos/total*100:.1f}%)")

    # Count trades
    pos_changes = signals['Position'].diff().fillna(0)
    n_trades = int((pos_changes.abs() > 0).sum() // 2)
    print(f"Number of trades: {n_trades}")

    # Basic performance
    returns = df['close'].pct_change()
    strategy_returns = returns * signals['Position'].shift(1)
    strategy_returns = strategy_returns.dropna()
    equity = (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25
    cagr = (equity.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0

    # Win rate
    in_position = False
    entry_price = None
    trade_returns = []
    for i, (date, row) in enumerate(signals.iterrows()):
        if row['Position'] == 1.0 and not in_position:
            in_position = True
            entry_price = row['close']
        elif row['Position'] == 0.0 and in_position:
            in_position = False
            if entry_price is not None:
                trade_ret = (row['close'] - entry_price) / entry_price
                trade_returns.append(trade_ret)

    winning = sum(1 for r in trade_returns if r > 0)
    win_rate = winning / len(trade_returns) * 100 if trade_returns else 0

    print(f"\nBasic Performance:")
    print(f"  CAGR:   {cagr * 100:.2f}%")
    print(f"  Sharpe: {sharpe:.2f}")
    print(f"  Win Rate: {win_rate:.1f}% ({winning}/{len(trade_returns)})")
    print(f"  Trades: {len(trade_returns)}")
    print(f"  Avg Hold: {signals['Position'].sum() / max(1, len(trade_returns)):.0f} days")

    print("\n✅ Multi-Principle signal generator completed successfully.")
