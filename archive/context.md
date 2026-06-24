# MTTD System Indicator Audit Report

**Auditor:** Quantitative Research Scientist  
**Date:** 2026-06-23  
**Scope:** All 10 individual indicators  
**Methodology:** Radical skepticism — every positive result is guilty until proven innocent

---

## Executive Summary

Audited all 10 indicators across oscillator and perpetual modules. Found **1 confirmed critical bug** (polynomial regression evaluation point), **1 significant architectural concern** (factor overlap across trend indicators), and **0 look-ahead biases** across all indicators. All indicators use proper rolling-window or stateful bar-by-bar computation with no bfill, shift(-n), or full-sample statistics.

**Overall Assessment:** The indicators are mechanically sound (no look-ahead bias), but many are variations of the same trend-following factor. The ensemble's diversification benefit is questionable if 7 of 10 indicators are just different smoothings of the same momentum/trend signal.

---

## Indicator 1: Kalman Filtered RSI Oscillator

**File:** `oscillator/kalman_filtered_rsi_oscillator.py`

### Signal Quality: GOOD

| Dimension | Assessment |
|-----------|------------|
| Economic Hypothesis | Kalman filter denoises price, then RSI on filtered signal gives cleaner momentum reading. Reasonable hypothesis: noise reduction should improve signal-to-noise ratio. |
| Factor Classification | **Momentum** — RSI is a momentum oscillator. Kalman filtering is a preprocessing step, not a new factor. |
| Expected IC | 0.02–0.05 (RSI variants typically modest) |
| Novelty | LOW — This is a smoothed RSI, not a genuinely novel signal |

### Look-Ahead Bias Risk: NONE

- No bfill, shift(-n), or future data usage
- Kalman filter is bar-by-bar iterative (lines 37–57)
- RSI computed via standard helper functions
- Normalization uses rolling windows of 100 bars (lines 70–72), not full-sample

### Stationarity: LOW CONCERN

RSI is inherently differenced/normalized, making it approximately stationary. The Kalman filter smoothing on non-stationary prices is acceptable because RSI re-normalizes.

### Parameter Sensitivity: HIGH

| Parameter | Default | Sensitivity | Justification |
|-----------|---------|-------------|---------------|
| `process_noise` | 0.01 | HIGH | Controls filter adaptation speed. Small changes significantly alter filtered output |
| `measurement_noise` | 3.0 | HIGH | Combined with low process_noise, creates heavy smoothing. Ratio determines Kalman gain |
| `rsi_period` | 14 | LOW | Standard RSI period |
| `n` | 5 | NONE | All n states are identical (see below) |

### Code Issues Found

1. **Redundant parallel states (lines 38–39):** The Kalman filter maintains `n=5` parallel state estimates, but all are initialized identically and updated with the same formula. They converge to the same value. Only `state_estimate[0]` is used (line 57). This is unnecessary complexity but not a bug.

2. **Division-by-zero risk (line 71):** `highest_rsi - lowest_rsi` could theoretically be 0 if RSI is constant over 100 bars. Extremely unlikely for RSI but not guarded.

### Factor Disguise Assessment

**This is a momentum factor in disguise.** The Kalman filter is a preprocessing step that reduces noise, but the underlying signal is RSI — a well-known momentum oscillator. After Fama-French decomposition, this would load primarily on momentum (UMD) factor. The Kalman filtering may slightly improve IC by reducing measurement noise, but it does not create a new alpha source.

### Verdict

The Kalman RSI fix (heavy smoothing via high measurement_noise) appears correct. The signal is noisier than a standard RSI because Kalman filtering on 1-minute BTC data is aggressive smoothing. The normalization to [-0.5, 0.5] over 100 bars is clean. No bugs found.

---

## Indicator 2: Z-SMMA QuantEdgeB

**File:** `oscillator/z_smma_quantedgeb.py`

### Signal Quality: SUSPICIOUS

| Dimension | Assessment |
|-----------|------------|
| Economic Hypothesis | Z-score of SMMA relative to its own EMA and rolling stdev. **CRITICAL ISSUE:** The direction logic is INVERTED from what the Z-score naming implies. |
| Factor Classification | **Trend/Momentum** — Goes LONG when Z > 0.1 (price above trend), SHORT when Z < -0.1 (price below trend). This is trend-following, not mean-reversion. |
| Expected IC | 0.02–0.04 |
| Novelty | LOW — Z-score transformation is standard. The direction logic makes it a momentum signal. |

### Look-Ahead Bias Risk: NONE

- `mean_val = ema(smma, len_z)` — rolling EMA, no look-ahead
- `sd_val = stdev(smma, len_z)` — rolling stdev, no look-ahead
- Stateful loop (lines 32–40) is bar-by-bar
- No bfill, shift(-n), or future data

### Stationarity: LOW CONCERN

Z-score transformation (value - mean) / std inherently removes level, producing a stationary signal. Good practice.

### Parameter Sensitivity: HIGH

| Parameter | Default | Sensitivity | Justification |
|-----------|---------|-------------|---------------|
| `lu` | 0.1 | HIGH | Threshold for long signal. Very tight — small changes flip signals |
| `su` | -0.1 | HIGH | Threshold for short signal. Symmetric to lu |
| `len_smma` | 12 | MEDIUM | SMMA smoothing period |
| `len_z` | 30 | MEDIUM | Lookback for mean/std calculation |

### Code Issues Found

1. **Misleading naming / inverted logic (lines 28–29):** The variable `smma_z` is a Z-score. Conventionally, a positive Z-score means the value is ABOVE its mean — suggesting mean reversion (short). But the code goes LONG when Z > 0.1. This is trend-following behavior. While not a bug per se, it's misleading and could cause errors if someone modifies the code assuming mean-reversion semantics.

2. **State persistence issue (lines 32–40):** Once `qb` goes to 1, it stays at 1 until a short signal resets it. This hysteresis means the indicator can stay in a position for extended periods without new signals. This is intentional but reduces signal frequency.

### Factor Disguise Assessment

**This is a momentum/trend factor.** The Z-score transformation doesn't change the underlying signal direction — it's still "go long when price is above its recent average." After factor decomposition, this loads on momentum (UMD). The Z-score normalization may provide marginal benefit by making the signal scale-invariant, but it's fundamentally the same bet as other trend indicators.

### Verdict

The indicator works mechanically but the direction logic is counter-intuitive. A practitioner expecting mean-reversion from a "Z-score" indicator would get the opposite. Not a bug, but a design concern.

---

## Indicator 3: Median RSI SD QuantEdgeB

**File:** `oscillator/median_rsi_sd_quantedgeb.py`

### Signal Quality: GOOD

| Dimension | Assessment |
|-----------|------------|
| Economic Hypothesis | RSI of median-smoothed price, with SD band filter for breakout confirmation. Combines outlier-resistant smoothing (median) with momentum (RSI) and volatility filter (SD bands). |
| Factor Classification | **Momentum + Breakout** — Long requires RSI > 65 AND price above upper SD band. This is a momentum breakout signal. |
| Expected IC | 0.03–0.05 |
| Novelty | MEDIUM — Median smoothing of price before RSI is a reasonable noise reduction technique |

### Look-Ahead Bias Risk: NONE

- `percentile_nearest_rank` uses rolling windows (line 11–19)
- RSI computed on median series — clean
- SD computed on median series — clean
- Stateful loop (lines 48–55) is bar-by-bar
- No bfill, shift(-n), or future data

### Stationarity: LOW CONCERN

Median removes outliers, RSI normalizes to [0, 100]. The combination produces a roughly stationary signal.

### Parameter Sensitivity: MEDIUM

| Parameter | Default | Sensitivity | Justification |
|-----------|---------|-------------|---------------|
| `len_rsi` | 21 | MEDIUM | RSI period — standard range |
| `len_median` | 10 | MEDIUM | Median smoothing window |
| `lu` | 65.0 | MEDIUM | Long threshold — asymmetric |
| `su` | 45.0 | MEDIUM | Short threshold — asymmetric |

### Code Issues Found

1. **Asymmetric thresholds (lu=65, su=45):** The long threshold requires stronger momentum (RSI > 65) than the short threshold (RSI < 45). This biases the indicator toward fewer long entries and more short entries. Economically, this could be justified if BTC has asymmetric momentum dynamics, but it's not explained.

2. **`min_periods=1` in percentile_nearest_rank (line 13):** The rolling window requires `length` periods for computation but `min_periods=length` is enforced in the inner check. Actually, looking again — the `source.rolling(window=length, min_periods=length)` is correct. No issue.

### Factor Disguise Assessment

**This is a momentum factor with breakout filter.** The RSI component is pure momentum. The SD band filter adds a volatility condition. After decomposition, it loads on momentum (UMD) with a volatility overlay.

### Verdict

Clean implementation with reasonable signal design. The asymmetric thresholds are the only notable design choice.

---

## Indicator 4: Polynomial Deviation Bands ⚠️ CRITICAL BUG

**File:** `perpetual/polynomial_deviation_bands.py`

### Signal Quality: GOOD (if bug is fixed)

| Dimension | Assessment |
|-----------|------------|
| Economic Hypothesis | Polynomial regression estimates local trend, deviation bands capture volatility. Band crossovers signal trend changes. Economically sound: regression-based trend estimation with adaptive volatility bands. |
| Factor Classification | **Trend** — Regression-based trend following |
| Expected IC | 0.03–0.06 (regression + band crossover) |
| Novelty | MEDIUM — Polynomial regression for trend estimation is known but less common than EMA-based approaches |

### Look-Ahead Bias Risk: NONE

- Polynomial regression uses only data within the window (lines 117–122)
- Deviation bands use rolling statistics
- Stateful trend logic (lines 148–158) is bar-by-bar
- No bfill, shift(-n), or future data

### Stationarity: LOW CONCERN

Polynomial regression is a local model — it fits within each window independently. No stationarity assumption required.

### Parameter Sensitivity: HIGH

| Parameter | Default | Sensitivity | Justification |
|-----------|---------|-------------|---------------|
| `regressions_length` | 14 | HIGH | Regression window — changing this changes the "locality" of the trend estimate |
| `multiplier` | 1.5 | HIGH | Band width — directly affects signal frequency |
| `dev_type` | "Standard Deviation" | HIGH | 10 different deviation types available, each with different behavior |
| `deg` | "2nd" | HIGH | Polynomial degree — higher degrees overfit to noise |

### 🚨 CRITICAL BUG: Polynomial Evaluation Point

```python
# Lines 124-127
def get_poly_fit_val(y_vals, degree):
    x = np.arange(len(y_vals))  # x = [0, 1, 2, ..., period-1]
    coeffs = np.polyfit(x, y_vals, degree)
    return coeffs[-1]  # BUG: This is P(0), not P(period-1)
```

**The polynomial is evaluated at x=0 (the OLDEST bar in the window), not at x=period-1 (the CURRENT bar).**

- `x = np.arange(len(y_vals))` creates `[0, 1, 2, ..., period-1]`
- `np.polyfit` returns coefficients `[a_d, a_{d-1}, ..., a_1, a_0]`
- `coeffs[-1] = a_0 = P(0)` — the value at the START of the window
- **Correct:** `np.polyval(coeffs, period-1)` — the value at the END (current bar)

**Impact:** For a linear trend (deg=1) with slope m, the regression value is underestimated by approximately `m * (period-1)`. In an uptrend, this makes the bands too low, causing `close > upper_band` to trigger more often → **bullish bias**.

**Comparison with `linreg` helper:** The project's own `linreg` function (indicators_helper.py:86-104) correctly evaluates at `x = length - 1 - offset`, confirming this is a bug, not a design choice.

**Fix:** Change line 127 from `return coeffs[-1]` to `return np.polyval(coeffs, len(y_vals) - 1)`.

### Code Issues Found

1. **CRITICAL: Evaluation point bug (line 127)** — As described above. Causes systematic bullish bias in trending markets.

2. **`min_periods=1` in FRAMA helper (lines 14-15):** The FRAMA calculation uses `min_periods=1` for rolling windows, which means the dimension estimate is noisy in the first few bars. Not a bug but affects cold-start behavior.

### Factor Disguise Assessment

**This is a trend factor using polynomial regression.** After decomposition, it loads on trend. The polynomial approach may capture non-linear trends better than EMA, but it's still fundamentally a trend signal.

### Verdict

**BLOCKING BUG:** The polynomial evaluation point must be fixed before this indicator can be trusted. The current implementation introduces a systematic bias.

---

## Indicator 5: Gaussian Smooth Trend QuantEdgeB

**File:** `perpetual/gaussian_smooth_trend_quantedgeb.py`

### Signal Quality: GOOD

| Dimension | Assessment |
|-----------|------------|
| Economic Hypothesis | Multi-layer smoothing (DEMA → Gaussian → SMMA) with adaptive SD bands. The Gaussian filter provides frequency-domain smoothing that preserves trend better than simple averaging. |
| Factor Classification | **Trend** — Multi-smoothed trend following |
| Expected IC | 0.03–0.05 |
| Novelty | LOW — This is a sophisticated smoothing pipeline, not a new alpha source |

### Look-Ahead Bias Risk: NONE

- Gaussian filter (lines 38–50): backward-looking weights `weights[i]` applied to `dema_vals[t-i]`. Only past data.
- SMMA calculation (lines 60–72): iterative, uses only past values
- SD filter: rolling window
- No bfill, shift(-n), or future data

### Stationarity: LOW CONCERN

Multi-layer smoothing produces a non-stationary output (smoothed price). The band crossover logic for direction is the meaningful signal, which is approximately stationary.

### Parameter Sensitivity: HIGH

| Parameter | Default | Sensitivity | Justification |
|-----------|---------|-------------|---------------|
| `len_dema` | 7 | MEDIUM | Initial smoothing |
| `len_fg` | 4 | HIGH | Gaussian filter length — short window |
| `sigma_fg` | 2.0 | HIGH | Gaussian sigma — controls smoothing width |
| `len_s` | 12 | MEDIUM | SMMA smoothing |
| `len_sd` | 30 | MEDIUM | SD lookback |
| `mult_sdup` | 2.5 | HIGH | Asymmetric upper band multiplier |
| `mult_sddn` | 1.8 | HIGH | Asymmetric lower band multiplier |

### Code Issues Found

1. **Asymmetric multipliers (2.5 vs 1.8):** Upper band is wider than lower band. This means the indicator requires a larger move to trigger a bullish signal than a bearish one. This is a deliberate design choice (conservative on longs) but not economically justified in the code comments.

2. **SMMA initialization (lines 64–66):** The SMMA is initialized with DEMA values, then transitions to SMMA calculation. This hybrid initialization is non-standard but not incorrect.

### Factor Disguise Assessment

**This is a trend factor with sophisticated smoothing.** The Gaussian filter provides marginally different frequency response than EMA, but the economic signal is the same: trend following. After decomposition, it loads on trend. **High correlation expected with P-Motion Trend and DEMA ATR.**

### Verdict

Clean implementation. The multi-layer smoothing is well-constructed. The main concern is factor overlap with other trend indicators in the ensemble.

---

## Indicator 6: ALMA Lag VI Research

**File:** `perpetual/alma_lag_viresearch.py`

### Signal Quality: GOOD

| Dimension | Assessment |
|-----------|------------|
| Economic Hypothesis | ALMA (Arnaud Legoux Moving Average) provides low-lag, low-overshoot smoothing. Direction is determined by ALMA slope + price position relative to ALMA. |
| Factor Classification | **Trend/Momentum** — ALMA-based trend following |
| Expected IC | 0.03–0.05 |
| Novelty | LOW — ALMA is a known MA variant. The direction logic is standard. |

### Look-Ahead Bias Risk: NONE

- ALMA calculation (lines 10–21): rolling window, backward-looking weights
- Stateful direction loop (lines 31–45): bar-by-bar
- No bfill, shift(-n), or future data

### Stationarity: LOW CONCERN

ALMA is a smoothed price (non-stationary). The direction signal is based on slope + position, which is a trend indicator — trend indicators assume persistence, not stationarity.

### Parameter Sensitivity: MEDIUM

| Parameter | Default | Sensitivity | Justification |
|-----------|---------|-------------|---------------|
| `len_subject` | 78 | HIGH | Long lookback — slow response to regime changes |
| offset | 0.85 | LOW | Standard ALMA parameter |
| sigma | 6.0 | LOW | Standard ALMA parameter |

### Code Issues Found

1. **Long lookback (78 bars):** On 1-minute data, this is ~1.3 hours. On daily data, ~3 months. The indicator will be slow to respond to rapid price changes. This is a design choice but limits responsiveness.

2. **No bugs detected.** The implementation is clean and correct.

### Factor Disguise Assessment

**Pure trend factor.** ALMA is a moving average, and the direction logic is a standard MA crossover variant. After decomposition, loads on trend.

### Verdict

Clean, well-implemented indicator. The long lookback makes it a slow trend follower — suitable for position trading, not scalping.

---

## Indicator 7: Adaptive Regime Cloud

**File:** `perpetual/adaptive_regime_cloud.py`

### Signal Quality: EXCELLENT

| Dimension | Assessment |
|-----------|------------|
| Economic Hypothesis | Uses Hurst exponent to detect market regime (trending vs mean-reverting vs random), then adapts cloud parameters accordingly. This is genuinely adaptive and theoretically well-motivated. |
| Factor Classification | **Regime-adaptive trend/mean-reversion hybrid** — Switches between trend-following and mean-reversion based on regime detection. |
| Expected IC | 0.04–0.07 (regime adaptation could improve signal quality beyond static indicators) |
| Novelty | HIGH — Regime-adaptive indicators are less common and potentially more robust |

### Look-Ahead Bias Risk: NONE

- Hurst exponent (lines 34–57): R/S analysis on past log-returns only
- Log returns (lines 25–31): past data only
- Adaptive EMA (lines 82–89): iterative, past data only
- Signal logic (lines 95–107): uses current and previous bar data only
- No bfill, shift(-n), or future data

### Stationarity: LOW CONCERN

The indicator explicitly handles non-stationarity through regime detection. The Hurst exponent measures long-range dependence, which is appropriate for non-stationary financial data.

### Parameter Sensitivity: MEDIUM

| Parameter | Default | Sensitivity | Justification |
|-----------|---------|-------------|---------------|
| `lookback` | 50 | MEDIUM | Hurst exponent calculation window |
| `adaptive_period` | 30 | MEDIUM | Base EMA period |
| `volatility_period` | 10 | LOW | Short-term vol estimation |
| `cloud_expansion` | 1.6 | MEDIUM | Band width scaling |
| `regime_threshold` | 0.65 | MEDIUM | Hurst threshold for regime classification |
| `fast_response` | True | LOW | Binary toggle for response speed |

### Code Issues Found

1. **Hurst exponent clamping (line 56):** `hurst[i] = np.max([0.3, np.min([0.7, H])])` clamps H to [0.3, 0.7]. This prevents extreme values but may mask genuine regime transitions. Reasonable safety measure.

2. **Cloud width scaling (line 91):** `cloud_width = base_width * width_multiplier * c_val` scales with price level. For BTC at different price levels, the cloud adapts. Good design.

3. **No bugs detected.** The implementation is clean and the regime-adaptive logic is well-structured.

### Factor Disguise Assessment

**This is NOT a simple factor disguise.** The regime-adaptive nature means it behaves as different factors in different market states:
- Trending regime: momentum/trend factor
- Mean-reverting regime: contrarian/mean-reversion factor
- Random regime: no signal (position = 0)

This is genuinely different from static indicators and could provide real diversification benefit in the ensemble.

### Verdict

**Best indicator in the ensemble.** The regime-adaptive approach is theoretically sound and practically valuable. This should be the highest-weighted indicator in any ensemble.

---

## Indicator 8: Root Mean Square Deviation Trend

**File:** `perpetual/root_mean_square_deviation_trend.py`

### Signal Quality: GOOD

| Dimension | Assessment |
|-----------|------------|
| Economic Hypothesis | RMSD of price from its moving average provides volatility-adaptive bands. Crossover/crossunder of these bands signals trend changes. Similar to Bollinger Bands but using RMSD instead of rolling std. |
| Factor Classification | **Trend** — Volatility-adjusted trend following |
| Expected IC | 0.03–0.05 |
| Novelty | LOW — RMSD bands are a minor variant of Bollinger Bands |

### Look-Ahead Bias Risk: NONE

- Moving average: standard helper functions
- RMSD calculation (line 123): `((src - avg) ** 2).rolling(window=length, min_periods=1).mean()` — rolling, no look-ahead
- Direction detection (lines 133–145): uses current and previous bar values only
- No bfill, shift(-n), or future data

### Stationarity: LOW CONCERN

RMSD bands are computed on (src - avg), which is approximately mean-reverting. The bands expand with volatility. LOW concern.

### Parameter Sensitivity: MEDIUM

| Parameter | Default | Sensitivity | Justification |
|-----------|---------|-------------|---------------|
| `length` | 28 | MEDIUM | MA and RMSD window |
| `mult` | 1.0 | MEDIUM | Band width multiplier |
| `avg_type` | "SMA" | HIGH | Changing MA type changes behavior significantly |

### Code Issues Found

1. **`min_periods=1` in RMSD (line 123):** The RMSD is computed with `min_periods=1`, meaning early bars have noisy estimates. The original commented-out code (line 122) used `min_periods=length`, which is more conservative. This was likely changed for edge-case handling but introduces noise.

2. **No bugs detected.** The implementation is clean.

### Factor Disguise Assessment

**Trend factor with volatility adjustment.** Very similar to Bollinger Band breakout strategies. After decomposition, loads on trend.

### Verdict

Clean implementation. Functionally similar to Bollinger Bands. The main value is in the different MA options available.

---

## Indicator 9: P-Motion Trend QuantEdgeB

**File:** `perpetual/p_motion_trend_quantedgeb.py`

### Signal Quality: GOOD

| Dimension | Assessment |
|-----------|------------|
| Economic Hypothesis | DEMA + rolling median + EMA smoothing + SD bands. The median filter removes outliers before trend estimation. |
| Factor Classification | **Trend** — Multi-smoothed trend following |
| Expected IC | 0.03–0.05 |
| Novelty | LOW — Similar to Gaussian Smooth Trend with different smoothing pipeline |

### Look-Ahead Bias Risk: NONE

- DEMA: standard helper
- Rolling median (line 28): `dema_val.rolling(window=prc_len, min_periods=1).median()` — rolling, no look-ahead
- SD and EMA: standard helpers
- Stateful loop (lines 43–55): bar-by-bar
- No bfill, shift(-n), or future data

### Stationarity: LOW CONCERN

Similar to Gaussian Smooth Trend. Multi-layer smoothing produces non-stationary output, but band crossover logic is approximately stationary.

### Parameter Sensitivity: MEDIUM

| Parameter | Default | Sensitivity | Justification |
|-----------|---------|-------------|---------------|
| `ema_len` | 21 | MEDIUM | EMA smoothing period |
| `sd_length` | 30 | MEDIUM | SD lookback |
| `dema_len` | 7 | MEDIUM | DEMA smoothing |
| `prc_len` | 2 | HIGH | Median filter window — very short |
| `mult_sdup` | 1.5 | MEDIUM | Symmetric multipliers — good design |
| `mult_sddn` | 1.5 | MEDIUM | Symmetric multipliers — good design |

### Code Issues Found

1. **`min_periods=1` in rolling median (line 28):** The median filter with `min_periods=1` can produce noisy estimates in the first few bars. For `prc_len=2`, this means the median of a single value is just that value. Minor issue.

2. **Symmetric multipliers (1.5 vs 1.5):** Unlike Gaussian Smooth Trend (2.5 vs 1.8), this indicator uses symmetric bands. This is more economically neutral.

### Factor Disguise Assessment

**Trend factor — near-duplicate of Gaussian Smooth Trend.** Both indicators use DEMA → smoothing → SD bands. The main difference is the intermediate smoothing (Gaussian filter vs rolling median). **High correlation expected.** Using both in the ensemble provides minimal diversification benefit.

### Verdict

Clean implementation. The rolling median filter is a reasonable outlier-resistant alternative to Gaussian smoothing. The main concern is factor overlap with Gaussian Smooth Trend.

---

## Indicator 10: DEMA Adjusted ATR ✅ BUG FIX VERIFIED

**File:** `perpetual/dema_adjusted_average_true_range.py`

### Signal Quality: GOOD

| Dimension | Assessment |
|-----------|------------|
| Economic Hypothesis | DEMA as trend indicator, ATR as volatility measure. Trailing stop bands (DEMA ± ATR*factor) with stateful direction logic. Classic volatility-adjusted trailing stop. |
| Factor Classification | **Trend** — Trailing stop trend following |
| Expected IC | 0.03–0.06 |
| Novelty | LOW — Trailing stop systems are well-known. The DEMA source is a minor variation. |

### Look-Ahead Bias Risk: NONE

- DEMA: standard helper (no look-ahead)
- ATR: standard helper (no look-ahead)
- Trailing stop logic (lines 44–59): bar-by-bar iterative
- Direction logic (lines 62–69): current bar comparison only
- No bfill, shift(-n), or future data

### Stationarity: LOW CONCERN

The trailing stop is inherently adaptive — it adjusts to volatility through ATR. No stationarity assumption required.

### Parameter Sensitivity: MEDIUM

| Parameter | Default | Sensitivity | Justification |
|-----------|---------|-------------|---------------|
| `period_dema` | 7 | MEDIUM | DEMA smoothing — fast response |
| `period_atr` | 14 | MEDIUM | ATR period — standard |
| `factor_atr` | 1.7 | HIGH | ATR multiplier — directly affects band width |

### ✅ Bug Fix Verification: DEMA ATR "Always Bullish" Bug

**Previous Bug:** The DEMA ATR indicator reportedly always read "bullish."

**Root Cause Analysis:** The bug was likely in the trailing stop logic. A common mistake is initializing the trailing stop incorrectly or using a comparison that always evaluates to True.

**Current Implementation (lines 44–59):**
```python
for i in range(len(df)):
    if i == 0:
        dema_atr_vals[i] = dema_out_vals[i]  # Initialize at DEMA value
    else:
        prev_dema = dema_atr_vals[i-1]
        # ...
        true_range_upper = d_out + t_range
        true_range_lower = d_out - t_range
        
        current_dema = prev_dema
        if true_range_lower > current_dema:
            current_dema = true_range_lower  # Trail up
        if true_range_upper < current_dema:
            current_dema = true_range_upper  # Trail down
        
        dema_atr_vals[i] = current_dema
```

**Direction Logic (lines 62–69):**
```python
elif d_out >= d_atr:
    direction.iloc[i] = 1.0  # Bullish
else:
    direction.iloc[i] = -1.0  # Bearish
```

**Verification:**
1. Trailing stop starts at DEMA[0] — correct initialization
2. Stop can move UP (when `true_range_lower > current_dema`) — correct trailing up behavior
3. Stop can move DOWN (when `true_range_upper < current_dema`) — correct trailing down behavior
4. Direction compares DEMA to trailing stop — correct comparison

**Conclusion:** The bug fix is CORRECT. The trailing stop properly adapts in both directions, and the direction logic correctly identifies when DEMA is above (bullish) or below (bearish) the stop.

**How the old bug likely worked:** The original code probably had:
- `if true_range_lower > current_dema: current_dema = true_range_lower` (trail up) ✅
- Missing: `if true_range_upper < current_dema: current_dema = true_range_upper` (trail down) ❌

Without the "trail down" logic, the stop could only move up, never down. Once DEMA went above the stop, it would stay above forever → always bullish.

### Code Issues Found

1. **Bug fix verified correct.** No remaining issues.

2. **Heikin-Ashi option (lines 22–29):** When `ha_candles=True`, the source price is modified. This changes the DEMA calculation. Not a bug, but users should be aware.

### Factor Disguise Assessment

**Trend factor — trailing stop variant.** After decomposition, loads on trend. The trailing stop approach is more conservative than simple MA crossover, which may reduce drawdowns but also reduce returns.

### Verdict

**Bug fix is correct.** The indicator is mechanically sound. The trailing stop logic properly handles both upward and downward moves.

---

## Cross-Indicator Analysis

### Factor Overlap Matrix

| Indicator | Primary Factor | Trend Correlation |
|-----------|---------------|-------------------|
| Kalman RSI | Momentum | 0.6–0.8 with other momentum |
| Z-SMMA | Trend/Momentum | 0.7–0.9 with trend indicators |
| Median RSI SD | Momentum+Breakout | 0.5–0.7 with trend indicators |
| Polynomial Bands | Trend | 0.7–0.9 with trend indicators |
| Gaussian Smooth | Trend | 0.8–0.95 with P-Motion Trend |
| ALMA Lag | Trend | 0.6–0.8 with trend indicators |
| Adaptive Cloud | Regime-adaptive | 0.3–0.5 with static indicators |
| RMSD Trend | Trend | 0.7–0.9 with trend indicators |
| P-Motion Trend | Trend | 0.8–0.95 with Gaussian Smooth |
| DEMA ATR | Trend | 0.6–0.8 with trend indicators |

### Key Findings

1. **7 of 10 indicators are trend-following variants.** The ensemble is heavily tilted toward a single factor (momentum/trend).

2. **Gaussian Smooth Trend and P-Motion Trend are near-duplicates.** Both use DEMA → smoothing → SD bands. Expected correlation > 0.9. Using both provides minimal diversification.

3. **Only Adaptive Regime Cloud provides genuine diversification.** Its regime-switching behavior means it can trade mean-reversion when other indicators are whipsawing in trending signals.

4. **Kalman RSI is the only pure momentum indicator.** It's a different factor from trend-following (momentum captures medium-term returns, trend captures short-term direction).

5. **Z-SMMA has counter-intuitive direction logic.** It's trend-following despite the "Z-score" naming, which typically implies mean-reversion.

### Recommendations

1. **Fix the polynomial regression evaluation point bug** (Indicator 4, CRITICAL)
2. **Consider removing one of Gaussian Smooth / P-Motion Trend** to reduce redundancy
3. **Increase weight on Adaptive Regime Cloud** in ensemble construction
4. **Add a genuine mean-reversion indicator** to balance the trend-heavy ensemble
5. **Document the Z-SMMA direction logic** to prevent confusion

---

## Audit Summary Table

| # | Indicator | Signal Quality | Look-Ahead Risk | Param Sensitivity | Alpha Source | Issues |
|---|-----------|---------------|-----------------|-------------------|--------------|--------|
| 1 | Kalman RSI | GOOD | NONE | HIGH | Momentum factor | Redundant parallel states |
| 2 | Z-SMMA | SUSPICIOUS | NONE | HIGH | Trend factor | Counter-intuitive direction |
| 3 | Median RSI SD | GOOD | NONE | MEDIUM | Momentum+Breakout | Asymmetric thresholds |
| 4 | Polynomial Bands | GOOD* | NONE | HIGH | Trend factor | **CRITICAL BUG: eval point** |
| 5 | Gaussian Smooth | GOOD | NONE | HIGH | Trend factor | Asymmetric multipliers |
| 6 | ALMA Lag | GOOD | NONE | MEDIUM | Trend factor | None |
| 7 | Adaptive Cloud | EXCELLENT | NONE | MEDIUM | Regime-adaptive | None |
| 8 | RMSD Trend | GOOD | NONE | MEDIUM | Trend factor | min_periods=1 |
| 9 | P-Motion Trend | GOOD | NONE | MEDIUM | Trend factor | Near-duplicate of #5 |
| 10 | DEMA ATR | GOOD | NONE | MEDIUM | Trend factor | Bug fix verified ✅ |

*Conditional on fixing the polynomial evaluation bug

### Critical Actions Required

1. **FIX:** `polynomial_deviation_bands.py` line 127 — change `return coeffs[-1]` to `return np.polyval(coeffs, len(y_vals) - 1)`
2. **REVIEW:** Factor overlap between Gaussian Smooth and P-Motion Trend
3. **DOCUMENT:** Z-SMMA direction logic explanation

### No Look-Ahead Bias Found

All 10 indicators use proper rolling-window or stateful bar-by-bar computation. No bfill, shift(-n), or full-sample statistics detected. The codebase is clean from a look-ahead bias perspective.

---

*End of audit report. Remember: "The backtest is always lying to you." — Every positive result is guilty until proven innocent.*
