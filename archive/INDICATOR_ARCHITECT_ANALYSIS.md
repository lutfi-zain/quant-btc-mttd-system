# Technical Indicator Architect Analysis

**Date:** 2026-06-23
**Skill:** lz-technical-indicator-architect (10 Statistical Families Framework)
**Source:** quant-technical-indicator-bank repository

---

## Current Problem

**4 indicators in MTTD system are ALL trend-following:**

| Indicator | Statistical Family | Primary Factor |
|-----------|-------------------|----------------|
| Adaptive Regime Cloud | Fractal (Hurst) + Smoothing | Trend |
| Kalman RSI | Bayesian (Kalman) + Smoothing | Momentum (but correlated with trend) |
| ALMA Lag | Smoothing (ALMA) | Trend |
| RMSD Trend | Smoothing + Volatility | Trend |

**Correlation matrix shows HIGH correlation (>0.65):**
- Adaptive Cloud ↔ RMSD: 0.87
- Kalman RSI ↔ ALMA: 0.82
- All pairs >0.65

**Result:** Majority vote provides NO genuine diversification. System fails in holdout.

---

## Solution: Add Indicators from DIFFERENT Statistical Families

Based on the 10 Families Framework, we need indicators from families NOT currently represented:

| Family | Current Coverage | Gap |
|--------|-----------------|-----|
| 1. Smoothing | ✅ ALMA, RMSD, Adaptive Cloud | Over-represented |
| 2. Filtering | ❌ None | **NEED** |
| 3. Regression | ❌ None (Polynomial has bug) | **NEED** |
| 4. Spectral | ❌ None | **NEED** |
| 5. Fractal | ✅ Adaptive Cloud (Hurst) | Covered |
| 6. GARCH | ❌ None | **NEED** |
| 7. Entropy | ❌ None | **NEED** |
| 8. Chaos | ❌ None | Low priority |
| 9. Bayesian | ✅ Kalman RSI | Covered |
| 10. ML Hybrid | ❌ None | Low priority |

---

## Recommended New Indicators

### 1. Fourier For Loop — FAMILY 4 (Spectral)

**File:** `oscillator/fourier_for_loop.py`

**What it does:**
- Decomposes price into frequency components using Discrete Fourier Transform (DFT)
- Identifies dominant cycle periods
- Uses cycle phase for signal generation

**Why it's different:**
- Measures CYCLE, not TREND
- Detects when market is in cycle mode vs trend mode
- Fundamentally different from smoothing-based indicators

**Expected correlation with current indicators:** LOW (<0.4)

**Parameters to tune:**
- `n`: Number of DFT components (1-5)
- `start`: Start frequency (default 1)
- `end`: End frequency (default 45)

**Statistical foundation:** Spectral analysis decomposes signal into sinusoidal components. Each component has amplitude, phase, and frequency. Trading signal derived from phase alignment of dominant cycles.

---

### 2. Volatility Adaptive Oscillator Suite — FAMILY 6 (GARCH-like)

**File:** `oscillator/volatility_adaptive_oscillator_suite.py`

**What it does:**
- Combines RSI, CCI, CMO with volatility adaptation
- Adapts oscillator thresholds based on current volatility regime
- Multiple oscillators confirm each other

**Why it's different:**
- VOLATILITY-ADAPTIVE (not just trend-following)
- Uses CCI and CMO (different mathematical foundations than RSI)
- Adapts to market conditions

**Expected correlation with current indicators:** MEDIUM (0.4-0.6)

**Parameters to tune:**
- `rsi_len`: RSI period (default 35)
- `cci_len`: CCI period (default 35)
- `cmo_len`: CMO period (default 30)
- Volatility adaptation parameters

**Statistical foundation:** GARCH models time-varying conditional variance. This indicator adapts oscillator sensitivity based on recent volatility, similar to how GARCH adapts variance estimates.

---

### 3. Z-Score Adaptive Oscillator Suite — FAMILY 7 (Entropy/Statistics)

**File:** `oscillator/z_score_adaptive_oscillator_suite.py`

**What it does:**
- Computes Z-scores of RSI, CCI, CMO
- Identifies statistical outliers (Z > 1.5)
- Mean-reversion signals when oscillators are statistically extreme

**Why it's different:**
- MEAN-REVERSION (opposite of trend-following)
- Uses statistical extreme detection
- Fundamental contrarian approach

**Expected correlation with current indicators:** LOW-NEGATIVE (<0.3, possibly negative)

**Parameters to tune:**
- `z_score_len`: Lookback for Z-score calculation (default 14)
- `threshold`: Z-score threshold for extreme (default 1.5)

**Statistical foundation:** Z-score measures how many standard deviations a value is from the mean. High Z-score = statistical outlier = potential mean-reversion opportunity. This is the opposite of trend-following.

---

### 4. IRS Elder Force Volume Index — Volume Factor

**File:** `perpetual/irs_elder_force_volume_index.py`

**What it does:**
- Combines price change with volume (Force Index)
- Uses EMA of Force Index for trend
- HMA for direction confirmation

**Why it's different:**
- VOLUME-BASED (price × volume)
- Captures conviction behind price moves
- Different data source (volume, not just price)

**Expected correlation with current indicators:** MEDIUM (0.3-0.5)

**Parameters to tune:**
- `length`: EMA period for Force Index (default 40)

**Statistical foundation:** Volume confirms price. High volume + price up = strong bullish conviction. Low volume + price up = weak move. Volume-price divergence is a leading indicator.

---

### 5. Hull Supertrend RSI — Combined Family

**File:** `oscillator/hull_supertrend_rsi.py`

**What it does:**
- Hull MA for smooth trend
- Supertrend for volatility-adjusted direction
- RSI for momentum confirmation

**Why it's different:**
- COMPOSITE indicator (3 principles combined)
- Supertrend uses ATR for volatility adjustment
- Multi-factor confirmation

**Expected correlation with current indicators:** MEDIUM-HIGH (0.5-0.7)

**Parameters to tune:**
- Hull MA period
- Supertrend ATR period and multiplier
- RSI period

**Statistical foundation:** Combines smoothing (Hull), volatility (ATR/Supertrend), and momentum (RSI) in a single indicator. Multi-factor models often outperform single-factor.

---

## Recommended Implementation Plan

### Phase 1: Add 2 Genuinely Different Indicators

**Priority 1:** Fourier For Loop (Spectral - cycle detection)
**Priority 2:** Z-Score Adaptive Oscillator Suite (Mean-reversion)

These two provide the MOST different factors:
- Fourier: Cycle detection (when to trade)
- Z-Score: Mean-reversion (contrarian signals)

### Phase 2: Add 1 More Indicator

**Priority 3:** IRS Elder Force Volume Index (Volume confirmation)

Volume is a genuinely different data source.

### Phase 3: Test Ensemble

After adding these indicators:
1. Compute correlation matrix
2. Verify correlations <0.5 between new and existing indicators
3. Run grid search V3 with expanded indicator set
4. Validate on holdout (2025-2026)

---

## Expected Outcome

**Before (4 trend indicators):**
- Correlation: 0.65-0.87 (HIGH)
- Diversification: NONE
- Holdout: -14.4% CAGR

**After (6-7 mixed indicators):**
- Correlation: 0.3-0.5 (MEDIUM)
- Diversification: GENUINE
- Expected Holdout: Improved (less overfitting)

---

## Implementation Code Skeleton

```python
# New indicators to add to grid_search_v3.py

INDICATORS = [
    # EXISTING (Trend family)
    {
        'name': 'adaptive_regime_cloud',
        'category': 'perpetual',
        'params': {'hurst_period': [30, 40, 50, 60, 70]},
        'family': 'Fractal'
    },
    {
        'name': 'alma_lag_viresearch',
        'category': 'perpetual',
        'params': {'alma_length': [60, 70, 78, 85, 100]},
        'family': 'Smoothing'
    },
    
    # NEW (Different families)
    {
        'name': 'fourier_for_loop',  # SPECTRAL - Cycle detection
        'category': 'oscillator',
        'params': {'n': [1, 2, 3], 'start': [1, 5], 'end': [30, 45]},
        'family': 'Spectral'
    },
    {
        'name': 'z_score_adaptive_oscillator_suite',  # MEAN-REVERSION
        'category': 'oscillator',
        'params': {'z_score_len': [10, 14, 20], 'threshold': [1.0, 1.5, 2.0]},
        'family': 'Entropy/Statistics'
    },
    {
        'name': 'irs_elder_force_volume_index',  # VOLUME
        'category': 'perpetual',
        'params': {'length': [30, 40, 50]},
        'family': 'Volume'
    },
]
```

---

## Key Insight from Technical Indicator Architect

> *"Never rely on single-family indicator. Combine 2+ principles."*

Current MTTD system violates this rule — all 4 indicators are from Smoothing/Fractal families.

**The fix:** Add indicators from Spectral (Fourier), Entropy (Z-Score), and Volume families.

---

## Next Steps

1. Copy new indicators from quant-technical-indicator-bank to quant-btc-mttd-system
2. Modify grid_search_v3.py to include new indicators
3. Run grid search with expanded set
4. Verify correlation matrix shows genuine diversification
5. Validate on holdout (2025-2026)

---

> *"Diversification is the only free lunch in investing."* — Harry Markowitz
>
> But diversification must be across DIFFERENT statistical families, not just different indicator names.
