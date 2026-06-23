# Indicator Selection Report

**Date:** 2026-06-23
**Source:** AUDIT_REPORT.md + ISP_CHEATING_ANALYSIS.md findings
**Purpose:** Select 3–4 indicators for the MTTD system with genuine factor diversification, no bugs, and complementary signals.

---

## Selection Criteria

1. **Factor diversification** — indicators must cover distinct market factors (regime, momentum, trend, volatility)
2. **No critical bugs** — eliminate indicators with known defects
3. **Good signal quality** — audit rating of GOOD or EXCELLENT, no look-ahead bias
4. **Complementary** — minimal redundancy, each adds unique information

---

## Indicators Eliminated

| Indicator | Reason for Elimination |
|-----------|----------------------|
| **Polynomial Bands** | 🔴 CRITICAL BUG — evaluates polynomial at x=0 (oldest bar) instead of x=period-1 (current bar). Systematic bullish bias. Not fixable without revalidation of all parameters. |
| **Z-SMMA** | 🔴 SUSPICIOUS quality — counter-intuitive direction logic. Signal inverts what a practitioner would expect. High param sensitivity compounds the problem. |
| **Gaussian Smooth** | 🟡 Near-duplicate of P-Motion Trend (correlation >0.9). Redundant. Asymmetric multipliers add complexity without diversification. |
| **P-Motion Trend** | 🟡 Near-duplicate of Gaussian Smooth. Choosing one over the other adds no new factor exposure. |
| **Median RSI SD** | 🟡 Asymmetric thresholds create fragile parameterization. Momentum overlap with Kalman RSI but with worse quality signals. |
| **DEMA ATR** | 🟡 Bug was fixed but indicator is trend-following. Adds no new factor beyond what ALMA Lag and RMSD Trend already provide. Higher param sensitivity for equivalent information. |

---

## Recommended Indicators: The Core Four

### 1. Adaptive Regime Cloud — **REGIME FACTOR** ⭐ Core

| Property | Detail |
|----------|--------|
| **Audit Quality** | EXCELLENT (highest rated of all 10) |
| **Factor** | Regime detection — the ONLY non-trend indicator in the original set |
| **What It Captures** | Whether the market is in a trending, ranging, or transitional regime. Dynamically adapts signal interpretation based on current conditions. |
| **Why Essential** | Addresses Finding M2 (No Regime Awareness). Without this, the system treats bull markets, bear markets, and choppy ranges identically — which is the #1 reason momentum strategies fail in real trading. This single indicator provides more genuine diversification than the other 5 trend indicators combined. |
| **Param Sensitivity** | MEDIUM (robust) |
| **Look-Ahead** | NONE |
| **Bugs** | NONE |

**Role in ensemble:** Primary regime gate. Determines the context in which other signals are interpreted. Should receive the highest weight.

---

### 2. Kalman RSI — **MOMENTUM FACTOR**

| Property | Detail |
|----------|--------|
| **Audit Quality** | GOOD |
| **Factor** | Momentum — RSI filtered through a Kalman state estimator |
| **What It Captures** | Overbought/oversold momentum conditions with noise reduction. The Kalman filter smooths RSI noise while preserving genuine momentum shifts. |
| **Why Essential** | Only genuine momentum indicator after Median RSI SD elimination. Provides a completely different signal dimension from trend indicators — momentum can confirm or diverge from trend, which is where alpha lives. |
| **Param Sensitivity** | HIGH (requires careful tuning, but well-understood RSI mechanics) |
| **Look-Ahead** | NONE |
| **Bugs** | NONE (minor: redundant parallel states noted in audit — cosmetic, not functional) |

**Role in ensemble:** Momentum confirmation/divergence detector. Identifies overextended moves that trend indicators miss.

---

### 3. ALMA Lag — **TREND FACTOR (Primary)**

| Property | Detail |
|----------|--------|
| **Audit Quality** | GOOD |
| **Factor** | Trend — Arnaud Legoux Moving Average lag measurement |
| **What It Captures** | The lag (displacement) between the ALMA smoothing and raw price. Low lag = strong directional conviction; high lag = uncertainty or reversal. |
| **Why Essential** | Cleanest trend implementation in the set. No bugs, no asymmetric quirks, medium param sensitivity. Measures trend conviction rather than just direction — a subtle but important distinction. |
| **Param Sensitivity** | MEDIUM (robust) |
| **Look-Ahead** | NONE |
| **Bugs** | NONE |

**Role in ensemble:** Primary trend signal. Provides the directional (long/short/flat) backbone of the system.

---

### 4. RMSD Trend — **TREND + VOLATILITY FACTOR**

| Property | Detail |
|----------|--------|
| **Audit Quality** | GOOD |
| **Factor** | Trend quality via volatility adjustment — root-mean-square deviation of price from its smoothed trend line |
| **What It Captures** | How "clean" a trend is. Low RMSD = smooth, reliable trend. High RMSD = noisy, unreliable trend. This is fundamentally different from ALMA Lag, which measures direction, not quality. |
| **Why Essential** | While both ALMA Lag and RMSD Trend are trend-family indicators, they measure orthogonal properties: ALMA Lag measures *direction and conviction*, RMSD Trend measures *trend smoothness and reliability*. In choppy markets, ALMA Lag may still signal direction but RMSD Trend will flag that the signal is noisy and unreliable. This is genuine within-factor diversification. |
| **Param Sensitivity** | MEDIUM (robust) |
| **Look-Ahead** | NONE |
| **Bugs** | NONE (min_periods=1 is noted but acceptable for the calculation) |

**Role in ensemble:** Trend quality filter. Down-weights or blocks trend signals when the trend is too noisy to trade reliably.

---

## Factor Diversification Matrix

| Indicator | Regime | Momentum | Trend Direction | Trend Quality | Volatility |
|-----------|--------|----------|----------------|---------------|------------|
| Adaptive Regime Cloud | ✅ | — | — | — | ✅ |
| Kalman RSI | — | ✅ | — | — | — |
| ALMA Lag | — | — | ✅ | — | — |
| RMSD Trend | — | — | — | ✅ | ✅ |

**Coverage:** 4 distinct factors across 4 indicators. No two indicators measure the same thing.

---

## How the Four Indicators Work Together

```
Market Data
    │
    ▼
┌─────────────────────────┐
│  Adaptive Regime Cloud   │  ← "What kind of market is this?"
│  (Regime Gate)           │
└────────┬────────────────┘
         │ Regime context
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────────────┐
│ Kalman │ │  ALMA Lag      │
│ RSI    │ │  (Trend Dir)   │
│(Momen) │ │                │
└───┬────┘ └───────┬────────┘
    │               │
    ▼               ▼
┌─────────────────────────┐
│     RMSD Trend          │  ← "Is this trend clean enough to trade?"
│  (Trend Quality Gate)   │
└────────┬────────────────┘
         │
         ▼
    ENSEMBLE SIGNAL
```

**Signal flow:**
1. **Adaptive Cloud** sets the regime context (trending/ranging/transitioning)
2. **Kalman RSI** reads momentum within that context
3. **ALMA Lag** reads trend direction and conviction
4. **RMSD Trend** acts as a quality gate — blocks noisy signals even if direction and momentum agree

**Example scenario — choppy ranging market:**
- Adaptive Cloud: "Ranging regime — reduce trend signal weight"
- Kalman RSI: "Slightly oversold"
- ALMA Lag: "Weak upward signal"
- RMSD Trend: "High noise — trend signal unreliable"
- **Result:** System stays flat (correct behavior — most trend systems get destroyed in ranges)

---

## Comparison to Eliminated Indicators

| Metric | Selected 4 | Original 10 |
|--------|-----------|-------------|
| Unique factors | 4 | 2 (trend + momentum) |
| Critical bugs | 0 | 1 (Polynomial Bands) |
| Suspicious indicators | 0 | 1 (Z-SMMA) |
| Near-duplicate pairs | 0 | 1 (Gaussian/P-Motion) |
| Avg param sensitivity | MEDIUM | HIGH |
| Look-ahead bias | NONE | NONE |
| Total parameters to tune | ~8–12 | ~25–30 |

---

## Risk Factors for the Selected Set

| Risk | Severity | Mitigation |
|------|----------|------------|
| ALMA Lag and RMSD Trend are both trend-family | LOW | They measure orthogonal properties (direction vs quality). Monitor correlation in backtest — if >0.7, consider replacing RMSD with a volatility-only indicator. |
| Kalman RSI has high param sensitivity | MEDIUM | Use wider param ranges in grid search. Consider fixing RSI period to standard 14 and only tuning Kalman process noise. |
| Adaptive Cloud is the sole regime indicator | LOW | Acceptable — it genuinely is the only regime indicator in the universe. If regime detection proves insufficient, add a simple regime classifier (e.g., 50/200 SMA + ATR percentile) as a second opinion. |
| 4 indicators may still overfit to historical data | MEDIUM | Addressed by audit recommendation: implement proper OOS holdout (2025-2026), deflated Sharpe ratio, and transaction costs. |

---

## Next Steps

1. **Rebuild ensemble** with only these 4 indicators
2. **Run grid search** optimized for risk-adjusted returns (NOT ISP coherence)
3. **Add transaction costs** (minimum 0.1% round-trip)
4. **Implement proper OOS holdout** — train on 2018–2024, test on 2025–2026
5. **Monitor ALMA Lag vs RMSD Trend correlation** — replace if redundant
6. **Paper trade** for 12+ months before any live deployment

---

> *"Simplicity is the ultimate sophistication."* — Leonardo da Vinci
>
> Four indicators, four factors, zero bugs. This is a foundation that can be validated, not a haystack of 10 indicators where alpha hides in the noise.
