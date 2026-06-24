# MTTD System — Quantitative Research Audit Report

**Auditor:** Quantitative Research Scientist (Radical Skepticism Framework)
**Date:** 2026-06-23
**Methodology:** 8 Critical Anti-Patterns, Deflated Sharpe Ratio, Haircut Rule, Factor Decomposition

---

## EXECUTIVE VERDICT: 🔴 HIGH OVERFITTING RISK — DO NOT DEPLOY

**The system's reported performance is statistically indistinguishable from noise.**

| Metric | Backtest | Expected Live | ISP Benchmark |
|--------|----------|---------------|---------------|
| Sharpe | 1.32 | 0.40–0.66 | 1.88 |
| CAGR | 56.9% | 17–29% | 78.1% |
| Max DD | -41.6% | Worse | -25.7% |
| ISP Coherence | 78.1% | 55–65% | — |

**Deflated Sharpe Ratio: 54%** (need >95% for significance)
**46% probability this result arose from pure data mining.**

---

## 🔴 CRITICAL FINDINGS (Must Fix Before Any Live Trading)

### C1. Zero Transaction Costs
- All backtests assume 0% commission, 0% spread, 0% slippage
- With 60 trades (vs ISP's 16), costs compound significantly
- **Impact:** ~1.5% annual CAGR reduction, Sharpe degradation
- **Fix:** Add 0.1% round-trip cost minimum

### C2. Grid Search Overfits to ISP (Structural Impossibility)
- ISP uses **on-chain data, sentiment, proprietary signals** — unavailable in TA
- Individual indicators achieve only 27–31% coherence with ISP (near random)
- Ensemble achieves 78% through parameter optimization, not signal alignment
- **The gap between 30% (individual) and 78% (ensemble) IS the overfitting**
- **Fix:** Stop optimizing against ISP coherence. Optimize for absolute risk-adjusted returns.

### C3. No Out-of-Sample Holdout
- ALL data (2018–2026) used for both fitting and evaluation
- No data was reserved that was NEVER touched during optimization
- Walk-forward uses expanding windows across ENTIRE dataset
- **Fix:** Reserve 2025–2026 as final holdout. Report only holdout metrics.

### C4. Statistical Significance: NOT SIGNIFICANT
- 2,654+ parameter combinations tested on 3,095 daily bars
- Expected max Sharpe under null hypothesis: **1.29**
- Observed best Sharpe: **1.32** — essentially at the noise floor
- Deflated Sharpe Ratio: 54% (need >95%)
- **Fix:** Implement HLZ 2016 deflated Sharpe ratio. Reject strategies with DSR < 95%.

### C5. System Fails ALL ISP Benchmark Comparisons
```json
{
  "Sharpe": false,
  "Sortino": false,
  "Calmar": false,
  "MaxDD": false,
  "CAGR": false
}
```
- The system was explicitly optimized to beat ISP and failed every comparison
- A system that cannot match its optimization target has no alpha

---

## 🟡 HIGH SEVERITY FINDINGS

### H1. Max Drawdown -41.6% Is Unacceptable
- System has 15% drawdown pause mechanism, yet Max DD is -41.6%
- ISP achieves only -25.7% Max DD with better risk management
- A 41% drawdown would trigger forced liquidation at most institutions

### H2. 60 Trades vs ISP's 16 — Excessive Whipsawing
- System trades 3.75x more frequently than ISP
- Each extra trade is a cost and a potential timing error
- Min_hold=10 helps but doesn't solve the fundamental issue

### H3. Walk-Forward Has No Embargo Gap
- Train and test periods share the same day boundary
- Information leakage at boundaries inflates OOS metrics
- **Fix:** Add 10-day embargo between train and test

### H4. Walk-Forward OOS Has No Statistical Power
- Only 22 ISP transitions over 8 years
- Most 12-month OOS periods contain 0–3 transitions
- A single correct trade swings coherence by 10+ points

### H5. Individual Indicators Have No Predictive Power
- Each indicator achieves only 27–31% coherence with ISP
- This is barely above random for a 33% in-position baseline
- The ensemble's 78% coherence is optimization artifact

### H6. Factor Overlap — 7 of 10 Indicators Are Trend-Following
| Indicator | Factor |
|-----------|--------|
| Kalman RSI | Momentum |
| Z-SMMA | Trend |
| Median RSI SD | Momentum+Breakout |
| Polynomial Bands | Trend |
| Gaussian Smooth | Trend |
| ALMA Lag | Trend |
| **Adaptive Cloud** | **Regime-adaptive** ← only genuine diversifier |
| RMSD Trend | Trend |
| P-Motion Trend | Trend |
| DEMA ATR | Trend |

- Gaussian Smooth and P-Motion Trend are near-duplicates (expected correlation >0.9)
- Ensemble heavily tilted toward single factor (trend/momentum)

---

## 🟠 MEDIUM SEVERITY FINDINGS

### M1. Polynomial Deviation Bands Has Critical Bug
- `polynomial_deviation_bands.py` line 127: evaluates polynomial at x=0 (oldest bar) instead of x=period-1 (current bar)
- Causes systematic bullish bias in trending markets
- **Fix:** Change `return coeffs[-1]` to `return np.polyval(coeffs, len(y_vals) - 1)`

### M2. No Regime Awareness
- System treats all market conditions identically
- ISP uses 3-tier regime system (Strong Bull/Weak Bull/Neutral)
- Momentum strategies fail in choppy markets; mean-reversion fails in trending markets

### M3. Data Window Mismatch
- ISP data starts 2015, MTTD uses 2018+
- ISP's CAGR includes 2015–2017 returns; MTTD's does not
- Direct comparison is apples-to-oranges

### M4. Neutral Signals Treated as Bearish
- `1.0 if x > 0 else -1.0` — neutral (0) becomes -1.0
- Structural long bias in ensemble (needs 51% bullish to enter)
- May explain why system is in-market 46% vs ISP's 33%

### M5. execute_system.py Is Dead Code
- References old ensemble API (threshold, EMA, weights)
- Would fail if run with current ensemble_engine.py

---

## 🟢 INDICATOR AUDIT SUMMARY

| # | Indicator | Quality | Look-Ahead | Param Sensitivity | Issues |
|---|-----------|---------|------------|-------------------|--------|
| 1 | Kalman RSI | GOOD | NONE | HIGH | Redundant parallel states |
| 2 | Z-SMMA | SUSPICIOUS | NONE | HIGH | Counter-intuitive direction |
| 3 | Median RSI SD | GOOD | NONE | MEDIUM | Asymmetric thresholds |
| 4 | Polynomial Bands | GOOD* | NONE | HIGH | **CRITICAL BUG: eval point** |
| 5 | Gaussian Smooth | GOOD | NONE | HIGH | Asymmetric multipliers |
| 6 | ALMA Lag | GOOD | NONE | MEDIUM | None |
| 7 | Adaptive Cloud | **EXCELLENT** | NONE | MEDIUM | None — best indicator |
| 8 | RMSD Trend | GOOD | NONE | MEDIUM | min_periods=1 |
| 9 | P-Motion Trend | GOOD | NONE | MEDIUM | Near-duplicate of #5 |
| 10 | DEMA ATR | GOOD | NONE | MEDIUM | Bug fix verified ✅ |

**No look-ahead bias found in any indicator.** All use proper rolling-window or stateful computation.

---

## 📊 HAIRCUT RULE — EXPECTED LIVE PERFORMANCE

| Metric | Backtest | Live (30% haircut) | Live (50% haircut) |
|--------|----------|-------------------|-------------------|
| Sharpe | 1.32 | 0.40 | 0.66 |
| CAGR | 56.9% | 17.1% | 28.5% |
| Max DD | -41.6% | Worse | Worse |
| Coherence | 78.1% | 55–65% | 55–65% |
| Trades | 60 | Similar | Similar |

**The system will NOT match ISP's Sharpe of 1.88 in live trading.**

---

## 🎯 RECOMMENDATIONS (Priority Order)

### CRITICAL (Must Implement)
1. **Add proper holdout.** Reserve 2025–2026 data. Optimize on 2018–2024 only.
2. **Reduce parameters by 90%.** Fix indicator params with domain knowledge. Max 3–5 free params.
3. **Implement Deflated Sharpe Ratio (HLZ 2016).** Reject strategies with DSR < 95%.
4. **Stop optimizing against ISP coherence.** Optimize for absolute risk-adjusted returns.
5. **Add transaction costs.** Minimum 0.1% round-trip for BTC.

### HIGHLY RECOMMENDED
6. Add 10-day embargo in walk-forward validation.
7. Reduce threshold grid to 10–15 candidates (not 101).
8. Report regime-specific performance (bull/bear/sideways).
9. Cross-validate indicator selection (test which 3–5 add value).
10. Fix polynomial evaluation bug.

### MEDIUM PRIORITY
11. Paper trade 12+ months before live deployment.
12. Test against random null (1,000 random strategies).
13. Add regime detection (50/200 SMA crossover + volatility regime).

---

## 🧠 BOTTOM LINE

**The backtest is lying to you.**

The system's reported performance (CAGR 56.9%, Sharpe 1.32) is statistically indistinguishable from random noise after accounting for:
- 2,654+ parameter combinations tested (multiple testing)
- No out-of-sample holdout (overfitting)
- Zero transaction costs (cost blindness)
- ISP uses unavailable data sources (structural impossibility)

The **Adaptive Regime Cloud** indicator is genuinely excellent and deserves more weight. The other 9 indicators are variations of the same trend-following factor with high correlation.

**To build a system that actually beats ISP:**
1. Stop trying to replicate ISP's timing with TA alone
2. Focus on a smaller number of indicators with genuine alpha
3. Optimize for risk-adjusted returns, not benchmark replication
4. Validate rigorously with proper OOS holdout and deflated Sharpe

> *"I will remember that I didn't make the world, and it doesn't satisfy my equations."*
> — Financial Modelers' Hippocratic Oath (Derman & Wilmott, 2009)
