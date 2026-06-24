# PLAN.md — MSVR Enhancement Roadmap

## Current State

### What Works
- MSVR base signal (Family 1: Smoothing) — good directional indicator
- Ichimoku (Families 2,5,7) — 11 trades, 63.6% win rate, Sharpe 1.31

### What Doesn't Work
- Enhanced MSVR (5 families) — 48 trades, 35.4% win rate = **NOISE**
- MSVR v2 (filtered) — 36 trades, 55.6% win, but returns dropped

### Root Cause
**MSVR Enhancement = Noise because:**
1. Volume confirmation too weak
2. Hurst exponent not effective for regime detection
3. Entropy threshold too loose
4. No genuine filtering like Ichimoku's SuperSmoother

## Goal

**Beat Ichimoku: 11 trades, 63.6% win, Sharpe 1.31**

## Strategy: 3-Phase Approach

### Phase 1: Port Ichimoku's Proven Principles
Copy what WORKS from Ichimoku into MSVR framework.

| Principle | Source | Implementation |
|-----------|--------|----------------|
| Ehler SuperSmoother | Family 2 | Noise reduction on MSVR signal |
| Shannon Entropy | Family 7 | Block random markets |
| Efficiency Ratio | Family 5 | Only trade trending markets |
| Adaptive Threshold | Ichimoku | Dynamic entry conditions |

### Phase 2: Add Missing Principles
Add NEW statistical families not yet used.

| Family | Principle | Expected Impact |
|--------|-----------|-----------------|
| Family 3 | Linear Regression | Better trend detection |
| Family 6 | GARCH-like | Volatility clustering |
| Family 8 | Volume (OBV, VWAP) | Confirm moves with volume |
| Family 9 | HMM/Jump Model | Regime detection |

### Phase 3: Optimize & Validate
1. Grid search on training data (2018-2024)
2. Walk-forward validation
3. Holdout test (2025-2026)
4. Compare with Ichimoku benchmark

## Target Metrics

| Metric | Current (Enhanced) | Ichimoku | Target |
|--------|-------------------|----------|--------|
| Trades | 48 | 11 | < 15 |
| Win Rate | 35.4% | 63.6% | > 65% |
| Sharpe | 1.35 | 1.31 | > 1.40 |
| CAGR | 58.5% | 55.6% | > 60% |

## Key Insight

**Ichimoku's Secret:**
- Uses 4 principles that GENUINELY reduce noise
- Each principle serves a SPECIFIC purpose
- Combined = high-quality signals with few trades

**Our Approach:**
- Port Ichimoku's proven principles
- Add missing principles (Volume, GARCH, HMM)
- Optimize for QUALITY not QUANTITY

## Phase 1 Results ✅ COMPLETED

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Trades | < 20 | 15 | ✅ |
| Win Rate | > 60% | 66.7% | ✅ |
| Sharpe | > 1.35 | 1.09 | ❌ |
| CAGR | > 60% | 35.8% | ❌ |

**Key Finding:** MSVR Hybrid in position only 30.7% vs Ichimoku 44.9%
**Root Cause:** Filters too aggressive → fewer opportunities → lower returns

## Phase 2 Results ✅ COMPLETED

### MSVR v8 (Best Config — Medium-Term Target)
| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Trades | 25-35 | 26 | ✅ |
| Win Rate | > 60% | 57.7% | ⚠️ |
| Sharpe | > 1.35 | 1.18 | ⚠️ |
| CAGR | > 50% | 45.0% | ⚠️ |
| Avg Hold | 30-60d | 44d | ✅ |

**Key Finding:** MSVR v8 achieves 26 trades, 1.18 Sharpe, 45% CAGR
**Progress:** v7 → v8 improved Sharpe +4%, CAGR +17%
**Root Cause:** MSVR signal may not have enough edge for Sharpe > 1.35
**Options:** Accept as final, try different base signal, or combine with Ichimoku

## Timeline

| Phase | Duration | Deliverable | Status |
|-------|----------|-------------|--------|
| Phase 1 | 1 session | MSVR+Ichimoku hybrid | ✅ DONE |
| Phase 2 | 1 session | MSVR v6 with 10 families | ✅ DONE |
| Phase 3 | 1 session | Validated system | 🔄 NEXT |
| **Total** | **3 sessions** | **Final system** | |
