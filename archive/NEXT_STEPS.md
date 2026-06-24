# Next Steps — Actionable Plan

## Problem Summary

1. **Structural**: ISP uses on-chain + sentiment data unavailable to TA
2. **Overfitting**: Parameters tuned to match ISP, not to predict prices
3. **Validation**: Holdout shows -79.5% Sharpe degradation

---

## Option A: Fix the System (Recommended)

### Step 1: Change Optimization Target
**BEFORE:** Optimize for ISP coherence
**AFTER:** Optimize for risk-adjusted returns (Sharpe, Calmar)

```python
# OLD (BAD):
score = coherence * 0.4 + sharpe_ratio * 20 + ...

# NEW (GOOD):
score = sharpe_ratio * 0.5 + calmar_ratio * 0.3 + sortino_ratio * 0.2
# ISP only for post-hoc comparison, NOT optimization target
```

### Step 2: Reduce Indicator Set
**BEFORE:** 10 indicators (7 are trend-following)
**AFTER:** 3-4 indicators with genuine diversification

**Keep:**
1. Adaptive Regime Cloud (genuinely excellent)
2. Kalman RSI (momentum — different factor)
3. One trend indicator (ALMA or RMSD)

**Remove:**
- Gaussian Smooth + P-Motion Trend (near-duplicates)
- Z-SMMA (counter-intuitive direction)
- Others (marginal value)

### Step 3: Add Transaction Costs
```python
# Add 0.1% round-trip cost per trade
transaction_cost = 0.001  # 0.1%
strategy_returns = returns * positions.shift(1) - transaction_cost * positions.diff().abs()
```

### Step 4: Proper Validation
```
1. Reserve 2025-2026 as HOLDOUT (already done)
2. Grid search on 2018-2024 only (already done)
3. Walk-forward on 2018-2024 with embargo gap
4. Final test on holdout — report ONLY holdout metrics
```

### Step 5: Realistic Expectations
- **Target:** Sharpe 0.8-1.2 (not 1.88 like ISP)
- **Accept:** ISP has structural advantage (better data)
- **Focus:** Generate independent alpha, not replicate ISP

---

## Option B: Start Fresh

### New Strategy: Adaptive Cloud Core

**Single indicator, properly optimized:**

1. Use Adaptive Regime Cloud as sole signal
2. Add regime filter (200-day SMA for bull/bear)
3. Optimize for Sharpe on 2018-2024
4. Validate on 2025-2026 holdout

**Advantages:**
- Simpler = less overfitting
- Adaptive Cloud is genuinely good
- No factor overlap issues
- Easier to understand and maintain

---

## Option C: Accept Limitations

### Reality Check

**Maybe TA alone cannot beat ISP.**

ISP has:
- On-chain data (exchange flows, whale movements)
- Sentiment data (fear & greed, social media)
- Proprietary signals
- Professional team

TA has:
- Price and volume only
- Publicly available
- No edge over ISP

**Honest assessment:**
- Technical analysis has limited alpha in BTC
- The market is increasingly efficient
- ISP's edge comes from data, not timing

**Recommendation:**
- Use MTTD as a TOOL, not a standalone system
- Combine with other analysis (on-chain, sentiment if available)
- Accept that perfect timing is impossible

---

## My Recommendation: Option A

### Why?

1. **Adaptive Cloud is genuinely good** — regime detection works
2. **Factor diversification is fixable** — reduce to 3-4 indicators
3. **Optimization target is fixable** — stop optimizing for ISP
4. **Validation is fixable** — proper holdout already implemented

### What Will Change?

| Aspect | Before | After |
|--------|--------|-------|
| Optimization target | ISP coherence | Risk-adjusted returns |
| Indicator count | 10 (7 trend) | 3-4 (diverse) |
| Transaction costs | 0% | 0.1% round-trip |
| Validation | In-sample only | Proper holdout |
| Expected Sharpe | 1.32 (fake) | 0.8-1.2 (realistic) |

### What Will Stay the Same?

- Ensemble majority vote (simple, robust)
- Binary position (100% BTC / 0% cash)
- Daily rebalancing
- Risk management (drawdown pause)

---

## Implementation Plan

### Phase 1: Fix Optimization (1-2 hours)
1. Modify `grid_search_v2_ho.py` to optimize for Sharpe/Calmar
2. Remove ISP coherence from fitness function
3. Run on 2018-2024 training data

### Phase 2: Reduce Indicators (30 min)
1. Select 3-4 best indicators
2. Remove redundant trend indicators
3. Re-run grid search with smaller set

### Phase 3: Add Costs (15 min)
1. Add 0.1% transaction cost to all calculations
2. Re-run grid search

### Phase 4: Validate (already done)
1. Holdout evaluation on 2025-2026
2. Report ONLY holdout metrics
3. Compare with ISP (post-hoc only)

### Phase 5: Report (30 min)
1. Generate new equity curve
2. Compare training vs holdout
3. Send to Telegram

---

## Success Criteria

**The system is ready for paper trading if:**

1. ✅ Holdout Sharpe > 0.5 (not 1.32 like training)
2. ✅ Holdout Max DD < -30%
3. ✅ Training→Holdout degradation < 50%
4. ✅ No ISP coherence in optimization
5. ✅ Transaction costs included
6. ✅ 3-4 diverse indicators (not 10)

**The system is ready for live trading if:**

1. ✅ Paper trading for 6+ months
2. ✅ Live Sharpe > 0.5
3. ✅ No blow-ups or unexpected behavior
4. ✅ Manual oversight for regime changes

---

## Timeline

| Task | Time | Priority |
|------|------|----------|
| Fix optimization target | 1-2h | CRITICAL |
| Reduce indicators | 30m | HIGH |
| Add transaction costs | 15m | HIGH |
| Re-run grid search | 10min | HIGH |
| Holdout evaluation | Done | DONE |
| Paper trading | 6+ months | MEDIUM |
| Live trading | After paper | LOW |

---

> *"The goal is not to be right, but to make money."* — Ed Seykota
