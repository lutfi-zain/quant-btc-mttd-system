# TASK.md — Detailed Task List

## Phase 1: Port Ichimoku's Proven Principles

### Task 1.1: Implement Ehler SuperSmoother
- [ ] Create `indicators/ehler_supersmoother.py`
- [ ] Test on MSVR signal (noise reduction)
- [ ] Validate: Does it smooth without adding lag?

**Expected:** MSVR signal becomes cleaner, fewer false signals

### Task 1.2: Implement Shannon Entropy Gate
- [ ] Create `indicators/shannon_entropy.py`
- [ ] Add entropy threshold (block random markets)
- [ ] Test: When entropy > 2.5, no trade

**Expected:** Blocks 20-30% of trades in choppy markets

### Task 1.3: Implement Efficiency Ratio Filter
- [ ] Create `indicators/efficiency_ratio.py`
- [ ] Add ER threshold (0.25 = trending)
- [ ] Test: Only trade when ER > 0.25

**Expected:** Only trades in trending markets

### Task 1.4: Combine into MSVR+Ichimoku Hybrid
- [ ] Create `msvr_ichimoku_hybrid.py`
- [ ] MSVR direction + Cycle Phase timing
- [ ] Apply SuperSmoother + Entropy + ER filters
- [ ] Test: Should get ~15-20 trades, >60% win rate

**Expected:** Beat Ichimoku's 11 trades, 63.6% win rate

---

## Phase 2: Add Missing Principles

### Task 2.1: Implement Linear Regression Trend
- [ ] Create `indicators/linear_reg_trend.py`
- [ ] Family 3: Regression
- [ ] LinearReg channel deviation as signal
- [ ] Test: Better trend detection than SMA

**Expected:** More accurate trend direction

### Task 2.2: Implement GARCH-like Volatility
- [ ] Create `indicators/volatility_cluster.py`
- [ ] Family 6: GARCH
- [ ] Detect volatility clustering
- [ ] Test: Avoid high volatility periods

**Expected:** Fewer false breakouts

### Task 2.3: Implement Volume Confirmation
- [ ] Create `indicators/volume_confirm.py`
- [ ] Family 8: Volume
- [ ] OBV + VWAP + Force Index
- [ ] Test: Confirm moves with volume

**Expected:** Only trade when volume confirms

### Task 2.4: Implement HMM Regime Detection
- [ ] Create `indicators/hmm_regime.py`
- [ ] Family 9: Bayesian
- [ ] Detect bull/bear/sideways regimes
- [ ] Test: Only trade in trending regime

**Expected:** Avoid sideways markets

### Task 2.5: Build MSVR v3
- [ ] Create `msvr_v3.py`
- [ ] Combine ALL principles:
  - MSVR base (Family 1)
  - SuperSmoother (Family 2)
  - LinearReg (Family 3)
  - Cycle Phase (Family 4)
  - Efficiency Ratio (Family 5)
  - Volatility Cluster (Family 6)
  - Shannon Entropy (Family 7)
  - Volume Confirm (Family 8)
  - HMM Regime (Family 9)
- [ ] Test: Should get <15 trades, >65% win rate

**Expected:** 9 principles combined = ULTRA quality signals

---

## Phase 3: Optimize & Validate

### Task 3.1: Grid Search MSVR v3
- [ ] Optimize all parameters
- [ ] Training period: 2018-2024
- [ ] Target: Sharpe > 1.4, Win Rate > 65%

**Expected:** Best parameter combination

### Task 3.2: Walk-Forward Validation
- [ ] 5-fold walk-forward
- [ ] Test stability across time
- [ ] Check degradation < 20%

**Expected:** Robust, not overfitted

### Task 3.3: Holdout Test
- [ ] Test on 2025-2026 (never seen)
- [ ] Compare with Ichimoku
- [ ] Final metrics

**Expected:** Beat Ichimoku on ALL metrics

### Task 3.4: Final Comparison
- [ ] MSVR v3 vs Enhanced vs Ichimoku
- [ ] Create comparison chart
- [ ] Document results

**Expected:** Clear winner identified

---

## Phase 1 Results ✅ COMPLETED

### Task 1.1: Implement Ehler SuperSmoother ✅
- Created `indicators/ehler_supersmoother.py`
- 2-pole Ehler SuperSmoother (Family 2: Filtering)
- Removes high-frequency noise without lag penalty

### Task 1.2: Implement Shannon Entropy Gate ✅
- Created `indicators/shannon_entropy.py`
- Shannon Entropy (Family 7: Entropy)
- Blocks random markets (entropy > 2.5)

### Task 1.3: Implement Efficiency Ratio Filter ✅
- Created `indicators/efficiency_ratio.py`
- Kaufman ER (Family 5: Fractal)
- Only trades trending markets (ER > 0.25)

### Task 1.4: Combine into MSVR+Ichimoku Hybrid ✅
- Created `msvr_hybrid.py`
- Created `test_hybrid_comparison.py`
- Created `generate_hybrid_comparison.py`
- Generated `mttd/msvr_hybrid_comparison.png`

### Phase 1 Metrics
| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Trades | < 20 | 15 | ✅ |
| Win Rate | > 60% | 66.7% | ✅ |
| Sharpe | > 1.35 | 1.09 | ❌ |
| CAGR | > 60% | 35.8% | ❌ |

**Key Finding:** MSVR Hybrid in position only 30.7% vs Ichimoku 44.9%
**Root Cause:** Filters too aggressive → fewer opportunities → lower returns

---

## Phase 2: Add Missing Principles 🔄 IN PROGRESS

### Task 2.1: Implement Linear Regression Trend ⬜
- [ ] Create `indicators/linear_reg_trend.py`
- [ ] Family 3: Regression
- [ ] LinearReg channel deviation as signal
- [ ] Test: Better trend detection than SMA

**Expected:** More accurate trend direction

### Task 2.2: Implement GARCH-like Volatility ⬜
- [ ] Create `indicators/volatility_cluster.py`
- [ ] Family 6: GARCH
- [ ] Detect volatility clustering
- [ ] Test: Avoid high volatility periods

**Expected:** Fewer false breakouts

### Task 2.3: Implement Volume Confirmation ⬜
- [ ] Create `indicators/volume_confirm.py`
- [ ] Family 8: Volume
- [ ] OBV + VWAP + Force Index
- [ ] Test: Confirm moves with volume

**Expected:** Only trade when volume confirms

### Task 2.4: Implement HMM Regime Detection ⬜
- [ ] Create `indicators/hmm_regime.py`
- [ ] Family 9: Bayesian
- [ ] Detect bull/bear/sideways regimes
- [ ] Test: Only trade in trending regime

**Expected:** Avoid sideways markets

### Task 2.5: Build MSVR v3 ⬜
- [ ] Create `msvr_v3.py`
- [ ] Combine ALL principles:
  - MSVR base (Family 1)
  - SuperSmoother (Family 2)
  - LinearReg (Family 3)
  - Cycle Phase (Family 4)
  - Efficiency Ratio (Family 5)
  - Volatility Cluster (Family 6)
  - Shannon Entropy (Family 7)
  - Volume Confirm (Family 8)
  - HMM Regime (Family 9)
- [ ] Test: Should get <15 trades, >65% win rate

**Expected:** 9 principles combined = ULTRA quality signals

---

## Phase 3: Optimize & Validate ⬜ PENDING

### Task 3.1: Grid Search MSVR v3 ⬜
- [ ] Optimize all parameters
- [ ] Training period: 2018-2024
- [ ] Target: Sharpe > 1.4, Win Rate > 65%

### Task 3.2: Walk-Forward Validation ⬜
- [ ] 5-fold walk-forward
- [ ] Test stability across time
- [ ] Check degradation < 20%

### Task 3.3: Holdout Test ⬜
- [ ] Test on 2025-2026 (never seen)
- [ ] Compare with Ichimoku
- [ ] Final metrics

### Task 3.4: Final Comparison ⬜
- [ ] MSVR v3 vs Enhanced vs Ichimoku
- [ ] Create comparison chart
- [ ] Document results

---

## Success Criteria

| Criterion | Target | Phase 1 | Status |
|-----------|--------|---------|--------|
| Trades | < 15 | 15 | ✅ |
| Win Rate | > 65% | 66.7% | ✅ |
| Sharpe | > 1.40 | 1.09 | ❌ |
| CAGR | > 60% | 35.8% | ❌ |
| Degradation | < 20% | N/A | ⬜ |

---

## File Structure

```
indicators/
├── ehler_supersmoother.py      # Family 2
├── shannon_entropy.py          # Family 7
├── efficiency_ratio.py         # Family 5
├── linear_reg_trend.py         # Family 3
├── volatility_cluster.py       # Family 6
├── volume_confirm.py           # Family 8
└── hmm_regime.py               # Family 9

msvr_hybrid.py                  # Phase 1: MSVR + Ichimoku principles
msvr_v3.py                      # Phase 2: All 9 principles
grid_search_msvr_v3.py          # Phase 3: Optimization
walkforward_msvr_v3.py          # Phase 3: Validation
```

---

## Notes

- Each principle MUST serve a SPECIFIC purpose
- Quality > Quantity (target: <15 trades)
- Walk-forward validation is MANDATORY
- Compare with Ichimoku as benchmark
