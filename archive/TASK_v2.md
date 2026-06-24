# TASK_v2.md — Combine Best Bases + Holdout Validation

## Task 1: Create Combination Engine ⬜

### Subtask 1.1: Create combine_bases.py
- [ ] Load Supertrend signal (vii > 0)
- [ ] Load Keltner signal (price > upper band)
- [ ] Implement 4 combination approaches:
  - AND: Both must agree
  - OR: Either can signal
  - Voting: Majority vote
  - Weighted: 50/50 average
- [ ] Apply common filters (MSVR, SuperSmoother, Cycle, Entropy)
- [ ] Output results for each approach

### Subtask 1.2: Test All Approaches
- [ ] Run AND combination
- [ ] Run OR combination
- [ ] Run Voting combination
- [ ] Run Weighted combination
- [ ] Print comparison table
- [ ] Identify best approach

**Expected:** 4 results to compare

---

## Task 2: Parameter Optimization ⬜

### Subtask 2.1: Grid Search on Best Approach
- [ ] Test min_hold: [20, 25, 30, 35]
- [ ] Test max_hold: [50, 60, 75, 90]
- [ ] Test gate_threshold: [2, 3, 4]
- [ ] Total: 4 × 4 × 3 = 48 combinations

### Subtask 2.2: Find Best Parameters
- [ ] Rank by Sharpe
- [ ] Rank by Win Rate
- [ ] Rank by CAGR
- [ ] Identify best balanced config

**Expected:** Best parameter set identified

---

## Task 3: Holdout Validation ⬜

### Subtask 3.1: Split Data
- [ ] Training: 2018-01-01 to 2024-12-31
- [ ] Holdout: 2025-01-01 to 2026-06-22

### Subtask 3.2: Train on Training Set
- [ ] Run best config on training data
- [ ] Record: Sharpe, Win Rate, Trades, CAGR, Max DD

### Subtask 3.3: Test on Holdout Set
- [ ] Run same config on holdout data
- [ ] Record: Sharpe, Win Rate, Trades, CAGR, Max DD

### Subtask 3.4: Compare Results
- [ ] Calculate degradation for each metric
- [ ] Check if degradation < 20%
- [ ] Pass/Fail holdout test

**Expected:** Training vs Holdout comparison

---

## Task 4: Final Comparison ⬜

### Subtask 4.1: Compare All Systems
- [ ] Supertrend-only
- [ ] Keltner-only
- [ ] Best Combination
- [ ] MSVR v8 (previous best)

### Subtask 4.2: Generate Chart
- [ ] Create comparison chart
- [ ] Save to mttd/combination_comparison.png

### Subtask 4.3: Document Findings
- [ ] Write COMBINATION_RESULTS.md
- [ ] Commit all files

**Expected:** Final comparison and documentation

---

## Success Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| Sharpe (Training) | > 1.20 | ⬜ |
| Sharpe (Holdout) | > 1.00 | ⬜ |
| Win Rate (Training) | > 55% | ⬜ |
| Win Rate (Holdout) | > 50% | ⬜ |
| Trades | 25-40 | ⬜ |
| CAGR (Training) | > 40% | ⬜ |
| Max DD | < -40% | ⬜ |
| Degradation | < 20% | ⬜ |

---

## File Structure

```
combine_bases.py              # Main combination engine
optimize_combination.py       # Parameter grid search
holdout_combination.py        # Holdout validation
compare_all_systems.py        # Final comparison
mttd/combination_comparison.png  # Chart
COMBINATION_RESULTS.md        # Documentation
```

---

## Timeline

| Task | Duration | Deliverable |
|------|----------|-------------|
| Task 1 | 5 min | Combination engine |
| Task 2 | 5 min | Best parameters |
| Task 3 | 5 min | Holdout validation |
| Task 4 | 5 min | Final comparison |
| **Total** | **20 min** | **Complete system** |
