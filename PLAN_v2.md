# PLAN_v2.md — Combine Best Bases + Holdout Validation

## Current State

### Best Individual Bases
| Base Signal | Trades | WinRate | Sharpe | CAGR |
|-------------|--------|---------|--------|------|
| Supertrend | 47 | 53.2% | 1.08 | 46.9% |
| Keltner | 39 | 56.4% | 1.06 | 45.8% |
| Ichimoku | 27 | 40.7% | 0.83 | 27.8% |

### Filtering Framework
- MSVR direction
- SuperSmoother direction
- Cycle Phase direction
- Shannon Entropy gate (< 2.8)
- Gate threshold: 3 (of 4 filters + base = 5 signals)

## Strategy: Combine Best Bases

### Approach 1: AND Logic (Strict)
```
Signal = Supertrend × Keltner × Filters
```
- Only trade when BOTH Supertrend AND Keltner agree
- Expected: Fewer trades, higher win rate

### Approach 2: OR Logic (Relaxed)
```
Signal = (Supertrend OR Keltner) × Filters
```
- Trade when EITHER Supertrend OR Keltner signals
- Expected: More trades, potentially higher Sharpe

### Approach 3: Voting (Majority)
```
Signal = (Supertrend + Keltner + Filters >= threshold)
```
- Majority vote among all signals
- Expected: Balanced approach

### Approach 4: Weighted Combination
```
Signal = (Supertrend × 0.5 + Keltner × 0.5) × Filters
```
- Weighted average of signals
- Expected: Smoothed signal

## Holdout Validation (CRITICAL)

### Split
- **Training**: 2018-01-01 to 2024-12-31 (7 years)
- **Holdout**: 2025-01-01 to 2026-06-22 (1.5 years)

### Metrics to Compare
| Metric | Training | Holdout | Degradation |
|--------|----------|---------|-------------|
| Sharpe | ? | ? | < 20% |
| Win Rate | ? | ? | < 10% |
| Trades | ? | ? | — |
| CAGR | ? | ? | < 30% |

## Target Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| Sharpe | > 1.20 | Conservative target |
| Win Rate | > 55% | Achievable with TA |
| Trades | 25-40 | Medium-term |
| CAGR | > 40% | Good returns |
| Max DD | < -40% | Acceptable risk |
| Degradation | < 20% | Robustness |

## Tasks

### Task 1: Create Combination Engine
- Test 4 combination approaches (AND, OR, Voting, Weighted)
- Use same filtering framework
- Output: trades, win rate, Sharpe, CAGR

### Task 2: Parameter Optimization
- Test different min_hold (20, 25, 30, 35)
- Test different max_hold (50, 60, 75, 90)
- Test different gate_threshold (2, 3, 4)

### Task 3: Holdout Validation
- Train on 2018-2024
- Test on 2025-2026
- Compare training vs holdout metrics

### Task 4: Final Comparison
- Compare best combination with individual bases
- Generate comparison chart
- Document findings

## Expected Outcome

```
Best Combination:
- Supertrend + Keltner + Filters
- 25-40 trades
- > 55% win rate
- > 1.20 Sharpe
- < 20% degradation
```
