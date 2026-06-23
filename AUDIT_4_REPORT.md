# Audit 4 — Current 15 Indicators: Full Report

**Date:** 2026-06-22
**Scope:** All 15 indicators in SELECTED_INDICATORS
**Benchmark:** ISP Signals (isp-signals-btcusd-2026-06-13.csv)
**Evaluation Range:** 2015-10-28 to 2025-08-21 (2790 days)

---

## Ranked by Coherence with ISP

| Rank | Indicator                                      | Coherence% | Trades | Sharpe | MaxDD%  | Tier   |
|------|------------------------------------------------|------------|--------|--------|---------|--------|
| 1    | DSMA \| viResearch                             | 68.57      | 53     | 2.17   | -25.65  | TOP    |
| 2    | HILO Interpolation \| QuantEdgeB               | 67.13      | 38     | 2.24   | -23.66  | TOP    |
| 3    | Gaussian Smooth Trend \| QuantEdgeB            | 65.09      | 31     | 2.15   | -30.86  | TOP    |
| 4    | Median Deviation Suite \| InvestorUnknown      | 64.62      | 49     | 2.34   | -23.17  | MID    |
| 5    | Linear % ST \| QuantEdgeB                      | 64.19      | 41     | 2.32   | -40.43  | MID    |
| 6    | alma lag \| viResearch                         | 63.58      | 36     | 2.10   | -26.23  | MID    |
| 7    | IRS`Elder Force Volume Index                   | 63.30      | 67     | 2.37   | -25.65  | MID    |
| 8    | Root Mean Square Deviation Trend               | 63.19      | 31     | 1.89   | -34.06  | MID    |
| 9    | Polynomial Deviation Bands                     | 61.25      | 39     | 1.85   | -35.48  | MID    |
| 10   | Adaptive Regime Cloud                          | 60.65      | 47     | 2.22   | -23.32  | MID    |
| 11   | MadTrend \| InvestorUnknown                    | 59.82      | 41     | 2.14   | -29.81  | BOTTOM |
| 12   | lsma \| viResearch                             | 59.39      | 38     | 2.26   | -27.42  | BOTTOM |
| 13   | Quantile DEMA Trend \| QuantEdgeB              | 58.75      | 33     | 1.99   | -29.62  | BOTTOM |
| 14   | DEGA RMA \| QuantEdgeB                         | 58.17      | 56     | 2.46   | -25.00  | BOTTOM |
| 15   | Adaptive Volatility Controlled LSMA \| QuantAlgo | 55.56    | 40     | 1.98   | -30.55  | BOTTOM |

---

## Tier Summary

### TOP TIER (3 indicators, avg coherence=66.9%, avg sharpe=2.19)
- DSMA | viResearch (68.6%)
- HILO Interpolation | QuantEdgeB (67.1%)
- Gaussian Smooth Trend | QuantEdgeB (65.1%)

### MID TIER (7 indicators, avg coherence=63.0%, avg sharpe=2.16)
- Median Deviation Suite | InvestorUnknown (64.6%)
- Linear % ST | QuantEdgeB (64.2%)
- alma lag | viResearch (63.6%)
- IRS`Elder Force Volume Index (63.3%)
- Root Mean Square Deviation Trend (63.2%)
- Polynomial Deviation Bands (61.3%)
- Adaptive Regime Cloud (60.6%)

### BOTTOM TIER (5 indicators, avg coherence=58.3%, avg sharpe=2.17)
- MadTrend | InvestorUnknown (59.8%)
- lsma | viResearch (59.4%)
- Quantile DEMA Trend | QuantEdgeB (58.7%)
- DEGA RMA | QuantEdgeB (58.2%)
- Adaptive Volatility Controlled LSMA | QuantAlgo (55.6%)

---

## Bottom 5 Indicators (Pruning Candidates)

| # | Indicator                                        | Coherence | Sharpe | Trades | MaxDD  |
|---|--------------------------------------------------|-----------|--------|--------|--------|
| 15| Adaptive Volatility Controlled LSMA \| QuantAlgo | 55.56%    | 1.98   | 40     | -30.55%|
| 14| DEGA RMA \| QuantEdgeB                           | 58.17%    | 2.46   | 56     | -25.00%|
| 13| Quantile DEMA Trend \| QuantEdgeB                | 58.75%    | 1.99   | 33     | -29.62%|
| 12| lsma \| viResearch                               | 59.39%    | 2.26   | 38     | -27.42%|
| 11| MadTrend \| InvestorUnknown                      | 59.82%    | 2.14   | 41     | -29.81%|

---

## Pruning Decisions

### KEEP (13 indicators)

| Indicator                                        | Coherence | Rationale                                    |
|--------------------------------------------------|-----------|----------------------------------------------|
| DSMA \| viResearch                               | 68.57%    | Top performer, excellent coherence           |
| HILO Interpolation \| QuantEdgeB                 | 67.13%    | Strong coherence, low drawdown               |
| Gaussian Smooth Trend \| QuantEdgeB              | 65.09%    | Top tier, smooth signals                     |
| Median Deviation Suite \| InvestorUnknown        | 64.62%    | Strong Sharpe (2.34), low drawdown           |
| Linear % ST \| QuantEdgeB                        | 64.19%    | Strong Sharpe (2.32)                         |
| alma lag \| viResearch                           | 63.58%    | Solid mid-tier performer                     |
| IRS`Elder Force Volume Index                     | 63.30%    | Best Sharpe (2.37) of mid-tier               |
| Root Mean Square Deviation Trend                 | 63.19%    | Stable performer, low trades (31)            |
| Polynomial Deviation Bands                       | 61.25%    | Above 60% coherence threshold                |
| Adaptive Regime Cloud                            | 60.65%    | Above 60% threshold, strong Sharpe (2.22)    |
| MadTrend \| InvestorUnknown                      | 59.82%    | Borderline coherence, strong Sharpe (2.14)   |
| lsma \| viResearch                               | 59.39%    | Borderline coherence, strong Sharpe (2.26)   |
| DEGA RMA \| QuantEdgeB                           | 58.17%    | Borderline coherence, best Sharpe (2.46)     |

### PRUNE (2 indicators)

| Indicator                                        | Coherence | Sharpe | Rationale                                        |
|--------------------------------------------------|-----------|--------|--------------------------------------------------|
| Adaptive Volatility Controlled LSMA \| QuantAlgo | 55.56%    | 1.98   | Lowest coherence by far, below 58% threshold     |
| Quantile DEMA Trend \| QuantEdgeB                | 58.75%    | 1.99   | Borderline coherence + low Sharpe (<2.0)         |

---

## Summary Statistics

| Metric              | Min    | Mean   | Median | Max    |
|---------------------|--------|--------|--------|--------|
| Coherence%          | 55.56  | 62.22  | 63.19  | 68.57  |
| Trades              | 31     | 42.67  | 40     | 67     |
| Avg Hold Days       | 21.5   | 35.0   | 36.5   | 46.6   |
| Sharpe Ratio        | 1.85   | 2.17   | 2.17   | 2.46   |
| Max Drawdown%       | -40.43 | -28.73 | -27.42 | -23.17 |

---

## Key Observations

1. **No individual indicator exceeds 70% coherence** — ensemble approach is essential
2. **Average coherence is 62.2%** — individual indicators are modest aligners
3. **Sharpe ratios are strong (1.85-2.46)** — indicators generate positive returns
4. **Drawdowns are significant (-23% to -40%)** — risk management is critical
5. **Trade frequency varies widely (31-67 trades)** — diversity in signal generation

## Recommendations for TODO 5/6

1. Search the indicator bank for higher-coherence indicators (>60% individually)
2. Target 5-10 new indicators to replace the 2 pruned ones
3. Ensemble aggregation should compensate for individual indicator weakness
4. Final selection should target 15-25 indicators total
