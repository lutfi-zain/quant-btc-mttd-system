# AGENTS.md — MTTD System

## Project Overview

MTTD (Medium-Term Trend following Consensus) is a Bitcoin trading system combining multiple statistical principles for signal generation.

## Architecture

```
mttd_system.py          # Main system - ROBUST config
generate_charts.py      # 4-panel performance chart
serve_web.py            # Web dashboard (port 8080)
indicators_helper.py    # SMA, EMA, ATR, etc.
ichimoku_quant.py       # Ichimoku (Family 2,5,7)
ensemble_engine.py      # Majority vote ensemble
ensemble_robust.py      # Robust ensemble with outlier rejection
inter_indicator_coherence.py  # Pairwise coherence metrics
report_generator.py     # Equity report
risk_management.py      # Position sizing, stop-loss
```

## Key Config (ROBUST)

```python
# T75/250_BB25_2.0s_MH45 — Most Robust Config
TREND_FAST = 75
TREND_SLOW = 250
BB_PERIOD = 25
BB_STD = 2.0
MIN_HOLD = 45
CYCLE_LOOKBACK = 40
```

## Performance

| Metric | Training | Holdout | Status |
|--------|----------|---------|--------|
| Sharpe | 0.62 | 0.62 | ✅ +0% |
| Win Rate | 47% | 60% | ✅ |
| CAGR | 16% | 16% | ✅ |

## Session Learnings

### 1. Ensemble Doesn't Help on Single Asset

**Finding:** Ensemble of correlated indicators doesn't improve performance on BTC (single asset).

**Reason:** All technical indicators are correlated on same asset. Majority vote = noise averaging.

**Lesson:** Use ensemble for multi-asset, not single asset.

### 2. Cycle Phase Timing is Key

**Finding:** FFT-based cycle phase timing combined with MSVR direction improves Sharpe from 0.40 to 1.48.

**Why:** MSVR provides DIRECTION (trend), Cycle Phase provides TIMING (trough/peak). Complementary questions.

**Lesson:** Ask complementary questions, not the same question twice.

### 3. ISP Behavior Pattern

**Finding:** ISP trades 17 times in 7 years, holds 62 days avg. Uses on-chain + sentiment data.

**Gap:** Technical indicators alone max out at ~55-60% win rate. ISP achieves 100% with proprietary data.

**Lesson:** Technical analysis has fundamental limits. On-chain/sentiment data is the edge.

### 4. Overfitting is Real

**Finding:** MSVR_ONLY_ICH showed 70% win rate in training but 33% in holdout (-100% degradation).

**Lesson:** Walk-forward validation is MANDATORY. Simple configs are more robust.

### 5. Robust > High Performance

**Finding:** T75/250_BB25_2.0s_MH45 has +0% degradation (Training ≈ Holdout). Higher-performing configs have -90% degradation.

**Lesson:** Prefer robustness over peak performance. Consistency > optimization.

## Anti-Patterns

1. **Over-optimization** — Testing 1000+ configs leads to overfitting
2. **Complex signals** — More parameters = more overfitting risk
3. **Ignoring regime** — Bull/bear markets behave differently
4. **No transaction costs** — Always include 0.1% round-trip

## Future Work

1. Add on-chain data (exchange flows, whale alerts)
2. Add sentiment data (Fear/Greed, funding rates)
3. Multi-asset ensemble (BTC + ETH + SOL)
4. Position sizing (Kelly criterion)
5. Stop-loss system (trailing stop)

## Commands

```bash
# Run system
python3 mttd_system.py

# Generate charts
python3 generate_charts.py

# Start web dashboard
python3 serve_web.py

# Grid search
python3 grid_search_best.py

# Walk-forward validation
python3 walkforward_all_options.py
```

## Data Sources

- BTC daily OHLCV: `data/btc_daily.json` (from BitView API)
- ISP signals: `isp-signals-btcusd-2026-06-21.csv`
- Indicator bank: `/home/ubuntu/projects/quant-technical-indicator-bank/`

## Dependencies

- Python 3.10+
- pandas, numpy, scipy, matplotlib
- No external ML libraries required
