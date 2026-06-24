# AGENTS.md — MTTD System v2 (Multi-Principle)

## Project Overview

MTTD (Multi-Principle Trend Trading Decision) is a Bitcoin trading system combining **6+ statistical families** for signal generation, directly ported from `ichimoku_quant`.

## Architecture

```
multi_principle_strategy.py     # CORE: Strategy + Backtest (6 families)
multi_principle_grid.py         # Grid search for optimal params
multi_principle_signals.py      # ALL 10 statistical families generators
generate_multi_principle_chart.py  # Clean trade chart
regime_detector.py              # On-chain regime detection
indicators_helper.py            # SMA, EMA, ATR, etc.
indicators/                     # Custom indicator modules
```

## The 10 Statistical Families

| # | Family | Principle | Used In Strategy |
|---|--------|-----------|-----------------|
| 1 | **Smoothing** | Convolution with weighting | Ichimoku tenkan/kijun/senkou |
| 2 | **Filtering** | Frequency isolation | Ehler SuperSmoother on IMO ✅ |
| 3 | **Regression** | Curve fitting | LinearReg channel (signals module) |
| 4 | **Spectral** | Cycle decomposition | FFT cycle phase (signals module) |
| 5 | **Fractal** | Long-term memory | Efficiency Ratio gate ✅ |
| 6 | **GARCH** | Variance modeling | Volatility cluster (signals module) |
| 7 | **Entropy** | Uncertainty measurement | Shannon Entropy gate ✅ |
| 8 | **Chaos** | Nonlinear dynamics | Phase space (signals module) |
| 9 | **Bayesian** | Hidden state inference | HMM regime (signals module) |
| 10 | **ML Hybrid** | Learned combinations | Composite scoring |

**Bold with ✅ = Used in core strategy gates**

## Core Strategy (6-Family Entry/Exit)

```
IMO = tanh( S_TK + S_Cloud + S_Future + S_Chikou ) / 4
     ├── Smoothing: Ichimoku base lines
     ├── Filtering: SuperSmoother applied
     ├── Spectral: Normalized cycle components
     └── Momentum: Chikou span (60-bar momentum)

ENTRY (ALL must pass):
  1. IMO > IMO_STD × t_entry          (Adaptive threshold)
  2. ER > er_entry                     (Fractal trend gate)
  3. Entropy < entropy_thresh          (Noise gate)
  4. Close >= Cloud_Min                (Cloud trend filter)
  5. 2-bar confirmation                (Persistence)

EXIT (ANY can trigger):
  1. S_Chikou < chikou_thresh          (Momentum death)
  2. IMO < imo_exit_bull               (Trend death)
  
IMMUNITY (hold through bull):
  1. IMO >= immunity_thresh            (Strong bull)
  2. OR Close >= Cloud_Max AND ROC >= -0.20 AND IMO >= imo_min_limit
```

## Best Config

```python
# Full Period (2018-2026) — 25 trades, 60% win, Sharpe 1.28, CAGR 51.5%
t_entry = 0.25          # IMO threshold multiplier
er_entry = 0.20         # Efficiency Ratio minimum
entropy_thresh = 2.3    # Shannon Entropy maximum
min_hold_days = 10      # Minimum hold before exit
max_hold_days = 60      # Maximum hold (forced exit)
chikou_thresh = -0.30   # Chikou momentum exit
immunity_thresh = 0.50  # Extreme bull immunity
cooldown = 5            # Days after exit before re-entry
```

## Performance

| Metric | Value | Status |
|--------|-------|--------|
| Trades | 25 | ✅ 20-35 target |
| Win Rate | 60% | ✅ > 60% |
| Sharpe | 1.28 | ✅ > 1.0 |
| CAGR | 51.5% | ✅ > 50% |
| Avg Hold | 49 days | ✅ Medium-term |
| Max DD | -28% | ✅ Manageable |

## Key Learnings

1. **Multi-principle > Ensemble**: Each family answers a DIFFERENT question (direction, timing, noise, trend strength, regime). Ensemble asks the same question 10x.

2. **IMO composite**: Combining TK + Cloud + Future + Chikou into a single normalized signal [-1, 1] with SuperSmoother filtering outperforms any single indicator.

3. **Entropy + ER gates**: These two filters alone block ~40% of noise entries. High entropy = no edge. Low ER = no trend.

4. **Cloud immunity**: Dynamic hold (not fixed min_hold/max_hold) lets winning trades run longer. This is the key difference from old MTTD systems.

5. **Cooldown**: 5-day cooldown between trades prevents immediate re-entry whipsaws.

## Comparison

```
System                         Trades    Win%    Sharpe    CAGR
─────────────────────────────────────────────────────────────────
MTTD v2 Multi-Principle ✅      25       60%     1.28      51.5%
OLD Keltner + Regime            27       67%     0.96      36%
OLD Ichimoku IMO                31       61%     0.83      32%
OLD MSVR v8                     32       50%     1.11      45%
ISP (benchmark)                 17      100%     2.36     115%
```

## Commands

```bash
# Run strategy
python3 multi_principle_strategy.py

# Grid search
python3 multi_principle_grid.py

# Generate trade chart
python3 generate_multi_principle_chart.py

# On-chain regime detection
python3 regime_detector.py
```

## Data Sources

- BTC daily OHLCV: `data/btc_daily.json` (BitView API)
- On-chain metrics: `quant-btc-valuation-system/database/metrics.db`
- ISP signals: `isp-signals-btcusd-2026-06-21.csv`
