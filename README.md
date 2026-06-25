# MTTD v2 — Multi-Principle Bitcoin Trend Following

**6+ statistical families** working together for robust signal generation.

## Overview

MTTD v2 uses a **multi-principle architecture** (ported from `ichimoku_quant`) combining Smoothing, Filtering, Fractal, Entropy, Spectral, and Momentum families into a single composite signal.

| Component | Family | Role |
|-----------|--------|------|
| Ichimoku lines | Smoothing | Base trend structure |
| Ehler SuperSmoother | Filtering | Noise reduction |
| IMO composite | Spectral | Normalized cycle signal |
| Efficiency Ratio | Fractal | Trend strength gate |
| Shannon Entropy | Entropy | Noise detection gate |
| S_Chikou | Momentum | Exit timing |

## Performance

| Metric | Value | Target |
|--------|-------|--------|
| Trades | **25** (2018-2026) | 20-35 ✅ |
| Win Rate | **60%** | > 60% ✅ |
| Sharpe Ratio | **1.28** | > 1.0 ✅ |
| CAGR | **51.5%** | > 50% ✅ |
| Max Drawdown | -39.5% | Manageable ✅ |
| Avg Hold | 49 days | Medium-term ✅ |
| Avg Win | +39.9% | — |
| Avg Loss | -10.0% | — |
| Win/Loss Ratio | **3.98** | Excellent ✅ |
| Total Return | **3,268.5%** | 33.69x |

## Quick Start

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

## Core Strategy

```
IMO = tanh( S_TK + S_Cloud + S_Future + S_Chikou ) / 4

ENTRY (ALL must pass):
  1. IMO > IMO_STD × 0.25    (Adaptive threshold)
  2. ER > 0.20               (Trend strength)
  3. Entropy < 2.3           (Noise gate)
  4. Close >= Cloud_Min      (Cloud filter)
  5. 2-bar confirmation      (Persistence)

EXIT (ANY can trigger):
  1. S_Chikou < -0.30        (Momentum death)
  2. IMO < -0.30             (Trend death)

IMMUNITY (hold through bull):
  IMO >= 0.50 OR Close >= Cloud_Max AND ROC >= -0.20 AND IMO >= -0.30
```

## Best Config

```python
t_entry = 0.25          # IMO threshold multiplier
er_entry = 0.20         # Efficiency Ratio minimum
entropy_thresh = 2.3    # Shannon Entropy maximum
min_hold_days = 10      # Minimum hold before exit
max_hold_days = 60      # Maximum hold (forced exit)
chikou_thresh = -0.30   # Chikou momentum exit
immunity_thresh = 0.50  # Extreme bull immunity
cooldown = 5            # Days after exit before re-entry
```

## The 10 Statistical Families

| # | Family | Used In Strategy |
|---|--------|-----------------|
| 1 | **Smoothing** | Ichimoku lines (tenkan/kijun/senkou) |
| 2 | **Filtering** | Ehler SuperSmoother on IMO ✅ |
| 3 | **Regression** | LinearReg channel (signals module) |
| 4 | **Spectral** | FFT cycle phase (signals module) |
| 5 | **Fractal** | Efficiency Ratio gate ✅ |
| 6 | **GARCH** | Volatility cluster (signals module) |
| 7 | **Entropy** | Shannon Entropy gate ✅ |
| 8 | **Chaos** | Phase space (signals module) |
| 9 | **Bayesian** | HMM regime (signals module) |
| 10 | **ML Hybrid** | Composite scoring (signals module) |

## Outputs

- `mttd/multi_principle/` — Strategy results and charts
- `mttd/multi_principle/trade_chart.png` — Clean trade chart
- `mttd/multi_principle/results.json` — Performance metrics
- `mttd/multi_principle/signals.csv` — Daily position signals

## Trade Examples

```
Entry → Exit          Return    Hold
───────────────────────────────────────
2019-03 → 2019-05     +99.6%    60d ✅
2020-11 → 2021-01    +117.8%    60d ✅
2021-01 → 2021-03     +71.9%    60d ✅
2024-01 → 2024-03     +63.2%    60d ✅
2024-10 → 2024-12     +57.1%    60d ✅
2023-06 → 2023-08     -13.1%    58d ❌
2021-04 → 2021-05     -21.9%    29d ❌
```

## Comparison

| System | Trades | Win% | Sharpe | CAGR | Total Return |
|--------|--------|------|--------|------|-------------|
| **MTTD v2** | **25** | **60%** | **1.28** | **51.5%** | **3,268%** |
| ISP (benchmark) | 17 | 100% | 2.36 | 114.7% | — |
| OLD MTTD v1 | 24 | 50% | 0.62 | 16.4% | — |

## Archive

Old systems, grid searches, and tests archived in `archive/`.
- `archive/old_systems/` — MSVR v2-v8, old ensemble, etc.
- `archive/old_grids/` — Old optimization runs
- `archive/old_tests/` — Test scripts and audits

## Data Sources

- BTC daily OHLCV: `data/btc_daily.json`
- On-chain metrics: `../quant-btc-valuation-system/database/metrics.db`
- ISP signals: `isp-signals-btcusd-2026-06-21.csv`

## License

Research use only. Not financial advice.
