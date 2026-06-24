# MTTD System — Bitcoin Trend Following

**Medium-Term Trend following Consensus** with spectral cycle phase timing.

## Overview

MTTD combines multiple statistical principles to generate trading signals for Bitcoin:

| Component | Principle | Family |
|-----------|-----------|--------|
| MSVR | Median + DEMA + ATR bands | Smoothing (Family 1) |
| Cycle Phase | FFT spectral analysis | Spectral (Family 4) |
| Trend Filter | 75/250 SMA crossover | Smoothing (Family 1) |
| Bollinger Filter | Volatility bands | Filtering (Family 2) |
| Min Hold | 45-day minimum hold | Behavioral |

## Performance

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 0.62 |
| CAGR | 16.4% |
| Max Drawdown | -57% |
| Win Rate | 50% |
| Trades | 24 (2018-2026) |
| Avg Hold | 47 days |

## Quick Start

```bash
# Run the system
python3 mttd_system.py

# Generate charts
python3 generate_charts.py

# Start web dashboard
python3 serve_web.py
# Open http://localhost:8080
```

## Architecture

```
mttd_system.py      # Main system - generates signals
generate_charts.py  # Creates performance charts
serve_web.py        # Web dashboard server
indicators_helper.py # Helper functions (SMA, EMA, etc.)
ichimoku_quant.py   # Ichimoku implementation (Family 2,5,7)
```

## Config

```python
TREND_FAST = 75      # Fast SMA period
TREND_SLOW = 250     # Slow SMA period
BB_PERIOD = 25       # Bollinger period
BB_STD = 2.0         # Bollinger standard deviations
MIN_HOLD = 45        # Minimum holding period (days)
CYCLE_LOOKBACK = 40  # FFT cycle detection lookback
```

## Signal Generation

1. **MSVR**: Direction (trend/breakout)
2. **Cycle Phase**: Timing (FFT trough/peak)
3. **Trend Filter**: Only trade with major trend
4. **Bollinger**: Only trade in normal volatility
5. **Min Hold**: Force patience (45 days minimum)

## Outputs

- `mttd/signals.csv` — Daily position signals
- `mttd/equity.csv` — Equity curve and drawdown
- `mttd/metrics.json` — Performance metrics
- `mttd/system_performance.png` — 4-panel chart

## Validation

- Walk-forward: 5 folds with 10-day embargo
- Degradation: +0% (Training ≈ Holdout)
- Statistical significance: p < 0.05

## Archive

Old grid search and test files moved to `archive/` directory.
See `archive/README.md` for details.

## License

Research use only. Not financial advice.
