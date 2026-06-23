# 🔬 MTTD Ensemble Trading System

## Medium-Term Trend Following Consensus for BTC/USD

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![TradingView](https://img.shields.io/badge/TradingView-Indicators-orange.svg)](#indicators)

---

## 📊 System Overview

The **MTTD (Medium-Term Trend Following Consensus)** system is an ensemble trading strategy that combines **10 technical indicators** to generate binary buy/sell signals for BTC/USD. The system uses equal-weighted voting across multiple indicator families to achieve robust trend detection.

### Key Metrics

| Metric | MTTD System | ISP Benchmark | Buy & Hold |
|--------|-------------|---------------|------------|
| **CAGR** | 260.69% | 104.40% | 59.04% |
| **Sharpe Ratio** | 3.04 | 0.66 | 1.01 |
| **Sortino Ratio** | 3.56 | 3.29 | 1.34 |
| **Max Drawdown** | -26.36% | -6.77% | -83.40% |
| **Calmar Ratio** | 9.89 | 15.42 | 0.71 |
| **Time-Coherence with ISP** | 80.22% | - | - |

---

## 🧠 Quantitative Research Foundation

### 1. Ensemble Theory

The system is built on the **wisdom of crowds** principle applied to technical analysis. By combining multiple independent indicators, we reduce idiosyncratic noise while preserving signal quality.

**Mathematical Framework:**

```
Ensemble_Signal(t) = (1/N) × Σ Indicator_i(t)
```

Where:
- `N` = number of indicators (10)
- `Indicator_i(t)` ∈ {-1, +1} (binary position)
- `Ensemble_Signal(t)` ∈ [-1, +1] (continuous vote)
- **Final Position** = 100% BTC if `Ensemble_Signal(t) > threshold`, else 0% cash

### 2. Time-Coherence Analysis

We measure **time-coherence** as the percentage of time our system's position (in/out of market) matches the ISP benchmark:

```
Time-Coherence = (Bars_Agree / Total_Bars) × 100%
```

**Coherence Breakdown:**
- Both in market: 36.85%
- Both out of market: 43.37%
- MTTD in, ISP out: 1.65%
- MTTD out, ISP in: 18.14%

### 3. Walk-Forward Validation

The system undergoes rigorous **walk-forward validation** with 5-day embargo between train/test windows:

| Cycle | Train Period | Test Period | OOS Coherence | Return |
|-------|--------------|-------------|---------------|--------|
| 1 | 2018 | 2019 | 91.51% | +87.11% |
| 2 | 2018-2019 | 2020 | 73.22% | +178.29% |
| 3 | 2018-2020 | 2021 | 85.75% | +67.99% |
| 4 | 2018-2021 | 2022 | 85.48% | -36.29% |
| 5 | 2018-2022 | 2023 | 51.23% | +23.38% |
| 6 | 2018-2023 | 2024 | 60.66% | +50.97% |
| 7 | 2018-2024 | 2025 | 86.70% | -3.11% |

**Average OOS Coherence:** 76.36%

---

## 📈 Selected Indicators

The system uses **10 indicators** selected through grid search optimization for maximum coherence with ISP benchmark.

| # | Indicator | Category | Coherence | TradingView |
|---|-----------|----------|-----------|-------------|
| 1 | **Polynomial Deviation Bands** | Perpetual | 85.56% | [Link](https://www.tradingview.com/script/u92vNY7X-Polynomial-Deviation-Bands/) |
| 2 | **Gaussian Smooth Trend** | Perpetual | 84.75% | [Link](https://www.tradingview.com/script/yanZWp7u-Gaussian-Smooth-Trend-QuantEdgeB/) |
| 3 | **ALMA Lag** | Perpetual | 84.68% | [Link](https://www.tradingview.com/script/kSU07jis-alma-lag-viResearch/) |
| 4 | **Adaptive Regime Cloud** | Perpetual | 84.56% | [Link](https://www.tradingview.com/script/23ILmPat-Adaptive-Regime-Cloud/) |
| 5 | **Root Mean Square Deviation Trend** | Perpetual | 83.94% | [Link](https://www.tradingview.com/script/) |
| 6 | **P-Motion Trend** | Perpetual | 83.68% | [Link](https://www.tradingview.com/script/Zdtfv3yc-P-Motion-Trend-QuantEdgeB/) |
| 7 | **Z-Score SMMA** | Oscillator | 83.49% | [Link](https://www.tradingview.com/script/8I50ufPD-Z-SMMA-QuantEdgeB/) |
| 8 | **Median RSI SD** | Oscillator | 83.33% | [Link](https://www.tradingview.com/script/iI4rCy2S-Median-RSI-SD-QuantEdgeB/) |
| 9 | **DEMA Adjusted ATR** | Perpetual | 83.00% | [Link](https://www.tradingview.com/script/FqwJYrJP-DEMA-Adjusted-Average-True-Range-BackQuant/) |
| 10 | **Kalman Filtered RSI** | Oscillator | 81.45% | [Link](https://www.tradingview.com/script/0YrX18dJ-Kalman-Filtered-RSI-Oscillator-BackQuant/) |

### Indicator Categories

**Oscillators (3):** Z-Score SMMA, Median RSI SD, Kalman Filtered RSI
- Measure overbought/oversold conditions
- Generate mean-reversion signals

**Perpetual (7):** Polynomial, Gaussian, ALMA, Adaptive, RMS, P-Motion, DEMA
- Measure trend direction and strength
- Generate trend-following signals

---

## 🔧 System Architecture

```
mttd/
├── execute_system.py          # Main orchestration
├── ensemble_engine.py         # Signal aggregation
├── coherence_metrics.py       # ISP comparison
├── calibrate_threshold.py     # Parameter optimization
├── walk_forward_validate.py   # Out-of-sample testing
├── risk_management.py         # Drawdown protection
├── indicators_helper.py       # Shared utilities
├── mttd_data.json             # Output signals
└── data/
    └── btc_daily.json         # Price cache
```

### Pipeline

1. **Data Loading** → Fetch BTC OHLCV from BitView API
2. **Indicator Calculation** → Compute all 10 indicators
3. **Signal Matrix** → Binary position for each indicator
4. **Ensemble Voting** → Equal-weighted average
5. **Threshold Calibration** → Optimize against ISP
6. **Risk Management** → Drawdown pause protection
7. **Output Generation** → JSON for dashboard

---

## 📊 ISP Benchmark Comparison

**ISP (Investment Signal Provider)** is a proprietary signal service that uses regime-based trend following.

**ISP Methodology (Reverse-Engineered):**
- 3-tier regime classification: Weak Bull (50%), Strong Bull (100%), Neutral (0%)
- Graduated position sizing: 0% → 50% → 100% → 50% → 0%
- ~2.85 trades per year
- Average holding period: ~162 days

**MTTD vs ISP:**
| Aspect | MTTD | ISP |
|--------|------|-----|
| Position Sizing | Binary (0/100%) | Graduated (0/50/100%) |
| Trade Frequency | 38 trades / 8 years | 28 trades / 10 years |
| Regime Detection | Technical indicators | On-chain + technical |
| Max Drawdown | -26.36% | -6.77% |

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy pyyaml
```

### Run System

```bash
python execute_system.py
```

### Output

- `mttd_data.json` → Full system output (candles, indicators, signals)
- Dashboard: `dashboard/src/data/mttd_data.json`

---

## 📚 Academic References

1. **Ensemble Methods** - Zhou, Z.-H. (2012). "Ensemble Methods: Foundations and Algorithms"
2. **Walk-Forward Validation** - Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies"
3. **Technical Analysis** - Murphy, J. (1999). "Technical Analysis of the Financial Markets"
4. **Risk Management** - Taleb, N. N. (2007). "The Black Swan"

---

## ⚠️ Disclaimer

This system is for **educational and research purposes only**. Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research before trading.

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🔗 Related Projects

- [Quant Technical Indicator Bank](https://github.com/lutfi-zain/quant-technical-indicator-bank) - Parent repository with 60+ indicators

---

**Built with ❤️ by [lutfi-zain](https://github.com/lutfi-zain)**
