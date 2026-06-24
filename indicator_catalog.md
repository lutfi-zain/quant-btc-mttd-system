# Indicator Bank Catalog

**Generated:** 2026-06-24
**Source:** `/home/ubuntu/projects/quant-technical-indicator-bank/perpetual/`
**Total Indicators:** 33 Python files

---

## Executive Summary

| Family | Count | Signal Types | Best For |
|--------|-------|--------------|----------|
| Smoothing/MA | 10 | vii (trend direction) | Trend filtering, noise reduction |
| Trend Following | 8 | qb, vii (binary) | Entry/exit signals, stop-loss |
| Volatility Band | 5 | band breakouts | Regime detection, mean-reversion |
| Momentum/RSI | 3 | vii (crossover) | Overbought/oversold, divergence |
| Volume-Based | 2 | vii (crossover) | Confirmation, accumulation/distribution |
| Multi-Timeframe | 1 | long_signal/short_signal | Trend confirmation across timeframes |
| Ichimoku-like | 1 | long_c/short_c | Support/resistance levels |
| Regression-Based | 3 | trend (direction) | Polynomial fitting, trend following |

---

## Family 1: Smoothing/Moving Average (10 indicators)

### 1. ALMA Lag | viResearch
- **File:** `alma_lag_viresearch.py`
- **Function:** `alma_lag_viresearch(df, len_subject=78)`
- **Statistical Family:** Gaussian Weighted Smoothing (ALMA)
- **Inputs:** OHLCV DataFrame, length parameter
- **Outputs:** `alma` (smoothed value), `vii` (1=bullish, -1=bearish, 0=neutral)
- **Signal Interface:** Stateful trend via ALMA slope + price position
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** Low-lag trend following

### 2. EWMA | viResearch
- **File:** `ewma_viresearch.py`
- **Function:** `ewma_viresearch(df, length=25)`
- **Statistical Family:** Weighted Smoothing (EMA of WMA)
- **Inputs:** OHLCV DataFrame, length
- **Outputs:** `ewma` (smoothed value), `vii` (1=bullish, -1=bearish)
- **Signal Interface:** Crossover of EWMA with itself (shifted)
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** Trend direction with minimal lag

### 3. DEMA SMA Standard Deviation | viResearch
- **File:** `dema_sma_standard_deviation_viresearch.py`
- **Function:** `dema_sma_standard_deviation_viresearch(df, len_dema=5, len_ma=60, len_sd=20)`
- **Statistical Family:** DEMA + Volatility Band
- **Inputs:** OHLCV DataFrame, DEMA length, SMA length, SD length
- **Outputs:** `hma`, `dema`, `sma`, `upper`, `lower`, `vii`
- **Signal Interface:** Price relative to band position
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** Trend with volatility-adjusted bands

### 4. DSMA | viResearch
- **File:** `dsma_viresearch.py`
- **Function:** `dsma_viresearch(df, len_sma=58, len_dsma=2)`
- **Statistical Family:** Double SMA Smoothing
- **Inputs:** OHLCV DataFrame, SMA lengths
- **Outputs:** `hma`, `dsma`, `vii`
- **Signal Interface:** Price relative to double-smoothed high
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** Ultra-smooth trend following

### 5. Double Source SMA Standard Deviation | viResearch
- **File:** `double_src_sma_standard_deviation_viresearch.py`
- **Function:** `double_src_sma_standard_deviation_viresearch(df, len_ma=60, len_sd=20)`
- **Statistical Family:** Source-smoothed SMA + SD Band
- **Inputs:** OHLCV DataFrame, MA length, SD length
- **Outputs:** `hma`, `sma`, `upper`, `lower`, `vii`
- **Signal Interface:** Price relative to band
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** Smooth trend with deviation bands

### 6. Adaptive Volatility Controlled LSMA | QuantAlgo
- **File:** `adaptive_volatility_controlled_lsma_quantalgo.py`
- **Function:** `adaptive_volatility_controlled_lsma_quantalgo(df, len_lsma=50, atr_len=14, atr_mul=1.5)`
- **Statistical Family:** Linear Regression + ATR Volatility Control
- **Inputs:** OHLCV DataFrame, LSMA length, ATR parameters
- **Outputs:** `lsma`, `upper_volatility_band`, `lower_volatility_band`, `trend_direction`
- **Signal Interface:** Price breakout from ATR-adjusted LSMA bands
- **Binary Signal:** ✅ Yes (`trend_direction`: 1=bull, -1=bear)
- **Best For:** Adaptive trend with volatility control

### 7. LSMA | viResearch
- **File:** `lsma_viresearch.py`
- **Function:** `lsma_viresearch(df, len_lsma=77, off=0)`
- **Statistical Family:** Linear Regression Smoothing
- **Inputs:** OHLCV DataFrame, length, offset
- **Outputs:** `lsma`, `vii`
- **Signal Interface:** LSMA slope + price position
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** Leading trend indicator

### 8. LSMA ATR | viResearch
- **File:** `lsma_atr_viresearch.py`
- **Function:** `lsma_atr_viresearch(df, len_lsma=87, atr_len=14)`
- **Statistical Family:** Linear Regression + ATR Bands
- **Inputs:** OHLCV DataFrame, LSMA length, ATR length
- **Outputs:** `lsma`, `atrl`, `atrs`, `vii`
- **Signal Interface:** Breakout from LSMA ± ATR bands
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** Trend with volatility breakout

### 9. Median Standard Deviation | viResearch
- **File:** `median_standard_deviation_viresearch.py`
- **Function:** `median_standard_deviation_viresearch(df, len_dema=7, median_len=61, atr_len=6, atr_mul=0.6, len_sd=27)`
- **Statistical Family:** DEMA + Median + ATR + SD
- **Inputs:** OHLCV DataFrame, multiple length/multiplier parameters
- **Outputs:** `median`, `upper`, `lower`, `vii`
- **Signal Interface:** Price relative to median ± ATR with SD filter
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** Robust trend with multiple filters

### 10. Median Deviation Suite | InvestorUnknown
- **File:** `median_deviation_suite_investorunknown.py`
- **Function:** `median_deviation_suite_investorunknown(df, median_len=28, dev_len=21, dev_mul=1.0)`
- **Statistical Family:** Multi-Metric Median Deviation (AAD, MAD, STDEV, ATR)
- **Inputs:** OHLCV DataFrame, median length, deviation length, multiplier
- **Outputs:** `median`, `upper`, `lower`, `upper2`, `lower2`, `trend`, `sig`
- **Signal Interface:** Composite score from 5 deviation metrics
- **Binary Signal:** ✅ Yes (`sig`: 1=bull, -1=bear)
- **Best For:** Multi-metric regime detection

---

## Family 2: Trend Following (8 indicators)

### 11. DEMA Supertrend | viResearch
- **File:** `dema_supertrend_viresearch.py`
- **Function:** `dema_supertrend_viresearch(df, subject=2, multiplier=3.35, dema_length=9)`
- **Statistical Family:** Supertrend with DEMA smoothing
- **Inputs:** OHLCV DataFrame, ATR length, multiplier, DEMA length
- **Outputs:** `dema`, `st`, `d`, `vii`
- **Signal Interface:** Supertrend direction flip (d crosses 0)
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** Classic trend following with reduced noise

### 12. Median Supertrend | viResearch
- **File:** `median_supertrend_viresearch.py`
- **Function:** `median_supertrend_viresearch(df, subject=10, multiplier=2.15, median_length=14)`
- **Statistical Family:** Supertrend with Median smoothing
- **Inputs:** OHLCV DataFrame, ATR length, multiplier, median length
- **Outputs:** `smooth`, `st`, `d`, `vii`
- **Signal Interface:** Supertrend direction flip
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** Robust Supertrend with outlier rejection

### 13. DEMA VStop | viResearch
- **File:** `dema_vstop_viresearch.py`
- **Function:** `dema_vstop_viresearch(df, dema_length=30, vstop_length=10, multiplier=2.0)`
- **Statistical Family:** Volatility Stop (VStop) with DEMA
- **Inputs:** OHLCV DataFrame, DEMA length, VStop length, multiplier
- **Outputs:** `dema`, `vStop`, `uptrend`, `vii`
- **Signal Interface:** Uptrend state change
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** Trailing stop-loss, trend following

### 14. vii`Stop
- **File:** `vii_stop.py`
- **Function:** `vii_stop(df, length=12, multiplier=2.8, hma_length=45)`
- **Statistical Family:** Volatility Stop (VStop) with HMA
- **Inputs:** OHLCV DataFrame, ATR length, multiplier, HMA length
- **Outputs:** `vStop`, `uptrend`, `vii`, `hma`
- **Signal Interface:** Uptrend state change
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** Trailing stop-loss with HMA confirmation

### 15. Linear % ST | QuantEdgeB
- **File:** `linear_st_quantedgeb.py`
- **Function:** `linear_st_quantedgeb(df, len_perc=35, len_lsma=24, len_atr=14, mult_up=0.8, mult_dn=1.9)`
- **Statistical Family:** Linear Regression Supertrend
- **Inputs:** OHLCV DataFrame, percentile length, LSMA length, ATR params
- **Outputs:** `median`, `base`, `upper_atr`, `lower_atr`, `st_up`, `st_dn`, `qb`
- **Signal Interface:** Supertrend with asymmetric ATR multipliers
- **Binary Signal:** ✅ Yes (`qb`: 1=bull, -1=bear)
- **Best For:** Trend following with asymmetric risk/reward

### 16. Quantile DEMA Trend | QuantEdgeB
- **File:** `quantile_dema_trend_quantedgeb.py`
- **Function:** `quantile_dema_trend_quantedgeb(df, len_dema=30, len_perc=10, len_atr=14, mult_up=1.2, mult_dn=1.2)`
- **Statistical Family:** DEMA + Percentile Bands Supertrend
- **Inputs:** OHLCV DataFrame, DEMA length, percentile length, ATR params
- **Outputs:** `filtered_src`, `dema_base`, `dema`, `sd_filter`, `super_trend`, `st_direction`, `qb`
- **Signal Interface:** Supertrend direction flip
- **Binary Signal:** ✅ Yes (`qb`)
- **Best For:** Trend with percentile-based volatility

### 17. DEGA RMA | QuantEdgeB
- **File:** `dega_rma_quantedgeb.py`
- **Function:** `dega_rma_quantedgeb(df, len_dema=30, len_fg=4, sigma_fg=2.0, len_rma=12, len_atr=40)`
- **Statistical Family:** DEMA + Gaussian + RMA + Custom ATR
- **Inputs:** OHLCV DataFrame, DEMA/Gaussian/RMA lengths, ATR params
- **Outputs:** `dema`, `gaussian`, `rma`, `long_r`, `short_r`, `qb`
- **Signal Interface:** Price breakout from Gaussian-filtered RMA bands
- **Binary Signal:** ✅ Yes (`qb`)
- **Best For:** Multi-smoothed trend with adaptive bands

### 18. Gaussian Smooth Trend | QuantEdgeB
- **File:** `gaussian_smooth_trend_quantedgeb.py`
- **Function:** `gaussian_smooth_trend_quantedgeb(df, len_dema=7, len_fg=4, sigma_fg=2.0, len_s=12, len_sd=30)`
- **Statistical Family:** DEMA + Gaussian + SMMA + SD Bands
- **Inputs:** OHLCV DataFrame, DEMA/Gaussian/SMMA lengths, SD params
- **Outputs:** `dema`, `filter_gaussian`, `smma`, `long_v`, `short_v`, `qb`
- **Signal Interface:** Price breakout from Gaussian-smoothed bands
- **Binary Signal:** ✅ Yes (`qb`)
- **Best For:** Ultra-smooth trend with adaptive volatility

---

## Family 3: Volatility Band (5 indicators)

### 19. DEMA Adjusted ATR
- **File:** `dema_adjusted_average_true_range.py`
- **Function:** `dema_adjusted_average_true_range(df, period_dema=7, period_atr=14, factor_atr=1.7)`
- **Statistical Family:** DEMA + ATR Channel
- **Inputs:** OHLCV DataFrame, DEMA length, ATR length, ATR factor
- **Outputs:** `dema_atr`, `moving_average`
- **Signal Interface:** DEMA-ATR channel with optional MA filter
- **Binary Signal:** ❌ No (continuous channel values)
- **Best For:** Dynamic support/resistance levels

### 20. DEMA RSI Overlay
- **File:** `dema_rsi_overlay.py`
- **Function:** `dema_rsi_overlay(df, sublen=30, len_rsi=14, long_threshold=70.0, short_threshold=55.0)`
- **Statistical Family:** DEMA + RSI + Volatility Bands
- **Inputs:** OHLCV DataFrame, DEMA length, RSI length, thresholds
- **Outputs:** `dema`, `u1/d1`, `u2/d2`, `u3/d3`, `back_quant`
- **Signal Interface:** RSI threshold with band filter
- **Binary Signal:** ✅ Yes (`back_quant`: 1=bull, -1=bear)
- **Best For:** RSI with volatility-adjusted thresholds

### 21. Inverted SD DEMA RSI | viResearch
- **File:** `inverted_sd_dema_rsi_viresearch.py`
- **Function:** `inverted_sd_dema_rsi_viresearch(df, dema_length=30, sd_length=30, rsi_length=14, threshold_l=70, threshold_s=55)`
- **Statistical Family:** DEMA + RSI + Inverted SD Filter
- **Inputs:** OHLCV DataFrame, DEMA/SD/RSI lengths, thresholds
- **Outputs:** `dema`, `rsi`, `stdev`, `u`, `d`, `vii`
- **Signal Interface:** RSI with inverted SD band filter
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** RSI with adaptive filtering

### 22. Polynomial Deviation Bands
- **File:** `polynomial_deviation_bands.py`
- **Function:** `polynomial_deviation_bands(df, deg="2nd", regressions_length=14, dev_type="Standard Deviation", dev_lookback=20, multiplier=1.5)`
- **Statistical Family:** Polynomial Regression + Multiple Deviation Types
- **Inputs:** OHLCV DataFrame, polynomial degree, regression length, deviation type, multiplier
- **Outputs:** `reg_val`, `upper_band`, `lower_band`, `trend`
- **Signal Interface:** Price breakout from polynomial regression bands
- **Binary Signal:** ✅ Yes (`trend`: 1=bull, -1=bear)
- **Best For:** Non-linear trend following with flexible deviation

### 23. HILO Interpolation | QuantEdgeB
- **File:** `hilo_interpolation_quantedgeb.py`
- **Function:** `hilo_interpolation_quantedgeb(df, prcl_len=35, prcl_period=4, prcl_high=75, prcl_low=50)`
- **Statistical Family:** Percentile Interpolation (HILO)
- **Inputs:** OHLCV DataFrame, percentile length, period, high/low thresholds
- **Outputs:** `prc_up`, `prc_dn`, `final_prcl`, `qb`
- **Signal Interface:** Price relative to interpolated percentile level
- **Binary Signal:** ✅ Yes (`qb`: 1=bull, -1=bear)
- **Best For:** Adaptive support/resistance with interpolation

---

## Family 4: Momentum/RSI (3 indicators)

### 24. DEMA EMA Crossover | viResearch
- **File:** `dema_ema_crossover_viresearch.py`
- **Function:** `dema_ema_crossover_viresearch(df, len_dema=15, len_fast_ema=12, len_slow_ema=26)`
- **Statistical Family:** DEMA + Dual EMA Crossover
- **Inputs:** OHLCV DataFrame, DEMA length, fast/slow EMA lengths
- **Outputs:** `dema`, `ema1st`, `ema2nd`, `vii`
- **Signal Interface:** Fast/slow EMA crossover of DEMA
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** Momentum crossover signals

### 25. DEMA DMI | viResearch
- **File:** `dema_dmi_viresearch.py`
- **Function:** `dema_dmi_viresearch(df, len_dema=15, adx_smoothing_len=18, di_len=18)`
- **Statistical Family:** DEMA + Directional Movement Index
- **Inputs:** OHLCV DataFrame, DEMA length, ADX/DI lengths
- **Outputs:** `plus`, `minus`, `adx`, `diff`, `vii`
- **Signal Interface:** ADX rising + DI crossover
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** Trend strength + direction

### 26. IRS Elder Force Volume Index
- **File:** `irs_elder_force_volume_index.py`
- **Function:** `irs_elder_force_volume_index(df, length=40)`
- **Statistical Family:** Elder Force Index (Volume × Price Change)
- **Inputs:** OHLCV DataFrame, length
- **Outputs:** `hma`, `efi`, `vii`
- **Signal Interface:** Zero-line crossover of Elder Force Index
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** Volume-confirmed momentum

---

## Family 5: Volume-Based (2 indicators)

### 27. Volume Trend Swing Points | viResearch
- **File:** `volume_trend_swing_points_viresearch.py`
- **Function:** `volume_trend_swing_points_viresearch(df, x=30, y=30)`
- **Statistical Family:** Price-Volume Trend (PVT) Swing Points
- **Inputs:** OHLCV DataFrame, lookback lengths
- **Outputs:** `pvt`, `h`, `l`, `vii`
- **Signal Interface:** PVT swing high/low detection
- **Binary Signal:** ✅ Yes (`vii`)
- **Best For:** Volume-confirmed swing points

### 28. P-Motion Trend | QuantEdgeB
- **File:** `p_motion_trend_quantedgeb.py`
- **Function:** `p_motion_trend_quantedgeb(df, ema_len=21, sd_length=30, mult_sdup=1.5, mult_sddn=1.5, dema_len=7, prc_len=2)`
- **Statistical Family:** DEMA + Median + EMA + SD Envelope
- **Inputs:** OHLCV DataFrame, EMA/SD/DEMA lengths, multipliers
- **Outputs:** `dema`, `prc_smooth`, `filter_sd`, `ema`, `long_e`, `short_e`, `qb`, `pl`
- **Signal Interface:** Price breakout from EMA ± SD envelope
- **Binary Signal:** ✅ Yes (`qb`)
- **Best For:** Trend with volume-confirmed motion

---

## Family 6: Multi-Timeframe (1 indicator)

### 29. TS Aggregated MAD | TobySimard
- **File:** `ts_aggregated_median_absolute_deviation_tobbysimard.py`
- **Function:** `ts_aggregated_median_absolute_deviation_tobbysimard(df, mad_time1="240", mad_time2="720", mad_time3="1D", ...)`
- **Statistical Family:** Multi-Timeframe Median Absolute Deviation
- **Inputs:** OHLCV DataFrame, 3 timeframes with MAD parameters, overbought/oversold thresholds
- **Outputs:** `percentage_deviation`, `normalized_upper_band`, `normalized_lower_band`, `aggregated_med`, `aggregated_upper_band`, `aggregated_short_band`, `aggregated_lower_band`, `long_signal`, `short_signal`
- **Signal Interface:** Aggregated MAD band breakout across timeframes
- **Binary Signal:** ✅ Yes (`long_signal`, `short_signal`)
- **Best For:** Multi-timeframe trend confirmation

---

## Family 7: Ichimoku-like (1 indicator)

### 30. Enhanced Kijun Sen Base
- **File:** `enhanced_kijun_sen_base.py`
- **Function:** `enhanced_kijun_sen_base(df, smf="NONE", per=14, mult=0.55, ma_length=50, percentage=0.28, kijun_sen_base_period=26)`
- **Statistical Family:** Ichimoku Kijun Sen + Multiple Filter Options
- **Inputs:** OHLCV DataFrame, filter type (ATR/SD/MAD/%OR/WMA), period, multiplier, Kijun period
- **Outputs:** `kijun`, `filter`, `upper_band`, `lower_band`, `long_c`, `short_c`
- **Signal Interface:** Price breakout from Kijun ± filter band
- **Binary Signal:** ✅ Yes (`long_c`, `short_c`)
- **Best For:** Ichimoku-style support/resistance with configurable filters

---

## Family 8: Regime Detection (3 indicators)

### 31. Adaptive Regime Cloud
- **File:** `adaptive_regime_cloud.py`
- **Function:** `adaptive_regime_cloud(df, lookback=50, adaptive_period=30, volatility_period=10, cloud_expansion=1.6, regime_threshold=0.65)`
- **Statistical Family:** Hurst Exponent Regime Detection + Adaptive EMA
- **Inputs:** OHLCV DataFrame, lookback periods, thresholds
- **Outputs:** `midline`, `upper_band`, `lower_band`, `hurst`, `volatility`, `adaptive_alpha`, `in_long_position`, `in_short_position`, `long_signal`, `short_signal`
- **Signal Interface:** Adaptive cloud breakout based on regime (trending/mean-reverting)
- **Binary Signal:** ✅ Yes (`long_signal`, `short_signal`)
- **Best For:** Regime-adaptive trend following

### 32. MAD Trend | InvestorUnknown
- **File:** `madtrend_investorunknown.py`
- **Function:** `madtrend_investorunknown(df, length=28, mad_mult=1.0, input_src="close")`
- **Statistical Family:** Median Absolute Deviation (MAD) Trend
- **Inputs:** OHLCV DataFrame, length, multiplier, source
- **Outputs:** `candle_center`, `median`, `med_p`, `med_m`, `dir`
- **Signal Interface:** Source crossover with MAD bands
- **Binary Signal:** ✅ Yes (`dir`: 1=bull, -1=bear)
- **Best For:** Robust trend detection with outlier resistance

### 33. Root Mean Square Deviation Trend
- **File:** `root_mean_square_deviation_trend.py`
- **Function:** `root_mean_square_deviation_trend(df, input_src="close", avg_type="SMA", length=28, mult=1.0)`
- **Statistical Family:** RMSD (Root Mean Square Deviation) Trend
- **Inputs:** OHLCV DataFrame, source, average type (SMA/EMA/HMA/DEMA/TEMA/RMA/FRAMA), length, multiplier
- **Outputs:** `direction`, `avg`, `avg_p`, `avg_m`, `direction_plot`, `avg_plot`, `avg_p_plot`, `avg_m_plot`, `candle_h_l`
- **Signal Interface:** Source crossover with RMSD bands
- **Binary Signal:** ✅ Yes (`direction`: 1=bull, -1=bear)
- **Best For:** Flexible trend detection with multiple smoothing options

---

## Signal Interface Summary

### Standard Signal Conventions

| Output Name | Meaning | Values |
|-------------|---------|--------|
| `vii` | Vertical Info Index (trend direction) | 1=bullish, -1=bearish, 0=neutral |
| `qb` | Quantitative Bias (binary signal) | 1=bullish, -1=bearish |
| `dir` / `direction` | Trend direction | 1=bullish, -1=bearish, 0=neutral |
| `trend` | Composite trend score | Positive=bullish, Negative=bearish |
| `long_signal` / `short_signal` | Event-based signals | True/False |
| `long_c` / `short_c` | Crossover signals | True/False |
| `back_quant` / `sig` | Composite signal | 1=bullish, -1=bearish |
| `in_long_position` / `in_short_position` | Position state | True/False |

### Input DataFrame Requirements

All indicators expect a pandas DataFrame with:
- **Required columns:** `open`, `high`, `low`, `close`, `volume`
- **Index:** DatetimeIndex (recommended) or integer index
- **Data type:** float64

### Helper Functions Available

All indicators import from `indicators_helper.py`:
- Moving Averages: `sma`, `ema`, `wma`, `hma`, `dema`, `rma`, `alma`, `frama`, `tema`, `t3`
- Volatility: `atr`, `tr`, `stdev`
- Extremes: `highest`, `lowest`
- Crossovers: `crossover`, `crossunder`
- Other: `rsi`, `linreg`, `pivotlow`, `pivothigh`, `valuewhen`, `barssince`, `vwma`

---

## Integration Recommendations

### For MTTD System Enhancement

Based on the AGENTS.md insights about asking complementary questions:

1. **Trend Direction (Direction Question):**
   - Use: `dema_supertrend_viresearch`, `median_supertrend_viresearch`, `linear_st_quantedgeb`
   - These provide clear binary trend signals

2. **Regime Detection (Timing Question):**
   - Use: `adaptive_regime_cloud` (Hurst-based), `madtrend_investorunknown`
   - These detect trending vs mean-reverting markets

3. **Volatility Filter (Risk Question):**
   - Use: `polynomial_deviation_bands`, `median_deviation_suite_investorunknown`
   - These provide adaptive volatility bands

4. **Volume Confirmation (Confirmation Question):**
   - Use: `irs_elder_force_volume_index`, `volume_trend_swing_points_viresearch`
   - These confirm moves with volume

### Filtering by Complexity

**Simple (Few Parameters):**
- `ewma_viresearch`, `lsma_viresearch`, `vii_stop`, `dsma_viresearch`

**Medium (3-5 Parameters):**
- `dema_supertrend_viresearch`, `median_supertrend_viresearch`, `dema_dmi_viresearch`

**Complex (6+ Parameters):**
- `adaptive_regime_cloud`, `ts_aggregated_median_absolute_deviation_tobbysimard`, `polynomial_deviation_bands`

---

## Verification Checklist

- [x] All 33 Python files from perpetual/ are cataloged
- [x] 8 statistical families identified (Smoothing, Trend Following, Volatility Band, Momentum, Volume, Multi-Timeframe, Ichimoku-like, Regime Detection)
- [x] Input/output signatures documented for each indicator
- [x] Binary signal interfaces identified (30 out of 33 have binary signals)
- [x] Signal conventions standardized (vii, qb, dir, trend, etc.)
