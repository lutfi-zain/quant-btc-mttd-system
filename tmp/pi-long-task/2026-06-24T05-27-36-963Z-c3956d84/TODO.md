# Pi Long Task TODO

## Global Constraints
- Each indicator MUST return a `direction` column (-1 or 1)
- Use existing indicators from `indicators/` package as reference
- Transaction costs: 0.1% round-trip
- Min hold: 45 days
- Training period: 2018-2026 (full data for Phase 2)
- Output: Sharpe >1.35, Win Rate >65%, Trades <15

## Progress

- [x] TODO 1 — Linear Regression Trend Indicator
- [x] TODO 2 — GARCH-like Volatility Indicator
- [x] TODO 3 — Volume Confirmation Indicator
- [x] TODO 4 — HMM Regime Detection Indicator
- [ ] TODO 5 — Build MSVR v3 Composite Engine
- [ ] TODO 6 — Test and Compare MSVR v3
- [ ] TODO 7 — Generate Comparison Chart

---

## TODO 1 — Linear Regression Trend Indicator

**Goal:** Create `indicators/linear_reg_trend.py` implementing Family 3 (Regression) indicator that fits a line to prices and uses channel deviation as signal.

**Status:**
- [x] File created at `indicators/linear_reg_trend.py`
- [x] Function `linear_reg_trend(df, length=50)` implemented
- [x] Returns DataFrame with `direction` column (-1 or 1)
- [x] Price > upper band = overbought (direction=-1), price < lower band = oversold (direction=1)
- [x] Works with existing DataFrame format from `mttd_system.py`

**Verify:**
- Import in Python REPL: `from indicators.linear_reg_trend import linear_reg_trend`
- Pass BTC daily DataFrame, check `direction` column exists and contains only -1 or 1
- Run on test data: `linear_reg_trend(btc_df)` returns valid output
- Visual sanity: direction flips align with trend changes

**Done when:** Indicator file exists, function is importable, returns correct `direction` column, and produces sensible signals on BTC daily data.

---

## TODO 2 — GARCH-like Volatility Indicator

**Goal:** Create `indicators/volatility_cluster.py` implementing Family 6 (GARCH) indicator that detects high/low volatility regimes using rolling std as GARCH proxy.

**Status:**
- [x] File created at `indicators/volatility_cluster.py`
- [x] Function `volatility_cluster(df, window=20, threshold=1.2)` implemented
- [x] Returns DataFrame with `direction` column (-1 or 1)
- [x] High volatility → direction=0 or -1 (avoid trading)
- [x] Low volatility → direction=1 (trade)
- [x] Compares rolling volatility to median

**Verify:**
- Import: `from indicators.volatility_cluster import volatility_cluster`
- Pass BTC daily DataFrame, check `direction` column exists
- Verify: during known high-vol periods (e.g., March 2020 crash), direction signals caution
- Verify: during calm periods, direction allows trading

**Done when:** Indicator file exists, function is importable, correctly identifies volatility regimes, and returns valid `direction` column.

---

## TODO 3 — Volume Confirmation Indicator

**Goal:** Create `indicators/volume_confirm.py` implementing Family 8 (Volume) indicator using OBV, volume spikes, and Force Index.

**Status:**
- [x] File created at `indicators/volume_confirm.py`
- [x] Function `volume_confirm(df, obv_length=20, spike_mult=1.5)` implemented
- [x] Returns DataFrame with `direction` column (-1 or 1)
- [x] OBV trend calculated (accumulation/distribution)
- [x] Volume spike detection implemented
- [x] Direction aligns with volume confirmation

**Verify:**
- Import: `from indicators.volume_confirm import volume_confirm`
- Pass BTC daily DataFrame (must have volume column)
- Check `direction` column exists with valid values
- Verify: volume spikes during breakouts produce confirming signals

**Done when:** Indicator file exists, function is importable, uses volume data correctly, and returns valid `direction` column.

---

## TODO 4 — HMM Regime Detection Indicator

**Goal:** Create `indicators/hmm_regime.py` implementing Family 9 (Bayesian) indicator that detects bull/bear/sideways regimes using simple HMM with 3 states.

**Status:**
- [x] File created at `indicators/hmm_regime.py`
- [x] Function `hmm_regime(df, n_states=3, window=100)` implemented
- [x] Returns DataFrame with `direction` column (-1 or 1)
- [x] Uses returns + volatility as features
- [x] Trending regime → direction=1 (trade)
- [x] Sideways/bear regime → direction=-1 (avoid)
- [x] No external ML library dependencies (use scipy/numpy only)

**Verify:**
- Import: `from indicators.hmm_regime import hmm_regime`
- Pass BTC daily DataFrame
- Check `direction` column exists
- Verify: during 2020-2021 bull market, regime detects "trending"
- Verify: during sideways chop, regime detects "sideways"

**Done when:** Indicator file exists, function is importable, classifies market regimes correctly, and returns valid `direction` column.

---

## TODO 5 — Build MSVR v3 Composite Engine

**Goal:** Create `msvr_v3.py` combining all 9 statistical families into a composite signal engine targeting Sharpe >1.35.

**Status:**
- [ ] File created at `msvr_v3.py`
- [ ] Layer 1: MSVR Base (Family 1) integrated
- [ ] Layer 2: SuperSmoother (Family 2) integrated
- [ ] Layer 3: LinearReg (Family 3) integrated
- [ ] Layer 4: Cycle Phase (Family 4) integrated
- [ ] Layer 5: Efficiency Ratio gate (Family 5) integrated
- [ ] Layer 6: GARCH Volatility gate (Family 6) integrated
- [ ] Layer 7: Shannon Entropy gate (Family 7) integrated
- [ ] Layer 8: Volume Confirm (Family 8) integrated
- [ ] Layer 9: HMM Regime gate (Family 9) integrated
- [ ] Composite signal = product of all layers
- [ ] Backtest function implemented
- [ ] Transaction costs: 0.1% round-trip
- [ ] Min hold: 45 days

**Verify:**
- Run: `python3 msvr_v3.py`
- Output shows: trades, win rate, Sharpe ratio
- Trades < 15, win rate > 60%
- Sharpe > 1.35

**Done when:** MSVR v3 runs end-to-end, produces backtest results meeting target metrics, and all 9 layers are visible in the composite signal.

---

## TODO 6 — Test and Compare MSVR v3

**Goal:** Compare MSVR v3 against MSVR Hybrid and Ichimoku baselines to validate improvement.

**Status:**
- [ ] MSVR v3 results documented (trades, win rate, Sharpe)
- [ ] MSVR Hybrid baseline: 15 trades, 66.7% win, Sharpe 1.09
- [ ] Ichimoku baseline: 11 trades, 63.6% win, Sharpe 1.31
- [ ] Comparison table created
- [ ] Time-in-market % calculated for MSVR v3
- [ ] Sharpe improvement confirmed (>1.35)

**Verify:**
- MSVR v3 Sharpe > Ichimoku Sharpe (1.31)
- MSVR v3 time-in-market > MSVR Hybrid (30.7%)
- Win rate maintained > 60%
- No degradation vs baselines

**Done when:** Comparison table exists showing MSVR v3 outperforms or matches baselines on Sharpe, win rate, and time-in-market.

---

## TODO 7 — Generate Comparison Chart

**Goal:** Create `mttd/msvr_v3_comparison.png` showing equity curves for MSVR v3, MSVR Hybrid, and Ichimoku with trade markers.

**Status:**
- [ ] Chart file created at `mttd/msvr_v3_comparison.png`
- [ ] MSVR v3 equity curve plotted
- [ ] MSVR Hybrid equity curve plotted
- [ ] Ichimoku equity curve plotted
- [ ] Trade markers (entries/exits) shown
- [ ] Legend with strategy names
- [ ] Title and axis labels

**Verify:**
- Open `mttd/msvr_v3_comparison.png`
- All three equity curves visible
- Trade markers present
- MSVR v3 curve ends higher than baselines (risk-adjusted)
- Image is clear and readable

**Done when:** PNG file exists, contains all three equity curves with markers, and visually demonstrates MSVR v3 improvement.
