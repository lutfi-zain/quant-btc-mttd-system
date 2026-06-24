# Pi Long Task TODO

## Global Context

- **Project:** `/home/ubuntu/projects/quant-btc-mttd-system/`
- **Target:** 25-35 trades, Sharpe > 1.35
- **Filtering Framework:** MSVR v8 (msvr_v8.py) — uses 10 statistical families
- **Current Baseline:** MSVR base → 26 trades, 1.18 Sharpe, 45% CAGR
- **Filter Families to Apply:** SuperSmoother (F2), LinearReg (F3), Cycle Phase (F4), Efficiency Ratio (F5), Volatility Cluster (F6), Shannon Entropy (F7), Volume Confirm (F8), HMM Regime (F9)
- **Trade Constraints:** min_hold=25, max_hold=90, gates=3

## Progress

- [ ] TODO 1 — Create Ichimoku Base Signal Test
- [ ] TODO 2 — Create Bollinger Breakout Base Signal Test
- [ ] TODO 3 — Create ADX Trend Base Signal Test
- [ ] TODO 4 — Create Supertrend Base Signal Test
- [ ] TODO 5 — Create Keltner Channel Base Signal Test
- [ ] TODO 6 — Create Comparison Script and Generate Chart

---

## TODO 1 — Create Ichimoku Base Signal Test

**Goal:** Create `test_base_ichimoku.py` that generates buy/sell signals using Ichimoku indicator, applies MSVR v8 filtering framework, and outputs performance metrics.

**Status:**
- [ ] Script created at `test_base_ichimoku.py`
- [ ] Signal logic: Buy when Tenkan > Kijun AND price > Cloud
- [ ] Signal logic: Sell when Tenkan < Kijun AND price < Cloud
- [ ] MSVR v8 filtering framework integrated (Families 2-9)
- [ ] Trade constraints applied (min_hold=25, max_hold=90, gates=3)
- [ ] Results printed: trades, win rate, Sharpe, CAGR, avg hold
- [ ] Achieves 25-35 trade range
- [ ] Achieves Sharpe > 1.35

**Verify:**
- Run `python3 test_base_ichimoku.py` and confirm it completes without errors
- Verify output shows all required metrics (trades, win rate, Sharpe, CAGR, avg hold)
- Check trade count is in 25-35 range
- Check Sharpe ratio is > 1.35
- Inspect signal logic matches Ichimoku rules in `ichimoku_quant.py`

**Done when:**
- Script runs successfully and produces complete metrics output
- Trade count is between 25-35
- Sharpe ratio exceeds 1.35

---

## TODO 2 — Create Bollinger Breakout Base Signal Test

**Goal:** Create `test_base_bollinger.py` that generates buy/sell signals using Bollinger Band breakout logic, applies MSVR v8 filtering framework, and outputs performance metrics.

**Status:**
- [ ] Script created at `test_base_bollinger.py`
- [ ] Signal logic: Buy when price breaks above upper band (25-period, 2.0 std)
- [ ] Signal logic: Sell when price breaks below lower band
- [ ] MSVR v8 filtering framework integrated (Families 2-9)
- [ ] Trade constraints applied (min_hold=25, max_hold=90, gates=3)
- [ ] Results printed: trades, win rate, Sharpe, CAGR, avg hold
- [ ] Achieves 25-35 trade range
- [ ] Achieves Sharpe > 1.35

**Verify:**
- Run `python3 test_base_bollinger.py` and confirm it completes without errors
- Verify output shows all required metrics
- Check trade count is in 25-35 range
- Check Sharpe ratio is > 1.35
- Confirm Bollinger parameters: 25-period, 2.0 standard deviations

**Done when:**
- Script runs successfully and produces complete metrics output
- Trade count is between 25-35
- Sharpe ratio exceeds 1.35

---

## TODO 3 — Create ADX Trend Base Signal Test

**Goal:** Create `test_base_adx.py` that generates buy/sell signals using ADX trend strength with directional indicators, applies MSVR v8 filtering framework, and outputs performance metrics.

**Status:**
- [ ] Script created at `test_base_adx.py`
- [ ] Signal logic: Buy when ADX > 25 AND +DI > -DI
- [ ] Signal logic: Sell when ADX > 25 AND -DI > +DI
- [ ] MSVR v8 filtering framework integrated (Families 2-9)
- [ ] Trade constraints applied (min_hold=25, max_hold=90, gates=3)
- [ ] Results printed: trades, win rate, Sharpe, CAGR, avg hold
- [ ] Achieves 25-35 trade range
- [ ] Achieves Sharpe > 1.35

**Verify:**
- Run `python3 test_base_adx.py` and confirm it completes without errors
- Verify output shows all required metrics
- Check trade count is in 25-35 range
- Check Sharpe ratio is > 1.35
- Confirm ADX threshold is 25 as specified

**Done when:**
- Script runs successfully and produces complete metrics output
- Trade count is between 25-35
- Sharpe ratio exceeds 1.35

---

## TODO 4 — Create Supertrend Base Signal Test

**Goal:** Create `test_base_supertrend.py` that generates buy/sell signals using Supertrend indicator from indicator bank, applies MSVR v8 filtering framework, and outputs performance metrics.

**Status:**
- [ ] Script created at `test_base_supertrend.py`
- [ ] Supertrend loaded from indicator bank: `perpetual/median_supertrend_viresearch.py`
- [ ] Signal logic: Buy when price > Supertrend
- [ ] Signal logic: Sell when price < Supertrend
- [ ] MSVR v8 filtering framework integrated (Families 2-9)
- [ ] Trade constraints applied (min_hold=25, max_hold=90, gates=3)
- [ ] Results printed: trades, win rate, Sharpe, CAGR, avg hold
- [ ] Achieves 25-35 trade range
- [ ] Achieves Sharpe > 1.35

**Verify:**
- Run `python3 test_base_supertrend.py` and confirm it completes without errors
- Verify output shows all required metrics
- Check trade count is in 25-35 range
- Check Sharpe ratio is > 1.35
- Confirm Supertrend is loaded from correct indicator bank path

**Done when:**
- Script runs successfully and produces complete metrics output
- Trade count is between 25-35
- Sharpe ratio exceeds 1.35

---

## TODO 5 — Create Keltner Channel Base Signal Test

**Goal:** Create `test_base_keltner.py` that generates buy/sell signals using Keltner Channel breakout logic, applies MSVR v8 filtering framework, and outputs performance metrics.

**Status:**
- [ ] Script created at `test_base_keltner.py`
- [ ] Signal logic: Buy when price breaks above upper band (20 EMA, 1.5x ATR)
- [ ] Signal logic: Sell when price breaks below lower band
- [ ] MSVR v8 filtering framework integrated (Families 2-9)
- [ ] Trade constraints applied (min_hold=25, max_hold=90, gates=3)
- [ ] Results printed: trades, win rate, Sharpe, CAGR, avg hold
- [ ] Achieves 25-35 trade range
- [ ] Achieves Sharpe > 1.35

**Verify:**
- Run `python3 test_base_keltner.py` and confirm it completes without errors
- Verify output shows all required metrics
- Check trade count is in 25-35 range
- Check Sharpe ratio is > 1.35
- Confirm Keltner parameters: 20-period EMA, 1.5x ATR multiplier

**Done when:**
- Script runs successfully and produces complete metrics output
- Trade count is between 25-35
- Sharpe ratio exceeds 1.35

---

## TODO 6 — Create Comparison Script and Generate Chart

**Goal:** Create `base_signal_comparison.py` that loads all 5 base signal test results plus MSVR baseline, generates a comparison table, identifies the best signal, and saves visualization to `mttd/base_signal_comparison.png`.

**Status:**
- [ ] Script created at `base_signal_comparison.py`
- [ ] Loads results from all test_base_*.py outputs
- [ ] Includes MSVR baseline in comparison
- [ ] Prints formatted comparison table (Base Signal, Trades, WinRate, Sharpe, CAGR, AvgHold)
- [ ] Identifies BEST base signal by Sharpe > 1.35 and trade count 25-35
- [ ] Generates and saves chart to `mttd/base_signal_comparison.png`
- [ ] Chart is clear and readable

**Verify:**
- Run `python3 base_signal_comparison.py` and confirm it completes without errors
- Verify comparison table output matches expected format
- Verify `mttd/base_signal_comparison.png` is created
- Open chart and confirm it displays all signals with metrics
- Confirm BEST signal is clearly identified

**Done when:**
- Script runs successfully
- Comparison table displays all 6 signals (MSVR + 5 new)
- Chart saved to `mttd/base_signal_comparison.png`
- Best signal identified based on Sharpe > 1.35 criteria
