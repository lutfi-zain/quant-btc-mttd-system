# Pi Long Task TODO

## Global Context

- MSVR base signal: `/home/ubuntu/projects/quant-btc-mttd-system/msvr_enhanced.py`
- Ichimoku: `/home/ubuntu/projects/quant-btc-mttd-system/ichimoku_quant.py`
- Indicator bank: `/home/ubuntu/projects/quant-technical-indicator-bank/perpetual/`
- Data: `/home/ubuntu/projects/quant-btc-mttd-system/data/btc_daily.json`

## Constraints

- All indicators must have `direction` column for consistency
- Transaction costs: 0.1% round-trip
- Training period: 2018-2026 (full data for Phase 1)
- Cycle Phase: FFT lookback=40, min_period=5, max_period=20
- Target: < 20 trades, > 60% win rate, Sharpe > 1.35

## Progress

- [x] TODO 1 — Create indicators/ directory and indicator files
- [x] TODO 2 — Create msvr_hybrid.py combining MSVR with Ichimoku principles
- [x] TODO 3 — Test hybrid performance and compare with baselines
- [x] TODO 4 — Generate comparison equity curve chart

---

## TODO 1 — Create indicators/ directory and indicator files

**Goal:** Create a new `indicators/` package with three custom indicator modules: Ehler's SuperSmoother, Shannon Entropy, and Efficiency Ratio.

**Status:**
- [x] Create `indicators/` directory
- [x] Create `indicators/__init__.py` (empty)
- [x] Implement `indicators/ehler_supersmoother.py` — 2-pole Ehler SuperSmoother (Family 2: Filtering), params: length (default 7), must output `direction` column
- [x] Implement `indicators/shannon_entropy.py` — Shannon Entropy (Family 7: Entropy), params: window (default 15), bins (default 6), threshold: entropy < 2.5 = tradeable, must output `direction` column
- [x] Implement `indicators/efficiency_ratio.py` — Efficiency Ratio (Family 5: Fractal), params: period (default 14), ER > 0.25 = trending, must output `direction` column

**Verify:**
- All three files exist in `indicators/` directory
- Each module has a callable function that accepts a DataFrame and returns a DataFrame with `direction` column
- Import test: `from indicators.ehler_supersmoother import ehler_supersmoother` works without errors
- Run each indicator on sample data from `data/btc_daily.json` and confirm output shape matches input

**Done when:** `import indicators; from indicators.ehler_supersmoother import ehler_supersmoother; from indicators.shannon_entropy import shannon_entropy; from indicators.efficiency_ratio import efficiency_ratio` all succeed, and each function returns a DataFrame with a `direction` column.

---

## TODO 2 — Create msvr_hybrid.py combining MSVR with Ichimoku principles

**Goal:** Build `msvr_hybrid.py` that combines MSVR direction signal with Ichimoku's proven filtering principles using cycle phase timing, SuperSmoother, Shannon Entropy gate, and Efficiency Ratio gate.

**Status:**
- [x] Create `msvr_hybrid.py` in project root
- [x] Implement MSVR base signal loading from `msvr_enhanced.py` or indicator bank
- [x] Implement Cycle Phase timing (FFT lookback=40, min_period=5, max_period=20)
- [x] Wire SuperSmoother to smooth MSVR signal
- [x] Wire Shannon Entropy gate (entropy < 2.5 to allow trades)
- [x] Wire Efficiency Ratio gate (ER > 0.25 to allow trades)
- [x] Implement composite signal: `signal = msvr_direction * cycle_direction * supersmoother_direction * entropy_gate * er_gate`
- [x] Set MIN_HOLD = 45 days
- [x] Include 0.1% round-trip transaction costs
- [x] Output trade list with entry/exit dates and returns

**Verify:**
- Run `python3 msvr_hybrid.py` without errors
- Output shows trade list with columns: entry_date, exit_date, return
- Signal is generated for full 2018-2026 period
- Composite signal only fires when ALL filters pass
- Min hold of 45 days is enforced

**Done when:** `python3 msvr_hybrid.py` produces a trade list DataFrame with < 20 trades and prints summary statistics (trade count, win rate, Sharpe).

---

## TODO 3 — Test hybrid performance and compare with baselines

**Goal:** Validate that MSVR+Ichimoku hybrid meets or beats target metrics: < 20 trades, > 60% win rate, Sharpe > 1.35.

**Status:**
- [x] Run `msvr_hybrid.py` and capture performance metrics
- [x] Run `msvr_enhanced.py` to confirm baseline: 48 trades, 35.4% win, Sharpe 1.35
- [x] Run `ichimoku_quant.py` to confirm baseline: 11 trades, 63.6% win, Sharpe 1.31
- [x] Compare hybrid vs baselines in summary table
- [x] Verify hybrid achieves: trades < 20, win rate > 60%, Sharpe > 1.35

**Verify:**
- Print comparison table showing all three strategies side-by-side
- Hybrid trade count is between 11 and 20 (closer to Ichimoku's selectivity)
- Hybrid win rate exceeds 60%
- Hybrid Sharpe exceeds 1.35
- No degradation in total return vs Ichimoku

**Done when:** A comparison table is printed showing hybrid beats or matches Ichimoku on selectivity and win rate while exceeding Sharpe 1.35.

---

## TODO 4 — Generate comparison equity curve chart

**Goal:** Create a publication-quality comparison chart showing MSVR Hybrid equity curve alongside Ichimoku, with trade markers.

**Status:**
- [x] Create chart generation logic in `msvr_hybrid.py` or separate script
- [x] Plot MSVR Hybrid equity curve
- [x] Plot Ichimoku equity curve for reference
- [x] Add trade entry/exit markers on hybrid curve
- [x] Add legend, title, and axis labels
- [x] Save to `mttd/msvr_hybrid_comparison.png`

**Verify:**
- File `mttd/msvr_hybrid_comparison.png` exists
- Chart shows two equity curves (hybrid and Ichimoku)
- Trade markers are visible at entry/exit points
- Chart is readable at 1200x800 resolution
- Legend clearly identifies each curve

**Done when:** `mttd/msvr_hybrid_comparison.png` exists and visually shows hybrid equity curve outperforming or matching Ichimoku with clear trade markers.
