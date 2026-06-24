# Pi Long Task TODO

## Global Constraints

- Project directory: `/home/ubuntu/projects/quant-btc-mttd-system/`
- Indicator bank: `/home/ubuntu/projects/quant-technical-indicator-bank/perpetual/`
- Baseline: Ichimoku + MSVR v8 Filtering (30 trades, 63.3% win, 1.18 Sharpe, 47.4% CAGR)
- Target: Sharpe > 1.35, 25-35 trades, win rate > 60%, CAGR > 50%
- Include 0.1% round-trip transaction costs in all calculations
- Use walk-forward validation (train 2018-2023, test 2024-2026)

## Progress

- [x] TODO 1 — Discover and Catalog Available Indicators from Bank
- [x] TODO 2 — Build Indicator Correlation and Screening Report
- [x] TODO 3 — Create optimize_ichimoku_advanced.py for Combination Testing
- [x] TODO 4 — Run Indicator Combination Optimization
- [ ] TODO 5 — Parameter Grid Search for Best Combination
- [ ] TODO 6 — Create walkforward_ichimoku_advanced.py for Validation
- [ ] TODO 7 — Execute Walk-Forward Validation and Generate Chart

---

## TODO 1 — Discover and Catalog Available Indicators from Bank

**Goal:** Generate a complete inventory of indicators available in the indicator bank, categorized by statistical family and suitability for filtering.

**Status:**
- [x] List all Python indicator files from `/home/ubuntu/projects/quant-technical-indicator-bank/perpetual/`
- [x] Categorize indicators by statistical family (Smoothing, Momentum, Volatility, Trend, Mean-Reversion, etc.)
- [x] Note input/output signatures for each indicator
- [x] Identify indicators suitable for binary signal generation

**Verify:**
- Run `ls /home/ubuntu/projects/quant-technical-indicator-bank/perpetual/*.py` and confirm all files are captured
- Check that categorization covers at least 3 different statistical families
- Output file should be readable and contain 15+ indicators

**Done when:** A complete catalog exists showing available indicators, their families, and signal interfaces suitable for integration testing.

---

## TODO 2 — Build Indicator Correlation and Screening Report

**Goal:** Test each candidate indicator individually against BTC and identify those with low correlation to Ichimoku and high standalone Sharpe.

**Status:**
- [x] Create script to test each indicator as standalone trading signal
- [x] Calculate Sharpe ratio for each indicator independently
- [x] Compute correlation matrix between Ichimoku signals and each indicator
- [x] Rank indicators by: low correlation + high Sharpe (multi-criteria ranking)
- [x] Output top 10 candidate indicators for combination testing

**Verify:**
- Script produces Sharpe > 0.4 for top candidates
- Correlation with Ichimoku < 0.3 for selected indicators
- Output file `mttd/indicator_screening_report.csv` exists with ranked results

**Done when:** A ranked list of 5-10 indicators exists with documented Sharpe values and correlation scores, ready for combination testing.

---

## TODO 3 — Create optimize_ichimoku_advanced.py for Combination Testing

**Goal:** Build the main optimization script that tests Ichimoku base signal combined with additional indicator filters.

**Status:**
- [x] Create `optimize_ichimoku_advanced.py` in project root
- [x] Implement Ichimoku base signal generation (reuse existing code)
- [x] Implement modular indicator filter addition system
- [x] Add majority-gate voting mechanism (configurable gate threshold)
- [x] Calculate performance metrics: Sharpe, win rate, trade count, CAGR, max drawdown
- [x] Support min_hold and max_hold parameters
- [x] Output results to CSV and console summary

**Verify:**
- Script runs without errors: `python3 optimize_ichimoku_advanced.py`
- Outputs CSV with columns: combination, sharpe, win_rate, trades, cagr, max_dd
- Uses baseline config (T75/250, BB25, 2.0s, MH45) as starting point
- Includes transaction cost of 0.1% round-trip

**Done when:** `optimize_ichimoku_advanced.py` exists, runs cleanly, and produces combinable filter test results.

---

## TODO 4 — Run Indicator Combination Optimization

**Goal:** Execute the optimization script to find the best indicator combination that improves Sharpe above 1.18 baseline.

**Status:**
- [x] Run optimization with Ichimoku + each top candidate indicator
- [x] Test 2-filter and 3-filter combinations
- [x] Record all combinations with metrics in `mttd/optimization_results.csv`
- [x] Identify top 3 combinations exceeding baseline Sharpe
- [x] Document best combination details (indicators, parameters, metrics)

**Verify:**
- Output file `mttd/optimization_results.csv` contains 20+ combination results
- At least one combination achieves Sharpe > 1.20 (interim target)
- Trade count remains in 25-35 range for top combinations
- Win rate > 60% for top combinations

**Done when:** Top-performing indicator combination identified with Sharpe > 1.20 and acceptable trade characteristics.

---

## TODO 5 — Parameter Grid Search for Best Combination

**Goal:** Optimize parameters for the best indicator combination found in TODO 4 to push Sharpe above 1.35.

**Status:**
- [ ] Implement grid search for min_hold values: [20, 25, 30, 35]
- [ ] Implement grid search for max_hold values: [60, 75, 90, 120]
- [ ] Implement grid search for gate requirements: [3, 4, 5]
- [ ] Test additional parameter variations specific to winning indicators
- [ ] Record all parameter combinations and metrics
- [ ] Identify optimal parameter set achieving Sharpe > 1.35

**Verify:**
- Grid search runs through at least 50 parameter combinations
- Output shows Sharpe > 1.35 for best configuration
- Trade count 25-35, win rate > 60%, CAGR > 50%
- Parameters are reasonable (not extreme edge values)

**Done when:** Optimal parameter set identified achieving all success criteria: Sharpe > 1.35, 25-35 trades, >60% win rate, >50% CAGR.

---

## TODO 6 — Create walkforward_ichimoku_advanced.py for Validation

**Goal:** Build walk-forward validation script to test the optimized configuration on out-of-sample data.

**Status:**
- [ ] Create `walkforward_ichimoku_advanced.py` in project root
- [ ] Implement train/test split: train 2018-2023, test 2024-2026
- [ ] Load best parameters from TODO 5 results
- [ ] Calculate metrics for both training and holdout periods
- [ ] Compute degradation percentage (train vs holdout)
- [ ] Output comparison table with both period metrics

**Verify:**
- Script runs without errors: `python3 walkforward_ichimoku_advanced.py`
- Outputs both training and holdout period metrics
- Degradation < 20% for Sharpe ratio
- No single metric degrades by more than 50%

**Done when:** `walkforward_ichimoku_advanced.py` exists and produces train/holdout comparison with degradation metrics.

---

## TODO 7 — Execute Walk-Forward Validation and Generate Chart

**Goal:** Run validation, confirm robustness, and generate comparison chart for documentation.

**Status:**
- [ ] Execute walk-forward validation script
- [ ] Verify holdout Sharpe > 1.08 (20% degradation from 1.35 target)
- [ ] Verify holdout win rate > 50%
- [ ] Verify holdout trade count 15-45
- [ ] Create `mttd/ichimoku_advanced_comparison.png` chart
- [ ] Chart shows: equity curve train vs holdout, metric comparison table
- [ ] Document final configuration in console output

**Verify:**
- Chart file `mttd/ichimoku_advanced_comparison.png` exists and is readable
- Holdout Sharpe > 1.08 (allowing 20% degradation)
- Final output matches expected format:
  ```
  BEST COMBINATION:
  - Base: Ichimoku
  - Additional filters: [list]
  - Parameters: [optimized values]
  - Metrics: [final validated metrics]
  ```

**Done when:** Walk-forward validation passed with < 20% degradation, chart generated, and final configuration documented meeting all success criteria.
