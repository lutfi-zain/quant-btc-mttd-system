# Archive - Test & Development Files

This directory contains files that were used during development and testing but are not part of the final production system.

## Files Moved

### Grid Search Files
- `grid_search_v2.py` - Initial grid search implementation
- `grid_search_v2_ho.py` - Grid search with holdout validation
- `grid_search_v3.py` - Grid search V3 with risk-adjusted optimization
- `grid_search_v4.py` - Grid search V4 with diversified indicators
- `grid_search_advanced.py` - Grid search with advanced indicators
- `grid_search_best.py` - Grid search to find best performance
- `grid_search_coherence.py` - Grid search for ISP coherence
- `grid_search_optimize.py` - Grid search optimization
- `grid_search_reduce_trades.py` - Grid search to reduce trades
- `grid_search_ichimoku.py` - Grid search with Ichimoku

### Test Files
- `test_all_indicators.py` - Test all 60 indicators from bank
- `test_robust_ensemble.py` - Test robust ensemble configurations
- `test_serf_msvr.py` - Test SERF + MSVR combination
- `test_serf_msvr_v2.py` - Test SERF + MSVR V2
- `test_serf_msvr_v3.py` - Test SERF + MSVR V3 with cycle phase

### Analysis Files
- `analyze_isp_behavior.py` - Analyze ISP trading behavior
- `analyze_new_isp.py` - Analyze new ISP signals
- `backtest_msvr_cycle.py` - Comprehensive backtest of MSVR + Cycle
- `optimize_msvr_trend_bollinger.py` - Optimize MSVR + Trend + Bollinger
- `walkforward_best.py` - Walk-forward validation of best config
- `walkforward_all_options.py` - Walk-forward validation of all options
- `walk_forward_validate.py` - Walk-forward validation engine

### Report Files
- `AUDIT_REPORT.md` - System audit report
- `ISP_CHEATING_ANALYSIS.md` - ISP cheating analysis
- `NEXT_STEPS.md` - Next steps document
- `TODO.md` - TODO list
- `TRADE_REPORT.md` - Trade report
- `INDICATOR_SELECTION.md` - Indicator selection analysis
- `INDICATOR_ARCHITECT_ANALYSIS.md` - Technical indicator architect analysis

### Result Files
- `all_indicators_test_results.json` - Results from testing all indicators
- `serf_msvr_v3_results.json` - SERF + MSVR V3 results
- `grid_search_*.json` - Various grid search results
- `robust_ensemble_results.json` - Robust ensemble results
- `backtest_msvr_cycle_results.json` - Backtest results

### Other Files
- `calibrate_threshold.py` - Threshold calibration script
- `generate_chart.py` - Chart generation script
- `chart_msvr_cycle.py` - MSVR + Cycle chart generation
- `indicator_config.py` - Indicator configuration
- `execute_system_backup.py` - Backup of execute_system.py
- `msvr_cycle_timing.py` - MSVR cycle timing implementation
- `msvr_isp_style.py` - MSVR ISP-style implementation
- `context.md` - Context document

## Why These Files Are Archived

1. **Development artifacts**: These files were used during the research and development phase
2. **Not production-ready**: They contain experimental code that may not be optimized
3. **Redundant functionality**: Many of these files overlap with the final system
4. **Testing only**: Some files are specifically for testing and validation

## Final Production System

The final production system consists of:
- `ensemble_engine.py` - Ensemble signal generation
- `ensemble_robust.py` - Robust ensemble implementation
- `inter_indicator_coherence.py` - Inter-indicator coherence analysis
- `run_system.py` - Main system runner
- `execute_system.py` - System execution
- `report_generator.py` - Report generation
- `risk_management.py` - Risk management
- `coherence_metrics.py` - Coherence metrics
- `indicators_helper.py` - Indicator helper functions
- `ichimoku_quant.py` - Ichimoku quant system
- `mttd_system.py` - Main MTTD system

---
*Last updated: 2026-06-24*
