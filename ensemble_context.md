# Code Context

## Files Retrieved
1. `/home/ubuntu/projects/quant-technical-indicator-bank/mttd/all_indicator_coherence_results.json` (60 indicators with coherence scores)
2. `/home/ubuntu/projects/quant-technical-indicator-bank/mttd/isp_target_metrics.json` (ISP benchmark: Sharpe=0.71, Sortino=3.69, MaxDD=-6.8%)
3. `/home/ubuntu/projects/quant-technical-indicator-bank/mttd/indicator_test_results.csv` (36 indicators with performance metrics)
4. `/home/ubuntu/projects/quant-technical-indicator-bank/mttd/all_indicator_test_results.csv` (36 indicators with performance metrics)
5. `/home/ubuntu/projects/quant-technical-indicator-bank/mttd/ensemble_engine.py` (ensemble engine with compute_ensemble_signal)
6. `/home/ubuntu/projects/quant-technical-indicator-bank/mttd/coherence_metrics.py` (ISP coherence measurement functions)
7. `/home/ubuntu/projects/quant-technical-indicator-bank/mttd/audit_indicators.py` (detect_direction_series, indicator_to_position)
8. `/home/ubuntu/projects/quant-technical-indicator-bank/mttd/execute_system.py` (load_data function, SELECTED_INDICATORS)
9. `/home/ubuntu/projects/quant-technical-indicator-bank/ensemble_combinations_5ind.json` (OUTPUT: top 10 ensemble combinations)

## Key Code

### ISP Target Metrics
```json
{
  "sharpe": 0.7086,
  "sortino": 3.6851,
  "max_drawdown": -0.0677
}
```

### Top 10 Ensemble Combinations (Binary 100% Equity)
The best combination achieves:
- **Coherence**: 82.69% (ISP time-alignment)
- **Sharpe**: 1.1851 (exceeds ISP's 0.71)
- **Sortino**: 1.2447
- **Max DD**: -45.51%
- **Trades**: 22 over 2790 days

### Key Indicators in Top Combinations
1. `median_rsi_sd_quantedgeb` - appears in ALL top 10 (most critical)
2. `dema_dmi_viresearch` - appears in 9/10
3. `lsma_for_loop_viresearch` - appears in 8/10
4. `median_for_loop_viresearch` - appears in 6/10
5. `hull_for_loop_viresearch` - appears in 4/10
6. `dema_sma_standard_deviation_viresearch` - appears in 4/10
7. `double_src_sma_standard_deviation_viresearch` - appears in 3/10
8. `mode_for_loop_viresearch` - appears in 3/10

### Binary Position Logic
```python
# Equal-weighted average of 5 indicator signals
avg_signal = pos_matrix.mean(axis=1)
# BINARY: 100% BTC if average > 0, else 0% cash
binary_position = (avg_signal > 0).astype(float)
```

## Architecture
- Indicators are Python files in `/perpetual/` and `/oscillator/` directories
- Each returns a DataFrame with a 'qb', 'dir', or similar direction column
- `detect_direction_series()` extracts the direction signal
- `indicator_to_position()` converts to binary position (1.0/0.0)
- `compute_ensemble_signal()` aggregates multiple indicators
- `compute_time_coherence()` measures ISP alignment

## Start Here
Open `/home/ubuntu/projects/quant-technical-indicator-bank/ensemble_combinations_5ind.json` for the final results.

## Notes
- 3 of the top 20 indicators failed to compute signals (adaptive_gaussian_ma_for_loop, dema_rsi_overlay, fourier_for_loop)
- 17/20 indicators successfully loaded, yielding C(17,5)=6188 combinations tested
- All top combinations feature median_rsi_sd_quantedgeb as the anchor indicator
- The best Sharpe (1.1887) is in combination #6 with median_rsi_sd, dema_sma_std, double_src_sma_std, lsma, median_for_loop
