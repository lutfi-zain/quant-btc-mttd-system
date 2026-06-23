# MTTD System Overhaul — TODO

## Phase 1: Bug Fixes ✅
- [x] 1a. Fix `dema_adjusted_average_true_range.py` — added `direction` column (dema_out >= dema_atr → bullish)
- [x] 1b. Fix `kalman_filtered_rsi_oscillator.py` — added `direction` column (normalized_rsi > 0 → bullish)

## Phase 2: Ensemble Simplification ✅
- [x] 2a. Simplify `ensemble_engine.py` — pure majority vote (mean > 0 → position 1), no threshold/EMA/weights
- [x] 2b. Created `run_system.py` — streamlined pipeline with min_hold, inter-indicator coherence

## Phase 3: Inter-Indicator Coherence Measurement ✅
- [x] 3a. Created `inter_indicator_coherence.py` — pairwise coherence, individual ISP coherence, flip rates, agreement windows

## Phase 4: Grid Search ✅
- [x] 4a. Created `grid_search_v2.py` — two-phase: optimize each indicator's params, then optimize min_hold
- [x] 4b. Fitness: trading metrics vs ISP benchmark, while maintaining ISP coherence

## Phase 5: Report Generation & Telegram ✅
- [x] 5a. Created report generation in `run_system.py` — equity curve chart (matplotlib) + full metrics
- [x] 5b. Sent to Telegram via telegram_attach + telegram_message

## Phase 6: Integration & Run ✅
- [x] 6a. Ran grid search → found optimal indicator params + min_hold=10
- [x] 6b. Updated system with optimized params
- [x] 6c. Ran full system → generated chart + metrics
- [x] 6d. Sent report to Telegram

## Results Summary
- **ISP Coherence**: 80.6% (target: ≥75%) ✅
- **CAGR**: 93.36% (ISP: 78.13%) — EXCEEDS ISP ✅
- **Sharpe**: 1.32 (ISP: 1.88) — below ISP ✗
- **Sortino**: 1.38 (ISP: 1.76) — below ISP ✗
- **Calmar**: 2.74 (ISP: 3.05) — below ISP ✗
- **Max DD**: -34.07% (ISP: -25.65%) — worse than ISP ✗
- **Total Return**: 26,557% (ISP: 13,202%) — EXCEEDS ISP ✅
- **Inter-indicator coherence**: 85.8% (well-synchronized)

## Key Files Created/Modified
- `ensemble_engine.py` — Simplified to pure majority vote
- `inter_indicator_coherence.py` — New module for coherence measurement
- `grid_search_v2.py` — New grid search with ISP benchmark comparison
- `run_system.py` — New streamlined execution pipeline
- `perpetual/dema_adjusted_average_true_range.py` — Bug fix (direction column)
- `oscillator/kalman_filtered_rsi_oscillator.py` — Bug fix (direction column)

## Next Steps (if desired)
1. Add trailing stop or volatility-based risk management to reduce Max DD
2. Increase min_hold further to reduce trade count
3. Test different agreement thresholds (e.g., require 7/10 indicators to agree)
4. Add position sizing based on consensus strength
