================================================================================
FINAL INDICATOR SELECTION DOCUMENTATION
================================================================================

Selection Criteria:
  1. Individual coherence with ISP benchmark > 50%
  2. Prefer higher coherence indicators
  3. Balance between perpetual and oscillator types
  4. Consider risk metrics (Sharpe ratio, max drawdown)
  5. Target 15-25 indicators for ensemble diversity

--------------------------------------------------------------------------------
SELECTION SUMMARY
--------------------------------------------------------------------------------
Total indicators evaluated: 51
Indicators meeting coherence threshold (>50%): 50
Final selected indicators: 20

Category breakdown:
  Perpetual: 8
  Oscillator: 12

Coherence statistics (selected):
  Mean:   69.9%
  Median: 68.8%
  Min:    63.0%
  Max:    79.2%

--------------------------------------------------------------------------------
SELECTED INDICATORS (ranked by coherence)
--------------------------------------------------------------------------------

Rank Indicator                                          Category      Coherence   Sharpe    MaxDD
----------------------------------------------------------------------------------------------------
   1 LSMA Z-Score                                       oscillator        79.2%    0.59    -2.3%
   2 Persistent Parabolic SAR Oscillator                oscillator        79.0%    1.59    -1.7%
   3 lsma for loop | viResearch                         oscillator        73.3%    2.18   -25.6%
   4 median for loop | viResearch                       oscillator        73.0%    1.68   -29.8%
   5 hull for loop | viResearch                         oscillator        71.8%    1.42   -37.3%
   6 dema dmi | viResearch                              perpetual         70.6%    1.48   -32.5%
   7 Two Pole Butterworth For Loop                      oscillator        69.9%    1.18   -46.5%
   8 Fourier For Loop                                   oscillator        69.7%    1.90   -27.1%
   9 mode for loop | viResearch                         oscillator        69.2%    1.90   -26.7%
  10 Median RSI SD| QuantEdgeB                          oscillator        69.1%    1.19   -52.6%
  11 DSMA | viResearch                                  perpetual         68.6%    2.17   -25.6%
  12 DEMA RSI Overlay                                   perpetual         68.5%    1.50   -37.4%
  13 Inverted SD Dema RSI | viResearch                  perpetual         68.5%    1.50   -37.4%
  14 Adaptive Gaussian MA For Loop                      oscillator        68.4%    1.50   -31.9%
  15 Double Src SMA Standard Deviation | viResearch     perpetual         68.1%    2.16   -25.6%
  16 DEMA SMA Standard Deviation | viResearch           perpetual         68.1%    2.14   -25.6%
  17 Median Standard Deviation | viResearch             perpetual         67.7%    2.76   -25.6%
  18 HILO Interpolation | QuantEdgeB                    perpetual         67.1%    2.24   -23.7%
  19 Z SMMA | QuantEdgeB                                oscillator        64.5%    1.25   -59.2%
  20 FDI Adaptive Oscillator Suite                      oscillator        63.0%    2.04   -39.8%

--------------------------------------------------------------------------------
SELECTION RATIONALE
--------------------------------------------------------------------------------

The final indicator set was selected based on the following rationale:

1. COHERENCE THRESHOLD (>50%):
   - All selected indicators demonstrate >50% time-coherence with the ISP benchmark
   - This ensures each indicator contributes meaningful signal alignment

2. DIVERSITY:
   - Mix of perpetual (trend-following) and oscillator (mean-reversion) indicators
   - Different calculation methodologies reduce correlation risk
   - Balanced representation prevents over-reliance on one indicator type

3. RISK-ADJUSTED PERFORMANCE:
   - Preference for indicators with positive Sharpe ratios
   - Consideration of maximum drawdown (lower is better)
   - Stability metric indicates consistent signal generation

4. ENSEMBLE DIVERSITY:
   - 20 indicators provide sufficient diversity for averaging
   - Equal weighting (1/N) applied to all indicators
   - Individual indicator failure won't collapse the ensemble

--------------------------------------------------------------------------------
INDICATOR DETAILS
--------------------------------------------------------------------------------

1. LSMA Z-Score
   Category: oscillator
   Normalized: lsma_z_score
   Coherence: 79.2%
   Trades: 7 | Avg Hold: 1 days
   Return: 0.4% | Sharpe: 0.59
   Stability: 0.003 | Max DD: -2.3%
   Pearson: -0.001 | Spearman: -0.001

2. Persistent Parabolic SAR Oscillator
   Category: oscillator
   Normalized: persistent_parabolic_sar_oscillator
   Coherence: 79.0%
   Trades: 25 | Avg Hold: 1 days
   Return: 193.8% | Sharpe: 1.59
   Stability: 0.009 | Max DD: -1.7%
   Pearson: -0.001 | Spearman: -0.001

3. lsma for loop | viResearch
   Category: oscillator
   Normalized: lsma_for_loop_viresearch
   Coherence: 73.3%
   Trades: 26 | Avg Hold: 42 days
   Return: 37029.1% | Sharpe: 2.18
   Stability: 0.390 | Max DD: -25.6%
   Pearson: 0.428 | Spearman: 0.428

4. median for loop | viResearch
   Category: oscillator
   Normalized: median_for_loop_viresearch
   Coherence: 73.0%
   Trades: 25 | Avg Hold: 44 days
   Return: 8117.5% | Sharpe: 1.68
   Stability: 0.395 | Max DD: -29.8%
   Pearson: 0.425 | Spearman: 0.425

5. hull for loop | viResearch
   Category: oscillator
   Normalized: hull_for_loop_viresearch
   Coherence: 71.8%
   Trades: 21 | Avg Hold: 54 days
   Return: 3795.9% | Sharpe: 1.42
   Stability: 0.409 | Max DD: -37.3%
   Pearson: 0.412 | Spearman: 0.412

6. dema dmi | viResearch
   Category: perpetual
   Normalized: dema_dmi_viresearch
   Coherence: 70.6%
   Trades: 27 | Avg Hold: 45 days
   Return: 4930.8% | Sharpe: 1.48
   Stability: 0.434 | Max DD: -32.5%
   Pearson: 0.417 | Spearman: 0.417

7. Two Pole Butterworth For Loop
   Category: oscillator
   Normalized: two_pole_butterworth_for_loop
   Coherence: 69.9%
   Trades: 23 | Avg Hold: 52 days
   Return: 2151.8% | Sharpe: 1.18
   Stability: 0.432 | Max DD: -46.5%
   Pearson: 0.402 | Spearman: 0.402

8. Fourier For Loop
   Category: oscillator
   Normalized: fourier_for_loop
   Coherence: 69.7%
   Trades: 24 | Avg Hold: 50 days
   Return: 16921.5% | Sharpe: 1.90
   Stability: 0.428 | Max DD: -27.1%
   Pearson: 0.398 | Spearman: 0.398

9. mode for loop | viResearch
   Category: oscillator
   Normalized: mode_for_loop_viresearch
   Coherence: 69.2%
   Trades: 26 | Avg Hold: 47 days
   Return: 18697.2% | Sharpe: 1.90
   Stability: 0.437 | Max DD: -26.7%
   Pearson: 0.388 | Spearman: 0.388

10. Median RSI SD| QuantEdgeB
   Category: oscillator
   Normalized: median_rsi_sd_quantedgeb
   Coherence: 69.1%
   Trades: 21 | Avg Hold: 62 days
   Return: 2182.9% | Sharpe: 1.19
   Stability: 0.466 | Max DD: -52.6%
   Pearson: 0.424 | Spearman: 0.424

11. DSMA | viResearch
   Category: perpetual
   Normalized: dsma_viresearch
   Coherence: 68.6%
   Trades: 53 | Avg Hold: 24 days
   Return: 39117.6% | Sharpe: 2.17
   Stability: 0.456 | Max DD: -25.6%
   Pearson: 0.398 | Spearman: 0.398

12. DEMA RSI Overlay
   Category: perpetual
   Normalized: dema_rsi_overlay
   Coherence: 68.5%
   Trades: 25 | Avg Hold: 50 days
   Return: 5679.8% | Sharpe: 1.50
   Stability: 0.449 | Max DD: -37.4%
   Pearson: 0.385 | Spearman: 0.385

13. Inverted SD Dema RSI | viResearch
   Category: perpetual
   Normalized: inverted_sd_dema_rsi_viresearch
   Coherence: 68.5%
   Trades: 25 | Avg Hold: 50 days
   Return: 5679.8% | Sharpe: 1.50
   Stability: 0.449 | Max DD: -37.4%
   Pearson: 0.385 | Spearman: 0.385

14. Adaptive Gaussian MA For Loop
   Category: oscillator
   Normalized: adaptive_gaussian_ma_for_loop
   Coherence: 68.4%
   Trades: 23 | Avg Hold: 55 days
   Return: 7189.9% | Sharpe: 1.50
   Stability: 0.453 | Max DD: -31.9%
   Pearson: 0.383 | Spearman: 0.383

15. Double Src SMA Standard Deviation | viResearch
   Category: perpetual
   Normalized: double_src_sma_standard_deviation_viresearch
   Coherence: 68.1%
   Trades: 55 | Avg Hold: 23 days
   Return: 38412.6% | Sharpe: 2.16
   Stability: 0.460 | Max DD: -25.6%
   Pearson: 0.392 | Spearman: 0.392

16. DEMA SMA Standard Deviation | viResearch
   Category: perpetual
   Normalized: dema_sma_standard_deviation_viresearch
   Coherence: 68.1%
   Trades: 56 | Avg Hold: 23 days
   Return: 36822.6% | Sharpe: 2.14
   Stability: 0.460 | Max DD: -25.6%
   Pearson: 0.390 | Spearman: 0.390

17. Median Standard Deviation | viResearch
   Category: perpetual
   Normalized: median_standard_deviation_viresearch
   Coherence: 67.7%
   Trades: 46 | Avg Hold: 28 days
   Return: 339035.1% | Sharpe: 2.76
   Stability: 0.453 | Max DD: -25.6%
   Pearson: 0.372 | Spearman: 0.372

18. HILO Interpolation | QuantEdgeB
   Category: perpetual
   Normalized: hilo_interpolation_quantedgeb
   Coherence: 67.1%
   Trades: 38 | Avg Hold: 34 days
   Return: 61641.5% | Sharpe: 2.24
   Stability: 0.458 | Max DD: -23.7%
   Pearson: 0.364 | Spearman: 0.364

19. Z SMMA | QuantEdgeB
   Category: oscillator
   Normalized: z_smma_quantedgeb
   Coherence: 64.5%
   Trades: 27 | Avg Hold: 54 days
   Return: 3136.2% | Sharpe: 1.25
   Stability: 0.523 | Max DD: -59.2%
   Pearson: 0.393 | Spearman: 0.393

20. FDI Adaptive Oscillator Suite
   Category: oscillator
   Normalized: fdi_adaptive_oscillator_suite
   Coherence: 63.0%
   Trades: 63 | Avg Hold: 23 days
   Return: 57254.1% | Sharpe: 2.04
   Stability: 0.527 | Max DD: -39.8%
   Pearson: 0.363 | Spearman: 0.363

================================================================================
END OF DOCUMENTATION
================================================================================