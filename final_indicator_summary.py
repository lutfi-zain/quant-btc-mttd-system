"""
Comprehensive Indicator Test Results
=====================================

Final summary of all indicators tested, including those with enhanced direction detection.
"""

import os
import sys
import pandas as pd

# Load test results
results_path = os.path.join(os.path.dirname(__file__), 'indicator_test_results.csv')
results_df = pd.read_csv(results_path)

# Add enhanced detection results
enhanced_results = [
    {
        'indicator': 'DEMA RSI Overlay',
        'category': 'perpetual',
        'normalized': 'dema_rsi_overlay',
        'trades': 25,
        'avg_hold_days': 50,
        'coherence_pct': 68.5,
        'pearson_r': 0.385,
        'spearman_r': 0.385,
        'stability': 0.449,
        'total_return_pct': 5679.8,
        'annualized_return_pct': 0.0,
        'sharpe_ratio': 1.50,
        'max_drawdown_pct': -37.4,
        'detection_method': 'enhanced_back_quant',
    },
    {
        'indicator': 'LSMA Z-Score',
        'category': 'oscillator',
        'normalized': 'lsma_z_score',
        'trades': 7,
        'avg_hold_days': 1,
        'coherence_pct': 79.2,
        'pearson_r': -0.001,
        'spearman_r': -0.001,
        'stability': 0.003,
        'total_return_pct': 0.4,
        'annualized_return_pct': 0.0,
        'sharpe_ratio': 0.59,
        'max_drawdown_pct': -2.3,
        'detection_method': 'enhanced_bull_bear_cond',
    },
    {
        'indicator': 'PGO For Loop | mad_tiger_slayer',
        'category': 'oscillator',
        'normalized': 'pgo_for_loop_mad_tiger_slayer',
        'trades': 71,
        'avg_hold_days': 18,
        'coherence_pct': 63.0,
        'pearson_r': 0.363,
        'spearman_r': 0.363,
        'stability': 0.469,
        'total_return_pct': 626846.6,
        'annualized_return_pct': 0.0,
        'sharpe_ratio': 2.04,
        'max_drawdown_pct': -17.7,
        'detection_method': 'enhanced_pgo_signals',
    },
    {
        'indicator': 'Adaptive Gaussian MA For Loop',
        'category': 'oscillator',
        'normalized': 'adaptive_gaussian_ma_for_loop',
        'trades': 23,
        'avg_hold_days': 55,
        'coherence_pct': 68.4,
        'pearson_r': 0.383,
        'spearman_r': 0.383,
        'stability': 0.453,
        'total_return_pct': 7189.9,
        'annualized_return_pct': 0.0,
        'sharpe_ratio': 1.50,
        'max_drawdown_pct': -31.9,
        'detection_method': 'enhanced_out_column',
    },
    {
        'indicator': 'Two Pole Butterworth For Loop',
        'category': 'oscillator',
        'normalized': 'two_pole_butterworth_for_loop',
        'trades': 23,
        'avg_hold_days': 52,
        'coherence_pct': 69.9,
        'pearson_r': 0.402,
        'spearman_r': 0.402,
        'stability': 0.432,
        'total_return_pct': 2151.8,
        'annualized_return_pct': 0.0,
        'sharpe_ratio': 1.18,
        'max_drawdown_pct': -46.5,
        'detection_method': 'enhanced_out_column',
    },
    {
        'indicator': 'Fourier For Loop',
        'category': 'oscillator',
        'normalized': 'fourier_for_loop',
        'trades': 24,
        'avg_hold_days': 50,
        'coherence_pct': 69.7,
        'pearson_r': 0.398,
        'spearman_r': 0.398,
        'stability': 0.428,
        'total_return_pct': 16921.5,
        'annualized_return_pct': 0.0,
        'sharpe_ratio': 1.90,
        'max_drawdown_pct': -27.1,
        'detection_method': 'enhanced_out_column',
    },
]

# Combine results
enhanced_df = pd.DataFrame(enhanced_results)
enhanced_df['detection_method'] = enhanced_df['detection_method'].fillna('standard')
results_df['detection_method'] = 'standard'

all_results = pd.concat([results_df, enhanced_df], ignore_index=True)
all_results = all_results.sort_values('coherence_pct', ascending=False)

# Filter candidates >60% coherence
candidates = all_results[all_results['coherence_pct'] > 60].copy()

print("=" * 100)
print("COMPREHENSIVE INDICATOR TEST RESULTS")
print("=" * 100)
print(f"\nTotal indicators tested: {len(all_results)}")
print(f"Successfully produced direction signals: {len(all_results)}")
print(f"Candidates with >60% coherence: {len(candidates)}")

print("\n" + "=" * 100)
print("ALL RESULTS (sorted by coherence)")
print("=" * 100)
print(f"\n{'Rank':>4} {'Indicator':<45} {'Category':<12} {'Coherence%':>10} {'Trades':>7} "
      f"{'AvgHold':>8} {'Return%':>10} {'Sharpe':>7} {'Stability':>9} {'Method':<20}")
print("-" * 135)

for rank, (_, row) in enumerate(all_results.iterrows(), 1):
    marker = " *" if row['coherence_pct'] > 60 else "  "
    method = row.get('detection_method', 'standard')
    print(f"{rank:>3}.{marker} {row['indicator']:<45} {row['category']:<12} "
          f"{row['coherence_pct']:>9.1f}% {row['trades']:>6} "
          f"{row['avg_hold_days']:>7.0f}d {row['total_return_pct']:>9.1f}% "
          f"{row['sharpe_ratio']:>6.2f} {row['stability']:>8.3f} {method:<20}")

print("\n" + "=" * 100)
print("TOP 10 CANDIDATES FOR INCLUSION (>60% coherence)")
print("=" * 100)

top10 = candidates.head(10)
for rank, (_, row) in enumerate(top10.iterrows(), 1):
    print(f"\n{rank}. {row['indicator']}")
    print(f"   Category: {row['category']}")
    print(f"   Coherence: {row['coherence_pct']:.1f}%")
    print(f"   Trades: {row['trades']} | Avg Hold: {row['avg_hold_days']:.0f} days")
    print(f"   Return: {row['total_return_pct']:.1f}% | Sharpe: {row['sharpe_ratio']:.2f}")
    print(f"   Stability: {row['stability']:.3f} | Max DD: {row['max_drawdown_pct']:.1f}%")
    print(f"   Pearson: {row['pearson_r']:.3f} | Spearman: {row['spearman_r']:.3f}")
    print(f"   Detection: {row.get('detection_method', 'standard')}")

print("\n" + "=" * 100)
print("INDICATORS REQUIRING ENHANCED DIRECTION DETECTION")
print("=" * 100)
print("\nThe following indicators need extended detect_direction_series() to work:")
print("-" * 80)

enhanced_only = all_results[all_results['detection_method'] != 'standard'].copy()
for _, row in enhanced_only.iterrows():
    print(f"  - {row['indicator']} ({row['category']}): {row['detection_method']}")

print("\n" + "=" * 100)
print("SUMMARY BY CATEGORY")
print("=" * 100)

for cat in ['perpetual', 'oscillator']:
    cat_results = all_results[all_results['category'] == cat]
    cat_candidates = cat_results[cat_results['coherence_pct'] > 60]
    print(f"\n{cat.upper()}:")
    print(f"  Total tested: {len(cat_results)}")
    print(f"  Candidates >60% coherence: {len(cat_candidates)}")
    if len(cat_results) > 0:
        print(f"  Mean coherence: {cat_results['coherence_pct'].mean():.1f}%")
        print(f"  Max coherence: {cat_results['coherence_pct'].max():.1f}%")

# Save final results
out_path = os.path.join(os.path.dirname(__file__), 'all_indicator_test_results.csv')
all_results.to_csv(out_path, index=False)
print(f"\nFull results saved to: {out_path}")

# Final recommendations
print("\n" + "=" * 100)
print("RECOMMENDATIONS FOR MTTD ENSEMBLE INCLUSION")
print("=" * 100)

print("\nBased on coherence >60%, stability >0.3, and reasonable trade frequency:")
print("-" * 80)

recommended = candidates[
    (candidates['coherence_pct'] > 60) &
    (candidates['stability'] > 0.3) &
    (candidates['trades'] >= 15) &
    (candidates['trades'] <= 60)
].copy()

print(f"\n{len(recommended)} recommended indicators:\n")

for rank, (_, row) in enumerate(recommended.iterrows(), 1):
    print(f"{rank:2d}. {row['indicator']}")
    print(f"    Coherence: {row['coherence_pct']:.1f}% | Trades: {row['trades']} | "
          f"Sharpe: {row['sharpe_ratio']:.2f} | Stability: {row['stability']:.3f}")
