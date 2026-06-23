"""
Test All Available Indicators
=============================

Discovers new high-coherence indicators from the indicator bank for potential inclusion.
Tests all unused perpetual and oscillator indicators that have detect_direction_series().
Measures individual coherence against ISP benchmark.
"""

import os
import sys
import re
import importlib.util
import yaml
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from indicators_helper import *

# Import audit functions
from audit_indicators import (
    detect_direction_series,
    indicator_to_position,
    compute_coherence,
    compute_trade_metrics,
    compute_stability,
    compute_strategy_returns,
    compute_max_drawdown,
    compute_pearson_correlation,
    compute_spearman_correlation,
    build_isp_position_series,
    normalize_name as audit_normalize_name,
)

# Execute system imports
from execute_system import load_data


def main():
    print("=" * 80)
    print("TEST ALL AVAILABLE INDICATORS")
    print("=" * 80)
    
    # Load price data
    print("\n[1] Loading price data...")
    df = load_data()
    print(f"    Loaded {len(df)} daily bars ({df.index[0]} to {df.index[-1]})")
    
    # Load ISP signals
    csv_path = os.path.join(project_root, "isp-signals-btcusd-2026-06-13.csv")
    isp_position_full = build_isp_position_series(df, csv_path)
    
    # Get ISP date range
    df_csv = pd.read_csv(csv_path)
    first_date = str(df_csv['Date'].iloc[0])
    last_date = str(df_csv['Date'].iloc[-1])
    
    df_eval = df.loc[first_date:last_date].copy()
    isp_eval = isp_position_full.loc[first_date:last_date].copy()
    price_eval = df_eval['close']
    
    print(f"    Evaluation range: {first_date} to {last_date} ({len(df_eval)} days)")
    print(f"    ISP in-position days: {int(isp_eval.sum())}")
    
    # Load library.yaml
    lib_path = os.path.join(project_root, "library.yaml")
    with open(lib_path, "r", encoding="utf-8") as f:
        content = f.read()
    yaml_lines = [line for line in content.splitlines() if not line.strip().startswith('#')]
    lib = yaml.safe_load("\n".join(yaml_lines))
    
    # Currently selected indicators (normalized names)
    SELECTED_INDICATORS = [
        "adaptive_regime_cloud",
        "adaptive_volatility_controlled_lsma_quantalgo",
        "polynomial_deviation_bands",
        "alma_lag_viresearch",
        "lsma_viresearch",
        "dsma_viresearch",
        "irs_elder_force_volume_index",
        "gaussian_smooth_trend_quantedgeb",
        "dega_rma_quantedgeb",
        "linear_st_quantedgeb",
        "quantile_dema_trend_quantedgeb",
        "hilo_interpolation_quantedgeb",
        "madtrend_investorunknown",
        "median_deviation_suite_investorunknown",
        "root_mean_square_deviation_trend",
    ]
    selected_set = set(SELECTED_INDICATORS)
    
    # Build list of unselected converted indicators
    unselected = []
    for cat in ['perpetual', 'oscillator']:
        for ind in lib.get(cat, []):
            status = ind.get('status', 'unknown')
            conv = ind.get('conversion_status', 'unknown')
            if status == 'fetched' and conv == 'converted':
                norm = audit_normalize_name(ind['indicator'])
                if norm not in selected_set:
                    unselected.append({'name': ind['indicator'], 'category': cat, 'normalized': norm})
    
    print(f"\n[2] Testing {len(unselected)} unselected indicators...")
    print("-" * 80)
    
    results = []
    errors = []
    
    for idx, ind_info in enumerate(unselected, 1):
        name = ind_info['name']
        cat = ind_info['category']
        normalized = ind_info['normalized']
        
        py_file = os.path.join(project_root, cat, f"{normalized}.py")
        
        if not os.path.exists(py_file):
            print(f"  [{idx:2d}/{len(unselected)}] SKIP: {name} (file not found)")
            errors.append({'name': name, 'category': cat, 'error': 'file not found'})
            continue
        
        try:
            # Load and execute the indicator module
            spec = importlib.util.spec_from_file_location(normalized, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            func = getattr(module, normalized)
            
            # Run indicator
            res_df = func(df_eval)
            
            # Check if function returns a DataFrame
            if not isinstance(res_df, pd.DataFrame):
                print(f"  [{idx:2d}/{len(unselected)}] SKIP: {name} (returned {type(res_df).__name__}, not DataFrame)")
                errors.append({'name': name, 'category': cat, 'error': f'returned {type(res_df).__name__}'})
                continue
            
            # Extract direction series
            direction = detect_direction_series(res_df)
            
            if direction is None:
                print(f"  [{idx:2d}/{len(unselected)}] SKIP: {name} (no direction column found)")
                # Show available columns for debugging
                cols = list(res_df.columns)
                print(f"           Available columns: {cols[:10]}{'...' if len(cols) > 10 else ''}")
                errors.append({'name': name, 'category': cat, 'error': 'no direction column', 'columns': cols[:10]})
                continue
            
            # Convert to position series
            direction_aligned = direction.reindex(df_eval.index).fillna(0.0)
            indicator_position = indicator_to_position(direction_aligned)
            
            # Compute coherence
            coherence = compute_coherence(indicator_position, isp_eval)
            
            # Compute other metrics
            trade_metrics = compute_trade_metrics(indicator_position)
            stability = compute_stability(indicator_position)
            returns = compute_strategy_returns(indicator_position, price_eval)
            max_dd = compute_max_drawdown(indicator_position, price_eval)
            pearson = compute_pearson_correlation(indicator_position, isp_eval)
            spearman = compute_spearman_correlation(indicator_position, isp_eval)
            
            result = {
                'indicator': name,
                'category': cat,
                'normalized': normalized,
                'trades': trade_metrics['trades'],
                'avg_hold_days': trade_metrics['avg_hold_days'],
                'coherence_pct': coherence,
                'pearson_r': pearson,
                'spearman_r': spearman,
                'stability': stability,
                'total_return_pct': returns['total_return_pct'],
                'annualized_return_pct': returns['annualized_return_pct'],
                'sharpe_ratio': returns['sharpe_ratio'],
                'max_drawdown_pct': max_dd['max_drawdown_pct'],
            }
            results.append(result)
            
            # Print result
            marker = "✓" if coherence > 60 else " "
            print(f"  [{idx:2d}/{len(unselected)}] {marker} {name}")
            print(f"           coherence={coherence:.1f}%  trades={trade_metrics['trades']}  "
                  f"hold={trade_metrics['avg_hold_days']:.0f}d  "
                  f"return={returns['total_return_pct']:.1f}%  sharpe={returns['sharpe_ratio']:.2f}  "
                  f"stability={stability:.3f}  pearson={pearson:.3f}  spearman={spearman:.3f}")
            
        except Exception as e:
            print(f"  [{idx:2d}/{len(unselected)}] ERROR: {name} - {str(e)[:100]}")
            errors.append({'name': name, 'category': cat, 'error': str(e)[:200]})
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('coherence_pct', ascending=False)
        
        # Display top results
        print(f"\nTotal indicators tested: {len(results)}")
        print(f"Successfully loaded and produced direction signals: {len(results)}")
        print(f"Failed/skipped: {len(errors)}")
        
        print("\n--- All Results (sorted by coherence) ---")
        print(f"{'Rank':>4} {'Indicator':<45} {'Category':<12} {'Coherence%':>10} {'Trades':>7} "
              f"{'AvgHold':>8} {'Return%':>10} {'Sharpe':>7} {'Pearson':>8} {'Spearman':>9}")
        print("-" * 130)
        
        for rank, (_, row) in enumerate(results_df.iterrows(), 1):
            marker = " *" if row['coherence_pct'] > 60 else "  "
            print(f"{rank:>3}.{marker} {row['indicator']:<45} {row['category']:<12} "
                  f"{row['coherence_pct']:>9.1f}% {row['trades']:>6} "
                  f"{row['avg_hold_days']:>7.0f}d {row['total_return_pct']:>9.1f}% "
                  f"{row['sharpe_ratio']:>6.2f} {row['pearson_r']:>7.3f} {row['spearman_r']:>8.3f}")
        
        # Filter candidates with >60% coherence
        candidates = results_df[results_df['coherence_pct'] > 60].copy()
        
        print("\n" + "=" * 80)
        print("CANDIDATE INDICATORS (>60% individual coherence)")
        print("=" * 80)
        
        if len(candidates) > 0:
            print(f"\nFound {len(candidates)} candidates with coherence > 60%:\n")
            for rank, (_, row) in enumerate(candidates.iterrows(), 1):
                print(f"  {rank}. {row['indicator']}")
                print(f"     Category: {row['category']}")
                print(f"     Coherence: {row['coherence_pct']:.1f}%")
                print(f"     Trades: {row['trades']} | Avg Hold: {row['avg_hold_days']:.0f} days")
                print(f"     Return: {row['total_return_pct']:.1f}% | Sharpe: {row['sharpe_ratio']:.2f}")
                print(f"     Stability: {row['stability']:.3f} | Max DD: {row['max_drawdown_pct']:.1f}%")
                print(f"     Pearson: {row['pearson_r']:.3f} | Spearman: {row['spearman_r']:.3f}")
                print()
        else:
            print("\nNo indicators found with coherence > 60%")
            print("Showing top 10 indicators by coherence instead:")
            top10 = results_df.head(10)
            for rank, (_, row) in enumerate(top10.iterrows(), 1):
                print(f"  {rank}. {row['indicator']} - Coherence: {row['coherence_pct']:.1f}%")
        
        # Save results
        out_path = os.path.join(project_root, "mttd", "indicator_test_results.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\nFull results saved to: {out_path}")
        
        # Print error summary
        if errors:
            print(f"\n--- Errors/Skipped ({len(errors)} total) ---")
            for err in errors:
                print(f"  [{err['category']}] {err['name']}: {err['error'][:80]}")
        
        # Final summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)
        print(f"Total unselected indicators: {len(unselected)}")
        print(f"Successfully tested: {len(results)}")
        print(f"Failed/skipped: {len(errors)}")
        print(f"Candidates with >60% coherence: {len(candidates) if len(candidates) > 0 else 0}")
        
        if len(results) > 0:
            print(f"\nCoherence statistics:")
            print(f"  Mean:   {results_df['coherence_pct'].mean():.1f}%")
            print(f"  Median: {results_df['coherence_pct'].median():.1f}%")
            print(f"  Max:    {results_df['coherence_pct'].max():.1f}%")
            print(f"  Min:    {results_df['coherence_pct'].min():.1f}%")
        
        return results_df, candidates
    else:
        print("\nNo indicators produced results!")
        if errors:
            print(f"\nAll {len(errors)} indicators failed:")
            for err in errors:
                print(f"  [{err['category']}] {err['name']}: {err['error'][:100]}")
        return pd.DataFrame(), pd.DataFrame()


if __name__ == "__main__":
    results_df, candidates = main()
