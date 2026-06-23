#!/usr/bin/env python3
"""
Find the Best 15-Indicator Ensemble for ISP Coherence
======================================================
Uses binary (100% BTC or 0% cash) position sizing.
Tests combinations from the top 30 performers.
Maximizes coherence with ISP target.
"""

import os
import sys
import re
import json
import importlib.util
import yaml
import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict
from datetime import datetime

project_root = "/home/ubuntu/projects/quant-technical-indicator-bank"
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "mttd"))

from indicators_helper import *
from audit_indicators import (
    detect_direction_series,
    indicator_to_position,
    compute_coherence,
    compute_trade_metrics,
    compute_stability,
    compute_strategy_returns,
    compute_max_drawdown,
    build_isp_position_series,
    normalize_name,
)

# ============================================================================
# Data Loading
# ============================================================================

def load_price_data():
    data_path = os.path.join(project_root, "mttd", "mttd_data.json")
    with open(data_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data['candles'])
    df.set_index('time', inplace=True)
    return df

def load_isp_position(df):
    csv_path = os.path.join(project_root, "isp-signals-btcusd-2026-06-13.csv")
    return build_isp_position_series(df, csv_path)

def load_indicator_results():
    results_path = os.path.join(project_root, "mttd", "all_indicator_coherence_results.json")
    with open(results_path, 'r') as f:
        return json.load(f)

def load_library():
    lib_path = os.path.join(project_root, "library.yaml")
    with open(lib_path, "r", encoding="utf-8") as f:
        content = f.read()
    yaml_lines = [line for line in content.splitlines() if not line.strip().startswith('#')]
    return yaml.safe_load("\n".join(yaml_lines))

# ============================================================================
# Compute Individual Indicator Signals
# ============================================================================

def compute_all_indicator_signals(df, top30_indicators, lib):
    signals = {}
    categories = {}
    errors = []

    for ind in top30_indicators:
        name = ind['indicator']
        cat = ind['category']
        normalized = name  # Already normalized in coherence results

        py_file = os.path.join(project_root, cat, f"{normalized}.py")
        if not os.path.exists(py_file):
            errors.append(f"File not found: {py_file}")
            continue

        try:
            spec = importlib.util.spec_from_file_location(normalized, py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            func = getattr(module, normalized)
            res_df = func(df)

            dir_series = detect_direction_series(res_df)
            if dir_series is None:
                errors.append(f"No direction series: {name}")
                continue

            dir_series = dir_series.reindex(df.index).fillna(0.0)
            position = indicator_to_position(dir_series)
            signals[name] = position
            categories[name] = cat

        except Exception as e:
            errors.append(f"Error computing {name}: {str(e)[:200]}")

    return signals, categories, errors

# ============================================================================
# Ensemble Computation
# ============================================================================

def compute_ensemble_performance(indicator_names, indicator_signals, isp_position,
                                  price_series, indicator_categories):
    n_ind = len(indicator_names)
    threshold = n_ind / 2.0

    sum_signals = pd.Series(0.0, index=price_series.index)
    valid_count = 0
    for name in indicator_names:
        if name in indicator_signals:
            sum_signals += indicator_signals[name]
            valid_count += 1

    if valid_count < n_ind:
        return None

    ensemble_position = (sum_signals > threshold).astype(float)

    coherence = compute_coherence(ensemble_position, isp_position)

    daily_returns = price_series.pct_change().fillna(0.0)
    strategy_returns = ensemble_position * daily_returns

    mean_ret = strategy_returns.mean()
    std_ret = strategy_returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(365) if std_ret > 0 else 0.0

    downside_returns = strategy_returns[strategy_returns < 0]
    std_downside = downside_returns.std()
    sortino = (mean_ret / std_downside) * np.sqrt(365) if std_downside > 0 else 0.0

    cumulative = (1 + strategy_returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(drawdown.min())

    trade_info = compute_trade_metrics(ensemble_position)

    categories = set()
    for name in indicator_names:
        cat = indicator_categories.get(name, 'unknown')
        categories.add(cat)

    return {
        'indicators': list(indicator_names),
        'n_indicators': len(indicator_names),
        'coherence': round(coherence, 2),
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'max_dd': round(max_dd, 4),
        'trades': trade_info['trades'],
        'avg_hold_days': trade_info['avg_hold_days'],
        'pct_in_position': round(float(ensemble_position.mean()) * 100, 2),
        'categories': sorted(categories),
    }

# ============================================================================
# Combination Search Strategy
# ============================================================================

def search_best_15_ensembles(df_eval, isp_eval, price_eval, indicator_signals,
                              indicator_categories, top30, n_random=5000):
    sorted_indicators = sorted(top30, key=lambda x: x['coherence'], reverse=True)
    valid_names = [ind['indicator'] for ind in sorted_indicators if ind['indicator'] in indicator_signals]
    valid_indicators = [ind for ind in sorted_indicators if ind['indicator'] in valid_names]

    print(f"\n[SEARCH] {len(valid_names)} valid indicators from top 30")

    cat_counts = defaultdict(int)
    for n in valid_names:
        cat_counts[indicator_categories.get(n, 'unknown')] += 1
    print(f"[SEARCH] Categories: {dict(cat_counts)}")

    # Individual metrics
    individual_metrics = {}
    for name in valid_names:
        result = compute_ensemble_performance(
            [name], indicator_signals, isp_eval, price_eval, indicator_categories
        )
        if result:
            individual_metrics[name] = result
            print(f"  {name}: coh={result['coherence']:.1f}%, sharpe={result['sharpe']:.3f}, "
                  f"maxDD={result['max_dd']:.2%}")

    all_results = []

    # Phase 1: Top-15 by individual coherence
    print("\n[Phase 1] Top-15 by individual coherence...")
    top15_by_coh = [ind['indicator'] for ind in valid_indicators[:15]]
    result = compute_ensemble_performance(
        top15_by_coh, indicator_signals, isp_eval, price_eval, indicator_categories
    )
    if result:
        all_results.append(result)
        print(f"  Coherence: {result['coherence']:.1f}%, Sharpe: {result['sharpe']:.3f}, "
              f"Sortino: {result['sortino']:.3f}, MaxDD: {result['max_dd']:.2%}")

    # Phase 2: Random sampling
    print(f"\n[Phase 2] Random sampling ({n_random} combinations)...")
    np.random.seed(42)

    for i in range(n_random):
        if i % 500 == 0:
            print(f"  Sampling: {i}/{n_random}")

        combo = list(np.random.choice(valid_names, size=min(15, len(valid_names)), replace=False))
        result = compute_ensemble_performance(
            combo, indicator_signals, isp_eval, price_eval, indicator_categories
        )
        if result:
            all_results.append(result)

    # Phase 3: Category-balanced combinations
    print("\n[Phase 3] Category-balanced combinations...")
    by_category = defaultdict(list)
    for ind in valid_indicators:
        by_category[ind['category']].append(ind['indicator'])

    n_perpetual = len(by_category.get('perpetual', []))
    n_oscillator = len(by_category.get('oscillator', []))
    print(f"  Perpetual: {n_perpetual}, Oscillator: {n_oscillator}")

    for _ in range(2000):
        min_perp = max(3, 15 - n_oscillator)
        max_perp = min(n_perpetual, 15 - 3)
        if min_perp > max_perp:
            continue
        n_perp = np.random.randint(min_perp, max_perp + 1)
        n_osc = 15 - n_perp

        perp_pool = by_category.get('perpetual', [])
        osc_pool = by_category.get('oscillator', [])
        if n_perp <= len(perp_pool) and n_osc <= len(osc_pool):
            perp_sample = list(np.random.choice(perp_pool, size=n_perp, replace=False))
            osc_sample = list(np.random.choice(osc_pool, size=n_osc, replace=False))
            combo = perp_sample + osc_sample
            result = compute_ensemble_performance(
                combo, indicator_signals, isp_eval, price_eval, indicator_categories
            )
            if result:
                all_results.append(result)

    # Phase 4: Greedy diversity-optimized search
    print("\n[Phase 4] Greedy diversity-optimized search...")
    for trial in range(1000):
        if trial % 200 == 0:
            print(f"  Greedy trial: {trial}/1000")

        # Random start with high-coherence indicator
        first_idx = np.random.randint(0, min(10, len(valid_indicators)))
        selected = [valid_indicators[first_idx]['indicator']]
        remaining = [n for n in valid_names if n not in selected]

        while len(selected) < 15 and remaining:
            best_next = None
            best_score = -np.inf

            for candidate in remaining[:30]:  # Limit to top candidates for speed
                coh = individual_metrics.get(candidate, {}).get('coherence', 0)
                # Compute diversity
                test_set = selected + [candidate]
                signals_df = pd.DataFrame({
                    n: indicator_signals[n]
                    for n in test_set if n in indicator_signals
                })
                if signals_df.shape[1] >= 2:
                    corr_matrix = signals_df.corr()
                    n_corr = corr_matrix.shape[0]
                    mask = np.ones((n_corr, n_corr), dtype=bool)
                    np.fill_diagonal(mask, False)
                    avg_corr = corr_matrix.values[mask].mean()
                else:
                    avg_corr = 0.0

                score = coh * 0.7 - avg_corr * 100 * 0.3 + np.random.normal(0, 0.5)
                if score > best_score:
                    best_score = score
                    best_next = candidate

            if best_next:
                selected.append(best_next)
                remaining.remove(best_next)

        if len(selected) == 15:
            result = compute_ensemble_performance(
                selected, indicator_signals, isp_eval, price_eval, indicator_categories
            )
            if result:
                all_results.append(result)

    # Phase 5: Top Sharpe and risk-optimized
    print("\n[Phase 5] Risk-optimized combinations...")
    indicators_by_sharpe = sorted(
        [n for n in valid_names if n in individual_metrics],
        key=lambda x: individual_metrics[x]['sharpe'],
        reverse=True
    )

    result = compute_ensemble_performance(
        indicators_by_sharpe[:15], indicator_signals, isp_eval, price_eval, indicator_categories
    )
    if result:
        all_results.append(result)
        print(f"  Top-15 Sharpe: coherence={result['coherence']:.1f}%, sharpe={result['sharpe']:.3f}")

    indicators_by_dd = sorted(
        [n for n in valid_names if n in individual_metrics],
        key=lambda x: individual_metrics[x]['max_dd'],
        reverse=True
    )

    result = compute_ensemble_performance(
        indicators_by_dd[:15], indicator_signals, isp_eval, price_eval, indicator_categories
    )
    if result:
        all_results.append(result)
        print(f"  Top-15 MaxDD: coherence={result['coherence']:.1f}%, maxDD={result['max_dd']:.2%}")

    # Phase 6: Cross-category optimal mixes
    print("\n[Phase 6] Cross-category optimal mixes...")
    best_perp = [n for n in indicators_by_sharpe if indicator_categories.get(n) == 'perpetual'][:10]
    best_osc = [n for n in indicators_by_sharpe if indicator_categories.get(n) == 'oscillator'][:10]

    for n_perp in range(5, 12):
        n_osc = 15 - n_perp
        if n_perp <= len(best_perp) and n_osc <= len(best_osc):
            for _ in range(100):
                perp_sample = list(np.random.choice(best_perp, size=n_perp, replace=False))
                osc_sample = list(np.random.choice(best_osc, size=n_osc, replace=False))
                combo = perp_sample + osc_sample
                result = compute_ensemble_performance(
                    combo, indicator_signals, isp_eval, price_eval, indicator_categories
                )
                if result:
                    all_results.append(result)

    # Phase 7: ISP-mimicking from top 20 by coherence
    print("\n[Phase 7] ISP-mimicking combinations...")
    top20_names = [ind['indicator'] for ind in valid_indicators[:20]]

    for _ in range(500):
        combo = list(np.random.choice(top20_names, size=15, replace=False))
        result = compute_ensemble_performance(
            combo, indicator_signals, isp_eval, price_eval, indicator_categories
        )
        if result:
            all_results.append(result)

    return all_results, individual_metrics

# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("FIND BEST 15-INDICATOR ENSEMBLE FOR ISP COHERENCE")
    print("BINARY STRATEGY: 100% BTC or 0% Cash")
    print("=" * 80)

    print("\n[1] Loading data...")
    df = load_price_data()
    isp_position = load_isp_position(df)
    price_series = df['close']
    print(f"    Price data: {len(df)} bars")
    print(f"    ISP in-position: {int(isp_position.sum())} / {len(isp_position)} days")

    csv_path = os.path.join(project_root, "isp-signals-btcusd-2026-06-13.csv")
    df_csv = pd.read_csv(csv_path)
    first_date = str(df_csv['Date'].iloc[0])
    last_date = str(df_csv['Date'].iloc[-1])
    df_eval = df.loc[first_date:last_date].copy()
    isp_eval = isp_position.loc[first_date:last_date].copy()
    price_eval = df_eval['close']
    print(f"    Eval range: {first_date} to {last_date} ({len(df_eval)} days)")
    print(f"    ISP eval in-position: {int(isp_eval.sum())} / {len(isp_eval)} days")

    print("\n[2] Loading indicator results...")
    indicator_results = load_indicator_results()
    sorted_results = sorted(indicator_results, key=lambda x: x['coherence'], reverse=True)
    top30 = sorted_results[:30]

    print("    Top 30 indicators by coherence:")
    for i, ind in enumerate(top30):
        print(f"    {i+1:2d}. {ind['indicator']:55s} [{ind['category']:10s}] {ind['coherence']:.2f}%")

    print("\n[3] Computing indicator signals...")
    lib = load_library()
    indicator_signals, indicator_categories, errors = compute_all_indicator_signals(df_eval, top30, lib)

    if errors:
        print(f"\n    Errors ({len(errors)}):")
        for err in errors:
            print(f"      - {err}")

    print(f"\n    Successfully computed {len(indicator_signals)} indicator signals")

    print("\n[4] Searching for best 15-indicator combinations...")
    all_results, individual_metrics = search_best_15_ensembles(
        df_eval, isp_eval, price_eval, indicator_signals, indicator_categories, top30,
        n_random=5000
    )

    print("\n[5] Ranking results...")
    seen = set()
    unique_results = []
    for r in all_results:
        key = tuple(sorted(r['indicators']))
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    unique_results.sort(key=lambda x: x['coherence'], reverse=True)
    top10 = unique_results[:10]

    print("\n" + "=" * 80)
    print("TOP 10 ENSEMBLE COMBINATIONS (by coherence)")
    print("=" * 80)

    for i, result in enumerate(top10):
        print(f"\n#{i+1}: Coherence={result['coherence']:.2f}%, "
              f"Sharpe={result['sharpe']:.3f}, "
              f"Sortino={result['sortino']:.3f}, "
              f"MaxDD={result['max_dd']:.2%}, "
              f"InPos={result['pct_in_position']:.1f}%")
        print(f"    Categories: {result['categories']}")
        print(f"    Indicators: {', '.join(result['indicators'])}")

    # Format for JSON output (simplified)
    output_data = []
    for r in top10:
        output_data.append({
            "indicators": r['indicators'],
            "coherence": r['coherence'],
            "sharpe": r['sharpe'],
            "sortino": r['sortino'],
            "max_dd": r['max_dd'],
            "pct_in_position": r['pct_in_position'],
            "trades": r['trades'],
            "categories": r['categories'],
        })

    output_path = os.path.join(project_root, "ensemble_combinations_15ind.json")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[6] Results saved to: {output_path}")
    print(f"    Total unique combinations tested: {len(unique_results)}")

    print("\n" + "=" * 80)
    print("ISP REFERENCE METRICS")
    print("=" * 80)
    isp_metrics_path = os.path.join(project_root, "mttd", "isp_target_metrics.json")
    with open(isp_metrics_path, 'r') as f:
        isp = json.load(f)
    print(f"    ISP Sharpe:   {isp['sharpe']:.4f}")
    print(f"    ISP Sortino:  {isp['sortino']:.4f}")
    print(f"    ISP MaxDD:    {isp['max_drawdown']:.2%}")
    print(f"    ISP Trades/yr: {isp['trades_per_year']:.2f}")
    print(f"    ISP Avg Hold:  {isp['avg_trade_duration']:.1f} hours")

    return top10

if __name__ == "__main__":
    top10 = main()
