#!/usr/bin/env python3
"""
MTTD Indicator Optimization — Grid search parameters for each indicator
to maximize time-coherence with ISP benchmark.

Goal: Each indicator should be 95% time coherent with ISP.
"""
import os
import sys
import json
import importlib.util
import re
import numpy as np
import pandas as pd
from itertools import product

project_root = "/home/ubuntu/projects/quant-technical-indicator-bank"
sys.path.append(project_root)
from indicators_helper import *

# ================================================================
# Load Data
# ================================================================
def load_data():
    cache_path = os.path.join(project_root, "mttd", "data", "btc_daily.json")
    with open(cache_path, 'r') as f:
        raw = json.load(f)
    
    if 'candles' in raw:
        candles = raw['candles']
    elif isinstance(raw, list):
        candles = raw
    else:
        for key in raw.keys():
            if isinstance(raw[key], list) and len(raw[key]) > 100:
                candles = raw[key]
                break
    
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df = df.sort_index()
    return df[df.index >= '2018-01-01'].copy()

# ================================================================
# ISP Position Series
# ================================================================
def build_isp_positions(df):
    csv_path = os.path.join(project_root, "isp-signals-btcusd-2026-06-13.csv")
    isp = pd.read_csv(csv_path)
    isp['Date'] = pd.to_datetime(isp['Date'])
    
    # ISP uses 3-tier: Strong Bull (100%), Weak Bull (50%), Neutral (0%)
    # For coherence: treat Weak Bull as "in" (1.0)
    isp_position = pd.Series(0.0, index=df.index)
    for _, row in isp.iterrows():
        date = row['Date']
        regime = row['Regime']
        if date in isp_position.index:
            if regime in ['Strong Bull', 'Weak Bull']:
                isp_position.loc[date:] = 1.0
            else:
                isp_position.loc[date:] = 0.0
        else:
            nearest_idx = df.index.get_indexer([date], method='nearest')[0]
            nearest_date = df.index[nearest_idx]
            if regime in ['Strong Bull', 'Weak Bull']:
                isp_position.loc[nearest_date:] = 1.0
            else:
                isp_position.loc[nearest_date:] = 0.0
    
    return isp_position

# ================================================================
# Time Coherence Calculation
# ================================================================
def calculate_time_coherence(signal, isp_position):
    """Calculate time-coherence between signal and ISP position."""
    # Align series
    common_idx = signal.index.intersection(isp_position.index)
    sig = signal.reindex(common_idx).fillna(0)
    isp = isp_position.reindex(common_idx).fillna(0)
    
    # Convert to binary (1 = in market, 0 = out)
    sig_binary = (sig > 0).astype(float)
    isp_binary = (isp > 0).astype(float)
    
    # Calculate coherence
    agreement = (sig_binary == isp_binary).sum()
    total = len(common_idx)
    coherence = (agreement / total) * 100 if total > 0 else 0
    
    return coherence

# ================================================================
# Indicator Loading with Parameter Search
# ================================================================
def normalize_name(name):
    n = name.replace("(", "").replace(")", "")
    n = n.replace("%", "")
    n = re.sub(r"[|:\-`]", " ", n)
    n = n.lower().strip()
    n = re.sub(r"\s+", "_", n)
    n = re.sub(r"_+", "_", n)
    return n

def detect_direction_series(res_df):
    for col in ['dir', 'sig', 'direction', 'vii', 'qb', 'st_direction', 'trend_direction', 'trend', 'out']:
        if col in res_df.columns:
            return res_df[col]
    if 'long_signal' in res_df.columns and 'short_signal' in res_df.columns:
        direction = pd.Series(0.0, index=res_df.index)
        curr = 0.0
        for i in range(len(res_df)):
            l = res_df['long_signal'].iloc[i]
            s = res_df['short_signal'].iloc[i]
            l_val = bool(l) if not pd.isna(l) else False
            s_val = bool(s) if not pd.isna(s) else False
            if l_val and not s_val:
                curr = 1.0
            elif s_val and not l_val:
                curr = -1.0
            direction.iloc[i] = curr
        return direction
    if 'in_long_position' in res_df.columns and 'in_short_position' in res_df.columns:
        direction = pd.Series(0.0, index=res_df.index)
        direction[res_df['in_long_position'] == 1] = 1.0
        direction[res_df['in_short_position'] == 1] = -1.0
        return direction
    for col in res_df.columns:
        col_lower = col.lower()
        if 'direction' in col_lower or 'signal' in col_lower or 'trend' in col_lower:
            unique_vals = res_df[col].dropna().unique()
            if len(unique_vals) <= 10:
                return res_df[col]
    return None

# ================================================================
# Indicator Parameters to Search
# ================================================================
INDICATOR_PARAMS = {
    "lsma_z_score": {
        "length": [20, 30, 40, 50],
        "lookback": [20, 30, 40, 50]
    },
    "persistent_parabolic_sar_oscillator": {
        "start": [0.01, 0.02],
        "inc": [0.001, 0.01, 0.02],
        "max_val": [0.1, 0.2]
    },
    "lsma_for_loop_viresearch": {
        "length": [20, 30, 40, 50, 60]
    },
    "median_for_loop_viresearch": {
        "length": [20, 30, 40, 50, 60]
    },
    "hull_for_loop_viresearch": {
        "length": [20, 30, 40, 50, 60]
    },
    "dema_dmi_viresearch": {
        "length": [20, 30, 40, 50, 60]
    },
    "two_pole_butterworth_for_loop": {
        "length": [20, 30, 40, 50, 60]
    },
    "fourier_for_loop": {
        "n": [1, 2, 3],
        "start": [1, 5, 10],
        "end": [30, 45, 50],
        "upper": [35, 40, 45],
        "lower": [-15, -10, -5]
    },
    "mode_for_loop_viresearch": {
        "length": [20, 30, 40, 50, 60]
    },
    "median_rsi_sd_quantedgeb": {
        "length": [20, 30, 40, 50, 60]
    },
    "dsma_viresearch": {
        "length": [20, 30, 40, 50, 60]
    },
    "inverted_sd_dema_rsi_viresearch": {
        "length": [20, 30, 40, 50, 60]
    },
    "adaptive_gaussian_ma_for_loop": {
        "length": [20, 30, 40, 50, 60]
    },
    "dema_sma_standard_deviation_viresearch": {
        "length": [20, 30, 40, 50, 60]
    },
    "median_standard_deviation_viresearch": {
        "length": [20, 30, 40, 50, 60]
    }
}

# ================================================================
# Main Optimization
# ================================================================
def optimize_indicator(df, ind_name, ind_category, param_grid, isp_position):
    """Grid search parameters for a single indicator."""
    
    normalized = normalize_name(ind_name)
    py_file = os.path.join(project_root, ind_category, f"{normalized}.py")
    
    if not os.path.exists(py_file):
        print(f"    File not found: {py_file}")
        return None
    
    # Load module
    spec = importlib.util.spec_from_file_location(normalized, py_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, normalized)
    
    # Get parameter names from function signature
    import inspect
    sig = inspect.signature(func)
    func_params = list(sig.parameters.keys())
    
    # Filter param_grid to only include parameters that exist in function
    valid_params = {k: v for k, v in param_grid.items() if k in func_params}
    
    if not valid_params:
        print(f"    No valid parameters to search for {ind_name}")
        # Try with default parameters
        try:
            res_df = func(df)
            dir_series = detect_direction_series(res_df)
            if dir_series is not None:
                dir_binary = (dir_series > 0).astype(float)
                coherence = calculate_time_coherence(dir_binary, isp_position)
                return {
                    'indicator': ind_name,
                    'category': ind_category,
                    'params': {},
                    'coherence': coherence,
                    'status': 'default_only'
                }
        except Exception as e:
            print(f"    Error with default params: {e}")
            return None
    
    # Generate parameter combinations
    param_names = list(valid_params.keys())
    param_values = list(valid_params.values())
    combinations = list(product(*param_values))
    
    print(f"    Testing {len(combinations)} parameter combinations...")
    
    best_coherence = 0
    best_params = None
    
    for combo in combinations:
        params = dict(zip(param_names, combo))
        
        try:
            res_df = func(df, **params)
            dir_series = detect_direction_series(res_df)
            
            if dir_series is not None:
                dir_binary = (dir_series > 0).astype(float)
                coherence = calculate_time_coherence(dir_binary, isp_position)
                
                if coherence > best_coherence:
                    best_coherence = coherence
                    best_params = params.copy()
        except Exception as e:
            continue
    
    return {
        'indicator': ind_name,
        'category': ind_category,
        'params': best_params,
        'coherence': best_coherence,
        'status': 'optimized'
    }

# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("MTTD INDICATOR OPTIMIZATION")
    print("Goal: Each indicator 95% time-coherent with ISP")
    print("=" * 80)
    print()
    
    # Load data
    print("[1] Loading BTC data...")
    df = load_data()
    print(f"    Loaded {len(df)} days")
    
    # Build ISP positions
    print("[2] Building ISP position series...")
    isp_position = build_isp_positions(df)
    print(f"    ISP in market: {(isp_position > 0).sum()} days ({(isp_position > 0).mean()*100:.1f}%)")
    
    # Load indicator library
    lib_path = os.path.join(project_root, "library.yaml")
    with open(lib_path, "r", encoding="utf-8") as f:
        import yaml
        content = f.read()
    yaml_lines = [line for line in content.splitlines() if not line.strip().startswith("#")]
    lib = yaml.safe_load("\n".join(yaml_lines))
    
    # Selected indicators
    SELECTED_INDICATORS = [
        {"name": "LSMA Z-Score", "category": "oscillator"},
        {"name": "Persistent Parabolic SAR Oscillator", "category": "oscillator"},
        {"name": "lsma for loop | viResearch", "category": "oscillator"},
        {"name": "median for loop | viResearch", "category": "oscillator"},
        {"name": "hull for loop | viResearch", "category": "oscillator"},
        {"name": "dema dmi | viResearch", "category": "perpetual"},
        {"name": "Two Pole Butterworth For Loop", "category": "oscillator"},
        {"name": "Fourier For Loop", "category": "oscillator"},
        {"name": "mode for loop | viResearch", "category": "oscillator"},
        {"name": "Median RSI SD | QuantEdgeB", "category": "oscillator"},
        {"name": "DSMA | viResearch", "category": "perpetual"},
        {"name": "Inverted SD Dema RSI | viResearch", "category": "perpetual"},
        {"name": "Adaptive Gaussian MA For Loop", "category": "oscillator"},
        {"name": "DEMA SMA Standard Deviation | viResearch", "category": "perpetual"},
        {"name": "Median Standard Deviation | viResearch", "category": "perpetual"}
    ]
    
    # Optimize each indicator
    print("[3] Optimizing each indicator...")
    results = []
    
    for idx, ind in enumerate(SELECTED_INDICATORS):
        name = ind['name']
        cat = ind['category']
        normalized = normalize_name(name)
        
        print(f"\n  [{idx+1}/15] {name}...")
        
        # Get parameter grid
        param_grid = INDICATOR_PARAMS.get(normalized, {})
        
        # Optimize
        result = optimize_indicator(df, name, cat, param_grid, isp_position)
        
        if result:
            results.append(result)
            status = "✅" if result['coherence'] >= 95 else "⚠️"
            print(f"    {status} Coherence: {result['coherence']:.2f}%")
            if result['params']:
                print(f"    Best params: {result['params']}")
        else:
            print(f"    ❌ Failed to optimize")
    
    # Summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Indicator':<45} {'Coherence':<12} {'Status':<10} {'Params'}")
    print("-" * 100)
    
    passed = 0
    for r in results:
        status = "✅ PASS" if r['coherence'] >= 95 else "⚠️ FAIL"
        if r['coherence'] >= 95:
            passed += 1
        params_str = json.dumps(r['params']) if r['params'] else "default"
        print(f"{r['indicator']:<45} {r['coherence']:<12.2f} {status:<10} {params_str}")
    
    print(f"\n{'='*80}")
    print(f"Results: {passed}/{len(results)} indicators achieved 95% coherence")
    print(f"{'='*80}")
    
    # Save results
    output_path = os.path.join(project_root, "mttd", "indicator_optimization_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")
