import os
import sys
import yaml
import re
import importlib.util
import urllib.request
import json
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from indicators_helper import *

# Make sure mttd directory exists
os.makedirs(os.path.join(project_root, "mttd"), exist_ok=True)

# Import ensemble engine and supporting modules
from ensemble_engine import compute_ensemble_signal, compute_ensemble_with_diagnostics
from coherence_metrics import load_isp_positions, measure_coherence, format_coherence_report
from calibrate_threshold import calibrate_threshold, format_calibration_report
from walk_forward_validate import run_walk_forward_validation, format_walk_forward_report
from risk_management import apply_drawdown_pause, compute_equity_curve, get_risk_metrics

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE = os.path.join(CACHE_DIR, "btc_daily.json")

SELECTED_INDICATORS = [
    # Top 10 indicators by coherence with ISP (82.21% ensemble coherence)
    {"name": "Z SMMA | QuantEdgeB", "category": "oscillator"},               # 83.49%
    {"name": "Median RSI SD| QuantEdgeB", "category": "oscillator"},         # 83.33%
    {"name": "Kalman Filtered RSI Oscillator", "category": "oscillator"},    # 81.45%
    {"name": "Polynomial Deviation Bands", "category": "perpetual"},         # 85.56%
    {"name": "Gaussian Smooth Trend | QuantEdgeB", "category": "perpetual"}, # 84.75%
    {"name": "alma lag | viResearch", "category": "perpetual"},              # 84.68%
    {"name": "Adaptive Regime Cloud", "category": "perpetual"},              # 84.56%
    {"name": "Root Mean Square Deviation Trend", "category": "perpetual"},   # 83.94%
    {"name": "P-Motion Trend | QuantEdgeB", "category": "perpetual"},        # 83.68%
    {"name": "DEMA Adjusted Average True Range", "category": "perpetual"}   # 83.00%
]

def normalize_name(name):
    n = name.replace("(", "").replace(")", "")
    n = n.replace("%", "")
    n = re.sub(r"[|:\-`]", " ", n)
    n = n.lower().strip()
    n = re.sub(r"\s+", "_", n)
    n = re.sub(r"_+", "_", n)
    return n

def fetch_series(series_name, index="day1"):
    url = f"https://bitview.space/api/series/{series_name}/{index}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode('utf-8'))

def _save_cache(aligned_data):
    """Save aligned price data to local JSON cache."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_payload = {
        'metadata': {
            'source': 'bitview.space',
            'description': 'BTC daily OHLC + volume, aligned by date',
            'records': len(aligned_data)
        },
        'aligned_data': aligned_data
    }
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache_payload, f, indent=2)
    print(f"Price data cached to: {CACHE_FILE} ({len(aligned_data)} records)")

def _load_cache():
    """Load aligned price data from local JSON cache.
    Returns list of dicts or None if cache is missing/invalid."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        aligned = payload.get('aligned_data', [])
        if not aligned or not isinstance(aligned, list):
            print("Warning: cache file is empty or malformed, ignoring.")
            return None
        # Validate structure of first record
        sample = aligned[0]
        required_keys = {'time', 'open', 'high', 'low', 'close', 'volume'}
        if not required_keys.issubset(sample.keys()):
            print(f"Warning: cache record missing keys {required_keys - sample.keys()}, ignoring.")
            return None
        print(f"Loaded price data from cache: {CACHE_FILE} ({len(aligned)} records)")
        return aligned
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: failed to parse cache file ({e}), ignoring.")
        return None

def load_data():
    # --- Check local cache first ---
    cached = _load_cache()
    if cached is not None:
        df = pd.DataFrame(cached)
        df.set_index('time', inplace=True)
        calc_df = df[df.index >= '2018-01-01'].copy()
        return calc_df
    
    # --- Cache miss: fetch from API ---
    print("No local cache found — fetching price data from API...")
    ohlc_res = fetch_series("price_ohlc")
    volume_res = fetch_series("transfer_volume_sum_24h_usd")
    date_res = fetch_series("date")
    
    ohlc_start, ohlc_end = ohlc_res['start'], ohlc_res['start'] + len(ohlc_res['data']) - 1
    vol_start, vol_end = volume_res['start'], volume_res['start'] + len(volume_res['data']) - 1
    date_start, date_end = date_res['start'], date_res['start'] + len(date_res['data']) - 1
    
    start_idx = max(ohlc_start, vol_start, date_start)
    end_idx = min(ohlc_end, vol_end, date_end)
    
    aligned_data = []
    for idx in range(start_idx, end_idx + 1):
        date_val = date_res['data'][idx - date_res['start']]
        ohlc_val = ohlc_res['data'][idx - ohlc_res['start']]
        vol_val = volume_res['data'][idx - volume_res['start']]
        aligned_data.append({
            'time': date_val,
            'open': ohlc_val[0],
            'high': ohlc_val[1],
            'low': ohlc_val[2],
            'close': ohlc_val[3],
            'volume': vol_val
        })
    
    # Save to local cache for future runs
    _save_cache(aligned_data)
    
    df = pd.DataFrame(aligned_data)
    df.set_index('time', inplace=True)
    calc_df = df[df.index >= '2018-01-01'].copy()
    return calc_df

def detect_direction_series(res_df):
    for col in ['dir', 'sig', 'direction', 'vii', 'qb', 'st_direction', 'trend_direction', 'trend']:
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

def clean_value(val):
    if pd.isna(val) or np.isinf(val):
        return 0.0
    return float(val)


def build_output_markers(ensemble_positions, df_index):
    """Build BUY/SELL markers from binary position transitions."""
    markers = []
    prev_pos = 0.0
    for i, date_str in enumerate(df_index):
        pos = ensemble_positions.iloc[i] if i < len(ensemble_positions) else 0.0
        if i > 0:
            if pos != prev_pos:
                if pos > prev_pos:
                    markers.append({
                        'time': date_str,
                        'position': 'belowBar',
                        'color': '#10b981',
                        'shape': 'arrowUp',
                        'text': 'BUY'
                    })
                else:
                    markers.append({
                        'time': date_str,
                        'position': 'aboveBar',
                        'color': '#f43f5e',
                        'shape': 'arrowDown',
                        'text': 'SELL'
                    })
        else:
            if pos > 0.0:
                markers.append({
                    'time': date_str,
                    'position': 'belowBar',
                    'color': '#10b981',
                    'shape': 'arrowUp',
                    'text': 'BUY'
                })
        prev_pos = pos
    return markers


def main():
    print("=" * 70)
    print("MTTD ENSEMBLE TRADING SYSTEM — Full Pipeline")
    print("=" * 70)

    # ================================================================
    # STEP 1: Load price data
    # ================================================================
    print("\n[Step 1] Fetching and aligning price data...")
    df = load_data()
    print(f"  Data loaded: {len(df)} daily bars ({df.index[0]} to {df.index[-1]})")

    candles_out = []
    for date_str, row in df.iterrows():
        candles_out.append({
            'time': date_str,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })

    # ================================================================
    # STEP 2: Load indicator library and ISP benchmark
    # ================================================================
    print("\n[Step 2] Loading indicator library and ISP benchmark...")
    lib_path = os.path.join(project_root, "library.yaml")
    with open(lib_path, "r", encoding="utf-8") as f:
        content = f.read()
    yaml_lines = [line for line in content.splitlines() if not line.strip().startswith("#")]
    lib = yaml.safe_load("\n".join(yaml_lines))

    author_map = {}
    for ind in lib.get("perpetual", []):
        author_map[normalize_name(ind["indicator"])] = ind["author"]

    csv_path = os.path.join(project_root, "isp-signals-btcusd-2026-06-13.csv")

    # Load ISP positions using the coherence_metrics module
    try:
        isp_positions = load_isp_positions(csv_path)
        print(f"  ISP positions loaded: {len(isp_positions)} daily bars")
    except Exception as e:
        print(f"  Warning: Could not load ISP positions ({e}). Coherence metrics will be skipped.")
        isp_positions = None

    # ================================================================
    # STEP 3: Calculate indicators and build signal matrix
    # ================================================================
    print("\n[Step 3] Calculating indicators...")
    indicators_data = {}
    indicator_directions = {}  # For ensemble engine

    for idx, ind in enumerate(SELECTED_INDICATORS):
        name = ind['name']
        cat = ind['category']
        normalized = normalize_name(name)
        author = author_map.get(normalized, "Creator")

        py_file = os.path.join(project_root, cat, f"{normalized}.py")
        print(f"  [{idx+1}/{len(SELECTED_INDICATORS)}] {name}...")

        spec = importlib.util.spec_from_file_location(normalized, py_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, normalized)

        res_df = func(df)

        # Get direction from indicator's actual output (no ISP override)
        detected = detect_direction_series(res_df)
        if detected is not None:
            dir_series = detected.reindex(df.index).fillna(0.0)
        else:
            dir_series = pd.Series(0.0, index=df.index)

        # Store direction for ensemble engine
        indicator_directions[normalized] = dir_series

        # Convert direction to binary signals (+1 for long, -1 for short/cash)
        binary_signals = []
        for i, (date_str, val) in enumerate(dir_series.items()):
            sig = 1.0 if val > 0 else -1.0
            binary_signals.append({
                'time': date_str,
                'value': sig
            })

        # Determine primary plot column for visualization
        primary_col_map = {
            "adaptive_regime_cloud": "midline",
            "adaptive_volatility_controlled_lsma_quantalgo": "lsma",
            "polynomial_deviation_bands": "reg_val",
            "alma_lag_viresearch": "alma",
            "lsma_viresearch": "lsma",
            "dsma_viresearch": "dsma",
            "irs_elder_force_volume_index": "efi",
            "gaussian_smooth_trend_quantedgeb": "filter_gaussian",
            "dega_rma_quantedgeb": "gaussian",
            "linear_st_quantedgeb": "base",
            "quantile_dema_trend_quantedgeb": "dema",
            "hilo_interpolation_quantedgeb": "final_prcl",
            "madtrend_investorunknown": "median",
            "median_deviation_suite_investorunknown": "median",
            "root_mean_square_deviation_trend": "avg"
        }

        primary_col = primary_col_map.get(normalized, res_df.columns[0])
        if normalized == "irs_elder_force_volume_index":
            raw_vals = res_df[primary_col].fillna(0.0)
        else:
            raw_vals = res_df[primary_col].fillna(df['close'])

        indicator_values = []
        for i, (date_str, val) in enumerate(raw_vals.items()):
            # Use signal value for color (1.0 = bullish, -1.0 = bearish)
            sig = 1.0 if i < len(binary_signals) and binary_signals[i]['value'] > 0 else -1.0
            color = '#10b981' if sig > 0 else '#f43f5e'
            clamped_val = max(-8e13, min(8e13, clean_value(val)))
            indicator_values.append({
                'time': date_str,
                'value': clamped_val,
                'color': color
            })

        indicators_data[normalized] = {
            'id': normalized,
            'name': name,
            'author': author,
            'signals': binary_signals,
            'values': indicator_values
        }

    print(f"  Calculated {len(indicators_data)} indicators")

    # ================================================================
    # STEP 4: Build signal matrix and compute ensemble
    # ================================================================
    print("\n[Step 4] Computing ensemble signal via ensemble engine...")

    # Build signal matrix: columns = indicators, rows = dates
    signal_matrix_data = {}
    for ind_name, direction in indicator_directions.items():
        # Convert direction to binary: 1.0 for bullish, -1.0 for bearish/neutral
        signal_matrix_data[ind_name] = direction.apply(lambda x: 1.0 if x > 0 else -1.0)

    signal_matrix = pd.DataFrame(signal_matrix_data, index=df.index)

    # Run ensemble with diagnostics (no look-ahead verification included)
    # No EMA smoothing - indicators are already coherent with ISP
    ensemble_result, ensemble_diagnostics = compute_ensemble_with_diagnostics(
        signal_matrix,
        threshold=0.0,   # Will be calibrated below
        ema_length=1,    # No smoothing (raw average)
        weights=None      # Equal weighting (1/N)
    )

    print(f"  Ensemble computed: {ensemble_diagnostics['n_indicators']} indicators, "
          f"{ensemble_diagnostics['n_bars']} bars")
    print(f"  Initial position breakdown: "
          f"{ensemble_diagnostics['pct_in_position']:.1f}% in position, "
          f"{ensemble_diagnostics['n_trades']} trades")
    print(f"  Look-ahead verification: {'PASS' if ensemble_diagnostics['look_ahead_verification']['passed'] else 'FAIL'}")

    # ================================================================
    # STEP 5: Threshold calibration
    # ================================================================
    print("\n[Step 5] Calibrating threshold against ISP benchmark...")
    calibration_result = None
    if isp_positions is not None and len(isp_positions) > 0:
        try:
            data_start_date = df.index[0]
            data_end_date = df.index[-1]

            # Use last 12 months for calibration training window
            calibration_result = calibrate_threshold(
                df=df,
                indicator_signals=signal_matrix,
                isp_positions=isp_positions,
                train_end_date=data_end_date,
                lookback_months=12,
                threshold_min=-0.3,
                threshold_max=0.5,
                threshold_step=0.03,  # Reduced grid to avoid overfitting (27 candidates)
                ema_length=1,    # No smoothing - indicators already coherent
                weights=None
            )

            optimal_threshold = calibration_result['optimal_threshold']
            
            print(f"  Optimal threshold: {optimal_threshold:.4f}")
            print(f"  Max in-sample coherence: {calibration_result['max_coherence']:.2f}%")
            print(format_calibration_report(calibration_result))

        except Exception as e:
            print(f"  Warning: Calibration failed ({e}). Using default threshold=0.0")
            optimal_threshold = 0.0
            calibration_result = {'error': str(e), 'optimal_threshold': 0.0}
    else:
        print("  Skipping calibration: no ISP positions available")
        optimal_threshold = 0.0

    # ================================================================
    # STEP 6: Re-compute ensemble with grid search optimized parameters
    # ================================================================
    # Use grid search optimized parameters instead of calibration
    OPTIMIZED_THRESHOLD = -0.10
    OPTIMIZED_EMA_LENGTH = 3
    # Use calibrated threshold (no override)
    FINAL_THRESHOLD = optimal_threshold
    FINAL_EMA_LENGTH = 1  # No smoothing - indicators already coherent
    print(f"\n[Step 6] Re-computing ensemble with calibrated parameters...")
    print(f"  Threshold: {FINAL_THRESHOLD:.4f}, EMA Length: {FINAL_EMA_LENGTH} (no smoothing)")
    ensemble_final, ensemble_final_diagnostics = compute_ensemble_with_diagnostics(
        signal_matrix,
        threshold=FINAL_THRESHOLD,
        ema_length=FINAL_EMA_LENGTH,
        weights=None
    )

    final_positions = ensemble_final['position']
    print(f"  Final position: {final_positions.sum():.0f} bars in ({final_positions.mean()*100:.1f}%)")

    # No regime filter (user selected: remove 200-SMA)

    # ================================================================
    # STEP 7: Apply risk management (drawdown pause)
    # ================================================================
    print("\n[Step 7] Applying risk management (15% max drawdown pause)...")
    price_series = df['close'].astype(float)
    initial_capital = 100000.0

    # Compute equity curve from final positions
    equity_curve = compute_equity_curve(final_positions, price_series, initial_capital)

    # Apply drawdown pause
    risk_result = apply_drawdown_pause(
        final_positions,
        equity_curve,
        max_dd_pct=0.15,
        pause_days=20
    )

    protected_positions = risk_result['position']
    n_pause_bars = int(risk_result['in_pause'].sum())
    print(f"  Drawdown pause applied: {n_pause_bars} bars in pause")

    # Recompute equity with protected positions
    protected_equity = compute_equity_curve(protected_positions, price_series, initial_capital)
    risk_metrics = get_risk_metrics(protected_positions, protected_equity)
    print(f"  Max drawdown: {risk_metrics['max_drawdown_pct']:.2f}%")
    print(f"  Entries: {risk_metrics['n_entries']}, Exits: {risk_metrics['n_exits']}")

    # ================================================================
    # STEP 8: Compute coherence metrics
    # ================================================================
    print("\n[Step 8] Computing coherence metrics against ISP benchmark...")
    coherence_result = None
    if isp_positions is not None and len(isp_positions) > 0:
        try:
            coherence_result = measure_coherence(
                protected_positions,
                isp_positions,
                price_series=price_series,
                coherence_threshold=95.0
            )
            print(format_coherence_report(coherence_result))
        except Exception as e:
            print(f"  Warning: Coherence computation failed ({e})")
            coherence_result = {'error': str(e)}
    else:
        print("  Skipping coherence metrics: no ISP positions available")

    # ================================================================
    # STEP 9: Walk-forward validation with 5-day embargo
    # ================================================================
    print("\n[Step 9] Running walk-forward validation (5-day embargo)...")
    walk_forward_result = None
    if isp_positions is not None and len(isp_positions) > 0:
        try:
            data_start_date = df.index[0]
            data_end_date = df.index[-1]

            walk_forward_result = run_walk_forward_validation(
                indicator_signals=signal_matrix,
                isp_positions=isp_positions,
                price_series=price_series,
                data_start_date=data_start_date,
                data_end_date=data_end_date,
                initial_train_months=12,
                test_months=12,
                min_cycles=3,
                ema_length=5,    # User selected
                weights=None,
                threshold_min=-0.3,
                threshold_max=0.5,
                threshold_step=0.03,  # Reduced grid (27 candidates)
                initial_capital=initial_capital,
                commission_rate=0.0  # No costs (user selected)
            )
            print(format_walk_forward_report(walk_forward_result))
        except Exception as e:
            print(f"  Warning: Walk-forward validation failed ({e})")
            walk_forward_result = {'error': str(e)}
    else:
        print("  Skipping walk-forward: no ISP positions available")

    # ================================================================
    # STEP 10: Build output JSON
    # ================================================================
    print("\n[Step 10] Building output JSON...")

    # Convert protected positions to agg_signals format for frontend
    agg_signals = []
    for i, date_str in enumerate(df.index):
        pos_val = protected_positions.iloc[i] if i < len(protected_positions) else 0.0
        agg_signals.append({
            'time': date_str,
            'value': float(pos_val)
        })

    # Build markers from position transitions
    markers = build_output_markers(protected_positions, df.index)

    # Build net vote (sum of individual binary signals)
    net_vote = []
    for i, date_str in enumerate(df.index):
        net_vote.append({
            'time': date_str,
            'value': float(signal_matrix.iloc[i].sum())
        })

    # Build ensemble diagnostic summary for JSON
    ensemble_summary = {
        'threshold': optimal_threshold,
        'ema_length': 5,
        'n_indicators': ensemble_final_diagnostics['n_indicators'],
        'pct_in_position': ensemble_final_diagnostics['pct_in_position'],
        'n_trades': ensemble_final_diagnostics['n_trades'],
        'n_entries': ensemble_final_diagnostics['n_entries'],
        'n_exits': ensemble_final_diagnostics['n_exits'],
        'signal_range': ensemble_final_diagnostics['signal_range'],
        'smoothed_range': ensemble_final_diagnostics['smoothed_range'],
        'look_ahead_verification': ensemble_final_diagnostics['look_ahead_verification']['passed']
    }

    # Build calibration summary for JSON
    calibration_summary = None
    if calibration_result is not None and 'error' not in calibration_result:
        calibration_summary = {
            'optimal_threshold': calibration_result['optimal_threshold'],
            'max_coherence_pct': calibration_result['max_coherence'],
            'train_start_date': calibration_result['train_start_date'],
            'train_end_date': calibration_result['train_end_date'],
            'n_train_bars': calibration_result['n_train_bars'],
            'threshold_grid_size': calibration_result['threshold_grid_size']
        }
    elif calibration_result and 'error' in calibration_result:
        calibration_summary = {'error': calibration_result['error']}

    # Build coherence summary for JSON
    coherence_summary = None
    if coherence_result is not None and 'error' not in coherence_result:
        coherence_summary = {
            'time_coherence_pct': coherence_result['time_coherence']['coherence_pct'],
            'n_agree': coherence_result['time_coherence']['n_agree'],
            'n_disagree': coherence_result['time_coherence']['n_disagree'],
            'n_total': coherence_result['time_coherence']['n_total'],
            'both_in_pct': coherence_result['time_coherence']['both_in_pct'],
            'both_out_pct': coherence_result['time_coherence']['both_out_pct'],
            'mttd_in_isp_out_pct': coherence_result['time_coherence']['mttd_in_isp_out_pct'],
            'mttd_out_isp_in_pct': coherence_result['time_coherence']['mttd_out_isp_in_pct'],
            'verdict': coherence_result['verdict']
        }
        # Add timing and correlation if available
        if 'timing_error' in coherence_result:
            te = coherence_result['timing_error']
            coherence_summary['timing_error'] = {
                'avg_entry_timing_error_days': te.get('avg_entry_timing_error_days'),
                'avg_exit_timing_error_days': te.get('avg_exit_timing_error_days'),
                'mttd_n_trades': te.get('mttd_n_trades'),
                'isp_n_trades': te.get('isp_n_trades')
            }
        if 'return_correlation' in coherence_result:
            rc = coherence_result['return_correlation']
            if not rc.get('skipped') and not rc.get('error'):
                coherence_summary['return_correlation'] = {
                    'pearson_corr': rc.get('pearson_corr'),
                    'spearman_corr': rc.get('spearman_corr'),
                    'n_common_days': rc.get('n_common_days')
                }
        if 'trade_comparison' in coherence_result:
            tc = coherence_result['trade_comparison']
            coherence_summary['trade_comparison'] = tc
    elif coherence_result and 'error' in coherence_result:
        coherence_summary = {'error': coherence_result['error']}

    # Build walk-forward summary for JSON
    walk_forward_summary = None
    if walk_forward_result is not None and 'error' not in walk_forward_result:
        wf_summary = walk_forward_result.get('summary', {})
        walk_forward_summary = {
            'cycles_completed': walk_forward_result.get('cycles_completed', 0),
            'min_cycles_met': walk_forward_result.get('min_cycles_met', False),
            'verdict': wf_summary.get('verdict', {}),
            'in_sample_coherence': wf_summary.get('in_sample_coherence', {}),
            'out_of_sample_coherence': wf_summary.get('out_of_sample_coherence', {}),
            'coherence_gap': wf_summary.get('coherence_gap', {}),
            'returns': wf_summary.get('returns', {}),
            'risk': wf_summary.get('risk', {}),
            'overfitting': wf_summary.get('overfitting', {}),
        }
        # Include per-cycle details (without equity_curve which isn't JSON serializable)
        cycle_details = []
        for cycle in walk_forward_result.get('cycles', []):
            cycle_copy = {k: v for k, v in cycle.items() if k != 'equity_curve'}
            cycle_details.append(cycle_copy)
        walk_forward_summary['cycles'] = cycle_details
    elif walk_forward_result and 'error' in walk_forward_result:
        walk_forward_summary = {'error': walk_forward_result['error']}

    # Build risk management summary for JSON
    risk_summary = {
        'max_dd_pause_pct': 0.15,
        'pause_days': 20,
        'n_pause_bars': n_pause_bars,
        'max_drawdown_pct': risk_metrics['max_drawdown_pct'],
        'n_entries': risk_metrics['n_entries'],
        'n_exits': risk_metrics['n_exits'],
        'pct_in_position': risk_metrics['pct_in_position']
    }

    # Build final output dictionary
    output_dict = {
        'candles': candles_out,
        'indicators': indicators_data,
        'aggregate': {
            'name': "MTTD Ensemble System",
            'signals': agg_signals,
            'markers': markers,
            'net_vote': net_vote
        },
        'ensemble': ensemble_summary,
        'calibration': calibration_summary,
        'coherence': coherence_summary,
        'walk_forward': walk_forward_summary,
        'risk_management': risk_summary
    }

    out_path = os.path.join(project_root, "mttd", "mttd_data.json")
    with open(out_path, "w", encoding="utf-8") as out_f:
        json.dump(output_dict, out_f, indent=2, default=str)
    print(f"\nMTTD System data generated successfully! Written to: {out_path}")

    # Also write a copy to the dashboard's data directory
    dashboard_data_dir = os.path.join(project_root, "dashboard/src/data")
    os.makedirs(dashboard_data_dir, exist_ok=True)
    dashboard_out_path = os.path.join(dashboard_data_dir, "mttd_data.json")
    with open(dashboard_out_path, "w", encoding="utf-8") as out_f:
        json.dump(output_dict, out_f, indent=2, default=str)
    print(f"Copied to dashboard: {dashboard_out_path}")

    # ================================================================
    # STEP 11: Print summary
    # ================================================================
    print("\n" + "=" * 70)
    print("EXECUTION SUMMARY")
    print("=" * 70)
    print(f"  Indicators:          {len(indicators_data)}")
    print(f"  Data range:          {df.index[0]} to {df.index[-1]} ({len(df)} bars)")
    print(f"  Calibrated threshold: {optimal_threshold:.4f}")
    print(f"  Position % in:       {final_positions.mean()*100:.1f}%")
    print(f"  Trades:              {ensemble_final_diagnostics['n_trades']}")
    print(f"  Pause bars:          {n_pause_bars}")
    if coherence_summary and 'time_coherence_pct' in coherence_summary:
        coh_pct = coherence_summary['time_coherence_pct']
        verdict_str = coherence_summary.get('verdict', {}).get('passed', False)
        print(f"  Time-coherence:      {coh_pct:.2f}% ({'PASS' if verdict_str else 'FAIL'} >= 95%)")
    if walk_forward_summary and 'verdict' in walk_forward_summary:
        wf_passed = walk_forward_summary['verdict'].get('passed', False)
        n_cycles = walk_forward_summary.get('cycles_completed', 0)
        avg_oos = walk_forward_summary.get('out_of_sample_coherence', {}).get('mean', 0)
        print(f"  Walk-forward:        {n_cycles} cycles, OOS coherence={avg_oos:.2f}% ({'PASS' if wf_passed else 'FAIL'})")
    print("=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
