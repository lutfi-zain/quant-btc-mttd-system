#!/usr/bin/env python3
"""
MTTD System Runner — Simplified Pipeline
=========================================

Uses optimized indicator parameters from grid search V2.
Pure majority-vote ensemble (no threshold/EMA/weights).
Generates full report for Telegram delivery.
"""

import os
import sys
import yaml
import re
import json
import importlib.util
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
from indicators_helper import *
from ensemble_engine import compute_ensemble_with_diagnostics
from inter_indicator_coherence import compute_all_metrics, format_coherence_report

# Make sure mttd directory exists
os.makedirs(os.path.join(project_root, "mttd"), exist_ok=True)

print("=" * 70)
print("MTTD ENSEMBLE TRADING SYSTEM — Simplified Pipeline")
print("=" * 70)

# ================================================================
# STEP 1: Load price data
# ================================================================
print("\n[Step 1] Loading price data...")

CACHE_FILE = os.path.join(project_root, "data", "btc_daily.json")
with open(CACHE_FILE) as f:
    btc_data = json.load(f)

df = pd.DataFrame(btc_data['aligned_data'])
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')
df = df[df.index >= '2018-01-01']
print(f"  Data loaded: {len(df)} daily bars ({df.index[0]} to {df.index[-1]})")

# ================================================================
# STEP 2: Load ISP benchmark
# ================================================================
print("\n[Step 2] Loading ISP benchmark...")

from coherence_metrics import load_isp_positions

csv_path = os.path.join(project_root, 'isp-signals-btcusd-2026-06-13.csv')
isp_positions_raw = load_isp_positions(csv_path)

# Convert ISP positions to DatetimeIndex for alignment with system
df_isp = pd.DataFrame({'position': isp_positions_raw})
df_isp.index = pd.to_datetime(df_isp.index)
df_isp = df_isp[df_isp.index >= '2018-01-01']
isp_positions = df_isp['position']

isp_transitions = (isp_positions.diff() != 0).sum()
print(f"  ISP: {isp_transitions} transitions, {(isp_positions == 1.0).mean()*100:.1f}% in position")

# ================================================================
# STEP 3: Load optimized indicator parameters
# ================================================================
print("\n[Step 3] Loading optimized indicator parameters...")

with open(os.path.join(project_root, 'grid_search_v2_results.json')) as f:
    gs_results = json.load(f)

optimized_params = gs_results['best_indicator_params']
MIN_HOLD = gs_results['best_min_hold']
print(f"  Optimized min_hold: {MIN_HOLD}")

# ================================================================
# STEP 4: Calculate indicators with optimized params
# ================================================================
print("\n[Step 4] Calculating indicators with optimized params...")

def detect_direction(res_df):
    """Extract direction from indicator output."""
    for col in ['dir', 'sig', 'direction', 'vii', 'qb', 'st_direction', 'trend_direction', 'trend']:
        if col in res_df.columns:
            return res_df[col]

    if 'long_signal' in res_df.columns and 'short_signal' in res_df.columns:
        direction = pd.Series(0.0, index=res_df.index)
        curr = 0.0
        for i in range(len(res_df)):
            l = bool(res_df['long_signal'].iloc[i]) if not pd.isna(res_df['long_signal'].iloc[i]) else False
            s = bool(res_df['short_signal'].iloc[i]) if not pd.isna(res_df['short_signal'].iloc[i]) else False
            if l and not s: curr = 1.0
            elif s and not l: curr = -1.0
            direction.iloc[i] = curr
        return direction

    if 'in_long_position' in res_df.columns and 'in_short_position' in res_df.columns:
        direction = pd.Series(0.0, index=res_df.index)
        direction[res_df['in_long_position'] == 1] = 1.0
        direction[res_df['in_short_position'] == 1] = -1.0
        return direction

    for col in res_df.columns:
        if 'direction' in col.lower() or 'signal' in col.lower() or 'trend' in col.lower():
            if len(res_df[col].dropna().unique()) <= 10:
                return res_df[col]
    return None

def load_indicator_func(indicator_name, category):
    """Dynamically load indicator function."""
    filename = f"{indicator_name}.py"
    module_path = os.path.join(project_root, category, filename)
    spec = importlib.util.spec_from_file_location(indicator_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, indicator_name)

def normalize_name(name):
    n = name.replace("(", "").replace(")", "")
    n = n.replace("%", "")
    n = re.sub(r"[|:\-`]", " ", n)
    n = n.lower().strip()
    n = re.sub(r"\s+", "_", n)
    n = re.sub(r"_+", "_", n)
    return n

# Indicator definitions
INDICATORS = [
    {"name": "kalman_filtered_rsi_oscillator", "category": "oscillator"},
    {"name": "z_smma_quantedgeb", "category": "oscillator"},
    {"name": "median_rsi_sd_quantedgeb", "category": "oscillator"},
    {"name": "polynomial_deviation_bands", "category": "perpetual"},
    {"name": "gaussian_smooth_trend_quantedgeb", "category": "perpetual"},
    {"name": "alma_lag_viresearch", "category": "perpetual"},
    {"name": "adaptive_regime_cloud", "category": "perpetual"},
    {"name": "root_mean_square_deviation_trend", "category": "perpetual"},
    {"name": "p_motion_trend_quantedgeb", "category": "perpetual"},
    {"name": "dema_adjusted_average_true_range", "category": "perpetual"}
]

# Load library.yaml for author mapping
lib_path = os.path.join(project_root, "library.yaml")
with open(lib_path, "r", encoding="utf-8") as f:
    content = f.read()
yaml_lines = [line for line in content.splitlines() if not line.strip().startswith("#")]
lib = yaml.safe_load("\n".join(yaml_lines))
author_map = {}
for ind in lib.get("perpetual", []):
    author_map[normalize_name(ind["indicator"])] = ind["author"]
for ind in lib.get("oscillator", []):
    author_map[normalize_name(ind["indicator"])] = ind["author"]

# Compute indicators
signal_matrix_data = {}
indicators_data = {}

for ind_def in INDICATORS:
    ind_name = ind_def['name']
    cat = ind_def['category']
    params = optimized_params.get(ind_name, {})
    author = author_map.get(ind_name, "Creator")

    print(f"  Computing {ind_name}...")

    try:
        func = load_indicator_func(ind_name, cat)

        # Filter params to only those accepted by the function
        import inspect
        sig = inspect.signature(func)
        valid_params = {k: v for k, v in params.items() if k in sig.parameters}

        res_df = func(df, **valid_params)
        direction = detect_direction(res_df)

        if direction is not None:
            binary = direction.apply(lambda x: 1.0 if x > 0 else -1.0)
            signal_matrix_data[ind_name] = binary

            # Store for visualization
            indicators_data[ind_name] = {
                'id': ind_name,
                'name': ind_name,
                'author': author,
                'params': params,
                'coherence': gs_results['indicator_isp_coherence'].get(ind_name, 0)
            }
            print(f"    ✓ {ind_name} (params: {params})")
        else:
            print(f"    ✗ {ind_name}: no direction detected")
    except Exception as e:
        print(f"    ✗ {ind_name}: {e}")

signal_matrix = pd.DataFrame(signal_matrix_data, index=df.index)
n_indicators = len(signal_matrix.columns)
print(f"\n  Active indicators: {n_indicators}")

# ================================================================
# STEP 5: Compute ensemble (pure majority vote)
# ================================================================
print(f"\n[Step 5] Computing ensemble (min_hold={MIN_HOLD})...")

ensemble_result, ensemble_diagnostics = compute_ensemble_with_diagnostics(
    signal_matrix, min_hold=MIN_HOLD
)

final_positions = ensemble_result['position']
print(f"  Position: {final_positions.mean()*100:.1f}% in market")
print(f"  Trades: {ensemble_diagnostics['n_trades']}")
print(f"  Entries: {ensemble_diagnostics['n_entries']}, Exits: {ensemble_diagnostics['n_exits']}")

# ================================================================
# STEP 6: Compute inter-indicator coherence
# ================================================================
print("\n[Step 6] Computing inter-indicator coherence...")

inter_metrics = compute_all_metrics(signal_matrix, isp_positions)
print(format_coherence_report(inter_metrics))

# ================================================================
# STEP 7: Compute ISP coherence
# ================================================================
print("\n[Step 7] Computing ISP coherence...")

from coherence_metrics import measure_coherence, format_coherence_report as fmt_coh

coherence_result = measure_coherence(
    final_positions,
    isp_positions,
    price_series=df['close'],
    coherence_threshold=75.0
)

coherence_pct = coherence_result['time_coherence']['coherence_pct']
print(f"  Time Coherence: {coherence_pct:.1f}%")
print(f"  Verdict: {'PASS' if coherence_result['verdict']['passed'] else 'FAIL'}")

# ================================================================
# STEP 8: Compute trading metrics
# ================================================================
print("\n[Step 8] Computing trading metrics...")

from risk_management import compute_equity_curve, get_risk_metrics

initial_capital = 100000.0
price_series = df['close'].astype(float)

equity_curve = compute_equity_curve(final_positions, price_series, initial_capital)
risk_metrics = get_risk_metrics(final_positions, equity_curve)

# Compute additional metrics
returns = price_series.pct_change()
strategy_returns = returns * final_positions.shift(1)
strategy_returns = strategy_returns.dropna()

years = len(strategy_returns) / 365.25
cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/years) - 1 if years > 0 else 0
sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0

downside = strategy_returns[strategy_returns < 0]
sortino = strategy_returns.mean() / downside.std() * np.sqrt(365) if len(downside) > 0 and downside.std() > 0 else 0

calmar = cagr / abs(risk_metrics['max_drawdown_pct']/100) if risk_metrics['max_drawdown_pct'] != 0 else 0
total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

system_metrics = {
    'cagr': round(cagr * 100, 2),
    'sharpe': round(sharpe, 2),
    'sortino': round(sortino, 2),
    'calmar': round(calmar, 2),
    'max_dd': round(risk_metrics['max_drawdown_pct'], 2),
    'total_return': round(total_return * 100, 2),
    'n_trades': ensemble_diagnostics['n_trades'],
    'pct_in': round(final_positions.mean() * 100, 2)
}

# ISP metrics for comparison
isp_metrics = gs_results['isp_metrics']

print(f"\n  {'Metric':<20} {'MTTD':>12} {'ISP':>12}")
print(f"  {'-'*44}")
print(f"  {'CAGR':<20} {system_metrics['cagr']:>11.2f}% {isp_metrics['cagr']:>11.2f}%")
print(f"  {'Sharpe':<20} {system_metrics['sharpe']:>12.2f} {isp_metrics['sharpe']:>12.2f}")
print(f"  {'Sortino':<20} {system_metrics['sortino']:>12.2f} {isp_metrics['sortino']:>12.2f}")
print(f"  {'Calmar':<20} {system_metrics['calmar']:>12.2f} {isp_metrics['calmar']:>12.2f}")
print(f"  {'Max DD':<20} {system_metrics['max_dd']:>11.2f}% {isp_metrics['max_dd']:>11.2f}%")
print(f"  {'Total Return':<20} {system_metrics['total_return']:>11.2f}% {isp_metrics['total_return']:>11.2f}%")
print(f"  {'Trades':<20} {system_metrics['n_trades']:>12} {isp_metrics['n_trades']:>12}")
print(f"  {'In Market':<20} {system_metrics['pct_in']:>11.1f}% {isp_metrics['pct_in']:>11.1f}%")

# ================================================================
# STEP 9: Build output JSON
# ================================================================
print("\n[Step 9] Building output JSON...")

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

# Build aggregate signals
agg_signals = []
for i, date_str in enumerate(df.index):
    pos_val = final_positions.iloc[i] if i < len(final_positions) else 0.0
    agg_signals.append({'time': date_str, 'value': float(pos_val)})

# Build markers
markers = []
prev_pos = 0.0
for i, date_str in enumerate(df.index):
    pos = final_positions.iloc[i] if i < len(final_positions) else 0.0
    if i > 0 and pos != prev_pos:
        if pos > prev_pos:
            markers.append({'time': date_str, 'position': 'belowBar', 'color': '#10b981', 'shape': 'arrowUp', 'text': 'BUY'})
        else:
            markers.append({'time': date_str, 'position': 'aboveBar', 'color': '#f43f5e', 'shape': 'arrowDown', 'text': 'SELL'})
    prev_pos = pos

# Build net vote
net_vote = []
for i, date_str in enumerate(df.index):
    net_vote.append({'time': date_str, 'value': float(signal_matrix.iloc[i].sum())})

output_dict = {
    'candles': candles_out,
    'indicators': indicators_data,
    'aggregate': {
        'name': "MTTD Ensemble System",
        'signals': agg_signals,
        'markers': markers,
        'net_vote': net_vote
    },
    'ensemble': {
        'n_indicators': n_indicators,
        'min_hold': MIN_HOLD,
        'pct_in_position': ensemble_diagnostics['pct_in_position'],
        'n_trades': ensemble_diagnostics['n_trades'],
    },
    'trading_metrics': system_metrics,
    'isp_metrics': isp_metrics,
    'coherence': {
        'time_coherence_pct': coherence_pct,
        'verdict': coherence_result['verdict']
    },
    'inter_indicator_coherence': {
        'avg_pairwise_pct': inter_metrics.get('pairwise_coherence', {}).get('avg_pct', 0),
        'min_pairwise_pct': inter_metrics.get('pairwise_coherence', {}).get('min_pct', 0),
        'avg_flip_rate': inter_metrics.get('avg_flip_rate', 0),
        'isp_coherence_per_indicator': inter_metrics.get('isp_coherence', {}),
    },
    'optimized_params': optimized_params
}

out_path = os.path.join(project_root, "mttd", "mttd_data.json")
with open(out_path, "w", encoding="utf-8") as out_f:
    json.dump(output_dict, out_f, indent=2, default=str)
print(f"  Written to: {out_path}")

# ================================================================
# STEP 10: Generate report
# ================================================================
print("\n[Step 10] Generating report...")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Generate equity curves
btc_returns = price_series.pct_change().fillna(0)
btc_equity = initial_capital * (1 + btc_returns).cumprod()

# ISP equity
# Align ISP positions with price series
isp_aligned = isp_positions.reindex(price_series.index, method='ffill').fillna(0)
isp_strategy_returns = price_series.pct_change() * isp_aligned.shift(1)
isp_strategy_returns = isp_strategy_returns.dropna()
isp_equity = initial_capital * (1 + isp_strategy_returns).cumprod()
isp_equity = pd.concat([pd.Series([initial_capital], index=[price_series.index[0]]), isp_equity])

# System equity
system_equity = equity_curve

# Generate chart
fig, axes = plt.subplots(4, 1, figsize=(16, 20), height_ratios=[3, 2, 1.5, 1.5])
fig.suptitle('MTTD Ensemble Trading System — Full Report', fontsize=16, fontweight='bold', y=0.98)

dates = df.index

# Panel 1: BTC Price + Position markers
ax1 = axes[0]
ax1.plot(dates, df['close'], color='#334155', linewidth=1, label='BTC Price', alpha=0.8)

# Shade positions
for i in range(1, len(dates)):
    date = dates[i]
    if date in final_positions.index and date in isp_positions.index:
        if final_positions.loc[date] == 1:
            ax1.axvspan(dates[i-1], dates[i], alpha=0.08, color='#10b981')
        elif isp_positions.loc[date] == 1:
            ax1.axvspan(dates[i-1], dates[i], alpha=0.05, color='#3b82f6')

# BUY/SELL markers
for m in markers:
    try:
        date = pd.Timestamp(m['time'])
        if date in df.index:
            if m['text'] == 'BUY':
                ax1.scatter(date, df.loc[date, 'close'], color='#10b981', marker='^', s=80, zorder=5)
            else:
                ax1.scatter(date, df.loc[date, 'close'], color='#f43f5e', marker='v', s=80, zorder=5)
    except:
        pass

ax1.set_ylabel('BTC Price ($)', fontsize=11)
ax1.set_title('Price & Position Signals (Green=MTTD, Blue=ISP)', fontsize=13)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Panel 2: Equity Curves
ax2 = axes[1]
ax2.plot(system_equity.index, system_equity.values, color='#10b981', linewidth=2, label='MTTD Ensemble')
ax2.plot(isp_equity.index, isp_equity.values, color='#3b82f6', linewidth=2, label='ISP Benchmark', linestyle='--')
ax2.plot(btc_equity.index, btc_equity.values, color='#94a3b8', linewidth=1, label='BTC Buy & Hold', alpha=0.6)
ax2.set_ylabel('Equity ($)', fontsize=11)
ax2.set_title('Equity Curves (Initial Capital: $100,000)', fontsize=13)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# Panel 3: Net Vote
ax3 = axes[2]
nv_values = [nv['value'] for nv in net_vote]
colors = ['#10b981' if v > 0 else '#f43f5e' if v < 0 else '#94a3b8' for v in nv_values]
ax3.bar(dates, nv_values, color=colors, alpha=0.7, width=1)
ax3.set_ylabel('Net Vote', fontsize=11)
ax3.set_title('Indicator Consensus (Positive=Bullish Majority)', fontsize=13)
ax3.grid(True, alpha=0.3)

# Panel 4: Rolling Coherence
ax4 = axes[3]
window = 90
coherence_rolling = pd.Series(index=dates, dtype=float)
# Align ISP positions with system positions
isp_aligned_for_chart = isp_positions.reindex(dates, method='ffill').fillna(0)
for i in range(window, len(dates)):
    seg_system = final_positions.iloc[i-window:i]
    seg_isp = isp_aligned_for_chart.iloc[i-window:i]
    coherence_rolling.iloc[i] = (seg_system == seg_isp).mean() * 100

ax4.plot(dates, coherence_rolling.values, color='#8b5cf6', linewidth=1.5, label=f'{window}-day Rolling Coherence')
ax4.axhline(y=75, color='#f59e0b', linestyle='--', alpha=0.7, label='75% Target')
ax4.axhline(y=95, color='#10b981', linestyle='--', alpha=0.7, label='95% Stretch Goal')
ax4.set_ylabel('Coherence %', fontsize=11)
ax4.set_title('Time Coherence with ISP (Rolling 90-Day)', fontsize=13)
ax4.set_ylim(0, 100)
ax4.legend(loc='lower left')
ax4.grid(True, alpha=0.3)

# Format x-axis
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout(rect=[0, 0, 1, 0.96])

chart_path = os.path.join(project_root, 'mttd', 'mttd_equity_report.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Chart saved to: {chart_path}")

# ================================================================
# STEP 11: Summary
# ================================================================
print("\n" + "=" * 70)
print("EXECUTION SUMMARY")
print("=" * 70)
print(f"  Indicators:          {n_indicators}")
print(f"  Data range:          {df.index[0]} to {df.index[-1]} ({len(df)} bars)")
print(f"  Min hold:            {MIN_HOLD} bars")
print(f"  Position % in:       {final_positions.mean()*100:.1f}%")
print(f"  Trades:              {ensemble_diagnostics['n_trades']}")
print(f"  Time coherence:      {coherence_pct:.1f}%")
print(f"  CAGR:                {system_metrics['cagr']:.2f}% (ISP: {isp_metrics['cagr']:.2f}%)")
print(f"  Sharpe:              {system_metrics['sharpe']:.2f} (ISP: {isp_metrics['sharpe']:.2f})")
print(f"  Max DD:              {system_metrics['max_dd']:.2f}% (ISP: {isp_metrics['max_dd']:.2f}%)")
print(f"  Chart:               {chart_path}")
print("=" * 70)
print("Done!")
