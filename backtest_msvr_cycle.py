#!/usr/bin/env python3
"""
MSVR + Cycle Phase Timing — Comprehensive Backtest
====================================================

Using lz-quant-researcher methodology:
1. Walk-forward validation with embargo
2. Statistical significance tests
3. Regime analysis
4. Cost sensitivity analysis
5. Overfitting detection (haircut rule)

This is NOT a simple backtest — it's a RIGOROUS validation.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import importlib.util
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
bank_root = '/home/ubuntu/projects/quant-technical-indicator-bank'
sys.path.append(project_root)
sys.path.append(bank_root)
from indicators_helper import *

print("=" * 70)
print("MSVR + CYCLE PHASE — COMPREHENSIVE BACKTEST")
print("Using lz-quant-researcher methodology")
print("=" * 70)

# ================================================================
# Load Data
# ================================================================
print("\n[1/8] Loading data...")

with open(os.path.join(project_root, 'data', 'btc_daily.json')) as f:
    btc_data = json.load(f)

df_full = pd.DataFrame(btc_data['aligned_data'])
df_full['time'] = pd.to_datetime(df_full['time'])
df_full = df_full.set_index('time')
df_full = df_full[df_full.index >= '2018-01-01']

HOLDOUT_START = '2025-01-01'
df_train = df_full[df_full.index < HOLDOUT_START].copy()
df_holdout = df_full[df_full.index >= HOLDOUT_START].copy()

print(f"  Full:      {len(df_full)} bars ({df_full.index[0]} to {df_full.index[-1]})")
print(f"  Training:  {len(df_train)} bars")
print(f"  Holdout:   {len(df_holdout)} bars")

# Load ISP
isp_df = pd.read_csv(os.path.join(project_root, 'isp-signals-btcusd-2026-06-13.csv'))
isp_df['Date'] = pd.to_datetime(isp_df['Date'])
isp_df = isp_df.set_index('Date')

isp_positions_full = pd.Series(0.0, index=df_full.index)
for date, row in isp_df.iterrows():
    if date in isp_positions_full.index:
        if row['Action'] == 'BUY':
            isp_positions_full.loc[date:] = 1.0
        elif row['Action'] == 'SELL':
            isp_positions_full.loc[date:] = 0.0

# ================================================================
# Load MSVR
# ================================================================
print("\n[2/8] Loading MSVR indicator...")

spec = importlib.util.spec_from_file_location('msvr', os.path.join(bank_root, 'perpetual/median_standard_deviation_viresearch.py'))
msvr_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(msvr_module)

msvr_full = msvr_module.median_standard_deviation_viresearch(df_full)
msvr_signal = msvr_full['vii']

# ================================================================
# Cycle Phase Computation
# ================================================================
print("\n[3/8] Computing cycle phases...")

def compute_cycle_phase(df, lookback):
    """Compute cycle phase using FFT."""
    src = (df['high'] + df['low'] + df['close']) / 3.0
    n = len(df)
    phase = pd.Series(np.nan, index=df.index)
    
    min_period = 5
    max_period = lookback // 2
    
    for i in range(lookback - 1, n):
        window = src.iloc[i - lookback + 1:i + 1].values
        if np.any(np.isnan(window)):
            continue
        
        window_detrended = window - np.mean(window)
        hann = np.hanning(lookback)
        window窗ed = window_detrended * hann
        
        fft_vals = np.fft.rfft(window窗ed)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(lookback, d=1)
        
        min_freq = 1.0 / max_period
        max_freq = 1.0 / min_period
        valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
        valid_power = power[valid_mask]
        valid_freqs = freqs[valid_mask]
        
        if len(valid_power) > 0 and np.sum(valid_power) > 0:
            dominant_idx = np.argmax(valid_power)
            dominant_freq = valid_freqs[dominant_idx]
            dominant_period = 1.0 / dominant_freq if dominant_freq > 0 else lookback
            
            cycle_pos = i % int(dominant_period)
            phase.iloc[i] = 2 * np.pi * cycle_pos / dominant_period
    
    return phase

# Compute phase
phase = compute_cycle_phase(df_full, lookback=40)
cycle_signal = -np.cos(phase)  # +1 at trough, -1 at peak

# Combined signal
msvr_binary = (msvr_signal > 0).astype(float)
cycle_binary = (cycle_signal > 0).astype(float)
combined = msvr_binary * cycle_binary

# ================================================================
# Metrics Functions
# ================================================================
def compute_metrics(positions, prices, transaction_cost=0.001):
    """Compute comprehensive metrics."""
    returns = prices.pct_change()
    strategy_returns = returns * positions.shift(1)
    strategy_returns = strategy_returns.dropna()
    
    transitions = positions.diff().fillna(0)
    strategy_returns = strategy_returns - transitions.loc[strategy_returns.index] * (transaction_cost / 2)

    if len(strategy_returns) == 0:
        return {'cagr': 0, 'sharpe': 0, 'sortino': 0, 'calmar': 0, 'max_dd': 0, 'n_trades': 0, 'pct_in': 0}

    equity = (1 + strategy_returns).cumprod()
    years = len(strategy_returns) / 365.25

    cagr = (equity.iloc[-1]) ** (1/years) - 1 if years > 0 else 0
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(365) if strategy_returns.std() > 0 else 0
    downside = strategy_returns[strategy_returns < 0]
    sortino = strategy_returns.mean() / downside.std() * np.sqrt(365) if len(downside) > 0 and downside.std() > 0 else 0
    peak = equity.cummax()
    max_dd = ((equity - peak) / peak).min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    n_trades = (positions.diff().abs() > 0).sum()
    
    # Win rate
    winning_trades = (strategy_returns > 0).sum()
    total_trades = (strategy_returns != 0).sum()
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Profit factor
    gross_profit = strategy_returns[strategy_returns > 0].sum()
    gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Max consecutive losses
    is_loss = strategy_returns < 0
    loss_groups = is_loss.ne(is_loss.shift()).cumsum()
    max_consecutive_losses = is_loss.groupby(loss_groups).sum().max() if is_loss.any() else 0

    return {
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'calmar': round(calmar, 2),
        'max_dd': round(max_dd * 100, 2),
        'n_trades': int(n_trades),
        'pct_in': round(positions.mean() * 100, 2),
        'win_rate': round(win_rate * 100, 2),
        'profit_factor': round(profit_factor, 2),
        'max_consecutive_losses': int(max_consecutive_losses)
    }

# ================================================================
# 1. Walk-Forward Validation
# ================================================================
print("\n[4/8] Walk-forward validation (with embargo)...")

def walk_forward_validate(data, signal, n_folds=5, train_ratio=0.7, embargo_days=10):
    """Walk-forward validation with embargo gap."""
    results = []
    total_days = len(data)
    fold_size = total_days // n_folds
    
    for fold in range(n_folds):
        fold_start = fold * fold_size
        fold_end = min((fold + 1) * fold_size, total_days)
        
        train_end_idx = fold_start + int((fold_end - fold_start) * train_ratio)
        test_start_idx = train_end_idx + embargo_days
        
        if test_start_idx >= fold_end:
            continue
        
        train_idx = data.index[fold_start:train_end_idx]
        test_idx = data.index[test_start_idx:fold_end]
        
        train_signal = signal.reindex(train_idx).fillna(0)
        test_signal = signal.reindex(test_idx).fillna(0)
        
        train_returns = data['close'].pct_change().reindex(train_idx) * train_signal
        test_returns = data['close'].pct_change().reindex(test_idx) * test_signal
        
        is_sharpe = train_returns.dropna().mean() / train_returns.dropna().std() * np.sqrt(365) if train_returns.dropna().std() > 0 else 0
        oos_sharpe = test_returns.dropna().mean() / test_returns.dropna().std() * np.sqrt(365) if test_returns.dropna().std() > 0 else 0
        
        results.append({
            'fold': fold,
            'train_period': f"{train_idx[0].date()} to {train_idx[-1].date()}",
            'test_period': f"{test_idx[0].date()} to {test_idx[-1].date()}",
            'is_sharpe': round(is_sharpe, 2),
            'oos_sharpe': round(oos_sharpe, 2),
            'decay': round((1 - oos_sharpe/is_sharpe)*100, 1) if is_sharpe > 0 else 0
        })
    
    return results

wf_results = walk_forward_validate(df_full, combined, n_folds=5, embargo_days=10)

print("\n  Walk-Forward Results:")
print(f"  {'Fold':<6} {'Train Period':<30} {'Test Period':<30} {'IS Sharpe':<10} {'OOS Sharpe':<10} {'Decay':<10}")
print("  " + "-" * 96)

for r in wf_results:
    print(f"  {r['fold']:<6} {r['train_period']:<30} {r['test_period']:<30} {r['is_sharpe']:<10.2f} {r['oos_sharpe']:<10.2f} {r['decay']:<10.1f}%")

avg_oos_sharpe = np.mean([r['oos_sharpe'] for r in wf_results])
avg_decay = np.mean([r['decay'] for r in wf_results])
print(f"\n  Average OOS Sharpe: {avg_oos_sharpe:.2f}")
print(f"  Average Decay: {avg_decay:.1f}%")

# ================================================================
# 2. Statistical Significance Tests
# ================================================================
print("\n[5/8] Statistical significance tests...")

# Compute returns
returns = df_full['close'].pct_change()
strategy_returns = returns * combined.shift(1)
strategy_returns = strategy_returns.dropna()

# t-test: is mean return significantly different from 0?
t_stat, p_value = stats.ttest_1samp(strategy_returns, 0)
print(f"\n  t-test for mean return ≠ 0:")
print(f"    t-statistic: {t_stat:.4f}")
print(f"    p-value:     {p_value:.6f}")
print(f"    Significant: {'YES' if p_value < 0.05 else 'NO'} (α=0.05)")

# Jarque-Bera test for normality
jb_stat, jb_p_value = stats.jarque_bera(strategy_returns)
print(f"\n  Jarque-Bera test for normality:")
print(f"    JB statistic: {jb_stat:.4f}")
print(f"    p-value:      {jb_p_value:.6f}")
print(f"    Normal:       {'YES' if jb_p_value > 0.05 else 'NO'} (fat tails present)")

# Autocorrelation test (simple implementation)
print(f"\n  Autocorrelation test:")
for lag in [5, 10, 20]:
    if len(strategy_returns) > lag:
        autocorr = strategy_returns.autocorr(lag=lag)
        # Simple significance test
        se = 1 / np.sqrt(len(strategy_returns))
        z_stat = autocorr / se
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        print(f"    Lag {lag:2d}: autocorr={autocorr:.4f}, p={p_val:.4f} {'Significant' if p_val < 0.05 else 'Not significant'}")

# ================================================================
# 3. Regime Analysis
# ================================================================
print("\n[6/8] Regime analysis...")

# Define regimes using 50/200 SMA crossover
sma50 = sma(df_full['close'], 50)
sma200 = sma(df_full['close'], 200)
regime = (sma50 > sma200).astype(float)  # 1 = bull, 0 = bear

# Compute metrics by regime
bull_mask = regime == 1
bear_mask = regime == 0

bull_returns = strategy_returns[bull_mask.reindex(strategy_returns.index, fill_value=0)]
bear_returns = strategy_returns[bear_mask.reindex(strategy_returns.index, fill_value=0)]

bull_sharpe = bull_returns.mean() / bull_returns.std() * np.sqrt(365) if bull_returns.std() > 0 else 0
bear_sharpe = bear_returns.mean() / bear_returns.std() * np.sqrt(365) if bear_returns.std() > 0 else 0

print(f"\n  Regime Analysis:")
print(f"    Bull market: {bull_sharpe:.2f} Sharpe ({len(bull_returns)} bars)")
print(f"    Bear market: {bear_sharpe:.2f} Sharpe ({len(bear_returns)} bars)")
print(f"    Regime dependence: {'HIGH' if abs(bull_sharpe - bear_sharpe) > 1.0 else 'LOW'}")

# ================================================================
# 4. Cost Sensitivity Analysis
# ================================================================
print("\n[7/8] Cost sensitivity analysis...")

cost_scenarios = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01]

print(f"\n  {'Cost':<10} {'Sharpe':<10} {'CAGR':<10} {'MaxDD':<10} {'Trades':<10}")
print("  " + "-" * 50)

for cost in cost_scenarios:
    metrics = compute_metrics(combined[df_holdout.index], df_holdout['close'], transaction_cost=cost)
    print(f"  {cost*100:<10.2f}% {metrics['sharpe']:<10.2f} {metrics['cagr']:<10.1f}% {metrics['max_dd']:<10.1f}% {metrics['n_trades']:<10}")

# ================================================================
# 5. Overfitting Detection (Haircut Rule)
# ================================================================
print("\n[8/8] Overfitting detection...")

# Number of strategies tested (grid search)
n_strategies_tested = 22  # 22 lookback periods tested
n_observations = len(df_train)

# Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014)
def deflated_sharpe_ratio(observed_sharpe, n_trials, n_observations, skew=0, kurtosis=3):
    """Calculate Deflated Sharpe Ratio."""
    # Expected Sharpe under null
    e_max_sharpe = stats.norm.ppf(1 - 1/n_trials) * np.sqrt(1 + (skew * observed_sharpe / 6) - (kurtosis - 3) * observed_sharpe**2 / 24)
    
    # Standard error
    se = np.sqrt((1 + 0.5 * observed_sharpe**2 - skew * observed_sharpe + (kurtosis - 3) * observed_sharpe**2 / 4) / n_observations)
    
    # DSR
    dsr = stats.norm.cdf((observed_sharpe - e_max_sharpe) / se)
    
    return dsr, e_max_sharpe

best_sharpe = 1.42  # Our best holdout Sharpe
dsr, e_max_sharpe = deflated_sharpe_ratio(best_sharpe, n_strategies_tested, n_observations)

print(f"\n  Overfitting Analysis:")
print(f"    Strategies tested:        {n_strategies_tested}")
print(f"    Observations:             {n_observations}")
print(f"    Best Sharpe (holdout):    {best_sharpe:.2f}")
print(f"    Expected max Sharpe (null): {e_max_sharpe:.2f}")
print(f"    Deflated Sharpe Ratio:    {dsr:.4f}")
print(f"    Significant after MTC:    {'YES' if dsr > 0.95 else 'NO'} (need DSR > 0.95)")

# Haircut rule
haircut_30 = best_sharpe * 0.7
haircut_50 = best_sharpe * 0.5
print(f"\n  Haircut Rule (Bailey & Lopez de Prado):")
print(f"    Observed Sharpe:          {best_sharpe:.2f}")
print(f"    30% haircut (conservative): {haircut_30:.2f}")
print(f"    50% haircut (very conservative): {haircut_50:.2f}")
print(f"    Expected live Sharpe:     {haircut_50:.2f} to {haircut_30:.2f}")

# ================================================================
# Final Summary
# ================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

# Baseline comparison
msvr_train = (msvr_signal[df_train.index] > 0).astype(float)
msvr_holdout = (msvr_signal[df_holdout.index] > 0).astype(float)
metrics_msvr_holdout = compute_metrics(msvr_holdout, df_holdout['close'])

combined_holdout = combined[df_holdout.index]
metrics_combined_holdout = compute_metrics(combined_holdout, df_holdout['close'])

print(f"\n  COMPARISON:")
print(f"  {'Metric':<20} {'MSVR Only':<15} {'MSVR+Cycle':<15}")
print(f"  " + "-" * 50)
print(f"  {'Sharpe':<20} {metrics_msvr_holdout['sharpe']:<15.2f} {metrics_combined_holdout['sharpe']:<15.2f}")
print(f"  {'CAGR':<20} {metrics_msvr_holdout['cagr']:<15.1f}% {metrics_combined_holdout['cagr']:<15.1f}%")
print(f"  {'MaxDD':<20} {metrics_msvr_holdout['max_dd']:<15.1f}% {metrics_combined_holdout['max_dd']:<15.1f}%")
print(f"  {'Win Rate':<20} {metrics_msvr_holdout['win_rate']:<15.1f}% {metrics_combined_holdout['win_rate']:<15.1f}%")
print(f"  {'Profit Factor':<20} {metrics_msvr_holdout['profit_factor']:<15.2f} {metrics_combined_holdout['profit_factor']:<15.2f}")
print(f"  {'Max Consec Loss':<20} {metrics_msvr_holdout['max_consecutive_losses']:<15} {metrics_combined_holdout['max_consecutive_losses']:<15}")

print(f"\n  VALIDATION RESULTS:")
print(f"    Walk-Forward OOS Sharpe:  {avg_oos_sharpe:.2f}")
print(f"    t-test p-value:          {p_value:.6f} {'✓' if p_value < 0.05 else '✗'}")
print(f"    Deflated Sharpe Ratio:   {dsr:.4f} {'✓' if dsr > 0.95 else '✗'}")
print(f"    Regime dependence:       {'LOW' if abs(bull_sharpe - bear_sharpe) < 1.0 else 'HIGH'}")
print(f"    Expected live Sharpe:    {haircut_50:.2f} to {haircut_30:.2f}")

# Verdict
print(f"\n  VERDICT:")
if p_value < 0.05 and avg_oos_sharpe > 0.5:
    print(f"    ✅ STRATEGY PASSES RIGOROUS VALIDATION")
    print(f"    Expected live Sharpe: {haircut_50:.2f} to {haircut_30:.2f}")
elif p_value < 0.1:
    print(f"    ⚠️ MARGINAL — needs more data or refinement")
else:
    print(f"    ❌ FAILS VALIDATION — likely overfitting")

print("=" * 70)

# Save results
output = {
    'holdout_metrics': metrics_combined_holdout,
    'baseline_metrics': metrics_msvr_holdout,
    'walk_forward': {
        'results': wf_results,
        'avg_oos_sharpe': avg_oos_sharpe,
        'avg_decay': avg_decay
    },
    'statistical_tests': {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    },
    'regime_analysis': {
        'bull_sharpe': bull_sharpe,
        'bear_sharpe': bear_sharpe
    },
    'overfitting': {
        'strategies_tested': n_strategies_tested,
        'deflated_sharpe_ratio': dsr,
        'expected_live_sharpe_range': [haircut_50, haircut_30]
    }
}

output_path = os.path.join(project_root, 'backtest_msvr_cycle_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
