"""
Validation Report Generator for MTTD Trading System
====================================================

Generates comprehensive system validation report including:
1. Walk-forward summary table
2. ISP comparison (CAGR, trade count, coherence)
3. Risk metrics (max DD, Sharpe, Sortino ratio)
4. Final PASS/FAIL verdict

PASS Requirements:
- Time-coherence ≥ 95%
- Max drawdown < 15%

The report reads from mttd_data.json which contains pre-computed results
from the full pipeline execution.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime


def load_mttd_data(json_path: str) -> Dict:
    """
    Load MTTD system data from JSON file.

    Parameters
    ----------
    json_path : str
        Path to mttd_data.json.

    Returns
    -------
    Dict
        Parsed JSON data.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_isp_benchmark(csv_path: str) -> Dict:
    """
    Load ISP benchmark statistics from CSV.

    Parameters
    ----------
    csv_path : str
        Path to ISP signals CSV file.

    Returns
    -------
    Dict
        ISP benchmark statistics.
    """
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    first_equity = df.iloc[0]['TotalEquity']
    last_equity = df.iloc[-1]['TotalEquity']
    n_years = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days / 365.25

    # Compute CAGR
    if first_equity > 0 and last_equity > 0 and n_years > 0:
        cagr = (last_equity / first_equity) ** (1 / n_years) - 1
    else:
        cagr = 0.0

    # Count trades
    buy_count = int((df['Action'] == 'BUY').sum())
    sell_count = int((df['Action'] == 'SELL').sum())
    total_trades = buy_count + sell_count

    # Regime distribution
    regime_dist = df['Regime'].value_counts().to_dict()

    return {
        'cagr_pct': round(cagr * 100, 2),
        'total_trades': total_trades,
        'buy_trades': buy_count,
        'sell_trades': sell_count,
        'years': round(n_years, 2),
        'first_equity': round(first_equity, 2),
        'last_equity': round(last_equity, 2),
        'regime_distribution': regime_dist,
        'data_start': df['Date'].min().strftime('%Y-%m-%d'),
        'data_end': df['Date'].max().strftime('%Y-%m-%d'),
    }


def compute_system_cagr(candles: List[Dict], positions: List[Dict],
                        initial_capital: float = 100000.0,
                        commission_rate: float = 0.001) -> Dict:
    """
    Compute system CAGR from candle data and position signals.

    Parameters
    ----------
    candles : List[Dict]
        Price candles with 'time' and 'close' fields.

    positions : List[Dict]
        Position signals with 'time' and 'value' fields (0.0 or 1.0).

    initial_capital : float
        Starting capital.

    commission_rate : float
        Transaction commission rate.

    Returns
    -------
    Dict
        CAGR and related metrics.
    """
    if not candles or not positions:
        return {'cagr_pct': 0.0, 'total_return_pct': 0.0, 'years': 0.0, 'final_equity': initial_capital}

    # Build aligned price and position series
    price_dict = {c['time']: float(c['close']) for c in candles}
    pos_dict = {p['time']: float(p['value']) for p in positions}

    # Get common dates
    common_dates = sorted(set(price_dict.keys()) & set(pos_dict.keys()))

    if len(common_dates) < 2:
        return {'cagr_pct': 0.0, 'total_return_pct': 0.0, 'years': 0.0, 'final_equity': initial_capital}

    prices = np.array([price_dict[d] for d in common_dates])
    pos = np.array([pos_dict[d] for d in common_dates])

    # Simulate trading
    cash = initial_capital
    btc = 0.0
    prev_target = 0.0

    for i, (price, target) in enumerate(zip(prices, pos)):
        if target != prev_target:
            total_equity = cash + btc * price
            target_btc_val = total_equity * target
            current_btc_val = btc * price
            trade_val = target_btc_val - current_btc_val

            if trade_val > 0:
                comm = abs(trade_val) * commission_rate
                btc_change = (trade_val - comm) / price
                btc += btc_change
                cash -= trade_val
            elif trade_val < 0:
                comm = abs(trade_val) * commission_rate
                btc_change = trade_val / price
                btc += btc_change
                cash += abs(trade_val) - comm

            prev_target = target

    final_equity = cash + btc * prices[-1]
    total_return = (final_equity - initial_capital) / initial_capital

    # Compute years
    start_date = pd.Timestamp(common_dates[0])
    end_date = pd.Timestamp(common_dates[-1])
    n_years = (end_date - start_date).days / 365.25

    # Compute CAGR
    if n_years > 0 and final_equity > 0 and initial_capital > 0:
        cagr = (final_equity / initial_capital) ** (1 / n_years) - 1
    else:
        cagr = 0.0

    return {
        'cagr_pct': round(cagr * 100, 2),
        'total_return_pct': round(total_return * 100, 2),
        'years': round(n_years, 2),
        'final_equity': round(final_equity, 2),
        'initial_capital': initial_capital,
    }


def compute_sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Compute Sortino ratio from daily returns.

    Sortino ratio = (mean_return - risk_free_rate) / downside_deviation

    Parameters
    ----------
    daily_returns : pd.Series
        Daily return series.

    risk_free_rate : float
        Daily risk-free rate (default 0).

    Returns
    -------
    float
        Sortino ratio.
    """
    if len(daily_returns) < 10:
        return 0.0

    mean_return = daily_returns.mean()
    downside_returns = daily_returns[daily_returns < risk_free_rate]

    if len(downside_returns) < 2:
        return float('inf') if mean_return > risk_free_rate else 0.0

    downside_std = downside_returns.std()

    if downside_std == 0:
        return float('inf') if mean_return > risk_free_rate else 0.0

    sortino = (mean_return - risk_free_rate) / downside_std * np.sqrt(365)
    return round(sortino, 4)


def compute_risk_metrics_from_candles(
    candles: List[Dict],
    positions: List[Dict],
    initial_capital: float = 100000.0,
    commission_rate: float = 0.001
) -> Dict:
    """
    Compute comprehensive risk metrics from candle data and positions.

    Parameters
    ----------
    candles : List[Dict]
        Price candles.

    positions : List[Dict]
        Position signals.

    initial_capital : float
        Starting capital.

    commission_rate : float
        Commission rate.

    Returns
    -------
    Dict
        Risk metrics including max DD, Sharpe, Sortino.
    """
    if not candles or not positions:
        return {
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'daily_volatility_pct': 0.0,
            'calmar_ratio': 0.0,
        }

    # Build aligned price and position series
    price_dict = {c['time']: float(c['close']) for c in candles}
    pos_dict = {p['time']: float(p['value']) for p in positions}

    common_dates = sorted(set(price_dict.keys()) & set(pos_dict.keys()))

    if len(common_dates) < 10:
        return {
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'daily_volatility_pct': 0.0,
            'calmar_ratio': 0.0,
        }

    prices = np.array([price_dict[d] for d in common_dates])
    pos = np.array([pos_dict[d] for d in common_dates])

    # Compute daily returns and equity curve
    daily_prices = pd.Series(prices, index=pd.to_datetime(common_dates))
    daily_positions = pd.Series(pos, index=pd.to_datetime(common_dates))

    daily_returns = daily_prices.pct_change().fillna(0.0)
    strategy_returns = daily_positions.shift(1).fillna(0.0) * daily_returns

    # Simulate with commissions for accurate equity curve
    cash = initial_capital
    btc = 0.0
    prev_target = 0.0
    equity_list = []

    for price, target in zip(prices, pos):
        if target != prev_target:
            total_equity = cash + btc * price
            target_btc_val = total_equity * target
            current_btc_val = btc * price
            trade_val = target_btc_val - current_btc_val

            if trade_val > 0:
                comm = abs(trade_val) * commission_rate
                btc_change = (trade_val - comm) / price
                btc += btc_change
                cash -= trade_val
            elif trade_val < 0:
                comm = abs(trade_val) * commission_rate
                btc_change = trade_val / price
                btc += btc_change
                cash += abs(trade_val) - comm

            prev_target = target

        eq = cash + btc * price
        equity_list.append(eq)

    equity = pd.Series(equity_list, index=pd.to_datetime(common_dates))

    # Max drawdown
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min()) * 100.0

    # Sharpe ratio
    strat_returns = equity.pct_change().fillna(0.0)
    mean_return = strat_returns.mean()
    std_return = strat_returns.std()
    sharpe = (mean_return / std_return) * np.sqrt(365) if std_return > 0 else 0.0

    # Sortino ratio
    sortino = compute_sortino_ratio(strat_returns)

    # Daily volatility
    daily_vol = float(std_return) * 100.0

    # Calmar ratio (annualized return / max drawdown)
    n_years = (pd.Timestamp(common_dates[-1]) - pd.Timestamp(common_dates[0])).days / 365.25
    if n_years > 0 and equity.iloc[-1] > 0:
        annualized_return = (equity.iloc[-1] / initial_capital) ** (1 / n_years) - 1
    else:
        annualized_return = 0.0

    calmar = annualized_return / (abs(max_dd) / 100.0) if abs(max_dd) > 0 else 0.0

    # Count trades
    pos_diff = np.diff(pos, prepend=pos[0])
    n_entries = int(np.sum(pos_diff == 1))
    n_exits = int(np.sum(pos_diff == -1))
    n_trades = min(n_entries, n_exits)

    return {
        'max_drawdown_pct': round(max_dd, 2),
        'sharpe_ratio': round(sharpe, 4),
        'sortino_ratio': round(sortino, 4),
        'daily_volatility_pct': round(daily_vol, 4),
        'calmar_ratio': round(calmar, 4),
        'annualized_return_pct': round(annualized_return * 100, 2),
        'n_trades': n_trades,
        'n_entries': n_entries,
        'n_exits': n_exits,
        'final_equity': round(float(equity.iloc[-1]), 2),
        'years': round(n_years, 2),
    }


def generate_validation_report(
    mttd_data: Dict,
    isp_csv_path: str,
    coherence_threshold: float = 95.0,
    max_dd_threshold: float = 15.0
) -> Dict:
    """
    Generate comprehensive validation report.

    Parameters
    ----------
    mttd_data : Dict
        MTTD system data from mttd_data.json.

    isp_csv_path : str
        Path to ISP signals CSV.

    coherence_threshold : float
        Minimum time-coherence for PASS.

    max_dd_threshold : float
        Maximum drawdown (absolute) for PASS.

    Returns
    -------
    Dict
        Complete validation report with all metrics and PASS/FAIL verdict.
    """
    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pass_thresholds': {
            'coherence_min_pct': coherence_threshold,
            'max_dd_max_pct': max_dd_threshold,
        },
    }

    # ================================================================
    # 1. ISP Benchmark Comparison
    # ================================================================
    try:
        isp_benchmark = load_isp_benchmark(isp_csv_path)
    except Exception as e:
        isp_benchmark = {'error': str(e)}

    report['isp_benchmark'] = isp_benchmark

    # ================================================================
    # 2. System Performance
    # ================================================================
    candles = mttd_data.get('candles', [])
    aggregate = mttd_data.get('aggregate', {})
    positions = aggregate.get('signals', [])

    system_perf = compute_system_cagr(candles, positions)
    report['system_performance'] = system_perf

    # Compare with ISP
    if 'error' not in isp_benchmark:
        report['isp_comparison'] = {
            'isp_cagr_pct': isp_benchmark['cagr_pct'],
            'system_cagr_pct': system_perf['cagr_pct'],
            'cagr_diff_pct': round(system_perf['cagr_pct'] - isp_benchmark['cagr_pct'], 2),
            'isp_trades': isp_benchmark['total_trades'],
            'system_trades': system_perf.get('n_trades', 0),
        }

    # ================================================================
    # 3. Risk Metrics
    # ================================================================
    risk_metrics = compute_risk_metrics_from_candles(candles, positions)
    report['risk_metrics'] = risk_metrics

    # Also include risk management results from pipeline
    risk_mgmt = mttd_data.get('risk_management', {})
    report['risk_management'] = {
        'max_dd_pause_configured_pct': risk_mgmt.get('max_dd_pause_pct', 0.15),
        'pause_days': risk_mgmt.get('pause_days', 20),
        'n_pause_bars': risk_mgmt.get('n_pause_bars', 0),
        'max_drawdown_pct': risk_mgmt.get('max_drawdown_pct', risk_metrics['max_drawdown_pct']),
        'pct_in_position': risk_mgmt.get('pct_in_position', 0.0),
    }

    # ================================================================
    # 4. Coherence Metrics
    # ================================================================
    coherence = mttd_data.get('coherence', {})
    if coherence and 'time_coherence_pct' in coherence:
        report['coherence'] = {
            'time_coherence_pct': coherence['time_coherence_pct'],
            'n_agree': coherence.get('n_agree', 0),
            'n_disagree': coherence.get('n_disagree', 0),
            'n_total': coherence.get('n_total', 0),
            'both_in_pct': coherence.get('both_in_pct', 0.0),
            'both_out_pct': coherence.get('both_out_pct', 0.0),
            'mttd_in_isp_out_pct': coherence.get('mttd_in_isp_out_pct', 0.0),
            'mttd_out_isp_in_pct': coherence.get('mttd_out_isp_in_pct', 0.0),
            'verdict': coherence.get('verdict', {}),
        }
        # Timing error
        timing = coherence.get('timing_error', {})
        if timing:
            report['coherence']['timing_error'] = {
                'avg_entry_timing_error_days': timing.get('avg_entry_timing_error_days'),
                'avg_exit_timing_error_days': timing.get('avg_exit_timing_error_days'),
                'mttd_n_trades': timing.get('mttd_n_trades', 0),
                'isp_n_trades': timing.get('isp_n_trades', 0),
            }
        # Return correlation
        ret_corr = coherence.get('return_correlation', {})
        if ret_corr and not ret_corr.get('skipped') and not ret_corr.get('error'):
            report['coherence']['return_correlation'] = {
                'pearson_corr': ret_corr.get('pearson_corr'),
                'spearman_corr': ret_corr.get('spearman_corr'),
                'n_common_days': ret_corr.get('n_common_days', 0),
            }
        # Trade comparison
        trade_comp = coherence.get('trade_comparison', {})
        if trade_comp:
            report['coherence']['trade_comparison'] = trade_comp
    else:
        report['coherence'] = {'error': 'No coherence data available'}

    # ================================================================
    # 5. Walk-Forward Summary Table
    # ================================================================
    walk_forward = mttd_data.get('walk_forward', {})
    if walk_forward and 'cycles' in walk_forward:
        wf_summary = {
            'cycles_completed': walk_forward.get('cycles_completed', 0),
            'min_cycles_met': walk_forward.get('min_cycles_met', False),
            'verdict': walk_forward.get('verdict', {}),
            'in_sample_coherence': walk_forward.get('in_sample_coherence', {}),
            'out_of_sample_coherence': walk_forward.get('out_of_sample_coherence', {}),
            'coherence_gap': walk_forward.get('coherence_gap', {}),
            'returns': walk_forward.get('returns', {}),
            'risk': walk_forward.get('risk', {}),
            'overfitting': walk_forward.get('overfitting', {}),
        }

        # Build cycle table for display
        cycles = walk_forward.get('cycles', [])
        cycle_table = []
        for cycle in cycles:
            cycle_table.append({
                'cycle': cycle.get('cycle'),
                'train_period': f"{cycle.get('train_start', '')} to {cycle.get('train_end', '')}",
                'test_period': f"{cycle.get('test_start', '')} to {cycle.get('test_end', '')}",
                'threshold': cycle.get('calibrated_threshold', 0.0),
                'is_coherence_pct': cycle.get('in_sample_coherence_pct', 0.0),
                'oos_coherence_pct': cycle.get('out_of_sample_coherence_pct', 0.0),
                'coherence_gap_pct': cycle.get('coherence_gap_pct', 0.0),
                'return_pct': cycle.get('test_return_pct', 0.0),
                'annualized_return_pct': cycle.get('annualized_return_pct', 0.0),
                'max_drawdown_pct': cycle.get('max_drawdown_pct', 0.0),
                'sharpe_ratio': cycle.get('sharpe_ratio', 0.0),
            })

        wf_summary['cycle_table'] = cycle_table
        report['walk_forward'] = wf_summary
    else:
        report['walk_forward'] = {'error': 'No walk-forward data available'}

    # ================================================================
    # 6. Ensemble Configuration
    # ================================================================
    ensemble = mttd_data.get('ensemble', {})
    report['ensemble_config'] = {
        'threshold': ensemble.get('threshold', 0.0),
        'ema_length': ensemble.get('ema_length', 5),
        'n_indicators': ensemble.get('n_indicators', 0),
        'pct_in_position': ensemble.get('pct_in_position', 0.0),
        'n_trades': ensemble.get('n_trades', 0),
        'look_ahead_verification': ensemble.get('look_ahead_verification', False),
    }

    calibration = mttd_data.get('calibration', {})
    if calibration and 'optimal_threshold' in calibration:
        report['ensemble_config']['calibration'] = {
            'optimal_threshold': calibration.get('optimal_threshold', 0.0),
            'max_coherence_pct': calibration.get('max_coherence_pct', 0.0),
            'train_period': f"{calibration.get('train_start_date', '')} to {calibration.get('train_end_date', '')}",
            'n_train_bars': calibration.get('n_train_bars', 0),
        }

    # ================================================================
    # 7. PASS/FAIL Verdict
    # ================================================================
    coherence_pct = report.get('coherence', {}).get('time_coherence_pct', 0.0)
    max_dd = abs(report.get('risk_metrics', {}).get('max_drawdown_pct', 0.0))

    coherence_pass = coherence_pct >= coherence_threshold
    dd_pass = max_dd < max_dd_threshold

    # Additional checks
    wf_data = report.get('walk_forward', {})
    wf_passed = wf_data.get('verdict', {}).get('passed', False)
    wf_cycles_met = wf_data.get('min_cycles_met', False)

    # Overall verdict
    overall_pass = coherence_pass and dd_pass

    report['verdict'] = {
        'overall_pass': overall_pass,
        'criteria': {
            'coherence': {
                'required': f'>= {coherence_threshold}%',
                'actual': f'{coherence_pct:.2f}%',
                'pass': coherence_pass,
            },
            'max_drawdown': {
                'required': f'< {max_dd_threshold}%',
                'actual': f'{max_dd:.2f}%',
                'pass': dd_pass,
            },
            'walk_forward': {
                'required': '>= 3 cycles',
                'actual': f"{wf_data.get('cycles_completed', 0)} cycles",
                'pass': wf_cycles_met,
            },
            'walk_forward_coherence': {
                'required': f'OOS coherence >= {coherence_threshold}%',
                'actual': f"{wf_data.get('out_of_sample_coherence', {}).get('mean', 0):.2f}%",
                'pass': wf_passed,
            },
        },
        'summary': (
            f"{'PASS' if overall_pass else 'FAIL'}: "
            f"Coherence={coherence_pct:.2f}% "
            f"({'PASS' if coherence_pass else 'FAIL'}>={coherence_threshold}%), "
            f"MaxDD={max_dd:.2f}% "
            f"({'PASS' if dd_pass else 'FAIL'}<{max_dd_threshold}%)"
        ),
    }

    return report


def format_validation_report(report: Dict) -> str:
    """
    Format validation report into human-readable text.

    Parameters
    ----------
    report : Dict
        Complete validation report from generate_validation_report().

    Returns
    -------
    str
        Formatted report string.
    """
    lines = []

    # Header
    lines.append("=" * 80)
    lines.append("MTTD TRADING SYSTEM — VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {report.get('generated_at', 'N/A')}")

    # ================================================================
    # PASS/FAIL VERDICT
    # ================================================================
    verdict = report.get('verdict', {})
    overall_pass = verdict.get('overall_pass', False)
    verdict_symbol = "✓ PASS" if overall_pass else "✗ FAIL"

    lines.append(f"\n{'=' * 80}")
    lines.append(f"FINAL VERDICT: {verdict_symbol}")
    lines.append(f"{'=' * 80}")
    lines.append(f"\n{verdict.get('summary', 'No summary available')}")

    lines.append(f"\nCriteria:")
    for criterion_name, criterion in verdict.get('criteria', {}).items():
        status = "✓" if criterion.get('pass', False) else "✗"
        lines.append(
            f"  {status} {criterion_name}: "
            f"{criterion.get('actual', 'N/A')} "
            f"(required: {criterion.get('required', 'N/A')})"
        )

    # ================================================================
    # ISP BENCHMARK COMPARISON
    # ================================================================
    lines.append(f"\n{'─' * 80}")
    lines.append("ISP BENCHMARK COMPARISON")
    lines.append(f"{'─' * 80}")

    isp = report.get('isp_benchmark', {})
    if 'error' not in isp:
        lines.append(f"  ISP CAGR:           {isp.get('cagr_pct', 0):.2f}%")
        lines.append(f"  ISP Total Trades:   {isp.get('total_trades', 0)}")
        lines.append(f"  ISP Years:          {isp.get('years', 0):.2f}")
        lines.append(f"  ISP Data Range:     {isp.get('data_start', '')} to {isp.get('data_end', '')}")
        lines.append(f"  ISP Regimes:        {isp.get('regime_distribution', {})}")
    else:
        lines.append(f"  Error loading ISP: {isp.get('error', 'Unknown')}")

    # System performance
    sys_perf = report.get('system_performance', {})
    lines.append(f"\n  System CAGR:        {sys_perf.get('cagr_pct', 0):.2f}%")
    lines.append(f"  System Total Return:{sys_perf.get('total_return_pct', 0):.2f}%")
    lines.append(f"  System Years:       {sys_perf.get('years', 0):.2f}")
    lines.append(f"  System Final Equity:{sys_perf.get('final_equity', 0):,.2f}")

    # Comparison
    comparison = report.get('isp_comparison', {})
    if comparison:
        lines.append(f"\n  CAGR Difference:    {comparison.get('cagr_diff_pct', 0):.2f}%")
        lines.append(f"  ISP Trades:         {comparison.get('isp_trades', 0)}")
        lines.append(f"  System Trades:      {comparison.get('system_trades', 0)}")

    # ================================================================
    # RISK METRICS
    # ================================================================
    lines.append(f"\n{'─' * 80}")
    lines.append("RISK METRICS")
    lines.append(f"{'─' * 80}")

    risk = report.get('risk_metrics', {})
    lines.append(f"  Max Drawdown:       {risk.get('max_drawdown_pct', 0):.2f}%")
    lines.append(f"  Sharpe Ratio:       {risk.get('sharpe_ratio', 0):.4f}")
    lines.append(f"  Sortino Ratio:      {risk.get('sortino_ratio', 0):.4f}")
    lines.append(f"  Daily Volatility:   {risk.get('daily_volatility_pct', 0):.4f}%")
    lines.append(f"  Calmar Ratio:       {risk.get('calmar_ratio', 0):.4f}")
    lines.append(f"  Annualized Return:  {risk.get('annualized_return_pct', 0):.2f}%")
    lines.append(f"  Final Equity:       {risk.get('final_equity', 0):,.2f}")
    lines.append(f"  Total Trades:       {risk.get('n_trades', 0)}")

    # Risk management
    risk_mgmt = report.get('risk_management', {})
    lines.append(f"\n  Risk Management:")
    lines.append(f"    Max DD Pause:     {risk_mgmt.get('max_dd_pause_configured_pct', 0)*100:.1f}%")
    lines.append(f"    Pause Days:       {risk_mgmt.get('pause_days', 0)}")
    lines.append(f"    Bars in Pause:    {risk_mgmt.get('n_pause_bars', 0)}")
    lines.append(f"    % Time in Mkt:    {risk_mgmt.get('pct_in_position', 0):.2f}%")

    # ================================================================
    # COHERENCE METRICS
    # ================================================================
    lines.append(f"\n{'─' * 80}")
    lines.append("COHERENCE WITH ISP BENCHMARK")
    lines.append(f"{'─' * 80}")

    coh = report.get('coherence', {})
    if 'error' not in coh:
        coh_pct = coh.get('time_coherence_pct', 0)
        coh_status = "✓ PASS" if coh_pct >= 95.0 else "✗ FAIL"
        lines.append(f"  Time Coherence:     {coh_pct:.2f}% {coh_status}")
        lines.append(f"  Bars in Agreement:  {coh.get('n_agree', 0)} / {coh.get('n_total', 0)}")
        lines.append(f"  Bars Disagreement:  {coh.get('n_disagree', 0)}")
        lines.append(f"  Both In Market:     {coh.get('both_in_pct', 0):.2f}%")
        lines.append(f"  Both Out of Market: {coh.get('both_out_pct', 0):.2f}%")
        lines.append(f"  MTTD In, ISP Out:   {coh.get('mttd_in_isp_out_pct', 0):.2f}%")
        lines.append(f"  MTTD Out, ISP In:   {coh.get('mttd_out_isp_in_pct', 0):.2f}%")

        # Timing error
        timing = coh.get('timing_error', {})
        if timing:
            lines.append(f"\n  Timing Error:")
            lines.append(f"    Avg Entry Error:  {timing.get('avg_entry_timing_error_days', 'N/A')} days")
            lines.append(f"    Avg Exit Error:   {timing.get('avg_exit_timing_error_days', 'N/A')} days")
            lines.append(f"    MTTD Trades:      {timing.get('mttd_n_trades', 0)}")
            lines.append(f"    ISP Trades:       {timing.get('isp_n_trades', 0)}")

        # Return correlation
        ret_corr = coh.get('return_correlation', {})
        if ret_corr and not ret_corr.get('skipped') and not ret_corr.get('error'):
            lines.append(f"\n  Return Correlation:")
            lines.append(f"    Pearson:          {ret_corr.get('pearson_corr', 'N/A')}")
            lines.append(f"    Spearman:         {ret_corr.get('spearman_corr', 'N/A')}")
            lines.append(f"    Common Days:      {ret_corr.get('n_common_days', 0)}")
    else:
        lines.append(f"  Error: {coh.get('error', 'Unknown')}")

    # ================================================================
    # WALK-FORWARD SUMMARY TABLE
    # ================================================================
    lines.append(f"\n{'─' * 80}")
    lines.append("WALK-FORWARD VALIDATION SUMMARY")
    lines.append(f"{'─' * 80}")

    wf = report.get('walk_forward', {})
    if 'error' not in wf:
        wf_verdict = wf.get('verdict', {})
        wf_status = "✓ PASS" if wf_verdict.get('passed', False) else "✗ FAIL"
        lines.append(f"  Walk-Forward Verdict: {wf_status}")
        lines.append(f"  {wf_verdict.get('reason', 'N/A')}")
        lines.append(f"  Cycles Completed:  {wf.get('cycles_completed', 0)}")
        lines.append(f"  Min Cycles Met:    {'Yes' if wf.get('min_cycles_met', False) else 'No'}")

        # In-sample coherence
        is_coh = wf.get('in_sample_coherence', {})
        if is_coh:
            lines.append(f"\n  In-Sample Coherence (Training):")
            lines.append(f"    Mean: {is_coh.get('mean', 0):.2f}%  "
                         f"Min: {is_coh.get('min', 0):.2f}%  "
                         f"Max: {is_coh.get('max', 0):.2f}%  "
                         f"Std: {is_coh.get('std', 0):.2f}%")

        # Out-of-sample coherence
        oos_coh = wf.get('out_of_sample_coherence', {})
        if oos_coh:
            lines.append(f"\n  Out-of-Sample Coherence (Test):")
            lines.append(f"    Mean: {oos_coh.get('mean', 0):.2f}%  "
                         f"Min: {oos_coh.get('min', 0):.2f}%  "
                         f"Max: {oos_coh.get('max', 0):.2f}%  "
                         f"Std: {oos_coh.get('std', 0):.2f}%")

        # Returns
        ret = wf.get('returns', {})
        if ret:
            lines.append(f"\n  Returns (Test Periods):")
            lines.append(f"    Mean Test Return:     {ret.get('mean_test_return_pct', 0):.2f}%")
            lines.append(f"    Mean Annual Return:   {ret.get('mean_annual_return_pct', 0):.2f}%")
            lines.append(f"    Best Test Return:     {ret.get('best_test_return_pct', 0):.2f}%")
            lines.append(f"    Worst Test Return:    {ret.get('worst_test_return_pct', 0):.2f}%")

        # Risk
        wf_risk = wf.get('risk', {})
        if wf_risk:
            lines.append(f"\n  Risk (Test Periods):")
            lines.append(f"    Mean Max Drawdown:    {wf_risk.get('mean_max_drawdown_pct', 0):.2f}%")
            lines.append(f"    Worst Max Drawdown:   {wf_risk.get('worst_max_drawdown_pct', 0):.2f}%")
            lines.append(f"    Mean Sharpe Ratio:    {wf_risk.get('mean_sharpe', 0):.4f}")

        # Cycle table
        cycle_table = wf.get('cycle_table', [])
        if cycle_table:
            lines.append(f"\n  {'Cycle':<7} {'Test Period':<25} {'IS Coh%':<10} {'OOS Coh%':<10} "
                         f"{'Return%':<10} {'MaxDD%':<10} {'Sharpe':<8}")
            lines.append(f"  {'─' * 80}")
            for cycle in cycle_table:
                lines.append(
                    f"  {cycle['cycle']:<7} "
                    f"{cycle['test_period']:<25} "
                    f"{cycle['is_coherence_pct']:<10.2f} "
                    f"{cycle['oos_coherence_pct']:<10.2f} "
                    f"{cycle['return_pct']:<10.2f} "
                    f"{cycle['max_drawdown_pct']:<10.2f} "
                    f"{cycle['sharpe_ratio']:<8.4f}"
                )

        # Overfitting analysis
        overfit = wf.get('overfitting', {})
        if overfit:
            lines.append(f"\n  Overfitting Analysis:")
            lines.append(f"    Mean Coherence Gap:  {overfit.get('mean_coherence_gap', 0):.2f}%")
            lines.append(f"    Coherence Stability: {overfit.get('coherence_stability', 0):.2f}% (std)")
            lines.append(f"    Avg Overfit Ratio:   {overfit.get('avg_overfit_ratio', 0):.4f} (OOS/IS)")
    else:
        lines.append(f"  Error: {wf.get('error', 'Unknown')}")

    # ================================================================
    # ENSEMBLE CONFIGURATION
    # ================================================================
    lines.append(f"\n{'─' * 80}")
    lines.append("ENSEMBLE CONFIGURATION")
    lines.append(f"{'─' * 80}")

    ens = report.get('ensemble_config', {})
    lines.append(f"  Threshold:          {ens.get('threshold', 0):.4f}")
    lines.append(f"  EMA Length:         {ens.get('ema_length', 5)}")
    lines.append(f"  N Indicators:       {ens.get('n_indicators', 0)}")
    lines.append(f"  % Time in Position: {ens.get('pct_in_position', 0):.2f}%")
    lines.append(f"  Total Trades:       {ens.get('n_trades', 0)}")
    lines.append(f"  Look-Ahead Check:   {'PASS' if ens.get('look_ahead_verification', False) else 'FAIL'}")

    cal = ens.get('calibration', {})
    if cal:
        lines.append(f"\n  Calibration:")
        lines.append(f"    Optimal Threshold: {cal.get('optimal_threshold', 0):.4f}")
        lines.append(f"    Max Coherence:     {cal.get('max_coherence_pct', 0):.2f}%")
        lines.append(f"    Train Period:      {cal.get('train_period', 'N/A')}")
        lines.append(f"    Train Bars:        {cal.get('n_train_bars', 0)}")

    # Footer
    lines.append(f"\n{'=' * 80}")
    lines.append("END OF VALIDATION REPORT")
    lines.append(f"{'=' * 80}")

    return "\n".join(lines)


def run_validation(
    mttd_json_path: str = None,
    isp_csv_path: str = None,
    output_path: str = None,
    coherence_threshold: float = 95.0,
    max_dd_threshold: float = 15.0
) -> Tuple[Dict, str]:
    """
    Run full validation and generate report.

    Parameters
    ----------
    mttd_json_path : str or None
        Path to mttd_data.json. If None, uses default location.

    isp_csv_path : str or None
        Path to ISP signals CSV. If None, uses default location.

    output_path : str or None
        Path to save report. If None, saves to mttd/validation_report.txt.

    coherence_threshold : float
        Minimum coherence for PASS.

    max_dd_threshold : float
        Maximum drawdown for PASS.

    Returns
    -------
    Tuple[Dict, str]
        (report_dict, formatted_report_text)
    """
    # Default paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if mttd_json_path is None:
        mttd_json_path = os.path.join(project_root, 'mttd', 'mttd_data.json')
    if isp_csv_path is None:
        isp_csv_path = os.path.join(project_root, 'isp-signals-btcusd-2026-06-13.csv')
    if output_path is None:
        output_path = os.path.join(project_root, 'mttd', 'validation_report.txt')

    print("=" * 80)
    print("MTTD VALIDATION REPORT GENERATOR")
    print("=" * 80)

    # Load data
    print(f"\nLoading MTTD data from: {mttd_json_path}")
    mttd_data = load_mttd_data(mttd_json_path)

    print(f"Loading ISP benchmark from: {isp_csv_path}")

    # Generate report
    print("\nGenerating validation report...")
    report = generate_validation_report(
        mttd_data,
        isp_csv_path,
        coherence_threshold=coherence_threshold,
        max_dd_threshold=max_dd_threshold
    )

    # Format report
    formatted = format_validation_report(report)

    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(formatted)
    print(f"\nReport saved to: {output_path}")

    # Print report
    print("\n" + formatted)

    # Also save JSON version
    json_output_path = output_path.replace('.txt', '.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report saved to: {json_output_path}")

    return report, formatted


def run_unit_tests():
    """
    Run unit tests for the validation report module.

    Returns
    -------
    bool
        True if all tests pass.
    """
    print("=" * 70)
    print("Running Validation Report Unit Tests")
    print("=" * 70)

    all_passed = True

    # Test 1: Sortino ratio computation
    print("\nTest 1: Sortino ratio computation")
    try:
        np.random.seed(42)
        # Use returns with clear positive trend so Sortino is positive
        returns = pd.Series([0.005 + np.random.normal(0, 0.005) for _ in range(100)])
        sortino = compute_sortino_ratio(returns)
        assert isinstance(sortino, float), f"Sortino should be float, got {type(sortino)}"
        print(f"  ✓ Sortino ratio computed: {sortino:.4f}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 2: CAGR computation
    print("\nTest 2: CAGR computation")
    try:
        candles = [
            {'time': '2020-01-01', 'close': 100.0},
            {'time': '2021-01-01', 'close': 200.0},
        ]
        positions = [
            {'time': '2020-01-01', 'value': 1.0},
            {'time': '2021-01-01', 'value': 1.0},
        ]
        result = compute_system_cagr(candles, positions, initial_capital=10000)
        assert result['cagr_pct'] > 0, f"CAGR should be positive, got {result['cagr_pct']}"
        assert abs(result['total_return_pct'] - 100.0) < 1.0, \
            f"Total return should be ~100%, got {result['total_return_pct']}"
        print(f"  ✓ CAGR computed: {result['cagr_pct']:.2f}%, Total: {result['total_return_pct']:.2f}%")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 3: Risk metrics computation
    print("\nTest 3: Risk metrics computation")
    try:
        candles = [{'time': f'2020-01-{i+1:02d}', 'close': 100.0 * (1.01 ** i)} for i in range(30)]
        positions = [{'time': f'2020-01-{i+1:02d}', 'value': 1.0} for i in range(30)]
        risk = compute_risk_metrics_from_candles(candles, positions)
        assert 'max_drawdown_pct' in risk
        assert 'sharpe_ratio' in risk
        assert 'sortino_ratio' in risk
        print(f"  ✓ Risk metrics computed: MaxDD={risk['max_drawdown_pct']:.2f}%, "
              f"Sharpe={risk['sharpe_ratio']:.4f}, Sortino={risk['sortino_ratio']:.4f}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 4: Report generation with mock data
    print("\nTest 4: Report generation with mock data")
    try:
        mock_mttd_data = {
            'candles': [{'time': f'2020-01-{i+1:02d}', 'close': 100.0 * (1.005 ** i)} for i in range(30)],
            'aggregate': {
                'signals': [{'time': f'2020-01-{i+1:02d}', 'value': 1.0 if i < 20 else 0.0} for i in range(30)]
            },
            'ensemble': {'threshold': 0.0, 'ema_length': 5, 'n_indicators': 5, 'pct_in_position': 66.67, 'n_trades': 1, 'look_ahead_verification': True},
            'calibration': {'optimal_threshold': 0.0, 'max_coherence_pct': 95.0, 'train_start_date': '2020-01-01', 'train_end_date': '2020-01-30', 'n_train_bars': 30},
            'coherence': {
                'time_coherence_pct': 96.0,
                'n_agree': 28, 'n_disagree': 2, 'n_total': 30,
                'both_in_pct': 50.0, 'both_out_pct': 46.67,
                'mttd_in_isp_out_pct': 1.67, 'mttd_out_isp_in_pct': 1.67,
                'verdict': {'passed': True, 'threshold': 95.0, 'actual_coherence': 96.0, 'reason': 'PASS'},
                'timing_error': {'avg_entry_timing_error_days': 1.0, 'avg_exit_timing_error_days': 2.0, 'mttd_n_trades': 5, 'isp_n_trades': 5},
                'return_correlation': {'pearson_corr': 0.9, 'spearman_corr': 0.85, 'n_common_days': 30},
                'trade_comparison': {'mttd_trades': 5, 'isp_trades': 5, 'trade_count_ratio': 1.0},
            },
            'walk_forward': {
                'cycles_completed': 3, 'min_cycles_met': True,
                'verdict': {'passed': True, 'reason': 'All criteria met'},
                'in_sample_coherence': {'mean': 96.0, 'min': 94.0, 'max': 98.0, 'std': 2.0},
                'out_of_sample_coherence': {'mean': 95.5, 'min': 93.0, 'max': 97.0, 'std': 2.0},
                'coherence_gap': {'mean': 0.5, 'max': 1.0},
                'returns': {'mean_test_return_pct': 50.0, 'mean_annual_return_pct': 50.0, 'best_test_return_pct': 80.0, 'worst_test_return_pct': 20.0},
                'risk': {'mean_max_drawdown_pct': -10.0, 'worst_max_drawdown_pct': -15.0, 'mean_sharpe': 1.5},
                'overfitting': {'mean_coherence_gap': 0.5, 'coherence_stability': 2.0, 'avg_overfit_ratio': 0.99},
                'cycles': [
                    {'cycle': 1, 'train_start': '2020-01-01', 'train_end': '2020-06-30', 'test_start': '2020-07-01', 'test_end': '2020-12-31', 'calibrated_threshold': 0.0, 'in_sample_coherence_pct': 96.0, 'out_of_sample_coherence_pct': 95.0, 'coherence_gap_pct': 1.0, 'test_return_pct': 50.0, 'annualized_return_pct': 50.0, 'max_drawdown_pct': -10.0, 'sharpe_ratio': 1.5},
                    {'cycle': 2, 'train_start': '2020-01-01', 'train_end': '2020-12-31', 'test_start': '2021-01-01', 'test_end': '2021-06-30', 'calibrated_threshold': 0.0, 'in_sample_coherence_pct': 97.0, 'out_of_sample_coherence_pct': 96.0, 'coherence_gap_pct': 1.0, 'test_return_pct': 60.0, 'annualized_return_pct': 60.0, 'max_drawdown_pct': -8.0, 'sharpe_ratio': 1.8},
                    {'cycle': 3, 'train_start': '2020-01-01', 'train_end': '2021-06-30', 'test_start': '2021-07-01', 'test_end': '2021-12-31', 'calibrated_threshold': 0.0, 'in_sample_coherence_pct': 95.0, 'out_of_sample_coherence_pct': 95.5, 'coherence_gap_pct': -0.5, 'test_return_pct': 40.0, 'annualized_return_pct': 40.0, 'max_drawdown_pct': -12.0, 'sharpe_ratio': 1.2},
                ],
            },
            'risk_management': {'max_dd_pause_pct': 0.15, 'pause_days': 20, 'n_pause_bars': 0, 'max_drawdown_pct': -10.0, 'n_entries': 1, 'n_exits': 1, 'pct_in_position': 66.67},
        }

        report = generate_validation_report(
            mock_mttd_data,
            '/home/ubuntu/projects/quant-technical-indicator-bank/isp-signals-btcusd-2026-06-13.csv',
            coherence_threshold=95.0,
            max_dd_threshold=15.0
        )

        assert 'verdict' in report
        assert 'coherence' in report
        assert 'risk_metrics' in report
        assert 'walk_forward' in report
        assert 'isp_benchmark' in report

        formatted = format_validation_report(report)
        assert 'VALIDATION REPORT' in formatted
        assert 'PASS' in formatted or 'FAIL' in formatted

        print(f"  ✓ Report generated successfully")
        print(f"    Verdict: {'PASS' if report['verdict']['overall_pass'] else 'FAIL'}")

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    # Test 5: PASS/FAIL logic
    print("\nTest 5: PASS/FAIL logic")
    try:
        # Case 1: High coherence, low DD → PASS
        mock_high = {
            'candles': [{'time': '2020-01-01', 'close': 100}],
            'aggregate': {'signals': [{'time': '2020-01-01', 'value': 1.0}]},
            'ensemble': {'threshold': 0, 'ema_length': 5, 'n_indicators': 5, 'pct_in_position': 100, 'n_trades': 0, 'look_ahead_verification': True},
            'calibration': {'optimal_threshold': 0, 'max_coherence_pct': 98, 'train_start_date': '2020-01-01', 'train_end_date': '2020-01-01', 'n_train_bars': 1},
            'coherence': {'time_coherence_pct': 98.0, 'verdict': {'passed': True}},
            'walk_forward': {'cycles_completed': 5, 'min_cycles_met': True, 'verdict': {'passed': True}, 'out_of_sample_coherence': {'mean': 97.0}},
            'risk_management': {'max_dd_pause_pct': 0.15, 'pause_days': 20, 'n_pause_bars': 0, 'max_drawdown_pct': -5.0, 'pct_in_position': 100},
        }
        report_pass = generate_validation_report(mock_high, '/home/ubuntu/projects/quant-technical-indicator-bank/isp-signals-btcusd-2026-06-13.csv')
        assert report_pass['verdict']['overall_pass'], "Should PASS with high coherence and low DD"
        print("  ✓ High coherence + low DD → PASS")

        # Case 2: Low coherence → FAIL
        mock_low_coh = dict(mock_high)
        mock_low_coh['coherence'] = {'time_coherence_pct': 80.0, 'verdict': {'passed': False}}
        report_fail1 = generate_validation_report(mock_low_coh, '/home/ubuntu/projects/quant-technical-indicator-bank/isp-signals-btcusd-2026-06-13.csv')
        assert not report_fail1['verdict']['overall_pass'], "Should FAIL with low coherence"
        print("  ✓ Low coherence → FAIL")

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 6: Empty data handling
    print("\nTest 6: Empty data handling")
    try:
        mock_empty = {
            'candles': [],
            'aggregate': {'signals': []},
            'ensemble': {},
            'calibration': {},
            'coherence': {},
            'walk_forward': {},
            'risk_management': {},
        }
        report_empty = generate_validation_report(mock_empty, '/home/ubuntu/projects/quant-technical-indicator-bank/isp-signals-btcusd-2026-06-13.csv')
        assert 'verdict' in report_empty
        formatted_empty = format_validation_report(report_empty)
        assert 'VALIDATION REPORT' in formatted_empty
        print("  ✓ Empty data handled gracefully")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MTTD Validation Report Generator')
    parser.add_argument('--mttd-json', type=str, help='Path to mttd_data.json')
    parser.add_argument('--isp-csv', type=str, help='Path to ISP signals CSV')
    parser.add_argument('--output', type=str, help='Output report path')
    parser.add_argument('--coherence-threshold', type=float, default=95.0, help='Min coherence for PASS')
    parser.add_argument('--max-dd-threshold', type=float, default=15.0, help='Max DD for PASS')
    parser.add_argument('--test', action='store_true', help='Run unit tests')

    args = parser.parse_args()

    if args.test:
        success = run_unit_tests()
        sys.exit(0 if success else 1)
    else:
        report, formatted = run_validation(
            mttd_json_path=args.mttd_json,
            isp_csv_path=args.isp_csv,
            output_path=args.output,
            coherence_threshold=args.coherence_threshold,
            max_dd_threshold=args.max_dd_threshold
        )
        sys.exit(0 if report['verdict']['overall_pass'] else 1)
