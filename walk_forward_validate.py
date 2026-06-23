"""
Walk-Forward Validation Module for MTTD Trading System
======================================================

Out-of-sample validation with expanding window to test robustness
of threshold calibration and prevent overfitting.

Walk-Forward Methodology:
1. Start with minimum training window (e.g., 12 months)
2. Calibrate threshold on training window (in-sample)
3. Apply calibrated threshold to test window (out-of-sample, 1 year)
4. Expand training window by test window length
5. Repeat until data is exhausted

Expanding Window Benefits:
- Training period grows with each cycle
- More data for calibration in later cycles
- Test window always fixed at 1 year
- No look-ahead bias across cycle boundaries

No Look-Ahead Bias:
- Training window strictly limited to data before test period
- Test window uses threshold calibrated on training data only
- No future data accessed in any computation
- Cycle boundaries are strictly enforced

Output per Cycle:
- Cycle number and dates
- Calibrated threshold (from training)
- In-sample coherence (training period)
- Out-of-sample coherence (test period)
- Test period returns
- Test period max drawdown
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Import existing modules
from mttd.ensemble_engine import compute_ensemble_signal
from mttd.coherence_metrics import compute_time_coherence, measure_coherence
from mttd.calibrate_threshold import calibrate_threshold
from mttd.risk_management import compute_equity_curve, get_risk_metrics


def generate_walk_forward_cycles(
    data_start_date: str,
    data_end_date: str,
    initial_train_months: int = 12,
    test_months: int = 12,
    min_cycles: int = 3
) -> List[Dict]:
    """
    Generate walk-forward cycle boundaries using expanding window.

    Parameters
    ----------
    data_start_date : str
        Start date of available data (format: 'YYYY-MM-DD').

    data_end_date : str
        End date of available data (format: 'YYYY-MM-DD').

    initial_train_months : int, default 12
        Initial training window length in months.

    test_months : int, default 12
        Test window length in months (fixed).

    min_cycles : int, default 3
        Minimum number of cycles required.

    Returns
    -------
    List[Dict]
        List of cycle dictionaries with:
        - cycle: cycle number (1-indexed)
        - train_start: training window start date
        - train_end: training window end date
        - test_start: test window start date
        - test_end: test window end date

    Raises
    ------
    ValueError
        If insufficient data for minimum cycles.
    """
    data_start = pd.Timestamp(data_start_date)
    data_end = pd.Timestamp(data_end_date)

    cycles = []
    cycle_num = 1

    # Start with initial training window
    train_start = data_start
    train_end = train_start + pd.DateOffset(months=initial_train_months) - pd.Timedelta(days=1)
    test_start = train_end + pd.Timedelta(days=1)
    test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

    while test_end <= data_end:
        cycles.append({
            'cycle': cycle_num,
            'train_start': train_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'test_start': test_start.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
        })

        cycle_num += 1

        # Expand training window by test period length
        train_end = test_end
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1)

    if len(cycles) < min_cycles:
        raise ValueError(
            f"Insufficient data for {min_cycles} walk-forward cycles. "
            f"Data range: {data_start_date} to {data_end_date}, "
            f"got {len(cycles)} cycles. "
            f"Need at least {initial_train_months + min_cycles * test_months} months of data."
        )

    return cycles


def compute_coherence_for_period(
    indicator_signals: pd.DataFrame,
    isp_positions: pd.Series,
    threshold: float,
    period_start: str,
    period_end: str,
    ema_length: int = 5,
    weights: Optional[pd.Series] = None
) -> Dict:
    """
    Compute time-coherence for a specific period using a given threshold.

    Parameters
    ----------
    indicator_signals : pd.DataFrame
        Full indicator signals DataFrame.

    isp_positions : pd.Series
        Full ISP positions Series.

    threshold : float
        Threshold to use for ensemble computation.

    period_start : str
        Start date of the period.

    period_end : str
        End date of the period.

    ema_length : int
        EMA smoothing length.

    weights : pd.Series or None
        Optional indicator weights.

    Returns
    -------
    Dict
        Coherence metrics for the period.
    """
    # Filter indicator signals to period
    if not isinstance(indicator_signals.index, pd.DatetimeIndex):
        sig_dt = indicator_signals.copy()
        sig_dt.index = pd.to_datetime(sig_dt.index)
    else:
        sig_dt = indicator_signals.copy()

    sig_period = sig_dt[(sig_dt.index >= period_start) & (sig_dt.index <= period_end)]

    if len(sig_period) == 0:
        return {
            'coherence_pct': 0.0,
            'n_bars': 0,
            'error': 'No indicator signals in period'
        }

    # Compute ensemble with given threshold
    sig_period_str = sig_period.copy()
    sig_period_str.index = sig_period_str.index.strftime('%Y-%m-%d')

    ensemble_result = compute_ensemble_signal(
        sig_period_str,
        threshold=threshold,
        ema_length=ema_length,
        weights=weights
    )

    mttd_positions = ensemble_result['position']

    # Filter ISP positions to period
    if not isinstance(isp_positions.index, pd.DatetimeIndex):
        isp_dt = pd.Series(
            isp_positions.values,
            index=pd.to_datetime(isp_positions.index)
        )
    else:
        isp_dt = isp_positions.copy()

    isp_period = isp_dt[(isp_dt.index >= period_start) & (isp_dt.index <= period_end)]

    if len(isp_period) == 0:
        return {
            'coherence_pct': 0.0,
            'n_bars': 0,
            'error': 'No ISP positions in period'
        }

    # Convert to string index for coherence computation
    isp_period_str = pd.Series(
        isp_period.values,
        index=isp_period.index.strftime('%Y-%m-%d')
    )

    # Compute coherence
    common_idx = mttd_positions.index.intersection(isp_period_str.index)
    if len(common_idx) == 0:
        return {
            'coherence_pct': 0.0,
            'n_bars': 0,
            'error': 'No overlapping dates'
        }

    coherence_result = compute_time_coherence(
        mttd_positions.loc[common_idx],
        isp_period_str.loc[common_idx]
    )

    return coherence_result


def compute_period_returns(
    indicator_signals: pd.DataFrame,
    price_series: pd.Series,
    threshold: float,
    period_start: str,
    period_end: str,
    ema_length: int = 5,
    weights: Optional[pd.Series] = None,
    initial_capital: float = 100000.0,
    commission_rate: float = 0.001
) -> Dict:
    """
    Compute returns and risk metrics for a specific period.

    Parameters
    ----------
    indicator_signals : pd.DataFrame
        Full indicator signals DataFrame.

    price_series : pd.Series
        Price series (daily close).

    threshold : float
        Threshold for ensemble computation.

    period_start : str
        Start date of the period.

    period_end : str
        End date of the period.

    ema_length : int
        EMA smoothing length.

    weights : pd.Series or None
        Optional indicator weights.

    initial_capital : float
        Starting capital.

    commission_rate : float
        Transaction commission rate.

    Returns
    -------
    Dict
        Returns and risk metrics for the period.
    """
    # Filter indicator signals to period
    if not isinstance(indicator_signals.index, pd.DatetimeIndex):
        sig_dt = indicator_signals.copy()
        sig_dt.index = pd.to_datetime(sig_dt.index)
    else:
        sig_dt = indicator_signals.copy()

    sig_period = sig_dt[(sig_dt.index >= period_start) & (sig_dt.index <= period_end)]

    if len(sig_period) == 0:
        return {
            'total_return_pct': 0.0,
            'annualized_return_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'n_bars': 0,
            'error': 'No data in period'
        }

    # Compute ensemble
    sig_period_str = sig_period.copy()
    sig_period_str.index = sig_period_str.index.strftime('%Y-%m-%d')

    ensemble_result = compute_ensemble_signal(
        sig_period_str,
        threshold=threshold,
        ema_length=ema_length,
        weights=weights
    )

    position = ensemble_result['position']

    # Filter price series to period
    if not isinstance(price_series.index, pd.DatetimeIndex):
        price_dt = pd.Series(
            price_series.values,
            index=pd.to_datetime(price_series.index)
        )
    else:
        price_dt = price_series.copy()

    price_period = price_dt[(price_dt.index >= period_start) & (price_dt.index <= period_end)]

    if len(price_period) == 0:
        return {
            'total_return_pct': 0.0,
            'annualized_return_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'n_bars': 0,
            'error': 'No price data in period'
        }

    # Align position and price
    price_period_str = pd.Series(
        price_period.values,
        index=price_period.index.strftime('%Y-%m-%d')
    )

    common_idx = position.index.intersection(price_period_str.index)
    if len(common_idx) == 0:
        return {
            'total_return_pct': 0.0,
            'annualized_return_pct': 0.0,
            'max_drawdown_pct': 0.0,
            'n_bars': 0,
            'error': 'No overlapping dates'
        }

    pos_aligned = position.loc[common_idx]
    price_aligned = price_period_str.loc[common_idx]

    # Compute daily returns and equity curve
    daily_returns = price_aligned.pct_change().fillna(0.0)

    # Simulate trading with commissions
    cash = initial_capital
    btc = 0.0
    prev_target = 0.0
    equity_list = []

    for date_str, (pos, price) in enumerate(zip(pos_aligned, price_aligned)):
        target = pos

        # Execute trade if position changed
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

    equity = pd.Series(equity_list, index=common_idx)

    # Compute metrics
    total_return_pct = (equity.iloc[-1] - initial_capital) / initial_capital * 100.0

    # Annualized return
    n_days = len(equity)
    n_years = n_days / 365.25
    if n_years > 0 and equity.iloc[-1] > 0:
        annualized_return_pct = ((equity.iloc[-1] / initial_capital) ** (1 / n_years) - 1) * 100.0
    else:
        annualized_return_pct = 0.0

    # Max drawdown
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    max_drawdown_pct = float(drawdown.min()) * 100.0

    # Sharpe ratio
    returns_series = equity.pct_change().fillna(0.0)
    mean_return = returns_series.mean()
    std_return = returns_series.std()
    sharpe = (mean_return / std_return) * np.sqrt(365) if std_return > 0 else 0.0

    return {
        'total_return_pct': round(total_return_pct, 2),
        'annualized_return_pct': round(annualized_return_pct, 2),
        'max_drawdown_pct': round(max_drawdown_pct, 2),
        'sharpe_ratio': round(sharpe, 4),
        'n_bars': n_days,
        'final_equity': round(equity.iloc[-1], 2),
        'equity_curve': equity
    }


def run_walk_forward_validation(
    indicator_signals: pd.DataFrame,
    isp_positions: pd.Series,
    price_series: pd.Series,
    data_start_date: str,
    data_end_date: str,
    initial_train_months: int = 12,
    test_months: int = 12,
    min_cycles: int = 3,
    ema_length: int = 5,
    weights: Optional[pd.Series] = None,
    threshold_min: float = -0.5,
    threshold_max: float = 0.5,
    threshold_step: float = 0.01,
    initial_capital: float = 100000.0,
    commission_rate: float = 0.001
) -> Dict:
    """
    Run walk-forward validation with expanding window.

    Parameters
    ----------
    indicator_signals : pd.DataFrame
        Matrix of indicator signals where each column is an indicator.
        Values should be +1.0 (bullish) or -1.0 (bearish).

    isp_positions : pd.Series
        ISP binary position series (1.0 or 0.0).

    price_series : pd.Series
        Daily close price series.

    data_start_date : str
        Start date of available data.

    data_end_date : str
        End date of available data.

    initial_train_months : int
        Initial training window length in months.

    test_months : int
        Test window length in months.

    min_cycles : int
        Minimum number of cycles required.

    ema_length : int
        EMA smoothing length.

    weights : pd.Series or None
        Optional indicator weights.

    threshold_min : float
        Minimum threshold for grid search.

    threshold_max : float
        Maximum threshold for grid search.

    threshold_step : float
        Step size for threshold grid search.

    initial_capital : float
        Starting capital for simulation.

    commission_rate : float
        Transaction commission rate.

    Returns
    -------
    Dict
        Walk-forward validation results with:
        - cycles: list of cycle results
        - summary: aggregate statistics
        - cycles_completed: number of cycles completed
        - min_cycles_met: whether minimum cycles requirement is met
    """
    # Generate cycle boundaries
    cycles_boundaries = generate_walk_forward_cycles(
        data_start_date,
        data_end_date,
        initial_train_months=initial_train_months,
        test_months=test_months,
        min_cycles=min_cycles
    )

    print(f"\nWalk-Forward Validation: {len(cycles_boundaries)} cycles planned")
    print("=" * 70)

    cycle_results = []

    for cycle_info in cycles_boundaries:
        cycle_num = cycle_info['cycle']
        train_start = cycle_info['train_start']
        train_end = cycle_info['train_end']
        test_start = cycle_info['test_start']
        test_end = cycle_info['test_end']

        print(f"\n--- Cycle {cycle_num} ---")
        print(f"  Training: {train_start} to {train_end}")
        print(f"  Testing:  {test_start} to {test_end}")

        # Step 1: Calibrate threshold on training period
        # For expanding window, lookback_months = total months from train_start to train_end
        train_start_ts = pd.Timestamp(train_start)
        train_end_ts = pd.Timestamp(train_end)
        lookback_months = (train_end_ts.year - train_start_ts.year) * 12 + (
            train_end_ts.month - train_start_ts.month
        ) + 1  # Add 1 to include partial months

        try:
            calibration = calibrate_threshold(
                df=pd.DataFrame({'close': price_series}),  # Dummy df, not used in calibration
                indicator_signals=indicator_signals,
                isp_positions=isp_positions,
                train_end_date=train_end,
                lookback_months=lookback_months,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                threshold_step=threshold_step,
                ema_length=ema_length,
                weights=weights
            )
            optimal_threshold = calibration['optimal_threshold']
            in_sample_coherence = calibration['max_coherence']

            print(f"  Calibrated threshold: {optimal_threshold:.4f}")
            print(f"  In-sample coherence: {in_sample_coherence:.2f}%")

        except Exception as e:
            print(f"  ⚠ Calibration failed: {e}")
            # Use default threshold if calibration fails
            optimal_threshold = 0.0
            in_sample_coherence = 0.0

        # Step 2: Measure out-of-sample coherence on test period
        oos_coherence_result = compute_coherence_for_period(
            indicator_signals,
            isp_positions,
            optimal_threshold,
            test_start,
            test_end,
            ema_length=ema_length,
            weights=weights
        )
        oos_coherence = oos_coherence_result.get('coherence_pct', 0.0)

        # Step 3: Compute returns and risk metrics for test period
        returns_result = compute_period_returns(
            indicator_signals,
            price_series,
            optimal_threshold,
            test_start,
            test_end,
            ema_length=ema_length,
            weights=weights,
            initial_capital=initial_capital,
            commission_rate=commission_rate
        )

        total_return = returns_result.get('total_return_pct', 0.0)
        annual_return = returns_result.get('annualized_return_pct', 0.0)
        max_dd = returns_result.get('max_drawdown_pct', 0.0)
        sharpe = returns_result.get('sharpe_ratio', 0.0)

        print(f"  Out-of-sample coherence: {oos_coherence:.2f}%")
        print(f"  Test period return: {total_return:.2f}%")
        print(f"  Annualized return: {annual_return:.2f}%")
        print(f"  Max drawdown: {max_dd:.2f}%")
        print(f"  Sharpe ratio: {sharpe:.4f}")

        # Store cycle results
        cycle_result = {
            'cycle': cycle_num,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'calibrated_threshold': optimal_threshold,
            'in_sample_coherence_pct': in_sample_coherence,
            'out_of_sample_coherence_pct': oos_coherence,
            'coherence_gap_pct': round(in_sample_coherence - oos_coherence, 2),
            'test_return_pct': total_return,
            'annualized_return_pct': annual_return,
            'max_drawdown_pct': max_dd,
            'sharpe_ratio': sharpe,
            'n_test_bars': returns_result.get('n_bars', 0),
        }

        # Check for overfitting
        if in_sample_coherence > 0 and oos_coherence > 0:
            overfit_ratio = oos_coherence / in_sample_coherence
            cycle_result['overfit_ratio'] = round(overfit_ratio, 4)
        else:
            cycle_result['overfit_ratio'] = 0.0

        cycle_results.append(cycle_result)

    # Compute summary statistics
    n_cycles = len(cycle_results)

    # Aggregate coherence
    is_coherences = [c['in_sample_coherence_pct'] for c in cycle_results]
    oos_coherences = [c['out_of_sample_coherence_pct'] for c in cycle_results]
    coherence_gaps = [c['coherence_gap_pct'] for c in cycle_results]

    # Aggregate returns
    test_returns = [c['test_return_pct'] for c in cycle_results]
    annual_returns = [c['annualized_return_pct'] for c in cycle_results]
    max_dds = [c['max_drawdown_pct'] for c in cycle_results]
    sharpes = [c['sharpe_ratio'] for c in cycle_results]

    # Stability metrics
    coherence_std = float(np.std(oos_coherences)) if oos_coherences else 0.0
    return_std = float(np.std(annual_returns)) if annual_returns else 0.0

    summary = {
        'cycles_completed': n_cycles,
        'min_cycles_met': n_cycles >= min_cycles,
        'data_range': {
            'start': data_start_date,
            'end': data_end_date,
        },
        'in_sample_coherence': {
            'mean': round(float(np.mean(is_coherences)), 2) if is_coherences else 0.0,
            'min': round(float(np.min(is_coherences)), 2) if is_coherences else 0.0,
            'max': round(float(np.max(is_coherences)), 2) if is_coherences else 0.0,
            'std': round(float(np.std(is_coherences)), 2) if is_coherences else 0.0,
        },
        'out_of_sample_coherence': {
            'mean': round(float(np.mean(oos_coherences)), 2) if oos_coherences else 0.0,
            'min': round(float(np.min(oos_coherences)), 2) if oos_coherences else 0.0,
            'max': round(float(np.max(oos_coherences)), 2) if oos_coherences else 0.0,
            'std': round(coherence_std, 2),
        },
        'coherence_gap': {
            'mean': round(float(np.mean(coherence_gaps)), 2) if coherence_gaps else 0.0,
            'max': round(float(np.max(coherence_gaps)), 2) if coherence_gaps else 0.0,
        },
        'returns': {
            'mean_test_return_pct': round(float(np.mean(test_returns)), 2) if test_returns else 0.0,
            'mean_annual_return_pct': round(float(np.mean(annual_returns)), 2) if annual_returns else 0.0,
            'return_std': round(return_std, 2),
            'worst_test_return_pct': round(float(np.min(test_returns)), 2) if test_returns else 0.0,
            'best_test_return_pct': round(float(np.max(test_returns)), 2) if test_returns else 0.0,
        },
        'risk': {
            'mean_max_drawdown_pct': round(float(np.mean(max_dds)), 2) if max_dds else 0.0,
            'worst_max_drawdown_pct': round(float(np.min(max_dds)), 2) if max_dds else 0.0,
            'mean_sharpe': round(float(np.mean(sharpes)), 4) if sharpes else 0.0,
        },
        'overfitting': {
            'mean_coherence_gap': round(float(np.mean(coherence_gaps)), 2) if coherence_gaps else 0.0,
            'coherence_stability': round(coherence_std, 2),
            'avg_overfit_ratio': round(float(np.mean([c.get('overfit_ratio', 0) for c in cycle_results])), 4),
        },
    }

    # Verdict
    avg_oos_coherence = summary['out_of_sample_coherence']['mean']
    coherence_pass = avg_oos_coherence >= 95.0
    cycles_pass = summary['min_cycles_met']

    summary['verdict'] = {
        'passed': coherence_pass and cycles_pass,
        'reason': (
            f"Average OOS coherence: {avg_oos_coherence:.2f}% "
            f"({'PASS' if coherence_pass else 'FAIL'} >= 95%), "
            f"Cycles: {n_cycles} ({'PASS' if cycles_pass else 'FAIL'} >= {min_cycles})"
        ),
    }

    return {
        'cycles': cycle_results,
        'summary': summary,
        'cycles_completed': n_cycles,
        'min_cycles_met': cycles_pass,
    }


def format_walk_forward_report(results: Dict) -> str:
    """
    Format walk-forward validation results into a human-readable report.

    Parameters
    ----------
    results : dict
        Output from run_walk_forward_validation().

    Returns
    -------
    str
        Formatted report string.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("WALK-FORWARD VALIDATION REPORT")
    lines.append("=" * 70)

    summary = results['summary']

    # Verdict
    verdict = summary['verdict']
    verdict_symbol = "✓ PASS" if verdict['passed'] else "✗ FAIL"
    lines.append(f"\nVerdict: {verdict_symbol}")
    lines.append(f"  {verdict['reason']}")

    # Summary Statistics
    lines.append(f"\n--- Summary Statistics ---")
    lines.append(f"  Cycles Completed: {summary['cycles_completed']}")
    lines.append(f"  Min Cycles Met:   {'Yes' if summary['min_cycles_met'] else 'No'}")

    # In-Sample Coherence
    is_coh = summary['in_sample_coherence']
    lines.append(f"\n--- In-Sample Coherence (Training) ---")
    lines.append(f"  Mean:  {is_coh['mean']:.2f}%")
    lines.append(f"  Min:   {is_coh['min']:.2f}%")
    lines.append(f"  Max:   {is_coh['max']:.2f}%")
    lines.append(f"  Std:   {is_coh['std']:.2f}%")

    # Out-of-Sample Coherence
    oos_coh = summary['out_of_sample_coherence']
    lines.append(f"\n--- Out-of-Sample Coherence (Test) ---")
    lines.append(f"  Mean:  {oos_coh['mean']:.2f}%")
    lines.append(f"  Min:   {oos_coh['min']:.2f}%")
    lines.append(f"  Max:   {oos_coh['max']:.2f}%")
    lines.append(f"  Std:   {oos_coh['std']:.2f}%")

    # Coherence Gap (Overfitting Indicator)
    gap = summary['coherence_gap']
    lines.append(f"\n--- Coherence Gap (IS - OOS) ---")
    lines.append(f"  Mean Gap:  {gap['mean']:.2f}%")
    lines.append(f"  Max Gap:   {gap['max']:.2f}%")

    # Returns
    ret = summary['returns']
    lines.append(f"\n--- Returns (Test Periods) ---")
    lines.append(f"  Mean Test Return:     {ret['mean_test_return_pct']:.2f}%")
    lines.append(f"  Mean Annual Return:   {ret['mean_annual_return_pct']:.2f}%")
    lines.append(f"  Return Std Dev:       {ret['return_std']:.2f}%")
    lines.append(f"  Best Test Return:     {ret['best_test_return_pct']:.2f}%")
    lines.append(f"  Worst Test Return:    {ret['worst_test_return_pct']:.2f}%")

    # Risk
    risk = summary['risk']
    lines.append(f"\n--- Risk ---")
    lines.append(f"  Mean Max Drawdown:    {risk['mean_max_drawdown_pct']:.2f}%")
    lines.append(f"  Worst Max Drawdown:   {risk['worst_max_drawdown_pct']:.2f}%")
    lines.append(f"  Mean Sharpe Ratio:    {risk['mean_sharpe']:.4f}")

    # Overfitting Analysis
    overfit = summary['overfitting']
    lines.append(f"\n--- Overfitting Analysis ---")
    lines.append(f"  Mean Coherence Gap:   {overfit['mean_coherence_gap']:.2f}%")
    lines.append(f"  Coherence Stability:  {overfit['coherence_stability']:.2f}% (std dev)")
    lines.append(f"  Avg Overfit Ratio:    {overfit['avg_overfit_ratio']:.4f} (OOS/IS)")

    # Individual Cycles
    lines.append(f"\n--- Individual Cycles ---")
    lines.append(f"{'Cycle':<7} {'Train Period':<25} {'Test Period':<25} "
                 f"{'IS Coh%':<10} {'OOS Coh%':<10} {'Return%':<10} {'MaxDD%':<10}")
    lines.append("-" * 97)

    for cycle in results['cycles']:
        train_period = f"{cycle['train_start']} to {cycle['train_end']}"
        test_period = f"{cycle['test_start']} to {cycle['test_end']}"
        lines.append(
            f"{cycle['cycle']:<7} {train_period:<25} {test_period:<25} "
            f"{cycle['in_sample_coherence_pct']:<10.2f} "
            f"{cycle['out_of_sample_coherence_pct']:<10.2f} "
            f"{cycle['test_return_pct']:<10.2f} "
            f"{cycle['max_drawdown_pct']:<10.2f}"
        )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def run_unit_tests():
    """
    Run unit tests for the walk-forward validation module.

    Returns
    -------
    bool
        True if all tests pass, False otherwise.
    """
    print("=" * 70)
    print("Running Walk-Forward Validation Unit Tests")
    print("=" * 70)

    all_passed = True

    # Test 1: Generate cycle boundaries
    print("\nTest 1: Generate cycle boundaries")
    try:
        cycles = generate_walk_forward_cycles(
            '2015-01-01',
            '2025-06-01',
            initial_train_months=12,
            test_months=12,
            min_cycles=3
        )

        assert len(cycles) >= 3, f"Expected at least 3 cycles, got {len(cycles)}"
        assert cycles[0]['cycle'] == 1, "First cycle should be 1"
        assert cycles[-1]['cycle'] == len(cycles), "Last cycle number should match count"

        # Verify expanding window (training period grows)
        for i in range(1, len(cycles)):
            prev_train_start = pd.Timestamp(cycles[i-1]['train_start'])
            curr_train_start = pd.Timestamp(cycles[i]['train_start'])
            assert curr_train_start == prev_train_start, \
                f"Training start should stay the same (expanding window)"

        print(f"  ✓ Generated {len(cycles)} cycles with expanding window")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 2: Insufficient data raises error
    print("\nTest 2: Insufficient data raises error")
    try:
        try:
            generate_walk_forward_cycles(
                '2020-01-01',
                '2020-06-01',  # Only 6 months
                initial_train_months=12,
                test_months=12,
                min_cycles=3
            )
            print("  ✗ FAILED: Should have raised ValueError")
            all_passed = False
        except ValueError as e:
            if "Insufficient data" in str(e):
                print("  ✓ Insufficient data correctly raises ValueError")
            else:
                print(f"  ✗ FAILED: Wrong error message: {e}")
                all_passed = False
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 3: Compute coherence for period
    print("\nTest 3: Compute coherence for period")
    try:
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        date_strs = dates.strftime('%Y-%m-%d').tolist()

        # Create indicator signals (mostly bullish in first half)
        signal_data = {}
        for i in range(5):
            signals = np.concatenate([
                np.random.choice([-1.0, 1.0], size=180, p=[0.2, 0.8]),
                np.random.choice([-1.0, 1.0], size=185, p=[0.8, 0.2])
            ])
            signal_data[f'ind_{i}'] = signals

        indicator_signals = pd.DataFrame(signal_data, index=date_strs)

        # ISP follows similar pattern
        isp_positions = pd.Series(
            [1.0 if i < 180 else 0.0 for i in range(365)],
            index=date_strs
        )

        # Test coherence for first half
        result = compute_coherence_for_period(
            indicator_signals,
            isp_positions,
            threshold=0.0,
            period_start='2020-01-01',
            period_end='2020-06-30',
            ema_length=5
        )

        assert 'coherence_pct' in result, "Missing coherence_pct"
        assert 0 <= result['coherence_pct'] <= 100, \
            f"Coherence out of range: {result['coherence_pct']}"

        print(f"  ✓ Coherence computed: {result['coherence_pct']:.2f}%")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 4: Compute period returns
    print("\nTest 4: Compute period returns")
    try:
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        date_strs = dates.strftime('%Y-%m-%d').tolist()

        # Simple bullish signals
        indicator_signals = pd.DataFrame({
            'ind_0': [1.0] * 365,
            'ind_1': [1.0] * 365,
        }, index=date_strs)

        # Price with upward trend
        price = pd.Series(
            [100 * (1.001 ** i) for i in range(365)],
            index=date_strs
        )

        # Compute returns
        result = compute_period_returns(
            indicator_signals,
            price,
            threshold=0.0,
            period_start='2020-01-01',
            period_end='2020-12-31',
            ema_length=5,
            initial_capital=10000
        )

        assert 'total_return_pct' in result, "Missing total_return_pct"
        assert 'max_drawdown_pct' in result, "Missing max_drawdown_pct"
        assert 'sharpe_ratio' in result, "Missing sharpe_ratio"

        print(f"  ✓ Returns computed: {result['total_return_pct']:.2f}%, "
              f"Sharpe: {result['sharpe_ratio']:.4f}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 5: Full walk-forward validation (synthetic data)
    print("\nTest 5: Full walk-forward validation (synthetic)")
    try:
        np.random.seed(42)
        # Create 4 years of data (enough for 3 cycles)
        dates = pd.date_range('2020-01-01', periods=1460, freq='D')  # 4 years
        date_strs = dates.strftime('%Y-%m-%d').tolist()

        # Create indicator signals with regime change
        signal_data = {}
        for i in range(5):
            signals = []
            for j in range(1460):
                if j < 365:
                    signals.append(np.random.choice([-1.0, 1.0], p=[0.3, 0.7]))
                elif j < 730:
                    signals.append(np.random.choice([-1.0, 1.0], p=[0.7, 0.3]))
                elif j < 1095:
                    signals.append(np.random.choice([-1.0, 1.0], p=[0.3, 0.7]))
                else:
                    signals.append(np.random.choice([-1.0, 1.0], p=[0.6, 0.4]))
            signal_data[f'ind_{i}'] = signals

        indicator_signals = pd.DataFrame(signal_data, index=date_strs)

        # ISP follows similar pattern
        isp_positions = pd.Series(
            [1.0 if (i % 365) < 250 else 0.0 for i in range(1460)],
            index=date_strs
        )

        # Price series
        price = pd.Series(
            [100 * (1.0003 ** i) for i in range(1460)],
            index=date_strs
        )

        # Run walk-forward
        results = run_walk_forward_validation(
            indicator_signals,
            isp_positions,
            price,
            data_start_date='2020-01-01',
            data_end_date='2023-12-31',
            initial_train_months=12,
            test_months=12,
            min_cycles=3,
            ema_length=5,
            threshold_step=0.1,  # Coarse grid for faster test
            initial_capital=10000
        )

        assert results['cycles_completed'] >= 3, \
            f"Expected at least 3 cycles, got {results['cycles_completed']}"
        assert results['min_cycles_met'], "Minimum cycles not met"
        assert 'cycles' in results, "Missing cycles"
        assert 'summary' in results, "Missing summary"

        # Verify each cycle has required fields
        for cycle in results['cycles']:
            assert 'calibrated_threshold' in cycle
            assert 'in_sample_coherence_pct' in cycle
            assert 'out_of_sample_coherence_pct' in cycle
            assert 'test_return_pct' in cycle
            assert 'max_drawdown_pct' in cycle

        print(f"  ✓ Walk-forward completed: {results['cycles_completed']} cycles")
        print(f"    Avg OOS coherence: {results['summary']['out_of_sample_coherence']['mean']:.2f}%")
        print(f"    Avg annual return: {results['summary']['returns']['mean_annual_return_pct']:.2f}%")

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 6: No look-ahead bias verification
    print("\nTest 6: No look-ahead bias verification")
    try:
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1460, freq='D')
        date_strs = dates.strftime('%Y-%m-%d').tolist()

        # Simple signals
        indicator_signals = pd.DataFrame({
            'ind_0': np.random.choice([-1.0, 1.0], size=1460),
            'ind_1': np.random.choice([-1.0, 1.0], size=1460),
        }, index=date_strs)

        isp_positions = pd.Series(
            [1.0 if i % 365 < 250 else 0.0 for i in range(1460)],
            index=date_strs
        )

        price = pd.Series(
            [100 * (1.0003 ** i) for i in range(1460)],
            index=date_strs
        )

        # Get cycle boundaries
        cycles = generate_walk_forward_cycles(
            '2020-01-01',
            '2023-12-31',
            initial_train_months=12,
            test_months=12,
            min_cycles=3
        )

        # Verify no overlap between train and test periods
        for cycle in cycles:
            train_end = pd.Timestamp(cycle['train_end'])
            test_start = pd.Timestamp(cycle['test_start'])

            assert test_start > train_end, \
                f"Test period overlaps with training period in cycle {cycle['cycle']}"

        # Verify expanding window
        for i in range(1, len(cycles)):
            assert cycles[i]['train_start'] == cycles[0]['train_start'], \
                "Training start should be constant (expanding window)"
            assert pd.Timestamp(cycles[i]['train_end']) > pd.Timestamp(cycles[i-1]['train_end']), \
                "Training end should expand"

        print("  ✓ No look-ahead bias: cycle boundaries are valid")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 7: Report formatting
    print("\nTest 7: Report formatting")
    try:
        # Create minimal results for report formatting
        results = {
            'cycles': [
                {
                    'cycle': 1,
                    'train_start': '2020-01-01',
                    'train_end': '2020-12-31',
                    'test_start': '2021-01-01',
                    'test_end': '2021-12-31',
                    'calibrated_threshold': 0.15,
                    'in_sample_coherence_pct': 85.5,
                    'out_of_sample_coherence_pct': 82.3,
                    'coherence_gap_pct': 3.2,
                    'test_return_pct': 25.5,
                    'annualized_return_pct': 25.5,
                    'max_drawdown_pct': -15.2,
                    'sharpe_ratio': 1.25,
                },
            ],
            'summary': {
                'cycles_completed': 1,
                'min_cycles_met': False,
                'in_sample_coherence': {'mean': 85.5, 'min': 85.5, 'max': 85.5, 'std': 0.0},
                'out_of_sample_coherence': {'mean': 82.3, 'min': 82.3, 'max': 82.3, 'std': 0.0},
                'coherence_gap': {'mean': 3.2, 'max': 3.2},
                'returns': {
                    'mean_test_return_pct': 25.5,
                    'mean_annual_return_pct': 25.5,
                    'return_std': 0.0,
                    'best_test_return_pct': 25.5,
                    'worst_test_return_pct': 25.5,
                },
                'risk': {
                    'mean_max_drawdown_pct': -15.2,
                    'worst_max_drawdown_pct': -15.2,
                    'mean_sharpe': 1.25,
                },
                'overfitting': {
                    'mean_coherence_gap': 3.2,
                    'coherence_stability': 0.0,
                    'avg_overfit_ratio': 0.96,
                },
                'verdict': {
                    'passed': False,
                    'reason': 'Insufficient cycles and coherence below 95%',
                },
            },
            'cycles_completed': 1,
            'min_cycles_met': False,
        }

        report = format_walk_forward_report(results)

        assert "WALK-FORWARD VALIDATION REPORT" in report
        assert "Cycle" in report
        assert "In-Sample Coherence" in report
        assert "Out-of-Sample Coherence" in report

        print("  ✓ Report formatting works correctly")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 8: Edge case - all same threshold
    print("\nTest 8: Edge case - consistent signals")
    try:
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1460, freq='D')
        date_strs = dates.strftime('%Y-%m-%d').tolist()

        # All bullish signals
        indicator_signals = pd.DataFrame({
            'ind_0': [1.0] * 1460,
            'ind_1': [1.0] * 1460,
        }, index=date_strs)

        # ISP always in market
        isp_positions = pd.Series([1.0] * 1460, index=date_strs)

        # Price with slight uptrend
        price = pd.Series(
            [100 * (1.0002 ** i) for i in range(1460)],
            index=date_strs
        )

        results = run_walk_forward_validation(
            indicator_signals,
            isp_positions,
            price,
            data_start_date='2020-01-01',
            data_end_date='2023-12-31',
            initial_train_months=12,
            test_months=12,
            min_cycles=3,
            ema_length=5,
            threshold_step=0.1,
            initial_capital=10000
        )

        # With all matching signals, coherence should be high
        avg_oos_coherence = results['summary']['out_of_sample_coherence']['mean']
        assert avg_oos_coherence >= 95.0, \
            f"Expected high coherence with matching signals, got {avg_oos_coherence:.2f}%"

        print(f"  ✓ Consistent signals handled: OOS coherence = {avg_oos_coherence:.2f}%")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 9: Different initial training periods
    print("\nTest 9: Different initial training periods")
    try:
        # Use 6 years of data to ensure both cases have enough cycles
        cycles_12m = generate_walk_forward_cycles(
            '2018-01-01', '2024-01-01',
            initial_train_months=12, test_months=12, min_cycles=3
        )

        cycles_24m = generate_walk_forward_cycles(
            '2018-01-01', '2024-01-01',
            initial_train_months=24, test_months=12, min_cycles=3
        )

        # Longer initial training should result in fewer cycles
        assert len(cycles_12m) > len(cycles_24m), \
            f"12m initial should have more cycles: {len(cycles_12m)} vs {len(cycles_24m)}"

        # Both should have at least 3 cycles
        assert len(cycles_12m) >= 3
        assert len(cycles_24m) >= 3

        print(f"  ✓ Different initial training periods work: "
              f"12m={len(cycles_12m)} cycles, 24m={len(cycles_24m)} cycles")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 10: Verify test window is always 1 year
    print("\nTest 10: Verify test window is always 1 year")
    try:
        cycles = generate_walk_forward_cycles(
            '2015-01-01', '2025-01-01',
            initial_train_months=12, test_months=12, min_cycles=3
        )

        for cycle in cycles:
            test_start = pd.Timestamp(cycle['test_start'])
            test_end = pd.Timestamp(cycle['test_end'])
            test_duration = (test_end - test_start).days

            # Allow 1 day tolerance for month boundaries
            assert 364 <= test_duration <= 366, \
                f"Test window should be ~1 year ({test_duration} days) in cycle {cycle['cycle']}"

        print(f"  ✓ All test windows are ~1 year (365 days)")
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
    success = run_unit_tests()
    if not success:
        exit(1)
