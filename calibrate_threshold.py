"""
Threshold Calibration Module for MTTD Trading System
=====================================================

Finds the optimal ensemble threshold that maximizes time-coherence
with the ISP benchmark during a training window.

Methodology:
1. Define training window using rolling 12-month lookback
2. Grid search thresholds in range [-0.5, 0.5] with step 0.01
3. For each threshold, compute ensemble positions and measure coherence
4. Return optimal threshold that maximizes in-sample coherence

No Look-Ahead Bias:
- Training window is strictly limited to data before train_end_date
- No future data is accessed in any computation
- Grid search only uses in-sample data for threshold selection
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta

# Import ensemble engine and coherence metrics
from mttd.ensemble_engine import compute_ensemble_signal
from mttd.coherence_metrics import compute_time_coherence


def compute_coherence_for_threshold(
    indicator_signals: pd.DataFrame,
    isp_positions: pd.Series,
    threshold: float,
    ema_length: int = 5,
    weights: Optional[pd.Series] = None
) -> float:
    """
    Compute time-coherence for a given threshold value.

    Parameters
    ----------
    indicator_signals : pd.DataFrame
        Indicator signals for the training window.
        Each column is an indicator, values are +1.0 (bullish) or -1.0 (bearish).

    isp_positions : pd.Series
        ISP binary position series (1.0 or 0.0) for the training window.

    threshold : float
        Threshold to test. If smoothed_average > threshold → position = 1.

    ema_length : int, default 5
        EMA smoothing length for ensemble computation.

    weights : pd.Series or None, default None
        Optional indicator weights. If None, equal weighting is used.

    Returns
    -------
    float
        Time-coherence percentage (0-100).
    """
    if indicator_signals.empty or isp_positions.empty:
        return 0.0

    # Compute ensemble signal with given threshold
    ensemble_result = compute_ensemble_signal(
        indicator_signals,
        threshold=threshold,
        ema_length=ema_length,
        weights=weights
    )

    # Extract binary positions
    mttd_positions = ensemble_result['position']

    # Align with ISP positions on common index
    common_idx = mttd_positions.index.intersection(isp_positions.index)
    if len(common_idx) == 0:
        return 0.0

    mttd_aligned = mttd_positions.loc[common_idx]
    isp_aligned = isp_positions.loc[common_idx]

    # Compute time-coherence
    coherence_result = compute_time_coherence(mttd_aligned, isp_aligned)

    return coherence_result['coherence_pct']


def calibrate_threshold(
    df: pd.DataFrame,
    indicator_signals: pd.DataFrame,
    isp_positions: pd.Series,
    train_end_date: str,
    lookback_months: int = 12,
    threshold_min: float = -0.5,
    threshold_max: float = 0.5,
    threshold_step: float = 0.01,
    ema_length: int = 5,
    weights: Optional[pd.Series] = None
) -> Dict:
    """
    Find optimal ensemble threshold maximizing ISP coherence within training window.

    This function performs grid search over threshold values to find the one
    that maximizes time-coherence between the MTTD ensemble and ISP benchmark.

    Parameters
    ----------
    df : pd.DataFrame
        Price data DataFrame with DatetimeIndex or date string index.

    indicator_signals : pd.DataFrame
        Matrix of indicator signals where each column is an indicator.
        Values should be +1.0 (bullish) or -1.0 (bearish).
        Index must match df index.

    isp_positions : pd.Series
        ISP binary position series (1.0 or 0.0).
        Index must be date strings or DatetimeIndex.

    train_end_date : str
        End date for training window (inclusive).
        Format: 'YYYY-MM-DD'.

    lookback_months : int, default 12
        Number of months to look back from train_end_date for training data.

    threshold_min : float, default -0.5
        Minimum threshold value to search.

    threshold_max : float, default 0.5
        Maximum threshold value to search.

    threshold_step : float, default 0.01
        Step size for threshold grid search.

    ema_length : int, default 5
        EMA smoothing length for ensemble computation.

    weights : pd.Series or None, default None
        Optional indicator weights. If None, equal weighting is used.

    Returns
    -------
    dict
        Dictionary with:
        - optimal_threshold: best threshold value found
        - max_coherence: coherence score at optimal threshold (0-100)
        - train_start_date: start of training window
        - train_end_date: end of training window
        - n_train_bars: number of bars in training window
        - n_indicators: number of indicators used
        - threshold_grid_size: number of thresholds tested
        - all_results: list of (threshold, coherence) pairs for all tested values

    Raises
    ------
    ValueError
        If inputs are invalid or training window has insufficient data.
    """
    # Validate inputs
    if indicator_signals.empty:
        raise ValueError("indicator_signals DataFrame is empty")
    if isp_positions.empty:
        raise ValueError("isp_positions Series is empty")
    if threshold_min >= threshold_max:
        raise ValueError(f"threshold_min ({threshold_min}) must be < threshold_max ({threshold_max})")
    if threshold_step <= 0:
        raise ValueError(f"threshold_step ({threshold_step}) must be > 0")

    # Parse train_end_date
    train_end = pd.Timestamp(train_end_date)

    # Compute training window start
    train_start = train_end - pd.DateOffset(months=lookback_months)

    # Convert ISP positions index to DatetimeIndex if needed
    if not isinstance(isp_positions.index, pd.DatetimeIndex):
        isp_dt = pd.Series(
            isp_positions.values,
            index=pd.to_datetime(isp_positions.index)
        )
    else:
        isp_dt = isp_positions.copy()

    # Filter ISP positions to training window
    isp_train = isp_dt[(isp_dt.index >= train_start) & (isp_dt.index <= train_end)]

    if len(isp_train) == 0:
        raise ValueError(
            f"No ISP positions found in training window [{train_start.date()}, {train_end.date()}]"
        )

    # Filter indicator signals to training window
    if not isinstance(indicator_signals.index, pd.DatetimeIndex):
        sig_dt = indicator_signals.copy()
        sig_dt.index = pd.to_datetime(sig_dt.index)
    else:
        sig_dt = indicator_signals.copy()

    sig_train = sig_dt[(sig_dt.index >= train_start) & (sig_dt.index <= train_end)]

    if len(sig_train) == 0:
        raise ValueError(
            f"No indicator signals found in training window [{train_start.date()}, {train_end.date()}]"
        )

    # Align ISP and signal indices
    common_idx = sig_train.index.intersection(isp_train.index)
    if len(common_idx) == 0:
        raise ValueError("No overlapping dates between indicator signals and ISP positions in training window")

    sig_train_aligned = sig_train.loc[common_idx]
    isp_train_aligned = isp_train.loc[common_idx]

    # Convert ISP positions to date strings for coherence computation
    isp_train_str = pd.Series(
        isp_train_aligned.values,
        index=isp_train_aligned.index.strftime('%Y-%m-%d')
    )

    # Generate threshold grid
    thresholds = np.arange(threshold_min, threshold_max + threshold_step/2, threshold_step)
    # Round to avoid floating point issues
    thresholds = np.round(thresholds, decimals=4)

    # Grid search
    results = []
    max_coherence = -1.0
    optimal_threshold = threshold_min

    for threshold in thresholds:
        # Convert signals index to date strings for ensemble computation
        sig_train_str = sig_train_aligned.copy()
        sig_train_str.index = sig_train_str.index.strftime('%Y-%m-%d')

        coherence = compute_coherence_for_threshold(
            sig_train_str,
            isp_train_str,
            threshold,
            ema_length=ema_length,
            weights=weights
        )

        results.append((float(threshold), coherence))

        if coherence > max_coherence:
            max_coherence = coherence
            optimal_threshold = float(threshold)

    # Build result dictionary
    calibration_result = {
        'optimal_threshold': optimal_threshold,
        'max_coherence': round(max_coherence, 4),
        'train_start_date': train_start.strftime('%Y-%m-%d'),
        'train_end_date': train_end.strftime('%Y-%m-%d'),
        'n_train_bars': len(common_idx),
        'n_indicators': indicator_signals.shape[1],
        'threshold_grid_size': len(thresholds),
        'threshold_range': (threshold_min, threshold_max),
        'threshold_step': threshold_step,
        'all_results': results
    }

    return calibration_result


def format_calibration_report(results: Dict) -> str:
    """
    Format calibration results into a human-readable report.

    Parameters
    ----------
    results : dict
        Output from calibrate_threshold().

    Returns
    -------
    str
        Formatted report string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("THRESHOLD CALIBRATION REPORT")
    lines.append("=" * 60)

    # Optimal result
    lines.append(f"\nOptimal Threshold: {results['optimal_threshold']:.4f}")
    lines.append(f"Max Coherence:     {results['max_coherence']:.2f}%")

    # Training window
    lines.append(f"\nTraining Window:")
    lines.append(f"  Start:  {results['train_start_date']}")
    lines.append(f"  End:    {results['train_end_date']}")
    lines.append(f"  Bars:   {results['n_train_bars']}")
    lines.append(f"  Indicators: {results['n_indicators']}")

    # Grid search summary
    lines.append(f"\nGrid Search:")
    lines.append(f"  Range:  [{results['threshold_range'][0]}, {results['threshold_range'][1]}]")
    lines.append(f"  Step:   {results['threshold_step']}")
    lines.append(f"  Tested: {results['threshold_grid_size']} thresholds")

    # Top 5 thresholds
    all_results = results['all_results']
    sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)
    lines.append(f"\nTop 5 Thresholds:")
    for i, (thresh, coh) in enumerate(sorted_results[:5]):
        lines.append(f"  {i+1}. Threshold={thresh:.4f}, Coherence={coh:.2f}%")

    # Bottom 5 thresholds
    lines.append(f"\nBottom 5 Thresholds:")
    for i, (thresh, coh) in enumerate(sorted_results[-5:]):
        lines.append(f"  {len(sorted_results)-4+i}. Threshold={thresh:.4f}, Coherence={coh:.2f}%")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def run_unit_tests():
    """
    Run unit tests for the threshold calibration module.

    Returns
    -------
    bool
        True if all tests pass, False otherwise.
    """
    print("=" * 60)
    print("Running Threshold Calibration Unit Tests")
    print("=" * 60)

    all_passed = True

    # Test 1: Basic calibration with synthetic data
    print("\nTest 1: Basic calibration with synthetic data")
    try:
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        date_strs = dates.strftime('%Y-%m-%d').tolist()

        # Create synthetic indicator signals
        n_indicators = 5
        signal_data = {}
        for i in range(n_indicators):
            # Random bullish/bearish signals
            signal_data[f'ind_{i}'] = np.random.choice([-1.0, 1.0], size=365)

        indicator_signals = pd.DataFrame(signal_data, index=date_strs)

        # Create ISP positions (simple trend-following pattern)
        isp_positions = pd.Series(
            [1.0 if i < 200 else 0.0 for i in range(365)],
            index=date_strs
        )

        # Calibrate threshold
        result = calibrate_threshold(
            df=pd.DataFrame({'close': np.random.randn(365) + 100}, index=date_strs),
            indicator_signals=indicator_signals,
            isp_positions=isp_positions,
            train_end_date='2020-12-31',
            lookback_months=12,
            threshold_min=-0.5,
            threshold_max=0.5,
            threshold_step=0.1  # Coarser grid for faster test
        )

        assert 'optimal_threshold' in result, "Missing optimal_threshold"
        assert 'max_coherence' in result, "Missing max_coherence"
        assert -0.5 <= result['optimal_threshold'] <= 0.5, \
            f"Optimal threshold {result['optimal_threshold']} out of range [-0.5, 0.5]"
        assert 0 <= result['max_coherence'] <= 100, \
            f"Coherence {result['max_coherence']} out of range [0, 100]"

        print(f"  ✓ Basic calibration works: threshold={result['optimal_threshold']:.2f}, "
              f"coherence={result['max_coherence']:.2f}%")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 2: Known optimal threshold recovery
    print("\nTest 2: Known optimal threshold recovery")
    try:
        np.random.seed(123)
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        date_strs = dates.strftime('%Y-%m-%d').tolist()

        # Create indicator signals that are mostly bullish in first 200 days
        # The ensemble average will be around 0.6, so threshold ~0.5 should be optimal
        signal_data = {}
        for i in range(10):
            # First 200 days: 80% bullish, last 165 days: 20% bullish
            signals = np.concatenate([
                np.random.choice([-1.0, 1.0], size=200, p=[0.2, 0.8]),
                np.random.choice([-1.0, 1.0], size=165, p=[0.8, 0.2])
            ])
            signal_data[f'ind_{i}'] = signals

        indicator_signals = pd.DataFrame(signal_data, index=date_strs)

        # ISP follows similar pattern
        isp_positions = pd.Series(
            [1.0 if i < 200 else 0.0 for i in range(365)],
            index=date_strs
        )

        # Calibrate with fine grid
        result = calibrate_threshold(
            df=pd.DataFrame({'close': np.random.randn(365) + 100}, index=date_strs),
            indicator_signals=indicator_signals,
            isp_positions=isp_positions,
            train_end_date='2020-12-31',
            lookback_months=12,
            threshold_min=-0.5,
            threshold_max=0.5,
            threshold_step=0.01
        )

        # With the signal pattern, threshold around 0.5 should give high coherence
        assert result['optimal_threshold'] >= 0.0, \
            f"Expected non-negative threshold for bullish-dominant signals, got {result['optimal_threshold']}"
        assert result['max_coherence'] > 50.0, \
            f"Expected >50% coherence, got {result['max_coherence']}%"

        print(f"  ✓ Threshold recovery works: threshold={result['optimal_threshold']:.2f}, "
              f"coherence={result['max_coherence']:.2f}%")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 3: Different train_end_date parameters
    print("\nTest 3: Different train_end_date parameters")
    try:
        np.random.seed(456)
        dates = pd.date_range('2020-01-01', periods=730, freq='D')  # 2 years
        date_strs = dates.strftime('%Y-%m-%d').tolist()

        # Create stable indicator signals
        signal_data = {}
        for i in range(5):
            signal_data[f'ind_{i}'] = np.random.choice([-1.0, 1.0], size=730)

        indicator_signals = pd.DataFrame(signal_data, index=date_strs)

        # ISP positions
        isp_positions = pd.Series(
            [1.0 if i % 365 < 200 else 0.0 for i in range(730)],
            index=date_strs
        )

        df = pd.DataFrame({'close': np.random.randn(730) + 100}, index=date_strs)

        # Test with different end dates
        result1 = calibrate_threshold(
            df, indicator_signals, isp_positions,
            train_end_date='2020-12-31',
            lookback_months=12,
            threshold_step=0.1
        )

        result2 = calibrate_threshold(
            df, indicator_signals, isp_positions,
            train_end_date='2021-06-30',
            lookback_months=12,
            threshold_step=0.1
        )

        assert result1['train_end_date'] == '2020-12-31', \
            f"Wrong train_end_date in result1: {result1['train_end_date']}"
        assert result2['train_end_date'] == '2021-06-30', \
            f"Wrong train_end_date in result2: {result2['train_end_date']}"
        assert result1['train_start_date'] == '2019-12-31', \
            f"Wrong train_start_date in result1: {result1['train_start_date']}"
        assert result2['train_start_date'] == '2020-06-30', \
            f"Wrong train_start_date in result2: {result2['train_start_date']}"

        print(f"  ✓ Different train_end_date parameters work correctly")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 4: Edge case - all same signals
    print("\nTest 4: Edge case - all bullish signals")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        date_strs = dates.strftime('%Y-%m-%d').tolist()

        # All bullish signals
        indicator_signals = pd.DataFrame({
            'ind_0': [1.0] * 100,
            'ind_1': [1.0] * 100,
        }, index=date_strs)

        # ISP always in market
        isp_positions = pd.Series([1.0] * 100, index=date_strs)

        result = calibrate_threshold(
            df=pd.DataFrame({'close': [100] * 100}, index=date_strs),
            indicator_signals=indicator_signals,
            isp_positions=isp_positions,
            train_end_date='2020-04-10',
            lookback_months=12,
            threshold_step=0.1
        )

        # With all bullish signals, any threshold < 1.0 should give 100% coherence
        assert result['max_coherence'] == 100.0, \
            f"Expected 100% coherence with all bullish signals, got {result['max_coherence']}%"
        print(f"  ✓ All bullish signals handled correctly: coherence={result['max_coherence']:.2f}%")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 5: Edge case - all bearish signals
    print("\nTest 5: Edge case - all bearish signals")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        date_strs = dates.strftime('%Y-%m-%d').tolist()

        # All bearish signals
        indicator_signals = pd.DataFrame({
            'ind_0': [-1.0] * 100,
            'ind_1': [-1.0] * 100,
        }, index=date_strs)

        # ISP always out of market
        isp_positions = pd.Series([0.0] * 100, index=date_strs)

        result = calibrate_threshold(
            df=pd.DataFrame({'close': [100] * 100}, index=date_strs),
            indicator_signals=indicator_signals,
            isp_positions=isp_positions,
            train_end_date='2020-04-10',
            lookback_months=12,
            threshold_step=0.1
        )

        # With all bearish signals, any threshold >= -1.0 should give 100% coherence
        assert result['max_coherence'] == 100.0, \
            f"Expected 100% coherence with all bearish signals, got {result['max_coherence']}%"
        print(f"  ✓ All bearish signals handled correctly: coherence={result['max_coherence']:.2f}%")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 6: Report formatting
    print("\nTest 6: Report formatting")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        date_strs = dates.strftime('%Y-%m-%d').tolist()

        indicator_signals = pd.DataFrame({
            'ind_0': [1.0] * 100,
            'ind_1': [-1.0] * 100,
        }, index=date_strs)

        isp_positions = pd.Series([1.0] * 100, index=date_strs)

        result = calibrate_threshold(
            df=pd.DataFrame({'close': [100] * 100}, index=date_strs),
            indicator_signals=indicator_signals,
            isp_positions=isp_positions,
            train_end_date='2020-04-10',
            lookback_months=12,
            threshold_step=0.2
        )

        report = format_calibration_report(result)

        assert "THRESHOLD CALIBRATION REPORT" in report
        assert "Optimal Threshold" in report
        assert "Max Coherence" in report
        assert "Top 5 Thresholds" in report

        print("  ✓ Report formatting works correctly")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 7: Error handling - empty inputs
    print("\nTest 7: Error handling - empty inputs")
    try:
        empty_signals = pd.DataFrame()
        empty_positions = pd.Series(dtype=float)

        try:
            calibrate_threshold(
                pd.DataFrame(),
                empty_signals,
                empty_positions,
                train_end_date='2020-12-31'
            )
            print("  ✗ FAILED: Should have raised ValueError for empty inputs")
            all_passed = False
        except ValueError:
            print("  ✓ Empty inputs correctly raises ValueError")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 8: Error handling - invalid threshold range
    print("\nTest 8: Error handling - invalid threshold range")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        date_strs = dates.strftime('%Y-%m-%d').tolist()

        indicator_signals = pd.DataFrame({'ind_0': [1.0] * 100}, index=date_strs)
        isp_positions = pd.Series([1.0] * 100, index=date_strs)

        try:
            calibrate_threshold(
                pd.DataFrame({'close': [100] * 100}, index=date_strs),
                indicator_signals,
                isp_positions,
                train_end_date='2020-04-10',
                threshold_min=0.5,
                threshold_max=-0.5  # Invalid: min > max
            )
            print("  ✗ FAILED: Should have raised ValueError for invalid range")
            all_passed = False
        except ValueError:
            print("  ✓ Invalid threshold range correctly raises ValueError")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 9: Custom weights
    print("\nTest 9: Custom weights")
    try:
        np.random.seed(789)
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        date_strs = dates.strftime('%Y-%m-%d').tolist()

        indicator_signals = pd.DataFrame({
            'ind_0': np.random.choice([-1.0, 1.0], size=100),
            'ind_1': np.random.choice([-1.0, 1.0], size=100),
        }, index=date_strs)

        isp_positions = pd.Series(
            [1.0 if i < 50 else 0.0 for i in range(100)],
            index=date_strs
        )

        weights = pd.Series({'ind_0': 0.7, 'ind_1': 0.3})

        result = calibrate_threshold(
            df=pd.DataFrame({'close': np.random.randn(100) + 100}, index=date_strs),
            indicator_signals=indicator_signals,
            isp_positions=isp_positions,
            train_end_date='2020-04-10',
            lookback_months=12,
            weights=weights,
            threshold_step=0.2
        )

        assert 'optimal_threshold' in result
        assert 'max_coherence' in result
        print(f"  ✓ Custom weights work correctly: threshold={result['optimal_threshold']:.2f}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 10: Coherence score verification
    print("\nTest 10: Coherence score is maximized")
    try:
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        date_strs = dates.strftime('%Y-%m-%d').tolist()

        # Create signals with clear pattern
        # First 100 days: bullish (avg ~0.8), last 100 days: bearish (avg ~-0.8)
        signal_data = {}
        for i in range(10):
            signals = np.concatenate([
                np.random.choice([-1.0, 1.0], size=100, p=[0.1, 0.9]),
                np.random.choice([-1.0, 1.0], size=100, p=[0.9, 0.1])
            ])
            signal_data[f'ind_{i}'] = signals

        indicator_signals = pd.DataFrame(signal_data, index=date_strs)

        # ISP matches the pattern
        isp_positions = pd.Series(
            [1.0 if i < 100 else 0.0 for i in range(200)],
            index=date_strs
        )

        # Test with fine grid
        result = calibrate_threshold(
            df=pd.DataFrame({'close': np.random.randn(200) + 100}, index=date_strs),
            indicator_signals=indicator_signals,
            isp_positions=isp_positions,
            train_end_date='2020-07-19',
            lookback_months=12,
            threshold_min=-0.5,
            threshold_max=0.5,
            threshold_step=0.05
        )

        # Verify the optimal threshold has higher coherence than random thresholds
        all_results = result['all_results']
        random_indices = np.random.choice(len(all_results), size=5, replace=False)
        random_coherences = [all_results[i][1] for i in random_indices]

        optimal_coherence = result['max_coherence']
        avg_random_coherence = np.mean(random_coherences)

        # Optimal should be at least as good as random average
        assert optimal_coherence >= avg_random_coherence - 1.0, \
            f"Optimal coherence ({optimal_coherence}) should be >= average random ({avg_random_coherence})"

        print(f"  ✓ Optimal coherence ({optimal_coherence:.2f}%) >= avg random ({avg_random_coherence:.2f}%)")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_unit_tests()
    if not success:
        exit(1)
