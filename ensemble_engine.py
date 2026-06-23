"""
Ensemble Engine for MTTD Trading System
========================================

Aggregates multiple indicator signals into a single binary position (100% or 0%).

Logic:
1. Sum individual binary signals (+1 for long, -1 for short/neutral)
2. Compute average signal (normalized by indicator count)
3. Apply EMA smoothing to reduce noise and whipsaws
4. Apply threshold to produce final binary position (100% or 0%)

No Look-Ahead Bias:
- EMA uses pandas ewm with adjust=False (causal filter, only uses past/current data)
- Threshold comparison is point-in-time (current bar only)
- No future data is accessed in any computation
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, List


def compute_ensemble_signal(
    indicator_signals: pd.DataFrame,
    threshold: float = 0.0,
    ema_length: int = 5,
    weights: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Compute ensemble binary position from multiple indicator signals.

    Parameters
    ----------
    indicator_signals : pd.DataFrame
        DataFrame where each column is an indicator signal.
        Values should be:
        - 1.0 for bullish/long signal
        - -1.0 for bearish/neutral signal
        - 0.0 for neutral (if supported by indicator)
        Index should be date strings or DatetimeIndex.

    threshold : float, default 0.0
        Threshold for binary position decision.
        If smoothed_average > threshold → position = 1 (100%)
        If smoothed_average <= threshold → position = 0 (0%)
        Range: typically between -1.0 and 1.0

    ema_length : int, default 5
        Length of EMA smoothing filter.
        Higher values = smoother but more lag.
        Lower values = more responsive but noisier.

    weights : pd.Series or None, default None
        Optional weights for each indicator (index = indicator names).
        If None, equal weighting (1/N) is applied.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'raw_average': Average of raw signals (before smoothing)
        - 'smoothed_average': EMA-smoothed average
        - 'position': Binary position (1.0 = 100%, 0.0 = 0%)
        Index matches input indicator_signals.

    Notes
    -----
    No look-ahead bias:
    - EMA is causal (uses only current and past values)
    - Threshold comparison is point-in-time
    - Weights are applied equally across all bars (no future adaptation)
    """
    if indicator_signals.empty:
        raise ValueError("indicator_signals DataFrame is empty")

    if indicator_signals.isna().all().all():
        raise ValueError("indicator_signals contains only NaN values")

    n_indicators = indicator_signals.shape[1]

    # Validate signal values
    valid_values = {-1.0, 0.0, 1.0, -1, 0, 1}
    all_values = set(indicator_signals.dropna().values.flatten())
    unexpected = all_values - valid_values
    if unexpected:
        # Allow any values in [-1, 1] range, just warn about unusual ones
        unusual = [v for v in unexpected if v < -1 or v > 1]
        if unusual:
            print(f"Warning: indicator_signals contains unexpected values: {unusual}")

    # Apply weights (equal weighting by default)
    if weights is None:
        weights = pd.Series(1.0 / n_indicators, index=indicator_signals.columns)
    else:
        # Normalize weights to sum to 1
        weights = weights / weights.sum()

    # Ensure weights align with indicator columns
    missing_cols = set(indicator_signals.columns) - set(weights.index)
    extra_cols = set(weights.index) - set(indicator_signals.columns)
    if missing_cols:
        raise ValueError(f"Weights missing for indicators: {missing_cols}")
    if extra_cols:
        # Drop extra weights not in indicator_signals
        weights = weights[indicator_signals.columns]

    # Step 1 & 2: Compute weighted average signal
    # raw_average = sum(signal_i * weight_i) for each time step
    raw_average = indicator_signals.mul(weights, axis=1).sum(axis=1)

    # Step 3: Apply EMA smoothing (causal filter - no look-ahead)
    # Using pandas ewm with adjust=False ensures the filter only uses past/current data
    smoothed_average = raw_average.ewm(span=ema_length, adjust=False, min_periods=1).mean()

    # Step 4: Apply threshold to produce binary position
    # position = 1 (100%) if smoothed_average > threshold, else 0 (0%)
    position = (smoothed_average > threshold).astype(float)

    # Build result DataFrame
    result = pd.DataFrame({
        'raw_average': raw_average,
        'smoothed_average': smoothed_average,
        'position': position
    }, index=indicator_signals.index)

    return result


def verify_no_look_ahead(
    ensemble_result: pd.DataFrame,
    indicator_signals: pd.DataFrame,
    ema_length: int = 5,
    n_test_bars: int = 50
) -> Dict:
    """
    Verify that the ensemble engine produces no look-ahead bias.

    This test checks that the position at time t does not depend on
    indicator signals at time t+1, t+2, etc.

    Parameters
    ----------
    ensemble_result : pd.DataFrame
        Output from compute_ensemble_signal.

    indicator_signals : pd.DataFrame
        Input indicator signals.

    ema_length : int
        EMA length used.

    n_test_bars : int
        Number of bars to test from the end.

    Returns
    -------
    dict
        Verification results with 'passed' bool and 'details' string.
    """
    results = {
        'test_name': 'no_look_ahead_verification',
        'passed': True,
        'details': '',
        'tests_run': [],
        'tests_passed': 0,
        'tests_failed': 0
    }

    if len(ensemble_result) < n_test_bars:
        n_test_bars = len(ensemble_result) // 2

    if n_test_bars < 5:
        results['passed'] = False
        results['details'] = 'Insufficient data for look-ahead verification'
        return results

    test_indices = list(range(len(ensemble_result) - n_test_bars, len(ensemble_result)))

    for test_idx in test_indices:
        # Compute ensemble on subset [0:test_idx+1]
        subset_signals = indicator_signals.iloc[:test_idx + 1]
        subset_result = compute_ensemble_signal(
            subset_signals,
            ema_length=ema_length,
            threshold=0.0  # Use default threshold for comparison
        )

        # Check: position at test_idx should be identical
        full_position = ensemble_result['position'].iloc[test_idx]
        subset_position = subset_result['position'].iloc[test_idx]

        test_passed = abs(full_position - subset_position) < 1e-10
        test_name = f'bar_{test_idx}'

        results['tests_run'].append({
            'bar_index': test_idx,
            'full_position': full_position,
            'subset_position': subset_position,
            'passed': test_passed
        })

        if test_passed:
            results['tests_passed'] += 1
        else:
            results['tests_failed'] += 1
            results['passed'] = False

    results['details'] = (
        f"Tested {len(results['tests_run'])} bars: "
        f"{results['tests_passed']} passed, {results['tests_failed']} failed"
    )

    return results


def compute_ensemble_with_diagnostics(
    indicator_signals: pd.DataFrame,
    threshold: float = 0.0,
    ema_length: int = 5,
    weights: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute ensemble signal with diagnostic information.

    Returns both the ensemble result and diagnostic metrics.

    Parameters
    ----------
    indicator_signals : pd.DataFrame
        Indicator signals as in compute_ensemble_signal.

    threshold : float
        Threshold for position decision.

    ema_length : int
        EMA smoothing length.

    weights : pd.Series or None
        Optional indicator weights.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        (ensemble_result, diagnostics)
    """
    # Compute ensemble
    result = compute_ensemble_signal(
        indicator_signals,
        threshold=threshold,
        ema_length=ema_length,
        weights=weights
    )

    # Run look-ahead verification
    look_ahead = verify_no_look_ahead(
        result,
        indicator_signals,
        ema_length=ema_length,
        n_test_bars=min(50, len(indicator_signals) // 2)
    )

    # Compute additional diagnostics
    position = result['position']
    n_bars = len(position)
    n_in_position = position.sum()
    pct_in_position = n_in_position / n_bars * 100 if n_bars > 0 else 0

    # Count position transitions (trades)
    position_diff = position.diff()
    n_entries = (position_diff == 1).sum()  # 0 -> 1 transitions
    n_exits = (position_diff == -1).sum()   # 1 -> 0 transitions
    n_trades = min(n_entries, n_exits)

    # Compute average signal strength by position
    in_position_avg = result.loc[position == 1, 'raw_average'].mean()
    out_position_avg = result.loc[position == 0, 'raw_average'].mean()

    diagnostics = {
        'threshold': threshold,
        'ema_length': ema_length,
        'n_indicators': indicator_signals.shape[1],
        'n_bars': n_bars,
        'pct_in_position': round(pct_in_position, 2),
        'n_entries': int(n_entries),
        'n_exits': int(n_exits),
        'n_trades': int(n_trades),
        'avg_signal_in_position': round(in_position_avg, 4) if not np.isnan(in_position_avg) else None,
        'avg_signal_out_position': round(out_position_avg, 4) if not np.isnan(out_position_avg) else None,
        'signal_range': (
            round(result['raw_average'].min(), 4),
            round(result['raw_average'].max(), 4)
        ),
        'smoothed_range': (
            round(result['smoothed_average'].min(), 4),
            round(result['smoothed_average'].max(), 4)
        ),
        'look_ahead_verification': look_ahead
    }

    return result, diagnostics


# --- Module-level helpers for integration with execute_system.py ---

def convert_indicator_direction_to_signals(
    direction_series: pd.Series
) -> pd.Series:
    """
    Convert an indicator's direction series to binary signals for ensemble.

    Parameters
    ----------
    direction_series : pd.Series
        Raw direction values from indicator.
        Positive = bullish, negative = bearish, zero = neutral.

    Returns
    -------
    pd.Series
        Binary signals: 1.0 for long, -1.0 for short/neutral.
    """
    return direction_series.apply(lambda x: 1.0 if x > 0 else -1.0)


def build_signal_matrix(
    indicator_directions: Dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Build a matrix of indicator signals for ensemble computation.

    Parameters
    ----------
    indicator_directions : dict
        Dictionary mapping indicator name to direction series.
        Each series should have same index as price data.

    Returns
    -------
    pd.DataFrame
        Matrix where each column is an indicator's binary signal.
    """
    signals = {}
    for name, direction in indicator_directions.items():
        signals[name] = convert_indicator_direction_to_signals(direction)

    return pd.DataFrame(signals)


# --- Unit test function ---

def run_unit_tests():
    """
    Run unit tests for the ensemble engine.
    Returns True if all tests pass, False otherwise.
    """
    print("=" * 60)
    print("Running Ensemble Engine Unit Tests")
    print("=" * 60)

    all_passed = True

    # Test 1: Basic functionality
    print("\nTest 1: Basic ensemble computation")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        signals = pd.DataFrame({
            'ind1': [1.0] * 50 + [-1.0] * 50,
            'ind2': [1.0] * 60 + [-1.0] * 40,
            'ind3': [-1.0] * 40 + [1.0] * 60,
        }, index=dates)

        result = compute_ensemble_signal(signals, threshold=0.0, ema_length=5)

        assert 'position' in result.columns, "Missing position column"
        assert 'smoothed_average' in result.columns, "Missing smoothed_average column"
        assert 'raw_average' in result.columns, "Missing raw_average column"

        positions = result['position'].values
        assert all(p in [0.0, 1.0] for p in positions), "Positions must be 0.0 or 1.0"
        print("  ✓ Basic computation works")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 2: Binary output verification
    print("\nTest 2: Verify output is strictly binary")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        # Use clear bullish then bearish signals
        signals = pd.DataFrame({
            'ind1': [1.0] * 50 + [-1.0] * 50,
            'ind2': [1.0] * 50 + [-1.0] * 50,
        }, index=dates)

        result = compute_ensemble_signal(signals, threshold=0.0, ema_length=1)

        unique_positions = set(result['position'].unique())
        assert unique_positions == {0.0, 1.0}, f"Positions must be binary, got: {unique_positions}"
        print("  ✓ Output is strictly binary (0.0 or 1.0)")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 3: Threshold effect
    print("\nTest 3: Threshold effect on position")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        # All bullish signals
        signals = pd.DataFrame({
            'ind1': [1.0] * 100,
            'ind2': [1.0] * 100,
            'ind3': [1.0] * 100,
        }, index=dates)

        # With threshold = 0.0, should be all in position
        result_low = compute_ensemble_signal(signals, threshold=0.0, ema_length=1)
        assert result_low['position'].sum() == 100, "All signals bullish, threshold 0: should be 100% in"

        # With threshold = 0.9, should be all in position (avg = 1.0 > 0.9)
        result_high = compute_ensemble_signal(signals, threshold=0.9, ema_length=1)
        assert result_high['position'].sum() == 100, "All signals bullish, threshold 0.9: should be 100% in"

        # With threshold = 1.0, should be all out (avg = 1.0, not > 1.0)
        result_very_high = compute_ensemble_signal(signals, threshold=1.0, ema_length=1)
        assert result_very_high['position'].sum() == 0, "All signals bullish, threshold 1.0: should be 0% in"

        print("  ✓ Threshold correctly affects position")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 4: EMA smoothing effect
    print("\nTest 4: EMA smoothing effect")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        # Alternating signals
        signals = pd.DataFrame({
            'ind1': [1.0 if i % 2 == 0 else -1.0 for i in range(100)],
            'ind2': [1.0 if i % 2 == 0 else -1.0 for i in range(100)],
        }, index=dates)

        # No smoothing (ema_length=1)
        result_no_smooth = compute_ensemble_signal(signals, threshold=0.0, ema_length=1)

        # Heavy smoothing (ema_length=20)
        result_smooth = compute_ensemble_signal(signals, threshold=0.0, ema_length=20)

        # Smoothed version should have fewer transitions
        transitions_no_smooth = (result_no_smooth['position'].diff().abs() > 0).sum()
        transitions_smooth = (result_smooth['position'].diff().abs() > 0).sum()

        assert transitions_smooth <= transitions_no_smooth, \
            f"Smoothed ({transitions_smooth}) should have fewer transitions than unsmoothed ({transitions_no_smooth})"

        print("  ✓ EMA smoothing reduces transitions")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 5: No look-ahead bias
    print("\nTest 5: No look-ahead bias verification")
    try:
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        np.random.seed(42)
        signals = pd.DataFrame({
            'ind1': np.random.choice([-1.0, 1.0], size=200),
            'ind2': np.random.choice([-1.0, 1.0], size=200),
            'ind3': np.random.choice([-1.0, 1.0], size=200),
        }, index=dates)

        result = compute_ensemble_signal(signals, threshold=0.0, ema_length=5)
        verification = verify_no_look_ahead(result, signals, ema_length=5, n_test_bars=30)

        assert verification['passed'], f"Look-ahead verification failed: {verification['details']}"
        print("  ✓ No look-ahead bias detected")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 6: Equal weighting by default
    print("\nTest 6: Equal weighting verification")
    try:
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        signals = pd.DataFrame({
            'ind1': [1.0] * 10,
            'ind2': [-1.0] * 10,
        }, index=dates)

        result = compute_ensemble_signal(signals, threshold=0.0, ema_length=1)

        # With equal weight and 1 bullish + 1 bearish, average should be 0.0
        # After EMA with min_periods=1, first bar should be 0.0
        assert abs(result['raw_average'].iloc[0]) < 1e-10, \
            f"Expected raw_average ~0.0, got {result['raw_average'].iloc[0]}"

        print("  ✓ Equal weighting works correctly")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 7: Custom weights
    print("\nTest 7: Custom weights")
    try:
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        signals = pd.DataFrame({
            'ind1': [1.0] * 10,
            'ind2': [-1.0] * 10,
        }, index=dates)

        weights = pd.Series({'ind1': 0.75, 'ind2': 0.25})

        result = compute_ensemble_signal(signals, threshold=0.0, ema_length=1, weights=weights)

        # With weights 0.75*1.0 + 0.25*(-1.0) = 0.75 - 0.25 = 0.5
        expected_raw = 0.75 * 1.0 + 0.25 * (-1.0)
        assert abs(result['raw_average'].iloc[0] - expected_raw) < 1e-10, \
            f"Expected raw_average ~{expected_raw}, got {result['raw_average'].iloc[0]}"

        print("  ✓ Custom weights work correctly")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 8: Position transitions create markers correctly
    print("\nTest 8: Position transitions")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        # Start neutral, flip to bullish, then bearish
        signals = pd.DataFrame({
            'ind1': [-1.0] * 10 + [1.0] * 40 + [-1.0] * 50,
            'ind2': [-1.0] * 10 + [1.0] * 40 + [-1.0] * 50,
        }, index=dates)

        result = compute_ensemble_signal(signals, threshold=0.0, ema_length=1)

        position_diff = result['position'].diff()
        entries = (position_diff == 1).sum()
        exits = (position_diff == -1).sum()

        # Should have 1 entry (at bar 10) and 1 exit (at bar 50)
        assert entries >= 1, f"Expected at least 1 entry, got {entries}"
        assert exits >= 1, f"Expected at least 1 exit, got {exits}"

        print(f"  ✓ Position transitions: {entries} entries, {exits} exits")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 9: Empty DataFrame handling
    print("\nTest 9: Error handling")
    try:
        empty_signals = pd.DataFrame()
        try:
            compute_ensemble_signal(empty_signals)
            print("  ✗ FAILED: Should have raised ValueError")
            all_passed = False
        except ValueError:
            print("  ✓ Empty DataFrame raises ValueError correctly")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 10: Verify no look-ahead with diagnostics
    print("\nTest 10: Compute with diagnostics")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(123)
        signals = pd.DataFrame({
            'ind1': np.random.choice([-1.0, 1.0], size=100),
            'ind2': np.random.choice([-1.0, 1.0], size=100),
            'ind3': np.random.choice([-1.0, 1.0], size=100),
        }, index=dates)

        result, diagnostics = compute_ensemble_with_diagnostics(
            signals, threshold=0.0, ema_length=5
        )

        assert 'look_ahead_verification' in diagnostics
        assert diagnostics['look_ahead_verification']['passed']
        assert diagnostics['n_indicators'] == 3
        assert diagnostics['n_bars'] == 100

        print("  ✓ Diagnostics computed correctly with look-ahead verification")
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
