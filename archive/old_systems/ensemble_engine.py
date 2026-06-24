"""
Ensemble Engine for MTTD Trading System (Simplified)
====================================================

Pure majority-vote ensemble:
- Each indicator votes +1 (bullish) or -1 (bearish/neutral)
- Ensemble = mean of all indicator votes
- Position = 1 (100% BTC) if mean > 0, else 0 (0% cash)
- No threshold calibration, no EMA smoothing, no weight optimization

No Look-Ahead Bias:
- Mean is computed on current-bar signals only
- No future data is accessed in any computation
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict


def compute_ensemble_signal(
    indicator_signals: pd.DataFrame,
    min_hold: int = 1
) -> pd.DataFrame:
    """
    Compute ensemble binary position from multiple indicator signals.

    Parameters
    ----------
    indicator_signals : pd.DataFrame
        DataFrame where each column is an indicator signal.
        Values: +1.0 (bullish), -1.0 (bearish/neutral).
        Index: date strings or DatetimeIndex.

    min_hold : int, default 1
        Minimum number of bars to hold a position before flipping.
        Reduces whipsaws from noisy consensus.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'raw_average': Mean of raw signals (before min_hold filter)
        - 'position': Binary position (1.0 = 100%, 0.0 = 0%)
    """
    if indicator_signals.empty:
        raise ValueError("indicator_signals DataFrame is empty")

    if indicator_signals.isna().all().all():
        raise ValueError("indicator_signals contains only NaN values")

    n_indicators = indicator_signals.shape[1]

    # Step 1: Equal-weight average of all binary signals
    raw_average = indicator_signals.mean(axis=1)

    # Step 2: Binary position via majority vote (mean > 0 → bullish)
    raw_position = (raw_average > 0).astype(float)

    # Step 3: Enforce minimum hold period
    if min_hold > 1:
        position = raw_position.copy()
        last_change_idx = 0
        last_pos = position.iloc[0]

        for i in range(1, len(position)):
            if position.iloc[i] != last_pos:
                if i - last_change_idx >= min_hold:
                    last_change_idx = i
                    last_pos = position.iloc[i]
                else:
                    position.iloc[i] = last_pos
    else:
        position = raw_position

    result = pd.DataFrame({
        'raw_average': raw_average,
        'position': position
    }, index=indicator_signals.index)

    return result


def compute_ensemble_with_diagnostics(
    indicator_signals: pd.DataFrame,
    min_hold: int = 1
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute ensemble signal with diagnostic information.

    Returns both the ensemble result and diagnostic metrics.

    Parameters
    ----------
    indicator_signals : pd.DataFrame
        Indicator signals (+1/-1 binary).

    min_hold : int, default 1
        Minimum hold period.

    Returns
    -------
    Tuple[pd.DataFrame, dict]
        (ensemble_result, diagnostics)
    """
    result = compute_ensemble_signal(indicator_signals, min_hold=min_hold)

    position = result['position']
    n_bars = len(position)
    n_in_position = position.sum()
    pct_in_position = n_in_position / n_bars * 100 if n_bars > 0 else 0

    # Count position transitions (trades)
    position_diff = position.diff()
    n_entries = int((position_diff == 1).sum())
    n_exits = int((position_diff == -1).sum())
    n_trades = min(n_entries, n_exits)

    # Average signal strength by position
    in_position_avg = result.loc[position == 1, 'raw_average'].mean()
    out_position_avg = result.loc[position == 0, 'raw_average'].mean()

    diagnostics = {
        'n_indicators': indicator_signals.shape[1],
        'n_bars': n_bars,
        'pct_in_position': round(pct_in_position, 2),
        'n_entries': n_entries,
        'n_exits': n_exits,
        'n_trades': n_trades,
        'avg_signal_in_position': round(float(in_position_avg), 4) if not np.isnan(in_position_avg) else None,
        'avg_signal_out_position': round(float(out_position_avg), 4) if not np.isnan(out_position_avg) else None,
        'signal_range': (
            round(float(result['raw_average'].min()), 4),
            round(float(result['raw_average'].max()), 4)
        ),
        'min_hold': min_hold
    }

    return result, diagnostics


# --- Module-level helpers ---

def convert_indicator_direction_to_signals(
    direction_series: pd.Series
) -> pd.Series:
    """Convert indicator direction to binary signals (+1/-1)."""
    return direction_series.apply(lambda x: 1.0 if x > 0 else -1.0)


def build_signal_matrix(
    indicator_directions: Dict[str, pd.Series]
) -> pd.DataFrame:
    """Build signal matrix from indicator directions."""
    signals = {}
    for name, direction in indicator_directions.items():
        signals[name] = convert_indicator_direction_to_signals(direction)
    return pd.DataFrame(signals)


# --- Unit test function ---

def run_unit_tests():
    """Run unit tests for the ensemble engine."""
    print("=" * 60)
    print("Running Ensemble Engine Unit Tests")
    print("=" * 60)

    all_passed = True

    # Test 1: Basic majority vote
    print("\nTest 1: Basic majority vote")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        # 3 indicators: 2 bullish, 1 bearish → mean = 1/3 > 0 → position 1
        signals = pd.DataFrame({
            'ind1': [1.0] * 100,
            'ind2': [1.0] * 100,
            'ind3': [-1.0] * 100,
        }, index=dates)

        result = compute_ensemble_signal(signals, min_hold=1)

        assert 'position' in result.columns, "Missing position column"
        assert result['position'].sum() == 100, "All bullish majority → should be 100% in"
        print("  ✓ Majority vote works correctly")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 2: Binary output
    print("\nTest 2: Binary output verification")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        signals = pd.DataFrame({
            'ind1': [1.0] * 50 + [-1.0] * 50,
            'ind2': [-1.0] * 50 + [1.0] * 50,
        }, index=dates)

        result = compute_ensemble_signal(signals, min_hold=1)
        unique_positions = set(result['position'].unique())
        assert unique_positions <= {0.0, 1.0}, f"Positions must be binary, got: {unique_positions}"
        print("  ✓ Output is strictly binary")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 3: Equal weight
    print("\nTest 3: Equal weight verification")
    try:
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        signals = pd.DataFrame({
            'ind1': [1.0] * 10,
            'ind2': [-1.0] * 10,
        }, index=dates)

        result = compute_ensemble_signal(signals, min_hold=1)

        # 1 bullish + 1 bearish → mean = 0 → NOT > 0 → position 0
        assert result['position'].iloc[0] == 0.0, f"Equal split → position should be 0, got {result['position'].iloc[0]}"
        print("  ✓ Equal weight works correctly (tie → out)")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 4: Min hold period
    print("\nTest 4: Min hold period")
    try:
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        # Flip every bar (noisy)
        signals = pd.DataFrame({
            'ind1': [1.0 if i % 2 == 0 else -1.0 for i in range(20)],
            'ind2': [1.0 if i % 2 == 0 else -1.0 for i in range(20)],
        }, index=dates)

        # No min hold → many transitions
        result_no_hold = compute_ensemble_signal(signals, min_hold=1)
        transitions_no_hold = (result_no_hold['position'].diff().abs() > 0).sum()

        # Min hold = 5 → fewer transitions
        result_hold = compute_ensemble_signal(signals, min_hold=5)
        transitions_hold = (result_hold['position'].diff().abs() > 0).sum()

        assert transitions_hold <= transitions_no_hold, \
            f"Min hold ({transitions_hold}) should have fewer transitions than none ({transitions_no_hold})"
        print(f"  ✓ Min hold reduces transitions: {transitions_no_hold} → {transitions_hold}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 5: No look-ahead
    print("\nTest 5: No look-ahead verification")
    try:
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        np.random.seed(42)
        signals = pd.DataFrame({
            'ind1': np.random.choice([-1.0, 1.0], size=200),
            'ind2': np.random.choice([-1.0, 1.0], size=200),
            'ind3': np.random.choice([-1.0, 1.0], size=200),
        }, index=dates)

        # Compute on full data
        result_full = compute_ensemble_signal(signals, min_hold=1)

        # Compute on subset (last 50 bars only) — should be identical for those bars
        for test_idx in range(150, 200):
            subset_signals = signals.iloc[:test_idx + 1]
            result_subset = compute_ensemble_signal(subset_signals, min_hold=1)
            assert abs(result_full['position'].iloc[test_idx] - result_subset['position'].iloc[test_idx]) < 1e-10, \
                f"Look-ahead detected at bar {test_idx}"

        print("  ✓ No look-ahead bias detected")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 6: Empty DataFrame handling
    print("\nTest 6: Error handling")
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

    # Test 7: Diagnostics
    print("\nTest 7: Diagnostics computation")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        np.random.seed(123)
        signals = pd.DataFrame({
            'ind1': np.random.choice([-1.0, 1.0], size=100),
            'ind2': np.random.choice([-1.0, 1.0], size=100),
            'ind3': np.random.choice([-1.0, 1.0], size=100),
        }, index=dates)

        result, diagnostics = compute_ensemble_with_diagnostics(signals, min_hold=1)

        assert diagnostics['n_indicators'] == 3
        assert diagnostics['n_bars'] == 100
        assert diagnostics['min_hold'] == 1
        print("  ✓ Diagnostics computed correctly")
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
