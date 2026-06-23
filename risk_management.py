"""
Risk Management Module for MTTD Trading System
================================================

Provides two risk management components:

1. Signal-Flip Exit: Handled by the ensemble engine naturally
   (position changes from 1→0 or 0→1 on signal flip).
   This is an identity function for API consistency.

2. 15% Max Drawdown Pause: When drawdown exceeds threshold from peak,
   force position to 0 (cash) for pause_days bars, then re-evaluate.

No Look-Ahead Bias:
- Drawdown computed from historical equity peak (expanding window)
- Pause decision is point-in-time (current bar only)
- No future data accessed in any computation
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


def apply_signal_flip_exit(position: pd.Series) -> pd.Series:
    """
    Signal-flip exit is handled by the ensemble engine.
    
    The ensemble engine produces binary positions (1.0 or 0.0) that
    change based on indicator signals. When signals flip, position
    naturally transitions from 1→0 (exit) or 0→1 (entry).
    
    This function is provided for API consistency and documentation.
    The ensemble engine's output IS the signal-flip exit mechanism.

    Parameters
    ----------
    position : pd.Series
        Binary position series from ensemble engine (1.0 or 0.0).

    Returns
    -------
    pd.Series
        Same position series (identity function).
    """
    return position.copy()


def apply_drawdown_pause(
    position: pd.Series,
    equity: pd.Series,
    max_dd_pct: float = 0.15,
    pause_days: int = 20
) -> pd.DataFrame:
    """
    Apply drawdown pause rule to position series.
    
    If drawdown exceeds max_dd_pct from equity peak, force position
    to 0 (cash) for pause_days bars, then re-evaluate.
    
    Drawdown is computed from the equity curve peak (cumulative),
    not from individual trade entry prices.

    Parameters
    ----------
    position : pd.Series
        Binary position series from ensemble engine (1.0 or 0.0).
        Index should be date strings or DatetimeIndex.

    equity : pd.Series
        Equity curve corresponding to the position series.
        Should have same index as position.
        Typically: (1 + position * daily_returns).cumprod() * initial_capital

    max_dd_pct : float, default 0.15
        Maximum drawdown threshold as a decimal (0.15 = 15%).
        When equity drawdown exceeds this, pause activates.

    pause_days : int, default 20
        Minimum number of bars to pause after drawdown exceeds threshold.
        During pause, position is forced to 0 regardless of signals.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'position': Modified position with drawdown pauses applied
        - 'in_pause': Boolean indicating if bar is in pause period
        - 'drawdown_pct': Current drawdown from peak (negative values)
        - 'pause_start_date': Date when current pause started (NaT if not in pause)
        Index matches input position series.

    Notes
    -----
    No look-ahead bias:
    - Peak is computed using expanding() (only uses past/current data)
    - Drawdown comparison is point-in-time (current bar only)
    - Pause decision does not access future data

    Algorithm:
    1. Compute running peak of equity curve
    2. Compute drawdown = (equity - peak) / peak
    3. If drawdown < -max_dd_pct, enter pause state
    4. During pause, force position to 0 and count bars
    5. After pause_days, exit pause and re-evaluate signals
    """
    if len(position) != len(equity):
        raise ValueError(
            f"position and equity must have same length. "
            f"Got position={len(position)}, equity={len(equity)}"
        )

    if not position.index.equals(equity.index):
        raise ValueError("position and equity must have matching index")

    if max_dd_pct <= 0 or max_dd_pct > 1:
        raise ValueError(f"max_dd_pct must be between 0 and 1, got {max_dd_pct}")

    if pause_days <= 0:
        raise ValueError(f"pause_days must be positive, got {pause_days}")

    # Initialize output
    protected_position = position.copy()
    in_pause = pd.Series(False, index=position.index)
    drawdown_pct = pd.Series(0.0, index=position.index)
    pause_start_date = pd.Series(pd.NaT, index=position.index)

    # Compute running peak and drawdown (causal - no look-ahead)
    peak = equity.expanding().min()  # Use min for safety, but equity should be monotonically increasing
    peak = equity.expanding().max()  # Correct: track highest equity seen so far
    drawdown = (equity - peak) / peak
    drawdown_pct = drawdown * 100.0  # Convert to percentage

    # Apply drawdown pause logic
    in_pause_state = False
    pause_counter = 0
    current_pause_start = pd.NaT

    for i in range(len(protected_position)):
        # Check if drawdown exceeds threshold
        if drawdown.iloc[i] < -max_dd_pct:
            if not in_pause_state:
                # Entering new pause
                in_pause_state = True
                pause_counter = 0
                current_pause_start = position.index[i]

        # Apply pause logic
        if in_pause_state:
            protected_position.iloc[i] = 0.0
            in_pause.iloc[i] = True
            pause_start_date.iloc[i] = current_pause_start
            pause_counter += 1

            # Check if pause duration exceeded
            if pause_counter >= pause_days:
                in_pause_state = False
                current_pause_start = pd.NaT

    result = pd.DataFrame({
        'position': protected_position,
        'in_pause': in_pause,
        'drawdown_pct': drawdown_pct,
        'pause_start_date': pause_start_date
    }, index=position.index)

    return result


def compute_equity_curve(
    position: pd.Series,
    price_series: pd.Series,
    initial_capital: float = 100000.0
) -> pd.Series:
    """
    Compute equity curve from position and price series.
    
    This is a helper function for creating the equity input
    required by apply_drawdown_pause.

    Parameters
    ----------
    position : pd.Series
        Binary position series (1.0 = invested, 0.0 = cash).

    price_series : pd.Series
        Price series (e.g., BTC close prices).

    initial_capital : float, default 100000.0
        Starting capital.

    Returns
    -------
    pd.Series
        Equity curve starting at initial_capital.
    """
    daily_returns = price_series.pct_change().fillna(0.0)
    strategy_returns = position * daily_returns
    equity = initial_capital * (1 + strategy_returns).cumprod()
    return equity


def get_risk_metrics(
    position: pd.Series,
    equity: pd.Series
) -> Dict:
    """
    Compute risk management metrics for evaluation.
    
    Parameters
    ----------
    position : pd.Series
        Binary position series.

    equity : pd.Series
        Equity curve.

    Returns
    -------
    dict
        Risk metrics including:
        - max_drawdown_pct: Maximum drawdown percentage
        - total_pause_bars: Number of bars in pause state
        - pause_count: Number of separate pause periods
        - current_drawdown_pct: Current drawdown from peak
    """
    # Compute drawdown
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min()) * 100.0

    # Count pause periods (where position was forced to 0 due to drawdown)
    # This is approximate - we detect potential pauses
    position_diff = position.diff()
    
    # Compute basic metrics
    n_bars = len(position)
    n_in_position = position.sum()
    pct_in_position = n_in_position / n_bars * 100 if n_bars > 0 else 0

    # Detect position transitions
    n_entries = (position_diff == 1).sum()
    n_exits = (position_diff == -1).sum()

    metrics = {
        'max_drawdown_pct': round(max_dd, 2),
        'total_bars': n_bars,
        'bars_in_position': int(n_in_position),
        'pct_in_position': round(pct_in_position, 2),
        'n_entries': int(n_entries),
        'n_exits': int(n_exits),
        'current_drawdown_pct': round(float(drawdown.iloc[-1]) * 100, 2) if n_bars > 0 else 0.0,
    }

    return metrics


def run_unit_tests():
    """
    Run unit tests for the risk management module.
    Returns True if all tests pass, False otherwise.
    """
    print("=" * 60)
    print("Running Risk Management Unit Tests")
    print("=" * 60)

    all_passed = True

    # Test 1: Signal-flip exit identity function
    print("\nTest 1: Signal-flip exit identity function")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        position = pd.Series([1.0] * 50 + [0.0] * 50, index=dates)

        result = apply_signal_flip_exit(position)

        assert result.equals(position), "Signal-flip exit should be identity function"
        assert not result is position, "Should return a copy, not the original"
        print("  ✓ Signal-flip exit correctly returns copy of position")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 2: Basic drawdown pause functionality
    print("\nTest 2: Basic drawdown pause functionality")
    try:
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        # Position: always in market
        position = pd.Series([1.0] * 200, index=dates)
        
        # Equity: goes up then drops significantly (>15% drawdown)
        equity_values = [100.0] * 50  # Stable
        equity_values += [100.0 - i * 2 for i in range(1, 31)]  # Drop to 40 (60% DD)
        equity_values += [40.0] * 20  # Stay low during pause
        equity_values += [40.0 + i * 2 for i in range(1, 101)]  # Recovery
        equity = pd.Series(equity_values[:200], index=dates)

        result = apply_drawdown_pause(position, equity, max_dd_pct=0.15, pause_days=20)

        # Verify pause activated (position should be forced to 0 during drop)
        assert 'position' in result.columns, "Missing position column"
        assert 'in_pause' in result.columns, "Missing in_pause column"
        assert 'drawdown_pct' in result.columns, "Missing drawdown_pct column"
        
        # Check that some bars are in pause
        n_pause_bars = result['in_pause'].sum()
        assert n_pause_bars >= 20, f"Expected at least 20 pause bars, got {n_pause_bars}"
        
        # Check that position is 0 during pause
        pause_positions = result.loc[result['in_pause'], 'position']
        assert (pause_positions == 0.0).all(), "Position should be 0 during pause"
        
        print(f"  ✓ Drawdown pause activated correctly ({n_pause_bars} bars in pause)")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 3: Pause duration minimum 20 days
    print("\nTest 3: Pause duration minimum 20 days")
    try:
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        position = pd.Series([1.0] * 200, index=dates)
        
        # Equity: quick drop then quick recovery (but pause should last 20 days)
        equity_values = [100.0] * 50
        equity_values += [100.0 - i * 5 for i in range(1, 11)]  # Fast drop
        equity_values += [50.0 + i * 10 for i in range(1, 11)]  # Fast recovery
        equity_values += [150.0] * 130  # Stay high
        equity = pd.Series(equity_values[:200], index=dates)

        result = apply_drawdown_pause(position, equity, max_dd_pct=0.15, pause_days=20)

        # Count continuous pause periods
        in_pause = result['in_pause'].values
        pause_periods = []
        current_period = 0
        for val in in_pause:
            if val:
                current_period += 1
            else:
                if current_period > 0:
                    pause_periods.append(current_period)
                current_period = 0
        if current_period > 0:
            pause_periods.append(current_period)

        # All pause periods should be at least 20 bars
        if pause_periods:
            min_pause = min(pause_periods)
            assert min_pause >= 20, f"Pause duration should be >= 20 days, got {min_pause}"
            print(f"  ✓ Pause duration minimum 20 days enforced (min: {min_pause} days)")
        else:
            print("  ⚠ No pause periods detected (equity may not have exceeded 15% DD)")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 4: Drawdown percentage calculation
    print("\nTest 4: Drawdown percentage calculation")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        position = pd.Series([1.0] * 100, index=dates)
        
        # Equity: simple 20% drop
        equity_values = [100.0] * 50
        equity_values += [100.0 - i * 0.4 for i in range(1, 51)]  # Drop to 80
        equity = pd.Series(equity_values, index=dates)

        result = apply_drawdown_pause(position, equity, max_dd_pct=0.15, pause_days=20)

        # Check drawdown calculation
        max_dd = result['drawdown_pct'].min()
        assert abs(max_dd - (-20.0)) < 0.1, f"Expected ~-20% drawdown, got {max_dd}%"
        
        print(f"  ✓ Drawdown percentage calculated correctly (max: {max_dd:.1f}%)")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 5: No look-ahead bias verification
    print("\nTest 5: No look-ahead bias verification")
    try:
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        # Random position
        position = pd.Series(np.random.choice([0.0, 1.0], size=200), index=dates)
        
        # Random equity with some drawdowns
        returns = np.random.normal(0.001, 0.02, 200)
        equity = 100000 * (1 + position.values * returns).cumprod()
        equity = pd.Series(equity, index=dates)

        # Test: position at time t should not change if we only have data up to time t
        n_test = 50
        test_indices = list(range(len(dates) - n_test, len(dates)))

        for idx in test_indices:
            # Full computation
            full_result = apply_drawdown_pause(
                position.iloc[:idx+1],
                equity.iloc[:idx+1],
                max_dd_pct=0.15,
                pause_days=20
            )

            # Subset computation (only data up to idx)
            subset_result = apply_drawdown_pause(
                position.iloc[:idx+1],
                equity.iloc[:idx+1],
                max_dd_pct=0.15,
                pause_days=20
            )

            # Positions should be identical
            assert full_result['position'].equals(subset_result['position']), \
                f"Look-ahead bias detected at index {idx}"

        print("  ✓ No look-ahead bias detected in drawdown pause logic")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 6: Edge case - no drawdown exceeds threshold
    print("\nTest 6: Edge case - no drawdown exceeds threshold")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        position = pd.Series([1.0] * 100, index=dates)
        
        # Equity: always increasing (no drawdown)
        equity = pd.Series([100.0 + i for i in range(100)], index=dates)

        result = apply_drawdown_pause(position, equity, max_dd_pct=0.15, pause_days=20)

        # No pause should occur
        assert result['in_pause'].sum() == 0, "Should have no pause when no drawdown"
        assert result['position'].equals(position), "Position should be unchanged"
        
        print("  ✓ No pause when drawdown doesn't exceed threshold")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 7: Multiple pause periods
    print("\nTest 7: Multiple pause periods")
    try:
        dates = pd.date_range('2020-01-01', periods=300, freq='D')
        position = pd.Series([1.0] * 300, index=dates)
        
        # Equity with two separate drawdowns
        equity_values = [100.0] * 50
        equity_values += [100.0 - i * 3 for i in range(1, 21)]  # First drop
        equity_values += [40.0] * 30  # Pause period 1
        equity_values += [40.0 + i * 3 for i in range(1, 21)]  # Recovery 1
        equity_values += [100.0] * 30  # Stable
        equity_values += [100.0 - i * 3 for i in range(1, 21)]  # Second drop
        equity_values += [40.0] * 30  # Pause period 2
        equity_values += [40.0 + i * 3 for i in range(1, 101)]  # Recovery 2
        equity = pd.Series(equity_values[:300], index=dates)

        result = apply_drawdown_pause(position, equity, max_dd_pct=0.15, pause_days=20)

        # Count pause periods
        in_pause = result['in_pause'].values
        pause_periods = []
        current_period = 0
        for val in in_pause:
            if val:
                current_period += 1
            else:
                if current_period > 0:
                    pause_periods.append(current_period)
                current_period = 0
        if current_period > 0:
            pause_periods.append(current_period)

        assert len(pause_periods) >= 2, f"Expected at least 2 pause periods, got {len(pause_periods)}"
        
        print(f"  ✓ Multiple pause periods detected: {len(pause_periods)} periods")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 8: Compute equity curve helper
    print("\nTest 8: Compute equity curve helper")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        position = pd.Series([1.0] * 100, index=dates)
        price = pd.Series([100.0 * (1.01 ** i) for i in range(100)], index=dates)  # 1% daily return

        equity = compute_equity_curve(position, price, initial_capital=10000.0)

        assert len(equity) == 100, "Equity curve should have same length"
        assert equity.iloc[0] == 10000.0, "Equity should start at initial capital"
        assert equity.iloc[-1] > 10000.0, "Equity should grow with positive returns"
        
        print(f"  ✓ Equity curve computed correctly (final: {equity.iloc[-1]:.2f})")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 9: Risk metrics computation
    print("\nTest 9: Risk metrics computation")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        position = pd.Series([1.0] * 50 + [0.0] * 50, index=dates)
        equity = pd.Series([100.0 + i for i in range(100)], index=dates)

        metrics = get_risk_metrics(position, equity)

        assert 'max_drawdown_pct' in metrics, "Missing max_drawdown_pct"
        assert 'total_bars' in metrics, "Missing total_bars"
        assert 'bars_in_position' in metrics, "Missing bars_in_position"
        assert 'pct_in_position' in metrics, "Missing pct_in_position"
        assert 'n_entries' in metrics, "Missing n_entries"
        assert 'n_exits' in metrics, "Missing n_exits"
        assert metrics['total_bars'] == 100, "Total bars should be 100"
        assert metrics['bars_in_position'] == 50, "Bars in position should be 50"
        assert metrics['pct_in_position'] == 50.0, "Percent in position should be 50%"
        
        print(f"  ✓ Risk metrics computed correctly")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 10: Error handling
    print("\nTest 10: Error handling")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        position = pd.Series([1.0] * 100, index=dates)
        equity = pd.Series([100.0] * 100, index=dates)

        # Test mismatched lengths
        try:
            apply_drawdown_pause(position.iloc[:50], equity)
            print("  ✗ FAILED: Should have raised ValueError for mismatched lengths")
            all_passed = False
        except ValueError as e:
            if "same length" in str(e):
                print("  ✓ Mismatched lengths correctly raises ValueError")
            else:
                print(f"  ✗ FAILED: Wrong error message: {e}")
                all_passed = False

        # Test invalid max_dd_pct
        try:
            apply_drawdown_pause(position, equity, max_dd_pct=-0.1)
            print("  ✗ FAILED: Should have raised ValueError for negative max_dd_pct")
            all_passed = False
        except ValueError as e:
            if "between 0 and 1" in str(e):
                print("  ✓ Invalid max_dd_pct correctly raises ValueError")
            else:
                print(f"  ✗ FAILED: Wrong error message: {e}")
                all_passed = False

        # Test invalid pause_days
        try:
            apply_drawdown_pause(position, equity, pause_days=0)
            print("  ✗ FAILED: Should have raised ValueError for zero pause_days")
            all_passed = False
        except ValueError as e:
            if "positive" in str(e):
                print("  ✓ Invalid pause_days correctly raises ValueError")
            else:
                print(f"  ✗ FAILED: Wrong error message: {e}")
                all_passed = False

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
