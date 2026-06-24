"""
Coherence Metrics Module for MTTD Trading System
==================================================

Measures alignment between MTTD position signals and ISP benchmark positions.

Metrics computed:
1. Time-Coherence % — percentage of bars where both agree on position (in/out)
2. Timing Error — average days difference between trade entries/exits
3. Return Correlation — Pearson and Spearman correlation of daily returns
4. Trade Counts — number of trades in each series
5. Agreement Stats — detailed breakdown of agreement/disagreement periods

PASS/FAIL Verdict:
- PASS if time-coherence ≥ 95%
- FAIL otherwise
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from scipy import stats


def load_isp_positions(
    csv_path: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.Series:
    """
    Load ISP benchmark positions from CSV file.

    Converts ISP's 3-tier regime (Strong Bull=100%, Weak Bull=50%, Neutral=0%)
    to a binary position series (1.0 for in-market, 0.0 for out-of-market).

    The ISP CSV format:
    - BUY trades enter positions (EquityPct = 50 for Weak Bull, 100 for Strong Bull)
    - SELL trades exit to Neutral (Regime = 'Neutral' means out of market)
    - The Regime column indicates the state AFTER the trade

    Parameters
    ----------
    csv_path : str
        Path to ISP signals CSV file.

    start_date : str or None, default None
        Start date filter (inclusive). Format: 'YYYY-MM-DD'.

    end_date : str or None, default None
        End date filter (inclusive). Format: 'YYYY-MM-DD'.

    Returns
    -------
    pd.Series
        Binary position series (1.0 or 0.0) indexed by date strings.
        Uses forward-fill to create daily positions from trade signals.
    """
    df = pd.read_csv(csv_path)

    if 'Date' not in df.columns or 'EquityPct' not in df.columns:
        raise ValueError(f"CSV must have 'Date' and 'EquityPct' columns. Got: {list(df.columns)}")

    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()

    # Convert Regime to binary position
    # Strong Bull = 1.0, Weak Bull = 1.0, Neutral = 0.0
    # The Regime column indicates the position state AFTER the trade
    if 'Regime' in df.columns:
        regime_map = {
            'Strong Bull': 1.0,
            'Weak Bull': 1.0,
            'Neutral': 0.0
        }
        df['position'] = df['Regime'].map(regime_map)
    else:
        # Fallback: use BUY/SELL action
        # BUY → in market, SELL to 0% → out of market
        df['position'] = 0.0
        df.loc[df['Action'] == 'BUY', 'position'] = 1.0
        # For SELL actions, check if going to Neutral (EquityPct indicates target)
        # If EquityPct = 100 on SELL, it means selling 100% to go to 0%
        sell_neutral = (df['Action'] == 'SELL') & (df['EquityPct'] == 100)
        df.loc[sell_neutral, 'position'] = 0.0

    # Create a daily index and forward-fill positions
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df_daily = df.reindex(full_idx)
    df_daily['position'] = df_daily['position'].ffill()

    # Apply date filters
    if start_date is not None:
        df_daily = df_daily[df_daily.index >= start_date]
    if end_date is not None:
        df_daily = df_daily[df_daily.index <= end_date]

    # Convert index to string for compatibility with MTTD system
    position_series = df_daily['position']
    position_series.index = position_series.index.strftime('%Y-%m-%d')

    return position_series


def build_mttd_position_series(
    position_data: list,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.Series:
    """
    Build a daily position series from MTTD position data.

    Parameters
    ----------
    position_data : list of dict
        List of position records with 'time' and 'value' keys.
        'value' should be 1.0 (in market) or 0.0 (out of market).

    start_date : str or None
        Start date filter.

    end_date : str or None
        End date filter.

    Returns
    -------
    pd.Series
        Daily position series (1.0 or 0.0) indexed by date strings.
    """
    df = pd.DataFrame(position_data)
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()

    # Ensure binary values
    df['position'] = df['value'].astype(float)

    # Create daily index and forward-fill
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df_daily = df.reindex(full_idx)
    df_daily['position'] = df_daily['position'].ffill()

    # Apply date filters
    if start_date is not None:
        df_daily = df_daily[df_daily.index >= start_date]
    if end_date is not None:
        df_daily = df_daily[df_daily.index <= end_date]

    position_series = df_daily['position']
    position_series.index = position_series.index.strftime('%Y-%m-%d')

    return position_series


def compute_time_coherence(
    mttd_positions: pd.Series,
    isp_positions: pd.Series
) -> Dict:
    """
    Compute time-coherence between MTTD and ISP positions.

    Time-coherence = (number of bars where both agree) / (total overlapping bars) * 100

    Agreement means:
    - Both in market (1.0)
    - Both out of market (0.0)

    Parameters
    ----------
    mttd_positions : pd.Series
        MTTD binary position series (1.0 or 0.0).

    isp_positions : pd.Series
        ISP binary position series (1.0 or 0.0).

    Returns
    -------
    dict
        Dictionary with:
        - coherence_pct: percentage of time in agreement
        - n_agree: number of bars in agreement
        - n_disagree: number of bars in disagreement
        - n_total: total overlapping bars
        - both_in_pct: percentage of time both are in market
        - both_out_pct: percentage of time both are out of market
        - mttd_in_isp_out_pct: time MTTD in, ISP out
        - mttd_out_isp_in_pct: time MTTD out, ISP in
    """
    # Align on common index
    common_idx = mttd_positions.index.intersection(isp_positions.index)

    if len(common_idx) == 0:
        return {
            'coherence_pct': 0.0,
            'n_agree': 0,
            'n_disagree': 0,
            'n_total': 0,
            'both_in_pct': 0.0,
            'both_out_pct': 0.0,
            'mttd_in_isp_out_pct': 0.0,
            'mttd_out_isp_in_pct': 0.0,
            'error': 'No overlapping dates found'
        }

    mttd = mttd_positions.loc[common_idx]
    isp = isp_positions.loc[common_idx]

    n_total = len(common_idx)

    # Compute agreement categories
    both_in = ((mttd == 1.0) & (isp == 1.0)).sum()
    both_out = ((mttd == 0.0) & (isp == 0.0)).sum()
    mttd_in_isp_out = ((mttd == 1.0) & (isp == 0.0)).sum()
    mttd_out_isp_in = ((mttd == 0.0) & (isp == 1.0)).sum()

    n_agree = both_in + both_out
    n_disagree = mttd_in_isp_out + mttd_out_isp_in

    coherence_pct = n_agree / n_total * 100 if n_total > 0 else 0.0

    return {
        'coherence_pct': round(coherence_pct, 4),
        'n_agree': int(n_agree),
        'n_disagree': int(n_disagree),
        'n_total': int(n_total),
        'both_in_pct': round(both_in / n_total * 100, 4) if n_total > 0 else 0.0,
        'both_out_pct': round(both_out / n_total * 100, 4) if n_total > 0 else 0.0,
        'mttd_in_isp_out_pct': round(mttd_in_isp_out / n_total * 100, 4) if n_total > 0 else 0.0,
        'mttd_out_isp_in_pct': round(mttd_out_isp_in / n_total * 100, 4) if n_total > 0 else 0.0,
    }


def extract_trade_dates(
    positions: pd.Series
) -> Tuple[List[str], List[str]]:
    """
    Extract entry and exit dates from a position series.

    Parameters
    ----------
    positions : pd.Series
        Binary position series (1.0 or 0.0).

    Returns
    -------
    Tuple[List[str], List[str]]
        (entry_dates, exit_dates) as lists of date strings.
    """
    entries = []
    exits = []

    prev_pos = 0.0
    for i, (date, pos) in enumerate(positions.items()):
        if prev_pos == 0.0 and pos == 1.0:
            entries.append(date)
        elif prev_pos == 1.0 and pos == 0.0:
            exits.append(date)
        prev_pos = pos

    return entries, exits


def compute_timing_error(
    mttd_positions: pd.Series,
    isp_positions: pd.Series
) -> Dict:
    """
    Compute average timing error between trade entries.

    For each ISP trade entry, find the nearest MTTD entry and compute
    the day difference.

    Parameters
    ----------
    mttd_positions : pd.Series
        MTTD binary position series.

    isp_positions : pd.Series
        ISP binary position series.

    Returns
    -------
    dict
        Dictionary with:
        - avg_entry_timing_error_days: average absolute days between entries
        - avg_exit_timing_error_days: average absolute days between exits
        - entry_errors: list of individual entry timing errors
        - exit_errors: list of individual exit timing errors
        - mttd_n_trades: number of MTtD round-trip trades
        - isp_n_trades: number of ISP round-trip trades
    """
    mttd_entries, mttd_exits = extract_trade_dates(mttd_positions)
    isp_entries, isp_exits = extract_trade_dates(isp_positions)

    def compute_nearest_date_errors(source_dates, target_dates):
        """For each source date, find nearest target date and compute error."""
        if not source_dates or not target_dates:
            return [], []

        source_dt = [pd.Timestamp(d) for d in source_dates]
        target_dt = [pd.Timestamp(d) for d in target_dates]

        errors = []
        for s in source_dt:
            min_err = min(abs(s - t).days for t in target_dt)
            errors.append(min_err)

        return errors, [f"{e} days" for e in errors]

    entry_errors, entry_error_strs = compute_nearest_date_errors(isp_entries, mttd_entries)
    exit_errors, exit_error_strs = compute_nearest_date_errors(isp_exits, mttd_exits)

    avg_entry_err = np.mean(entry_errors) if entry_errors else float('nan')
    avg_exit_err = np.mean(exit_errors) if exit_errors else float('nan')

    # Count round-trip trades (entries followed by exits)
    mttd_n_trades = min(len(mttd_entries), len(mttd_exits))
    isp_n_trades = min(len(isp_entries), len(isp_exits))

    return {
        'avg_entry_timing_error_days': round(float(avg_entry_err), 2) if not np.isnan(avg_entry_err) else None,
        'avg_exit_timing_error_days': round(float(avg_exit_err), 2) if not np.isnan(avg_exit_err) else None,
        'entry_errors': entry_error_strs,
        'exit_errors': exit_error_strs,
        'mttd_n_entries': len(mttd_entries),
        'mttd_n_exits': len(mttd_exits),
        'isp_n_entries': len(isp_entries),
        'isp_n_exits': len(isp_exits),
        'mttd_n_trades': mttd_n_trades,
        'isp_n_trades': isp_n_trades,
    }


def compute_return_correlation(
    mttd_positions: pd.Series,
    isp_positions: pd.Series,
    price_series: pd.Series
) -> Dict:
    """
    Compute return correlation between MTTD and ISP strategies.

    Parameters
    ----------
    mttd_positions : pd.Series
        MTTD binary position series.

    isp_positions : pd.Series
        ISP binary position series.

    price_series : pd.Series
        Daily close price series with matching index.

    Returns
    -------
    dict
        Dictionary with:
        - pearson_corr: Pearson correlation of daily strategy returns
        - spearman_corr: Spearman rank correlation of daily strategy returns
        - daily_returns_corr: correlation of daily return series
    """
    # Align all series on common index
    common_idx = mttd_positions.index.intersection(
        isp_positions.index
    ).intersection(price_series.index)

    if len(common_idx) < 10:
        return {
            'pearson_corr': None,
            'spearman_corr': None,
            'n_common_days': len(common_idx),
            'error': 'Insufficient common data points (< 10)'
        }

    mttd = mttd_positions.loc[common_idx]
    isp = isp_positions.loc[common_idx]
    price = price_series.loc[common_idx]

    # Compute daily returns
    daily_returns = price.pct_change().fillna(0.0)

    # Compute strategy returns
    mttd_returns = mttd.shift(1).fillna(0.0) * daily_returns
    isp_returns = isp.shift(1).fillna(0.0) * daily_returns

    # Remove first bar (NaN from shift)
    mttd_returns = mttd_returns.iloc[1:]
    isp_returns = isp_returns.iloc[1:]

    if len(mttd_returns) < 5:
        return {
            'pearson_corr': None,
            'spearman_corr': None,
            'n_common_days': len(common_idx),
            'error': 'Insufficient data after shift'
        }

    # Pearson correlation
    pearson_corr, pearson_pval = stats.pearsonr(mttd_returns, isp_returns)

    # Spearman correlation
    spearman_corr, spearman_pval = stats.spearmanr(mttd_returns, isp_returns)

    # Position agreement correlation (skip if constant)
    if mttd.nunique() > 1 and isp.nunique() > 1:
        pos_corr, pos_pval = stats.pearsonr(mttd, isp)
    else:
        pos_corr, pos_pval = 1.0 if (mttd == isp).all() else 0.0, 1.0

    return {
        'pearson_corr': round(float(pearson_corr), 4),
        'pearson_pval': round(float(pearson_pval), 6),
        'spearman_corr': round(float(spearman_corr), 4),
        'spearman_pval': round(float(spearman_pval), 6),
        'position_agreement_corr': round(float(pos_corr), 4),
        'n_common_days': len(common_idx),
    }


def measure_coherence(
    mttd_positions: pd.Series,
    isp_positions: pd.Series,
    price_series: Optional[pd.Series] = None,
    coherence_threshold: float = 95.0
) -> Dict:
    """
    Comprehensive coherence measurement between MTTD and ISP positions.

    This is the main entry point for coherence analysis.

    Parameters
    ----------
    mttd_positions : pd.Series
        MTTD binary position series (1.0 or 0.0), indexed by date strings.

    isp_positions : pd.Series
        ISP binary position series (1.0 or 0.0), indexed by date strings.

    price_series : pd.Series or None, default None
        Daily close price series for return correlation analysis.
        If None, return correlation metrics will be skipped.

    coherence_threshold : float, default 95.0
        Minimum time-coherence percentage for PASS verdict.

    Returns
    -------
    dict
        Complete coherence measurement results including:
        - time_coherence: time-coherence metrics
        - timing_error: trade timing error metrics
        - return_correlation: return correlation metrics (if price provided)
        - trade_comparison: trade count comparison
        - verdict: PASS/FAIL with reason
    """
    # Validate inputs
    if mttd_positions.empty:
        raise ValueError("mttd_positions is empty")
    if isp_positions.empty:
        raise ValueError("isp_positions is empty")

    # Validate that positions are binary
    mttd_unique = set(mttd_positions.unique())
    isp_unique = set(isp_positions.unique())

    expected = {0.0, 1.0}
    if not mttd_unique.issubset(expected):
        raise ValueError(f"mttd_positions must be binary (0.0 or 1.0). Got: {mttd_unique}")
    if not isp_unique.issubset(expected):
        raise ValueError(f"isp_positions must be binary (0.0 or 1.0). Got: {isp_unique}")

    # Compute time-coherence
    time_coherence = compute_time_coherence(mttd_positions, isp_positions)

    # Compute timing error
    timing_error = compute_timing_error(mttd_positions, isp_positions)

    # Compute return correlation (if price data provided)
    if price_series is not None:
        return_correlation = compute_return_correlation(
            mttd_positions, isp_positions, price_series
        )
    else:
        return_correlation = {'skipped': True, 'reason': 'No price_series provided'}

    # Compute trade count comparison
    trade_comparison = {
        'mttd_trades': timing_error['mttd_n_trades'],
        'isp_trades': timing_error['isp_n_trades'],
        'trade_count_ratio': round(
            timing_error['mttd_n_trades'] / timing_error['isp_n_trades'], 2
        ) if timing_error['isp_n_trades'] > 0 else float('inf'),
    }

    # Determine PASS/FAIL verdict
    coherence_pct = time_coherence['coherence_pct']
    passed = coherence_pct >= coherence_threshold

    verdict = {
        'passed': passed,
        'threshold': coherence_threshold,
        'actual_coherence': coherence_pct,
        'reason': (
            f"Time-coherence {coherence_pct:.2f}% {'>=' if passed else '<'} "
            f"{coherence_threshold:.2f}% threshold"
        ),
    }

    return {
        'time_coherence': time_coherence,
        'timing_error': timing_error,
        'return_correlation': return_correlation,
        'trade_comparison': trade_comparison,
        'verdict': verdict,
    }


def format_coherence_report(results: Dict) -> str:
    """
    Format coherence measurement results into a human-readable report.

    Parameters
    ----------
    results : dict
        Output from measure_coherence().

    Returns
    -------
    str
        Formatted report string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("COHERENCE METRICS REPORT")
    lines.append("=" * 60)

    # Verdict
    verdict = results['verdict']
    verdict_symbol = "✓ PASS" if verdict['passed'] else "✗ FAIL"
    lines.append(f"\nVerdict: {verdict_symbol}")
    lines.append(f"  Threshold: {verdict['threshold']:.2f}%")
    lines.append(f"  Actual:    {verdict['actual_coherence']:.2f}%")
    lines.append(f"  Reason:    {verdict['reason']}")

    # Time Coherence
    tc = results['time_coherence']
    lines.append(f"\n--- Time Coherence ---")
    lines.append(f"  Coherence %:        {tc['coherence_pct']:.2f}%")
    lines.append(f"  Bars in agreement:  {tc['n_agree']} / {tc['n_total']}")
    lines.append(f"  Bars in disagreement: {tc['n_disagree']}")
    lines.append(f"  Both in market:     {tc['both_in_pct']:.2f}%")
    lines.append(f"  Both out of market: {tc['both_out_pct']:.2f}%")
    lines.append(f"  MTTD in, ISP out:   {tc['mttd_in_isp_out_pct']:.2f}%")
    lines.append(f"  MTTD out, ISP in:   {tc['mttd_out_isp_in_pct']:.2f}%")

    # Timing Error
    te = results['timing_error']
    lines.append(f"\n--- Timing Error ---")
    lines.append(f"  MTTD trades:  {te['mttd_n_trades']} ({te['mttd_n_entries']} entries, {te['mttd_n_exits']} exits)")
    lines.append(f"  ISP trades:   {te['isp_n_trades']} ({te['isp_n_entries']} entries, {te['isp_n_exits']} exits)")
    if te['avg_entry_timing_error_days'] is not None:
        lines.append(f"  Avg entry error: {te['avg_entry_timing_error_days']:.1f} days")
        lines.append(f"  Avg exit error:  {te['avg_exit_timing_error_days']:.1f} days")
        if te['entry_errors']:
            lines.append(f"  Entry errors: {', '.join(te['entry_errors'][:5])}")
            lines.append(f"  Exit errors:  {', '.join(te['exit_errors'][:5])}")
    else:
        lines.append(f"  No trade timing errors to compute (missing entries or exits)")

    # Return Correlation
    rc = results['return_correlation']
    lines.append(f"\n--- Return Correlation ---")
    if rc.get('skipped'):
        lines.append(f"  Skipped: {rc['reason']}")
    elif rc.get('error'):
        lines.append(f"  Error: {rc['error']}")
    else:
        lines.append(f"  Pearson correlation:  {rc['pearson_corr']:.4f} (p={rc['pearson_pval']:.6f})")
        lines.append(f"  Spearman correlation: {rc['spearman_corr']:.4f} (p={rc['spearman_pval']:.6f})")
        lines.append(f"  Position agreement:   {rc['position_agreement_corr']:.4f}")
        lines.append(f"  Common days:          {rc['n_common_days']}")

    # Trade Comparison
    tr = results['trade_comparison']
    lines.append(f"\n--- Trade Comparison ---")
    lines.append(f"  MTTD trades: {tr['mttd_trades']}")
    lines.append(f"  ISP trades:  {tr['isp_trades']}")
    lines.append(f"  Ratio:       {tr['trade_count_ratio']:.2f}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def run_unit_tests():
    """
    Run unit tests for the coherence metrics module.

    Returns
    -------
    bool
        True if all tests pass, False otherwise.
    """
    print("=" * 60)
    print("Running Coherence Metrics Unit Tests")
    print("=" * 60)

    all_passed = True

    # Test 1: Identical positions → 100% coherence
    print("\nTest 1: Identical positions → 100% coherence")
    try:
        dates = [f"2020-01-{i+1:02d}" for i in range(30)]
        positions = pd.Series([1.0] * 15 + [0.0] * 15, index=dates)

        tc = compute_time_coherence(positions, positions)
        assert tc['coherence_pct'] == 100.0, f"Expected 100%, got {tc['coherence_pct']}%"
        assert tc['n_disagree'] == 0, f"Expected 0 disagreements, got {tc['n_disagree']}"
        print("  ✓ Identical positions yield 100% coherence")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 2: Opposite positions → ~0% coherence
    print("\nTest 2: Opposite positions → ~0% coherence")
    try:
        dates = [f"2020-01-{i+1:02d}" for i in range(30)]
        mttd_pos = pd.Series([1.0] * 15 + [0.0] * 15, index=dates)
        isp_pos = pd.Series([0.0] * 15 + [1.0] * 15, index=dates)

        tc = compute_time_coherence(mttd_pos, isp_pos)
        assert tc['coherence_pct'] == 0.0, f"Expected 0%, got {tc['coherence_pct']}%"
        assert tc['n_agree'] == 0, f"Expected 0 agreements, got {tc['n_agree']}"
        print("  ✓ Opposite positions yield 0% coherence")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 3: Partial match → correct percentage
    print("\nTest 3: Partial match → correct percentage")
    try:
        dates = [f"2020-01-{i+1:02d}" for i in range(20)]
        # Both agree on first 15, disagree on last 5
        mttd_pos = pd.Series([1.0] * 20, index=dates)
        isp_pos = pd.Series([1.0] * 15 + [0.0] * 5, index=dates)

        tc = compute_time_coherence(mttd_pos, isp_pos)
        expected_pct = 15 / 20 * 100  # 75%
        assert abs(tc['coherence_pct'] - expected_pct) < 0.01, \
            f"Expected {expected_pct}%, got {tc['coherence_pct']}%"
        print(f"  ✓ Partial match: {tc['coherence_pct']:.2f}% (expected {expected_pct}%)")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 4: Trade date extraction
    print("\nTest 4: Trade date extraction")
    try:
        dates = [f"2020-01-{i+1:02d}" for i in range(10)]
        positions = pd.Series([0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], index=dates)

        entries, exits = extract_trade_dates(positions)
        assert len(entries) == 2, f"Expected 2 entries, got {len(entries)}"
        assert len(exits) == 2, f"Expected 2 exits, got {len(exits)}"
        assert entries[0] == '2020-01-03', f"First entry should be 2020-01-03, got {entries[0]}"
        assert exits[0] == '2020-01-06', f"First exit should be 2020-01-06, got {exits[0]}"
        print(f"  ✓ Trade dates extracted: {len(entries)} entries, {len(exits)} exits")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 5: Timing error computation
    print("\nTest 5: Timing error computation")
    try:
        dates = [f"2020-01-{i+1:02d}" for i in range(30)]
        # MTTD enters on day 3, ISP enters on day 5 (2 day error)
        mttd_pos = pd.Series([0.0] * 2 + [1.0] * 28, index=dates)
        isp_pos = pd.Series([0.0] * 4 + [1.0] * 26, index=dates)

        te = compute_timing_error(mttd_pos, isp_pos)
        assert te['avg_entry_timing_error_days'] == 2.0, \
            f"Expected 2.0 days error, got {te['avg_entry_timing_error_days']}"
        print(f"  ✓ Timing error computed: {te['avg_entry_timing_error_days']} days")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 6: Return correlation with matching positions
    print("\nTest 6: Return correlation with matching positions")
    try:
        np.random.seed(42)
        dates = [f"2020-01-{i+1:02d}" for i in range(30)]
        price = pd.Series([100.0 * (1.01 ** i) for i in range(30)], index=dates)

        # Same positions → should have high correlation
        positions = pd.Series([1.0] * 20 + [0.0] * 10, index=dates)
        rc = compute_return_correlation(positions, positions, price)

        assert rc['pearson_corr'] == 1.0, f"Expected pearson=1.0, got {rc['pearson_corr']}"
        assert rc['spearman_corr'] == 1.0, f"Expected spearman=1.0, got {rc['spearman_corr']}"
        print(f"  ✓ Matching positions: Pearson={rc['pearson_corr']}, Spearman={rc['spearman_corr']}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 7: measure_coherence full pipeline
    print("\nTest 7: measure_coherence full pipeline")
    try:
        np.random.seed(42)
        dates = [f"2020-01-{i+1:02d}" for i in range(50)]
        price = pd.Series([100.0 * (1.005 ** i) for i in range(50)], index=dates)

        mttd_pos = pd.Series([1.0] * 30 + [0.0] * 20, index=dates)
        isp_pos = pd.Series([1.0] * 28 + [0.0] * 22, index=dates)

        results = measure_coherence(mttd_pos, isp_pos, price)

        assert 'time_coherence' in results
        assert 'timing_error' in results
        assert 'return_correlation' in results
        assert 'trade_comparison' in results
        assert 'verdict' in results

        tc_pct = results['time_coherence']['coherence_pct']
        assert 90.0 <= tc_pct <= 100.0, f"Coherence should be 90-100%, got {tc_pct}%"

        print(f"  ✓ Full pipeline works: coherence={tc_pct:.2f}%, verdict={'PASS' if results['verdict']['passed'] else 'FAIL'}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 8: PASS/FAIL verdict at 95% threshold
    print("\nTest 8: PASS/FAIL verdict at 95% threshold")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D').strftime('%Y-%m-%d').tolist()

        # 98% agreement → PASS
        mttd_pos = pd.Series([1.0] * 50 + [0.0] * 50, index=dates)
        isp_pos = pd.Series([1.0] * 52 + [0.0] * 48, index=dates)

        results = measure_coherence(mttd_pos, isp_pos, coherence_threshold=95.0)
        assert results['verdict']['passed'], "98% coherence should PASS at 95% threshold"

        # 80% agreement → FAIL
        mttd_pos2 = pd.Series([1.0] * 50 + [0.0] * 50, index=dates)
        isp_pos2 = pd.Series([1.0] * 30 + [0.0] * 70, index=dates)

        results2 = measure_coherence(mttd_pos2, isp_pos2, coherence_threshold=95.0)
        assert not results2['verdict']['passed'], "80% coherence should FAIL at 95% threshold"

        print("  ✓ PASS/FAIL verdict works correctly at 95% threshold")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 9: Format report
    print("\nTest 9: Format report")
    try:
        dates = [f"2020-01-{i+1:02d}" for i in range(50)]
        mttd_pos = pd.Series([1.0] * 30 + [0.0] * 20, index=dates)
        isp_pos = pd.Series([1.0] * 28 + [0.0] * 22, index=dates)

        results = measure_coherence(mttd_pos, isp_pos)
        report = format_coherence_report(results)

        assert "COHERENCE METRICS REPORT" in report
        assert "PASS" in report or "FAIL" in report
        assert "Time Coherence" in report
        assert "Timing Error" in report

        print("  ✓ Report formatted successfully")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 10: Empty input handling
    print("\nTest 10: Edge case - no overlapping dates")
    try:
        dates1 = [f"2020-01-{i+1:02d}" for i in range(10)]
        dates2 = [f"2020-02-{i+1:02d}" for i in range(10)]

        mttd_pos = pd.Series([1.0] * 10, index=dates1)
        isp_pos = pd.Series([1.0] * 10, index=dates2)

        tc = compute_time_coherence(mttd_pos, isp_pos)
        assert tc['n_total'] == 0, "Should have 0 overlapping bars"
        assert tc['coherence_pct'] == 0.0, "Coherence should be 0% with no overlap"
        print("  ✓ No overlapping dates handled gracefully")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 11: Invalid input handling
    print("\nTest 11: Invalid input handling")
    try:
        dates = [f"2020-01-{i+1:02d}" for i in range(10)]
        # Non-binary positions
        bad_pos = pd.Series([0.5] * 10, index=dates)
        good_pos = pd.Series([1.0] * 10, index=dates)

        try:
            measure_coherence(bad_pos, good_pos)
            print("  ✗ FAILED: Should have raised ValueError for non-binary positions")
            all_passed = False
        except ValueError as e:
            if "binary" in str(e).lower():
                print("  ✓ Non-binary positions correctly raises ValueError")
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
