"""
Inter-Indicator Coherence Measurement for MTTD Trading System
=============================================================

Measures time-alignment between individual indicators and with ISP benchmark.

Metrics:
- Pairwise coherence: fraction of time two indicators agree
- Individual ISP coherence: fraction of time each indicator matches ISP
- Aggregate coherence: average/min pairwise coherence across all pairs
- Flip rate: how often each indicator changes signal (stability)
- Agreement window: consecutive bars where majority agrees
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def compute_pairwise_coherence(signal_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise coherence matrix between all indicators.

    Parameters
    ----------
    signal_matrix : pd.DataFrame
        Binary signals (+1/-1) for each indicator. Columns = indicators.

    Returns
    -------
    pd.DataFrame
        N×N matrix where entry (i,j) = fraction of time indicators i and j agree.
        Diagonal = 1.0 (self-coherence).
    """
    indicators = signal_matrix.columns.tolist()
    n = len(indicators)
    coherence = pd.DataFrame(np.eye(n), index=indicators, columns=indicators)

    for i in range(n):
        for j in range(i + 1, n):
            agree = (signal_matrix.iloc[:, i] == signal_matrix.iloc[:, j]).sum()
            total = len(signal_matrix)
            coh = agree / total if total > 0 else 0.0
            coherence.iloc[i, j] = coh
            coherence.iloc[j, i] = coh

    return coherence


def compute_individual_isp_coherence(
    signal_matrix: pd.DataFrame,
    isp_positions: pd.Series
) -> Dict[str, float]:
    """
    Compute coherence of each individual indicator with ISP benchmark.

    Parameters
    ----------
    signal_matrix : pd.DataFrame
        Binary signals (+1/-1) for each indicator.
    isp_positions : pd.Series
        ISP position series (1.0 = in market, 0.0 = out).

    Returns
    -------
    dict
        {indicator_name: coherence_pct}
    """
    # Convert ISP to binary (+1/-1) for comparison
    isp_binary = isp_positions.apply(lambda x: 1.0 if x > 0 else -1.0)

    result = {}
    for col in signal_matrix.columns:
        aligned = pd.DataFrame({
            'signal': signal_matrix[col],
            'isp': isp_binary
        }).dropna()

        if len(aligned) == 0:
            result[col] = 0.0
            continue

        agree = (aligned['signal'] == aligned['isp']).sum()
        result[col] = round(agree / len(aligned) * 100, 2)

    return result


def compute_flip_rates(signal_matrix: pd.DataFrame) -> Dict[str, float]:
    """
    Compute flip rate (signal changes per bar) for each indicator.

    Parameters
    ----------
    signal_matrix : pd.DataFrame
        Binary signals (+1/-1) for each indicator.

    Returns
    -------
    dict
        {indicator_name: flip_rate} where flip_rate = number_of_flips / n_bars
    """
    result = {}
    n_bars = len(signal_matrix)

    for col in signal_matrix.columns:
        flips = (signal_matrix[col].diff().abs() > 0).sum()
        result[col] = round(flips / n_bars, 4) if n_bars > 0 else 0.0

    return result


def compute_agreement_windows(
    signal_matrix: pd.DataFrame,
    min_agreement: float = 0.6
) -> Dict:
    """
    Compute statistics about agreement windows.

    An agreement window is a consecutive run where at least min_agreement
    fraction of indicators agree on the same signal.

    Parameters
    ----------
    signal_matrix : pd.DataFrame
        Binary signals (+1/-1) for each indicator.
    min_agreement : float
        Minimum fraction of indicators that must agree (0.0 to 1.0).

    Returns
    -------
    dict
        - 'n_windows': number of agreement windows
        - 'avg_length': average window length in bars
        - 'max_length': longest window in bars
        - 'bullish_pct': percentage of bars in bullish agreement
        - 'bearish_pct': percentage of bars in bearish agreement
        - 'neutral_pct': percentage of bars without agreement
    """
    n_indicators = signal_matrix.shape[1]
    n_bars = len(signal_matrix)

    if n_bars == 0 or n_indicators == 0:
        return {
            'n_windows': 0, 'avg_length': 0, 'max_length': 0,
            'bullish_pct': 0, 'bearish_pct': 0, 'neutral_pct': 100
        }

    # Compute agreement ratio per bar
    bullish_count = (signal_matrix == 1.0).sum(axis=1)
    bearish_count = (signal_matrix == -1.0).sum(axis=1)
    max_agreement = pd.concat([bullish_count, bearish_count], axis=1).max(axis=1)
    agreement_ratio = max_agreement / n_indicators

    # Classify each bar
    bullish_bars = (bullish_count / n_indicators >= min_agreement)
    bearish_bars = (bearish_count / n_indicators >= min_agreement)
    agreement_bars = bullish_bars | bearish_bars

    # Find windows (consecutive runs)
    windows = []
    current_signal = None
    current_length = 0

    for i in range(n_bars):
        if bullish_bars.iloc[i]:
            if current_signal == 'bullish':
                current_length += 1
            else:
                if current_length > 0:
                    windows.append((current_signal, current_length))
                current_signal = 'bullish'
                current_length = 1
        elif bearish_bars.iloc[i]:
            if current_signal == 'bearish':
                current_length += 1
            else:
                if current_length > 0:
                    windows.append((current_signal, current_length))
                current_signal = 'bearish'
                current_length = 1
        else:
            if current_length > 0:
                windows.append((current_signal, current_length))
            current_signal = None
            current_length = 0

    # Don't forget last window
    if current_length > 0:
        windows.append((current_signal, current_length))

    if not windows:
        return {
            'n_windows': 0, 'avg_length': 0, 'max_length': 0,
            'bullish_pct': 0, 'bearish_pct': 0, 'neutral_pct': 100
        }

    lengths = [w[1] for w in windows]
    bullish_windows = [w[1] for w in windows if w[0] == 'bullish']
    bearish_windows = [w[1] for w in windows if w[0] == 'bearish']

    return {
        'n_windows': len(windows),
        'avg_length': round(np.mean(lengths), 1),
        'max_length': max(lengths),
        'min_length': min(lengths),
        'bullish_pct': round(sum(bullish_windows) / n_bars * 100, 1),
        'bearish_pct': round(sum(bearish_windows) / n_bars * 100, 1),
        'neutral_pct': round((n_bars - sum(lengths)) / n_bars * 100, 1)
    }


def compute_all_metrics(
    signal_matrix: pd.DataFrame,
    isp_positions: Optional[pd.Series] = None
) -> Dict:
    """
    Compute all inter-indicator coherence metrics.

    Parameters
    ----------
    signal_matrix : pd.DataFrame
        Binary signals (+1/-1) for each indicator.
    isp_positions : pd.Series, optional
        ISP position series for ISP coherence measurement.

    Returns
    -------
    dict
        All coherence metrics combined.
    """
    pairwise = compute_pairwise_coherence(signal_matrix)

    # Aggregate pairwise metrics (excluding diagonal)
    n = len(pairwise)
    off_diagonal = []
    for i in range(n):
        for j in range(i + 1, n):
            off_diagonal.append(pairwise.iloc[i, j])

    avg_pairwise = round(np.mean(off_diagonal) * 100, 2) if off_diagonal else 0.0
    min_pairwise = round(np.min(off_diagonal) * 100, 2) if off_diagonal else 0.0
    max_pairwise = round(np.max(off_diagonal) * 100, 2) if off_diagonal else 0.0

    flip_rates = compute_flip_rates(signal_matrix)
    avg_flip_rate = round(np.mean(list(flip_rates.values())), 4)

    agreement = compute_agreement_windows(signal_matrix, min_agreement=0.6)

    metrics = {
        'pairwise_coherence': {
            'avg_pct': avg_pairwise,
            'min_pct': min_pairwise,
            'max_pct': max_pairwise,
            'matrix': pairwise.round(3).to_dict()
        },
        'flip_rates': flip_rates,
        'avg_flip_rate': avg_flip_rate,
        'agreement_windows': agreement
    }

    if isp_positions is not None:
        isp_coherence = compute_individual_isp_coherence(signal_matrix, isp_positions)
        metrics['isp_coherence'] = isp_coherence
        metrics['avg_isp_coherence'] = round(np.mean(list(isp_coherence.values())), 2)
        metrics['min_isp_coherence'] = round(np.min(list(isp_coherence.values())), 2)

    return metrics


def format_coherence_report(metrics: Dict) -> str:
    """
    Format coherence metrics as a human-readable report.

    Parameters
    ----------
    metrics : dict
        Output from compute_all_metrics().

    Returns
    -------
    str
        Formatted report string.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("INTER-INDICATOR COHERENCE REPORT")
    lines.append("=" * 60)

    # Pairwise coherence
    pw = metrics.get('pairwise_coherence', {})
    lines.append(f"\nPairwise Coherence:")
    lines.append(f"  Average: {pw.get('avg_pct', 0):.1f}%")
    lines.append(f"  Range:   {pw.get('min_pct', 0):.1f}% — {pw.get('max_pct', 0):.1f}%")

    # ISP coherence
    if 'isp_coherence' in metrics:
        lines.append(f"\nISP Coherence (per indicator):")
        for name, coh in sorted(metrics['isp_coherence'].items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {name:40s} {coh:.1f}%")
        lines.append(f"  {'AVERAGE':40s} {metrics['avg_isp_coherence']:.1f}%")

    # Flip rates
    fr = metrics.get('flip_rates', {})
    lines.append(f"\nFlip Rates (signal changes per bar):")
    for name, rate in sorted(fr.items(), key=lambda x: x[1]):
        lines.append(f"  {name:40s} {rate:.4f}")
    lines.append(f"  {'AVERAGE':40s} {metrics.get('avg_flip_rate', 0):.4f}")

    # Agreement windows
    aw = metrics.get('agreement_windows', {})
    lines.append(f"\nAgreement Windows (≥60% consensus):")
    lines.append(f"  Total windows:   {aw.get('n_windows', 0)}")
    lines.append(f"  Avg length:      {aw.get('avg_length', 0):.1f} bars")
    lines.append(f"  Max length:      {aw.get('max_length', 0)} bars")
    lines.append(f"  Bullish:         {aw.get('bullish_pct', 0):.1f}%")
    lines.append(f"  Bearish:         {aw.get('bearish_pct', 0):.1f}%")
    lines.append(f"  Neutral:         {aw.get('neutral_pct', 0):.1f}%")

    lines.append("=" * 60)
    return "\n".join(lines)


# --- Unit test ---

def run_unit_tests():
    """Run unit tests for inter-indicator coherence."""
    print("=" * 60)
    print("Running Inter-Indicator Coherence Unit Tests")
    print("=" * 60)

    all_passed = True

    # Test 1: Perfectly correlated indicators
    print("\nTest 1: Perfectly correlated indicators")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        signals = pd.DataFrame({
            'ind1': [1.0] * 100,
            'ind2': [1.0] * 100,
            'ind3': [-1.0] * 100,
        }, index=dates)

        metrics = compute_all_metrics(signals)

        # With 2 bullish + 1 bearish: pairs (ind1,ind2)=100%, (ind1,ind3)=0%, (ind2,ind3)=0%
        # Average = 33.33%
        assert metrics['pairwise_coherence']['avg_pct'] > 30.0, \
            f"Expected ~33% pairwise coherence, got {metrics['pairwise_coherence']['avg_pct']}"
        print(f"  ✓ Pairwise coherence: {metrics['pairwise_coherence']['avg_pct']:.1f}%")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 2: Flip rates
    print("\nTest 2: Flip rates")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        signals = pd.DataFrame({
            'stable': [1.0] * 50 + [-1.0] * 50,       # 1 flip
            'noisy': [1.0 if i % 2 == 0 else -1.0 for i in range(100)],  # 99 flips
        }, index=dates)

        metrics = compute_all_metrics(signals)

        assert metrics['flip_rates']['stable'] < metrics['flip_rates']['noisy'], \
            "Stable should have lower flip rate than noisy"
        print(f"  ✓ Flip rates: stable={metrics['flip_rates']['stable']}, noisy={metrics['flip_rates']['noisy']}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 3: ISP coherence
    print("\nTest 3: ISP coherence")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        signals = pd.DataFrame({
            'ind1': [1.0] * 100,
            'ind2': [-1.0] * 100,
        }, index=dates)
        isp = pd.Series([1.0] * 100, index=dates)

        metrics = compute_all_metrics(signals, isp)

        assert metrics['isp_coherence']['ind1'] == 100.0, "ind1 should match ISP perfectly"
        assert metrics['isp_coherence']['ind2'] == 0.0, "ind2 should not match ISP"
        print(f"  ✓ ISP coherence: ind1={metrics['isp_coherence']['ind1']}, ind2={metrics['isp_coherence']['ind2']}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 4: Agreement windows
    print("\nTest 4: Agreement windows")
    try:
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        signals = pd.DataFrame({
            'ind1': [1.0] * 10 + [-1.0] * 10,
            'ind2': [1.0] * 10 + [-1.0] * 10,
            'ind3': [1.0] * 10 + [-1.0] * 10,
        }, index=dates)

        metrics = compute_all_metrics(signals)

        assert metrics['agreement_windows']['n_windows'] == 2, \
            f"Expected 2 windows, got {metrics['agreement_windows']['n_windows']}"
        assert metrics['agreement_windows']['avg_length'] == 10.0, \
            f"Expected avg length 10, got {metrics['agreement_windows']['avg_length']}"
        print(f"  ✓ Agreement windows: {metrics['agreement_windows']['n_windows']} windows, avg {metrics['agreement_windows']['avg_length']} bars")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    # Test 5: Format report
    print("\nTest 5: Format report")
    try:
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        signals = pd.DataFrame({
            'ind1': [1.0] * 100,
            'ind2': [-1.0] * 100,
        }, index=dates)
        isp = pd.Series([1.0] * 100, index=dates)

        metrics = compute_all_metrics(signals, isp)
        report = format_coherence_report(metrics)

        assert "Pairwise Coherence" in report
        assert "ISP Coherence" in report
        print("  ✓ Report formatted correctly")
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
