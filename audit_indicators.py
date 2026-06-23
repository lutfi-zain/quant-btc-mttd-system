"""
Indicator Audit Framework
=========================

Evaluates individual indicator performance against the ISP benchmark.
Produces a metrics DataFrame with per-indicator evaluation including:
- Coherence %: How well indicator aligns with ISP target (time-coherence)
- Trade count: Number of signal transitions (entry/exit cycles)
- Avg hold duration: Average days in position per trade
- Return: Cumulative return from indicator signals
- Stability: Consistency of position allocation (0 to 1)
- Max Drawdown: Worst peak-to-trough decline

Usage:
    from audit_indicators import run_full_audit, build_isp_position_series
    results = run_full_audit(indicators_list, df, csv_path)
"""

import os
import sys
import re
import importlib.util
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


# ---------------------------------------------------------------------------
# ISP Position Series Builder
# ---------------------------------------------------------------------------

def build_isp_position_series(df, csv_path):
    """
    Convert ISP CSV to daily position series (0 or 1).

    ISP CSV format:
        Date, Action, Price, EquityPct, Cost, BTCHeld, TotalEquity, Regime

    Regime mapping (binary for audit purposes):
        Strong Bull  → 1.0  (100% BTC)
        Weak Bull    → 0.0  (0% BTC — below binary threshold)
        Neutral      → 0.0  (0% BTC)

    Args:
        df: DataFrame with date string index (same as price data)
        csv_path: Path to ISP signals CSV

    Returns:
        pd.Series with 0.0 or 1.0 values aligned to df.index
    """
    df_csv = pd.read_csv(csv_path)
    position_series = pd.Series(0.0, index=df.index)

    for _, row in df_csv.iterrows():
        date = str(row['Date'])
        regime = str(row['Regime']).strip()
        if regime == 'Strong Bull':
            position_series.loc[date:] = 1.0
        else:
            position_series.loc[date:] = 0.0

    return position_series


# ---------------------------------------------------------------------------
# Direction Series Extraction (reused from execute_system.py)
# ---------------------------------------------------------------------------

def detect_direction_series(res_df):
    """
    Extract direction series from indicator output DataFrame.

    Priority order for signal column detection:
    1. Named signal columns: dir, sig, direction, vii, qb, etc.
    2. long_signal + short_signal pair → stateful direction
    3. in_long_position + in_short_position pair → binary
    4. Fallback heuristic on remaining columns

    Returns:
        pd.Series of direction values (-1.0, 0.0, 1.0) or None
    """
    for col in ['dir', 'sig', 'direction', 'vii', 'qb',
                'st_direction', 'trend_direction', 'trend']:
        if col in res_df.columns:
            return res_df[col]

    if 'long_signal' in res_df.columns and 'short_signal' in res_df.columns:
        direction = pd.Series(0.0, index=res_df.index)
        curr = 0.0
        for i in range(len(res_df)):
            l_val = bool(res_df['long_signal'].iloc[i]) if not pd.isna(
                res_df['long_signal'].iloc[i]) else False
            s_val = bool(res_df['short_signal'].iloc[i]) if not pd.isna(
                res_df['short_signal'].iloc[i]) else False
            if l_val and not s_val:
                curr = 1.0
            elif s_val and not l_val:
                curr = -1.0
            direction.iloc[i] = curr
        return direction

    if 'in_long_position' in res_df.columns and 'in_short_position' in res_df.columns:
        direction = pd.Series(0.0, index=res_df.index)
        direction[res_df['in_long_position'] == 1] = 1.0
        direction[res_df['in_short_position'] == 1] = -1.0
        return direction

    for col in res_df.columns:
        col_lower = col.lower()
        if 'direction' in col_lower or 'signal' in col_lower or 'trend' in col_lower:
            unique_vals = res_df[col].dropna().unique()
            if len(unique_vals) <= 10:
                return res_df[col]

    return None


# ---------------------------------------------------------------------------
# Name Normalization
# ---------------------------------------------------------------------------

def normalize_name(name):
    """Normalize indicator name to module filename format."""
    n = name.replace("(", "").replace(")", "")
    n = n.replace("%", "")
    n = re.sub(r"[|:\-`]", " ", n)
    n = n.lower().strip()
    n = re.sub(r"\s+", "_", n)
    n = re.sub(r"_+", "_", n)
    return n


# ---------------------------------------------------------------------------
# Position Conversion
# ---------------------------------------------------------------------------

def indicator_to_position(direction_series):
    """
    Convert direction series (-1/0/1) to binary position series (0/1).

    Bullish (direction > 0) → 1.0
    Bearish/Neutral (direction ≤ 0) → 0.0
    """
    return (direction_series > 0).astype(float)


# ---------------------------------------------------------------------------
# Core Metric Calculations
# ---------------------------------------------------------------------------

def compute_coherence(indicator_position, isp_position):
    """
    Compute time-coherence between indicator and ISP position series.

    Coherence = % of daily periods where both agree (both 0 or both 1).

    Returns:
        float: coherence percentage (0.0 to 100.0)
    """
    if len(indicator_position) != len(isp_position):
        # Align to common index
        common_idx = indicator_position.index.intersection(isp_position.index)
        indicator_position = indicator_position.reindex(common_idx)
        isp_position = isp_position.reindex(common_idx)

    agreement = (indicator_position == isp_position).sum()
    total = len(indicator_position)

    return (agreement / total) * 100.0 if total > 0 else 0.0


def compute_trade_metrics(position_series):
    """
    Compute trade count, avg hold duration from position series.

    A "trade" is one complete entry→exit cycle. Partial periods at
    start/end are counted as open trades.

    Returns:
        dict with keys: trades, avg_hold_days
    """
    in_position = (position_series == 1.0)

    if not in_position.any():
        return {'trades': 0, 'avg_hold_days': 0.0}

    # Create group labels for consecutive True blocks
    # Transition points mark new groups
    transitions = in_position.ne(in_position.shift(1))
    groups = transitions.cumsum()

    # Count consecutive True values per group
    hold_durations = in_position.groupby(groups).sum()

    # Filter to only groups that are in-position (True)
    in_groups = groups[in_position]
    if in_groups.empty:
        return {'trades': 0, 'avg_hold_days': 0.0}

    # Count durations per in-position group
    hold_counts = in_groups.value_counts().sort_index()
    avg_hold = float(hold_counts.mean()) if len(hold_counts) > 0 else 0.0
    trade_count = len(hold_counts)

    return {
        'trades': int(trade_count),
        'avg_hold_days': round(avg_hold, 1),
    }


def compute_stability(position_series):
    """
    Compute signal stability as position allocation ratio.

    0.0 = never in position (all cash)
    0.5 = half the time in position
    1.0 = always in position (100% BTC)

    Returns:
        float: stability value (0.0 to 1.0)
    """
    return float(position_series.mean())


def compute_strategy_returns(indicator_position, price_series):
    """
    Compute cumulative return from indicator position and price series.

    Returns:
        dict with total_return_pct, annualized_return_pct, sharpe_ratio
    """
    daily_returns = price_series.pct_change().fillna(0.0)
    strategy_returns = indicator_position * daily_returns
    cumulative = (1 + strategy_returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1.0) * 100.0

    # Annualized return
    n_days = len(price_series)
    annualized = ((cumulative.iloc[-1]) ** (365.0 / n_days) - 1.0) * 100.0 if n_days > 0 else 0.0

    # Sharpe ratio
    mean_ret = strategy_returns.mean()
    std_ret = strategy_returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(365) if std_ret > 0 else 0.0

    return {
        'total_return_pct': round(total_return, 2),
        'annualized_return_pct': round(annualized, 2),
        'sharpe_ratio': round(sharpe, 4),
    }


def compute_max_drawdown(indicator_position, price_series):
    """
    Compute maximum drawdown from position and price series.

    Returns:
        dict with max_drawdown_pct (negative value)
    """
    daily_returns = price_series.pct_change().fillna(0.0)
    strategy_returns = indicator_position * daily_returns
    cumulative = (1 + strategy_returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd = float(drawdown.min()) * 100.0

    return {'max_drawdown_pct': round(max_dd, 2)}


def compute_pearson_correlation(indicator_position, isp_position):
    """
    Pearson correlation between position series.
    Measures linear co-movement of positions.
    """
    if len(indicator_position) != len(isp_position):
        common_idx = indicator_position.index.intersection(isp_position.index)
        indicator_position = indicator_position.reindex(common_idx)
        isp_position = isp_position.reindex(common_idx)

    # Check for zero variance
    if indicator_position.std() == 0 or isp_position.std() == 0:
        return 0.0

    return round(float(indicator_position.corr(isp_position)), 4)


def compute_spearman_correlation(indicator_position, isp_position):
    """
    Spearman rank correlation between position series.
    Measures monotonic relationship of positions.
    """
    if len(indicator_position) != len(isp_position):
        common_idx = indicator_position.index.intersection(isp_position.index)
        indicator_position = indicator_position.reindex(common_idx)
        isp_position = isp_position.reindex(common_idx)

    # Check for zero variance
    if indicator_position.std() == 0 or isp_position.std() == 0:
        return 0.0

    return round(float(indicator_position.corr(isp_position, method='spearman')), 4)


# ---------------------------------------------------------------------------
# Single Indicator Audit
# ---------------------------------------------------------------------------

def audit_single_indicator(indicator_name, indicator_category, df, isp_position,
                           price_series):
    """
    Audit a single indicator against ISP benchmark.

    Args:
        indicator_name: Display name of indicator
        indicator_category: Category folder ('perpetual', 'oscillator')
        df: Price DataFrame (subset to evaluation range)
        isp_position: ISP binary position series (subset to evaluation range)
        price_series: Close price series (subset to evaluation range)

    Returns:
        dict with all metrics, or None if indicator fails to load
    """
    normalized = normalize_name(indicator_name)
    py_file = os.path.join(project_root, indicator_category, f"{normalized}.py")

    if not os.path.exists(py_file):
        print(f"  SKIP: File not found: {py_file}")
        return None

    try:
        spec = importlib.util.spec_from_file_location(normalized, py_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, normalized)
        res_df = func(df)
    except Exception as e:
        print(f"  ERROR loading {indicator_name}: {e}")
        return None

    direction = detect_direction_series(res_df)
    if direction is None:
        print(f"  SKIP: No direction column found in {indicator_name}")
        return None

    # Align to evaluation index
    direction = direction.reindex(df.index).fillna(0.0)
    indicator_position = indicator_to_position(direction)

    # Compute all metrics
    trade_metrics = compute_trade_metrics(indicator_position)
    coherence = compute_coherence(indicator_position, isp_position)
    stability = compute_stability(indicator_position)
    returns = compute_strategy_returns(indicator_position, price_series)
    max_dd = compute_max_drawdown(indicator_position, price_series)
    pearson = compute_pearson_correlation(indicator_position, isp_position)
    spearman = compute_spearman_correlation(indicator_position, isp_position)

    # Buy-and-hold reference
    bnh_returns = compute_strategy_returns(
        pd.Series(1.0, index=price_series.index), price_series
    )

    return {
        'indicator': indicator_name,
        'normalized': normalized,
        'trades': trade_metrics['trades'],
        'avg_hold_days': trade_metrics['avg_hold_days'],
        'coherence_pct': coherence,
        'pearson_r': pearson,
        'spearman_r': spearman,
        'stability': stability,
        'total_return_pct': returns['total_return_pct'],
        'annualized_return_pct': returns['annualized_return_pct'],
        'sharpe_ratio': returns['sharpe_ratio'],
        'max_drawdown_pct': max_dd['max_drawdown_pct'],
        'bnh_total_return_pct': bnh_returns['total_return_pct'],
        'bnh_annualized_return_pct': bnh_returns['annualized_return_pct'],
    }


# ---------------------------------------------------------------------------
# Full Audit Runner
# ---------------------------------------------------------------------------

def run_full_audit(indicators_list, df, csv_path):
    """
    Run full audit across all indicators against ISP benchmark.

    Args:
        indicators_list: List of dicts with 'name' and 'category' keys
                         e.g., [{"name": "DSMA | viResearch", "category": "perpetual"}]
        df: Price data DataFrame with date string index
        csv_path: Path to ISP signals CSV

    Returns:
        pd.DataFrame with all metrics per indicator, sorted by coherence
    """
    # Build ISP position series
    isp_position_full = build_isp_position_series(df, csv_path)

    # Limit evaluation to ISP date range
    df_csv = pd.read_csv(csv_path)
    first_date = str(df_csv['Date'].iloc[0])
    last_date = str(df_csv['Date'].iloc[-1])

    df_eval = df.loc[first_date:last_date].copy()
    isp_eval = isp_position_full.loc[first_date:last_date].copy()
    price_eval = df_eval['close']

    print(f"Evaluation range: {first_date} to {last_date} ({len(df_eval)} days)")
    print(f"ISP position sum (in-position days): {int(isp_eval.sum())}")
    print(f"Auditing {len(indicators_list)} indicators...\n")

    results = []

    for idx, ind in enumerate(indicators_list, 1):
        name = ind['name']
        cat = ind['category']
        print(f"[{idx}/{len(indicators_list)}] {name}...")
        try:
            metrics = audit_single_indicator(
                indicator_name=name,
                indicator_category=cat,
                df=df_eval,
                isp_position=isp_eval,
                price_series=price_eval,
            )
            if metrics:
                results.append(metrics)
                print(f"  → coherence={metrics['coherence_pct']:.1f}%, "
                      f"trades={metrics['trades']}, "
                      f"return={metrics['total_return_pct']:.1f}%, "
                      f"sharpe={metrics['sharpe_ratio']:.2f}")
        except Exception as e:
            print(f"  ERROR: {e}")

    if not results:
        print("\nNo indicators produced results.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('coherence_pct', ascending=False)
    results_df = results_df.reset_index(drop=True)

    return results_df


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Import SELECTED_INDICATORS and load_data from execute_system
    from execute_system import load_data, SELECTED_INDICATORS

    print("=" * 70)
    print("INDICATOR AUDIT FRAMEWORK — Standalone Mode")
    print("=" * 70)

    print("\nLoading price data...")
    df = load_data()
    print(f"Loaded {len(df)} daily bars.\n")

    csv_path = os.path.join(project_root, "isp-signals-btcusd-2026-06-13.csv")

    print("Running full audit...")
    results_df = run_full_audit(SELECTED_INDICATORS, df, csv_path)

    print("\n" + "=" * 70)
    print("INDICATOR AUDIT RESULTS")
    print("=" * 70)
    print(results_df.to_string(index=False))

    print("\n" + "-" * 70)
    print("Summary Statistics:")
    print("-" * 70)
    print(results_df[['coherence_pct', 'trades', 'avg_hold_days',
                       'total_return_pct', 'sharpe_ratio',
                       'stability', 'max_drawdown_pct']].describe().round(2))

    # Save results
    out_path = os.path.join(project_root, "mttd", "audit_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
