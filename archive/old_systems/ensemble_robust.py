#!/usr/bin/env python3
"""
MTTD Robust Ensemble Engine
=============================

Combines multiple indicators with robustness features:
1. Minimum agreement threshold (not just majority)
2. Outlier rejection (ignore extreme signals)
3. Voting with configurable minimum votes
4. Regime-aware indicator selection

Goal: When 1 indicator gives false signal, ensemble still correct.
"""

import pandas as pd
import numpy as np


def compute_robust_ensemble(
    signal_matrix: pd.DataFrame,
    min_hold: int = 5,
    min_agreement: float = 0.6,  # Need 60% agreement (e.g., 3/5 or 4/6)
    reject_outliers: bool = True,
    outlier_threshold: float = 2.0,  # Std devs from mean
    position_col: str = 'position'
) -> pd.DataFrame:
    """
    Compute robust ensemble signal with majority vote + outlier rejection.
    
    Args:
        signal_matrix: DataFrame with indicator signals (-1, 0, 1)
        min_hold: Minimum holding period in days
        min_agreement: Minimum fraction of indicators agreeing (0.6 = 60%)
        reject_outliers: Whether to reject outlier signals
        outlier_threshold: Number of std devs for outlier detection
        
    Returns:
        DataFrame with 'position' and 'agreement' columns
    """
    n_indicators = len(signal_matrix.columns)
    
    if n_indicators == 0:
        return pd.DataFrame({
            'position': 0.0,
            'agreement': 0.0
        }, index=signal_matrix.index)
    
    # Step 1: Outlier rejection (optional)
    if reject_outliers and n_indicators >= 3:
        # For each bar, compute mean and std of signals
        signal_mean = signal_matrix.mean(axis=1)
        signal_std = signal_matrix.std(axis=1)
        
        # Flag outlier signals (more than outlier_threshold std devs from mean)
        is_outlier = pd.DataFrame(False, index=signal_matrix.index, columns=signal_matrix.columns)
        
        for col in signal_matrix.columns:
            deviation = (signal_matrix[col] - signal_mean).abs()
            is_outlier[col] = deviation > outlier_threshold * signal_std
        
        # Replace outlier signals with NaN (will be ignored in voting)
        signal_clean = signal_matrix.copy()
        for col in signal_matrix.columns:
            signal_clean.loc[is_outlier[col], col] = np.nan
        
        n_outliers = is_outlier.sum().sum()
        if n_outliers > 0:
            print(f"    Outlier rejection: {n_outliers} outlier signals removed")
    else:
        signal_clean = signal_matrix
    
    # Step 2: Compute agreement score
    # Count how many indicators agree on direction
    bullish_count = (signal_clean > 0).sum(axis=1)
    bearish_count = (signal_clean < 0).sum(axis=1)
    neutral_count = (signal_clean == 0).sum(axis=1)
    valid_count = signal_clean.notna().sum(axis=1)  # Non-NaN indicators
    
    # Avoid division by zero
    valid_count = valid_count.replace(0, 1)
    
    # Agreement fraction for majority direction
    majority_count = np.maximum(bullish_count, bearish_count)
    agreement = majority_count / valid_count
    
    # Step 3: Determine ensemble direction
    # Need minimum agreement threshold
    ensemble_signal = pd.Series(0.0, index=signal_matrix.index)
    
    # Bullish: majority bullish AND meets agreement threshold
    bullish_mask = (bullish_count > bearish_count) & (agreement >= min_agreement)
    ensemble_signal[bullish_mask] = 1.0
    
    # Bearish: majority bearish AND meets agreement threshold
    bearish_mask = (bearish_count > bullish_count) & (agreement >= min_agreement)
    ensemble_signal[bearish_mask] = -1.0
    
    # Step 4: Convert to position (1 = in market, 0 = out)
    position = (ensemble_signal > 0).astype(float)
    
    # Step 5: Apply min_hold filter
    if min_hold > 1:
        result = apply_min_hold(position, min_hold)
    else:
        result = position
    
    # Create output DataFrame
    output = pd.DataFrame({
        'position': result,
        'agreement': agreement,
        'bullish_pct': bullish_count / valid_count,
        'bearish_pct': bearish_count / valid_count
    }, index=signal_matrix.index)
    
    return output


def apply_min_hold(position: pd.Series, min_hold: int) -> pd.Series:
    """
    Apply minimum holding period filter.
    
    Once position changes, it must stay for at least min_hold bars.
    """
    result = position.copy()
    
    if len(result) == 0:
        return result
    
    last_change_idx = 0
    last_position = result.iloc[0]
    
    for i in range(1, len(result)):
        current_position = result.iloc[i]
        
        if current_position != last_position:
            # Position changed
            if i - last_change_idx < min_hold:
                # Too soon to change - revert to previous position
                result.iloc[i] = last_position
            else:
                # OK to change
                last_change_idx = i
                last_position = current_position
        else:
            # Same position, update last position
            last_position = current_position
    
    return result


def compute_weighted_ensemble(
    signal_matrix: pd.DataFrame,
    weights: dict,
    min_hold: int = 5,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Compute weighted ensemble with indicator weights.
    
    Args:
        signal_matrix: DataFrame with indicator signals (-1, 0, 1)
        weights: Dict of {indicator_name: weight} where weight > 0
        min_hold: Minimum holding period in days
        threshold: Weighted vote threshold for position (0.5 = majority)
        
    Returns:
        DataFrame with 'position' column
    """
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight == 0:
        return pd.DataFrame({'position': 0.0}, index=signal_matrix.index)
    
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    # Compute weighted vote
    weighted_vote = pd.Series(0.0, index=signal_matrix.index)
    
    for col in signal_matrix.columns:
        if col in normalized_weights:
            weighted_vote += signal_matrix[col] * normalized_weights[col]
    
    # Determine position based on threshold
    position = (weighted_vote > threshold).astype(float)
    
    # Apply min_hold
    if min_hold > 1:
        position = apply_min_hold(position, min_hold)
    
    return pd.DataFrame({'position': position}, index=signal_matrix.index)


# ================================================================
# Pre-defined robust ensembles based on test results
# ================================================================

ENSEMBLE_CONFIGS = {
    'conservative_5': {
        'description': '5 best indicators, need 4/5 agreement (80%)',
        'indicators': [
            'median_standard_deviation_viresearch',
            'lsma_for_loop_viresearch', 
            'irs_elder_force_volume_index',
            'dema_sma_standard_deviation_viresearch',
            'mode_for_loop_viresearch'
        ],
        'min_agreement': 0.8,  # Need 4/5
        'min_hold': 5,
        'reject_outliers': True
    },
    'moderate_6': {
        'description': '6 indicators, need 4/6 agreement (67%)',
        'indicators': [
            'median_standard_deviation_viresearch',
            'lsma_for_loop_viresearch',
            'irs_elder_force_volume_index',
            'gaussian_smooth_trend_quantedgeb',
            'median_for_loop_viresearch',
            'hull_for_loop_viresearch'
        ],
        'min_agreement': 0.67,  # Need 4/6
        'min_hold': 5,
        'reject_outliers': True
    },
    'diverse_6': {
        'description': '6 diverse indicators, need 4/6 agreement (67%)',
        'indicators': [
            'median_standard_deviation_viresearch',  # Best overall
            'irs_elder_force_volume_index',          # Volume-based
            'adaptive_regime_cloud',                 # Regime detection
            'kalman_filtered_rsi_oscillator',        # Momentum
            'z_score_adaptive_oscillator_suite',     # Mean-reversion
            'alma_lag_viresearch'                    # Trend
        ],
        'min_agreement': 0.67,  # Need 4/6
        'min_hold': 5,
        'reject_outliers': True
    },
    'ultra_conservative_7': {
        'description': '7 indicators, need 5/7 agreement (71%)',
        'indicators': [
            'median_standard_deviation_viresearch',
            'lsma_for_loop_viresearch',
            'irs_elder_force_volume_index',
            'dema_sma_standard_deviation_viresearch',
            'gaussian_smooth_trend_quantedgeb',
            'median_for_loop_viresearch',
            'hull_for_loop_viresearch'
        ],
        'min_agreement': 0.71,  # Need 5/7
        'min_hold': 7,
        'reject_outliers': True
    }
}


if __name__ == '__main__':
    print("=" * 70)
    print("MTTD ROBUST ENSEMBLE ENGINE")
    print("=" * 70)
    
    print("\nAvailable ensemble configurations:")
    for name, config in ENSEMBLE_CONFIGS.items():
        print(f"\n  {name}:")
        print(f"    Description: {config['description']}")
        print(f"    Indicators: {len(config['indicators'])}")
        print(f"    Min agreement: {config['min_agreement']*100:.0f}%")
        print(f"    Min hold: {config['min_hold']} days")
        print(f"    Reject outliers: {config['reject_outliers']}")
