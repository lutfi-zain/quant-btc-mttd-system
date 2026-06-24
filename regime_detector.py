#!/usr/bin/env python3
"""
Regime Detector — On-Chain & Sentiment Regime Classification
============================================================

Loads 17 metrics from the BTC Valuation System database, normalizes them
to a -2 to +2 scale (already done in the database), and computes a
composite regime signal based on majority vote.

Regime Logic:
- Bull:  Majority of metrics have normalized_value > 0.5
- Bear:  Majority of metrics have normalized_value < -0.5
- Neutral: Otherwise

Output: mttd/regime_data.csv with columns: date, regime, composite_score
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

# ================================================================
# Configuration
# ================================================================
VALUATION_DB_PATH = '/home/ubuntu/projects/quant-btc-valuation-system/database/metrics.db'
MTTD_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mttd')
OUTPUT_FILE = os.path.join(MTTD_OUTPUT_DIR, 'regime_data.csv')

# MTTD system starts from 2018-01-01
MTTD_START_DATE = '2018-01-01'

# Thresholds for regime classification
BULL_THRESHOLD = 0.5
BEAR_THRESHOLD = -0.5

# Minimum number of metrics required for a valid regime classification
MIN_METRICS_REQUIRED = 10

print("=" * 70)
print("REGIME DETECTOR — On-Chain & Sentiment Regime Classification")
print("=" * 70)

# ================================================================
# Load Metrics from Database
# ================================================================
print("\n[1/4] Loading metrics from valuation database...")

conn = sqlite3.connect(VALUATION_DB_PATH)

# Load all normalized metrics
df = pd.read_sql('''
    SELECT date, metric_name, normalized_value 
    FROM timeseries_metrics 
    WHERE normalized_value IS NOT NULL
''', conn)

conn.close()

print(f"  Loaded {len(df)} rows across {df['metric_name'].nunique()} metrics")
print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

# ================================================================
# Pivot and Align Data
# ================================================================
print("\n[2/4] Pivoting and aligning data...")

# Pivot to get metrics as columns
pivot = df.pivot_table(
    index='date', 
    columns='metric_name', 
    values='normalized_value', 
    aggfunc='first'
)

# Parse dates and filter to MTTD start date
pivot.index = pd.to_datetime(pivot.index).tz_localize(None)
pivot = pivot[pivot.index >= MTTD_START_DATE]

print(f"  Pivoted data: {len(pivot)} dates, {len(pivot.columns)} metrics")
print(f"  Date range: {pivot.index.min()} to {pivot.index.max()}")

# Count non-null metrics per date
metrics_per_date = pivot.count(axis=1)
print(f"  Metrics per date: min={metrics_per_date.min()}, max={metrics_per_date.max()}, mean={metrics_per_date.mean():.1f}")

# ================================================================
# Compute Regime Signal
# ================================================================
print("\n[3/4] Computing regime signal using majority vote...")

def compute_regime(row, bull_thresh, bear_thresh, min_metrics):
    """
    Compute regime for a single date based on majority vote.
    
    Rules:
    - Bull:  majority of metrics > bull_thresh
    - Bear:  majority of metrics < bear_thresh
    - Neutral: otherwise
    
    Returns: (regime_label, composite_score)
    """
    valid_metrics = row.dropna()
    n_valid = len(valid_metrics)
    
    if n_valid < min_metrics:
        return ('Neutral', 0.0)
    
    # Count bullish and bearish signals
    n_bull = (valid_metrics > bull_thresh).sum()
    n_bear = (valid_metrics < bear_thresh).sum()
    n_neutral = n_valid - n_bull - n_bear
    
    # Majority threshold
    majority = n_valid / 2.0
    
    # Composite score: weighted average normalized to -1 to +1
    # Positive = bullish, Negative = bearish
    composite_score = valid_metrics.mean()
    
    # Determine regime
    if n_bull > majority:
        regime = 'Bull'
    elif n_bear > majority:
        regime = 'Bear'
    else:
        regime = 'Neutral'
    
    return (regime, round(composite_score, 4))

# Apply regime computation
results = pivot.apply(
    lambda row: compute_regime(row, BULL_THRESHOLD, BEAR_THRESHOLD, MIN_METRICS_REQUIRED), 
    axis=1, 
    result_type='expand'
)
results.columns = ['regime', 'composite_score']

# Create output dataframe
regime_df = pd.DataFrame({
    'date': results.index,
    'regime': results['regime'].values,
    'composite_score': results['composite_score'].values
})

print(f"  Computed regime for {len(regime_df)} dates")

# ================================================================
# Analyze Regime Distribution
# ================================================================
print("\n  Regime Distribution:")
regime_counts = regime_df['regime'].value_counts()
for regime in ['Bull', 'Neutral', 'Bear']:
    count = regime_counts.get(regime, 0)
    pct = count / len(regime_df) * 100
    print(f"    {regime:10s}: {count:5d} ({pct:5.1f}%)")

print(f"\n  Composite Score Statistics:")
print(f"    Mean:   {regime_df['composite_score'].mean():.4f}")
print(f"    Std:    {regime_df['composite_score'].std():.4f}")
print(f"    Min:    {regime_df['composite_score'].min():.4f}")
print(f"    Max:    {regime_df['composite_score'].max():.4f}")

# ================================================================
# Spot-Check Regime Classifications
# ================================================================
print("\n  Spot-Check Regime Classifications:")

# Sample a few dates with clear regimes
bull_dates = regime_df[regime_df['regime'] == 'Bull']['date'].head(2)
bear_dates = regime_df[regime_df['regime'] == 'Bear']['date'].head(2)

for date in list(bull_dates) + list(bear_dates):
    row = pivot.loc[date] if date in pivot.index else None
    if row is not None:
        regime_row = regime_df[regime_df['date'] == date].iloc[0]
        print(f"\n  Date: {date.strftime('%Y-%m-%d')} | Regime: {regime_row['regime']} | Score: {regime_row['composite_score']:.4f}")
        valid = row.dropna()
        above_thresh = (valid > BULL_THRESHOLD).sum()
        below_thresh = (valid < BEAR_THRESHOLD).sum()
        print(f"    Valid metrics: {len(valid)}, Bullish: {above_thresh}, Bearish: {below_thresh}")
        print(f"    Top bullish: {valid.nlargest(3).index.tolist()}")
        print(f"    Top bearish: {valid.nsmallest(3).index.tolist()}")

# ================================================================
# Save Results
# ================================================================
print("\n[4/4] Saving regime data...")

os.makedirs(MTTD_OUTPUT_DIR, exist_ok=True)

# Save to CSV
regime_df.to_csv(OUTPUT_FILE, index=False)
print(f"  Saved: {OUTPUT_FILE}")
print(f"  Rows: {len(regime_df)}")

# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Database: {VALUATION_DB_PATH}")
print(f"  Output:   {OUTPUT_FILE}")
print(f"  Date Range: {regime_df['date'].min()} to {regime_df['date'].max()}")
print(f"  Total Dates: {len(regime_df)}")
print(f"  Regime Distribution:")
for regime in ['Bull', 'Neutral', 'Bear']:
    count = regime_counts.get(regime, 0)
    pct = count / len(regime_df) * 100
    print(f"    {regime:10s}: {count:5d} ({pct:5.1f}%)")
print("=" * 70)
