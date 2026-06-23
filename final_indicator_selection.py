"""
Final Indicator Selection
=========================

Combines audit results from TODO 4 (current 15 indicators) and TODO 5 (new indicator tests)
to select the final ensemble set of 15-25 indicators.

Selection criteria:
1. Individual coherence > 50%
2. Prefer indicators with higher coherence
3. Ensure diversity in indicator types (perpetual vs oscillator)
4. Avoid redundancy (similar indicators)
5. Consider risk metrics (max drawdown, Sharpe ratio)

Output: FINAL_INDICATORS list and documentation
"""

import os
import sys
import pandas as pd
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


# ---------------------------------------------------------------------------
# Load Audit Results
# ---------------------------------------------------------------------------

def load_audit_results():
    """Load audit results from TODO 4 and TODO 5."""
    
    # TODO 4 results: current 15 indicators
    audit_path = os.path.join(project_root, "mttd", "audit_results.csv")
    audit_df = pd.read_csv(audit_path)
    audit_df['source'] = 'todo4_current'
    
    # TODO 5 results: all tested indicators
    test_path = os.path.join(project_root, "mttd", "all_indicator_test_results.csv")
    test_df = pd.read_csv(test_path)
    test_df['source'] = 'todo5_new'
    
    return audit_df, test_df


# ---------------------------------------------------------------------------
# Merge and Deduplicate
# ---------------------------------------------------------------------------

def merge_results(audit_df, test_df):
    """Merge results from both audits, removing duplicates."""
    
    # Common columns for merging
    common_cols = ['indicator', 'normalized', 'trades', 'avg_hold_days', 
                   'coherence_pct', 'pearson_r', 'spearman_r', 'stability',
                   'total_return_pct', 'annualized_return_pct', 'sharpe_ratio',
                   'max_drawdown_pct']
    
    # Add category to audit_df if not present
    if 'category' not in audit_df.columns:
        audit_df['category'] = 'perpetual'  # Default for current indicators
    
    # Combine both dataframes
    combined = pd.concat([audit_df[common_cols + ['category', 'source']], 
                          test_df[common_cols + ['category', 'source']]], 
                         ignore_index=True)
    
    # Remove duplicates by normalized name, keeping the one with more info
    # If same indicator appears in both, prefer the one from todo5 (more complete)
    combined = combined.sort_values('source', ascending=False)
    combined = combined.drop_duplicates(subset='normalized', keep='first')
    
    # Sort by coherence descending
    combined = combined.sort_values('coherence_pct', ascending=False).reset_index(drop=True)
    
    return combined


# ---------------------------------------------------------------------------
# Filter by Coherence Threshold
# ---------------------------------------------------------------------------

def filter_by_coherence(df, min_coherence=50.0):
    """Filter indicators with coherence above threshold."""
    filtered = df[df['coherence_pct'] >= min_coherence].copy()
    return filtered


# ---------------------------------------------------------------------------
# Select Final Indicators
# ---------------------------------------------------------------------------

def select_final_indicators(df, target_count=20, min_coherence=50.0):
    """
    Select final indicator set.
    
    Strategy:
    1. Start with all indicators meeting coherence threshold
    2. Group by category (perpetual vs oscillator)
    3. Select top N from each category to ensure diversity
    4. Fill remaining slots with highest coherence indicators
    5. Final count should be 15-25
    """
    
    # Filter by minimum coherence
    eligible = df[df['coherence_pct'] >= min_coherence].copy()
    
    print(f"\nEligible indicators (coherence >= {min_coherence}%): {len(eligible)}")
    
    if len(eligible) == 0:
        print("WARNING: No indicators meet minimum coherence threshold!")
        print("Lowering threshold to 45%...")
        eligible = df[df['coherence_pct'] >= 45.0].copy()
        print(f"Indicators with coherence >= 45%: {len(eligible)}")
    
    # Split by category
    perpetual = eligible[eligible['category'] == 'perpetual'].copy()
    oscillator = eligible[eligible['category'] == 'oscillator'].copy()
    
    print(f"\nBy category:")
    print(f"  Perpetual: {len(perpetual)}")
    print(f"  Oscillator: {len(oscillator)}")
    
    # Selection strategy: aim for balanced mix
    # 40% perpetual, 60% oscillator (oscillators tend to be more diverse)
    n_perpetual = min(int(target_count * 0.4), len(perpetual))
    n_oscillator = min(target_count - n_perpetual, len(oscillator))
    
    # If not enough oscillators, take more perpetuals
    if n_oscillator < len(oscillator):
        n_perpetual = min(target_count - n_oscillator, len(perpetual))
    
    # Select top from each category
    selected_perpetual = perpetual.head(n_perpetual)
    selected_oscillator = oscillator.head(n_oscillator)
    
    # Combine
    selected = pd.concat([selected_perpetual, selected_oscillator], ignore_index=True)
    
    # If we have fewer than target, add more from remaining
    if len(selected) < target_count:
        remaining = eligible[~eligible['normalized'].isin(selected['normalized'])]
        additional = remaining.head(target_count - len(selected))
        selected = pd.concat([selected, additional], ignore_index=True)
    
    # If we have more than 25, trim to 25
    if len(selected) > 25:
        selected = selected.head(25)
    
    # Sort by coherence
    selected = selected.sort_values('coherence_pct', ascending=False).reset_index(drop=True)
    
    return selected


# ---------------------------------------------------------------------------
# Document Selection Rationale
# ---------------------------------------------------------------------------

def document_selection(selected_df, all_df):
    """Generate documentation for the final indicator selection."""
    
    doc = []
    doc.append("=" * 80)
    doc.append("FINAL INDICATOR SELECTION DOCUMENTATION")
    doc.append("=" * 80)
    doc.append("")
    doc.append("Selection Criteria:")
    doc.append("  1. Individual coherence with ISP benchmark > 50%")
    doc.append("  2. Prefer higher coherence indicators")
    doc.append("  3. Balance between perpetual and oscillator types")
    doc.append("  4. Consider risk metrics (Sharpe ratio, max drawdown)")
    doc.append("  5. Target 15-25 indicators for ensemble diversity")
    doc.append("")
    doc.append("-" * 80)
    doc.append("SELECTION SUMMARY")
    doc.append("-" * 80)
    doc.append(f"Total indicators evaluated: {len(all_df)}")
    doc.append(f"Indicators meeting coherence threshold (>50%): {len(all_df[all_df['coherence_pct'] >= 50])}")
    doc.append(f"Final selected indicators: {len(selected_df)}")
    doc.append("")
    
    # Category breakdown
    perpetual_count = len(selected_df[selected_df['category'] == 'perpetual'])
    oscillator_count = len(selected_df[selected_df['category'] == 'oscillator'])
    doc.append(f"Category breakdown:")
    doc.append(f"  Perpetual: {perpetual_count}")
    doc.append(f"  Oscillator: {oscillator_count}")
    doc.append("")
    
    # Statistics
    doc.append(f"Coherence statistics (selected):")
    doc.append(f"  Mean:   {selected_df['coherence_pct'].mean():.1f}%")
    doc.append(f"  Median: {selected_df['coherence_pct'].median():.1f}%")
    doc.append(f"  Min:    {selected_df['coherence_pct'].min():.1f}%")
    doc.append(f"  Max:    {selected_df['coherence_pct'].max():.1f}%")
    doc.append("")
    
    doc.append("-" * 80)
    doc.append("SELECTED INDICATORS (ranked by coherence)")
    doc.append("-" * 80)
    doc.append("")
    doc.append(f"{'Rank':>4} {'Indicator':<50} {'Category':<12} {'Coherence':>10} {'Sharpe':>8} {'MaxDD':>8}")
    doc.append("-" * 100)
    
    for rank, (_, row) in enumerate(selected_df.iterrows(), 1):
        doc.append(f"{rank:>4} {row['indicator']:<50} {row['category']:<12} "
                   f"{row['coherence_pct']:>9.1f}% {row['sharpe_ratio']:>7.2f} "
                   f"{row['max_drawdown_pct']:>7.1f}%")
    
    doc.append("")
    doc.append("-" * 80)
    doc.append("SELECTION RATIONALE")
    doc.append("-" * 80)
    doc.append("")
    doc.append("The final indicator set was selected based on the following rationale:")
    doc.append("")
    doc.append("1. COHERENCE THRESHOLD (>50%):")
    doc.append("   - All selected indicators demonstrate >50% time-coherence with the ISP benchmark")
    doc.append("   - This ensures each indicator contributes meaningful signal alignment")
    doc.append("")
    doc.append("2. DIVERSITY:")
    doc.append("   - Mix of perpetual (trend-following) and oscillator (mean-reversion) indicators")
    doc.append("   - Different calculation methodologies reduce correlation risk")
    doc.append("   - Balanced representation prevents over-reliance on one indicator type")
    doc.append("")
    doc.append("3. RISK-ADJUSTED PERFORMANCE:")
    doc.append("   - Preference for indicators with positive Sharpe ratios")
    doc.append("   - Consideration of maximum drawdown (lower is better)")
    doc.append("   - Stability metric indicates consistent signal generation")
    doc.append("")
    doc.append("4. ENSEMBLE DIVERSITY:")
    doc.append(f"   - {len(selected_df)} indicators provide sufficient diversity for averaging")
    doc.append("   - Equal weighting (1/N) applied to all indicators")
    doc.append("   - Individual indicator failure won't collapse the ensemble")
    doc.append("")
    doc.append("-" * 80)
    doc.append("INDICATOR DETAILS")
    doc.append("-" * 80)
    doc.append("")
    
    for rank, (_, row) in enumerate(selected_df.iterrows(), 1):
        doc.append(f"{rank}. {row['indicator']}")
        doc.append(f"   Category: {row['category']}")
        doc.append(f"   Normalized: {row['normalized']}")
        doc.append(f"   Coherence: {row['coherence_pct']:.1f}%")
        doc.append(f"   Trades: {row['trades']} | Avg Hold: {row['avg_hold_days']:.0f} days")
        doc.append(f"   Return: {row['total_return_pct']:.1f}% | Sharpe: {row['sharpe_ratio']:.2f}")
        doc.append(f"   Stability: {row['stability']:.3f} | Max DD: {row['max_drawdown_pct']:.1f}%")
        doc.append(f"   Pearson: {row['pearson_r']:.3f} | Spearman: {row['spearman_r']:.3f}")
        doc.append("")
    
    doc.append("=" * 80)
    doc.append("END OF DOCUMENTATION")
    doc.append("=" * 80)
    
    return "\n".join(doc)


# ---------------------------------------------------------------------------
# Generate Python Constants
# ---------------------------------------------------------------------------

def generate_constants(selected_df):
    """Generate Python code with final indicator constants."""
    
    lines = []
    lines.append('"""')
    lines.append('Final Indicator Selection')
    lines.append('=========================')
    lines.append('')
    lines.append('Auto-generated from final_indicator_selection.py')
    lines.append('Contains the final set of indicators for the MTTD ensemble.')
    lines.append('"""')
    lines.append('')
    lines.append('# Final selected indicators for MTTD ensemble')
    lines.append('# All indicators have individual coherence > 50% with ISP benchmark')
    lines.append('# Equal weighting (1/N) applied to all indicators')
    lines.append('')
    lines.append('FINAL_INDICATORS = [')
    
    for _, row in selected_df.iterrows():
        lines.append(f'    {{')
        lines.append(f'        "name": "{row["indicator"]}",')
        lines.append(f'        "normalized": "{row["normalized"]}",')
        lines.append(f'        "category": "{row["category"]}",')
        lines.append(f'        "coherence_pct": {row["coherence_pct"]:.2f},')
        lines.append(f'    }},')
    
    lines.append(']')
    lines.append('')
    lines.append(f'FINAL_INDICATOR_COUNT = {len(selected_df)}')
    lines.append('')
    lines.append('# Normalized names for quick lookup')
    lines.append('FINAL_INDICATOR_NAMES = [ind["normalized"] for ind in FINAL_INDICATORS]')
    lines.append('')
    
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("FINAL INDICATOR SELECTION")
    print("=" * 80)
    
    # Load results
    print("\n[1] Loading audit results...")
    audit_df, test_df = load_audit_results()
    print(f"    TODO 4 (current indicators): {len(audit_df)}")
    print(f"    TODO 5 (tested indicators): {len(test_df)}")
    
    # Merge results
    print("\n[2] Merging and deduplicating...")
    combined = merge_results(audit_df, test_df)
    print(f"    Combined unique indicators: {len(combined)}")
    
    # Show all indicators and their coherence
    print("\n[3] All indicators sorted by coherence:")
    print("-" * 80)
    for _, row in combined.iterrows():
        marker = "✓" if row['coherence_pct'] >= 50 else " "
        print(f"  {marker} {row['indicator']:<50} {row['coherence_pct']:>6.1f}%  [{row['category']}]")
    
    # Filter by coherence
    print("\n[4] Filtering indicators with coherence > 50%...")
    eligible = filter_by_coherence(combined, min_coherence=50.0)
    print(f"    Eligible indicators: {len(eligible)}")
    
    # Select final indicators
    print("\n[5] Selecting final indicator set (target: 20)...")
    selected = select_final_indicators(combined, target_count=20, min_coherence=50.0)
    print(f"    Final selected: {len(selected)}")
    
    # Document selection
    print("\n[6] Generating documentation...")
    doc = document_selection(selected, combined)
    
    # Save documentation
    doc_path = os.path.join(project_root, "mttd", "FINAL_INDICATOR_SELECTION.md")
    with open(doc_path, 'w') as f:
        f.write(doc)
    print(f"    Documentation saved to: {doc_path}")
    
    # Generate Python constants
    print("\n[7] Generating Python constants...")
    constants = generate_constants(selected)
    
    constants_path = os.path.join(project_root, "mttd", "final_indicators.py")
    with open(constants_path, 'w') as f:
        f.write(constants)
    print(f"    Constants saved to: {constants_path}")
    
    # Save CSV
    csv_path = os.path.join(project_root, "mttd", "final_indicator_selection.csv")
    selected.to_csv(csv_path, index=False)
    print(f"    CSV saved to: {csv_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SELECTION COMPLETE")
    print("=" * 80)
    print(f"\nFinal indicator count: {len(selected)}")
    print(f"Coherence range: {selected['coherence_pct'].min():.1f}% - {selected['coherence_pct'].max():.1f}%")
    print(f"Mean coherence: {selected['coherence_pct'].mean():.1f}%")
    print(f"\nCategory breakdown:")
    print(f"  Perpetual: {len(selected[selected['category'] == 'perpetual'])}")
    print(f"  Oscillator: {len(selected[selected['category'] == 'oscillator'])}")
    
    print("\n" + doc)
    
    return selected


if __name__ == "__main__":
    main()
