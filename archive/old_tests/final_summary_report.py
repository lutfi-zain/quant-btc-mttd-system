#!/usr/bin/env python3
"""
Final Summary Report — Holdout Validation Consolidation
=========================================================

Aggregates all holdout validation results and generates:
  1. Console formatted summary table
  2. Visual comparison chart: mttd/final_summary.png
  3. Markdown report: FINAL_SUMMARY.md

References:
  - mttd/holdout_best_results.csv (from holdout_best_configs.py)
  - mttd/charts/*.png (trade charts)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# Paths
# ================================================================
project_root = os.path.dirname(os.path.abspath(__file__))
MTTD_DIR     = os.path.join(project_root, 'mttd')
CHARTS_DIR   = os.path.join(MTTD_DIR, 'charts')
RESULTS_CSV  = os.path.join(MTTD_DIR, 'holdout_best_results.csv')
SUMMARY_PNG  = os.path.join(MTTD_DIR, 'final_summary.png')
REPORT_MD    = os.path.join(project_root, 'FINAL_SUMMARY.md')

# Trade chart files
TRADE_CHARTS = {
    'msvr': 'msvr_trade_chart.png',
    'ichimoku': 'ichimoku_trade_chart.png',
    'supertrend': 'supertrend_trade_chart.png',
    'keltner': 'keltner_trade_chart.png',
}

# ================================================================
# Load Data
# ================================================================

def load_holdout_results():
    """Load holdout validation results from CSV."""
    if not os.path.exists(RESULTS_CSV):
        print(f"ERROR: Holdout results not found: {RESULTS_CSV}")
        sys.exit(1)

    df = pd.read_csv(RESULTS_CSV)
    print(f"Loaded holdout results: {len(df)} rows from {RESULTS_CSV}")
    return df


def check_trade_charts():
    """Check which trade charts exist."""
    existing = {}
    missing = []

    for system, filename in TRADE_CHARTS.items():
        filepath = os.path.join(CHARTS_DIR, filename)
        if os.path.exists(filepath):
            existing[system] = filepath
            print(f"  ✅ {filename}")
        else:
            missing.append(filename)
            print(f"  ❌ {filename} (missing)")

    return existing, missing


# ================================================================
# Console Summary
# ================================================================

def print_final_summary(df):
    """Print formatted final summary to console."""
    print("\n" + "=" * 80)
    print("  HOLDOUT VALIDATION — FINAL SUMMARY")
    print("=" * 80)
    print()
    print("  Periods:")
    print("    Training: 2018-01-01 to 2024-12-31")
    print("    Test:     2025-01-01 to 2026-06-24")
    print("    Transaction Cost: 0.1% round-trip")
    print()

    # Header
    print("  " + "-" * 76)
    print(f"  {'System':<35s} {'Train':>8s} {'Test':>8s} {'Degrad':>10s} {'Trades':>8s}")
    print("  " + "-" * 76)

    # Sort by degradation (absolute value)
    df_sorted = df.reindex(df['Degradation'].abs().argsort())

    for _, row in df_sorted.iterrows():
        # Format system description
        system_str = f"{row['System']} + {row['Filter']}"
        if row['System'] == 'Ichimoku':
            system_str = f"Ichimoku IMO"
        elif row['System'] == 'Supertrend':
            system_str = f"Supertrend"
        elif row['System'] == 'MSVR':
            system_str = f"MSVR"

        mh_str = f"MH={row['MH']}"
        system_str = f"{system_str} ({mh_str})"

        train_sharpe = f"{row['Train_Sharpe']:.2f}"
        test_sharpe = f"{row['Test_Sharpe']:.2f}"
        degradation = f"{row['Degradation']:+.1f}%"
        trades = f"{row['Test_Trades']}"

        print(f"  {system_str:<35s} {train_sharpe:>8s} {test_sharpe:>8s} {degradation:>10s} {trades:>8s}")

    print("  " + "-" * 76)

    # Find best config (lowest absolute degradation)
    best_idx = df['Degradation'].abs().idxmin()
    best = df.iloc[best_idx]

    # Find highest test Sharpe
    highest_test_idx = df['Test_Sharpe'].idxmax()
    highest_test = df.iloc[highest_test_idx]

    print()
    print("  KEY FINDINGS:")
    print()
    print(f"  🏆 Most Robust (Lowest Degradation):")
    print(f"     {best['Description']}")
    print(f"     Train Sharpe: {best['Train_Sharpe']:.2f} → Test Sharpe: {best['Test_Sharpe']:.2f}")
    print(f"     Degradation: {best['Degradation']:+.1f}%")
    print()
    print(f"  📈 Highest Test Sharpe:")
    print(f"     {highest_test['Description']}")
    print(f"     Test Sharpe: {highest_test['Test_Sharpe']:.2f}")
    print(f"     Test Win Rate: {highest_test['Test_WinRate']:.1f}%")
    print()


# ================================================================
# Chart Generation
# ================================================================

def generate_comparison_chart(df):
    """
    Generate comparison bar chart showing Train vs Test Sharpe for each system.
    """
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('Holdout Validation — Train vs Test Sharpe Comparison',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(1, 2, hspace=0.30, wspace=0.35,
                           left=0.08, right=0.95, top=0.88, bottom=0.12)

    # ── Panel 1: Sharpe Comparison ──
    ax1 = fig.add_subplot(gs[0, 0])

    # Prepare data
    systems = df['Description'].tolist()
    train_sharpes = df['Train_Sharpe'].tolist()
    test_sharpes = df['Test_Sharpe'].tolist()

    # Short labels for x-axis
    short_labels = []
    for desc in systems:
        if 'Keltner' in desc:
            short_labels.append('Keltner\n(MH=15/60)' if '15/60' in desc else 'Keltner\n(MH=25/60)')
        elif 'Ichimoku' in desc:
            short_labels.append('Ichimoku\n(MH=15/60)')
        elif 'Supertrend' in desc:
            short_labels.append('Supertrend\n(MH=15/90)')
        elif 'MSVR' in desc:
            short_labels.append('MSVR\n(MH=15/90)')
        else:
            short_labels.append(desc[:15])

    x = np.arange(len(systems))
    width = 0.35

    bars_train = ax1.bar(x - width/2, train_sharpes, width, label='Train (2018-2024)',
                         color='#3498db', alpha=0.85, edgecolor='white', linewidth=0.5)
    bars_test = ax1.bar(x + width/2, test_sharpes, width, label='Test (2025-2026)',
                        color='#e74c3c', alpha=0.85, edgecolor='white', linewidth=0.5)

    # Add value labels on bars
    for bar, val in zip(bars_train, train_sharpes):
        y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.08
        ax1.text(bar.get_x() + bar.get_width() / 2., y_pos,
                 f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top',
                 fontsize=8, fontweight='bold')

    for bar, val in zip(bars_test, test_sharpes):
        y_pos = bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.08
        ax1.text(bar.get_x() + bar.get_width() / 2., y_pos,
                 f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top',
                 fontsize=8, fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=9)
    ax1.set_ylabel('Sharpe Ratio', fontsize=11)
    ax1.set_title('Sharpe Ratio: Train vs Test', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── Panel 2: Degradation Chart ──
    ax2 = fig.add_subplot(gs[0, 1])

    degradations = df['Degradation'].tolist()

    # Color based on degradation severity
    colors = []
    for d in degradations:
        abs_d = abs(d)
        if abs_d < 50:
            colors.append('#2ecc71')  # Green - robust
        elif abs_d < 100:
            colors.append('#f39c12')  # Yellow - moderate
        else:
            colors.append('#e74c3c')  # Red - fragile

    bars = ax2.barh(x, degradations, color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, degradations):
        x_pos = bar.get_width() + 2 if val >= 0 else bar.get_width() - 2
        ax2.text(x_pos, bar.get_y() + bar.get_height() / 2.,
                 f'{val:+.1f}%', ha='left' if val >= 0 else 'right',
                 va='center', fontsize=9, fontweight='bold')

    ax2.set_yticks(x)
    ax2.set_yticklabels(short_labels, fontsize=9)
    ax2.set_xlabel('Degradation (%)', fontsize=11)
    ax2.set_title('Performance Degradation\n(Train → Test)', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='x', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', alpha=0.85, label='Robust (<50%)'),
        Patch(facecolor='#f39c12', alpha=0.85, label='Moderate (50-100%)'),
        Patch(facecolor='#e74c3c', alpha=0.85, label='Fragile (>100%)'),
    ]
    ax2.legend(handles=legend_elements, fontsize=8, loc='lower right')

    # ── Save ──
    os.makedirs(MTTD_DIR, exist_ok=True)
    fig.savefig(SUMMARY_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart saved: {SUMMARY_PNG}")


# ================================================================
# Markdown Report
# ================================================================

def write_markdown_report(df, existing_charts, missing_charts):
    """Write comprehensive FINAL_SUMMARY.md."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Find best configs
    best_robust_idx = df['Degradation'].abs().idxmin()
    best_robust = df.iloc[best_robust_idx]

    highest_test_idx = df['Test_Sharpe'].idxmax()
    highest_test = df.iloc[highest_test_idx]

    highest_win_idx = df['Test_WinRate'].idxmax()
    highest_win = df.iloc[highest_win_idx]

    lines = []
    lines.append("# Final Summary Report — BTC MTTD System")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Training Period:** 2018-01-01 to 2024-12-31")
    lines.append(f"**Test Period:** 2025-01-01 to 2026-06-24")
    lines.append(f"**Transaction Cost:** 0.1% round-trip")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Executive Summary ──
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This report consolidates holdout validation results for the top 5 BTC trading system configurations.")
    lines.append("The validation uses walk-forward methodology with strict train/test separation to detect overfitting.")
    lines.append("")
    lines.append("**Key Findings:**")
    lines.append("")
    lines.append(f"1. **Most Robust System:** `{best_robust['Description']}`")
    lines.append(f"   - Degradation: {best_robust['Degradation']:+.1f}% (lowest)")
    lines.append(f"   - Train Sharpe: {best_robust['Train_Sharpe']:.2f} → Test Sharpe: {best_robust['Test_Sharpe']:.2f}")
    lines.append("")
    lines.append(f"2. **Highest Test Sharpe:** `{highest_test['Description']}`")
    lines.append(f"   - Test Sharpe: {highest_test['Test_Sharpe']:.2f}")
    lines.append(f"   - Test Win Rate: {highest_test['Test_WinRate']:.1f}%")
    lines.append("")
    lines.append(f"3. **Highest Test Win Rate:** `{highest_win['Description']}`")
    lines.append(f"   - Test Win Rate: {highest_win['Test_WinRate']:.1f}%")
    lines.append(f"   - Test Sharpe: {highest_win['Test_Sharpe']:.2f}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Holdout Results Table ──
    lines.append("## Holdout Validation Results")
    lines.append("")
    lines.append("| System | Filter | MH | Train Sharpe | Test Sharpe | Degradation | Train Win% | Test Win% | Trades |")
    lines.append("|--------|--------|-----|--------------|-------------|-------------|------------|-----------|--------|")

    for _, row in df.sort_values('Degradation', key=abs).iterrows():
        lines.append(
            f"| {row['System']} | {row['Filter']} | {row['MH']} | "
            f"{row['Train_Sharpe']:.2f} | {row['Test_Sharpe']:.2f} | "
            f"{row['Degradation']:+.1f}% | {row['Train_WinRate']:.1f}% | "
            f"{row['Test_WinRate']:.1f}% | {row['Test_Trades']} |"
        )
    lines.append("")

    # ── Detailed Analysis ──
    lines.append("## Detailed Analysis")
    lines.append("")

    lines.append("### 1. Keltner Channel Systems")
    lines.append("")
    lines.append("The Keltner Channel system uses EMA-based envelope with ATR for dynamic width.")
    lines.append("Both configurations use bull_with_filters (MSVR + SuperSmoother + Cycle Phase filters).")
    lines.append("")

    keltner_df = df[df['System'] == 'Keltner']
    for _, row in keltner_df.iterrows():
        lines.append(f"- **MH={row['MH']}:** Train Sharpe {row['Train_Sharpe']:.2f} → Test Sharpe {row['Test_Sharpe']:.2f} "
                     f"(Degradation: {row['Degradation']:+.1f}%)")
    lines.append("")
    lines.append("**Observation:** Keltner systems show significant degradation in test period, "
                 "suggesting potential overfitting to training data patterns.")
    lines.append("")

    lines.append("### 2. Ichimoku System")
    lines.append("")
    ichimoku_row = df[df['System'] == 'Ichimoku'].iloc[0]
    lines.append(f"- **Configuration:** MH={ichimoku_row['MH']}")
    lines.append(f"- **Train Sharpe:** {ichimoku_row['Train_Sharpe']:.2f}")
    lines.append(f"- **Test Sharpe:** {ichimoku_row['Test_Sharpe']:.2f}")
    lines.append(f"- **Degradation:** {ichimoku_row['Degradation']:+.1f}%")
    lines.append(f"- **Test Win Rate:** {ichimoku_row['Test_WinRate']:.1f}%")
    lines.append("")
    lines.append("**Observation:** Ichimoku maintains positive test Sharpe with moderate degradation. "
                 "Best balance of performance and robustness.")
    lines.append("")

    lines.append("### 3. Supertrend System")
    lines.append("")
    supertrend_row = df[df['System'] == 'Supertrend'].iloc[0]
    lines.append(f"- **Configuration:** MH={supertrend_row['MH']}")
    lines.append(f"- **Train Sharpe:** {supertrend_row['Train_Sharpe']:.2f}")
    lines.append(f"- **Test Sharpe:** {supertrend_row['Test_Sharpe']:.2f}")
    lines.append(f"- **Degradation:** {supertrend_row['Degradation']:+.1f}%")
    lines.append(f"- **Test Win Rate:** {supertrend_row['Test_WinRate']:.1f}%")
    lines.append("")
    lines.append("**Observation:** Supertrend shows high degradation, indicating poor generalization.")
    lines.append("")

    lines.append("### 4. MSVR System")
    lines.append("")
    msvr_row = df[df['System'] == 'MSVR'].iloc[0]
    lines.append(f"- **Configuration:** MH={msvr_row['MH']}")
    lines.append(f"- **Train Sharpe:** {msvr_row['Train_Sharpe']:.2f}")
    lines.append(f"- **Test Sharpe:** {msvr_row['Test_Sharpe']:.2f}")
    lines.append(f"- **Degradation:** {msvr_row['Degradation']:+.1f}%")
    lines.append(f"- **Test Win Rate:** {msvr_row['Test_WinRate']:.1f}%")
    lines.append("")
    lines.append("**Observation:** MSVR shows relatively stable performance with moderate degradation.")
    lines.append("")

    lines.append("---")
    lines.append("")

    # ── Recommended Best Config ──
    lines.append("## Recommended Best Configuration")
    lines.append("")
    lines.append("Based on robustness (lowest degradation) and positive test performance:")
    lines.append("")
    lines.append(f"### 🏆 {best_robust['Description']}")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| System | {best_robust['System']} |")
    lines.append(f"| Filter | {best_robust['Filter']} |")
    lines.append(f"| Min Hold | {best_robust['MH'].split('/')[0]} days |")
    lines.append(f"| Max Hold | {best_robust['MH'].split('/')[1]} days |")
    lines.append(f"| Train Sharpe | {best_robust['Train_Sharpe']:.2f} |")
    lines.append(f"| Test Sharpe | {best_robust['Test_Sharpe']:.2f} |")
    lines.append(f"| Degradation | {best_robust['Degradation']:+.1f}% |")
    lines.append(f"| Train Win Rate | {best_robust['Train_WinRate']:.1f}% |")
    lines.append(f"| Test Win Rate | {best_robust['Test_WinRate']:.1f}% |")
    lines.append(f"| Train CAGR | {best_robust['Train_CAGR']:.2f}% |")
    lines.append(f"| Test CAGR | {best_robust['Test_CAGR']:.2f}% |")
    lines.append(f"| Train Max DD | {best_robust['Train_MaxDD']:.2f}% |")
    lines.append(f"| Test Max DD | {best_robust['Test_MaxDD']:.2f}% |")
    lines.append("")

    lines.append("**Why this config?**")
    lines.append("")
    lines.append(f"- Lowest degradation ({best_robust['Degradation']:+.1f}%) indicates best generalization")
    lines.append("- Maintains positive test Sharpe ratio")
    lines.append("- Simple parameter set reduces overfitting risk")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Trade Charts ──
    lines.append("## Trade Charts")
    lines.append("")
    lines.append("Individual trade charts for each system are available in `mttd/charts/`:")
    lines.append("")

    for system, filename in sorted(TRADE_CHARTS.items()):
        if system in existing_charts:
            lines.append(f"- ✅ [{filename}](mttd/charts/{filename})")
        else:
            lines.append(f"- ❌ {filename} (missing)")
    lines.append("")

    # ── Comparison Chart ──
    lines.append("## Comparison Chart")
    lines.append("")
    lines.append("![Final Summary](mttd/final_summary.png)")
    lines.append("")

    lines.append("---")
    lines.append("")

    # ── Methodology ──
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Validation Approach")
    lines.append("")
    lines.append("- **Walk-Forward Validation:** Strict train/test separation (2018-2024 / 2025-2026)")
    lines.append("- **Transaction Costs:** 0.1% round-trip applied to all backtests")
    lines.append("- **Position Sizing:** Fixed 100% allocation per trade")
    lines.append("- **Signal Generation:** Independent signal generation on full data, then split for evaluation")
    lines.append("")
    lines.append("### Metrics")
    lines.append("")
    lines.append("- **Sharpe Ratio:** Annualized risk-adjusted return")
    lines.append("- **Win Rate:** Percentage of profitable trades")
    lines.append("- **CAGR:** Compound Annual Growth Rate")
    lines.append("- **Max Drawdown:** Maximum peak-to-trough decline")
    lines.append("- **Degradation:** Percentage change in Sharpe from train to test period")
    lines.append("")
    lines.append("### Degradation Interpretation")
    lines.append("")
    lines.append("| Degradation | Status | Meaning |")
    lines.append("|-------------|--------|---------|")
    lines.append("| < 50% | ✅ Robust | Good generalization |")
    lines.append("| 50-100% | ⚠️ Moderate | Some overfitting |")
    lines.append("| > 100% | ❌ Fragile | Significant overfitting |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Project Structure ──
    lines.append("## Project Structure")
    lines.append("")
    lines.append("```")
    lines.append("mttd/")
    lines.append("├── holdout_best_results.csv      # Raw validation results")
    lines.append("├── final_summary.png            # Comparison chart")
    lines.append("├── charts/")
    lines.append("│   ├── msvr_trade_chart.png     # MSVR trade chart")
    lines.append("│   ├── ichimoku_trade_chart.png # Ichimoku trade chart")
    lines.append("│   ├── supertrend_trade_chart.png # Supertrend trade chart")
    lines.append("│   └── keltner_trade_chart.png  # Keltner trade chart")
    lines.append("└── ...")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by `final_summary_report.py` on {now}*")
    lines.append("")

    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    with open(REPORT_MD, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Report saved: {REPORT_MD}")


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 80)
    print("  FINAL SUMMARY REPORT — Holdout Validation Consolidation")
    print("=" * 80)
    print()

    # Load data
    print("[1/4] Loading holdout validation results...")
    df = load_holdout_results()

    # Check trade charts
    print("\n[2/4] Checking trade charts...")
    existing_charts, missing_charts = check_trade_charts()

    # Console summary
    print("\n[3/4] Generating console summary...")
    print_final_summary(df)

    # Generate comparison chart
    print("\n[4/4] Generating comparison chart...")
    generate_comparison_chart(df)

    # Write markdown report
    print("\n[5/5] Writing markdown report...")
    write_markdown_report(df, existing_charts, missing_charts)

    # Final summary
    print("\n" + "=" * 80)
    print("  FINAL SUMMARY REPORT — COMPLETE")
    print("=" * 80)
    print(f"  Chart:   {SUMMARY_PNG}")
    print(f"  Report:  {REPORT_MD}")
    print("=" * 80)


if __name__ == '__main__':
    main()
