#!/usr/bin/env python3
"""
Final Audit Report — Grid + Audit Consolidation
=================================================

Consolidates grid test results and audit results into:
  1. Console summary tables (Grid, Audit, Ranking)
  2. Visual comparison chart: mttd/top3_audit_comparison.png
  3. Markdown report: TOP3_AUDIT_REPORT.md

References:
  - mttd/top3_grid_results.csv  (from grid_test_top3.py — Task 1)
  - mttd/top3_audit_results.csv (from audit_top3.py — Task 2)
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
GRID_CSV     = os.path.join(MTTD_DIR, 'top3_grid_results.csv')
AUDIT_CSV    = os.path.join(MTTD_DIR, 'top3_audit_results.csv')
CHART_PNG    = os.path.join(MTTD_DIR, 'top3_audit_comparison.png')
REPORT_MD    = os.path.join(project_root, 'TOP3_AUDIT_REPORT.md')

# ================================================================
# Load Data
# ================================================================

def load_data():
    """Load grid and audit CSV files."""
    if not os.path.exists(GRID_CSV):
        print(f"ERROR: Grid results not found: {GRID_CSV}")
        sys.exit(1)
    if not os.path.exists(AUDIT_CSV):
        print(f"ERROR: Audit results not found: {AUDIT_CSV}")
        sys.exit(1)

    grid_df = pd.read_csv(GRID_CSV)
    audit_df = pd.read_csv(AUDIT_CSV)
    print(f"Loaded grid results:  {len(grid_df)} rows from {GRID_CSV}")
    print(f"Loaded audit results: {len(audit_df)} rows from {AUDIT_CSV}")
    return grid_df, audit_df


# ================================================================
# Console Tables
# ================================================================

def print_grid_summary(grid_df):
    """Print top configs per system from grid search."""
    print("\n" + "=" * 90)
    print("  TABLE 1 — GRID TEST RESULTS (Top Config per System by Test Sharpe)")
    print("=" * 90)
    header = (f"  {'System':<32s} {'mh':>4s} {'MH':>4s} {'rTh':>5s} "
              f"{'Train_SH':>9s} {'Test_SH':>9s} {'Degrad%':>8s} "
              f"{'Train_WR':>9s} {'Test_WR':>8s}")
    print(header)
    print("  " + "-" * 88)

    for sys_name in grid_df['system'].unique():
        sub = grid_df[grid_df['system'] == sys_name]
        valid = sub[(sub['train_trades'] > 0) & (sub['test_trades'] > 0)]
        if len(valid) == 0:
            print(f"  {sys_name:<32s}  — NO VALID CONFIGS —")
            continue
        best = valid.nlargest(1, 'test_sharpe').iloc[0]
        print(f"  {sys_name:<32s} {int(best['min_hold']):>4d} {int(best['max_hold']):>4d} "
              f"{best['regime_threshold']:>5.1f} "
              f"{best['train_sharpe']:>9.2f} {best['test_sharpe']:>9.2f} {best['degradation']:>+8.1f} "
              f"{best['train_win_rate']:>8.1f}% {best['test_win_rate']:>8.1f}%")

    # Print all configs count
    print(f"\n  Total grid combos tested: {len(grid_df)}")
    for sys_name in grid_df['system'].unique():
        n = len(grid_df[grid_df['system'] == sys_name])
        n_valid = len(grid_df[(grid_df['system'] == sys_name) &
                              (grid_df['train_trades'] > 0) &
                              (grid_df['test_trades'] > 0)])
        print(f"    {sys_name}: {n} combos, {n_valid} valid")


def print_audit_summary(audit_df):
    """Print audit scores for each system."""
    print("\n" + "=" * 90)
    print("  TABLE 2 — AUDIT RESULTS (lz-quant-researcher Methodology)")
    print("=" * 90)
    header = (f"  {'System':<32s} {'Config':>12s} {'WF_OOS_SH':>10s} {'Degrad%':>8s} "
              f"{'Overfit':>8s} {'Robust':>8s} {'Stats':>8s} {'AntiPat':>8s} {'TOTAL':>8s}")
    print(header)
    print("  " + "-" * 88)

    for _, row in audit_df.sort_values('total_audit_score', ascending=False).iterrows():
        config_str = f"mh{int(row['best_min_hold'])}_MH{int(row['best_max_hold'])}"
        print(f"  {row['system']:<32s} {config_str:>12s} "
              f"{row['wf_avg_oos_sharpe']:>10.2f} {row['avg_degradation_pct']:>+8.1f} "
              f"{row['overfitting_score']:>8.1f} {row['robustness_score']:>8.1f} "
              f"{row['statistical_score']:>8.1f} {row['antipattern_score']:>8.1f} "
              f"{row['total_audit_score']:>8.1f}")

    # Print regime breakdown
    print(f"\n  {'System':<32s} {'Bull_SH':>8s} {'Bear_SH':>8s} {'Neut_SH':>8s} "
          f"{'TC@0.1%':>8s} {'TC@0.5%':>8s} {'Sharpe_p':>9s} {'Binom_p':>9s}")
    print("  " + "-" * 88)
    for _, row in audit_df.sort_values('total_audit_score', ascending=False).iterrows():
        print(f"  {row['system']:<32s} "
              f"{row['bull_sharpe']:>8.2f} {row['bear_sharpe']:>8.2f} {row['neutral_sharpe']:>8.2f} "
              f"{row['sharpe_at_0p1tc']:>8.2f} {row['sharpe_at_0p5tc']:>8.2f} "
              f"{row['sharpe_p_value']:>9.4f} {row['winrate_binom_p']:>9.4f}")


def print_final_ranking(audit_df):
    """Print final ranking by total audit score."""
    print("\n" + "=" * 90)
    print("  TABLE 3 — FINAL RANKING (by Total Audit Score, 0–100)")
    print("=" * 90)

    ranked = audit_df.sort_values('total_audit_score', ascending=False).reset_index(drop=True)
    for i, row in ranked.iterrows():
        rank = i + 1
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        config_str = f"mh={int(row['best_min_hold'])} MH={int(row['best_max_hold'])} rTh={row['best_regime_threshold']:.1f}"
        print(f"\n  {medal} #{rank}  {row['system']}")
        print(f"     Score: {row['total_audit_score']:.1f}/100   |   Config: {config_str}")
        print(f"     Overfitting: {row['overfitting_score']:.1f}  |  Robustness: {row['robustness_score']:.1f}  |  "
              f"Statistical: {row['statistical_score']:.1f}  |  Anti-Pattern: {row['antipattern_score']:.1f}")
        print(f"     WF OOS Sharpe: {row['wf_avg_oos_sharpe']:.2f}  |  Degradation: {row['avg_degradation_pct']:+.1f}%  |  "
              f"Sharpe p={row['sharpe_p_value']:.4f}")


# ================================================================
# Chart Generation
# ================================================================

def generate_comparison_chart(grid_df, audit_df):
    """
    Generate 3-panel comparison chart:
      Panel 1: Audit score breakdown (radar-style bar chart)
      Panel 2: Grid search best config comparison
      Panel 3: Regime performance comparison
    """
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle('Top 3 Regime-Filtered Systems — Final Audit Comparison',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(1, 3, hspace=0.35, wspace=0.30,
                           left=0.06, right=0.96, top=0.90, bottom=0.08)

    systems = audit_df.sort_values('total_audit_score', ascending=False)['system'].tolist()
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # green, blue, red
    score_cols = ['overfitting_score', 'robustness_score', 'statistical_score', 'antipattern_score']
    score_labels = ['Overfitting\n(inverted)', 'Robustness', 'Statistical\nSignificance', 'Anti-Pattern']

    # ── Panel 1: Audit Score Breakdown ──
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(score_cols))
    width = 0.25
    for i, sys_name in enumerate(systems):
        row = audit_df[audit_df['system'] == sys_name].iloc[0]
        # Invert overfitting for display (higher = better)
        scores = [
            100.0 - row['overfitting_score'],
            row['robustness_score'],
            row['statistical_score'],
            row['antipattern_score'],
        ]
        bars = ax1.bar(x + i * width, scores, width, label=sys_name.replace('_', '\n'),
                       color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)
        # Add score labels on bars
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                     f'{score:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(score_labels, fontsize=9)
    ax1.set_ylabel('Score (0–100)', fontsize=10)
    ax1.set_title('Audit Score Breakdown', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── Panel 2: Grid Search Performance ──
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ['train_sharpe', 'test_sharpe', 'train_win_rate', 'test_win_rate']
    metric_labels = ['Train\nSharpe', 'Test\nSharpe', 'Train\nWinRate%', 'Test\nWinRate%']
    x2 = np.arange(len(metrics))
    width2 = 0.25

    for i, sys_name in enumerate(systems):
        sub = grid_df[grid_df['system'] == sys_name]
        valid = sub[(sub['train_trades'] > 0) & (sub['test_trades'] > 0)]
        if len(valid) == 0:
            vals = [0, 0, 0, 0]
        else:
            best = valid.nlargest(1, 'test_sharpe').iloc[0]
            vals = [best['train_sharpe'], best['test_sharpe'],
                    best['train_win_rate'], best['test_win_rate']]
        bars = ax2.bar(x2 + i * width2, vals, width2,
                       label=sys_name.replace('_', '\n'),
                       color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax2.set_xticks(x2 + width2)
    ax2.set_xticklabels(metric_labels, fontsize=9)
    ax2.set_ylabel('Value', fontsize=10)
    ax2.set_title('Best Config Performance (Grid)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ── Panel 3: Regime Performance ──
    ax3 = fig.add_subplot(gs[0, 2])
    regimes = ['Bull', 'Bear', 'Neutral']
    x3 = np.arange(len(regimes))
    width3 = 0.25

    for i, sys_name in enumerate(systems):
        row = audit_df[audit_df['system'] == sys_name].iloc[0]
        sharpes = [
            row.get('bull_sharpe', 0),
            row.get('bear_sharpe', 0),
            row.get('neutral_sharpe', 0),
        ]
        bars = ax3.bar(x3 + i * width3, sharpes, width3,
                       label=sys_name.replace('_', '\n'),
                       color=colors[i], alpha=0.85, edgecolor='white', linewidth=0.5)
        for bar, val in zip(bars, sharpes):
            ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax3.set_xticks(x3 + width3)
    ax3.set_xticklabels(regimes, fontsize=10)
    ax3.set_ylabel('Sharpe Ratio', fontsize=10)
    ax3.set_title('Regime Performance', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ── Save ──
    os.makedirs(MTTD_DIR, exist_ok=True)
    fig.savefig(CHART_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nChart saved: {CHART_PNG}")


# ================================================================
# Markdown Report
# ================================================================

def write_markdown_report(grid_df, audit_df):
    """Write comprehensive TOP3_AUDIT_REPORT.md."""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    ranked = audit_df.sort_values('total_audit_score', ascending=False).reset_index(drop=True)

    lines = []
    lines.append("# Top 3 Regime-Filtered Systems — Final Audit Report")
    lines.append("")
    lines.append(f"**Generated:** {now}")
    lines.append(f"**Methodology:** lz-quant-researcher audit (overfitting, robustness, statistics, anti-patterns)")
    lines.append(f"**Grid combos tested:** {len(grid_df)}")
    lines.append(f"**Systems audited:** {len(audit_df)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Summary Table ──
    lines.append("## Final Ranking")
    lines.append("")
    lines.append("| Rank | System | Total Score | Overfitting | Robustness | Statistical | Anti-Pattern |")
    lines.append("|------|--------|-------------|-------------|------------|-------------|--------------|")
    for i, row in ranked.iterrows():
        rank = i + 1
        lines.append(
            f"| {rank} | `{row['system']}` | **{row['total_audit_score']:.1f}** | "
            f"{row['overfitting_score']:.1f} | {row['robustness_score']:.1f} | "
            f"{row['statistical_score']:.1f} | {row['antipattern_score']:.1f} |"
        )
    lines.append("")

    # ── Per-System Details ──
    for i, row in ranked.iterrows():
        rank = i + 1
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        sys_name = row['system']
        lines.append(f"## {medal} #{rank} — {sys_name}")
        lines.append("")
        lines.append(f"**Total Audit Score: {row['total_audit_score']:.1f}/100**")
        lines.append("")

        # Config
        lines.append("### Best Configuration")
        lines.append("")
        lines.append(f"- **min_hold:** {int(row['best_min_hold'])}")
        lines.append(f"- **max_hold:** {int(row['best_max_hold'])}")
        lines.append(f"- **regime_threshold:** {row['best_regime_threshold']:.1f}")
        lines.append("")

        # Grid search
        lines.append("### Grid Search Results")
        lines.append("")
        sys_grid = grid_df[grid_df['system'] == sys_name]
        valid_grid = sys_grid[(sys_grid['train_trades'] > 0) & (sys_grid['test_trades'] > 0)]
        lines.append(f"- **Total combos tested:** {len(sys_grid)}")
        lines.append(f"- **Valid combos:** {len(valid_grid)}")
        if len(valid_grid) > 0:
            best = valid_grid.nlargest(1, 'test_sharpe').iloc[0]
            lines.append(f"- **Best test Sharpe:** {best['test_sharpe']:.2f}")
            lines.append(f"- **Train Sharpe:** {best['train_sharpe']:.2f}")
            lines.append(f"- **Degradation:** {best['degradation']:+.1f}%")
            lines.append(f"- **Train Win Rate:** {best['train_win_rate']:.1f}%")
            lines.append(f"- **Test Win Rate:** {best['test_win_rate']:.1f}%")
        lines.append("")

        # Walk-forward
        lines.append("### Walk-Forward Validation")
        lines.append("")
        lines.append(f"- **Folds:** {int(row['wf_n_folds'])}")
        lines.append(f"- **Avg OOS Sharpe:** {row['wf_avg_oos_sharpe']:.2f}")
        lines.append(f"- **Avg OOS Win Rate:** {row['wf_avg_oos_winrate']:.1f}%")
        lines.append(f"- **Degradation:** {row['avg_degradation_pct']:+.1f}%")
        lines.append("")

        # Regime
        lines.append("### Regime Performance")
        lines.append("")
        lines.append("| Regime | Sharpe |")
        lines.append("|--------|--------|")
        lines.append(f"| Bull | {row['bull_sharpe']:.2f} |")
        lines.append(f"| Bear | {row['bear_sharpe']:.2f} |")
        lines.append(f"| Neutral | {row['neutral_sharpe']:.2f} |")
        lines.append("")

        # Transaction cost
        lines.append("### Transaction Cost Sensitivity")
        lines.append("")
        lines.append(f"- **Sharpe @ 0.1% TC:** {row['sharpe_at_0p1tc']:.2f}")
        lines.append(f"- **Sharpe @ 0.5% TC:** {row['sharpe_at_0p5tc']:.2f}")
        tc_drop = row['sharpe_at_0p1tc'] - row['sharpe_at_0p5tc']
        lines.append(f"- **Sharpe drop (0.1% → 0.5%):** {tc_drop:+.2f}")
        lines.append("")

        # Statistical
        lines.append("### Statistical Significance")
        lines.append("")
        sig_status = "✅ SIGNIFICANT" if row['sharpe_p_value'] < 0.05 else "❌ NOT SIGNIFICANT"
        lines.append(f"- **Sharpe t-test:** t={row['sharpe_t_stat']:.3f}, p={row['sharpe_p_value']:.4f} ({sig_status})")
        binom_status = "✅ SIGNIFICANT" if row['winrate_binom_p'] < 0.05 else "❌ NOT SIGNIFICANT"
        lines.append(f"- **Win rate binomial test:** p={row['winrate_binom_p']:.4f} ({binom_status})")
        lines.append("")

        # Score breakdown
        lines.append("### Audit Score Breakdown")
        lines.append("")
        lines.append("| Criterion | Score | Notes |")
        lines.append("|-----------|-------|-------|")
        inv_overfit = 100.0 - row['overfitting_score']
        overfit_note = "LOW risk" if inv_overfit >= 70 else "MEDIUM risk" if inv_overfit >= 40 else "HIGH risk"
        lines.append(f"| Overfitting (inverted) | {inv_overfit:.1f} | {overfit_note} |")
        robust_note = "ROBUST" if row['robustness_score'] >= 70 else "MODERATE" if row['robustness_score'] >= 40 else "FRAGILE"
        lines.append(f"| Robustness | {row['robustness_score']:.1f} | {robust_note} |")
        stat_note = "SOUND" if row['statistical_score'] >= 60 else "WEAK" if row['statistical_score'] >= 30 else "INSUFFICIENT"
        lines.append(f"| Statistical | {row['statistical_score']:.1f} | {stat_note} |")
        ap_note = "CLEAN" if row['antipattern_score'] >= 90 else "MINOR FLAGS" if row['antipattern_score'] >= 70 else "CONCERNS"
        lines.append(f"| Anti-Pattern | {row['antipattern_score']:.1f} | {ap_note} |")
        lines.append("")

    # ── Comparison Chart ──
    lines.append("---")
    lines.append("")
    lines.append("## Comparison Chart")
    lines.append("")
    lines.append("![Top 3 Audit Comparison](mttd/top3_audit_comparison.png)")
    lines.append("")

    # ── Methodology ──
    lines.append("---")
    lines.append("")
    lines.append("## Methodology Notes")
    lines.append("")
    lines.append("### Audit Criteria")
    lines.append("")
    lines.append("1. **Overfitting Risk (0–100):** Walk-forward Sharpe decay, overall degradation, OOS consistency. Score is *inverted* (100 = no overfitting risk).")
    lines.append("2. **Robustness (0–100):** Regime consistency, transaction cost resilience, degradation tolerance.")
    lines.append("3. **Statistical Significance (0–100):** Sharpe t-test p-value, win rate binomial test, trade count power.")
    lines.append("4. **Anti-Pattern (0–100):** Look-ahead bias, survivorship bias, data snooping, hardcoded dates, missing TC.")
    lines.append("")
    lines.append("**Total Score = mean(Inverted Overfitting, Robustness, Statistical, Anti-Pattern)**")
    lines.append("")
    lines.append("### Grid Search")
    lines.append("")
    lines.append("- Parameter matrix: min_hold × max_hold × regime_threshold")
    lines.append("- Holdout: 2018–2024 train, 2025–2026 test")
    lines.append("- Transaction cost: 0.1% round-trip")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated by `final_audit_report.py` on {now}*")
    lines.append("")

    os.makedirs(os.path.dirname(REPORT_MD), exist_ok=True)
    with open(REPORT_MD, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Report saved: {REPORT_MD}")


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 90)
    print("  FINAL AUDIT REPORT — Grid + Audit Consolidation")
    print("=" * 90)

    # Load data
    grid_df, audit_df = load_data()

    # Console tables
    print_grid_summary(grid_df)
    print_audit_summary(audit_df)
    print_final_ranking(audit_df)

    # Generate chart
    print("\n" + "=" * 90)
    print("  Generating comparison chart...")
    print("=" * 90)
    generate_comparison_chart(grid_df, audit_df)

    # Write markdown report
    print("\n" + "=" * 90)
    print("  Writing markdown report...")
    print("=" * 90)
    write_markdown_report(grid_df, audit_df)

    # Final summary
    print("\n" + "=" * 90)
    print("  FINAL AUDIT REPORT — COMPLETE")
    print("=" * 90)
    print(f"  Chart:   {CHART_PNG}")
    print(f"  Report:  {REPORT_MD}")
    print("=" * 90)


if __name__ == '__main__':
    main()
