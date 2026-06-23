#!/usr/bin/env python3
"""
MTTD System Chart — BTC + Composite + Individual Indicators
Single image output showing the full ensemble system.
"""
import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

project_root = "/home/ubuntu/projects/quant-technical-indicator-bank"
sys.path.append(project_root)

# ─── Load Data ───────────────────────────────────────────────────────────────
data_path = os.path.join(project_root, "mttd", "mttd_data.json")
with open(data_path, "r") as f:
    data = json.load(f)

# Candles
candles = data["candles"]
df = pd.DataFrame(candles)
df["time"] = pd.to_datetime(df["time"])
df.set_index("time", inplace=True)

# Aggregate signals
agg_signals = data["aggregate"]["signals"]
agg_df = pd.DataFrame(agg_signals)
agg_df["time"] = pd.to_datetime(agg_df["time"])
agg_df.set_index("time", inplace=True)

# Net vote
net_vote = data["aggregate"]["net_vote"]
nv_df = pd.DataFrame(net_vote)
nv_df["time"] = pd.to_datetime(nv_df["time"])
nv_df.set_index("time", inplace=True)

# Individual indicators
indicators = data["indicators"]
indicator_names = list(indicators.keys())

# ISP signals for comparison
csv_path = os.path.join(project_root, "isp-signals-btcusd-2026-06-13.csv")
isp_df = pd.read_csv(csv_path)
isp_df["Date"] = pd.to_datetime(isp_df["Date"])

# ─── Build ISP Position Series ───────────────────────────────────────────────
isp_position = pd.Series(0.0, index=df.index)
for _, row in isp_df.iterrows():
    date = row["Date"]
    regime = row["Regime"]
    if date in isp_position.index:
        if regime in ["Strong Bull", "Weak Bull"]:
            isp_position.loc[date:] = 1.0
        else:
            isp_position.loc[date:] = 0.0

# ─── Plot Setup ──────────────────────────────────────────────────────────────
n_indicators = len(indicator_names)
n_rows = 3 + n_indicators  # BTC + Composite + NetVote + individual indicators

fig = plt.figure(figsize=(24, 6 + n_indicators * 1.8), facecolor="#0d1117")
gs = GridSpec(n_rows, 1, figure=fig, height_ratios=[3, 1.5, 1.2] + [1.2] * n_indicators,
              hspace=0.08, left=0.06, right=0.94, top=0.95, bottom=0.03)

# Color scheme
bg_color = "#0d1117"
text_color = "#e6edf3"
green = "#10b981"
red = "#f43f5e"
blue = "#3b82f6"
yellow = "#f59e0b"
purple = "#a855f7"
grid_color = "#21262d"

# ─── Row 1: BTC Price Chart ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.set_facecolor(bg_color)
ax1.plot(df.index, df["close"], color=blue, linewidth=1.5, label="BTC/USD", zorder=3)

# Shade positions: green when long, gray when cash
for i in range(1, len(agg_df)):
    if agg_df["value"].iloc[i] == 1.0:
        ax1.axvspan(agg_df.index[i-1], agg_df.index[i], alpha=0.15, color=green, zorder=1)
    elif agg_df["value"].iloc[i] == 0.0:
        ax1.axvspan(agg_df.index[i-1], agg_df.index[i], alpha=0.08, color="#333", zorder=1)

# ISP markers
for _, row in isp_df.iterrows():
    if row["Action"] == "BUY":
        ax1.axvline(row["Date"], color=green, alpha=0.4, linewidth=0.8, linestyle="--", zorder=2)
    elif row["Action"] == "SELL":
        ax1.axvline(row["Date"], color=red, alpha=0.4, linewidth=0.8, linestyle="--", zorder=2)

ax1.set_ylabel("BTC/USD", color=text_color, fontsize=11, fontweight="bold")
ax1.set_title("MTTD Ensemble System — BTC + Composite + 20 Individual Indicators",
              color=text_color, fontsize=14, fontweight="bold", pad=12)
ax1.legend(loc="upper left", fontsize=9, facecolor="#161b22", edgecolor="#30363d",
           labelcolor=text_color)
ax1.tick_params(colors=text_color, labelsize=9)
ax1.set_xlim(df.index[0], df.index[-1])
ax1.grid(True, alpha=0.15, color=grid_color)
ax1.set_xticklabels([])
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

# ─── Row 2: Composite Position ──────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.set_facecolor(bg_color)
ax2.fill_between(agg_df.index, 0, agg_df["value"], color=green, alpha=0.5, label="MTTD Position")
ax2.plot(agg_df.index, agg_df["value"], color=green, linewidth=1)
ax2.set_ylabel("Position", color=text_color, fontsize=10, fontweight="bold")
ax2.set_ylim(-0.1, 1.2)
ax2.set_yticks([0, 0.5, 1.0])
ax2.set_yticklabels(["0% Cash", "50%", "100% BTC"], fontsize=8)
ax2.legend(loc="upper left", fontsize=8, facecolor="#161b22", edgecolor="#30363d",
           labelcolor=text_color)
ax2.tick_params(colors=text_color, labelsize=8)
ax2.set_xticklabels([])
ax2.grid(True, alpha=0.15, color=grid_color)

# ISP overlay
ax2.step(isp_position.index, isp_position.values, color=yellow, linewidth=1.2,
         alpha=0.7, linestyle="--", label="ISP Target", where="post")
ax2.legend(loc="upper left", fontsize=8, facecolor="#161b22", edgecolor="#30363d",
           labelcolor=text_color)

# ─── Row 3: Net Vote ────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.set_facecolor(bg_color)
colors_vote = [green if v > 0 else red if v < 0 else "#666" for v in nv_df["value"]]
ax3.bar(nv_df.index, nv_df["value"], color=colors_vote, alpha=0.6, width=1.5)
ax3.axhline(y=0, color=text_color, linewidth=0.5, alpha=0.3)
ax3.axhline(y=10, color=green, linewidth=0.5, alpha=0.3, linestyle="--")
ax3.axhline(y=-10, color=red, linewidth=0.5, alpha=0.3, linestyle="--")
ax3.set_ylabel("Net Vote", color=text_color, fontsize=10, fontweight="bold")
ax3.set_ylim(-16, 16)
ax3.set_yticks([-15, -10, -5, 0, 5, 10, 15])
ax3.tick_params(colors=text_color, labelsize=8)
ax3.set_xticklabels([])
ax3.grid(True, alpha=0.15, color=grid_color)

# ─── Rows 4+: Individual Indicators ──────────────────────────────────────────
for i, ind_name in enumerate(indicator_names):
    ax = fig.add_subplot(gs[3 + i], sharex=ax1)
    ax.set_facecolor(bg_color)

    ind_data = indicators[ind_name]
    ind_signals = ind_data["signals"]
    ind_values = ind_data["values"]

    # Build signal series
    sig_df = pd.DataFrame(ind_signals)
    sig_df["time"] = pd.to_datetime(sig_df["time"])
    sig_df.set_index("time", inplace=True)

    # Build value series
    val_df = pd.DataFrame(ind_values)
    val_df["time"] = pd.to_datetime(val_df["time"])
    val_df.set_index("time", inplace=True)

    # Plot indicator value (normalized to 0-1 range for comparison)
    if "value" in val_df.columns:
        vals = val_df["value"].reindex(df.index)
        # Normalize to 0-1 range
        v_min, v_max = vals.min(), vals.max()
        if v_max > v_min:
            vals_norm = (vals - v_min) / (v_max - v_min)
        else:
            vals_norm = vals * 0 + 0.5

        # Color based on signal
        sig_vals = sig_df["value"].reindex(df.index).fillna(0)
        colors_ind = [green if s > 0 else red for s in sig_vals]

        ax.scatter(df.index, vals_norm, c=colors_ind, s=1, alpha=0.6, zorder=2)
        ax.plot(df.index, vals_norm, color=text_color, linewidth=0.5, alpha=0.3, zorder=1)

    # Display name
    display_name = ind_data.get("name", ind_name)[:40]
    ax.set_ylabel(display_name, color=text_color, fontsize=7, fontweight="bold", rotation=0, labelpad=100, ha="right")
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([])
    ax.tick_params(colors=text_color, labelsize=7)
    ax.grid(True, alpha=0.1, color=grid_color)

    # Add x-axis labels only on last row
    if i == n_indicators - 1:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", colors=text_color, labelsize=9)
    else:
        ax.set_xticklabels([])

# ─── Save ────────────────────────────────────────────────────────────────────
output_path = os.path.join(project_root, "mttd", "mttd_chart.png")
fig.savefig(output_path, dpi=150, facecolor=bg_color, bbox_inches="tight")
plt.close()
print(f"Chart saved to: {output_path}")
print(f"Dimensions: {n_rows} rows (1 BTC + 1 Position + 1 NetVote + {n_indicators} indicators)")
