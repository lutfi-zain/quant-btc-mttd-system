"""
Final Indicator Selection
=========================

Auto-generated from final_indicator_selection.py
Contains the final set of indicators for the MTTD ensemble.
"""

# Final selected indicators for MTTD ensemble
# All indicators have individual coherence > 50% with ISP benchmark
# Equal weighting (1/N) applied to all indicators

FINAL_INDICATORS = [
    {
        "name": "LSMA Z-Score",
        "normalized": "lsma_z_score",
        "category": "oscillator",
        "coherence_pct": 79.20,
    },
    {
        "name": "Persistent Parabolic SAR Oscillator",
        "normalized": "persistent_parabolic_sar_oscillator",
        "category": "oscillator",
        "coherence_pct": 78.96,
    },
    {
        "name": "lsma for loop | viResearch",
        "normalized": "lsma_for_loop_viresearch",
        "category": "oscillator",
        "coherence_pct": 73.33,
    },
    {
        "name": "median for loop | viResearch",
        "normalized": "median_for_loop_viresearch",
        "category": "oscillator",
        "coherence_pct": 73.01,
    },
    {
        "name": "hull for loop | viResearch",
        "normalized": "hull_for_loop_viresearch",
        "category": "oscillator",
        "coherence_pct": 71.76,
    },
    {
        "name": "dema dmi | viResearch",
        "normalized": "dema_dmi_viresearch",
        "category": "perpetual",
        "coherence_pct": 70.61,
    },
    {
        "name": "Two Pole Butterworth For Loop",
        "normalized": "two_pole_butterworth_for_loop",
        "category": "oscillator",
        "coherence_pct": 69.90,
    },
    {
        "name": "Fourier For Loop",
        "normalized": "fourier_for_loop",
        "category": "oscillator",
        "coherence_pct": 69.70,
    },
    {
        "name": "mode for loop | viResearch",
        "normalized": "mode_for_loop_viresearch",
        "category": "oscillator",
        "coherence_pct": 69.25,
    },
    {
        "name": "Median RSI SD| QuantEdgeB",
        "normalized": "median_rsi_sd_quantedgeb",
        "category": "oscillator",
        "coherence_pct": 69.10,
    },
    {
        "name": "DSMA | viResearch",
        "normalized": "dsma_viresearch",
        "category": "perpetual",
        "coherence_pct": 68.57,
    },
    {
        "name": "DEMA RSI Overlay",
        "normalized": "dema_rsi_overlay",
        "category": "perpetual",
        "coherence_pct": 68.50,
    },
    {
        "name": "Inverted SD Dema RSI | viResearch",
        "normalized": "inverted_sd_dema_rsi_viresearch",
        "category": "perpetual",
        "coherence_pct": 68.46,
    },
    {
        "name": "Adaptive Gaussian MA For Loop",
        "normalized": "adaptive_gaussian_ma_for_loop",
        "category": "oscillator",
        "coherence_pct": 68.40,
    },
    {
        "name": "Double Src SMA Standard Deviation | viResearch",
        "normalized": "double_src_sma_standard_deviation_viresearch",
        "category": "perpetual",
        "coherence_pct": 68.14,
    },
    {
        "name": "DEMA SMA Standard Deviation | viResearch",
        "normalized": "dema_sma_standard_deviation_viresearch",
        "category": "perpetual",
        "coherence_pct": 68.06,
    },
    {
        "name": "Median Standard Deviation | viResearch",
        "normalized": "median_standard_deviation_viresearch",
        "category": "perpetual",
        "coherence_pct": 67.71,
    },
    {
        "name": "HILO Interpolation | QuantEdgeB",
        "normalized": "hilo_interpolation_quantedgeb",
        "category": "perpetual",
        "coherence_pct": 67.13,
    },
    {
        "name": "Z SMMA | QuantEdgeB",
        "normalized": "z_smma_quantedgeb",
        "category": "oscillator",
        "coherence_pct": 64.52,
    },
    {
        "name": "FDI Adaptive Oscillator Suite",
        "normalized": "fdi_adaptive_oscillator_suite",
        "category": "oscillator",
        "coherence_pct": 63.05,
    },
]

FINAL_INDICATOR_COUNT = 20

# Normalized names for quick lookup
FINAL_INDICATOR_NAMES = [ind["normalized"] for ind in FINAL_INDICATORS]
