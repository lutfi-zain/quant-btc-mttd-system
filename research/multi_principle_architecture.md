# Multi-Principle MTTD — Architecture & Research Plan

> **Goal:** Rebuild the MTTD trading system using all 10 statistical families from
> `lz-technical-indicator-architect`, matching the sophistication of the proven
> `ichimoku_quant` benchmark (6+ families, Sharpe 1.5+, Win Rate 70%+).

---

## 1. The 10 Statistical Families — Mapping to Concrete Indicators

Each of the 10 families contributes a distinct signal or data stream. The table
below maps each family to (a) the specific indicator, (b) the source module,
(c) the output columns, and (d) the role in the layered signal flow.

| # | Family | Indicator | Source | Output Signals | Role |
|---|--------|-----------|--------|----------------|------|
| 1 | **Smoothing** | KAMA (Kaufman Adaptive MA) | `indicators/kama.py` *(new)* | `kama`, `kama_direction` (+1/-1) | **Entry baseline** — adaptive trend tracking |
| 2 | **Filtering** | Ehler SuperSmoother | `indicators/ehler_supersmoother.py` *(existing)* | `smooth`, `direction` (+1/-1) | **Input denoiser** — pre-filter for all families |
| 3 | **Regression** | LinearReg Trend Channel | `indicators/linear_reg_trend.py` *(existing)* | `lr_line`, `upper_band`, `lower_band`, `slope`, `direction` (+1/-1) | **Trend strength gate** — slope confirms momentum |
| 4 | **Spectral** | FFT Cycle Phase + MAMA | `indicators/fft_cycle_phase.py` *(new)* | `phase`, `cycle_signal`, `cycle_period` | **Entry timing** — align entries with troughs |
| 5 | **Fractal** | Efficiency Ratio + Hurst Exponent | `indicators/efficiency_ratio.py` *(existing)* + `indicators/hurst_exponent.py` *(new)* | `er`, `hurst`, `er_direction` (+1/-1 if ER > 0.25) | **Trend gate** — ER ensures trending, Hurst ensures persistent |
| 6 | **GARCH** | Volatility Cluster | `indicators/volatility_cluster.py` *(existing)* | `rolling_vol`, `vol_ratio`, `direction` (+1/-1) | **Noise gate** — don't trade during high vol |
| 7 | **Entropy** | Shannon Entropy + Permutation Entropy | `indicators/shannon_entropy.py` *(existing)* + `indicators/permutation_entropy.py` *(new)* | `entropy`, `entropy_direction` (+1/-1) | **Randomness gate** — don't trade random markets |
| 8 | **Chaos** | Phase Space Reconstruction | `indicators/phase_space_reconstruction.py` *(new)* | `psr_instability`, `psr_direction` (+1/-1) | **Crash gate / exit trigger** — instability precedes crashes |
| 9 | **Bayesian** | HMM Regime Detection | `indicators/hmm_regime.py` *(existing)* | `state`, `bull_prob`, `direction` (+1/-1) | **Regime gate** — only trade in BULL regime |
| 10 | **ML Hybrid** | Composite Scoring | *(integrated in strategy)* | Weighted combination of all family signals | **Final signal aggregation** |

### Family 1: Smoothing — KAMA (Kaufman Adaptive MA)

**Why:** KAMA adapts its smoothing parameter based on the Efficiency Ratio.
When ER is high (strong trend), KAMA responds faster. When ER is low (noise),
KAMA smooths aggressively. This gives us an adaptive trend baseline that
adjusts to market conditions automatically.

**Implementation plan:**
- Compute ER over window (default 14)
- `fastSC = 2/3`, `slowSC = 2/31`
- `sc = (ER × (fastSC - slowSC) + slowSC)²`
- `KAMA = KAMA[prev] + sc × (price - KAMA[prev])`

**Output columns:** `kama`, `kama_direction` (+1 when KAMA rising)

### Family 4: Spectral — FFT Cycle Phase

**Why:** The old MTTD system already proved that cycle phase timing works.
Combining FFT-based cycle trough identification with our other signals gives
us a timing edge. We extract:
- Dominant cycle period via FFT
- Phase position (0 to 2π)
- Cycle trough signal (+1 when near trough)

**Implementation plan:** Port from `mttd_system.py` `compute_cycle_phase()`
function into a standalone indicator module.

### Family 4: Spectral — MAMA (MESA Adaptive Moving Average)

**Why:** MAMA uses the Hilbert Transform to measure phase rate of change
and adapts alpha dynamically. It provides fast attack (quick to respond to
new trends) and slow decay (holds onto trends longer). This is the advanced
version of adaptive smoothing from the DSP lineage.

**Implementation plan:**
- Use Hilbert Transform to compute instantaneous phase and phase rate
- `alpha_fast = 0.5`, `alpha_slow = 0.05`
- `alpha = alpha_fast` when phase rate is high (fast attack)
- `alpha = alpha_slow` when phase rate is low (slow decay)

### Family 5: Fractal — Hurst Exponent

**Why:** Hurst exponent H tells us whether the market is trending (H > 0.5)
or mean-reverting (H < 0.5). We only want to take trend-following trades
when H > 0.5. Combined with ER, this gives a rigorous "is this a real trend?"
test.

**Implementation plan:** Use R/S analysis over a rolling window (default 60).
Take log(R/S) / log(n) to estimate H. This is the same approach used in
`adaptive_regime_cloud.py` from the indicator bank.

### Family 7: Entropy — Permutation Entropy

**Why:** Shannon entropy measures return distribution randomness, but
Permutation Entropy is more robust because it's based on ordinal patterns
(rank order) rather than exact values. It's less sensitive to noise and
better at detecting deterministic structure in short windows.

**Implementation plan:**
- Embed dimension m = 3, delay τ = 1
- For each window, compute ordinal patterns and their probabilities
- `PermEn = -Σ p(π) × log₂(p(π)) / log₂(m!)` (normalized 0 to 1)

### Family 8: Chaos — Phase Space Reconstruction

**Why:** Phase Space Reconstruction (via Takens' Theorem) allows us to
reconstruct the system's attractor from a single time series. High
instability in the phase space (trajectories diverging) indicates
impending crashes or regime changes.

**Implementation plan:**
- Embed dimension m = 3, delay τ determined by mutual information
- For each window, compute the L largest Lyapunov exponent proxy
- If trajectories diverge rapidly (high instability), signal is bearish

---

## 2. Layered Signal Flow Architecture

The signal flow follows a strict **multi-layer pipeline** inspired by the
`ichimoku_quant.py` architecture but expanded to 10 families:

```
                          ┌──────────────────────┐
                          │   RAW OHLCV DATA      │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  LAYER 0: DENOISE     │
                          │  Ehler SuperSmoother  │  ← Family 2 (Filtering)
                          │  (pre-filter all cols) │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │  LAYER 1: FEATURES    │
                          │  Compute all 10       │
                          │  family indicators    │
                          └──────────┬───────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
    ┌─────────▼─────────┐  ┌────────▼───────┐  ┌──────────▼────────┐
    │  ENTRY GATES       │  │  TRIGGER        │  │  EXIT GATES      │
    │  (must ALL pass)   │  │  (at least one) │  │  (any triggers)  │
    └───────────────────┘  └────────────────┘  └───────────────────┘
```

### 2.1 Layer 0: Input Denoising (Family 2)

Before any indicator computation, apply Ehler SuperSmoother to `close`, `high`,
and `low` with length = 5–10. This removes high-frequency noise common in
crypto data while preserving phase.

```python
close_smooth = ehler_supersmoother(df['close'], length=7)['smooth']
high_smooth  = ehler_supersmoother(df['high'],  length=7)['smooth']
low_smooth   = ehler_supersmoother(df['low'],   length=7)['smooth']
```

All subsequent indicator computations use the smoothed series.

### 2.2 Layer 1: Feature Computation

Compute all indicator families. Each family produces:
- A **numeric signal** (continuous value, e.g., ER from 0 to 1, entropy from 0 to max)
- A **binary direction** (+1 = bullish/tradeable, -1 = bearish/avoid)

### 2.3 Entry Logic — Multi-Layer Gate System

For an entry to trigger, **all gates must pass** AND at least one trigger
must fire.

#### Entry Gates (AND — all must pass):

| Gate # | Condition | Family | Parameter |
|--------|-----------|--------|-----------|
| G1 | `ER > er_entry` (trend must exist) | Family 5 (Fractal) | `er_entry: 0.25` |
| G2 | `Hurst > 0.5` (trend is persistent, not random walk) | Family 5 (Fractal) | `hurst_entry: 0.5` |
| G3 | `Entropy < entropy_thresh` (not random noise) | Family 7 (Entropy) | `entropy_thresh: 2.0` |
| G4 | `VolRatio < vol_thresh` (not in high volatility) | Family 6 (GARCH) | `vol_thresh: 1.2` |
| G5 | `HMM_direction == +1` (regime is BULL) | Family 9 (Bayesian) | — |
| G6 | `KAMA_direction == +1` (trend is UP) | Family 1 (Smoothing) | — |
| G7 | Price above LinearReg lower band | Family 3 (Regression) | — |

#### Entry Triggers (OR — at least one must fire):

| Trigger # | Condition | Family | Parameter |
|-----------|-----------|--------|-----------|
| T1 | `Cycle trough` (phase near 0 or 2π) | Family 4 (Spectral) | — |
| T2 | `MAMA crossover` (fast > slow line) | Family 1/4 (Smoothing/Spectral) | — |
| T3 | Price above LinearReg upper band (strong breakout) | Family 3 (Regression) | — |
| T4 | `HMM_bull_prob > 0.7` (strong regime conviction) | Family 9 (Bayesian) | — |

#### Entry Confirmation:

Once gates pass and a trigger fires:
- **Confirmation bars:** Wait `confirm_entry` bars (default 2) with sustained
  signal before entering.
- **Min hold:** After entry, `min_hold` days must pass before any exit.
  This prevents whipsaw exits.

### 2.4 Exit Logic — Multi-Layer Exit with Immunity

Exits are triggered when **any** exit signal fires, **unless** immunity applies.

#### Exit Signals (OR — any can trigger):

| Signal # | Condition | Family | Parameter |
|----------|-----------|--------|-----------|
| X1 | `ER < er_exit` (trend died) | Family 5 (Fractal) | `er_exit: 0.15` |
| X2 | `KAMA_direction == -1` (trend reversed) | Family 1 (Smoothing) | — |
| X3 | `Chikou momentum < chikou_thresh` | Family 2 (Filtering) | `chikou_thresh: -0.30` |
| X4 | `ROC < roc_gate_limit` (crash gate) | Family 2 (Filtering) | `roc_gate_limit: -0.20` |
| X5 | `PSR_instability > instability_thresh` (chaos) | Family 8 (Chaos) | `instability_thresh: 0.7` |
| X6 | `VolRatio > vol_exit` (volatility explosion) | Family 6 (GARCH) | `vol_exit: 1.5` |
| X7 | `Entropy > entropy_exit` (market turned random) | Family 7 (Entropy) | `entropy_exit: 2.5` |

#### Immunity Conditions:

If **any** immunity condition is met, no exit signal can fire (hold position):

| Immunity # | Condition | Parameter |
|------------|-----------|-----------|
| I1 | `IMO >= immunity_thresh` (strong composite trend) | `immunity_thresh: 0.50` |
| I2 | `Hurst > 0.7` (very strong trend persistence) | — |
| I3 | Price above cloud AND not crashing | — |
| I4 | `HMM_bull_prob > 0.85` (overwhelming bull regime) | — |

#### Exit Confirmation:

- **Confirmation bars:** Wait `confirm_exit` bars (default 1) with sustained
  exit signal before exiting.

### 2.5 Composite IMO (Ichimoku Momentum Oscillator)

The IMO is the primary **composite trend meter**, exactly as in `ichimoku_quant`.
It combines multiple family signals into one bounded [-1, 1] oscillator:

```
IMO = tanh(S_TK + S_Cloud + S_Future + S_Chikou) / 4
```

Where:
- `S_TK` = normalized Tenkan-Kijun spread → **Smoothing (Family 1)**
- `S_Cloud` = normalized distance to cloud → **Filtering (Family 2)**
- `S_Future` = normalized cloud spread → **Spectral (Family 4)**
- `S_Chikou` = normalized lagging span (SuperSmoothed) → **Filtering (Family 2)**

---

## 3. Indicator Implementation Details

### 3.1 New Indicators to Build

The following indicators need to be created for TODO 2:

| Module | Families | Description |
|--------|----------|-------------|
| `indicators/kama.py` | 1 (Smoothing) | Kaufman Adaptive Moving Average |
| `indicators/mama.py` | 1, 4 (Smoothing, Spectral) | MESA Adaptive Moving Average |
| `indicators/fft_cycle_phase.py` | 4 (Spectral) | FFT cycle phase and period detection |
| `indicators/hurst_exponent.py` | 5 (Fractal) | Rolling Hurst exponent via R/S analysis |
| `indicators/permutation_entropy.py` | 7 (Entropy) | Permutation Entropy (ordinal patterns) |
| `indicators/phase_space_reconstruction.py` | 8 (Chaos) | Phase Space Reconstruction + Lyapunov proxy |

### 3.2 Indicator Signatures

```python
# indicators/kama.py
def kama(df, source_col='close', period=14, fast_sc=2/3, slow_sc=2/31):
    """Returns DataFrame with columns: kama, direction (+1 rising, -1 falling)"""
    
# indicators/mama.py
def mama(df, source_col='close', fast_limit=0.5, slow_limit=0.05):
    """Returns DataFrame with columns: mama, mama_direction (+1/ -1), phase, phase_rate"""
    
# indicators/fft_cycle_phase.py
def fft_cycle_phase(df, source_col='close', lookback=40, min_period=5, max_period=None):
    """Returns DataFrame with columns: phase, cycle_signal, cycle_period, trough_signal"""
    
# indicators/hurst_exponent.py
def hurst_exponent(df, source_col='close', lookback=60):
    """Returns DataFrame with columns: hurst, direction (+1 if H > 0.5)"""
    
# indicators/permutation_entropy.py
def permutation_entropy(df, source_col='close', window=30, embed_dim=3, delay=1):
    """Returns DataFrame with columns: perm_en, direction (+1 if low entropy)"""
    
# indicators/phase_space_reconstruction.py
def phase_space_reconstruction(df, source_col='close', window=100, embed_dim=3, delay=3):
    """Returns DataFrame with columns: lyapunov_proxy, instability, direction (+1 stable)"""
```

### 3.3 Existing Indicators to Reuse (Minor Adaptations)

| Module | File | Changes Needed |
|--------|------|----------------|
| Efficiency Ratio | `indicators/efficiency_ratio.py` | None — already exports `er` and `direction` |
| Ehler SuperSmoother | `indicators/ehler_supersmoother.py` | None — already exports `smooth` and `direction` |
| Linear Reg Trend | `indicators/linear_reg_trend.py` | None — already exports bands, slope, direction |
| Volatility Cluster | `indicators/volatility_cluster.py` | None — already exports `vol_ratio` and `direction` |
| Shannon Entropy | `indicators/shannon_entropy.py` | None — already exports `entropy` and `direction` |
| HMM Regime | `indicators/hmm_regime.py` | May need performance tuning (sliding window refit) |

---

## 4. Signal Aggregation — Multi-Principle Scoring

### 4.1 Raw Composite Score

For each bar, compute a composite score from all families:

```python
# Weighted sum of all family direction signals
score = (
    w1 * kama_direction      +   # Family 1: Smoothing
    w2 * super_direction     +   # Family 2: Filtering
    w3 * lr_direction        +   # Family 3: Regression
    w4 * cycle_signal        +   # Family 4: Spectral
    w5 * er_direction        +   # Family 5: Fractal (ER)
    w6 * hurst_direction     +   # Family 5: Fractal (Hurst)
    w7 * vol_direction       +   # Family 6: GARCH
    w8 * entropy_direction   +   # Family 7: Entropy
    w9 * psr_direction       +   # Family 8: Chaos
    w10 * hmm_direction      +   # Family 9: Bayesian
) / sum(weights)
```

Default weights (all equal = 1.0 each, for 11 signals including ER+Hurst
as separate Family 5 signals):

```python
weights = {
    'kama': 1.0, 'supersmoother': 1.0, 'linearreg': 1.0, 'cycle': 1.5,
    'er': 1.5, 'hurst': 1.0, 'volatility': 1.0, 'entropy': 1.0,
    'psr': 0.5, 'hmm': 1.5
}
```

Rationale:
- **Cycle** gets higher weight because timing is critical
- **ER** gets higher weight because it's the primary trend/strength measure
- **HMM** gets higher weight as a regime classifier
- **PSR** gets lower weight because chaos theory evidence is controversial

### 4.2 Entry Decision

```
ENTRY_ORDER = score > 0.5 AND ALL_GATES_PASS
```

Where `ALL_GATES_PASS` means:
1. `er > er_entry` (0.25)
2. `hurst > 0.5`
3. `entropy < entropy_thresh` (2.0)
4. `vol_ratio < vol_thresh` (1.2)
5. `hmm_direction == 1`
6. `kama_direction == 1`
7. Price above LinearReg lower band

### 4.3 Exit Decision

```
EXIT_ORDER = ANY_EXIT_TRIGGER AND NOT IMMUNE
```

---

## 5. Ichimoku Quant Integration (Optional Enhancement)

To ensure we match or exceed the `ichimoku_quant` benchmark, we can optionally
include the full Ichimoku machinery as a Family-1/Family-2/Family-4 composite
module. This gives us:

- `tenkan_sen / kijun_sen` (Smoothing)
- `senkou_span_a / senkou_span_b` (Filtering)
- `chikou_span` (Spectral, lagging)
- Cloud visualization for immunity detection

The `generate_ichimoku_features` function from `ichimoku_quant.py` can be
reused directly and its IMO becomes an additional composite signal in our
scoring system.

---

## 6. Grid Search Parameter Space

The following parameter space will be covered in TODO 4 (Grid Search):

### 6.1 Entry Parameters (8 parameters)

| Parameter | Type | Values | Rationale |
|-----------|------|--------|-----------|
| `min_hold` | int | [10, 20, 30, 45, 60] | Short to medium hold periods |
| `confirm_entry` | int | [1, 2, 3] | Number of confirmation bars before entry |
| `er_entry` | float | [0.15, 0.20, 0.25, 0.30] | Efficiency Ratio minimum for entry |
| `entropy_thresh` | float | [1.8, 2.0, 2.27, 2.5] | Maximum entropy for tradeable state |
| `hurst_entry` | float | [0.45, 0.50, 0.55] | Minimum Hurst for trending market |
| `vol_entry` | float | [1.0, 1.1, 1.2, 1.5] | Maximum vol ratio for entry |
| `t_entry` | float | [0.3, 0.4, 0.5, 0.6] | IMO threshold multiplier (× IMO_Std) |
| `score_entry` | float | [0.3, 0.4, 0.5, 0.6] | Composite score minimum for entry |

### 6.2 Exit Parameters (7 parameters)

| Parameter | Type | Values | Rationale |
|-----------|------|--------|-----------|
| `confirm_exit` | int | [1, 2] | Confirmation bars before exit |
| `er_exit` | float | [0.10, 0.15, 0.20] | ER minimum to stay in trade |
| `chikou_thresh` | float | [-0.4, -0.3, -0.2, -0.1] | Chikou momentum drop threshold |
| `roc_gate_limit` | float | [-0.25, -0.20, -0.15] | Crash gate (30-day ROC) |
| `vol_exit` | float | [1.3, 1.5, 2.0] | Max vol ratio before exit |
| `instability_thresh` | float | [0.6, 0.7, 0.8] | PSR instability threshold |
| `entropy_exit` | float | [2.0, 2.27, 2.5] | Entropy max before exit |

### 6.3 Immunity Parameters (2 parameters)

| Parameter | Type | Values | Rationale |
|-----------|------|--------|-----------|
| `immunity_thresh` | float | [0.3, 0.4, 0.5, 0.6] | IMO level for strong immunity |
| `hurst_immune` | float | [0.6, 0.65, 0.7] | Hurst above which immunity kicks in |

### 6.4 Total Combinations

**Estimated: ~1,080 to ~2,160 configurations** depending on chaining.

If we grid the most critical parameters first (branch 1: min_hold × er_entry ×
entropy_thresh × t_entry × immunity_thresh) = 5 × 4 × 4 × 4 × 4 = 1,280,
then optimize secondary parameters in a second pass.

### 6.5 Optimization Strategy

**Two-phase grid search:**

- **Phase 1 (Coarse):** Vary min_hold (5), er_entry (4), entropy_thresh (4),
  t_entry (4), immunity_thresh (4) = 1,280 runs. Fix other params at defaults.
- **Phase 2 (Fine):** Take top 3 configurations from Phase 1, vary secondary
  params (confirm_entry, confirm_exit, er_exit, vol_exit, chikou_thresh, etc.)
  = 3 × 432 ≈ 1,296 runs.

Total: ~2,500 runs.

---

## 7. Validation Protocol

From AGENTS.md learnings, we know:
- Walk-forward validation is MANDATORY
- Robustness over peak performance

### 7.1 Walk-Forward Schedule

| Period | Start | End | Role |
|--------|-------|-----|------|
| Train (in-sample) | 2018-01-01 | 2023-12-31 | Parameter optimization |
| Validation | 2024-01-01 | 2024-12-31 | Early stopping / parameter selection |
| Test (out-of-sample) | 2025-01-01 | 2026-06-22 | Final evaluation |

**Degradation check:**

```
degradation = (holdout_metric - train_metric) / abs(train_metric) × 100

Accept if degradation < 50% for Sharpe and CAGR.
```

### 7.2 Anti-Overfitting Rules

1. **No cherry-picking** — The best config must work on both train AND holdout
2. **Degradation < 50%** — Otherwise discard the config
3. **Minimum 20 trades in holdout** — Otherwise not statistically meaningful
4. **Transaction costs always included** — 0.1% round-trip
5. **Final config must be simple** — Prefer fewer params that generalize

---

## 8. Implementation Roadmap

```
TODO 1 (this document)  → Research & Architecture Plan      [DONE]
                        ↓
TODO 2                  → Build Signal Generators
                          (Create 6 new indicator modules)
                        ↓
TODO 3                  → Build Strategy + Backtest
                          (Create multi_principle_strategy.py)
                        ↓
TODO 4                  → Grid Search + Optimization
                          (Create multi_principle_grid.py)
                        ↓
TODO 5                  → Compare Results
                          (Create compare_multi_principle.py)
```

---

## 9. Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| Win Rate | > 65% | Minimum acceptable for trend-following |
| Sharpe | > 1.0 | Risk-adjusted outperformance vs. buy-and-hold |
| CAGR | > 40% | Meaningful alpha over BTC buy-and-hold |
| Trades | 20–35 | Enough for statistical significance (fewer = overfit) |
| Degradation | < 50% | Generalization requirement on holdout data |

### Expected Comparison (Final Output)

| System | Trades | WinRate | Sharpe | CAGR | Degradation |
|--------|--------|---------|--------|------|-------------|
| ISP (benchmark) | 17 | 100% | 2.36 | 114.7% | N/A |
| ichimoku_quant | 13–27 | 70%+ | 1.5+ | — | — |
| NEW Multi-Principle MTTD | 20–35 | >65% | >1.0 | >40% | <50% |
| OLD MTTD (Keltner+Regime) | 27 | 66.7% | 0.96 | 36.4% | -136.6% |

---

## Appendix A: Indicator Family Dependency Graph

```
                    Raw OHLCV
                        │
              ┌─────────▼─────────┐
              │   SuperSmoother   │  ← Family 2: Input denoising
              │   (length=7)      │
              └─────────┬─────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
    │ KAMA    │   │ Linear  │   │ FFT     │   ← Family 1, 3, 4
    │         │   │ Reg     │   │ Cycle   │
    └────┬────┘   └────┬────┘   └────┬────┘
         │              │              │
    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
    │ ER      │   │ Hurst   │   │ MAMA    │   ← Family 5, 5, 1/4
    │ (Frac)  │   │ (Frac)  │   │ (Spec)  │
    └────┬────┘   └────┬────┘   └────┬────┘
         │              │              │
    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
    │ Vol     │   │ Entropy │   │ HMM     │   ← Family 6, 7, 9
    │ Cluster │   │(Shannon)│   │ Regime  │
    └────┬────┘   └────┬────┘   └────┬────┘
         │              │              │
    ┌────▼────┐   ┌────▼────┐   ┌────▼────┐
    │ Perm    │   │ Phase   │   │ Ichimoku│   ← Family 7, 8, 1/2/4
    │ Entropy │   │ Space   │   │ Cloud   │
    └─────────┘   └─────────┘   └─────────┘
```

## Appendix B: Existing Code References

### Key Files to Study Before Implementation

| File | What It Teaches |
|------|----------------|
| `ichimoku_quant.py` | Complete 6-family signal architecture, layered entry/exit, immunity logic, confirmation bars, ATR normalization |
| `mttd_system.py` | Current system structure, data loading, metric computation, min-hold application |
| `indicators/*.py` | Pattern for indicator modules (return DataFrame with direction column) |
| `indicators/efficiency_ratio.py` | Clean indicator pattern with `__main__` test block |
| `indicators/ehler_supersmoother.py` | Recursive filter implementation pattern |
| `indicators/hmm_regime.py` | Complex indicator with sliding window, stateful computation |

### Reusable Function Signatures

From `indicators_helper.py` (bank):
```python
def sma(source, length)     → Simple Moving Average
def ema(source, length)     → Exponential Moving Average
def wma(source, length)     → Weighted Moving Average
def hma(source, length)     → Hull Moving Average
def dema(source, length)    → Double EMA
def tr(high, low, close)    → True Range
def atr(high, low, close, length) → Average True Range
def stdev(source, length)   → Standard Deviation
def linreg(source, length)  → Linear Regression
```

From `ichimoku_quant.py`:
```python
def compute_atr(df, window)              → Average True Range
def ehler_supersmoother(series, length)  → 2-pole SuperSmoother filter
def shannon_entropy(series, window, bins) → Rolling Shannon Entropy
def generate_ichimoku_features(df, ...)  → Full Ichimoku feature set
def generate_ichimoku_signals(df, ...)   → Full signal logic with layers
```

---

*End of Architecture Document*
