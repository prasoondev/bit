# Phase 2: Causality Analysis

## Overview

Investigates the directional relationship between sentiment and returns using Granger causality tests and cross-correlation analysis to explain why sentiment fails during crises.

## Files

- **`phase2_causality_analysis.py`** - Granger tests, cross-correlation, sentiment volatility analysis
- **`phase2_summary_viz.py`** - Summary visualization generator
- **`plots/`** - Causality plots and temporal lag analysis

## Methodology

### 1. Granger Causality Test
Tests whether past sentiment values improve forecasting of current returns using F-tests on VAR models (lags 1-5).

### 2. Cross-Correlation Analysis
Computes correlation between Returns(t) and Sentiment(t+τ) for τ ∈ [-10, +10] days to identify temporal relationships.

### 3. Sentiment Volatility
Tests if sentiment instability (7-day rolling std) is more predictive than raw sentiment direction.

## Key Results

### Granger Causality (All Periods, n=13,646)

| Lag | F-Statistic | p-value | Significant |
|-----|-------------|---------|-------------|
| 1 | 0.0403 | 0.841 | No |
| **2** | **3.1646** | **0.042** | **Yes** |
| 3 | 2.1281 | 0.094 | No |
| 4 | 1.8943 | 0.173 | No |
| 5 | 1.6234 | 0.246 | No |

### Cross-Correlation

**Peak correlation:** 0.0215 at lag **-2** (negative lag means Returns lead Sentiment)

### Interpretation

```
Timeline:
T=0: Economic shock → Prices adjust instantly (efficient market)
T+1: Journalists investigate, write articles
T+2: Articles published → Sentiment aggregated
```

**Conclusion:** Sentiment is **reactive**, not predictive. It reflects information already priced into the market 2 days earlier.

### Sentiment Volatility

- Raw Sentiment correlation: 0.0043
- Sentiment Volatility correlation: -0.0055 (28.5% stronger in absolute terms)
- Both remain extremely weak predictors

## Key Finding

**The Sentiment Paradox Explained:**

Granger "causality" (sentiment at T-2 → returns at T) is a statistical artifact. The true causal chain is:

1. Economic shock drives returns (T=0)
2. Same shock eventually manifests in sentiment (T+2)
3. This creates spurious predictive relationship when looking backward

Cross-correlation confirms the actual direction: **Returns → Sentiment** (not vice versa)

## Usage

```bash
python phase2_causality_analysis.py
```

**Expected Runtime:** ~3 minutes  
**Output:** Plots saved to `plots/`

## Visualizations

- `granger_causality_all.png` - F-statistics across lags (all periods)
- `granger_causality_crisis.png` - F-statistics across lags (crisis only)
- `lagged_cross_correlation.png` - Correlation vs lag (-10 to +10 days)
- `sentiment_volatility_analysis.png` - Raw sentiment vs volatility comparison
- `PHASE2_QUICK_SUMMARY.png` - Visual summary of key findings
