# Phase 3: Regime-Switching Solution

## Overview

Implements a regime-switching architecture using Hidden Markov Models for crisis detection and specialized models optimized for each market regime. Validates findings through A/B testing and feature importance analysis.

## Files

- **`phase3_regime_switching.py`** - HMM regime detection + specialist models
- **`phase3_permutation_importance.py`** - Model-agnostic feature ranking
- **`phase3_shap_analysis.py`** - SHAP analysis (failed, archived)
- **`phase3_feature_importance.csv`** - Detailed feature rankings with statistics
- **`plots/`** - A/B testing results and feature importance visualizations

## Methodology

### 1. A/B Testing
**Model A:** 20 features (with sentiment + derivatives)  
**Model B:** 15 features (technical indicators only)

Trains identical Bi-LSTM architectures to isolate sentiment's impact.

### 2. Hidden Markov Model (HMM)
- **Input:** Volatility + Volume
- **States:** Normal (81.45%) vs Crisis (18.55%)
- **Method:** Gaussian HMM with full covariance

### 3. Regime-Specific Models
- **Normal Specialist:** 20 features (any reasonable set works well)
- **Crisis Specialist:** 8 features (technical-only, optimized for high volatility)

### 4. Permutation Importance
Shuffles each feature 5 times, measures F1-score drop:
```
Importance(feature) = baseline_F1 - permuted_F1
```

## Key Results

### A/B Testing

| Model | Features | Params | Accuracy | F1-Score | Winner |
|-------|----------|--------|----------|----------|--------|
| A (Sentiment+) | 20 | 98,433 | 59.71% | 0.5784 | — |
| B (Technical) | 15 | 92,573 | **61.33%** | **0.6020** | ✓ |

**Technical-only wins by +4.1% F1**

### Regime-Switching Performance

| Model | Features | F1-Score | Improvement |
|-------|----------|----------|-------------|
| Phase 1 General | 15 | 0.558 (crisis) | Baseline |
| Normal Specialist | 20 | 0.6244 | +11.9% |
| **Crisis Specialist** | **8** | **0.6849** | **+22.7%** |

### Feature Importance Rankings

| Rank | Feature | Importance | Verdict |
|------|---------|------------|---------|
| 1 | **RSI** | **+0.1126** | CRITICAL |
| 2 | MACD_Hist | +0.0328 | Important |
| 3 | Pct_Change | +0.0209 | Important |
| 4 | Log_Return | +0.0092 | Useful |
| 5 | MACD | +0.0046 | Useful |
| ... | ... | ... | ... |
| 14 | **Sentiment** | **-0.0049** | HARMFUL |
| 15 | Sentiment_Vol | -0.0029 | Harmful |

**Category Comparison:**
- Technical features (13): +105.3% total importance
- Sentiment features (2): **-5.3%** total importance

**Ratio:** Technical is **19.9× more important** than sentiment

## Crisis Specialist Configuration

**Optimal 8 Features:**
1. RSI (momentum indicator)
2. MACD_Hist (trend strength)
3. Pct_Change (returns)
4. Log_Return (log returns)
5. MACD (trend)
6. Volume (market participation)
7. RSI_Change (momentum derivative)
8. Volatility (regime indicator)

**Architecture:** Bi-LSTM (64→32 units), 79,905 parameters (31.6% smaller than baseline)

## Key Findings

### 1. Sentiment is Actively Harmful
- Permutation analysis: -0.0049 importance (negative)
- A/B test: Removing improves F1 by +4.1%
- Contribution: -5.3% vs +105.3% for technical features

### 2. RSI Dominates
- 3.4× more important than second-ranked feature
- Accounts for 76% of all positive importance
- Bounded [0-100] range stabilizes neural network training

### 3. Crisis Specialization Works
- +22.7% F1 improvement over baseline
- Separate models needed: crisis periods are fundamentally different, not just "harder"
- Smaller model (8 features) outperforms larger (15 features)

### 4. Simpler is Better
- Fewer features → better generalization
- Technical-only → less noise
- Momentum indicators >> complex derivatives

## Usage

### A/B Testing + Regime-Switching
```bash
python phase3_regime_switching.py
```
**Expected Runtime:** ~15 minutes  
**Output:** Models trained, plots saved

### Feature Importance
```bash
python phase3_permutation_importance.py
```
**Expected Runtime:** ~10 minutes (requires Phase 1 model)  
**Output:** `phase3_feature_importance.csv`, plots saved

## Visualizations

- `phase3_ab_testing_results.png` - Model A vs B comparison (4-panel)
- `phase3_permutation_importance.png` - Feature rankings with sentiment vs technical breakdown

## Practical Recommendations

1. **Remove all sentiment features** from production systems
2. **Implement HMM regime detection** using volatility + volume
3. **Deploy crisis specialist** during high-volatility periods (F1 = 0.6849)
4. **Prioritize RSI derivatives** in feature engineering
5. **Expected overall improvement:** +3.81% F1 vs Phase 1 baseline
