# Phase 1: Baseline Crisis-Aware Model

## Overview

Implements a Bidirectional LSTM with attention mechanism to establish baseline performance for stock price prediction during crisis and normal periods.

## Files

- **`crisis_stock_prediction.py`** - Main training script
- **`crisis_aware_bilstm_model.h5`** - Trained model (117,441 parameters)
- **`plots/`** - Visualizations (training curves, confusion matrices, sentiment analysis)

## Architecture

```
Input (60 days × 15 features)
    ↓
Bidirectional LSTM (64 units)
    ↓
Layer Normalization + Attention
    ↓
Bidirectional LSTM (32 units)
    ↓
Dense (16) + Dropout (0.3)
    ↓
Output (sigmoid)
```

## Results

| Metric | Overall | Normal | Crisis |
|--------|---------|--------|--------|
| **Accuracy** | 61.01% | 63.64% | 58.19% |
| **F1-Score** | 0.6180 | 0.6630 | 0.5580 |
| **Precision** | 0.6245 | 0.6578 | 0.5912 |
| **Recall** | 0.6116 | 0.6683 | 0.5283 |

**Crisis Penalty:** 18.8% F1-score degradation during volatility spikes

## Key Finding

**Hypothesis Rejected:** Sentiment correlation with returns **decreases by 41%** during crises:
- Normal periods: 0.0664
- Crisis periods: 0.0392

This contradicts the assumption that sentiment serves as a leading indicator during market stress.

## Usage

```bash
python crisis_stock_prediction.py
```

**Expected Runtime:** ~5 minutes (CPU)  
**Output:** Model saved to `crisis_aware_bilstm_model.h5`, plots saved to `plots/`

## Visualizations

- `training_history.png` - Loss and accuracy curves over 24 epochs
- `confusion_matrix_overall.png` - Overall test set performance
- `confusion_matrix_crisis_periods.png` - Crisis-specific performance
- `sentiment_analysis.png` - Sentiment vs prediction correlation (crisis vs normal)
