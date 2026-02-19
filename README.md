# Crisis-Aware Stock Market Prediction

**A deep learning study investigating sentiment's role in predicting stock movements during market crises.**

## Overview

This project analyzes whether financial sentiment serves as a leading indicator during crisis periods using Bi-LSTM neural networks. The study spans three phases: baseline modeling, causality analysis, and regime-switching optimization.

**Key Finding:** Sentiment lags returns by 2 days and contributes negative importance (-5.3%), contrary to conventional assumptions. A regime-switching approach achieves 22.7% improvement in crisis prediction.

## Results Summary

| Phase | Method | Crisis F1-Score | Key Finding |
|-------|--------|----------------|-------------|
| **Phase 1** | Bi-LSTM + Attention | 0.558 | Sentiment correlation drops 41% during crises |
| **Phase 2** | Causality Analysis | — | Sentiment lags returns by 2 days (p=0.042) |
| **Phase 3** | Regime-Switching | **0.6849** | +22.7% improvement, RSI dominates |

## Repository Structure

```
.
├── data/
│   └── stock_market_data_large.csv    # 13,647 trading days
│
├── phase1/                             # Baseline Model
│   ├── crisis_stock_prediction.py      # Bi-LSTM with attention (117K params)
│   ├── crisis_aware_bilstm_model.h5    # Trained model
│   └── plots/                          # Training curves, confusion matrices
│
├── phase2/                             # Causality Analysis
│   ├── phase2_causality_analysis.py    # Granger tests, cross-correlation
│   ├── phase2_summary_viz.py           # Summary visualizations
│   └── plots/                          # Causality plots, lag analysis
│
├── phase3/                             # Regime-Switching
│   ├── phase3_regime_switching.py      # HMM + specialist models
│   ├── phase3_permutation_importance.py # Feature importance analysis
│   ├── phase3_shap_analysis.py         # SHAP analysis (archived)
│   ├── phase3_feature_importance.csv   # Detailed rankings
│   └── plots/                          # A/B testing, feature importance
│
├── requirements.txt
└── README.md
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Experiments

**Phase 1: Baseline Model**
```bash
python phase1/crisis_stock_prediction.py
```
Output: F1 = 0.6180 (overall), 0.558 (crisis)

**Phase 2: Causality Analysis**
```bash
python phase2/phase2_causality_analysis.py
```
Output: 2-day sentiment lag, Granger p=0.042

**Phase 3: Regime-Switching**
```bash
python phase3/phase3_regime_switching.py
python phase3/phase3_permutation_importance.py
```
Output: F1 = 0.6849 (+22.7%), sentiment importance = -0.0049

## Methodology

### Phase 1: Baseline
- **Architecture:** Bidirectional LSTM with attention mechanism
- **Dataset:** 13,647 days (2010-2062), 15 features (OHLCV, RSI, MACD, sentiment)
- **Results:** 61% accuracy, but 18.8% crisis penalty discovered

### Phase 2: Causality
- **Tests:** Granger causality (lag 1-5) + lagged cross-correlation
- **Discovery:** Sentiment at T-2 "Granger-causes" returns, but cross-correlation shows returns lead sentiment
- **Interpretation:** Market reacts instantly (T=0) → journalists write (T+1) → sentiment aggregated (T+2)

### Phase 3: Optimization
- **Innovation:** Hidden Markov Model for regime detection → separate specialist models
- **A/B Test:** Technical-only beats sentiment+ by +4.1% F1
- **Feature Importance:** RSI (+0.1126) >> Sentiment (-0.0049)
- **Crisis Specialist:** 8-feature model achieves F1 = 0.6849

## Key Findings

1. **Sentiment Paradox:** Correlation decreases 41% during crises (0.0664 → 0.0392)
2. **Temporal Lag:** Sentiment lags returns by 2 days due to information propagation delay
3. **Negative Utility:** Sentiment contributes -5.3% importance, actively harming predictions
4. **RSI Dominance:** RSI is 3.4× more important than any other feature
5. **Regime Specialization:** Crisis periods require fundamentally different models

## Practical Implications

- Remove sentiment features from production trading systems
- Use HMM-based regime detection (volatility + volume)
- Deploy crisis specialist during high-volatility periods
- Prioritize RSI and momentum indicators in feature engineering

## Technical Stack

- **Python:** 3.13.12
- **TensorFlow:** 2.20.0
- **Key Libraries:** pandas, numpy, statsmodels, hmmlearn, scikit-learn

## Citation

```bibtex
@misc{prasoon2026sentiment,
  title={Crisis-Aware Stock Market Prediction using Regime-Switching Deep Learning},
  author={Prasoon},
  year={2026}
}
```

---

**Last Updated:** February 19, 2026
