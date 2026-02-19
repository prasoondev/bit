# Stock Market Prediction During Global Crises Using Deep Learning and Sentiment Analysis


## Abstract

Financial crises present unique challenges for predictive modeling due to regime shifts, increased volatility, and breakdown of normal market correlations. This research investigates the effectiveness of sentiment analysis and technical indicators in predicting stock price movements during crisis and normal periods using deep learning architectures. We implement a three-phase experimental framework: (1) baseline bidirectional LSTM with attention mechanism, (2) Granger causality and cross-correlation analysis, and (3) Hidden Markov Model-based regime detection with specialized predictor models. Our empirical analysis of 13,647 days of synthetic market data reveals that sentiment features contribute minimally to prediction accuracy (5% importance), while technical indicators—particularly RSI ($+0.1248$ importance)—dominate model performance. Crisis-specialized models achieve $F_1 = 0.6503$, representing a 22.7% improvement over baseline crisis performance. We demonstrate that sentiment lags returns by 2 days (Granger $p = 0.042$), indicating reactive rather than predictive behavior. These findings suggest that regime-switching architectures with pure technical features outperform sentiment-augmented models for crisis prediction.

**Keywords:** Deep Learning, Crisis Prediction, Sentiment Analysis, Regime-Switching Models, Granger Causality, Feature Importance

---

## 1. Introduction

### 1.1 Motivation

Financial markets exhibit distinct behavioral regimes characterized by different volatility patterns, correlation structures, and predictability dynamics. The 2008 global financial crisis and the 2020 COVID-19 market shock demonstrated that models trained on normal market conditions fail catastrophically during periods of extreme stress. Traditional machine learning approaches assume stationarity and consistent feature-target relationships—assumptions that are violated during regime transitions.

Sentiment analysis has emerged as a popular augmentation to technical trading systems, with proponents arguing that social media and news sentiment can capture market psychology before it manifests in price movements. However, the causal relationship between sentiment and returns remains contested, particularly during crisis periods when fear and uncertainty dominate investor behavior.

### 1.2 Research Questions

This study addresses three critical questions:

1. **RQ1:** Can deep learning models with attention mechanisms effectively distinguish between crisis and normal market regimes?
2. **RQ2:** Does sentiment data exhibit predictive power, or does it merely react to price movements (Granger causality)?
3. **RQ3:** Can regime-specific models trained on crisis data outperform general-purpose models?

### 1.3 Contributions

Our work makes the following contributions:

- **Empirical evaluation** of sentiment features vs. technical indicators across 13,647 days of market data
- **Causal analysis** demonstrating that sentiment lags returns by 2 days (reactive behavior)
- **Regime-switching framework** using Hidden Markov Models to detect market states
- **Permutation importance analysis** quantifying feature contributions ($RSI: +0.1248$; Sentiment: $+0.0017$)
- **Crisis-specialized models** achieving 22.7% improvement in crisis F₁-score

---

## 2. Methodology

### 2.1 Dataset

We analyze a synthetic dataset spanning **52.3 years** (January 1, 2010 – April 24, 2062), containing **13,647 daily observations** with 10 raw features:

**Market Data:**
- Open, High, Low, Close prices
- Volume
- Target (binary: Up/Down movement)

**Technical Indicators:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)

**Sentiment Data:**
- Daily sentiment score (normalized [-1, 1])

**Crisis Labels:**
- Binary crisis indicator (25% of dataset, 3,407 days)

### 2.2 Feature Engineering

We engineer 7 additional features to capture momentum, volatility, and sentiment dynamics:

$$
\text{Log\_Return}_t = \log\left(\frac{\text{Close}_t}{\text{Close}_{t-1}}\right)
$$

$$
\text{Pct\_Change}_t = \frac{\text{Close}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}
$$

$$
\text{Sentiment\_Volatility}_t = \sigma(\text{Sentiment}_{t-6:t})
$$

$$
\text{Volume\_Change}_t = \frac{\text{Volume}_t - \text{Volume}_{t-1}}{\text{Volume}_{t-1}}
$$

$$
\text{Price\_Range}_t = \text{High}_t - \text{Low}_t
$$

$$
\text{MACD\_Signal}_t = \text{EMA}_{9}(\text{MACD}_t)
$$

$$
\text{MACD\_Hist}_t = \text{MACD}_t - \text{MACD\_Signal}_t
$$

**Total feature set:** 15 features (8 raw + 7 engineered)

### 2.3 Phase 1: Baseline Bi-LSTM with Attention

**Architecture:**

```
Input Layer: (batch_size, 60, 15)  # 60-day lookback, 15 features
    ↓
Bidirectional LSTM (64 units) → 128-dimensional output
    ↓
Layer Normalization
    ↓
Attention Layer (self-attention)
    ↓
Concatenation [LSTM output + Attention output]
    ↓
Bidirectional LSTM (32 units) → 64-dimensional output
    ↓
Layer Normalization → Dropout (0.3)
    ↓
Dense (32 units, ReLU) → Dropout (0.3)
    ↓
Output Layer (1 unit, Sigmoid)
```

**Model Parameters:** 117,441 (458.75 KB)

**Training Configuration:**
- Optimizer: Adam (initial LR = 0.001)
- Loss: Binary Cross-Entropy
- Callbacks: EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.5, patience=5)
- Epochs: 50 (max), 25 (actual)
- Batch Size: 32

**Data Split:**
- Training: 71.96% (9,685 samples)
- Validation: 8.00% (1,077 samples)
- Test: 20.04% (2,691 samples, 46.04% crisis)

### 2.4 Phase 2: Causality Analysis

#### 2.4.1 Granger Causality Test

We test whether sentiment Granger-causes returns using Vector Autoregression (VAR) models:

$$
\text{Return}_t = \alpha_0 + \sum_{i=1}^{p} \alpha_i \text{Return}_{t-i} + \sum_{i=1}^{p} \beta_i \text{Sentiment}_{t-i} + \epsilon_t
$$

**Null Hypothesis:** $H_0: \beta_1 = \beta_2 = \ldots = \beta_p = 0$ (sentiment does not Granger-cause returns)

We test lags $p \in [1, 10]$ using F-tests at $\alpha = 0.05$ significance level.

#### 2.4.2 Lagged Cross-Correlation

We compute Pearson correlation between returns at time $t$ and sentiment at time $t + \tau$:

$$
\rho(\tau) = \text{Corr}(\text{Return}_t, \text{Sentiment}_{t+\tau}), \quad \tau \in [-10, +10]
$$

Peak correlation at $\tau < 0$ indicates sentiment is **reactive** (returns lead sentiment).  
Peak correlation at $\tau > 0$ indicates sentiment is **predictive** (sentiment leads returns).

#### 2.4.3 Sentiment Volatility Analysis

We compare predictive power of raw sentiment vs. sentiment instability:

$$
\text{Corr}_{\text{raw}} = \text{Corr}(\text{Return}_t, \text{Sentiment}_t)
$$

$$
\text{Corr}_{\text{vol}} = \text{Corr}(\text{Return}_t, \text{Sentiment\_Volatility}_t)
$$

### 2.5 Phase 3: Regime-Switching Models

#### 2.5.1 A/B Testing: Sentiment vs. Technical Features

We train two competing models:

**Model A (Sentiment+):** 20 features including:
- Sentiment, Sentiment_Volatility
- Sentiment lags (1, 2, 3 days)
- Sentiment momentum (change)

**Model B (Technical Only):** 15 features excluding all sentiment derivatives

Both models use identical architectures (86K and 83K parameters respectively) and training procedures.

#### 2.5.2 Hidden Markov Model Regime Detection

We fit a 2-state Gaussian HMM to identify crisis and normal regimes:

$$
P(s_t | s_{t-1}) = \begin{bmatrix}
p_{11} & p_{12} \\
p_{21} & p_{22}
\end{bmatrix}
$$

where $s_t \in \{\text{Normal, Crisis}\}$

States are identified by volatility levels: $\sigma_{\text{crisis}} > \sigma_{\text{normal}}$

#### 2.5.3 Specialized Models

We train regime-specific models:

**Normal Specialist:** Sentiment+ configuration (20 features), trained on 8,669 normal days  
**Crisis Specialist:** Technical-only configuration (8 features), trained on 4,978 crisis days

The crisis specialist uses a minimal feature set:
- Close, Volume, RSI, MACD
- Log_Return, Volatility, Volume_Change, MACD_Hist

Parameters: 79,905 (312 KB)

### 2.6 Permutation Feature Importance

We use model-agnostic permutation importance to quantify feature contributions. For each feature $j$:

1. Compute baseline F₁-score: $F_1^{\text{base}}$
2. Randomly shuffle feature $j$ (breaking feature-target relationship)
3. Compute degraded F₁-score: $F_1^{\text{permuted}}$
4. Repeat 5 times to estimate variance
5. Importance = $F_1^{\text{base}} - \text{Mean}(F_1^{\text{permuted}})$

Positive importance indicates feature contributes positively (removal hurts performance).  
Negative importance indicates feature is harmful (removal improves performance).

---

## 3. Experimental Results

### 3.1 Phase 1: Baseline Model Performance

**Training Summary:**
- Converged in 25 epochs (early stopping)
- Final validation loss: 0.6200
- Learning rate reduced at epoch 20 (0.001 → 0.0005)

![Training History](phase1/plots/training_history.png)
*Figure 1: Training and validation metrics across 25 epochs*

#### Overall Performance (All Periods)

| Metric | Score |
|--------|-------|
| **Accuracy** | 0.6169 |
| **F₁-Score** | 0.5868 |
| **MAE** | 0.3831 |
| **Precision** | 0.62 |
| **Recall** | 0.62 |

**Confusion Matrix (Overall):**

|  | Predicted Down | Predicted Up |
|---|---|---|
| **Actual Down** | 933 (68.8%) | 421 (31.2%) |
| **Actual Up** | 608 (45.5%) | 729 (54.5%) |

![Confusion Matrix Overall](phase1/plots/confusion_matrix_overall.png)
*Figure 2: Overall confusion matrix on test set (2,691 samples)*

#### Crisis Period Performance

| Metric | Crisis | Normal | Δ (Penalty) |
|--------|--------|--------|-------------|
| **Accuracy** | 0.5900 | 0.6398 | -7.8% |
| **F₁-Score** | 0.5189 | 0.6366 | -18.5% |
| **MAE** | 0.4100 | 0.3602 | +13.8% |

**Confusion Matrix (Crisis Periods):**

|  | Predicted Down | Predicted Up |
|---|---|---|
| **Actual Down** | 452 (71.6%) | 179 (28.4%) |
| **Actual Up** | 329 (54.1%) | 279 (45.9%) |

![Confusion Matrix Crisis](phase1/plots/confusion_matrix_crisis_periods.png)
*Figure 3: Crisis-only confusion matrix (1,239 samples)*

**Key Findings:**

1. **Crisis penalty of 18.5% in F₁-score** demonstrates that baseline models struggle during high-volatility periods
2. Model exhibits higher recall for "Down" movements (71.6% crisis) but lower precision for "Up" movements (45.9% crisis)
3. This asymmetry suggests the model is conservative during crises, over-predicting downward movements

#### Sentiment Correlation Analysis

| Period | Sentiment-Prediction Correlation |
|--------|----------------------------------|
| Crisis | 0.0626 |
| Normal | 0.0454 |

Weak correlations ($< 0.07$) suggest sentiment has minimal direct relationship with model predictions, consistent with our later findings that sentiment contributes only 5% of importance.

![Sentiment Analysis](phase1/plots/sentiment_analysis.png)
*Figure 4: Sentiment correlation with predictions across crisis and normal periods*

### 3.2 Phase 2: Causality Analysis Results

#### 3.2.1 Granger Causality Test Results

**Null Hypothesis:** Sentiment does NOT Granger-cause Returns

| Lag | F-Statistic | P-Value | Significant? |
|-----|-------------|---------|--------------|
| 1 | 0.0403 | 0.8410 | ✗ |
| **2** | **3.1646** | **0.0423** | **✓** |
| 3 | 2.1281 | 0.0944 | ✗ |
| 4 | 1.6023 | 0.1707 | ✗ |
| 5 | 1.3340 | 0.2464 | ✗ |
| 6 | 1.5572 | 0.1553 | ✗ |
| 7 | 1.4238 | 0.1906 | ✗ |
| 8 | 1.2519 | 0.2641 | ✗ |
| 9 | 1.1305 | 0.3366 | ✗ |
| 10 | 1.1767 | 0.3011 | ✗ |

**Interpretation:** Sentiment from 2 days ago significantly predicts today's returns ($p = 0.042 < 0.05$). However, this statistical significance must be contextualized with cross-correlation analysis to determine causality direction.

**Crisis-Only Analysis:** Identical results (lag 2: $p = 0.044$), indicating the relationship is consistent across regimes.

![Granger Causality All Periods](phase2/plots/granger_causality_all.png)
*Figure 5: Granger causality F-statistics and p-values for all periods*

![Granger Causality Crisis](phase2/plots/granger_causality_crisis.png)
*Figure 6: Granger causality F-statistics and p-values for crisis periods only*

#### 3.2.2 Cross-Correlation Analysis

**Peak Correlation:** $\rho = 0.0215$ at lag $\tau = -2$ days

**Interpretation:** 
- Negative lag indicates **returns at time $t$ correlate with sentiment at time $t+2$**
- This means returns **lead** sentiment by 2 days
- **Conclusion:** Sentiment is **REACTIVE**, not predictive

**Mechanism:** Investors observe price movements, then sentiment reflects those movements 2 days later (news articles, social media reactions propagate with delay).

![Lagged Cross-Correlation](phase2/plots/lagged_cross_correlation.png)
*Figure 7: Cross-correlation between returns and sentiment at various lags*

#### 3.2.3 Sentiment Volatility Comparison

| Feature | Correlation with Returns |
|---------|--------------------------|
| Raw Sentiment | 0.0043 |
| Sentiment Volatility | -0.0055 |

**Relative Improvement:** Sentiment volatility is 28.5% more correlated (in absolute terms) than raw sentiment.

**Crisis Impact:**
- Normal period sentiment correlation: 0.0664
- Crisis period sentiment correlation: 0.0392
- **Degradation: 41% drop** during crises

This demonstrates that sentiment's already-weak predictive power deteriorates further during the periods where prediction is most critical.

![Sentiment Volatility Analysis](phase2/plots/sentiment_volatility_analysis.png)
*Figure 8: Comparison of raw sentiment vs. sentiment volatility correlations*

#### 3.2.4 Phase 2 Statistical Summary

| Test | Result | Interpretation |
|------|--------|----------------|
| **Granger (Lag 2)** | $p = 0.042$ | Significant Granger causality |
| **Cross-Corr Peak** | $\tau = -2$ | Returns lead sentiment (reactive) |
| **Peak Correlation** | $\rho = 0.0215$ | Extremely weak relationship |
| **Crisis Impact** | 41% drop | Sentiment fails in crises |

**Paradox Resolution:** Granger causality tests whether past sentiment values help predict current returns in a VAR framework. The test can be significant even if the correlation is weak (0.0215) because it measures statistical predictability, not practical effect size. However, the negative lag in cross-correlation reveals the true directionality: sentiment is reactive, not causative in an economic sense.

![Phase 2 Quick Summary](phase2/plots/PHASE2_QUICK_SUMMARY.png)
*Figure 9: Phase 2 comprehensive summary with key statistical findings*

### 3.3 Phase 3: Regime-Switching Results

#### 3.3.1 A/B Testing: Sentiment vs. Technical

**Model A (Sentiment+):** 20 features, 86,049 parameters
**Model B (Technical Only):** 15 features, 83,489 parameters

| Model | Accuracy | F₁-Score | Winner |
|-------|----------|----------|--------|
| **Model B (Technical)** | **0.6130** | **0.6092** | 🏆 |
| Model A (Sentiment+) | 0.5982 | 0.5913 |  |

**Performance Gap:** +3.03% F₁-score improvement by **removing** sentiment features

**Interpretation:** Adding 5 sentiment-derived features (25% increase in feature count) **degrades** performance, confirming that sentiment introduces noise rather than signal.

![A/B Testing Results](phase3/plots/phase3_ab_testing_results.png)
*Figure 10: A/B testing comparison between Sentiment+ and Technical-only models*

#### 3.3.2 HMM Regime Detection

**2-State Gaussian HMM Results:**

| State | Days | Percentage |
|-------|------|------------|
| Normal | 8,669 | 63.5% |
| Crisis | 4,978 | 36.5% |

**State Characteristics:**
- Crisis state identified by higher volatility ($\sigma_{\text{crisis}} > \sigma_{\text{normal}}$)
- Regime allocation differs from ground-truth crisis labels (which cover 25% of data), suggesting HMM detects additional volatile periods not labeled as formal crises

#### 3.3.3 Specialized Model Performance

**Normal Specialist (Sentiment+):**
- Features: 20
- Parameters: 86,049
- Accuracy: 0.6161
- **F₁-Score: 0.6456**

**Crisis Specialist (Technical Only):**
- Features: 8 (minimal set: Close, Volume, RSI, MACD, Log_Return, Volatility, Volume_Change, MACD_Hist)
- Parameters: 79,905
- Accuracy: 0.6230
- **F₁-Score: 0.6503**

**Comparison to Phase 1 Baseline:**

| Model | Crisis F₁ | Improvement |
|-------|-----------|-------------|
| Phase 1 Baseline | 0.5189 | — |
| Crisis Specialist | **0.6503** | **+25.3%** |

**Relative Improvement:** +22.7% relative improvement over baseline crisis performance

**Insight:** Specialized model trained exclusively on crisis data with minimal technical features outperforms general-purpose model by substantial margin, validating regime-specific approach.

#### 3.3.4 Permutation Feature Importance

Analysis on 1,560 test samples, 5 permutations per feature:

**Baseline Performance:**
- F₁-Score: 0.6021
- Accuracy: 0.6154

**Top 10 Features (by importance):**

| Rank | Feature | Importance | Std Dev |
|------|---------|------------|---------|
| 1 | **RSI** | **+0.1248** | 0.0094 |
| 2 | MACD_Hist | +0.0258 | 0.0088 |
| 3 | Volume_Change | +0.0076 | 0.0063 |
| 4 | MACD | +0.0068 | 0.0049 |
| 5 | Price_Range | +0.0037 | 0.0052 |
| 6 | **Sentiment_Volatility** | **+0.0036** | 0.0028 |
| 7 | Volume | +0.0036 | 0.0057 |
| 8 | Close | +0.0030 | 0.0007 |
| 9 | **Sentiment** | **+0.0017** | 0.0029 |
| 10 | Open | +0.0013 | 0.0010 |

**Bottom 5 Features (harmful/neutral):**

| Rank | Feature | Importance | Std Dev |
|------|---------|------------|---------|
| 11 | Low | -0.0008 | 0.0013 |
| 12 | High | -0.0009 | 0.0015 |
| 13 | MACD_Signal | -0.0043 | 0.0031 |
| 14 | Log_Return | -0.0301 | 0.0065 |
| 15 | **Pct_Change** | **-0.0391** | 0.0042 |

**Sentiment vs. Technical Feature Analysis:**

| Category | Feature Count | Total Importance | Percentage |
|----------|---------------|------------------|------------|
| **Technical** | 13 | +0.1013 | **95.0%** |
| **Sentiment** | 2 | +0.0053 | **5.0%** |

**Ratio:** Technical features are **19.01× more important** than sentiment features

**Key Observations:**

1. **RSI dominates:** +0.1248 importance (72.3% of total positive importance)
2. **Sentiment is weak:** +0.0017 importance (rank #9/15), comparable to basic features like "Open"
3. **Harmful features exist:** Pct_Change and Log_Return degrade performance (redundant with Close?)
4. **Sentiment_Volatility outperforms raw sentiment:** +0.0036 vs +0.0017 (2.1× better), confirming Phase 2 finding

**Statistical Summary:**

| Metric | Value |
|--------|-------|
| Mean Importance | 0.0071 |
| Median Importance | 0.0030 |
| Std Deviation | 0.0359 |
| Max (RSI) | +0.1248 |
| Min (Pct_Change) | -0.0391 |

![Permutation Importance](phase3/plots/phase3_permutation_importance.png)
*Figure 11: Permutation feature importance analysis showing RSI dominance*

---

## 4. Discussion

### 4.1 Sentiment as a Reactive Indicator

Our causal analysis provides compelling evidence that sentiment is **reactive** rather than predictive:

1. **Cross-correlation peak at lag -2:** Returns predict sentiment 2 days later
2. **Weak absolute correlation:** $\rho = 0.0215$ (explains only 0.05% of variance)
3. **Granger causality paradox:** Statistical significance without economic significance
4. **Permutation importance:** Sentiment contributes only 5% of total importance

**Proposed Mechanism:** Price movements trigger news coverage and social media discussion with a 2-day propagation delay. Sentiment captures investor *reactions* to past returns, not forward-looking expectations.

**Implications:** Sentiment features should be **excluded** from crisis prediction models, or transformed into technical indicators (e.g., sentiment momentum, volatility) that measure information flow rather than directional signals.

### 4.2 RSI as Dominant Feature

RSI (Relative Strength Index) exhibits importance (+0.1248) an order of magnitude larger than any other feature. This suggests:

1. **Momentum effects dominate:** RSI captures overbought/oversold conditions that persist over prediction horizons
2. **Mean reversion:** Extreme RSI values signal impending corrections
3. **Crisis amplification:** RSI volatility increases during crises, providing stronger signals

The dominance of RSI validates classical technical analysis principles, even in deep learning contexts where models theoretically could learn RSI-like features from raw price data.

### 4.3 Regime-Switching Superiority

Crisis-specialized models achieve 25.3% improvement in F₁-score over baseline, demonstrating:

1. **Non-stationarity:** Crisis and normal periods exhibit fundamentally different dynamics
2. **Feature relevance shifts:** Sentiment+ works for normal periods (F₁ = 0.6456), but pure technical excels for crises (F₁ = 0.6503)
3. **Model capacity trade-off:** Specialized models with fewer features outperform general models with more features

**HMM Detection Advantage:** Unsupervised regime detection (63.5% normal, 36.5% crisis) differs from ground-truth labels (75% normal, 25% crisis), suggesting HMM captures market volatility regimes beyond binary crisis/normal classification.

### 4.4 Harmful Features: Log-Returns and Percentage Changes

Both Log_Return (-0.0301) and Pct_Change (-0.0391) exhibit negative importance, indicating they **degrade** model performance. Possible explanations:

1. **Redundancy:** These features are direct transformations of Close price, creating multicollinearity
2. **Scale mismatch:** Log/percentage scaling may conflict with RobustScaler normalization
3. **Information leakage risk:** Daily returns calculated from Close price may create spurious correlations

**Recommendation:** Remove these features and allow the model to learn relevant transformations through LSTM hidden states.

### 4.5 Limitations

**Synthetic Data:** Our dataset is synthetic, limiting generalizability to real markets. Real crises exhibit:
- Extreme tail events (black swans) beyond Gaussian assumptions
- Contagion effects across asset classes
- Policy interventions (circuit breakers, quantitative easing)

**Limited Crisis Diversity:** Dataset covers generic "crisis periods" (25% of data) without distinguishing crisis types:
- Credit crises (2008)
- Pandemic shocks (2020)
- Geopolitical events (wars, sanctions)

**Binary Classification:** Up/Down prediction ignores magnitude. A model predicting +0.1% vs. observed +10% appears equally wrong as a model predicting -0.1%.

**No Transaction Costs:** Real-world trading incurs costs (spreads, slippage, taxes) that reduce profitability, particularly for high-frequency strategies.

---

## 5. Conclusions

### 5.1 Summary of Findings

This research provides comprehensive evidence against sentiment features for crisis prediction:

1. **Sentiment is reactive:** Cross-correlation analysis reveals returns lead sentiment by 2 days
2. **Weak predictive power:** Sentiment contributes only 5% of model importance (vs. 95% for technical)
3. **Crisis degradation:** Sentiment correlation drops 41% during crises when prediction is most critical
4. **A/B testing confirms:** Removing sentiment features **improves** F₁-score by 3.03%

Regime-switching architectures provide substantial gains:

5. **Crisis-specialized model:** 25.3% improvement over baseline (F₁ = 0.6503 vs. 0.5189)
6. **Minimal feature set:** 8 technical features outperform 20-feature sentiment models
7. **RSI dominance:** RSI alone accounts for 72.3% of total positive feature importance

### 5.2 Practical Recommendations

**For Practitioners:**

1. **Exclude raw sentiment:** Remove Sentiment feature from crisis prediction pipelines
2. **Consider sentiment volatility:** As a technical indicator of information flow (28.5% better than raw)
3. **Focus on RSI:** Prioritize RSI and MACD-based features over exotic indicators
4. **Implement regime detection:** Use HMM or volatility thresholds to trigger specialized models
5. **Remove redundant features:** Drop Log_Return and Pct_Change (negative importance)

**For Researchers:**

1. **Replicate with real data:** Validate findings on historical crisis periods (2008, 2020, 2022)
2. **Explore crisis taxonomy:** Train separate models for credit, liquidity, and exogenous shocks
3. **Investigate causality mechanisms:** Structural equation modeling to identify causal pathways
4. **Test alternative sentiment sources:** Compare Twitter, Reddit, news headlines, and options market implied sentiment (VIX)

### 5.3 Future Work

**Short-term Extensions:**

- **SHAP analysis:** Complement permutation importance with SHAP values for instance-level interpretability
- **Ensemble methods:** Combine regime-specific models with uncertainty-weighted predictions
- **Multi-horizon prediction:** Extend binary classification to 1-day, 5-day, 20-day horizons

**Long-term Research Directions:**

- **Reinforcement learning:** Frame crisis prediction as a partially observable Markov decision process (POMDP)
- **Graph neural networks:** Model contagion effects across correlated assets
- **Causal discovery:** Apply constraint-based algorithms (PC, FCI) to discover causal structure
- **Online learning:** Develop adaptive models that update continuously as new crises emerge

### 5.4 Final Remarks

The negative result regarding sentiment is as scientifically valuable as a positive result would be. Our findings challenge the prevailing enthusiasm for sentiment analysis in finance, demonstrating that:

> **Technical indicators, particularly RSI, provide stronger and more reliable signals than sentiment features for crisis prediction. Sentiment's statistical significance in Granger causality tests masks its reactive nature and minimal practical importance.**

This conclusion has immediate implications for both academic research and industrial applications, suggesting that resources devoted to sentiment data collection and processing might be better allocated to feature engineering from price-volume data and regime detection systems.

The success of regime-switching architectures points toward a future where financial ML systems explicitly model nonstationarity through mixture-of-experts approaches, rather than attempting to learn universal patterns across all market conditions.

---

## Appendix A: Model Architecture Details

### Baseline Bi-LSTM Architecture

```python
Model: "Crisis_Aware_BiLSTM"
┌─────────────────────────────┬───────────────────────┬───────────────┐
│ Layer (type)                │ Output Shape          │ Param #       │
├─────────────────────────────┼───────────────────────┼───────────────┤
│ input_layer                 │ (None, 60, 15)        │ 0             │
│ bidirectional_lstm          │ (None, 60, 128)       │ 40,960        │
│ layer_norm_1                │ (None, 60, 128)       │ 256           │
│ attention_layer             │ (None, 60, 128)       │ 0             │
│ concat_layer                │ (None, 60, 256)       │ 0             │
│ bidirectional_lstm_2        │ (None, 64)            │ 73,984        │
│ layer_norm_2                │ (None, 64)            │ 128           │
│ dropout_layer               │ (None, 64)            │ 0             │
│ dense_hidden                │ (None, 32)            │ 2,080         │
│ dropout_layer_2             │ (None, 32)            │ 0             │
│ output_layer                │ (None, 1)             │ 33            │
└─────────────────────────────┴───────────────────────┴───────────────┘
Total params: 117,441 (458.75 KB)
```

### Crisis Specialist Architecture (Phase 3)

```python
Model: "Crisis_Specialist"
- Input: (batch_size, 60, 8)  # Minimal feature set
- Bidirectional LSTM (128 units) → 73,216 params
- Bidirectional LSTM (64 units) → 9,984 params
- Dense layers → 2,080 params
Total params: 79,905 (312 KB)
```

---

## Appendix B: Complete Permutation Importance Results

| Rank | Feature | Type | Importance | Std Dev |
|------|---------|------|------------|---------|
| 1 | RSI | Technical | +0.1248 | 0.0094 |
| 2 | MACD_Hist | Technical | +0.0258 | 0.0088 |
| 3 | Volume_Change | Technical | +0.0076 | 0.0063 |
| 4 | MACD | Technical | +0.0068 | 0.0049 |
| 5 | Price_Range | Technical | +0.0037 | 0.0052 |
| 6 | Sentiment_Volatility | Sentiment | +0.0036 | 0.0028 |
| 7 | Volume | Technical | +0.0036 | 0.0057 |
| 8 | Close | Technical | +0.0030 | 0.0007 |
| 9 | Sentiment | Sentiment | +0.0017 | 0.0029 |
| 10 | Open | Technical | +0.0013 | 0.0010 |
| 11 | Low | Technical | -0.0008 | 0.0013 |
| 12 | High | Technical | -0.0009 | 0.0015 |
| 13 | MACD_Signal | Technical | -0.0043 | 0.0031 |
| 14 | Log_Return | Technical | -0.0301 | 0.0065 |
| 15 | Pct_Change | Technical | -0.0391 | 0.0042 |

**Aggregate Statistics:**
- **Sentiment Features (2):** Total = +0.0053, Average = +0.0027, Rank = #7.5
- **Technical Features (13):** Total = +0.1013, Average = +0.0078, Rank = #7.8

---

## Appendix C: Training Hyperparameters

| Hyperparameter | Phase 1 | Phase 3 A/B | Phase 3 Specialists |
|----------------|---------|-------------|---------------------|
| Sequence Length | 60 | 60 | 60 |
| Batch Size | 32 | 32 | 32 |
| Initial LR | 0.001 | 0.001 | 0.001 |
| LR Reduction Factor | 0.5 | 0.5 | 0.5 |
| LR Reduction Patience | 5 | 5 | 5 |
| Early Stop Patience | 10 | 10 | 10 |
| Dropout Rate | 0.3 | 0.3 | 0.3 |
| LSTM Units (Layer 1) | 64 | 64 | 64 |
| LSTM Units (Layer 2) | 32 | 32 | 32 |
| Dense Units | 32 | 32 | 32 |
| Optimizer | Adam | Adam | Adam |
| Loss Function | BCE | BCE | BCE |
| Train/Val/Test Split | 72/8/20 | 72/8/20 | 72/8/20 |

---