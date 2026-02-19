#!/usr/bin/env python3
"""
Phase 3: SHAP Feature Importance Analysis

Quantify exactly how much each feature contributes to predictions.
Answer the key question: Are sentiment features truly useful or just noise?

Author: Prasoon
Date: February 19, 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
import shap
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 3: SHAP FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)
print()


def load_and_prepare_data():
    """Load data and create Phase 3 features"""
    print("📂 Loading and preparing data...")
    
    data = pd.read_csv('../data/stock_market_data_large.csv')
    
    # Create sentiment features
    data['Sentiment_Volatility'] = data['Sentiment'].rolling(window=7).std()
    for lag in [1, 2, 3]:
        data[f'Sentiment_Lag_{lag}'] = data['Sentiment'].shift(lag)
    data['Sentiment_Change'] = data['Sentiment'].diff()
    
    # Create technical features
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    data['Pct_Change'] = data['Close'].pct_change()
    data['Volume_Change'] = data['Volume'].pct_change()
    data['RSI_Change'] = data['RSI'].diff()
    data['MACD_Signal'] = data['MACD'].rolling(window=9).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    data['Price_Range'] = data['High'] - data['Low']
    data['Volatility'] = data['Log_Return'].rolling(window=20).std()
    
    # Fill NaN
    data = data.ffill().bfill()
    
    print(f"✓ Data prepared: {len(data):,} rows\n")
    
    return data


def create_feature_sets():
    """Define feature sets"""
    
    # Full feature set (with sentiment)
    features_with_sentiment = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'Log_Return', 'Pct_Change', 'Price_Range', 'Volatility',
        'Sentiment_Volatility',
        'Sentiment_Lag_1',
        'Sentiment_Lag_2',
        'Sentiment_Lag_3',
        'Sentiment_Change',
        'Volume_Change',
        'RSI_Change'
    ]
    
    # Technical only (no sentiment)
    features_technical = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'Log_Return', 'Pct_Change', 'Price_Range', 'Volatility',
        'Volume_Change',
        'RSI_Change'
    ]
    
    return features_with_sentiment, features_technical


def prepare_sequences(data, feature_cols, seq_length=60, n_samples=1000):
    """Create sequences for SHAP analysis (limited samples)"""
    print(f"📊 Creating sequences ({n_samples} samples for SHAP)...")
    
    # Normalize features
    scaler = RobustScaler()
    scaled_data = data[feature_cols].copy()
    scaled_data[feature_cols] = scaler.fit_transform(scaled_data[feature_cols])
    
    X, y = [], []
    
    for i in range(seq_length, min(len(data), seq_length + n_samples + 500)):
        X.append(scaled_data.iloc[i-seq_length:i].values)
        y.append(data['Target'].iloc[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Take subset for SHAP
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    X_shap = X[indices]
    y_shap = y[indices]
    
    print(f"✓ Sequences created: {X_shap.shape}\n")
    
    return X_shap, y_shap, feature_cols


def analyze_model_shap(model_path, X_shap, feature_names, model_name="Model"):
    """Run SHAP analysis on a trained model"""
    print(f"🔬 Running SHAP analysis on {model_name}...")
    print("-" * 60)
    
    # Load model
    model = tf.keras.models.load_model(model_path)
    print(f"✓ Model loaded: {model.count_params():,} parameters")
    
    # Create background dataset (smaller for speed)
    background = X_shap[:100]
    
    print(f"✓ Creating SHAP explainer (this may take 2-3 minutes)...")
    
    # Use DeepExplainer for deep learning models
    explainer = shap.DeepExplainer(model, background)
    
    print(f"✓ Computing SHAP values for {len(X_shap)} samples...")
    shap_values = explainer.shap_values(X_shap[:200])  # Limit for speed
    
    # SHAP values shape: (n_samples, seq_length, n_features)
    # Average over sequence dimension to get feature importance
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    shap_values_mean = np.abs(shap_values).mean(axis=1)  # Average over time steps
    
    print(f"✓ SHAP values computed: {shap_values_mean.shape}\n")
    
    return shap_values_mean, explainer


def plot_shap_summary(shap_values, feature_names, title="SHAP Feature Importance"):
    """Create SHAP summary plots"""
    
    # Calculate mean absolute SHAP value for each feature
    feature_importance = np.abs(shap_values).mean(axis=0)
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = feature_importance[sorted_idx]
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Bar plot of feature importance
    colors = ['darkred' if 'Sentiment' in f else 'steelblue' for f in sorted_features[:15]]
    
    ax1.barh(range(15), sorted_importance[:15], color=colors, alpha=0.8)
    ax1.set_yticks(range(15))
    ax1.set_yticklabels(sorted_features[:15])
    ax1.set_xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
    ax1.set_title('Top 15 Most Important Features', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(sorted_importance[:15]):
        ax1.text(v, i, f' {v:.4f}', va='center', fontsize=9)
    
    # 2. Sentiment vs Technical comparison
    sentiment_features = [f for f in feature_names if 'Sentiment' in f]
    technical_features = [f for f in feature_names if 'Sentiment' not in f]
    
    sentiment_importance = np.array([feature_importance[feature_names.index(f)] 
                                     for f in sentiment_features if f in feature_names])
    technical_importance = np.array([feature_importance[feature_names.index(f)] 
                                     for f in technical_features if f in feature_names])
    
    sentiment_total = sentiment_importance.sum()
    technical_total = technical_importance.sum()
    
    categories = ['Sentiment\nFeatures', 'Technical\nFeatures']
    totals = [sentiment_total, technical_total]
    colors_pie = ['darkred', 'steelblue']
    
    ax2.bar(categories, totals, color=colors_pie, alpha=0.8, width=0.5)
    ax2.set_ylabel('Total SHAP Importance', fontsize=12, fontweight='bold')
    ax2.set_title('Sentiment vs Technical Importance', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # Add percentages
    total = sentiment_total + technical_total
    for i, (cat, val) in enumerate(zip(categories, totals)):
        pct = (val / total) * 100
        ax2.text(i, val, f'{pct:.1f}%\n({val:.4f})', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return fig, sorted_features[:15], sorted_importance[:15]


def generate_shap_report(sorted_features, sorted_importance, feature_names):
    """Generate text report"""
    print("\n" + "=" * 80)
    print("SHAP ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    # Overall ranking
    print("🏆 TOP 15 MOST IMPORTANT FEATURES:")
    print("-" * 60)
    for i, (feat, imp) in enumerate(zip(sorted_features[:15], sorted_importance[:15]), 1):
        marker = "📈" if 'Sentiment' in feat else "🔧"
        print(f"{i:2d}. {marker} {feat:<25} {imp:.6f}")
    print()
    
    # Sentiment analysis
    sentiment_features = [f for f in feature_names if 'Sentiment' in f]
    sentiment_importance = np.array([sorted_importance[sorted_features.index(f)] 
                                     for f in sentiment_features if f in sorted_features])
    
    if len(sentiment_importance) > 0:
        sentiment_total = sentiment_importance.sum()
        sentiment_rank_best = sorted_features.index(sentiment_features[0]) + 1 if sentiment_features[0] in sorted_features else 999
        
        print("📊 SENTIMENT FEATURE ANALYSIS:")
        print("-" * 60)
        print(f"Total sentiment features: {len(sentiment_features)}")
        print(f"Combined importance: {sentiment_total:.6f}")
        print(f"Best sentiment feature rank: #{sentiment_rank_best}")
        print(f"Best sentiment feature: {sentiment_features[0]}")
        print()
        
        # Individual sentiment features
        print("Sentiment feature breakdown:")
        for i, feat in enumerate(sentiment_features, 1):
            if feat in sorted_features:
                idx = sorted_features.index(feat)
                imp = sorted_importance[idx]
                print(f"  {i}. {feat:<25} Rank: #{idx+1:2d}  Importance: {imp:.6f}")
        print()
    
    # Technical vs Sentiment
    technical_features = [f for f in feature_names if 'Sentiment' not in f]
    technical_importance = np.array([sorted_importance[sorted_features.index(f)] 
                                     for f in technical_features if f in sorted_features])
    technical_total = technical_importance.sum()
    
    total = sentiment_total + technical_total
    
    print("⚖️ SENTIMENT VS TECHNICAL:")
    print("-" * 60)
    print(f"Sentiment total:  {sentiment_total:.6f} ({(sentiment_total/total)*100:.1f}%)")
    print(f"Technical total:  {technical_total:.6f} ({(technical_total/total)*100:.1f}%)")
    print(f"Ratio (Tech/Sent): {technical_total/sentiment_total:.2f}x")
    print()
    
    # Verdict
    print("💡 VERDICT:")
    print("-" * 60)
    if sentiment_total / total < 0.20:
        print("❌ SENTIMENT FEATURES ARE WEAK")
        print("   - Contribute <20% of model importance")
        print("   - RECOMMENDATION: Remove sentiment, use technical only")
    elif sentiment_total / total < 0.35:
        print("⚠️ SENTIMENT FEATURES ARE MARGINAL")
        print("   - Contribute 20-35% of importance")
        print("   - RECOMMENDATION: Keep only Sentiment_Volatility and Lag_2")
    else:
        print("✓ SENTIMENT FEATURES ARE USEFUL")
        print("   - Contribute >35% of importance")
        print("   - RECOMMENDATION: Keep sentiment features")
    print()


def main():
    """Main SHAP analysis pipeline"""
    
    # 1. Load data
    data = load_and_prepare_data()
    
    # 2. Define feature sets
    features_with_sentiment, features_technical = create_feature_sets()
    
    # 3. Try to analyze the best model from A/B test
    # Check which models exist
    import os
    
    if os.path.exists('crisis_aware_bilstm_model.h5'):
        print("📁 Found Phase 1 model: crisis_aware_bilstm_model.h5")
        print("   (Using this for SHAP analysis)\n")
        
        # Prepare data with Phase 1 features (simpler)
        phase1_features = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'RSI', 'MACD', 'Sentiment',
            'Log_Return', 'Pct_Change', 'Sentiment_Volatility',
            'Volume_Change', 'Price_Range', 'MACD_Signal', 'MACD_Hist'
        ]
        
        X_shap, y_shap, feature_names = prepare_sequences(
            data, phase1_features, seq_length=60, n_samples=500
        )
        
        # Run SHAP
        shap_values, explainer = analyze_model_shap(
            '../phase1/crisis_aware_bilstm_model.h5',
            X_shap,
            feature_names,
            "Phase 1 Crisis-Aware Bi-LSTM"
        )
        
        # Plot and report
        fig, sorted_features, sorted_importance = plot_shap_summary(
            shap_values,
            feature_names,
            "SHAP Feature Importance: Phase 1 Model"
        )
        plt.savefig('plots/phase3_shap_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: phase3_shap_analysis.png\n")
        
        generate_shap_report(sorted_features, sorted_importance, feature_names)
        
    else:
        print("❌ No trained models found!")
        print("   Please run crisis_stock_prediction.py or phase3_regime_switching.py first")
    
    print("\n" + "=" * 80)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
