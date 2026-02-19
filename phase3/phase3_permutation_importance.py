#!/usr/bin/env python3
"""
Phase 3: Permutation Feature Importance Analysis

Alternative to SHAP - uses permutation importance which is model-agnostic
and works with any architecture including attention layers.

Author: Prasoon
Date: February 19, 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, accuracy_score
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PHASE 3: PERMUTATION FEATURE IMPORTANCE ANALYSIS")
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


def prepare_sequences(data, feature_cols, seq_length=60, n_samples=2000):
    """Create sequences for analysis"""
    print(f"📊 Creating sequences ({n_samples} samples for analysis)...")
    
    # Normalize features
    scaler = RobustScaler()
    scaled_data = data[feature_cols].copy()
    scaled_data[feature_cols] = scaler.fit_transform(scaled_data[feature_cols])
    
    X, y = [], []
    
    start_idx = len(data) - n_samples - seq_length
    for i in range(start_idx, len(data)):
        if i < seq_length:
            continue
        X.append(scaled_data.iloc[i-seq_length:i].values)
        y.append(data['Target'].iloc[i])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✓ Sequences created: {X.shape}\n")
    
    return X, y, scaler


def permutation_importance(model, X, y, feature_names, n_repeats=5):
    """
    Calculate permutation importance for each feature.
    
    Method: Shuffle each feature and measure drop in performance.
    Higher drop = more important feature.
    """
    print("🔬 Calculating permutation importance...")
    print(f"   - Running {n_repeats} permutations per feature")
    print(f"   - Evaluating on {len(X):,} samples")
    print("-" * 60)
    
    # Baseline performance
    y_pred_baseline = (model.predict(X, verbose=0) > 0.5).astype(int).flatten()
    baseline_f1 = f1_score(y, y_pred_baseline)
    baseline_acc = accuracy_score(y, y_pred_baseline)
    
    print(f"✓ Baseline F1-Score: {baseline_f1:.4f}")
    print(f"✓ Baseline Accuracy: {baseline_acc:.4f}\n")
    
    importances = []
    
    for feat_idx, feat_name in enumerate(feature_names):
        print(f"  [{feat_idx+1}/{len(feature_names)}] Testing {feat_name}...", end="")
        
        feat_importance = []
        
        for repeat in range(n_repeats):
            # Create copy and shuffle this feature across all timesteps
            X_permuted = X.copy()
            for t in range(X.shape[1]):  # For each timestep
                np.random.shuffle(X_permuted[:, t, feat_idx])
            
            # Evaluate with permuted feature
            y_pred_permuted = (model.predict(X_permuted, verbose=0) > 0.5).astype(int).flatten()
            permuted_f1 = f1_score(y, y_pred_permuted)
            
            # Importance = drop in F1-score
            importance = baseline_f1 - permuted_f1
            feat_importance.append(importance)
        
        mean_importance = np.mean(feat_importance)
        std_importance = np.std(feat_importance)
        importances.append({
            'feature': feat_name,
            'importance': mean_importance,
            'std': std_importance
        })
        
        print(f" {mean_importance:+.4f} ± {std_importance:.4f}")
    
    print()
    return pd.DataFrame(importances).sort_values('importance', ascending=False)


def plot_importance_results(importance_df, title="Permutation Feature Importance"):
    """Visualize importance results"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Top 15 features
    top_15 = importance_df.head(15)
    colors = ['darkred' if 'Sentiment' in f else 'steelblue' 
              for f in top_15['feature']]
    
    ax1.barh(range(15), top_15['importance'], xerr=top_15['std'], 
             color=colors, alpha=0.8, capsize=3)
    ax1.set_yticks(range(15))
    ax1.set_yticklabels(top_15['feature'])
    ax1.set_xlabel('Importance (Drop in F1-Score)', fontsize=12, fontweight='bold')
    ax1.set_title('Top 15 Most Important Features', fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (imp, std) in enumerate(zip(top_15['importance'], top_15['std'])):
        ax1.text(imp, i, f' {imp:.4f}', va='center', fontsize=9)
    
    # 2. Bottom 10 features (least important)
    bottom_10 = importance_df.tail(10)
    colors_bottom = ['darkred' if 'Sentiment' in f else 'steelblue' 
                     for f in bottom_10['feature']]
    
    ax2.barh(range(10), bottom_10['importance'], xerr=bottom_10['std'],
             color=colors_bottom, alpha=0.8, capsize=3)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(bottom_10['feature'])
    ax2.set_xlabel('Importance (Drop in F1-Score)', fontsize=12, fontweight='bold')
    ax2.set_title('Bottom 10 Least Important Features', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, axis='x')
    ax2.invert_yaxis()
    
    # 3. Sentiment vs Technical comparison
    sentiment_features = importance_df[importance_df['feature'].str.contains('Sentiment')]
    technical_features = importance_df[~importance_df['feature'].str.contains('Sentiment')]
    
    sentiment_total = sentiment_features['importance'].sum()
    technical_total = technical_features['importance'].sum()
    
    categories = ['Sentiment\nFeatures', 'Technical\nFeatures']
    totals = [sentiment_total, technical_total]
    colors_bar = ['darkred', 'steelblue']
    
    ax3.bar(categories, totals, color=colors_bar, alpha=0.8, width=0.5, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Total Importance', fontsize=12, fontweight='bold')
    ax3.set_title('Sentiment vs Technical Importance', fontsize=13, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')
    
    # Add percentages
    total = sentiment_total + technical_total
    for i, (cat, val) in enumerate(zip(categories, totals)):
        pct = (val / total) * 100 if total > 0 else 0
        ax3.text(i, val, f'{pct:.1f}%\n({val:.4f})', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # 4. Importance distribution
    ax4.hist(importance_df['importance'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axvline(importance_df['importance'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {importance_df['importance'].mean():.4f}")
    ax4.axvline(0, color='black', linestyle='-', linewidth=1)
    ax4.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Distribution of Feature Importances', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def generate_report(importance_df):
    """Generate comprehensive text report"""
    print("\n" + "=" * 80)
    print("PERMUTATION IMPORTANCE ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    # Overall ranking
    print("🏆 TOP 10 MOST IMPORTANT FEATURES:")
    print("-" * 80)
    for i, row in importance_df.head(10).iterrows():
        marker = "📈" if 'Sentiment' in row['feature'] else "🔧"
        print(f"{i+1:2d}. {marker} {row['feature']:<25} {row['importance']:+.6f} ± {row['std']:.6f}")
    print()
    
    print("⬇️  BOTTOM 5 LEAST IMPORTANT FEATURES:")
    print("-" * 80)
    bottom_5 = importance_df.tail(5)
    for i, row in bottom_5.iterrows():
        marker = "📈" if 'Sentiment' in row['feature'] else "🔧"
        rank = len(importance_df) - len(importance_df) + list(importance_df.index).index(i) + 1
        print(f"{rank:2d}. {marker} {row['feature']:<25} {row['importance']:+.6f} ± {row['std']:.6f}")
    print()
    
    # Sentiment analysis
    sentiment_features = importance_df[importance_df['feature'].str.contains('Sentiment')]
    technical_features = importance_df[~importance_df['feature'].str.contains('Sentiment')]
    
    sentiment_total = sentiment_features['importance'].sum()
    technical_total = technical_features['importance'].sum()
    total_importance = sentiment_total + technical_total
    
    print("📊 SENTIMENT FEATURE ANALYSIS:")
    print("-" * 80)
    print(f"Number of sentiment features: {len(sentiment_features)}")
    print(f"Combined importance: {sentiment_total:.6f}")
    print(f"Average importance per feature: {sentiment_total/len(sentiment_features):.6f}")
    print()
    
    print("Individual sentiment features (sorted by importance):")
    for i, (idx, row) in enumerate(sentiment_features.iterrows(), 1):
        rank = list(importance_df.index).index(idx) + 1
        print(f"  {i}. {row['feature']:<25} Rank: #{rank:2d}  Importance: {row['importance']:+.6f}")
    print()
    
    print("⚖️  SENTIMENT VS TECHNICAL:")
    print("-" * 80)
    print(f"Sentiment features:  {len(sentiment_features):2d} features  {sentiment_total:+.6f} ({(sentiment_total/total_importance)*100 if total_importance > 0 else 0:.1f}%)")
    print(f"Technical features:  {len(technical_features):2d} features  {technical_total:+.6f} ({(technical_total/total_importance)*100 if total_importance > 0 else 0:.1f}%)")
    if sentiment_total > 0:
        print(f"Ratio (Tech/Sent):   {technical_total/sentiment_total:.2f}x")
    print()
    
    # Statistical summary
    print("📈 STATISTICAL SUMMARY:")
    print("-" * 80)
    print(f"Mean importance: {importance_df['importance'].mean():.6f}")
    print(f"Median importance: {importance_df['importance'].median():.6f}")
    print(f"Std deviation: {importance_df['importance'].std():.6f}")
    print(f"Max importance: {importance_df['importance'].max():.6f} ({importance_df.iloc[0]['feature']})")
    print(f"Min importance: {importance_df['importance'].min():.6f} ({importance_df.iloc[-1]['feature']})")
    print()
    
    # Verdict
    print("💡 VERDICT:")
    print("-" * 80)
    
    sentiment_pct = (sentiment_total / total_importance * 100) if total_importance > 0 else 0
    
    # Calculate average rank for sentiment features
    sentiment_ranks = [list(importance_df.index).index(idx) + 1 for idx in sentiment_features.index]
    sentiment_avg_rank = np.mean(sentiment_ranks) if len(sentiment_ranks) > 0 else 999
    
    if sentiment_pct < 15:
        verdict = "❌ SENTIMENT FEATURES ARE WEAK"
        rec = "   RECOMMENDATION: Remove all sentiment features"
        explanation = f"   - Sentiment contributes only {sentiment_pct:.1f}% of total importance"
        explanation2 = f"   - Average rank: #{sentiment_avg_rank:.0f} out of {len(importance_df)}"
    elif sentiment_pct < 30:
        verdict = "⚠️  SENTIMENT FEATURES ARE MARGINAL"
        rec = "   RECOMMENDATION: Keep only Sentiment_Volatility and Sentiment_Lag_2"
        explanation = f"   - Sentiment contributes {sentiment_pct:.1f}% of importance"
        explanation2 = f"   - Some sentiment features useful, others are noise"
    else:
        verdict = "✅ SENTIMENT FEATURES ARE USEFUL"
        rec = "   RECOMMENDATION: Keep all sentiment features"
        explanation = f"   - Sentiment contributes {sentiment_pct:.1f}% of importance"
        explanation2 = f"   - Multiple sentiment features in top 10"
    
    print(verdict)
    print(explanation)
    print(explanation2)
    print(rec)
    print()
    
    # Comparison to Phase 2 findings
    print("🔗 CONNECTION TO PHASE 2 FINDINGS:")
    print("-" * 80)
    print("Phase 2 showed sentiment LAGS returns by 2 days (reactive, not predictive).")
    print("This permutation analysis quantifies EXACTLY how much each feature matters.")
    print()
    
    if 'Sentiment_Lag_2' in importance_df['feature'].values:
        lag2_rank = list(importance_df['feature']).index('Sentiment_Lag_2') + 1
        lag2_imp = importance_df[importance_df['feature'] == 'Sentiment_Lag_2']['importance'].values[0]
        print(f"→ Sentiment_Lag_2 (Granger significant): Rank #{lag2_rank}, Importance {lag2_imp:+.6f}")
    
    if 'Sentiment_Volatility' in importance_df['feature'].values:
        vol_rank = list(importance_df['feature']).index('Sentiment_Volatility') + 1
        vol_imp = importance_df[importance_df['feature'] == 'Sentiment_Volatility']['importance'].values[0]
        print(f"→ Sentiment_Volatility (+68% corr): Rank #{vol_rank}, Importance {vol_imp:+.6f}")
    
    if 'Sentiment' in importance_df['feature'].values:
        sent_rank = list(importance_df['feature']).index('Sentiment') + 1
        sent_imp = importance_df[importance_df['feature'] == 'Sentiment']['importance'].values[0]
        print(f"→ Raw Sentiment (baseline): Rank #{sent_rank}, Importance {sent_imp:+.6f}")
    print()


def main():
    """Main analysis pipeline"""
    
    # 1. Load data
    data = load_and_prepare_data()
    
    # 2. Define Phase 1 features (matching the trained model)
    phase1_features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI', 'MACD', 'Sentiment',
        'Log_Return', 'Pct_Change', 'Sentiment_Volatility',
        'Volume_Change', 'Price_Range', 'MACD_Signal', 'MACD_Hist'
    ]
    
    print(f"Using {len(phase1_features)} features from Phase 1 model\n")
    
    # 3. Prepare test data
    X, y, scaler = prepare_sequences(data, phase1_features, seq_length=60, n_samples=1500)
    
    # 4. Load model
    print("📁 Loading trained model...")
    model = tf.keras.models.load_model('../phase1/crisis_aware_bilstm_model.h5')
    print(f"✓ Model loaded: {model.count_params():,} parameters\n")
    
    # 5. Calculate permutation importance
    importance_df = permutation_importance(
        model, X, y, phase1_features, n_repeats=5
    )
    
    # 6. Plot results
    fig = plot_importance_results(
        importance_df,
        "Permutation Feature Importance: Phase 1 Crisis-Aware Bi-LSTM"
    )
    plt.savefig('plots/phase3_permutation_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: phase3_permutation_importance.png\n")
    
    # 7. Generate report
    generate_report(importance_df)
    
    # 8. Save results to CSV
    importance_df.to_csv('phase3_feature_importance.csv', index=False)
    print("✓ Saved: phase3_feature_importance.csv")
    
    print("\n" + "=" * 80)
    print("PERMUTATION IMPORTANCE ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
