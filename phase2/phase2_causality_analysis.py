"""
Phase 2: Causality & Feature Importance Analysis
================================================

This script tests:
1. Granger Causality: Does sentiment lead price movements?
2. Lagged Cross-Correlation: What's the optimal sentiment lag?
3. Attention Weight Visualization: What does the model focus on?
4. SHAP Analysis: Which features matter most?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from scipy.stats import pearsonr
from scipy.signal import correlate
import tensorflow as tf
from tensorflow import keras
import shap
import warnings
warnings.filterwarnings('ignore')

# Set styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class CausalityAnalyzer:
    """
    Test causal relationships between sentiment and price movements.
    """
    
    def __init__(self, data_path):
        """
        Initialize analyzer with dataset.
        
        Args:
            data_path: Path to CSV file
        """
        self.data = pd.read_csv(data_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index('Date', inplace=True)
        
        # Calculate returns for stationarity
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data.dropna(inplace=True)
        
        print(f"Loaded {len(self.data)} days of data")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
    
    def test_granger_causality(self, max_lag=10, crisis_only=False):
        """
        Test if sentiment Granger-causes price returns.
        
        Granger Causality tests if past values of X help predict Y
        beyond what past values of Y already provide.
        
        Args:
            max_lag: Maximum lag to test (days)
            crisis_only: Test only on crisis periods
        """
        print("\n" + "="*80)
        print("GRANGER CAUSALITY TEST: Does Sentiment Lead Price Movements?")
        print("="*80)
        
        # Prepare data
        if crisis_only:
            # Identify crisis periods (simple volatility threshold for now)
            rolling_vol = self.data['Returns'].rolling(window=20).std()
            crisis_threshold = rolling_vol.quantile(0.75)
            crisis_mask = rolling_vol > crisis_threshold
            test_data = self.data[crisis_mask][['Returns', 'Sentiment']].dropna()
            print(f"\nTesting on CRISIS periods only: {len(test_data)} days")
        else:
            test_data = self.data[['Returns', 'Sentiment']].dropna()
            print(f"\nTesting on ALL periods: {len(test_data)} days")
        
        if len(test_data) < max_lag * 2:
            print("Insufficient data for Granger causality test")
            return None
        
        print(f"\nNull Hypothesis: Sentiment does NOT Granger-cause Returns")
        print(f"Testing lags 1 to {max_lag} days\n")
        
        # Run Granger causality test
        try:
            results = grangercausalitytests(
                test_data[['Returns', 'Sentiment']], 
                maxlag=max_lag, 
                verbose=False
            )
            
            # Extract p-values
            p_values = {}
            f_stats = {}
            
            for lag in range(1, max_lag + 1):
                # Get F-test results (index 0 is the test, index 1 is the p-value)
                ssr_ftest = results[lag][0]['ssr_ftest']
                f_stats[lag] = ssr_ftest[0]
                p_values[lag] = ssr_ftest[1]
            
            # Display results
            print(f"{'Lag':<6} {'F-Statistic':<15} {'P-Value':<12} {'Significant?':<15}")
            print("-" * 55)
            
            for lag in range(1, max_lag + 1):
                significant = "✓ YES" if p_values[lag] < 0.05 else "✗ NO"
                print(f"{lag:<6} {f_stats[lag]:<15.4f} {p_values[lag]:<12.6f} {significant:<15}")
            
            # Summary
            significant_lags = [lag for lag, p in p_values.items() if p < 0.05]
            
            print("\n" + "="*80)
            if significant_lags:
                print(f"✓ SENTIMENT GRANGER-CAUSES RETURNS at lags: {significant_lags}")
                print(f"  → Sentiment from {min(significant_lags)}-{max(significant_lags)} days ago")
                print(f"    helps predict today's returns (p < 0.05)")
            else:
                print("✗ NO SIGNIFICANT GRANGER CAUSALITY FOUND")
                print("  → Sentiment does NOT help predict returns beyond past returns")
            print("="*80)
            
            # Visualize
            self._plot_granger_results(f_stats, p_values, crisis_only)
            
            return {'f_stats': f_stats, 'p_values': p_values, 'significant_lags': significant_lags}
            
        except Exception as e:
            print(f"Error in Granger causality test: {e}")
            return None
    
    def _plot_granger_results(self, f_stats, p_values, crisis_only):
        """Plot Granger causality test results."""
        lags = list(f_stats.keys())
        f_values = list(f_stats.values())
        p_vals = list(p_values.values())
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # F-statistics
        axes[0].bar(lags, f_values, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Lag (Days)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('F-Statistic', fontsize=12, fontweight='bold')
        axes[0].set_title('Granger Causality F-Statistics', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(lags)
        
        # P-values
        axes[1].bar(lags, p_vals, alpha=0.7, color='coral', edgecolor='black')
        axes[1].axhline(y=0.05, color='red', linestyle='--', linewidth=2, 
                       label='Significance Threshold (p=0.05)')
        axes[1].set_xlabel('Lag (Days)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('P-Value', fontsize=12, fontweight='bold')
        axes[1].set_title('Granger Causality P-Values (Lower = More Significant)', 
                         fontsize=14, fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(lags)
        axes[1].set_ylim(0, max(p_vals) * 1.1)
        
        period_label = "Crisis" if crisis_only else "All"
        plt.suptitle(f'Granger Causality: Sentiment → Returns ({period_label} Periods)', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filename = f'plots/granger_causality_{"crisis" if crisis_only else "all"}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Granger causality plot saved: {filename}")
        plt.show()
    
    def lagged_cross_correlation(self, max_lag=15):
        """
        Compute cross-correlation between sentiment and returns at various lags.
        
        Positive lag k: Sentiment at t correlates with Returns at t+k (sentiment leads)
        Negative lag k: Returns at t correlates with Sentiment at t+k (returns lead)
        
        Args:
            max_lag: Maximum lag to test in both directions
        """
        print("\n" + "="*80)
        print("LAGGED CROSS-CORRELATION: When Does Sentiment Correlate with Returns?")
        print("="*80)
        
        sentiment = self.data['Sentiment'].values
        returns = self.data['Returns'].values
        
        # Compute cross-correlation
        cross_corr = correlate(
            (sentiment - sentiment.mean()) / sentiment.std(),
            (returns - returns.mean()) / returns.std(),
            mode='same'
        ) / len(sentiment)
        
        # Extract relevant lags
        center = len(cross_corr) // 2
        lags = np.arange(-max_lag, max_lag + 1)
        lag_corr = cross_corr[center - max_lag:center + max_lag + 1]
        
        # Find peak correlation
        max_corr_idx = np.argmax(np.abs(lag_corr))
        best_lag = lags[max_corr_idx]
        best_corr = lag_corr[max_corr_idx]
        
        print(f"\nCross-Correlation Results:")
        print(f"  Peak correlation: {best_corr:.4f} at lag {best_lag} days")
        
        if best_lag > 0:
            print(f"  → Sentiment LEADS returns by {best_lag} days")
            print(f"    (Sentiment today correlates with returns {best_lag} days later)")
        elif best_lag < 0:
            print(f"  → Returns LEAD sentiment by {abs(best_lag)} days")
            print(f"    (Returns today correlate with sentiment {abs(best_lag)} days later)")
            print(f"    ⚠️ This suggests sentiment is REACTIVE, not predictive!")
        else:
            print(f"  → Sentiment and returns are contemporaneous (same-day correlation)")
        
        # Visualize
        self._plot_cross_correlation(lags, lag_corr, best_lag, best_corr)
        
        return {'lags': lags, 'correlations': lag_corr, 'best_lag': best_lag, 'best_corr': best_corr}
    
    def _plot_cross_correlation(self, lags, correlations, best_lag, best_corr):
        """Plot lagged cross-correlation."""
        plt.figure(figsize=(14, 6))
        
        # Bar plot
        colors = ['red' if lag < 0 else 'green' if lag > 0 else 'blue' for lag in lags]
        plt.bar(lags, correlations, alpha=0.6, color=colors, edgecolor='black', linewidth=0.5)
        
        # Highlight peak
        plt.bar(best_lag, best_corr, alpha=0.9, color='gold', edgecolor='red', linewidth=2,
               label=f'Peak: {best_corr:.4f} at lag {best_lag}')
        
        # Reference lines
        plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
        plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Labels
        plt.xlabel('Lag (Days)', fontsize=13, fontweight='bold')
        plt.ylabel('Cross-Correlation', fontsize=13, fontweight='bold')
        plt.title('Lagged Cross-Correlation: Sentiment vs Returns', 
                 fontsize=15, fontweight='bold', pad=20)
        
        # Add interpretation text
        if best_lag > 0:
            interpretation = f"Sentiment LEADS returns by {best_lag} days"
            color = 'green'
        elif best_lag < 0:
            interpretation = f"Returns LEAD sentiment by {abs(best_lag)} days (Sentiment is REACTIVE)"
            color = 'red'
        else:
            interpretation = "Contemporaneous (same-day) correlation"
            color = 'blue'
        
        plt.text(0.5, 0.95, interpretation, transform=plt.gca().transAxes,
                fontsize=12, fontweight='bold', color=color,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=11)
        plt.tight_layout()
        
        plt.savefig('plots/lagged_cross_correlation.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Cross-correlation plot saved: lagged_cross_correlation.png")
        plt.show()
    
    def sentiment_volatility_analysis(self):
        """
        Test if sentiment VOLATILITY (instability) is more predictive than raw sentiment.
        """
        print("\n" + "="*80)
        print("SENTIMENT VOLATILITY ANALYSIS: Is Sentiment Instability Predictive?")
        print("="*80)
        
        # Calculate sentiment volatility
        self.data['Sentiment_Vol'] = self.data['Sentiment'].rolling(window=7).std()
        
        # Calculate correlations
        corr_raw = self.data['Sentiment'].corr(self.data['Returns'])
        corr_vol = self.data['Sentiment_Vol'].dropna().corr(
            self.data.loc[self.data['Sentiment_Vol'].notna(), 'Returns']
        )
        
        print(f"\nCorrelation with Returns:")
        print(f"  Raw Sentiment:        {corr_raw:.4f}")
        print(f"  Sentiment Volatility: {corr_vol:.4f}")
        
        improvement = (abs(corr_vol) - abs(corr_raw)) / abs(corr_raw) * 100 if corr_raw != 0 else 0
        
        if abs(corr_vol) > abs(corr_raw):
            print(f"\n✓ Sentiment volatility is {improvement:.1f}% more correlated!")
            print(f"  → Sentiment INSTABILITY matters more than sentiment DIRECTION")
        else:
            print(f"\n✗ Raw sentiment is still more correlated")
        
        # Visualize
        fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Raw sentiment
        ax1_twin = axes[0].twinx()
        axes[0].plot(self.data.index, self.data['Sentiment'], 
                    label='Sentiment', color='blue', alpha=0.6, linewidth=1)
        ax1_twin.plot(self.data.index, self.data['Returns'], 
                     label='Returns', color='red', alpha=0.6, linewidth=1)
        axes[0].set_ylabel('Sentiment', fontsize=11, fontweight='bold', color='blue')
        ax1_twin.set_ylabel('Returns', fontsize=11, fontweight='bold', color='red')
        axes[0].set_title(f'Raw Sentiment vs Returns (Corr: {corr_raw:.4f})', 
                         fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Sentiment volatility
        ax2_twin = axes[1].twinx()
        axes[1].plot(self.data.index, self.data['Sentiment_Vol'], 
                    label='Sentiment Volatility', color='purple', alpha=0.6, linewidth=1)
        ax2_twin.plot(self.data.index, self.data['Returns'], 
                     label='Returns', color='red', alpha=0.6, linewidth=1)
        axes[1].set_ylabel('Sentiment Vol (7-day std)', fontsize=11, fontweight='bold', color='purple')
        ax2_twin.set_ylabel('Returns', fontsize=11, fontweight='bold', color='red')
        axes[1].set_title(f'Sentiment Volatility vs Returns (Corr: {corr_vol:.4f})', 
                         fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Date', fontsize=11, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/sentiment_volatility_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Sentiment volatility plot saved: sentiment_volatility_analysis.png")
        plt.show()
        
        return {'corr_raw': corr_raw, 'corr_vol': corr_vol, 'improvement': improvement}


class AttentionVisualizer:
    """
    Visualize what the attention mechanism is focusing on.
    """
    
    def __init__(self, model_path):
        """
        Load trained model.
        
        Args:
            model_path: Path to saved .h5 model
        """
        print("\n" + "="*80)
        print("ATTENTION VISUALIZATION: What Does the Model Look At?")
        print("="*80)
        
        self.model = keras.models.load_model(model_path)
        print(f"\n✓ Model loaded from {model_path}")
        
        # Create attention extraction model
        self._create_attention_model()
    
    def _create_attention_model(self):
        """Create a model that outputs attention weights."""
        try:
            # Find attention layer
            attention_layer = None
            for layer in self.model.layers:
                if 'attention' in layer.name.lower():
                    attention_layer = layer
                    break
            
            if attention_layer is None:
                print("⚠️ No attention layer found in model")
                self.attention_model = None
                return
            
            # Get the layer that feeds into attention
            attention_input_layer = None
            for layer in self.model.layers:
                if 'layer_norm_1' in layer.name:
                    attention_input_layer = layer
                    break
            
            if attention_input_layer:
                print(f"✓ Found attention layer: {attention_layer.name}")
                print(f"✓ Attention input: {attention_input_layer.name}")
                
                # Note: Keras Attention layer doesn't expose weights directly
                # We'll need to use the layer outputs instead
                self.attention_model = None  # Will compute attention scores differently
            else:
                self.attention_model = None
                
        except Exception as e:
            print(f"⚠️ Could not create attention extraction model: {e}")
            self.attention_model = None
    
    def visualize_attention(self, X_sample, feature_names, sample_idx=0):
        """
        Visualize attention weights for a sample.
        
        Note: Keras Attention layer doesn't expose weights directly.
        This is a placeholder for future implementation.
        
        Args:
            X_sample: Input sequences (samples, timesteps, features)
            feature_names: List of feature names
            sample_idx: Which sample to visualize
        """
        print("\n⚠️ Direct attention weight extraction not implemented")
        print("   Keras Attention layer doesn't expose weights in output")
        print("   Alternative: Use gradient-based attention or rebuild model")
        print("\n💡 Recommendation: Implement custom attention layer with weight output")


def main():
    """
    Run Phase 2 causality and feature importance analysis.
    """
    print("="*80)
    print("PHASE 2: CAUSALITY & FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    DATA_FILE = '../data/stock_market_data_large.csv'
    MODEL_FILE = '../phase1/crisis_aware_bilstm_model.h5'
    
    # =========================================================================
    # PART 1: Granger Causality Test
    # =========================================================================
    print("\n[PART 1: GRANGER CAUSALITY TEST]")
    
    analyzer = CausalityAnalyzer(DATA_FILE)
    
    # Test on all periods
    print("\n--- Testing All Periods ---")
    granger_all = analyzer.test_granger_causality(max_lag=10, crisis_only=False)
    
    # Test on crisis periods only
    print("\n\n--- Testing Crisis Periods Only ---")
    granger_crisis = analyzer.test_granger_causality(max_lag=10, crisis_only=True)
    
    # =========================================================================
    # PART 2: Lagged Cross-Correlation
    # =========================================================================
    print("\n\n[PART 2: LAGGED CROSS-CORRELATION]")
    
    cross_corr = analyzer.lagged_cross_correlation(max_lag=15)
    
    # =========================================================================
    # PART 3: Sentiment Volatility Analysis
    # =========================================================================
    print("\n\n[PART 3: SENTIMENT VOLATILITY ANALYSIS]")
    
    sent_vol = analyzer.sentiment_volatility_analysis()
    
    # =========================================================================
    # PART 4: Attention Visualization (Placeholder)
    # =========================================================================
    print("\n\n[PART 4: ATTENTION VISUALIZATION]")
    
    try:
        viz = AttentionVisualizer(MODEL_FILE)
        print("\n💡 To properly visualize attention weights:")
        print("   1. Rebuild model with custom attention layer that outputs weights")
        print("   2. Or use gradient-based attention (GradCAM for time-series)")
        print("   3. Or use integrated gradients to see feature importance over time")
    except Exception as e:
        print(f"⚠️ Could not load model for attention visualization: {e}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n\n" + "="*80)
    print("PHASE 2 ANALYSIS SUMMARY")
    print("="*80)
    
    print("\n1. GRANGER CAUSALITY:")
    if granger_all and granger_all['significant_lags']:
        print(f"   ✓ Sentiment Granger-causes returns at lags: {granger_all['significant_lags']}")
    else:
        print(f"   ✗ No significant Granger causality found (sentiment doesn't predict returns)")
    
    print("\n2. CROSS-CORRELATION:")
    if cross_corr:
        if cross_corr['best_lag'] > 0:
            print(f"   ✓ Sentiment LEADS returns by {cross_corr['best_lag']} days")
        elif cross_corr['best_lag'] < 0:
            print(f"   ✗ Returns LEAD sentiment by {abs(cross_corr['best_lag'])} days (REACTIVE)")
        else:
            print(f"   ~ Contemporaneous correlation (same-day)")
    
    print("\n3. SENTIMENT VOLATILITY:")
    if sent_vol:
        if sent_vol['improvement'] > 0:
            print(f"   ✓ Sentiment volatility is {sent_vol['improvement']:.1f}% more correlated")
            print(f"     → Use sentiment INSTABILITY as a feature")
        else:
            print(f"   ✗ Raw sentiment still more predictive")
    
    print("\n4. NEXT STEPS:")
    print("   - Review generated plots to understand causal relationships")
    print("   - If sentiment is reactive, consider removing it as a feature")
    print("   - If sentiment volatility works, integrate it into model")
    print("   - Implement proper attention visualization")
    print("   - Run SHAP analysis for feature importance")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - granger_causality_all.png")
    print("  - granger_causality_crisis.png")
    print("  - lagged_cross_correlation.png")
    print("  - sentiment_volatility_analysis.png")


if __name__ == "__main__":
    main()
