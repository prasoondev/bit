#!/usr/bin/env python3
"""
Phase 3: Regime-Switching Crisis Prediction with Smart Sentiment Usage

Key Innovations:
1. Sentiment Volatility instead of raw sentiment (68% higher correlation)
2. Lag features (T-1, T-2, T-3) to capture reactive patterns
3. HMM-based regime detection (Normal vs Crisis)
4. Separate specialist models for each regime
5. A/B testing framework to validate improvements

Author: Prasoon
Date: February 19, 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional, 
    Concatenate, LayerNormalization, MultiHeadAttention
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 80)
print("PHASE 3: REGIME-SWITCHING CRISIS PREDICTION")
print("=" * 80)
print()


class AdvancedFeatureEngineer:
    """
    Phase 3 Feature Engineering:
    - Sentiment volatility (7-day rolling std)
    - Lag features (T-1, T-2, T-3)
    - Removes raw sentiment score
    """
    
    def __init__(self, data):
        self.data = data.copy()
        
    def create_sentiment_features(self):
        """Replace raw sentiment with volatility + lags"""
        print("📊 Phase 3 Feature Engineering:")
        print("-" * 40)
        
        # 1. Sentiment Volatility (7-day rolling std)
        self.data['Sentiment_Volatility'] = (
            self.data['Sentiment']
            .rolling(window=7, min_periods=1)
            .std()
            .fillna(0)
        )
        
        # 2. Lag Features (T-1, T-2, T-3)
        for lag in [1, 2, 3]:
            self.data[f'Sentiment_Lag_{lag}'] = self.data['Sentiment'].shift(lag)
        
        # Fill NaN lags with forward fill
        for lag in [1, 2, 3]:
            self.data[f'Sentiment_Lag_{lag}'] = self.data[f'Sentiment_Lag_{lag}'].ffill().bfill()
        
        # 3. Sentiment Change Rate (momentum)
        self.data['Sentiment_Change'] = self.data['Sentiment'].diff().fillna(0)
        
        print(f"✓ Created Sentiment_Volatility (7-day std)")
        print(f"✓ Created Sentiment_Lag_1, Lag_2, Lag_3")
        print(f"✓ Created Sentiment_Change (momentum)")
        print()
        
        return self.data
    
    def create_technical_features(self):
        """Enhanced technical indicators"""
        print("🔧 Enhanced Technical Features:")
        print("-" * 40)
        
        # Log returns and price changes
        self.data['Log_Return'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['Pct_Change'] = self.data['Close'].pct_change()
        
        # Volume momentum
        self.data['Volume_Change'] = self.data['Volume'].pct_change()
        
        # RSI momentum
        self.data['RSI_Change'] = self.data['RSI'].diff()
        
        # MACD signal and histogram
        self.data['MACD_Signal'] = self.data['MACD'].rolling(window=9).mean()
        self.data['MACD_Hist'] = self.data['MACD'] - self.data['MACD_Signal']
        
        # Price range (volatility proxy)
        self.data['Price_Range'] = self.data['High'] - self.data['Low']
        
        # Rolling volatility (20-day)
        self.data['Volatility'] = self.data['Log_Return'].rolling(window=20).std()
        
        # Fill NaN values
        self.data = self.data.ffill().bfill()
        
        print(f"✓ Created Log_Return, Pct_Change")
        print(f"✓ Created Volume_Change (volume momentum)")
        print(f"✓ Created RSI_Change (RSI momentum)")
        print(f"✓ Created MACD_Signal, MACD_Hist")
        print(f"✓ Created Price_Range, Volatility")
        print()
        
        return self.data
    
    def prepare_features(self):
        """Full preprocessing pipeline"""
        self.create_sentiment_features()
        self.create_technical_features()
        
        # Define feature sets for different models
        
        # Model A: With Sentiment Volatility + Lags (Phase 3 approach)
        self.features_with_sentiment = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'Log_Return', 'Pct_Change', 'Price_Range', 'Volatility',
            'Sentiment_Volatility',  # NEW: Replaces raw sentiment
            'Sentiment_Lag_1',       # NEW: T-1 lag
            'Sentiment_Lag_2',       # NEW: T-2 lag (Granger significant)
            'Sentiment_Lag_3',       # NEW: T-3 lag
            'Sentiment_Change',      # NEW: Momentum
            'Volume_Change',         # NEW: Volume momentum
            'RSI_Change'             # NEW: RSI momentum
        ]
        
        # Model B: Pure Technical (No sentiment at all)
        self.features_no_sentiment = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'Log_Return', 'Pct_Change', 'Price_Range', 'Volatility',
            'Volume_Change',
            'RSI_Change'
        ]
        
        # Model C: Crisis-Specific (Focus on momentum)
        self.features_crisis = [
            'Close', 'Volume', 'Volatility',
            'RSI', 'RSI_Change',
            'MACD_Hist',
            'Volume_Change',
            'Log_Return'
        ]
        
        print(f"✓ Model A (Sentiment+): {len(self.features_with_sentiment)} features")
        print(f"✓ Model B (Technical): {len(self.features_no_sentiment)} features")
        print(f"✓ Model C (Crisis): {len(self.features_crisis)} features")
        print()
        
        return self.data


class RegimeDetector:
    """
    Hidden Markov Model for regime detection
    States: 0 = Normal, 1 = Crisis
    """
    
    def __init__(self, n_states=2):
        self.n_states = n_states
        self.model = None
        
    def fit(self, data):
        """Fit HMM on volatility and volume"""
        print("🔍 Training Regime Detector (HMM):")
        print("-" * 40)
        
        # Use volatility and volume as regime indicators
        features = data[['Volatility', 'Volume']].values
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Train Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.model.fit(features_scaled)
        self.scaler = scaler
        
        # Predict regimes
        regimes = self.model.predict(features_scaled)
        
        # Map states: Higher volatility state = Crisis
        state_volatility = [
            data[regimes == i]['Volatility'].mean()
            for i in range(self.n_states)
        ]
        crisis_state = np.argmax(state_volatility)
        
        # Remap so 1 = Crisis, 0 = Normal
        if crisis_state == 0:
            regimes = 1 - regimes
        
        print(f"✓ HMM trained with {self.n_states} states")
        print(f"✓ Crisis state identified (higher volatility)")
        print(f"✓ Normal periods: {(regimes == 0).sum()} days ({(regimes == 0).mean()*100:.1f}%)")
        print(f"✓ Crisis periods: {(regimes == 1).sum()} days ({(regimes == 1).mean()*100:.1f}%)")
        print()
        
        return regimes
    
    def predict(self, data):
        """Predict regime for new data"""
        features = data[['Volatility', 'Volume']].values
        features_scaled = self.scaler.transform(features)
        regimes = self.model.predict(features_scaled)
        
        # Apply same mapping as fit()
        state_volatility = [
            data[regimes == i]['Volatility'].mean()
            for i in range(self.n_states)
        ]
        crisis_state = np.argmax(state_volatility)
        if crisis_state == 0:
            regimes = 1 - regimes
            
        return regimes


class TransformerBiLSTM:
    """
    Simplified Bi-LSTM architecture (proven from Phase 1)
    """
    
    def __init__(self, seq_length, n_features, model_name="Hybrid"):
        self.seq_length = seq_length
        self.n_features = n_features
        self.model_name = model_name
        self.model = None
        
    def build(self):
        """Build Bi-LSTM architecture (Phase 1 approach)"""
        inputs = Input(shape=(self.seq_length, self.n_features))
        
        # First Bi-LSTM layer
        lstm1 = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        lstm1 = LayerNormalization()(lstm1)
        lstm1 = Dropout(0.3)(lstm1)
        
        # Second Bi-LSTM layer
        lstm2 = Bidirectional(LSTM(32, return_sequences=False))(lstm1)
        lstm2 = Dropout(0.3)(lstm2)
        
        # Dense layers
        dense1 = Dense(16, activation='relu')(lstm2)
        dense1 = Dropout(0.2)(dense1)
        output = Dense(1, activation='sigmoid')(dense1)
        
        self.model = Model(inputs=inputs, outputs=output)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # Higher LR
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train with early stopping"""
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=64,  # Larger batch size
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save(self, filepath):
        """Save model"""
        self.model.save(filepath)
        print(f"✓ Model saved: {filepath}")


class ABTestingFramework:
    """
    A/B Testing: Compare multiple model configurations
    """
    
    def __init__(self, data, feature_engineer):
        self.data = data
        self.fe = feature_engineer
        self.results = {}
        
    def create_sequences(self, feature_cols, seq_length=60):
        """Create sequences for training with normalization"""
        # Normalize features first
        scaler = RobustScaler()
        scaled_data = self.data[feature_cols].copy()
        scaled_data[feature_cols] = scaler.fit_transform(scaled_data[feature_cols])
        
        X, y = [], []
        
        for i in range(seq_length, len(self.data)):
            X.append(scaled_data.iloc[i-seq_length:i].values)
            y.append(self.data['Target'].iloc[i])
        
        return np.array(X), np.array(y)
    
    def train_model(self, model_name, feature_cols, seq_length=60):
        """Train a single model configuration"""
        print(f"\n{'='*60}")
        print(f"Training Model: {model_name}")
        print(f"Features: {len(feature_cols)}")
        print(f"{'='*60}\n")
        
        # Create sequences
        X, y = self.create_sequences(feature_cols, seq_length)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Further split train into train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Build and train model
        model_builder = TransformerBiLSTM(
            seq_length=seq_length,
            n_features=len(feature_cols),
            model_name=model_name
        )
        model_builder.build()
        
        print(f"Model Parameters: {model_builder.model.count_params():,}")
        
        history = model_builder.train(X_train, y_train, X_val, y_val, epochs=30)
        
        # Evaluate
        y_pred_proba = model_builder.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print(f"Results for {model_name}:")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print()
        
        # Store results
        self.results[model_name] = {
            'model': model_builder,
            'accuracy': accuracy,
            'f1': f1,
            'y_test': y_test,
            'y_pred': y_pred,
            'history': history
        }
        
        return model_builder, accuracy, f1
    
    def run_ab_tests(self):
        """Run all A/B tests"""
        print("\n" + "="*80)
        print("A/B TESTING: COMPARING MODEL CONFIGURATIONS")
        print("="*80 + "\n")
        
        # Model A: With Sentiment Volatility + Lags
        print("\n🔬 Model A: Sentiment Volatility + Lag Features")
        self.train_model("A_Sentiment_Plus", self.fe.features_with_sentiment)
        
        # Model B: Pure Technical (No Sentiment)
        print("\n🔬 Model B: Pure Technical (No Sentiment)")
        self.train_model("B_Technical_Only", self.fe.features_no_sentiment)
        
        return self.results
    
    def compare_results(self):
        """Generate comparison report"""
        print("\n" + "="*80)
        print("A/B TEST RESULTS COMPARISON")
        print("="*80 + "\n")
        
        # Sort by F1-score
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['f1'],
            reverse=True
        )
        
        print(f"{'Model':<30} {'Accuracy':<12} {'F1-Score':<12} {'Winner'}")
        print("-" * 80)
        
        for i, (name, res) in enumerate(sorted_results):
            winner = "🏆 WINNER" if i == 0 else ""
            print(f"{name:<30} {res['accuracy']:<12.4f} {res['f1']:<12.4f} {winner}")
        
        print()
        
        # Statistical comparison
        best_model = sorted_results[0][0]
        best_f1 = sorted_results[0][1]['f1']
        
        print(f"✓ Best Model: {best_model}")
        print(f"✓ Best F1-Score: {best_f1:.4f}")
        
        if len(sorted_results) > 1:
            second_best_f1 = sorted_results[1][1]['f1']
            if second_best_f1 > 0:
                improvement = ((best_f1 - second_best_f1) / second_best_f1) * 100
                print(f"✓ Improvement: {improvement:+.2f}% vs second-best")
            else:
                print(f"✓ Improvement: {best_f1:.4f} vs {second_best_f1:.4f} (second model failed to learn)")
        
        print()
        
        return best_model, self.results[best_model]
    
    def plot_comparison(self):
        """Visualize A/B test results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Phase 3 A/B Testing Results', fontsize=16, fontweight='bold')
        
        # 1. Accuracy comparison
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy'] for m in models]
        f1_scores = [self.results[m]['f1'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax1.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='steelblue')
        ax1.bar(x + width/2, f1_scores, width, label='F1-Score', alpha=0.8, color='darkgreen')
        ax1.set_xlabel('Model', fontweight='bold')
        ax1.set_ylabel('Score', fontweight='bold')
        ax1.set_title('Model Performance Comparison', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. Training history (best model)
        best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['f1'])
        history = self.results[best_model_name]['history']
        
        ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_xlabel('Epoch', fontweight='bold')
        ax2.set_ylabel('Loss', fontweight='bold')
        ax2.set_title(f'Training History: {best_model_name}', fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Confusion matrices comparison
        for i, (model_name, res) in enumerate(self.results.items()):
            ax = [ax3, ax4][i]
            
            cm = confusion_matrix(res['y_test'], res['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{model_name}\nF1: {res["f1"]:.4f}', fontweight='bold')
            ax.set_xlabel('Predicted', fontweight='bold')
            ax.set_ylabel('Actual', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/phase3_ab_testing_results.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: phase3_ab_testing_results.png")
        print()


class RegimeSwitchingSystem:
    """
    Complete regime-switching prediction system
    """
    
    def __init__(self, data, feature_engineer):
        self.data = data
        self.fe = feature_engineer
        self.regime_detector = RegimeDetector(n_states=2)
        self.normal_model = None
        self.crisis_model = None
        
    def train_specialist_models(self):
        """Train separate models for Normal and Crisis regimes"""
        print("\n" + "="*80)
        print("REGIME-SWITCHING SYSTEM: TRAINING SPECIALIST MODELS")
        print("="*80 + "\n")
        
        # Detect regimes
        regimes = self.regime_detector.fit(self.data)
        self.data['Regime'] = regimes
        
        # Split data by regime
        normal_data = self.data[self.data['Regime'] == 0].copy()
        crisis_data = self.data[self.data['Regime'] == 1].copy()
        
        print(f"📊 Data Split:")
        print(f"  Normal: {len(normal_data):,} days ({len(normal_data)/len(self.data)*100:.1f}%)")
        print(f"  Crisis: {len(crisis_data):,} days ({len(crisis_data)/len(self.data)*100:.1f}%)")
        print()
        
        # Train Normal model (with sentiment features)
        print("\n🟢 Training NORMAL Regime Model (Sentiment+):")
        print("-" * 60)
        normal_ab = ABTestingFramework(normal_data, self.fe)
        self.normal_model, _, _ = normal_ab.train_model(
            "Normal_Specialist",
            self.fe.features_with_sentiment
        )
        
        # Train Crisis model (technical only)
        print("\n🔴 Training CRISIS Regime Model (Technical Only):")
        print("-" * 60)
        crisis_ab = ABTestingFramework(crisis_data, self.fe)
        self.crisis_model, _, _ = crisis_ab.train_model(
            "Crisis_Specialist",
            self.fe.features_crisis
        )
        
        print("\n✓ Regime-switching system ready!")
        print()


def main():
    """Main execution pipeline"""
    
    # 1. Load data
    print("📂 Loading data...")
    data = pd.read_csv('../data/stock_market_data_large.csv')
    print(f"✓ Loaded {len(data):,} rows, {len(data.columns)} columns\n")
    
    # 2. Feature engineering
    fe = AdvancedFeatureEngineer(data)
    data = fe.prepare_features()
    
    # 3. A/B Testing Framework
    ab_test = ABTestingFramework(data, fe)
    results = ab_test.run_ab_tests()
    
    # 4. Compare results
    best_model_name, best_result = ab_test.compare_results()
    ab_test.plot_comparison()
    
    # 5. Regime-switching system
    regime_system = RegimeSwitchingSystem(data, fe)
    regime_system.train_specialist_models()
    
    # 6. Final summary
    print("\n" + "="*80)
    print("PHASE 3 COMPLETE")
    print("="*80)
    print(f"\n🏆 Winner: {best_model_name}")
    print(f"📊 F1-Score: {best_result['f1']:.4f}")
    print(f"📊 Accuracy: {best_result['accuracy']:.4f}")
    print()
    print("✓ Next steps:")
    print("  1. Run SHAP analysis on best model")
    print("  2. Collect real 2008/2020 crisis data")
    print("  3. Deploy regime-switching system")
    print()


if __name__ == "__main__":
    main()
