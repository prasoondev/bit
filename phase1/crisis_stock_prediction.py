"""
Crisis-Aware Stock Prediction via Deep Learning & Sentiment Integration

This module implements a hybrid Bi-LSTM model with attention mechanism
to predict stock price movements during crisis and normal periods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, mean_absolute_error, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, 
    Attention, Concatenate, LayerNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class StockDataPreprocessor:
    """
    Handles data loading, cleaning, and feature engineering for stock market data.
    """
    
    def __init__(self, filepath):
        """
        Initialize preprocessor with data file path.
        
        Args:
            filepath: Path to the CSV file containing stock market data
        """
        self.filepath = filepath
        self.df = None
        self.scaler = RobustScaler()
        self.feature_columns = None
        
    def load_data(self):
        """Load CSV and handle missing values using linear interpolation."""
        print("Loading data...")
        self.df = pd.read_csv(self.filepath)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df.set_index('Date', inplace=True)
        
        print(f"Data loaded: {len(self.df)} rows")
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")
        print(f"Missing values before interpolation:\n{self.df.isnull().sum()}")
        
        # Handle missing values with linear interpolation
        self.df.interpolate(method='linear', inplace=True)
        self.df.bfill(inplace=True)  # Fill any remaining NaN at the start
        
        print(f"Missing values after interpolation:\n{self.df.isnull().sum()}")
        return self
    
    def engineer_features(self):
        """
        Create additional features:
        - Log returns from Close prices
        - Daily percentage changes
        - Sentiment volatility (7-day rolling std)
        """
        print("\nEngineering features...")
        
        # Calculate log returns for stationarity
        self.df['Log_Return'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        
        # Daily percentage change
        self.df['Pct_Change'] = self.df['Close'].pct_change()
        
        # Sentiment volatility (7-day rolling standard deviation)
        self.df['Sentiment_Volatility'] = self.df['Sentiment'].rolling(window=7).std()
        
        # Volume change
        self.df['Volume_Change'] = self.df['Volume'].pct_change()
        
        # Price range (High - Low)
        self.df['Price_Range'] = self.df['High'] - self.df['Low']
        
        # MACD signal line and histogram
        self.df['MACD_Signal'] = self.df['MACD'].rolling(window=9).mean()
        self.df['MACD_Hist'] = self.df['MACD'] - self.df['MACD_Signal']
        
        # Drop NaN values created by rolling operations
        self.df.dropna(inplace=True)
        
        print(f"Features engineered. New shape: {self.df.shape}")
        print(f"New features added: Log_Return, Pct_Change, Sentiment_Volatility, Volume_Change, Price_Range, MACD_Signal, MACD_Hist")
        
        return self
    
    def identify_crisis_periods(self):
        """
        Identify crisis periods based on extreme volatility and market conditions.
        Crisis periods:
        - 2008 Financial Crisis: Not in dataset (starts from 2010)
        - 2020 COVID-19 Crisis: March-April 2020
        - 2024-25 potential crisis: Check latest data
        """
        print("\nIdentifying crisis periods...")
        
        # Mark crisis periods
        self.df['Crisis'] = 0
        
        # COVID-19 Crisis (March-April 2020)
        covid_mask = (self.df.index >= '2020-03-01') & (self.df.index <= '2020-04-30')
        self.df.loc[covid_mask, 'Crisis'] = 1
        
        # Recent potential crisis (2024-2025) - identify by high volatility
        recent_mask = self.df.index >= '2024-01-01'
        if recent_mask.any():
            # Calculate rolling volatility for recent period
            recent_vol = self.df.loc[recent_mask, 'Log_Return'].rolling(window=20).std()
            high_vol_threshold = recent_vol.quantile(0.75)
            high_vol_mask = recent_mask & (self.df['Log_Return'].rolling(window=20).std() > high_vol_threshold)
            self.df.loc[high_vol_mask, 'Crisis'] = 1
        
        crisis_count = (self.df['Crisis'] == 1).sum()
        print(f"Crisis periods identified: {crisis_count} days ({crisis_count/len(self.df)*100:.2f}%)")
        
        return self
    
    def normalize_features(self, features_to_scale=None):
        """
        Apply RobustScaler to numerical features to mitigate outlier impact.
        
        Args:
            features_to_scale: List of feature names to normalize. If None, uses default set.
        """
        print("\nNormalizing features with RobustScaler...")
        
        if features_to_scale is None:
            # Default features to scale (excluding target and crisis label)
            features_to_scale = [
                'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD',
                'Sentiment', 'Log_Return', 'Pct_Change', 'Sentiment_Volatility',
                'Volume_Change', 'Price_Range', 'MACD_Signal', 'MACD_Hist'
            ]
        
        self.feature_columns = features_to_scale
        
        # Fit and transform
        self.df[features_to_scale] = self.scaler.fit_transform(self.df[features_to_scale])
        
        print(f"Normalized {len(features_to_scale)} features")
        return self
    
    def get_processed_data(self):
        """Return the processed dataframe."""
        return self.df


class SequenceGenerator:
    """
    Generate sliding window sequences for time-series modeling.
    """
    
    def __init__(self, lookback=60):
        """
        Initialize sequence generator.
        
        Args:
            lookback: Number of time steps to look back (default: 60 days)
        """
        self.lookback = lookback
        
    def create_sequences(self, data, feature_columns, target_column='Target'):
        """
        Transform data into 3D tensor (samples, time_steps, features).
        
        Args:
            data: DataFrame with processed features
            feature_columns: List of column names to use as features
            target_column: Name of the target column
            
        Returns:
            X: 3D numpy array of shape (samples, lookback, n_features)
            y: 1D numpy array of targets
            dates: Array of dates corresponding to each sample
        """
        print(f"\nCreating sequences with {self.lookback}-day lookback window...")
        
        X, y, dates = [], [], []
        
        feature_data = data[feature_columns].values
        target_data = data[target_column].values
        date_index = data.index
        
        for i in range(self.lookback, len(data)):
            X.append(feature_data[i-self.lookback:i])
            y.append(target_data[i])
            dates.append(date_index[i])
        
        X = np.array(X)
        y = np.array(y)
        dates = np.array(dates)
        
        print(f"Sequences created: X shape {X.shape}, y shape {y.shape}")
        return X, y, dates


class AttentionBiLSTM:
    """
    Hybrid Bidirectional LSTM model with Attention mechanism for stock prediction.
    """
    
    def __init__(self, input_shape, lstm_units=64, dropout_rate=0.3):
        """
        Initialize the Bi-LSTM model with attention.
        
        Args:
            input_shape: Tuple (time_steps, n_features)
            lstm_units: Number of LSTM units (default: 64)
            dropout_rate: Dropout rate for regularization (default: 0.3)
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build the Bi-LSTM model architecture with attention layer.
        
        The attention mechanism helps the model focus on important time steps,
        especially during volatility spikes.
        """
        print("\nBuilding Bi-LSTM model with Attention...")
        
        # Input layer
        inputs = Input(shape=self.input_shape, name='input_layer')
        
        # First Bidirectional LSTM layer (returns sequences for attention)
        lstm_out = Bidirectional(
            LSTM(self.lstm_units, return_sequences=True, name='bi_lstm_1'),
            name='bidirectional_lstm'
        )(inputs)
        
        # Layer normalization
        lstm_out = LayerNormalization(name='layer_norm_1')(lstm_out)
        
        # Attention mechanism
        # This helps the model focus on important time steps (e.g., sentiment spikes during crisis)
        attention_out = Attention(name='attention_layer')([lstm_out, lstm_out])
        
        # Concatenate attention output with LSTM output
        concat = Concatenate(name='concat_layer')([lstm_out, attention_out])
        
        # Second Bidirectional LSTM layer
        lstm_out_2 = Bidirectional(
            LSTM(self.lstm_units // 2, return_sequences=False, name='bi_lstm_2'),
            name='bidirectional_lstm_2'
        )(concat)
        
        # Layer normalization
        lstm_out_2 = LayerNormalization(name='layer_norm_2')(lstm_out_2)
        
        # Dropout for regularization (prevent overfitting)
        dropout = Dropout(self.dropout_rate, name='dropout_layer')(lstm_out_2)
        
        # Dense hidden layer
        dense = Dense(32, activation='relu', name='dense_hidden')(dropout)
        dropout_2 = Dropout(self.dropout_rate / 2, name='dropout_layer_2')(dense)
        
        # Output layer (binary classification)
        outputs = Dense(1, activation='sigmoid', name='output_layer')(dropout_2)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=outputs, name='Crisis_Aware_BiLSTM')
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc'), 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        print("\nModel architecture:")
        self.model.summary()
        
        return self
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the model with early stopping and learning rate reduction.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Maximum number of epochs
            batch_size: Batch size for training
        """
        print("\nTraining model...")
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print("\nTraining completed!")
        return self
    
    def predict(self, X):
        """Make predictions."""
        return (self.model.predict(X) > 0.5).astype(int).flatten()
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.model.predict(X).flatten()
    
    def save_model(self, filepath):
        """Save trained model."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Crisis-Aware Bi-LSTM Training History', fontsize=16, fontweight='bold', y=0.995)
        
        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 0].plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Binary Cross-Entropy Loss', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch', fontsize=11)
        axes[0, 0].set_ylabel('Loss', fontsize=11)
        axes[0, 0].legend(loc='best', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 1].plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Classification Accuracy', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch', fontsize=11)
        axes[0, 1].set_ylabel('Accuracy', fontsize=11)
        axes[0, 1].legend(loc='best', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # AUC
        axes[1, 0].plot(self.history.history['auc'], label='Train AUC-ROC', linewidth=2)
        axes[1, 0].plot(self.history.history['val_auc'], label='Validation AUC-ROC', linewidth=2)
        axes[1, 0].set_title('Area Under ROC Curve', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=11)
        axes[1, 0].set_ylabel('AUC', fontsize=11)
        axes[1, 0].legend(loc='best', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precision & Recall
        axes[1, 1].plot(self.history.history['precision'], label='Train Precision', linewidth=2, linestyle='-')
        axes[1, 1].plot(self.history.history['val_precision'], label='Val Precision', linewidth=2, linestyle='-')
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall', linewidth=2, linestyle='--')
        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall', linewidth=2, linestyle='--')
        axes[1, 1].set_title('Precision & Recall Metrics', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=11)
        axes[1, 1].set_ylabel('Score', fontsize=11)
        axes[1, 1].legend(loc='best', fontsize=9, ncol=2)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()


class CrisisBacktester:
    """
    Backtesting framework specifically for crisis periods.
    """
    
    def __init__(self, model, preprocessor):
        """
        Initialize backtester.
        
        Args:
            model: Trained AttentionBiLSTM model
            preprocessor: StockDataPreprocessor instance with processed data
        """
        self.model = model
        self.preprocessor = preprocessor
        
    def backtest_crisis_period(self, X_test, y_test, dates_test, crisis_labels):
        """
        Evaluate model specifically on crisis periods.
        
        Args:
            X_test: Test features
            y_test: Test targets
            dates_test: Test dates
            crisis_labels: Binary labels indicating crisis periods
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("\n" + "="*80)
        print("CRISIS PERIOD BACKTESTING")
        print("="*80)
        
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Overall metrics
        print("\nOVERALL PERFORMANCE:")
        print("-"*40)
        accuracy = np.mean(y_pred == y_test)
        f1 = f1_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"MAE: {mae:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
        
        # Crisis-specific metrics
        crisis_mask = crisis_labels == 1
        
        if crisis_mask.sum() > 0:
            print("\n" + "="*80)
            print("CRISIS-SPECIFIC PERFORMANCE:")
            print("="*80)
            
            y_test_crisis = y_test[crisis_mask]
            y_pred_crisis = y_pred[crisis_mask]
            
            accuracy_crisis = np.mean(y_pred_crisis == y_test_crisis)
            f1_crisis = f1_score(y_test_crisis, y_pred_crisis)
            mae_crisis = mean_absolute_error(y_test_crisis, y_pred_crisis)
            
            print(f"\nCrisis Period Samples: {crisis_mask.sum()} ({crisis_mask.sum()/len(y_test)*100:.2f}%)")
            print(f"Accuracy (Crisis): {accuracy_crisis:.4f}")
            print(f"F1-Score (Crisis): {f1_crisis:.4f}")
            print(f"MAE (Crisis): {mae_crisis:.4f}")
            
            print("\nClassification Report (Crisis Periods):")
            print(classification_report(y_test_crisis, y_pred_crisis, target_names=['Down', 'Up']))
            
            # Normal period metrics for comparison
            normal_mask = crisis_labels == 0
            if normal_mask.sum() > 0:
                print("\n" + "="*80)
                print("NORMAL PERIOD PERFORMANCE (for comparison):")
                print("="*80)
                
                y_test_normal = y_test[normal_mask]
                y_pred_normal = y_pred[normal_mask]
                
                accuracy_normal = np.mean(y_pred_normal == y_test_normal)
                f1_normal = f1_score(y_test_normal, y_pred_normal)
                mae_normal = mean_absolute_error(y_test_normal, y_pred_normal)
                
                print(f"\nNormal Period Samples: {normal_mask.sum()} ({normal_mask.sum()/len(y_test)*100:.2f}%)")
                print(f"Accuracy (Normal): {accuracy_normal:.4f}")
                print(f"F1-Score (Normal): {f1_normal:.4f}")
                print(f"MAE (Normal): {mae_normal:.4f}")
        else:
            print("\nNo crisis periods found in test set.")
            f1_crisis = None
            mae_crisis = None
        
        # Confusion matrix
        self._plot_confusion_matrix(y_test, y_pred, "Overall")
        
        if crisis_mask.sum() > 0:
            self._plot_confusion_matrix(y_test_crisis, y_pred_crisis, "Crisis Periods")
        
        results = {
            'overall': {
                'accuracy': accuracy,
                'f1_score': f1,
                'mae': mae
            },
            'crisis': {
                'accuracy': accuracy_crisis if crisis_mask.sum() > 0 else None,
                'f1_score': f1_crisis,
                'mae': mae_crisis,
                'n_samples': crisis_mask.sum()
            }
        }
        
        return results
    
    def _plot_confusion_matrix(self, y_true, y_pred, title="Confusion Matrix"):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percentages = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(10, 8))
        
        # Create annotations with both counts and percentages
        annotations = np.array([[f'{count}\n({pct:.1f}%)' 
                                for count, pct in zip(row_counts, row_pcts)]
                               for row_counts, row_pcts in zip(cm, cm_percentages)])
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                   xticklabels=['Down (0)', 'Up (1)'],
                   yticklabels=['Down (0)', 'Up (1)'],
                   cbar_kws={'label': 'Number of Predictions'},
                   linewidths=2, linecolor='white')
        
        plt.title(f'{title}', fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'plots/confusion_matrix_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_sentiment_as_leading_indicator(self, X_test, y_test, dates_test, 
                                               crisis_labels, sentiment_col_idx):
        """
        Analyze whether sentiment acts as a leading indicator during crises.
        
        Args:
            X_test: Test features
            y_test: Test targets
            dates_test: Test dates
            crisis_labels: Binary crisis labels
            sentiment_col_idx: Index of sentiment column in feature array
        """
        print("\n" + "="*80)
        print("SENTIMENT AS LEADING INDICATOR ANALYSIS")
        print("="*80)
        
        # Extract sentiment values (last time step of each sequence)
        sentiment_values = X_test[:, -1, sentiment_col_idx]
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Create DataFrame for analysis
        analysis_df = pd.DataFrame({
            'Date': dates_test,
            'Sentiment': sentiment_values,
            'True_Target': y_test,
            'Pred_Proba': y_pred_proba,
            'Crisis': crisis_labels
        })
        
        # Calculate correlation during crisis vs normal periods
        crisis_df = analysis_df[analysis_df['Crisis'] == 1]
        normal_df = analysis_df[analysis_df['Crisis'] == 0]
        
        if len(crisis_df) > 0:
            corr_crisis = crisis_df['Sentiment'].corr(crisis_df['Pred_Proba'])
            print(f"\nSentiment-Prediction Correlation (Crisis): {corr_crisis:.4f}")
        
        if len(normal_df) > 0:
            corr_normal = normal_df['Sentiment'].corr(normal_df['Pred_Proba'])
            print(f"Sentiment-Prediction Correlation (Normal): {corr_normal:.4f}")
        
        # Plot sentiment vs prediction probability
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Sentiment Analysis: Leading Indicator Evaluation', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        # Crisis periods
        if len(crisis_df) > 0:
            scatter1 = axes[0].scatter(crisis_df['Sentiment'], crisis_df['Pred_Proba'], 
                          alpha=0.6, c=crisis_df['True_Target'], cmap='RdYlGn',
                          s=50, edgecolors='black', linewidth=0.5)
            axes[0].set_title(f'Crisis Periods (n={len(crisis_df)}, corr={corr_crisis:.4f})', 
                            fontsize=12, fontweight='bold')
            axes[0].set_xlabel('Sentiment Score (Normalized)', fontsize=11, fontweight='bold')
            axes[0].set_ylabel('Predicted Probability (Price Up)', fontsize=11, fontweight='bold')
            axes[0].grid(True, alpha=0.3, linestyle='--')
            axes[0].axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Decision Threshold')
            axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            axes[0].legend(loc='upper left', fontsize=10)
            
            # Add colorbar
            cbar1 = plt.colorbar(scatter1, ax=axes[0])
            cbar1.set_label('Actual Outcome (0=Down, 1=Up)', fontsize=10, fontweight='bold')
            cbar1.set_ticks([0, 1])
            cbar1.set_ticklabels(['Down', 'Up'])
        
        # Normal periods
        if len(normal_df) > 0:
            scatter2 = axes[1].scatter(normal_df['Sentiment'], normal_df['Pred_Proba'], 
                          alpha=0.6, c=normal_df['True_Target'], cmap='RdYlGn',
                          s=50, edgecolors='black', linewidth=0.5)
            axes[1].set_title(f'Normal Periods (n={len(normal_df)}, corr={corr_normal:.4f})', 
                            fontsize=12, fontweight='bold')
            axes[1].set_xlabel('Sentiment Score (Normalized)', fontsize=11, fontweight='bold')
            axes[1].set_ylabel('Predicted Probability (Price Up)', fontsize=11, fontweight='bold')
            axes[1].grid(True, alpha=0.3, linestyle='--')
            axes[1].axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Decision Threshold')
            axes[1].axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
            axes[1].legend(loc='upper left', fontsize=10)
            
            # Add colorbar
            cbar2 = plt.colorbar(scatter2, ax=axes[1])
            cbar2.set_label('Actual Outcome (0=Down, 1=Up)', fontsize=10, fontweight='bold')
            cbar2.set_ticks([0, 1])
            cbar2.set_ticklabels(['Down', 'Up'])
        
        plt.tight_layout()
        plt.savefig('plots/sentiment_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Main execution pipeline for crisis-aware stock prediction.
    """
    print("="*80)
    print("CRISIS-AWARE STOCK PREDICTION PIPELINE")
    print("="*80)
    
    # Configuration
    DATA_FILE = '../data/stock_market_data_large.csv'
    LOOKBACK = 60
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    
    # Step 1: Data Preprocessing
    print("\n[STEP 1: DATA PREPROCESSING]")
    preprocessor = StockDataPreprocessor(DATA_FILE)
    preprocessor.load_data()
    preprocessor.engineer_features()
    preprocessor.identify_crisis_periods()
    
    # Select features for modeling
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD',
        'Sentiment', 'Log_Return', 'Pct_Change', 'Sentiment_Volatility',
        'Volume_Change', 'Price_Range', 'MACD_Signal', 'MACD_Hist'
    ]
    
    preprocessor.normalize_features(feature_columns)
    df = preprocessor.get_processed_data()
    
    # Step 2: Sequence Generation
    print("\n[STEP 2: SEQUENCE GENERATION]")
    seq_gen = SequenceGenerator(lookback=LOOKBACK)
    X, y, dates = seq_gen.create_sequences(df, feature_columns, target_column='Target')
    
    # Extract crisis labels for test set
    crisis_labels = df['Crisis'].values[LOOKBACK:]
    
    # Step 3: Train/Val/Test Split
    print("\n[STEP 3: DATA SPLITTING]")
    n_samples = len(X)
    test_start = int(n_samples * (1 - TEST_SIZE))
    val_start = int(test_start * (1 - VAL_SIZE))
    
    X_train = X[:val_start]
    y_train = y[:val_start]
    
    X_val = X[val_start:test_start]
    y_val = y[val_start:test_start]
    
    X_test = X[test_start:]
    y_test = y[test_start:]
    dates_test = dates[test_start:]
    crisis_test = crisis_labels[test_start:]
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Crisis samples in test set: {crisis_test.sum()} ({crisis_test.sum()/len(crisis_test)*100:.2f}%)")
    
    # Step 4: Model Building and Training
    print("\n[STEP 4: MODEL BUILDING & TRAINING]")
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    model = AttentionBiLSTM(input_shape=input_shape, lstm_units=64, dropout_rate=0.3)
    model.build_model()
    model.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    # Plot training history
    model.plot_training_history(save_path='plots/training_history.png')
    
    # Step 5: Backtesting
    print("\n[STEP 5: BACKTESTING]")
    backtester = CrisisBacktester(model, preprocessor)
    results = backtester.backtest_crisis_period(X_test, y_test, dates_test, crisis_test)
    
    # Step 6: Sentiment Analysis
    print("\n[STEP 6: SENTIMENT ANALYSIS]")
    sentiment_idx = feature_columns.index('Sentiment')
    backtester.analyze_sentiment_as_leading_indicator(
        X_test, y_test, dates_test, crisis_test, sentiment_idx
    )
    
    # Save model
    print("\n[SAVING MODEL]")
    model.save_model('crisis_aware_bilstm_model.h5')
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print("\nKey Results:")
    print(f"  - Overall F1-Score: {results['overall']['f1_score']:.4f}")
    print(f"  - Overall MAE: {results['overall']['mae']:.4f}")
    if results['crisis']['f1_score'] is not None:
        print(f"  - Crisis F1-Score: {results['crisis']['f1_score']:.4f}")
        print(f"  - Crisis MAE: {results['crisis']['mae']:.4f}")
    print("\nGenerated files:")
    print("  - crisis_aware_bilstm_model.h5")
    print("  - training_history.png")
    print("  - confusion_matrix_overall.png")
    print("  - confusion_matrix_crisis_periods.png")
    print("  - sentiment_analysis.png")


if __name__ == "__main__":
    main()
