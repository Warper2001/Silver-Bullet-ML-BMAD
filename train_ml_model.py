#!/usr/bin/env python3
"""Train XGBoost model to filter trading signals on real MNQ data."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from src.research.historical_data_loader import HistoricalDataLoader

def calculate_technical_indicators(df):
    """Calculate technical indicators for features."""
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    df['bb_std'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    # Momentum
    df['momentum'] = df['close'] - df['close'].shift(10)
    df['momentum_pct'] = df['momentum'] / df['close']

    # Volume features
    df['volume_ma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Price rate of change
    df['roc'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100

    # Volatility
    df['volatility'] = df['close'].rolling(window=20).std()
    df['volatility_pct'] = df['volatility'] / df['close']

    return df

def generate_training_signals(df):
    """Generate signals and calculate future returns for training."""
    signals = []

    for i in range(100, len(df) - 100):  # Leave room for forward/outcome calculation
        current_bar = df.iloc[i]
        prev_bar = df.iloc[i-1]

        # Feature-based signals
        features = {
            'timestamp': current_bar.name,
            'rsi': current_bar['rsi'],
            'rsi_change': current_bar['rsi'] - prev_bar['rsi'],
            'sma_20_above_sma_50': 1 if current_bar['sma_20'] > current_bar['sma_50'] else 0,
            'price_vs_sma20': (current_bar['close'] - current_bar['sma_20']) / current_bar['sma_20'],
            'bb_position': current_bar['bb_position'],
            'bb_width': current_bar['bb_width'],
            'atr_pct': current_bar['atr_pct'],
            'momentum_pct': current_bar['momentum_pct'],
            'volume_ratio': current_bar['volume_ratio'],
            'roc': current_bar['roc'],
            'volatility_pct': current_bar['volatility_pct'],
        }

        # Calculate forward return (next 10 bars)
        future_bars = df.iloc[i+1:i+11]
        if len(future_bars) == 10:
            future_return = (future_bars['close'].iloc[-1] - current_bar['close']) / current_bar['close']

            # Binary label: 1 if positive return, 0 otherwise
            label = 1 if future_return > 0 else 0

            # Signal type
            if current_bar['rsi'] < 30:
                signal_type = 'RSI_OVERSOLD'
            elif current_bar['rsi'] > 70:
                signal_type = 'RSI_OVERBOUGHT'
            elif current_bar['momentum_pct'] > 0 and prev_bar['momentum_pct'] <= 0:
                signal_type = 'MOMENTUM_UP'
            elif current_bar['momentum_pct'] < 0 and prev_bar['momentum_pct'] >= 0:
                signal_type = 'MOMENTUM_DOWN'
            elif current_bar['close'] > current_bar['bb_upper']:
                signal_type = 'BB_BREAKOUT_UP'
            elif current_bar['close'] < current_bar['bb_lower']:
                signal_type = 'BB_BREAKOUT_DOWN'
            else:
                continue  # Skip if no clear signal

            features.update({
                'signal_type': signal_type,
                'future_return': future_return,
                'label': label,
                'close': current_bar['close'],
            })

            signals.append(features)

    return pd.DataFrame(signals)

def prepare_features(df):
    """Prepare features for ML training."""
    # One-hot encode signal_type
    signal_dummies = pd.get_dummies(df['signal_type'], prefix='signal')
    df = pd.concat([df, signal_dummies], axis=1)

    # Select feature columns
    feature_cols = [
        'rsi', 'rsi_change',
        'sma_20_above_sma_50', 'price_vs_sma20',
        'bb_position', 'bb_width',
        'atr_pct', 'momentum_pct',
        'volume_ratio', 'roc', 'volatility_pct'
    ] + list(signal_dummies.columns)

    X = df[feature_cols].fillna(0)
    y = df['label']

    return X, y, feature_cols

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier."""
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model

def main():
    """Train ML model on real MNQ data."""

    print("🤖 TRAINING ML MODEL ON REAL MNQ DATA")
    print("=" * 60)

    # Load data (use more data for training)
    print("\n📊 Loading training data...")
    loader = HistoricalDataLoader(
        data_directory="data/processed/dollar_bars/",
        min_completeness=0.1
    )

    # Use wider date range for training
    data = loader.load_data('2024-12-01', '2025-03-01')
    print(f"✅ Loaded {len(data)} bars for training")

    # Calculate indicators
    print("\n🔬 Calculating technical indicators...")
    data = calculate_technical_indicators(data)

    # Generate training signals
    print("\n🎯 Generating training signals...")
    signals_df = generate_training_signals(data)
    print(f"✅ Generated {len(signals_df)} training samples")

    if len(signals_df) == 0:
        print("❌ No training signals generated!")
        return

    # Prepare features
    print("\n📋 Preparing features...")
    X, y, feature_cols = prepare_features(signals_df)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"✅ Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Train model
    print("\n🚀 Training XGBoost model...")
    model = train_xgboost_model(X_train, y_train, X_val, y_val)

    # Evaluate
    print("\n📈 Model Evaluation:")

    # Validation set
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)

    print(f"\n   Validation AUC: {val_auc:.4f}")

    val_accuracy = (val_pred == y_val).sum() / len(y_val)
    print(f"   Validation Accuracy: {val_accuracy:.2%}")

    # Test set
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_proba)

    print(f"   Test AUC: {test_auc:.4f}")

    test_accuracy = (test_pred == y_test).sum() / len(y_test)
    print(f"   Test Accuracy: {test_accuracy:.2%}")

    # Feature importance
    print("\n🔍 Feature Importance:")
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(importance_df.head(10).to_string(index=False))

    # Save model
    model_path = 'data/models/xgboost_mnq_classifier.pkl'
    Path('data/models').mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to {model_path}")

    # Save feature columns
    feature_cols_path = 'data/models/feature_columns.pkl'
    joblib.dump(feature_cols, feature_cols_path)
    print(f"✅ Feature columns saved to {feature_cols_path}")

    print("\n✅ Training complete!")
    print(f"   Model ready for backtesting with AUC: {test_auc:.4f}")

if __name__ == '__main__':
    main()
