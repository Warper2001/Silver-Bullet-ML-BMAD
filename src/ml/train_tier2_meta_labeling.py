#!/usr/bin/env python3
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
from pathlib import Path

def train_model():
    csv_path = Path("data/ml_training/tier2_meta_labeling.csv")
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    # Features and Label
    X = df.drop(columns=["label"])
    y = df["label"]

    print(f"Feature columns: {list(X.columns)}")
    print(f"Label distribution: {y.value_counts(normalize=True).to_dict()}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost
    # Small dataset (672 samples), so keep it simple
    model = XGBClassifier(
        n_estimators=30,
        max_depth=2,
        learning_rate=0.03,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate
    y_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_proba)
    acc = accuracy_score(y_val, (y_proba > 0.5).astype(int))

    print(f"Validation AUC: {auc:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")

    # Save model
    model_dir = Path("models/xgboost")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "tier2_meta_labeling_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
