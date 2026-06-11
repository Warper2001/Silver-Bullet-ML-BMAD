#!/usr/bin/env python3
"""Retrain Silver Bullet XGBoost meta-labeling model and run SHAP analysis.

Trains on the full 1182-signal dataset (Dec 2023 – Mar 2026) with the
new LR channel regime-context features (lr_slope_50/100, lr_dev_50/100,
lr_price_pos_50) added to the existing 52-feature set.

Outputs:
  models/xgboost/lr_enriched/         — retrained model + metadata
  data/reports/shap_lr_enriched.txt   — SHAP summary (terminal-friendly)
"""

import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.label_mapper import map_signals_to_outcomes
from src.ml.meta_training_data_builder import MetaLabelingDatasetBuilder
from src.ml.signal_feature_extractor import SignalFeatureExtractor
from src.ml.training_data import split_data, select_features
from src.ml.xgboost_trainer import XGBoostTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

SIGNALS_PATH  = Path("data/ml_training/silver_bullet_signals_full.parquet")
TRADES_PATH   = Path("data/ml_training/silver_bullet_trades_full.parquet")
METADATA_PATH = Path("data/ml_training/metadata_full.json")
TIME_BARS_DIR = Path("data/processed/time_bars")
MODEL_DIR     = "models/xgboost/lr_enriched"
REPORT_PATH   = Path("data/reports/shap_lr_enriched.txt")

LR_FEATURE_NAMES = [
    "lr_slope_50", "lr_slope_100",
    "lr_dev_50",   "lr_dev_100",
    "lr_price_pos_50",
]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_time_bars(start: str, end: str) -> pd.DataFrame:
    start_dt = pd.Timestamp(start)
    end_dt   = pd.Timestamp(end)
    current  = start_dt.replace(day=1)
    files    = []
    while current <= end_dt:
        p = TIME_BARS_DIR / f"MNQ_time_bars_5min_{current.strftime('%Y%m')}.h5"
        if p.exists():
            files.append(p)
        current += pd.DateOffset(months=1)

    import h5py
    frames = []
    for p in files:
        try:
            with h5py.File(p, "r") as f:
                data = f["dollar_bars"][:]
            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume", "notional_value"
            ])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            frames.append(df)
        except Exception as e:
            logger.warning(f"Skipping {p.name}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp").set_index("timestamp")
    combined = combined.loc[(combined.index >= start_dt) & (combined.index <= end_dt)]
    logger.info(f"Loaded {len(combined):,} time bars ({start} → {end})")
    return combined


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # 1. Load training data
    logger.info("Loading training data …")
    signals_df = pd.read_parquet(SIGNALS_PATH)
    trades_df  = pd.read_parquet(TRADES_PATH)
    with open(METADATA_PATH) as f:
        meta = json.load(f)
    logger.info(f"  {len(signals_df)} signals | {meta['date_range']['start']} → {meta['date_range']['end']}")

    # 2. Load price data
    logger.info("Loading time bars …")
    price_data = load_time_bars(meta["date_range"]["start"], meta["date_range"]["end"])

    # 3. Build feature matrix
    logger.info("Extracting features at signal timestamps …")
    extractor = SignalFeatureExtractor(lookback_bars=100)
    builder   = MetaLabelingDatasetBuilder(feature_extractor=extractor)
    dataset   = builder.build_dataset(signals_df, trades_df, price_data, verbose=True)
    logger.info(f"Dataset: {len(dataset)} samples × {len(dataset.columns)} columns")

    # Confirm LR features made it into the dataset
    lr_present = [c for c in LR_FEATURE_NAMES if c in dataset.columns]
    lr_missing = [c for c in LR_FEATURE_NAMES if c not in dataset.columns]
    logger.info(f"LR features present: {lr_present}")
    if lr_missing:
        logger.warning(f"LR features MISSING: {lr_missing}")

    # 4. Feature selection (remove high-correlation, keep top-k)
    logger.info("Selecting features …")
    feature_cols = [c for c in dataset.columns if c not in ("label", "return_pct")]
    X = dataset[feature_cols]
    y = dataset["label"]

    selected_data, preproc_meta = select_features(X, max_correlation=0.9, top_k=25)
    selected_data["label"] = y.values
    n_features = len([c for c in selected_data.columns if c != "label"])
    logger.info(f"Selected {n_features} features after correlation pruning")

    lr_selected = [c for c in LR_FEATURE_NAMES if c in selected_data.columns]
    lr_dropped  = [c for c in LR_FEATURE_NAMES if c not in selected_data.columns]
    logger.info(f"LR features kept after selection: {lr_selected}")
    if lr_dropped:
        logger.info(f"LR features dropped by correlation filter: {lr_dropped}")

    # 5. Time-based train / val / test split
    logger.info("Splitting data …")
    selected_data = selected_data.reset_index().rename(columns={"index": "timestamp"})
    train, val, test, split_meta = split_data(selected_data, 0.7, 0.15, 0.15)
    logger.info(f"  Train {len(train)} | Val {len(val)} | Test {len(test)}")

    # 6. Train XGBoost
    logger.info("Training XGBoost …")
    trainer  = XGBoostTrainer(model_dir=MODEL_DIR)
    datasets = {30: {"train": train, "val": val, "preprocessing_metadata": preproc_meta}}
    models   = trainer.train_models(datasets, time_horizons=[30], n_estimators=200,
                                    max_depth=6, learning_rate=0.05)

    model_data = models[30]
    model      = model_data["model"]
    metrics    = model_data["metrics"]
    importance = model_data["feature_importance"]

    # 7. SHAP analysis
    logger.info("Running SHAP TreeExplainer …")
    import shap

    # Use the model's own feature_names to guarantee column alignment
    feat_cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else [
        c for c in train.columns
        if c not in ("label", "timestamp", "return_pct", "trading_session",
                     "signal_direction", "time_horizon", "open", "high", "low", "close", "volume")
        and pd.api.types.is_numeric_dtype(train[c])
    ]
    X_test        = test[feat_cols].fillna(0).values
    feature_names = feat_cols

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Mean |SHAP| per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_series   = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)

    # 8. Build text report
    lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines += [
        f"Silver Bullet XGBoost — LR-Enriched Model  ({ts})",
        f"{'='*62}",
        f"Training data:  {len(train)} | Val: {len(val)} | Test: {len(test)}",
        f"Features:       {n_features} (incl. {len(lr_selected)} LR channel features)",
        f"",
        f"── Test-set metrics ─────────────────────────────────────────",
        f"  Accuracy:   {metrics['accuracy']:.4f}",
        f"  Precision:  {metrics['precision']:.4f}",
        f"  Recall:     {metrics['recall']:.4f}",
        f"  F1:         {metrics['f1']:.4f}",
        f"  ROC-AUC:    {metrics['roc_auc']:.4f}",
        f"",
        f"── SHAP feature importance (mean |SHAP|) ────────────────────",
    ]

    for rank, (feat, val_) in enumerate(shap_series.items(), 1):
        bar_len = int(val_ / shap_series.iloc[0] * 30)
        bar     = "█" * bar_len
        marker  = " ◄ LR" if feat in LR_FEATURE_NAMES else ""
        lines.append(f"  {rank:2d}. {feat:<35s} {val_:.4f}  {bar}{marker}")
        if rank >= 30:
            lines.append(f"  ... ({len(shap_series) - 30} more features omitted)")
            break

    lines += [
        f"",
        f"── LR channel features summary ──────────────────────────────",
    ]
    for feat in LR_FEATURE_NAMES:
        if feat in shap_series.index:
            rank  = list(shap_series.index).index(feat) + 1
            shval = shap_series[feat]
            xgb_imp = importance.get(feat, 0.0)
            lines.append(f"  {feat:<35s}  SHAP rank #{rank:2d}  |SHAP|={shval:.4f}  XGB_imp={xgb_imp:.4f}")
        else:
            lines.append(f"  {feat:<35s}  NOT in selected feature set")

    report = "\n".join(lines)
    print("\n" + report)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report + "\n")
    logger.info(f"Report saved → {REPORT_PATH}")

    # Save feature list for reference
    feature_list = {
        "selected_features": feat_cols,
        "lr_features_selected": lr_selected,
        "lr_features_dropped": lr_dropped,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "shap_top10": {k: float(v) for k, v in shap_series.head(10).items()},
    }
    json_path = Path(MODEL_DIR) / "30_minute" / "lr_enriched_metadata.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(feature_list, f, indent=2)
    logger.info(f"Feature metadata → {json_path}")


if __name__ == "__main__":
    main()
