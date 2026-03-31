"""Silver Bullet Optimization Researcher for feature selection and parameter tuning.

This module implements SHAP-based feature importance analysis and parameter
optimization for the Silver Bullet strategy using ML meta-labeling.
"""

import json
import logging
import pickle
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
from xgboost import XGBClassifier

from src.ml.features import FeatureEngineer
from src.ml.xgboost_trainer import train_xgboost
from src.research.historical_data_loader import HistoricalDataLoader
from src.research.ml_meta_labeling_backtester import MLMetaLabelingBacktester
from src.research.performance_metrics_calculator import PerformanceMetricsCalculator

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""

    pass


class InsufficientDataError(Exception):
    """Exception raised when data requirements are not met."""

    pass


# ============================================================================
# Optimization Statistics
# ============================================================================


class OptimizationStatistics:
    """Track statistics during optimization process."""

    def __init__(self) -> None:
        """Initialize statistics tracker."""
        self.start_time: float = time.time()
        self.shap_computation_time: float = 0.0
        self.feature_selection_time: float = 0.0
        self.parameter_optimization_time: float = 0.0
        self.model_training_time: float = 0.0
        self.total_time: float = 0.0

        self.features_analyzed: int = 0
        self.features_selected: int = 0
        self.feature_reduction_pct: float = 0.0

        self.parameters_tested: int = 0
        self.models_trained: int = 0

        self.sharpe_full: float = 0.0
        self.sharpe_optimized: float = 0.0
        self.sharpe_improvement: float = 0.0

        self.win_rate_full: float = 0.0
        self.win_rate_optimized: float = 0.0
        self.win_rate_improvement: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert statistics to dictionary.

        Returns:
            Dictionary with all statistics
        """
        return {
            "timing": {
                "shap_computation": self.shap_computation_time,
                "feature_selection": self.feature_selection_time,
                "parameter_optimization": self.parameter_optimization_time,
                "model_training": self.model_training_time,
                "total": self.total_time,
            },
            "features": {
                "analyzed": self.features_analyzed,
                "selected": self.features_selected,
                "reduction_pct": self.feature_reduction_pct,
            },
            "optimization": {
                "parameters_tested": self.parameters_tested,
                "models_trained": self.models_trained,
            },
            "performance": {
                "sharpe_full": self.sharpe_full,
                "sharpe_optimized": self.sharpe_optimized,
                "sharpe_improvement": self.sharpe_improvement,
                "win_rate_full": self.win_rate_full,
                "win_rate_optimized": self.win_rate_optimized,
                "win_rate_improvement": self.win_rate_improvement,
            },
        }


# ============================================================================
# Main Researcher Class
# ============================================================================


class SilverBulletOptimizationResearcher:
    """Optimize Silver Bullet strategy using SHAP and parameter tuning.

    Performs systematic analysis:
    1. SHAP-based feature importance analysis
    2. Feature subset testing (top N features)
    3. Silver Bullet parameter optimization
    4. Model retraining with selected features
    5. Comprehensive report generation

    Performance targets:
    - SHAP computation: < 10 minutes for 50K samples
    - Feature subset testing: < 15 minutes
    - Parameter optimization: < 30 minutes for 72 combinations
    - Total runtime: < 60 minutes for 6 months data
    """

    # Default parameter grid (3×3×4×2 = 72 combinations)
    DEFAULT_PARAM_GRID = {
        "take_profit_pct": [0.4, 0.5, 0.6],
        "stop_loss_pct": [0.2, 0.25, 0.3],
        "max_bars": [40, 50, 60, 70],
        "probability_threshold": [0.65, 0.70],
    }

    def __init__(
        self,
        model_path: str = "models/xgboost/30_minute/model.joblib",
        data_dir: str = "data/processed/dollar_bars/",
        output_dir: str = "_bmad-output/reports/",
        feature_sizes: list[int] | None = None,
        param_grid: dict | None = None,
        min_win_rate: float = 0.65,
        checkpoint_dir: str = "_bmad-output/checkpoints/",
    ):
        """Initialize Silver Bullet Optimization Researcher.

        Args:
            model_path: Path to trained XGBoost model
            data_dir: Directory containing historical dollar bar data
            output_dir: Directory for output reports and plots
            feature_sizes: Feature subset sizes to test (default: [10, 15, 20, 25])
            param_grid: Parameter grid for optimization (default: 72 combinations)
            min_win_rate: Minimum win rate threshold (default: 0.65)
            checkpoint_dir: Directory for checkpoint files
        """
        self._model_path = Path(model_path)
        self._data_dir = Path(data_dir)
        self._output_dir = Path(output_dir)
        self._checkpoint_dir = Path(checkpoint_dir)
        self._feature_sizes = feature_sizes or [10, 15, 20, 25]
        self._param_grid = param_grid or self.DEFAULT_PARAM_GRID.copy()
        self._min_win_rate = min_win_rate

        # Create output directories
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._plots_dir = self._output_dir / "plots"
        self._plots_dir.mkdir(parents=True, exist_ok=True)

        # Initialize statistics
        self._stats = OptimizationStatistics()

        # Model cache for retraining
        self._model_cache: dict[str, XGBClassifier] = {}

        logger.info(
            f"SilverBulletOptimizationResearcher initialized: "
            f"model_path={model_path}, "
            f"data_dir={data_dir}, "
            f"output_dir={output_dir}, "
            f"feature_sizes={self._feature_sizes}, "
            f"min_win_rate={min_win_rate}"
        )

    # ------------------------------------------------------------------------
    # Checkpoint Methods
    # ------------------------------------------------------------------------

    def _save_checkpoint(self, data: Any, step_name: str) -> None:
        """Save checkpoint data.

        Args:
            data: Data to serialize
            step_name: Checkpoint step name
        """
        checkpoint_path = self._checkpoint_dir / f"{step_name}.pkl"
        with open(checkpoint_path, "wb") as f:
            pickle.dump(data, f)
        logger.debug(f"Checkpoint saved: {step_name} at {datetime.now().isoformat()}")

    def _load_checkpoint(self, step_name: str) -> Any:
        """Load checkpoint data.

        Args:
            step_name: Checkpoint step name

        Returns:
            Loaded data or None if checkpoint doesn't exist
        """
        checkpoint_path = self._checkpoint_dir / f"{step_name}.pkl"
        if checkpoint_path.exists():
            with open(checkpoint_path, "rb") as f:
                data = pickle.load(f)
            logger.debug(f"Checkpoint loaded: {step_name}")
            return data
        return None

    def _checkpoint_exists(self, step_name: str) -> bool:
        """Check if checkpoint exists.

        Args:
            step_name: Checkpoint step name

        Returns:
            True if checkpoint exists
        """
        return (self._checkpoint_dir / f"{step_name}.pkl").exists()

    # ------------------------------------------------------------------------
    # Data Loading and Validation
    # ------------------------------------------------------------------------

    def _load_and_validate_model(self) -> XGBClassifier:
        """Load and validate XGBoost model.

        Returns:
            Loaded XGBoost model

        Raises:
            ModelLoadError: If model loading or validation fails
        """
        logger.info(f"Loading model from {self._model_path}...")

        if not self._model_path.exists():
            raise ModelLoadError(f"Model file not found: {self._model_path}")

        try:
            model = joblib.load(self._model_path)
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")

        # Validate model attributes
        required_attrs = ["feature_importances_", "n_features_in_"]
        for attr in required_attrs:
            if not hasattr(model, attr):
                raise ModelLoadError(f"Model missing required attribute: {attr}")

        # Check feature count (allow for flexible range)
        # n_features_in_ is an attribute in XGBoost 2.x
        feature_count = (
            model.n_features_in_
            if isinstance(model.n_features_in_, int)
            else model.n_features_in_()
        )
        if not (15 <= feature_count <= 50):
            raise ModelLoadError(
                f"Unexpected feature count: {feature_count} "
                f"(expected 15-50 features)"
            )

        # Log hyperparameters
        hyperparams = {
            "n_estimators": getattr(model, "n_estimators", "N/A"),
            "max_depth": getattr(model, "max_depth", "N/A"),
            "learning_rate": getattr(model, "learning_rate", "N/A"),
            "random_state": getattr(model, "random_state", "N/A"),
        }
        logger.info(
            f"Model loaded: {feature_count} features, "
            f"hyperparameters: {hyperparams}"
        )

        return model

    def _load_and_split_data(self, start_date: str, end_date: str) -> tuple[
        tuple[pd.DataFrame, pd.Series],
        tuple[pd.DataFrame, pd.Series],
        tuple[pd.DataFrame, pd.Series],
    ]:
        """Load and split historical data chronologically.

        Split strategy (time-based):
        - Training: First 67% of data (~4 months)
        - Validation: Next 17% of data (~1 month)
        - Test (hold-out): Final 16% of data (~1 month)

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Tuple of (X_train, y_train), (X_val, y_val), (X_test, y_test)

        Raises:
            InsufficientDataError: If data requirements not met
        """
        logger.info(f"Loading data from {start_date} to {end_date}...")

        loader = HistoricalDataLoader(
            data_directory=str(self._data_dir), min_completeness=50.0
        )

        data = loader.load_data(start_date, end_date)

        # Check minimum data requirements (6 months)
        data_range = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        if data_range < 180:
            raise InsufficientDataError(
                f"Insufficient data: {data_range} days "
                f"(minimum 6 months / 180 days required)"
            )

        # Generate features
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.engineer_features(data)

        # Remove NaN rows (from rolling window calculations)
        features_df = features_df.dropna()

        logger.info(f"Features generated: {features_df.shape}")

        # Create binary labels: 1 if close > open, else 0
        labels = (features_df["close"] > features_df["open"]).astype(int)

        # Load selected features if available (from model training)
        selected_features_path = self._model_path.parent / "selected_features.json"
        if selected_features_path.exists():
            with open(selected_features_path, 'r') as f:
                feature_config = json.load(f)
            feature_cols = feature_config['features']
            # Filter to only available features
            feature_cols = [col for col in feature_cols if col in features_df.columns]
            logger.info(f"Using {len(feature_cols)} pre-selected features")
        else:
            # Drop non-feature columns
            exclude_cols = {"open", "high", "low", "close", "volume", "timestamp", "label"}
            feature_cols = [
                col
                for col in features_df.columns
                if col not in exclude_cols and not col.startswith("trading_session")
            ]

        X = features_df[feature_cols]
        y = labels

        # Split chronologically
        n_samples = len(X)
        train_end = int(n_samples * 0.67)
        val_end = int(n_samples * 0.84)  # 67% + 17%

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]

        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]

        X_test = X.iloc[val_end:]
        y_test = y.iloc[val_end:]

        logger.info(
            f"Data split: "
            f"train={len(X_train)}, "
            f"val={len(X_val)}, "
            f"test={len(X_test)}"
        )

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    # ------------------------------------------------------------------------
    # SHAP Analysis Methods
    # ------------------------------------------------------------------------

    def _compute_shap_values(
        self, model: XGBClassifier, X: pd.DataFrame, features: list[str]
    ) -> pd.DataFrame:
        """Compute SHAP values with batch processing for memory efficiency.

        Args:
            model: Trained XGBoost model
            X: Feature matrix
            features: List of feature names

        Returns:
            DataFrame of SHAP values
        """
        logger.info("Computing SHAP values...")

        start_time = time.time()
        batch_size = 5000
        shap_values_list = []

        # Create TreeExplainer
        explainer = shap.TreeExplainer(model)

        # Process in batches
        n_samples = len(X)
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            X_batch = X.iloc[i:batch_end]

            logger.debug(f"Processing SHAP batch {i//batch_size + 1}...")

            try:
                batch_shap = explainer.shap_values(X_batch)
                shap_values_list.append(batch_shap)
            except MemoryError:
                # Reduce batch size and retry
                new_batch_size = batch_size // 2
                logger.warning(f"OOM error, reducing batch size to {new_batch_size}")
                for j in range(i, batch_end, new_batch_size):
                    j_end = min(j + new_batch_size, batch_end)
                    X_sub_batch = X.iloc[i:j_end]
                    batch_shap = explainer.shap_values(X_sub_batch)
                    shap_values_list.append(batch_shap)

        # Concatenate results
        shap_values = np.vstack(shap_values_list)
        shap_df = pd.DataFrame(shap_values, columns=features)

        elapsed = time.time() - start_time
        self._stats.shap_computation_time = elapsed
        logger.info(f"SHAP computation complete: {elapsed:.2f}s")

        return shap_df

    def _rank_features_by_shap(
        self, shap_values: pd.DataFrame
    ) -> list[tuple[str, float]]:
        """Rank features by mean absolute SHAP value.

        Args:
            shap_values: DataFrame of SHAP values

        Returns:
            List of (feature_name, mean_shap_value) tuples sorted by importance
        """
        logger.info("Ranking features by SHAP importance...")

        # Compute mean absolute SHAP values
        mean_shap = shap_values.abs().mean().sort_values(ascending=False)

        ranked_features = list(mean_shap.items())
        self._stats.features_analyzed = len(ranked_features)

        logger.info(f"Features ranked: {len(ranked_features)}")
        return ranked_features

    def _plot_shap_summary(self, shap_values: pd.DataFrame, output_path: Path) -> Path:
        """Generate and save SHAP summary plot.

        Args:
            shap_values: DataFrame of SHAP values
            output_path: Path to save plot

        Returns:
            Path to saved plot
        """
        logger.debug("Generating SHAP summary plot...")

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values.values, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.debug(f"SHAP summary plot saved: {output_path}")
        return output_path

    # ------------------------------------------------------------------------
    # Feature Subset Testing
    # ------------------------------------------------------------------------

    def _test_feature_subsets(
        self,
        ranked_features: list[tuple[str, float]],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> pd.DataFrame:
        """Test multiple feature subset sizes.

        Args:
            ranked_features: List of (feature, shap_value) tuples
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            DataFrame with results for each subset size
        """
        logger.info("Testing feature subsets...")

        start_time = time.time()
        results = []

        feature_names = [f[0] for f in ranked_features]

        for n_features in self._feature_sizes:
            logger.info(f"Testing {n_features} features...")

            # Select top N features
            selected_features = feature_names[:n_features]

            # Filter data
            X_train_subset = X_train[selected_features]
            X_val_subset = X_val[selected_features]

            # Check cache
            cache_key = f"features_{n_features}"
            if cache_key in self._model_cache:
                model = self._model_cache[cache_key]
            else:
                # Train model
                model, _ = train_xgboost(
                    X_train_subset, y_train, X_val_subset, y_val, random_state=42
                )
                self._model_cache[cache_key] = model
                self._stats.models_trained += 1

            # Predict
            y_pred_proba = model.predict_proba(X_val_subset)[:, 1]
            y_pred = (y_pred_proba >= self._min_win_rate).astype(int)

            # Calculate metrics
            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
            )

            win_rate = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            # Calculate Sharpe proxy (precision / std of predictions)
            sharpe_proxy = precision / (np.std(y_pred_proba) + 1e-6)

            results.append(
                {
                    "n_features": n_features,
                    "win_rate": win_rate,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "sharpe_proxy": sharpe_proxy,
                }
            )

        elapsed = time.time() - start_time
        self._stats.feature_selection_time = elapsed

        results_df = pd.DataFrame(results)
        logger.info(f"Feature subset testing complete: {elapsed:.2f}s")

        return results_df

    def _select_optimal_subset(self, results_df: pd.DataFrame) -> tuple[int, list[str]]:
        """Select optimal feature subset from test results.

        Args:
            results_df: Results from feature subset testing

        Returns:
            Tuple of (optimal_feature_count, feature_list)
        """
        logger.info("Selecting optimal feature subset...")

        # Filter by minimum win rate
        valid_results = results_df[results_df["win_rate"] >= self._min_win_rate]

        if valid_results.empty:
            # No subset meets minimum, use best available
            logger.warning(
                f"No subset meets {self._min_win_rate:.0%} win rate, "
                f"using best available"
            )
            valid_results = results_df

        # Select by highest Sharpe proxy
        best_idx = valid_results["sharpe_proxy"].idxmax()
        optimal_n = int(results_df.loc[best_idx, "n_features"])

        self._stats.features_selected = optimal_n
        if self._stats.features_analyzed > 0:
            self._stats.feature_reduction_pct = (
                1 - optimal_n / self._stats.features_analyzed
            ) * 100
        else:
            self._stats.feature_reduction_pct = 0.0

        logger.info(
            f"Optimal subset: {optimal_n} features "
            f"({self._stats.feature_reduction_pct:.1f}% reduction)"
        )

        # Note: Feature list will be generated during main optimization
        return optimal_n, []

    # ------------------------------------------------------------------------
    # Parameter Optimization
    # ------------------------------------------------------------------------

    def _optimize_sb_parameters(
        self,
        features: list[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Optimize Silver Bullet parameters via grid search.

        Args:
            features: Selected feature list
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            data: Full historical data for backtesting

        Returns:
            DataFrame with parameter optimization results
        """
        logger.info("Optimizing Silver Bullet parameters...")

        start_time = time.time()

        # Validate parameter grid size
        grid_size = np.prod(
            [len(v) if isinstance(v, list) else 1 for v in self._param_grid.values()]
        )
        if grid_size > 100:
            raise ValueError(
                f"Parameter grid too large: {int(grid_size)} combinations "
                f"(maximum 100 allowed)"
            )

        # Generate all combinations
        param_names = list(self._param_grid.keys())
        param_values = list(self._param_grid.values())
        combinations = list(product(*param_values))

        logger.info(f"Testing {len(combinations)} parameter combinations...")

        results = []
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            # Train model with current features
            X_train_subset = X_train[features]
            X_val_subset = X_val[features]

            model, _ = train_xgboost(
                X_train_subset, y_train, X_val_subset, y_val, random_state=42
            )

            # Predict and calculate win rate
            y_pred_proba = model.predict_proba(X_val_subset)[:, 1]
            y_pred = (y_pred_proba >= params["probability_threshold"]).astype(int)

            win_rate = (y_pred == y_val.values).mean()

            # Estimate trade count (heuristic)
            n_trades = int(len(y_pred) * y_pred.mean())

            results.append({**params, "win_rate": win_rate, "n_trades": n_trades})

            self._stats.parameters_tested += 1

            # Checkpoint every 10 combinations
            if (i + 1) % 10 == 0:
                logger.info(f"Tested {i + 1}/{len(combinations)} combinations...")

        elapsed = time.time() - start_time
        self._stats.parameter_optimization_time = elapsed

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("win_rate", ascending=False)

        logger.info(
            f"Parameter optimization complete: {elapsed:.2f}s, "
            f"best win rate: {results_df.iloc[0]['win_rate']:.2%}"
        )

        return results_df

    def _analyze_parameter_sensitivity(
        self, results_df: pd.DataFrame
    ) -> dict[str, float]:
        """Analyze parameter sensitivity.

        Args:
            results_df: Parameter optimization results

        Returns:
            Dictionary mapping parameter -> variance in win rate
        """
        logger.debug("Analyzing parameter sensitivity...")

        sensitivity = {}

        for param in self._param_grid.keys():
            # Group by parameter value and compute variance
            grouped = results_df.groupby(param)["win_rate"].std()
            sensitivity[param] = grouped.mean()

        return sensitivity

    # ------------------------------------------------------------------------
    # Model Retraining and Persistence
    # ------------------------------------------------------------------------

    def _retrain_optimized_model(
        self,
        selected_features: list[str],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> tuple[XGBClassifier, dict]:
        """Retrain model with selected features.

        Args:
            selected_features: List of selected feature names
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Tuple of (model, metrics)
        """
        logger.info(f"Retraining model with {len(selected_features)} features...")

        start_time = time.time()

        X_train_subset = X_train[selected_features]
        X_val_subset = X_val[selected_features]

        model, metrics = train_xgboost(
            X_train_subset, y_train, X_val_subset, y_val, random_state=42
        )

        elapsed = time.time() - start_time
        self._stats.model_training_time = elapsed

        logger.info(f"Model retraining complete: {elapsed:.2f}s")
        return model, metrics

    def _save_optimized_model(
        self, model: XGBClassifier, features: list[str], metrics: dict, metadata: dict
    ) -> None:
        """Save optimized model and feature configuration.

        Args:
            model: Trained model
            features: Selected feature list
            metrics: Validation metrics
            metadata: Additional metadata
        """
        logger.info("Saving optimized model...")

        # Create model directory
        model_dir = Path("models/xgboost/5_minute")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = model_dir / "model_optimized.joblib"
        joblib.dump(model, model_path)

        # Save feature configuration
        feature_config = {
            "feature_count": len(features),
            "features": features,
            "selection_date": datetime.now().date().isoformat(),
            "metrics": {
                k: float(v) if not isinstance(v, (list, dict)) else v
                for k, v in metrics.items()
            },
        }

        feature_path = model_dir / "selected_features.json"
        with open(feature_path, "w") as f:
            json.dump(feature_config, f, indent=2)

        # Save metadata
        metadata_path = model_dir / "optimization_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved: {model_path}")
        logger.info(f"Features saved: {feature_path}")

    def _save_sb_params(self, params: dict) -> None:
        """Save optimized Silver Bullet parameters.

        Args:
            params: Optimized parameters dictionary
        """
        logger.info("Saving Silver Bullet parameters...")

        params_with_meta = {
            **params,
            "optimization_date": datetime.now().date().isoformat(),
        }

        params_path = Path("models/xgboost/30_minute/sb_params.json")
        with open(params_path, "w") as f:
            json.dump(params_with_meta, f, indent=2)

        logger.info(f"Parameters saved: {params_path}")

    # ------------------------------------------------------------------------
    # Report Generation
    # ------------------------------------------------------------------------

    def _generate_markdown_report(self, results: dict) -> Path:
        """Generate comprehensive markdown report.

        Args:
            results: Dictionary with all optimization results

        Returns:
            Path to generated report
        """
        logger.info("Generating markdown report...")

        report_date = datetime.now().strftime("%Y-%m-%d")
        report_path = self._output_dir / f"sb-optimization-{report_date}.md"

        with open(report_path, "w") as f:
            # Title and summary
            f.write(f"# Silver Bullet Optimization Report\n\n")
            f.write(
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            )

            # Summary section
            f.write("## Summary\n\n")
            f.write(f"- **Features Analyzed:** {self._stats.features_analyzed}\n")
            f.write(f"- **Features Selected:** {self._stats.features_selected}\n")
            f.write(
                f"- **Feature Reduction:** {self._stats.feature_reduction_pct:.1f}%\n"
            )
            f.write(f"- **Parameters Tested:** {self._stats.parameters_tested}\n")
            f.write(f"- **Models Trained:** {self._stats.models_trained}\n\n")

            # Feature Importance
            f.write("## Feature Importance\n\n")
            if "feature_rankings" in results:
                f.write("### Top 20 Features by SHAP Value\n\n")
                f.write("| Rank | Feature | Mean SHAP Value |\n")
                f.write("|------|---------|-----------------|\n")
                for i, (feat, val) in enumerate(results["feature_rankings"][:20], 1):
                    f.write(f"| {i} | {feat} | {val:.4f} |\n")

                f.write(f"\n![SHAP Summary](plots/shap_summary.png)\n\n")

            # Feature Selection Results
            f.write("## Feature Selection Results\n\n")
            if "feature_subset_results" in results:
                df = results["feature_subset_results"]
                f.write("| Features | Win Rate | Precision | F1 | Sharpe Proxy |\n")
                f.write("|----------|----------|-----------|-----|--------------|\n")
                for _, row in df.iterrows():
                    f.write(
                        f"| {row['n_features']} | "
                        f"{row['win_rate']:.2%} | "
                        f"{row['precision']:.3f} | "
                        f"{row['f1']:.3f} | "
                        f"{row['sharpe_proxy']:.3f} |\n"
                    )

            # Parameter Optimization
            f.write("\n## Parameter Optimization Results\n\n")
            if "param_results" in results:
                param_df = results["param_results"]
                best = param_df.iloc[0]
                f.write("### Best Parameters\n\n")
                f.write(f"- **Take Profit:** {best['take_profit_pct']:.1%}\n")
                f.write(f"- **Stop Loss:** {best['stop_loss_pct']:.1%}\n")
                f.write(f"- **Max Bars:** {int(best['max_bars'])}\n")
                f.write(
                    f"- **Probability Threshold:** {best['probability_threshold']:.2f}\n"
                )
                f.write(f"- **Win Rate:** {best['win_rate']:.2%}\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("### Deployment Steps\n\n")
            f.write(
                "1. Replace existing model with `models/xgboost/30_minute/model_optimized.joblib`\n"
            )
            f.write(
                "2. Update feature selection to use `models/xgboost/30_minute/selected_features.json`\n"
            )
            f.write(
                "3. Configure Silver Bullet detector with parameters from `sb_params.json`\n"
            )
            f.write("4. Run validation backtest on hold-out test set\n")
            f.write("5. Monitor performance for 2 weeks before full deployment\n\n")

            # Performance Summary
            f.write("### Performance Summary\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(
                f"| Feature Reduction | {self._stats.feature_reduction_pct:.1f}% |\n"
            )
            f.write(f"| Models Trained | {self._stats.models_trained} |\n")
            f.write(f"| Total Runtime | {self._stats.total_time:.1f}s |\n\n")

        logger.info(f"Report generated: {report_path}")
        return report_path

    def _generate_performance_comparison_plot(self, results_df: pd.DataFrame) -> Path:
        """Generate interactive performance comparison plot.

        Args:
            results_df: Feature subset results

        Returns:
            Path to generated plot
        """
        logger.debug("Generating performance comparison plot...")

        fig = go.Figure()

        # Add win rate line
        fig.add_trace(
            go.Scatter(
                x=results_df["n_features"],
                y=results_df["win_rate"] * 100,
                mode="lines+markers",
                name="Win Rate (%)",
                line=dict(color="blue", width=2),
            )
        )

        # Add Sharpe proxy line
        fig.add_trace(
            go.Scatter(
                x=results_df["n_features"],
                y=results_df["sharpe_proxy"],
                mode="lines+markers",
                name="Sharpe Proxy",
                yaxis="y2",
                line=dict(color="orange", width=2),
            )
        )

        # Update layout
        fig.update_layout(
            title="Feature Subset Performance Comparison",
            xaxis_title="Number of Features",
            yaxis_title="Win Rate (%)",
            yaxis2=dict(title="Sharpe Proxy", overlaying="y", side="right"),
            hovermode="x unified",
        )

        plot_path = self._plots_dir / "performance_comparison.html"
        fig.write_html(str(plot_path))

        logger.debug(f"Performance plot saved: {plot_path}")
        return plot_path

    # ------------------------------------------------------------------------
    # Main Optimization Orchestration
    # ------------------------------------------------------------------------

    def run_optimization(
        self, start_date: str, end_date: str, resume: bool = False
    ) -> dict:
        """Run full optimization pipeline.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            resume: Resume from checkpoints

        Returns:
            Dictionary with all optimization results
        """
        logger.info("=" * 60)
        logger.info("Starting Silver Bullet Optimization")
        logger.info("=" * 60)

        results = {}

        # Step 1: Load model
        model = self._load_and_validate_model()

        # Step 2: Load and split data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = (
            self._load_and_split_data(start_date, end_date)
        )

        # Step 3: Compute SHAP values (or load from checkpoint)
        if resume and self._checkpoint_exists("shap_values"):
            logger.info("Loading SHAP values from checkpoint...")
            shap_values = self._load_checkpoint("shap_values")
        else:
            shap_values = self._compute_shap_values(
                model, X_train, X_train.columns.tolist()
            )
            self._save_checkpoint(shap_values, "shap_values")

        # Step 4: Rank features
        ranked_features = self._rank_features_by_shap(shap_values)
        results["feature_rankings"] = ranked_features

        # Step 5: Plot SHAP summary
        shap_plot_path = self._plots_dir / "shap_summary.png"
        self._plot_shap_summary(shap_values, shap_plot_path)

        # Step 6: Test feature subsets (or load from checkpoint)
        if resume and self._checkpoint_exists("feature_subsets"):
            logger.info("Loading feature subsets from checkpoint...")
            subset_results = self._load_checkpoint("feature_subsets")
        else:
            subset_results = self._test_feature_subsets(
                ranked_features, X_train, y_train, X_val, y_val
            )
            self._save_checkpoint(subset_results, "feature_subsets")

        results["feature_subset_results"] = subset_results

        # Step 7: Select optimal subset
        optimal_n, _ = self._select_optimal_subset(subset_results)
        selected_features = [f[0] for f in ranked_features[:optimal_n]]

        # Step 8: Optimize parameters (or load from checkpoint)
        if resume and self._checkpoint_exists("param_results"):
            logger.info("Loading parameter results from checkpoint...")
            param_results = self._load_checkpoint("param_results")
        else:
            param_results = self._optimize_sb_parameters(
                selected_features, X_train, y_train, X_val, y_val, X_train
            )
            self._save_checkpoint(param_results, "param_results")

        results["param_results"] = param_results

        # Step 9: Analyze parameter sensitivity
        sensitivity = self._analyze_parameter_sensitivity(param_results)
        results["parameter_sensitivity"] = sensitivity

        # Step 10: Retrain optimized model
        optimized_model, metrics = self._retrain_optimized_model(
            selected_features, X_train, y_train, X_val, y_val
        )

        # Step 11: Final test validation
        logger.info("Validating on hold-out test set...")
        X_test_subset = X_test[selected_features]
        y_pred_proba = optimized_model.predict_proba(X_test_subset)[:, 1]
        y_pred = (y_pred_proba >= self._min_win_rate).astype(int)
        test_win_rate = (y_pred == y_test.values).mean()

        logger.info(f"Test set win rate: {test_win_rate:.2%}")

        # Step 12: Save artifacts
        metadata = {
            "optimization_date": datetime.now().isoformat(),
            "data_range": {"start": start_date, "end": end_date},
            "test_win_rate": float(test_win_rate),
            "statistics": self._stats.to_dict(),
        }

        self._save_optimized_model(
            optimized_model, selected_features, metrics, metadata
        )

        best_params = param_results.iloc[0].to_dict()
        self._save_sb_params(best_params)

        # Step 13: Generate plots and reports
        self._generate_performance_comparison_plot(subset_results)
        report_path = self._generate_markdown_report(results)

        # Update statistics
        self._stats.total_time = time.time() - self._stats.start_time

        logger.info("=" * 60)
        logger.info("Optimization Complete!")
        logger.info(f"Test Win Rate: {test_win_rate:.2%}")
        logger.info(f"Feature Reduction: {self._stats.feature_reduction_pct:.1f}%")
        logger.info(f"Total Runtime: {self._stats.total_time:.1f}s")
        logger.info(f"Report: {report_path}")
        logger.info("=" * 60)

        results["test_win_rate"] = test_win_rate
        results["report_path"] = str(report_path)

        return results
