"""Feature Importance Analyzer for XGBoost models.

Analyzes feature importance from trained XGBoost models, including
gain, weight, and cover importance types.

Performance: Completes in < 10 seconds.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # type: ignore

logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """Analyze feature importance from trained XGBoost model.

    Extracts importance scores (gain, weight, cover), ranks features,
    calculates cumulative importance, generates visualizations.

    Performance: Completes in < 10 seconds.
    """

    def __init__(
        self,
        model_path: str = "data/models/xgboost_classifier.pkl",
        importance_type: str = "gain",
        top_n: int = 20,
        output_directory: str = "data/reports"
    ):
        """Initialize feature importance analyzer.

        Args:
            model_path: Path to trained XGBoost model
            importance_type: Type of importance ('gain', 'weight', 'cover')
            top_n: Number of top features to visualize
            output_directory: Directory to save reports
        """
        self._model_path = Path(model_path)
        self._importance_type = importance_type
        self._top_n = top_n
        self._output_directory = Path(output_directory)

        # Create output directory
        self._output_directory.mkdir(parents=True, exist_ok=True)

        # Load XGBoost model
        self._model = None
        self._load_xgboost_model()

        logger.debug(
            f"FeatureImportanceAnalyzer initialized: "
            f"model_path={model_path}, "
            f"importance_type={importance_type}, "
            f"top_n={top_n}"
        )

    def _load_xgboost_model(self) -> None:
        """Load XGBoost model from pickle file.

        Raises:
            FileNotFoundError: If model file not found
            RuntimeError: If model doesn't have feature_importances
        """
        logger.debug(f"Loading XGBoost model from {self._model_path}...")

        try:
            with open(self._model_path, 'rb') as f:
                self._model = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model file not found: {self._model_path}"
            )

        # Validate model has feature_importances
        if not hasattr(self._model, 'feature_importances_'):
            raise RuntimeError(
                "Model does not have feature_importances_ attribute. "
                "Is this a valid XGBoost model?"
            )

        logger.debug("XGBoost model loaded successfully")

    def extract_importance_scores(self) -> dict[str, np.ndarray]:
        """Extract importance scores from XGBoost model.

        Returns:
            Dictionary with importance scores (normalized to 100%)
        """
        logger.debug("Extracting importance scores...")

        # Get raw importance scores
        raw_scores = self._model.feature_importances_

        # Normalize to percentages (sum = 100)
        normalized_scores = (raw_scores / raw_scores.sum()) * 100

        return {
            self._importance_type: normalized_scores
        }

    def rank_features(
        self,
        importance_scores: np.ndarray,
        feature_names: list[str]
    ) -> pd.DataFrame:
        """Rank features by importance (descending).

        Args:
            importance_scores: Array of importance scores
            feature_names: List of feature names

        Returns:
            DataFrame with ranked features
        """
        logger.debug("Ranking features by importance...")

        # Create DataFrame
        df = pd.DataFrame({
            'feature_name': feature_names,
            'importance_gain': importance_scores
        })

        # Sort by importance (descending)
        df = df.sort_values('importance_gain', ascending=False)

        # Add rank
        df['rank'] = range(1, len(df) + 1)

        return df.reset_index(drop=True)

    def calculate_cumulative_importance(
        self,
        ranked_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate cumulative importance percentage.

        Args:
            ranked_df: DataFrame with ranked features

        Returns:
            DataFrame with cumulative_pct column added
        """
        logger.debug("Calculating cumulative importance...")

        df = ranked_df.copy()

        # Calculate cumulative sum
        df['cumulative_pct'] = df['importance_gain'].cumsum()

        return df

    def generate_bar_chart(
        self,
        importance_df: pd.DataFrame
    ) -> plt.Figure:
        """Generate horizontal bar chart of top features.

        Args:
            importance_df: DataFrame with feature importance

        Returns:
            Matplotlib figure
        """
        logger.debug(f"Generating bar chart for top {self._top_n} features...")

        # Get top N features
        top_features = importance_df.head(self._top_n)

        # Reverse order for horizontal bar chart (most important at top)
        top_features = top_features.iloc[::-1]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

        # Create horizontal bar chart
        y_pos = np.arange(len(top_features))
        bars = ax.barh(
            y_pos,
            top_features['importance_gain'],
            color='steelblue',
            edgecolor='black'
        )

        # Set y-axis labels (feature names)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature_name'])
        ax.invert_yaxis()  # Highest importance at top

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_features['importance_gain'])):
            ax.text(
                value + 0.5,
                i,
                f'{value:.1f}%',
                va='center',
                fontsize=9
            )

        # Set labels and title
        ax.set_xlabel("Importance Gain (%)", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title(
            f"Feature Importance (Top {self._top_n} Features)",
            fontsize=14,
            fontweight='bold'
        )

        # Add grid
        ax.grid(True, axis='x', alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        return fig

    def create_importance_table(
        self,
        ranked_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create importance table with formatted columns.

        Args:
            ranked_df: DataFrame with ranked features

        Returns:
            DataFrame with formatted table
        """
        logger.debug("Creating importance table...")

        # Select and format columns
        table = pd.DataFrame({
            'feature_name': ranked_df['feature_name'],
            'importance_gain': ranked_df['importance_gain'].round(2),
            'cumulative_pct': ranked_df['cumulative_pct'].round(2)
        })

        # Add rank if available
        if 'rank' in ranked_df.columns:
            table.insert(0, 'rank', ranked_df['rank'])
        else:
            table.insert(0, 'rank', range(1, len(table) + 1))

        return table

    def save_results(
        self,
        fig: plt.Figure,
        table: pd.DataFrame
    ) -> tuple[str, str]:
        """Save chart and table to files.

        Args:
            fig: Matplotlib figure
            table: Importance table DataFrame

        Returns:
            Tuple of (png_file_path, csv_file_path)
        """
        logger.debug("Saving results...")

        # Generate filename with timestamp
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d")

        png_filename = f"feature_importance_{timestamp}.png"
        csv_filename = f"feature_importance_data_{timestamp}.csv"

        png_path = self._output_directory / png_filename
        csv_path = self._output_directory / csv_filename

        # Save figure
        fig.savefig(png_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

        # Save CSV
        table.to_csv(csv_path, index=False)

        logger.debug(f"Saved chart to {png_path}, data to {csv_path}")

        return str(png_path), str(csv_path)

    def analyze_feature_importance(
        self,
        feature_names: list[str] | None = None
    ) -> dict[str, Any]:
        """Perform complete feature importance analysis.

        Args:
            feature_names: List of feature names (optional, uses generic names if None)

        Returns:
            Dictionary with:
                - importance_df: DataFrame with full importance analysis
                - png_path: Path to saved chart
                - csv_path: Path to saved CSV table
        """
        logger.info("Starting feature importance analysis...")

        # Extract importance scores
        importance_dict = self.extract_importance_scores()
        importance_scores = importance_dict[self._importance_type]

        # Use generic feature names if not provided
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance_scores))]
            logger.warning("No feature names provided, using generic names")

        # Rank features
        ranked_df = self.rank_features(importance_scores, feature_names)

        # Calculate cumulative importance
        ranked_df = self.calculate_cumulative_importance(ranked_df)

        # Generate bar chart
        fig = self.generate_bar_chart(ranked_df)

        # Create importance table
        table = self.create_importance_table(ranked_df)

        # Save results
        png_path, csv_path = self.save_results(fig, table)

        logger.info(
            f"Feature importance analysis complete: "
            f"Top feature: {ranked_df.iloc[0]['feature_name']} "
            f"({ranked_df.iloc[0]['importance_gain']:.1f}%), "
            f"Top {self._top_n} features account for "
            f"{ranked_df.iloc[min(self._top_n-1, len(ranked_df)-1)]['cumulative_pct']:.1f}% "  # noqa: E501
            f"of importance"
        )

        return {
            'importance_df': ranked_df,
            'png_path': png_path,
            'csv_path': csv_path
        }
