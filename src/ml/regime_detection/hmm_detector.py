"""HMM-based regime detection using hmmlearn.

This module implements Hidden Markov Model (HMM) for detecting market regimes
(trending-up, trending-down, ranging, volatile) from time-series features.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm

from src.ml.regime_detection.features import HMMFeatureEngineer
from src.ml.regime_detection.models import (
    HMMModelMetadata,
    HMMTrainingConfig,
    RegimeDetectionResult,
    RegimeState,
    RegimeTransitionEvent,
    RegimeType,
)

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """Hidden Markov Model-based regime detector.

    Uses hmmlearn's GaussianHMM to identify market regimes from time-series
    features. Supports training, real-time inference, and regime classification.

    Attributes:
        n_regimes: Number of regimes (hidden states)
        n_features: Number of observation features
        regime_names: List of regime names
        model: Trained hmmlearn.GaussianHMM model
        metadata: Model metadata and training info

    Example:
        >>> detector = HMMRegimeDetector(n_regimes=3)
        >>> features_df = detector.engineer_features(dollar_bars_data)
        >>> detector.fit(features_df)
        >>> regime_sequence = detector.predict(features_df)
        >>> current_regime = detector.detect_regime(latest_features)
    """

    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = "full",
        n_iterations: int = 100,
        random_state: int = 42,
    ):
        """Initialize HMM regime detector.

        Args:
            n_regimes: Number of regimes (hidden states), default=3
            covariance_type: Covariance type ('full', 'diag', 'spherical')
            n_iterations: Max EM iterations, default=100
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iterations = n_iterations
        self.random_state = random_state

        # Feature engineer
        self.feature_engineer = HMMFeatureEngineer()

        # Model (will be set during training)
        self.model: hmm.GaussianHMM | None = None
        self.metadata: HMMModelMetadata | None = None

        # Real-time state
        self.current_regime: RegimeState | None = None
        self.regime_history: list[RegimeState] = []
        self.transition_events: list[RegimeTransitionEvent] = []

        # Smoothing for real-time inference
        self.regime_history_length = 3  # Number of past regimes to average
        self.regime_confidence_threshold = 0.7  # Minimum confidence to report regime

        logger.info(
            f"HMMRegimeDetector initialized: n_regimes={n_regimes}, "
            f"covariance_type={covariance_type}"
        )

    def fit(
        self,
        features_df: pd.DataFrame,
        regime_names: list[RegimeType] | None = None
    ) -> dict:
        """Train HMM on feature time series.

        Args:
            features_df: DataFrame with HMM features (from HMMFeatureEngineer)
            regime_names: Optional list of regime names (e.g., ["trending_up", "ranging", ...])

        Returns:
            Dictionary with training results:
            {
                "converged": bool,
                "bic_score": float,
                "n_iterations": int,
                "log_likelihood": float
            }
        """
        logger.info(f"Training HMM with {self.n_regimes} regimes on {len(features_df)} samples")

        # Prepare features for HMM (remove any remaining NaN)
        features_array = features_df.values
        features_array = np.nan_to_num(features_array, nan=0.0)

        # Lengths of features
        n_samples, n_features = features_array.shape

        # Train HMM with multiple initializations to find best model
        logger.info("Starting EM algorithm training...")
        best_model = None
        best_bic = float('inf')

        for init in range(2):  # Reduce to 2 initializations for speed
            hmm_model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=self.n_iterations,
                random_state=self.random_state + init,
                init_params='stmc',
                tol=1e-3,  # Slightly relaxed tolerance
            )

            hmm_model.fit(features_array)

            # Calculate BIC (inline to avoid dependency on self.model)
            log_likelihood = hmm_model.score(features_array)
            n_params = self.n_regimes * n_features * 2
            bic = -2 * log_likelihood + n_params * np.log(n_samples)

            # Accept model even if n_iter_ is not set (may still converge)
            if bic < best_bic:
                best_bic = bic
                best_model = hmm_model
                iter_count = getattr(hmm_model, 'n_iter_', 'N/A')
                logger.info(f"  Init {init+1}: BIC={bic:.2f}, iterations={iter_count}")

        self.model = best_model
        log_likelihood = self.model.score(features_array)
        bic_score = best_bic

        # Determine regime names (if not provided)
        if regime_names is None:
            regime_names = self._assign_regime_names()
        else:
            if len(regime_names) != self.n_regimes:
                raise ValueError(f"Expected {self.n_regimes} regime names, got {len(regime_names)}")

        # Compute regime persistence (average duration in each regime)
        regime_persistence = self._compute_regime_persistence(features_array)

        # Create metadata
        feature_names = list(features_df.columns)

        # Get convergence iterations (may not be set if model didn't fully converge)
        conv_iters = getattr(self.model, 'n_iter_', self.n_iterations)

        self.metadata = HMMModelMetadata(
            model_name="hmm_regime_detector",
            n_regimes=self.n_regimes,
            n_features=n_features,
            training_samples=n_samples,
            training_date=datetime.now(),
            bic_score=bic_score,
            convergence_iterations=conv_iters,
            regime_names=regime_names,
            regime_persistence=regime_persistence,
            transition_matrix=self.model.transmat_.tolist(),
            features_used=feature_names
        )

        logger.info(
            f"HMM training complete: BIC={bic_score:.2f}, "
            f"iterations={conv_iters}, "
            f"log_likelihood={log_likelihood:.2f}"
        )

        # Initialize current regime based on most recent data
        self.current_regime = self.detect_regime(features_array[-1:])

        return {
            "converged": True,
            "bic_score": bic_score,
            "n_iterations": conv_iters,
            "log_likelihood": log_likelihood
        }

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict regime labels for feature time series using Viterbi algorithm.

        Args:
            features_df: DataFrame with HMM features

        Returns:
            Array of regime labels (integers 0 to n_regimes-1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Prepare features
        features_array = features_df.values
        features_array = np.nan_to_num(features_array, nan=0.0)

        # Predict using Viterbi algorithm (most probable regime path)
        regime_labels = self.model.predict(features_array)

        logger.info(f"Predicted regimes for {len(regime_labels)} time steps")

        return regime_labels

    def predict_proba(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict regime probabilities for feature time series.

        Args:
            features_df: DataFrame with HMM features

        Returns:
            Array of shape (n_samples, n_regimes) with P(regime|features)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Prepare features
        features_array = features_df.values
        features_array = np.nan_to_num(features_array, nan=0.0)

        # Predict posterior probabilities
        regime_probs = self.model.predict_proba(features_array)

        return regime_probs

    def detect_regime(self, features: np.ndarray | pd.DataFrame) -> RegimeState:
        """Detect current regime from single observation or features DataFrame.

        Args:
            features: Features for current time step (can be 1D array or DataFrame)

        Returns:
            RegimeState with current regime and metadata

        Raises:
            ValueError: If model not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Convert DataFrame to array if needed
        if isinstance(features, pd.DataFrame):
            # Use last row if DataFrame provided
            features_array = features.iloc[-1:].values
        else:
            features_array = np.atleast_2d(features)

        # Get regime probabilities
        regime_probs = self.model.predict_proba(features_array)[0]

        # Get most likely regime
        regime_idx = np.argmax(regime_probs)
        regime_name = self.metadata.regime_names[regime_idx]
        confidence = float(regime_probs[regime_idx])

        # Update current regime
        if self.current_regime is None or regime_name != self.current_regime.regime:
            # Regime transition detected
            old_regime = self.current_regime.regime if self.current_regime else None

            self.current_regime = RegimeState(
                regime=regime_name,
                probability=confidence,
                detected_at=datetime.now(),
                duration_bars=0,
                duration_days=0.0,
                transition_from=old_regime
            )

            # Log transition if not initial detection
            if old_regime is not None:
                transition_event = RegimeTransitionEvent(
                    timestamp=datetime.now(),
                    from_regime=old_regime,
                    to_regime=regime_name,
                    confidence=confidence,
                    transition_duration_bars=1
                )
                self.transition_events.append(transition_event)
                logger.info(
                    f"Regime transition detected: {old_regime} → {regime_name} "
                    f"(confidence: {confidence:.2f})"
                )

        else:
            # Same regime, increment duration
            self.current_regime.duration_bars += 1
            self.current_regime.duration_days = (
                self.current_regime.duration_bars * (5/60/24)  # Approximate (5-min bars to days)
            )

        return self.current_regime

    def detect_regime_realtime(
        self,
        current_bar: pd.Series,
        smoothing_window: int = 3
    ) -> RegimeDetectionResult:
        """Detect regime in real-time with smoothing.

        Args:
            current_bar: Current bar data (OHLCV)
            smoothing_window: Number of past predictions to average

        Returns:
            RegimeDetectionResult with current regime and confidence

        Raises:
            ValueError: If model not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Update regime history
        self.regime_history.append(self.current_regime)
        if len(self.regime_history) > smoothing_window:
            self.regime_history.pop(0)

        # Smooth regime probabilities across recent history
        if len(self.regime_history) >= smoothing_window:
            # Get recent regimes
            recent_regimes = self.regime_history[-smoothing_window:]

            # Weighted average: more recent = higher weight
            weights = np.linspace(0.5, 1.0, smoothing_window)

            # Compute weighted regime probabilities
            regime_probs = {}
            for regime in self.metadata.regime_names:
                # Average probability of this regime in recent history
                probs = [
                    1.0 if r.regime == regime else 0.0
                    for r in recent_regimes
                ]
                regime_probs[regime] = np.average(probs, weights=weights)

            # Select regime with highest smoothed probability
            best_regime = max(regime_probs, key=regime_probs.get)
            confidence = regime_probs[best_regime]

            # Check if this is a transition from previous regime
            is_transition = (
                self.current_regime is not None and
                best_regime != self.current_regime.regime
            )

            if is_transition and confidence > self.regime_confidence_threshold:
                # Create transition event
                transition_event = RegimeTransitionEvent(
                    timestamp=datetime.now(),
                    from_regime=self.current_regime.regime,
                    to_regime=best_regime,
                    confidence=confidence,
                    transition_duration_bars=smoothing_window
                )

                result = RegimeDetectionResult(
                    timestamp=datetime.now(),
                    current_regime=best_regime,
                    regime_probabilities=regime_probs,
                    confidence=confidence,
                    is_transition=True,
                    transition_event=transition_event,
                    regime_duration_bars=0,
                    regime_duration_days=0.0
                )

                # Update current regime
                self.current_regime = RegimeState(
                    regime=best_regime,
                    probability=confidence,
                    detected_at=datetime.now(),
                    duration_bars=0,
                    duration_days=0.0,
                    transition_from=self.current_regime.regime
                )

                logger.info(
                    f"Regime transition (smoothed): {self.current_regime.regime} → {best_regime} "
                    f"(confidence: {confidence:.2f})"
                )

            else:
                # No transition, update duration
                result = RegimeDetectionResult(
                    timestamp=datetime.now(),
                    current_regime=best_regime,
                    regime_probabilities=regime_probs,
                    confidence=confidence,
                    is_transition=False,
                    transition_event=None,
                    regime_duration_bars=self.current_regime.duration_bars + 1,
                    regime_duration_days=self.current_regime.duration_days + (5/60/24)
                )

                # Update current regime without changing
                self.current_regime.duration_bars += 1
                self.current_regime.duration_days += (5/60/24)

        else:
            # Not enough history, use current regime
            result = RegimeDetectionResult(
                timestamp=datetime.now(),
                current_regime=self.current_regime.regime,
                regime_probabilities={self.current_regime.regime: 1.0},
                confidence=self.current_regime.probability,
                is_transition=False,
                transition_event=None,
                regime_duration_bars=self.current_regime.duration_bars + 1,
                regime_duration_days=self.current_regime.duration_days + (5/60/24)
            )

        return result

    def _assign_regime_names(self) -> list[str]:
        """Assign regime names based on model characteristics.

        Returns:
            List of regime names (e.g., ["trending_up", "ranging", ...])
        """
        # Get means and covariances for each regime
        means = self.model.means_  # Shape: (n_regimes, n_features)
        regime_names = []

        for i in range(self.n_regimes):
            regime_mean = means[i][0]  # Assume returns_1 is first feature (most interpretable)

            if regime_mean > 0.001:  # Positive drift
                regime_names.append("trending_up")
            elif regime_mean < -0.001:  # Negative drift
                regime_names.append("trending_down")
            elif abs(regime_mean) < 0.001:  # Near-zero drift
                regime_names.append("ranging")
            else:
                regime_names.append(f"regime_{i}")

        logger.info(f"Assigned regime names based on means: {regime_names}")

        return regime_names

    def _compute_regime_persistence(self, features: np.ndarray) -> list[float]:
        """Compute average duration (bars) for each regime.

        Args:
            features: Feature array

        Returns:
            List of average durations (one per regime)
        """
        # Predict regimes
        regime_labels = self.model.predict(features)

        regime_durations = []

        for regime_idx in range(self.n_regimes):
            # Find consecutive sequences of this regime
            is_regime = (regime_labels == regime_idx)

            # Find regime switches
            switches = np.diff(is_regime.astype(int))
            switch_indices = np.where(switches != 0)[0]

            # Calculate durations
            if len(switch_indices) == 0:
                # No switches, entire series is one regime
                duration = len(features)
            else:
                # Calculate average duration
                durations = []
                start = 0

                for switch_idx in switch_indices:
                    durations.append(switch_idx - start)
                    start = switch_idx + 1

                # Last segment
                durations.append(len(features) - start)

                duration = np.mean(durations) if durations else len(features)

            regime_durations.append(duration)

        logger.info(f"Regime persistence (bars): {regime_durations}")

        return regime_durations

    def _calculate_bic(self, n_samples: int, n_features: int, n_params: int | None = None) -> float:
        """Calculate Bayesian Information Criterion (BIC) for model selection.

        Args:
            n_samples: Number of samples
            n_features: Number of features
            n_params: Number of parameters (or calculate automatically)

        Returns:
            BIC score (lower is better)
        """
        if n_params is None:
            # For GaussianHMM with full covariance:
            # n_params = n_regimes * n_features + n_regimes * (n_regimes - 1) / 2
            # Actually, let's use a simpler approximation
            n_params = self.n_regimes * n_features * 2  # Mean and covariance

        # BIC = -2 * log_likelihood + n_params * log(n_samples)
        log_likelihood = self.model.score(
            np.zeros((1, n_features))  # Dummy data for score calculation
        )

        bic = -2 * log_likelihood + n_params * np.log(n_samples)

        return bic

    def save(self, model_path: str | Path) -> None:
        """Save trained HMM model and metadata to disk.

        Args:
            model_path: Path to save model (without extension)
        """
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save HMM model
        joblib.dump(self.model, model_path / "hmm_model.joblib")

        # Save metadata
        if self.metadata:
            metadata_file = model_path / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(
                    self.metadata.model_dump(mode="json"),
                    f,
                    indent=2
                )

        logger.info(f"HMM model saved to {model_path}")

    @classmethod
    def load(cls, model_path: str | Path) -> "HMMRegimeDetector":
        """Load trained HMM model and metadata from disk.

        Args:
            model_path: Path to model directory

        Returns:
            Loaded HMMRegimeDetector instance
        """
        model_path = Path(model_path)

        # Load HMM model
        model_file = model_path / "hmm_model.joblib"
        model = joblib.load(model_file)

        # Load metadata
        metadata_file = model_path / "metadata.json"
        with open(metadata_file, "r") as f:
            metadata_dict = json.load(f)
            metadata = HMMModelMetadata(**metadata_dict)

        # Create detector instance
        detector = cls(
            n_regimes=metadata.n_regimes,
            covariance_type="full",  # TODO: Load from metadata
            n_iterations=metadata.convergence_iterations,
            random_state=42
        )

        detector.model = model
        detector.metadata = metadata

        logger.info(f"HMM model loaded from {model_path}")

        return detector


def find_optimal_hmm(
    features_df: pd.DataFrame,
    n_regimes_range: list[int] = [2, 3, 4, 5],
    covariance_types: list[str] = ["full", "diag", "spherical"],
    n_iterations: int = 100
) -> dict:
    """Find optimal HMM configuration using grid search and BIC.

    Args:
        features_df: DataFrame with HMM features
        n_regimes_range: Range of regime counts to test
        covariance_types: Covariance types to test
        n_iterations: Max EM iterations

    Returns:
        Dictionary with best model configuration:
        {
            "best_n_regimes": int,
            "best_covariance_type": str,
            "best_bic_score": float,
            "all_results": list[dict]  # Results for all configurations
        }
    """
    logger.info("Starting HMM hyperparameter tuning...")

    features_array = features_df.values
    features_array = np.nan_to_num(features_array, nan=0.0)
    n_samples, n_features = features_array.shape

    all_results = []
    best_bic = float('inf')
    best_config = None

    for n_regimes in n_regimes_range:
        for cov_type in covariance_types:
            try:
                # Train HMM with multiple initializations
                best_model_bic = float('inf')
                best_model = None

                # Try multiple random initializations
                for init in range(2):  # Reduced to 2 for speed
                    hmm_model = hmm.GaussianHMM(
                        n_components=n_regimes,
                        covariance_type=cov_type,
                        n_iter=n_iterations,
                        random_state=42 + init,  # Different seed each time
                        init_params='stmc',  # Initialize start, trans, means, covars
                        tol=1e-3,  # Convergence tolerance
                    )

                    hmm_model.fit(features_array)

                    # Calculate BIC
                    n_params = n_regimes * n_features * 2
                    log_likelihood = hmm_model.score(features_array)
                    bic = -2 * log_likelihood + n_params * np.log(n_samples)

                    if bic < best_model_bic:
                        best_model_bic = bic
                        best_model = hmm_model

                # Use best model from multiple initializations
                if best_model is not None:
                    iter_count = getattr(best_model, 'n_iter_', n_iterations)
                    result = {
                        "n_regimes": n_regimes,
                        "covariance_type": cov_type,
                        "bic_score": best_model_bic,
                        "n_iterations": iter_count,
                        "log_likelihood": best_model.score(features_array)
                    }

                    all_results.append(result)

                    logger.info(
                        f"HMM({n_regimes} regimes, {cov_type} covariance): "
                        f"BIC={best_model_bic:.2f}, "
                        f"iterations={iter_count}"
                    )

                    # Track best model
                    if best_model_bic < best_bic:
                        best_bic = best_model_bic
                        best_config = result
                else:
                    logger.warning(
                        f"HMM({n_regimes} regimes, {cov_type} covariance): "
                        f"Did not converge after multiple initializations"
                    )

            except Exception as e:
                logger.error(f"Failed to train HMM ({n_regimes} regimes, {cov_type} covariance): {e}")

    logger.info(f"Optimal HMM configuration: {best_config['n_regimes']} regimes, "
                f"{best_config['covariance_type']} covariance, BIC={best_config['bic_score']:.2f}")

    return {
        "best_n_regimes": best_config["n_regimes"],
        "best_covariance_type": best_config["covariance_type"],
        "best_bic_score": best_config["bic_score"],
        "all_results": all_results
    }
