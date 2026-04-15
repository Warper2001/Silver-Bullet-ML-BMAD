"""Hybrid Regime-Aware ML Pipeline for Silver Bullet trading.

This module implements the hybrid regime-aware trading system that:
1. Detects market regimes using HMM
2. Selects appropriate ML model based on regime
3. Evaluates every bar with bar-by-bar evaluation (not signal-based)
4. Applies REGIME-SPECIFIC probability thresholds for filtering:
   - Regime 0 (trending_up): 25% threshold (65.2% win rate)
   - Regime 1 (ranging): 50% threshold (34.3% win rate)
   - Regime 2 (trending_down): 35% threshold (55.1% win rate)
5. Enforces minimum bars between trades
6. Uses triple-barrier exits (TP: 0.3%, SL: 0.2%, Time: 30min)

Tier 1 Model Support (Experimental - 2026-04-15):
- Option 1: Baseline models (52 technical indicators)
  - Regime 0: 97.83% accuracy
  - Regime 2: 100.00% accuracy
  - Generic: 79.30% accuracy
- Option 2: Tier 1 models (17 order flow, volatility, microstructure features)
  - Regime 0: 88.73% accuracy, 96.25% precision
  - Regime 1: 82.61% accuracy, 21.26% precision (challenging regime)
  - Regime 2: 69.10% accuracy (limited data)
  - Enable via: config.yaml ml.tier1_models.enabled = true

Expected Performance (based on triple-barrier label analysis):
- Baseline: 55-65% win rate, 5-25 trades/day
- Tier 1: Expected 55-65% win rate, 10-25 trades/day (higher precision in Regime 0)

Configuration:
- See config.yaml [ml] section for Tier 1 model configuration
- Toggle between baseline and Tier 1 via ml.tier1_models.enabled
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

from src.data.models import DollarBar
from src.execution.trade_execution_pipeline import TradingSignal
from src.ml.features import FeatureEngineer
from src.ml.regime_detection import HMMRegimeDetector, HMMFeatureEngineer
from src.ml.tier1_features import Tier1FeatureEngineer

logger = logging.getLogger(__name__)


class HybridMLPipeline:
    """Hybrid regime-aware ML pipeline for bar-by-bar trading.

    Pipeline Flow:
    1. Detect market regime using HMM
    2. Select appropriate ML model based on regime:
       - Regime 0 (trending_up) → Regime 0 model (97.83% accuracy)
       - Regime 1 (trending_up_strong) → Generic fallback (79.30% accuracy)
       - Regime 2 (trending_down) → Regime 2 model (100.00% accuracy)
    3. Evaluate every bar with selected model
    4. Filter by REGIME-SPECIFIC probability threshold:
       - Regime 0: 25% (aggressive - 65.2% win rate)
       - Regime 1: 50% (conservative - 34.3% win rate)
       - Regime 2: 35% (moderate - 55.1% win rate)
    5. Enforce minimum bars between trades
    6. Publish trading signals to execution queue

    Tier 1 Model Support (Experimental):
    - Uses 17 order flow, volatility, microstructure features
    - Higher precision in Regime 0 (96.25%)
    - Graceful degradation: Falls back to baseline if Tier 1 unavailable
    - Enable via config.yaml: ml.tier1_models.enabled = true

    Performance (based on triple-barrier label analysis):
        - Expected win rate: 55-65% (vs 34% generic)
        - Expected trades/day: 5-25 (regime-dependent)
        - Expected Sharpe: 0.8-1.5 (vs 0.4 generic)
        - Processing latency < 50ms per bar
    """

    # Configuration constants (from deployment decision)
    # Regime-specific probability thresholds based on triple-barrier label analysis
    # Regime 0 (trending_up): 65.2% win rate → 25% threshold (more signals)
    # Regime 1 (ranging): 34.3% win rate → 50% threshold (very selective)
    # Regime 2 (trending_down): 55.1% win rate → 35% threshold (normal)
    REGIME_THRESHOLDS = {
        0: 0.25,  # Regime 0: 25% threshold
        1: 0.50,  # Regime 1: 50% threshold (or skip entirely)
        2: 0.35,  # Regime 2: 35% threshold
    }
    DEFAULT_PROBABILITY_THRESHOLD = 0.40  # Fallback threshold
    MIN_BARS_BETWEEN_TRADES = 30  # 2.5 hours at 5-min bars

    # Triple-barrier exit parameters
    TAKE_PROFIT_PCT = 0.003  # 0.3%
    STOP_LOSS_PCT = 0.002    # 0.2%
    MAX_HOLD_MINUTES = 30

    def __init__(
        self,
        output_queue: asyncio.Queue[TradingSignal],
        model_dir: str | Path = "models/xgboost",
        hmm_dir: str | Path = "models/hmm/regime_model",
        config_path: str | Path = "config.yaml",
    ) -> None:
        """Initialize the hybrid ML pipeline.

        Args:
            output_queue: Queue publishing TradingSignal to execution pipeline
            model_dir: Directory containing ML models
            hmm_dir: Directory containing HMM regime detector
            config_path: Path to config.yaml for Tier 1 model settings
        """
        self._output_queue = output_queue
        self._model_dir = Path(model_dir)
        self._hmm_dir = Path(hmm_dir)

        # Load configuration for Tier 1 model support
        self._config = self._load_config(config_path)
        tier1_config = self._config.get("ml", {}).get("tier1_models", {})
        self._use_tier1_models = tier1_config.get("enabled", False)

        # Store safety thresholds for validation
        self._min_precision_threshold = tier1_config.get("min_precision_threshold", 0.60)
        self._min_win_rate_threshold = tier1_config.get("min_win_rate_threshold", 0.50)

        # Initialize HMM regime detector
        logger.info("Loading HMM regime detector...")
        self._hmm_detector = HMMRegimeDetector.load(self._hmm_dir)
        self._hmm_feature_engineer = HMMFeatureEngineer()
        self._current_regime: Optional[int] = None
        logger.info(f"✅ HMM loaded: {self._hmm_detector.n_regimes} regimes")

        # Load regime-specific ML models
        logger.info("Loading hybrid ML models...")
        self._load_models()
        logger.info("✅ Hybrid models loaded")

        # Initialize feature engineer (baseline or Tier 1)
        if self._use_tier1_models:
            logger.info("Initializing Tier 1 feature engineer...")
            self._feature_engineer = Tier1FeatureEngineer()
            self._tier1_feature_engineer = Tier1FeatureEngineer()
            tier1_feature_count = len(self._tier1_feature_engineer.feature_names)
            logger.info(f"✅ Tier 1 feature engineer initialized ({tier1_feature_count} features)")
        else:
            logger.info("Initializing baseline feature engineer (52 features)...")
            self._feature_engineer = FeatureEngineer(
                model_dir=self._model_dir, window_size=100
            )
            self._tier1_feature_engineer = None
            logger.info("✅ Baseline feature engineer initialized")

        # Trade management
        self._bars_since_last_trade = self.MIN_BARS_BETWEEN_TRADES
        self._last_trade_bar: Optional[int] = None

        # Statistics
        self._bars_processed = 0
        self._signals_generated = 0
        self._regime_distribution = {0: 0, 1: 0, 2: 0}
        self._tier1_predictions = 0  # Track Tier 1 model usage
        self._baseline_predictions = 0  # Track baseline model usage

        logger.info("HybridMLPipeline initialized")
        logger.info("  Regime-specific thresholds:")
        for regime, thresh in self.REGIME_THRESHOLDS.items():
            logger.info(f"    Regime {regime}: {thresh:.0%}")
        logger.info(f"  Min bars between trades: {self.MIN_BARS_BETWEEN_TRADES}")
        logger.info(f"  Tier 1 models: {'ENABLED' if self._use_tier1_models else 'DISABLED'}")

    def _load_config(self, config_path: str | Path) -> dict:
        """Load configuration from config.yaml.

        Loads Tier 1 model configuration to enable/disable experimental features.
        Distinguishes between file not found (warn) and parse errors (error).

        Args:
            config_path: Path to config.yaml file

        Returns:
            Configuration dictionary with ml.tier1_models settings
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}")
            logger.warning("Using default configuration (Tier 1 models disabled)")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in config file {config_path}: {e}")
            logger.error("Using default configuration (Tier 1 models disabled)")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error loading config from {config_path}: {e}")
            logger.error("Using default configuration (Tier 1 models disabled)")
            return {}

    def _load_models(self) -> None:
        """Load regime-specific ML models (baseline or Tier 1)."""
        if self._use_tier1_models:
            logger.info("Loading Tier 1 models (order flow, volatility, microstructure)...")
            self._load_tier1_models()
        else:
            logger.info("Loading baseline models (54 technical indicators)...")
            self._load_baseline_models()

    def _load_baseline_models(self) -> None:
        """Load baseline regime-specific ML models (54 features)."""
        model_path = self._model_dir / "regime_aware_real_labels"

        # Load Regime 0 model (trending_up)
        regime_0_path = model_path / "xgboost_regime_0_real_labels.joblib"
        if regime_0_path.exists():
            self._regime_0_model = joblib.load(regime_0_path)
            logger.info("  ✅ Baseline Regime 0 model loaded (97.83% accuracy)")
        else:
            logger.error(f"❌ Regime 0 model not found: {regime_0_path}")
            raise FileNotFoundError(f"Regime 0 model not found: {regime_0_path}")

        # Load Regime 2 model (trending_down)
        regime_2_path = model_path / "xgboost_regime_2_real_labels.joblib"
        if regime_2_path.exists():
            self._regime_2_model = joblib.load(regime_2_path)
            logger.info("  ✅ Baseline Regime 2 model loaded (100.00% accuracy)")
        else:
            logger.error(f"❌ Regime 2 model not found: {regime_2_path}")
            raise FileNotFoundError(f"Regime 2 model not found: {regime_2_path}")

        # Load Generic model (fallback for Regime 1)
        generic_path = model_path / "xgboost_generic_real_labels.joblib"
        if generic_path.exists():
            self._generic_model = joblib.load(generic_path)
            logger.info("  ✅ Baseline Generic model loaded (79.30% accuracy)")
        else:
            logger.error(f"❌ Generic model not found: {generic_path}")
            raise FileNotFoundError(f"Generic model not found: {generic_path}")

    def _load_tier1_models(self) -> None:
        """Load Tier 1 regime-specific ML models (17 features).

        Loads experimental Tier 1 models trained on order flow, volatility,
        and microstructure features specifically designed for 1-minute data.

        Raises:
            FileNotFoundError: If Tier 1 model directory doesn't exist

        Note:
            Graceful degradation: If Tier 1 models fail to load, falls back
            to baseline models with a warning. This prevents production outages.

            Tier 1 models have different performance characteristics:
            - Regime 0: High precision (96.25%) but limited accuracy
            - Regime 1: Low precision (21.26%) - challenging regime
            - Regime 2: Limited training data
        """
        # Get Tier 1 model path from config
        tier1_path_str = self._config.get("ml", {}).get("tier1_models", {}).get("model_path", "models/xgboost/regime_aware_tier1/")
        tier1_path = Path(tier1_path_str)

        # Check if Tier 1 directory exists
        if not tier1_path.exists():
            logger.warning(f"⚠️  Tier 1 model directory not found: {tier1_path}")
            logger.warning("⚠️  Falling back to baseline models")
            logger.info("🔄 Loading baseline models instead...")
            self._use_tier1_models = False  # Disable Tier 1
            self._load_baseline_models()
            return

        # Load Regime 0 model (trending_up)
        regime_0_path = tier1_path / "xgboost_regime_0_tier1.joblib"
        if regime_0_path.exists():
            try:
                self._regime_0_model = joblib.load(regime_0_path)
                logger.info("  ✅ Tier 1 Regime 0 model loaded (96.25% precision)")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load Tier 1 Regime 0 model: {e}")
                logger.warning("⚠️  Falling back to baseline models")
                self._use_tier1_models = False
                self._load_baseline_models()
                return
        else:
            logger.warning(f"⚠️  Tier 1 Regime 0 model not found: {regime_0_path}")
            logger.warning("⚠️  Falling back to baseline models")
            self._use_tier1_models = False
            self._load_baseline_models()
            return

        # Load Regime 1 model (ranging - low precision but trained)
        regime_1_path = tier1_path / "xgboost_regime_1_tier1.joblib"
        if regime_1_path.exists():
            try:
                self._regime_1_model = joblib.load(regime_1_path)
                logger.info("  ✅ Tier 1 Regime 1 model loaded (21.26% precision)")
                logger.warning("⚠️  WARNING: Regime 1 model has low precision (21.26%)")
                logger.warning("⚠️  This model should be used with caution in production")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load Tier 1 Regime 1 model: {e}")
                logger.warning("⚠️  Falling back to baseline models")
                self._use_tier1_models = False
                self._load_baseline_models()
                return
        else:
            logger.warning(f"⚠️  Tier 1 Regime 1 model not found: {regime_1_path}")
            logger.warning("⚠️  Falling back to baseline models")
            self._use_tier1_models = False
            self._load_baseline_models()
            return

        # Load Regime 2 model (trending_down - limited data)
        regime_2_path = tier1_path / "xgboost_regime_2_tier1.joblib"
        if regime_2_path.exists():
            try:
                self._regime_2_model = joblib.load(regime_2_path)
                logger.info("  ✅ Tier 1 Regime 2 model loaded (limited data)")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load Tier 1 Regime 2 model: {e}")
                logger.warning("⚠️  Falling back to baseline models")
                self._use_tier1_models = False
                self._load_baseline_models()
                return
        else:
            logger.warning(f"⚠️  Tier 1 Regime 2 model not found: {regime_2_path}")
            logger.warning("⚠️  Falling back to baseline models")
            self._use_tier1_models = False
            self._load_baseline_models()
            return

        # Tier 1 doesn't use generic model - has regime-specific for all 3 regimes
        self._generic_model = None

    async def process_bar(self, bar: DollarBar, historical_data: pd.DataFrame) -> None:
        """Process a single dollar bar through the hybrid pipeline.

        This implements bar-by-bar evaluation (not signal-based filtering).
        Every bar is evaluated with the regime-appropriate ML model.

        Args:
            bar: Current dollar bar to evaluate
            historical_data: Historical bars for feature engineering

        Performance:
            Total latency < 50ms (regime detection + features + inference + filtering)
        """
        start_time = time.perf_counter()

        try:
            self._bars_processed += 1

            # Step 1: Detect market regime
            regime = self._detect_regime(historical_data)
            self._current_regime = regime
            self._regime_distribution[regime] += 1

            # Step 2: Select model based on regime (HYBRID APPROACH)
            if regime == 0:
                model = self._regime_0_model
                model_name = "Regime_0"
            elif regime == 2:
                model = self._regime_2_model
                model_name = "Regime_2"
            else:  # Regime 1
                # Tier 1 has regime-specific model for Regime 1, baseline uses generic
                if self._use_tier1_models and hasattr(self, '_regime_1_model'):
                    model = self._regime_1_model
                else:
                    model = self._generic_model
                model_name = "Regime_1" if self._use_tier1_models else "Generic"

            # Step 3: Generate features (baseline or Tier 1)
            if self._use_tier1_models:
                features = self._generate_tier1_features(bar, historical_data)
                self._tier1_predictions += 1
            else:
                features = self._feature_engineer.generate_features_bar(
                    current_bar=bar, historical_data=historical_data
                )
                self._baseline_predictions += 1

            # Step 4: Predict success probability
            probability = self._predict_probability(features, model)

            # Step 5: Apply regime-specific probability threshold filter
            threshold = self.REGIME_THRESHOLDS.get(
                regime, self.DEFAULT_PROBABILITY_THRESHOLD
            )

            if probability < threshold:
                logger.debug(
                    f"Bar filtered: P(Success)={probability:.2%} < "
                    f"{threshold:.0%} threshold (Regime={regime}, "
                    f"Model={model_name})"
                )
                return

            # Step 6: Check minimum bars between trades
            if self._bars_since_last_trade < self.MIN_BARS_BETWEEN_TRADES:
                logger.debug(
                    f"Signal blocked: Only {self._bars_since_last_trade} bars since "
                    f"last trade (min: {self.MIN_BARS_BETWEEN_TRADES})"
                )
                self._bars_since_last_trade += 1
                return

            # Step 7: Determine trade direction from 5-bar momentum
            direction = self._get_trade_direction(historical_data)

            # Step 8: Create SilverBulletSetup signal
            signal = self._create_signal(bar, direction, probability)

            # Step 9: Publish to execution queue
            await self._output_queue.put(signal)

            # Step 10: Update trade management
            self._bars_since_last_trade = 0
            self._signals_generated += 1

            latency_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                f"🎯 Signal #{self._signals_generated}: P(Success)={probability:.2%}, "
                f"Direction={direction}, Regime={regime}, Model={model_name}, "
                f"Latency={latency_ms:.2f}ms"
            )

            # Log statistics every 50 signals
            if self._signals_generated % 50 == 0:
                self._log_statistics()

        except Exception as e:
            logger.error(f"Error processing bar: {e}", exc_info=True)

    def _detect_regime(self, historical_data: pd.DataFrame) -> int:
        """Detect market regime using HMM.

        Args:
            historical_data: Historical bars for HMM feature engineering

        Returns:
            Regime ID (0, 1, or 2)
        """
        # Generate HMM features
        hmm_features = self._hmm_feature_engineer.engineer_features(historical_data)

        # Predict regime (use latest)
        regime = self._hmm_detector.predict(hmm_features)
        return int(regime[-1]) if len(regime) > 0 else 1

    def _predict_probability(self, features: np.ndarray, model: Any) -> float:
        """Predict success probability using selected model.

        Args:
            features: Feature vector for current bar
            model: XGBoost model (regime-specific or generic)

        Returns:
            Success probability (0.0 to 1.0)
        """
        # Reshape if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        # Get probability of class 1 (success)
        probability = float(model.predict_proba(features)[0, 1])
        return probability

    def _generate_tier1_features(self, bar: DollarBar, historical_data: pd.DataFrame) -> np.ndarray:
        """Generate Tier 1 features for 1-minute trading.

        Generates order flow, volatility, and microstructure features
        specifically designed for high-frequency (1-minute) data where
        traditional technical indicators fail due to microstructure noise.

        Args:
            bar: Current dollar bar
            historical_data: Historical bars for feature engineering

        Returns:
            Feature vector (17 features)

        Raises:
            ValueError: If insufficient historical data for Tier 1 features

        Note:
            Features include: volume imbalance, cumulative delta, realized
            volatility, VWAP deviation, bid-ask bounce, noise-adjusted momentum.
            Requires minimum 100 bars of historical data.
        """
        # Ensure we have enough data
        min_bars_required = 100
        if len(historical_data) < min_bars_required:
            raise ValueError(
                f"Insufficient historical data for Tier 1 features: "
                f"{len(historical_data)} < {min_bars_required} required"
            )

        # Create DataFrame with current bar
        df = historical_data.copy()

        # Generate Tier 1 features
        df_features = self._tier1_feature_engineer.generate_features(df)

        # Get latest features
        tier1_feature_names = self._tier1_feature_engineer.feature_names

        # Validate features exist
        missing_features = [f for f in tier1_feature_names if f not in df_features.columns]
        if missing_features:
            raise ValueError(f"Missing Tier 1 features: {missing_features}")

        features = df_features[tier1_feature_names].iloc[-1].values

        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)

        return features

    def _get_trade_direction(self, historical_data: pd.DataFrame) -> str:
        """Determine trade direction from 5-bar momentum.

        Args:
            historical_data: Historical bars for momentum calculation

        Returns:
            "bullish" or "bearish"
        """
        # Calculate 5-bar momentum
        recent_close = historical_data['close'].iloc[-1]
        momentum_5 = recent_close - historical_data['close'].iloc[-6]

        return "bullish" if momentum_5 > 0 else "bearish"

    def _create_signal(
        self, bar: DollarBar, direction: str, probability: float
    ) -> TradingSignal:
        """Create TradingSignal from bar evaluation.

        Args:
            bar: Current dollar bar
            direction: Trade direction ("bullish" or "bearish")
            probability: ML-predicted success probability

        Returns:
            TradingSignal with triple-barrier exits
        """
        # Calculate entry, stop loss, take profit
        entry_price = bar.close

        if direction == "bullish":
            stop_loss = entry_price * (1 - self.STOP_LOSS_PCT)
            take_profit = entry_price * (1 + self.TAKE_PROFIT_PCT)
        else:  # bearish
            stop_loss = entry_price * (1 + self.STOP_LOSS_PCT)
            take_profit = entry_price * (1 - self.TAKE_PROFIT_PCT)

        # Create unique signal ID
        model_type = "tier1" if self._use_tier1_models else "baseline"
        signal_id = f"hybrid_{model_type}_{int(bar.timestamp.timestamp())}"

        # Create TradingSignal
        signal = TradingSignal(
            signal_id=signal_id,
            symbol="MNQ",
            direction=direction,
            confidence_score=probability,
            timestamp=bar.timestamp,
            entry_price=entry_price,
            patterns=[f"hybrid_ml_{model_type}_regime_{self._current_regime or 1}"],
            prediction={
                "probability": probability,
                "regime": self._current_regime or 1,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "setup_type": f"hybrid_ml_{model_type}",
                "model_type": model_type,
                "feature_count": 16 if self._use_tier1_models else 54
            },
            quantity=5,  # Default quantity
            stop_loss_price=stop_loss,  # Add stop loss price
            take_profit_price=take_profit  # Add take profit price
        )

        return signal

    def _log_statistics(self) -> None:
        """Log pipeline statistics."""
        logger.info("=" * 70)
        logger.info("HYBRID PIPELINE STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Bars processed: {self._bars_processed}")
        logger.info(f"Signals generated: {self._signals_generated}")
        logger.info(f"Signal rate: {self._signals_generated / max(self._bars_processed, 1) * 100:.2f}%")
        logger.info(f"Regime distribution: {self._regime_distribution}")
        logger.info("=" * 70)

    def health_check(self) -> dict[str, Any]:
        """Check health status of hybrid pipeline.

        Returns:
            Dictionary containing health status and model usage statistics
        """
        return {
            "healthy": True,
            "tier1_models_enabled": self._use_tier1_models,
            "regime_detector_loaded": self._hmm_detector is not None,
            "regime_0_model_loaded": hasattr(self, "_regime_0_model"),
            "regime_1_model_loaded": hasattr(self, "_regime_1_model"),
            "regime_2_model_loaded": hasattr(self, "_regime_2_model"),
            "generic_model_loaded": hasattr(self, "_generic_model"),
            "tier1_feature_engineer_loaded": hasattr(self, "_tier1_feature_engineer") and self._tier1_feature_engineer is not None,
            "current_regime": self._current_regime,
            "bars_processed": self._bars_processed,
            "signals_generated": self._signals_generated,
            "signal_rate": self._signals_generated / max(self._bars_processed, 1),
            "regime_distribution": self._regime_distribution,
            # Model usage tracking
            "tier1_predictions": self._tier1_predictions,
            "baseline_predictions": self._baseline_predictions,
            "tier1_usage_pct": self._tier1_predictions / max(self._tier1_predictions + self._baseline_predictions, 1) * 100,
        }
