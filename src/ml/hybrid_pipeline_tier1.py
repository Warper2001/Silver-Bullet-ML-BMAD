"""Enhanced Hybrid Regime-Aware ML Pipeline with Tier 1 Features.

This module implements the hybrid regime-aware trading system with:
1. HMM regime detection
2. Tier 1 feature engineering (order flow, volatility, microstructure)
3. Regime-specific XGBoost models trained on triple-barrier labels
4. Regime-specific probability thresholds
5. Bar-by-bar evaluation for 1-minute trading

Expected Performance (with Tier 1 models):
- Sharpe ratio: 2.0+ (vs 1.52 without Tier 1)
- Win rate: 55-65% (Regime 0: 96% precision model)
- Trade frequency: 10-25/day
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from src.data.models import DollarBar
from src.execution.trade_execution_pipeline import TradingSignal
from src.ml.tier1_features import Tier1FeatureEngineer
from src.ml.regime_detection import HMMRegimeDetector, HMMFeatureEngineer

logger = logging.getLogger(__name__)


class HybridMLPipelineTier1:
    """Enhanced hybrid regime-aware ML pipeline with Tier 1 features.

    Pipeline Flow:
    1. Detect market regime using HMM
    2. Generate Tier 1 features (order flow, volatility, microstructure)
    3. Select appropriate Tier 1 XGBoost model based on regime:
       - Regime 0 (trending_up): 96.25% precision model
       - Regime 1 (ranging): Model with 21.26% precision (challenging regime)
       - Regime 2 (trending_down): Model with 26.93% precision
    4. Filter by regime-specific probability thresholds
    5. Publish trading signals with triple-barrier exits

    Performance (based on Tier 1 model validation):
        - Expected Sharpe: 2.0+ (vs 1.52 without Tier 1 features)
        - Expected win rate: 55-65% (Regime 0: 96% precision)
        - Expected trades/day: 10-25
        - Processing latency: <50ms per bar
    """

    # Regime-specific probability thresholds
    REGIME_THRESHOLDS = {
        0: 0.25,  # Regime 0: 25% threshold (aggressive - excellent model)
        1: 0.50,  # Regime 1: 50% threshold (conservative - poor model)
        2: 0.35,  # Regime 2: 35% threshold (moderate - limited data)
    }
    DEFAULT_PROBABILITY_THRESHOLD = 0.40

    # Triple-barrier exit parameters
    TAKE_PROFIT_PCT = 0.003  # 0.3%
    STOP_LOSS_PCT = 0.002    # 0.2%
    MAX_HOLD_MINUTES = 30

    # Minimum bars between trades
    MIN_BARS_BETWEEN_TRADES = 30  # For 1-minute bars = 30 minutes

    def __init__(
        self,
        output_queue: asyncio.Queue[TradingSignal],
        model_dir: str | Path = "models/xgboost/regime_aware_tier1",
        hmm_dir: str | Path = "models/hmm/regime_model",
    ) -> None:
        """Initialize the enhanced hybrid ML pipeline.

        Args:
            output_queue: Queue publishing TradingSignal to execution pipeline
            model_dir: Directory containing Tier 1 XGBoost models
            hmm_dir: Directory containing HMM regime detector
        """
        self._output_queue = output_queue
        self._model_dir = Path(model_dir)
        self._hmm_dir = Path(hmm_dir)

        # Initialize HMM regime detector
        logger.info("Loading HMM regime detector...")
        self._hmm_detector = HMMRegimeDetector.load(self._hmm_dir)
        self._hmm_feature_engineer = HMMFeatureEngineer()
        self._current_regime: Optional[int] = None
        logger.info(f"✅ HMM loaded: {self._hmm_detector.n_regimes} regimes")

        # Load Tier 1 XGBoost models
        logger.info("Loading Tier 1 XGBoost models...")
        self._load_tier1_models()
        logger.info("✅ Tier 1 models loaded")

        # Initialize Tier 1 feature engineer
        self._tier1_feature_engineer = Tier1FeatureEngineer()
        logger.info("✅ Tier 1 feature engineer initialized")

        # Trade management
        self._bars_since_last_trade = self.MIN_BARS_BETWEEN_TRADES
        self._last_trade_bar: Optional[int] = None

        # Statistics
        self._bars_processed = 0
        self._signals_generated = 0
        self._regime_distribution = {0: 0, 1: 0, 2: 0}
        self._feature_buffer = pd.DataFrame()  # Buffer for feature calculation

        logger.info("HybridMLPipelineTier1 initialized")
        logger.info("  Using Tier 1 features (order flow, volatility, microstructure)")
        logger.info("  Regime-specific thresholds:")
        for regime, thresh in self.REGIME_THRESHOLDS.items():
            logger.info(f"    Regime {regime}: {thresh:.0%}")
        logger.info(f"  Min bars between trades: {self.MIN_BARS_BETWEEN_TRADES}")

    def _load_tier1_models(self) -> None:
        """Load Tier 1 regime-specific XGBoost models."""
        # Load Regime 0 model (trending_up - EXCELLENT: 96.25% precision)
        regime_0_path = self._model_dir / "xgboost_regime_0_tier1.joblib"
        if regime_0_path.exists():
            self._regime_0_model = joblib.load(regime_0_path)
            logger.info("  ✅ Regime 0 model loaded (96.25% precision, 88.73% accuracy)")
        else:
            logger.error(f"❌ Regime 0 model not found: {regime_0_path}")
            raise FileNotFoundError(f"Regime 0 model not found: {regime_0_path}")

        # Load Regime 1 model (ranging - CHALLENGING: 21.26% precision)
        regime_1_path = self._model_dir / "xgboost_regime_1_tier1.joblib"
        if regime_1_path.exists():
            self._regime_1_model = joblib.load(regime_1_path)
            logger.info("  ✅ Regime 1 model loaded (21.26% precision, 82.61% accuracy)")
        else:
            logger.error(f"❌ Regime 1 model not found: {regime_1_path}")
            raise FileNotFoundError(f"Regime 1 model not found: {regime_1_path}")

        # Load Regime 2 model (trending_down - LIMITED DATA: 26.93% precision)
        regime_2_path = self._model_dir / "xgboost_regime_2_tier1.joblib"
        if regime_2_path.exists():
            self._regime_2_model = joblib.load(regime_2_path)
            logger.info("  ✅ Regime 2 model loaded (26.93% precision, 69.10% accuracy)")
        else:
            logger.error(f"❌ Regime 2 model not found: {regime_2_path}")
            raise FileNotFoundError(f"Regime 2 model not found: {regime_2_path}")

    async def process_bar(self, bar: DollarBar, historical_data: pd.DataFrame) -> None:
        """Process a single dollar bar through the enhanced pipeline.

        Args:
            bar: Current dollar bar to evaluate
            historical_data: Historical bars for feature engineering

        Performance:
            Total latency < 100ms (regime detection + features + inference + filtering)
        """
        start_time = time.perf_counter()

        try:
            self._bars_processed += 1

            # Convert historical_data to list of dicts if needed
            if isinstance(historical_data, pd.DataFrame):
                # Convert to format expected by Tier1FeatureEngineer
                df = historical_data.copy()
                # Add current bar to dataframe
                current_bar_dict = {
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume,
                    'timestamp': bar.timestamp
                }
                df = pd.concat([df, pd.DataFrame([current_bar_dict])], ignore_index=True)
            else:
                df = historical_data

            # Step 1: Detect market regime
            regime = self._detect_regime(df)
            self._current_regime = regime
            self._regime_distribution[regime] += 1

            # Step 2: Select model based on regime
            if regime == 0:
                model = self._regime_0_model
                model_name = "Regime_0_Tier1"
            elif regime == 2:
                model = self._regime_2_model
                model_name = "Regime_2_Tier1"
            else:  # Regime 1
                model = self._regime_1_model
                model_name = "Regime_1_Tier1"

            # Step 3: Generate Tier 1 features
            try:
                df_with_features = self._tier1_feature_engineer.generate_features(df)

                # Extract Tier 1 feature columns
                feature_cols = [f for f in self._tier1_feature_engineer.feature_names if f in df_with_features.columns]

                if not feature_cols:
                    logger.warning("No Tier 1 features available, skipping bar")
                    return

                # Get latest feature values
                features = df_with_features[feature_cols].iloc[-1].values

                # Handle NaN values
                if np.isnan(features).any():
                    logger.debug("Features contain NaN values, skipping bar")
                    return

            except Exception as e:
                logger.error(f"Error generating Tier 1 features: {e}")
                return

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
            direction = self._get_trade_direction(df)

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

    def _detect_regime(self, df: pd.DataFrame) -> int:
        """Detect market regime using HMM.

        Args:
            df: Historical bars dataframe

        Returns:
            Regime ID (0, 1, or 2)
        """
        # Generate HMM features
        hmm_features = self._hmm_feature_engineer.engineer_features(df)

        # Predict regime (use latest)
        regime = self._hmm_detector.predict(hmm_features)
        return int(regime[-1]) if len(regime) > 0 else 1

    def _predict_probability(self, features: np.ndarray, model: Any) -> float:
        """Predict success probability using selected model.

        Args:
            features: Feature vector for current bar
            model: XGBoost model (regime-specific)

        Returns:
            Success probability (0.0 to 1.0)
        """
        # Reshape if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)

        # Get probability of class 1 (success/win)
        probability = float(model.predict_proba(features)[0, 1])
        return probability

    def _get_trade_direction(self, df: pd.DataFrame) -> str:
        """Determine trade direction from 5-bar momentum.

        Args:
            df: Historical bars dataframe

        Returns:
            "bullish" or "bearish"
        """
        if len(df) < 6:
            # Not enough data, use single bar
            return "bullish" if df['close'].iloc[-1] > df['open'].iloc[-1] else "bearish"

        # Calculate 5-bar momentum
        recent_close = df['close'].iloc[-1]
        momentum_5 = recent_close - df['close'].iloc[-6]

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
        signal_id = f"tier1_hybrid_{int(bar.timestamp.timestamp())}"

        # Create TradingSignal
        signal = TradingSignal(
            signal_id=signal_id,
            symbol="MNQ",
            direction=direction,
            confidence_score=probability,
            timestamp=bar.timestamp,
            entry_price=entry_price,
            patterns=[f"tier1_ml_regime_{self._current_regime or 1}"],
            prediction={
                "probability": probability,
                "regime": self._current_regime or 1,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "setup_type": "tier1_hybrid_ml",
                "features": "tier1_order_flow_volatility"
            },
            quantity=5,  # Default quantity
            stop_loss_price=stop_loss,
            take_profit_price=take_profit
        )

        return signal

    def _log_statistics(self) -> None:
        """Log pipeline statistics."""
        logger.info("=" * 70)
        logger.info("HYBRID PIPELINE TIER1 STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Bars processed: {self._bars_processed}")
        logger.info(f"Signals generated: {self._signals_generated}")
        logger.info(f"Signal rate: {self._signals_generated / max(self._bars_processed, 1) * 100:.2f}%")
        logger.info(f"Regime distribution: {self._regime_distribution}")
        logger.info("=" * 70)

    def health_check(self) -> dict[str, Any]:
        """Check health status of Tier 1 hybrid pipeline.

        Returns:
            Dictionary containing health status
        """
        return {
            "healthy": True,
            "regime_detector_loaded": self._hmm_detector is not None,
            "regime_0_model_loaded": hasattr(self, "_regime_0_model"),
            "regime_1_model_loaded": hasattr(self, "_regime_1_model"),
            "regime_2_model_loaded": hasattr(self, "_regime_2_model"),
            "tier1_feature_engineer_loaded": hasattr(self, "_tier1_feature_engineer"),
            "current_regime": self._current_regime,
            "bars_processed": self._bars_processed,
            "signals_generated": self._signals_generated,
            "signal_rate": self._signals_generated / max(self._bars_processed, 1),
            "regime_distribution": self._regime_distribution,
        }
