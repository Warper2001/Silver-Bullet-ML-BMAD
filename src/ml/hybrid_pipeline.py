"""Hybrid Regime-Aware ML Pipeline for Silver Bullet trading.

This module implements the hybrid regime-aware trading system that:
1. Detects market regimes using HMM
2. Selects appropriate ML model based on regime
3. Evaluates every bar with bar-by-bar evaluation (not signal-based)
4. Applies 40% probability threshold for filtering
5. Enforces minimum 30 bars between trades
6. Uses triple-barrier exits (TP: 0.3%, SL: 0.2%, Time: 30min)

Expected Performance (based on backtest):
- Trades per day: 3.92
- Win rate: 51.80%
- Sharpe ratio: 0.74
- Max drawdown: -2.78%
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
from src.ml.features import FeatureEngineer
from src.ml.regime_detection import HMMRegimeDetector, HMMFeatureEngineer

logger = logging.getLogger(__name__)


class HybridMLPipeline:
    """Hybrid regime-aware ML pipeline for bar-by-bar trading.

    Pipeline Flow:
    1. Detect market regime using HMM
    2. Select appropriate ML model based on regime:
       - Regime 0 (trending_up) → Regime 0 model (97.83% accuracy)
       - Regime 1 (trending_up_strong) → Generic fallback (79.30% accuracy)
       - Regime 2 (trending_down) → Regime 2 model (100.00% accuracy)
    3. Evaluate every 5-minute bar with selected model
    4. Filter by 40% probability threshold
    5. Enforce minimum 30 bars between trades
    6. Publish trading signals to execution queue

    Performance:
        - Expected trades/day: 3.92
        - Expected win rate: 51.80%
        - Expected Sharpe: 0.74
        - Processing latency < 50ms per bar
    """

    # Configuration constants (from deployment decision)
    PROBABILITY_THRESHOLD = 0.40  # 40% threshold
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
    ) -> None:
        """Initialize the hybrid ML pipeline.

        Args:
            output_queue: Queue publishing TradingSignal to execution pipeline
            model_dir: Directory containing ML models
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

        # Load regime-specific ML models
        logger.info("Loading hybrid ML models...")
        self._load_models()
        logger.info("✅ Hybrid models loaded")

        # Initialize feature engineer
        self._feature_engineer = FeatureEngineer(
            model_dir=self._model_dir, window_size=100
        )

        # Trade management
        self._bars_since_last_trade = self.MIN_BARS_BETWEEN_TRADES
        self._last_trade_bar: Optional[int] = None

        # Statistics
        self._bars_processed = 0
        self._signals_generated = 0
        self._regime_distribution = {0: 0, 1: 0, 2: 0}

        logger.info("HybridMLPipeline initialized")
        logger.info(f"  Probability threshold: {self.PROBABILITY_THRESHOLD:.0%}")
        logger.info(f"  Min bars between trades: {self.MIN_BARS_BETWEEN_TRADES}")

    def _load_models(self) -> None:
        """Load regime-specific ML models."""
        model_path = self._model_dir / "regime_aware_real_labels"

        # Load Regime 0 model (trending_up)
        regime_0_path = model_path / "xgboost_regime_0_real_labels.joblib"
        if regime_0_path.exists():
            self._regime_0_model = joblib.load(regime_0_path)
            logger.info("  ✅ Regime 0 model loaded (97.83% accuracy)")
        else:
            logger.error(f"❌ Regime 0 model not found: {regime_0_path}")
            raise FileNotFoundError(f"Regime 0 model not found: {regime_0_path}")

        # Load Regime 2 model (trending_down)
        regime_2_path = model_path / "xgboost_regime_2_real_labels.joblib"
        if regime_2_path.exists():
            self._regime_2_model = joblib.load(regime_2_path)
            logger.info("  ✅ Regime 2 model loaded (100.00% accuracy)")
        else:
            logger.error(f"❌ Regime 2 model not found: {regime_2_path}")
            raise FileNotFoundError(f"Regime 2 model not found: {regime_2_path}")

        # Load Generic model (fallback for Regime 1)
        generic_path = model_path / "xgboost_generic_real_labels.joblib"
        if generic_path.exists():
            self._generic_model = joblib.load(generic_path)
            logger.info("  ✅ Generic model loaded (79.30% accuracy)")
        else:
            logger.error(f"❌ Generic model not found: {generic_path}")
            raise FileNotFoundError(f"Generic model not found: {generic_path}")

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
                model = self._generic_model
                model_name = "Generic"

            # Step 3: Generate features
            features = self._feature_engineer.generate_features_bar(
                current_bar=bar, historical_data=historical_data
            )

            # Step 4: Predict success probability
            probability = self._predict_probability(features, model)

            # Step 5: Apply probability threshold filter (40%)
            if probability < self.PROBABILITY_THRESHOLD:
                logger.debug(
                    f"Bar filtered: P(Success)={probability:.2%} < "
                    f"{self.PROBABILITY_THRESHOLD:.0%} threshold (Regime={regime}, "
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
        signal_id = f"hybrid_{int(bar.timestamp.timestamp())}"

        # Create TradingSignal
        signal = TradingSignal(
            signal_id=signal_id,
            symbol="MNQ",
            direction=direction,
            confidence_score=probability,
            timestamp=bar.timestamp,
            entry_price=entry_price,
            patterns=[f"hybrid_ml_regime_{self._current_regime or 1}"],
            prediction={
                "probability": probability,
                "regime": self._current_regime or 1,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "setup_type": "hybrid_ml"
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
            Dictionary containing health status
        """
        return {
            "healthy": True,
            "regime_detector_loaded": self._hmm_detector is not None,
            "regime_0_model_loaded": hasattr(self, "_regime_0_model"),
            "regime_2_model_loaded": hasattr(self, "_regime_2_model"),
            "generic_model_loaded": hasattr(self, "_generic_model"),
            "current_regime": self._current_regime,
            "bars_processed": self._bars_processed,
            "signals_generated": self._signals_generated,
            "signal_rate": self._signals_generated / max(self._bars_processed, 1),
            "regime_distribution": self._regime_distribution,
        }
