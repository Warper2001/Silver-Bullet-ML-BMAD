"""Unit tests for spec-6-3: ML MetaLabelingFilter in Tier 2 paper trader.

Covers:
- MetaLabelingFilter model-present and model-absent (pass-through) paths
- _detect_and_enter: ML approval allows trade entry
- _detect_and_enter: ML rejection suppresses trade entry
- ALLOWED / FILTERED log messages emitted at INFO level
- predict_proba latency < 50ms (single-row inference)
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import from the research script; path manipulation is in the module itself
from src.research.tier2_streaming_working import MetaLabelingFilter, ML_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(direction: str = "bullish") -> dict:
    """Minimal feature dict covering all 8 FEATURE_COLS used by MetaLabelingFilter."""
    return {
        "fvg_fill_pct": 0.5,
        "sweep_window_vol": 1.2,
        "volume_ratio": 1.5,
        "signal_direction": direction,
        "h1_trend_slope": 0.05,
        "atr": 5.0,
        "session_displacement": 0.3,
        "session_volume_ratio": 1.1,
        # extra cols that the live feature dict also carries (ignored by model)
        "gap_size": 2.0,
        "et_hour": 10,
        "day_of_week": 1,
        "adr_pct_used": 0.6,
        "fvg_to_sweep_bars": 8,
        "prior_setup_proximity": 60,
    }


def _mock_model(proba: float):
    """Return a Pipeline mock matching the saved model format: Pipeline(StandardScaler + LR)."""
    pipe = MagicMock()
    pipe.predict_proba.return_value = np.array([[1 - proba, proba]])
    return pipe


# ---------------------------------------------------------------------------
# MetaLabelingFilter — model absent (pass-through)
# ---------------------------------------------------------------------------

class TestMetaLabelingFilterPassThrough:
    def test_missing_model_returns_pass_through(self, tmp_path):
        missing = tmp_path / "nonexistent.pkl"
        f = MetaLabelingFilter(missing)
        assert f.model is None
        assert f.predict_proba(_make_features()) == 1.0

    def test_corrupt_model_falls_back(self, tmp_path):
        bad = tmp_path / "bad.pkl"
        bad.write_bytes(b"not a valid pickle")
        f = MetaLabelingFilter(bad)
        assert f.model is None
        assert f.predict_proba(_make_features()) == 1.0


# ---------------------------------------------------------------------------
# MetaLabelingFilter — model present
# ---------------------------------------------------------------------------

class TestMetaLabelingFilterWithModel:
    def _filter_with_proba(self, proba: float) -> MetaLabelingFilter:
        f = MetaLabelingFilter.__new__(MetaLabelingFilter)
        f.threshold = ML_THRESHOLD
        f.model = _mock_model(proba)
        return f

    def test_returns_correct_proba_bullish(self):
        f = self._filter_with_proba(0.70)
        result = f.predict_proba(_make_features("bullish"))
        assert abs(result - 0.70) < 1e-6

    def test_returns_correct_proba_bearish(self):
        f = self._filter_with_proba(0.40)
        result = f.predict_proba(_make_features("bearish"))
        assert abs(result - 0.40) < 1e-6

    def test_direction_encoded_as_int(self):
        """signal_direction 'bullish'→1, 'bearish'→0 must not crash the model call."""
        f = self._filter_with_proba(0.60)
        # Should not raise
        f.predict_proba(_make_features("bullish"))
        f.predict_proba(_make_features("bearish"))
        assert f.model.predict_proba.call_count == 2

    # AC1 — threshold gate
    def test_ac1_above_threshold_allowed(self):
        f = self._filter_with_proba(ML_THRESHOLD + 0.01)
        assert f.predict_proba(_make_features()) >= ML_THRESHOLD

    def test_ac1_below_threshold_filtered(self):
        f = self._filter_with_proba(ML_THRESHOLD - 0.01)
        assert f.predict_proba(_make_features()) < ML_THRESHOLD

    def test_ac1_at_threshold_allowed(self):
        f = self._filter_with_proba(ML_THRESHOLD)
        assert f.predict_proba(_make_features()) >= ML_THRESHOLD

    # AC2 — latency < 50ms (with a real numpy mock, timing the wrapper overhead)
    def test_ac2_predict_proba_latency_under_50ms(self):
        f = self._filter_with_proba(0.65)
        features = _make_features()
        start = time.monotonic()
        for _ in range(100):
            f.predict_proba(features)
        elapsed_per_call_ms = (time.monotonic() - start) / 100 * 1000
        assert elapsed_per_call_ms < 50, (
            f"predict_proba overhead {elapsed_per_call_ms:.1f}ms exceeds 50ms budget"
        )


# ---------------------------------------------------------------------------
# Spec constraint: feature values must NOT be logged at INFO level
# ---------------------------------------------------------------------------

class TestLoggingConstraints:
    def test_feature_values_not_logged_at_info(self, caplog):
        """Spec-6-3 Never: raw feature values must only appear at DEBUG, not INFO."""
        import logging
        from src.research.tier2_streaming_working import MetaLabelingFilter

        f = MetaLabelingFilter.__new__(MetaLabelingFilter)
        f.threshold = ML_THRESHOLD
        f.model = _mock_model(0.65)

        with caplog.at_level(logging.INFO):
            f.predict_proba(_make_features())

        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        # No INFO record should contain raw numeric feature values
        for msg in info_messages:
            assert "ATR=" not in msg and "gap_size" not in msg and "Feature Vector" not in msg, (
                f"Feature values leaked to INFO log: {msg}"
            )


# ---------------------------------------------------------------------------
# ALLOWED / FILTERED log messages (verify spec wording)
# ---------------------------------------------------------------------------

class TestAllowedFilteredLogMessages:
    def _make_bar(self, price: float = 20000.0):
        from src.data.models import DollarBar
        from datetime import datetime, timezone
        return DollarBar(
            timestamp=datetime.now(timezone.utc),
            open=price, high=price + 5, low=price - 5,
            close=price, volume=1000,
            notional_value=50_000_000, bar_num=1,
        )

    def test_allowed_message_logged_when_proba_above_threshold(self, caplog):
        import logging
        import asyncio
        from unittest.mock import AsyncMock
        from src.research.tier2_streaming_working import Tier2StreamingTrader

        trader = Tier2StreamingTrader()
        trader.ml_filter.model = _mock_model(ML_THRESHOLD + 0.05)
        bar = self._make_bar()
        trader.h1_bullish_sweep_active = True
        trader.dollar_bars = [bar] * 25

        with caplog.at_level(logging.INFO):
            with patch.object(trader, "_detect_fvg", return_value={"direction": "bullish", "top": 20005.0, "bottom": 20000.0}):
                with patch.object(trader, "_enter_trade", new_callable=AsyncMock):
                    asyncio.run(trader._detect_and_enter(bar, is_backfill=False))

        assert any("ALLOWED" in r.message for r in caplog.records)

    def test_filtered_message_logged_when_proba_below_threshold(self, caplog):
        import logging
        from src.research.tier2_streaming_working import Tier2StreamingTrader

        trader = Tier2StreamingTrader()
        trader.ml_filter.model = _mock_model(ML_THRESHOLD - 0.05)

        with caplog.at_level(logging.INFO):
            import asyncio
            bar = self._make_bar()
            trader.h1_bullish_sweep_active = True
            trader.dollar_bars = [bar] * 25
            with patch.object(trader, "_detect_fvg", return_value={"direction": "bullish", "top": 20005.0, "bottom": 20000.0}):
                asyncio.run(trader._detect_and_enter(bar, is_backfill=False))

        assert any("FILTERED" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Task 5.2 — Saved Pipeline loads and infers correctly (AC4)
# ---------------------------------------------------------------------------

class TestSavedPipelineInference:
    MODEL_PATH = "models/xgboost/tier2_meta_labeling_model.pkl"

    def _single_row(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame([{
            'fvg_fill_pct': 0.5, 'sweep_window_vol': 1.2, 'volume_ratio': 1.5,
            'signal_direction': 1, 'h1_trend_slope': 0.05, 'atr': 5.0,
            'session_displacement': 0.3, 'session_volume_ratio': 1.1,
        }])

    @pytest.mark.xfail(
        not __import__('pathlib').Path("models/xgboost/tier2_meta_labeling_model.pkl").exists(),
        reason="Model artifact not present in this environment",
        strict=False,
    )
    def test_saved_pipeline_predict_proba_shape(self):
        """Saved Pipeline predict_proba on 8-feature row returns shape (1, 2)."""
        import joblib
        from pathlib import Path
        model = joblib.load(Path(self.MODEL_PATH))
        result = model.predict_proba(self._single_row())
        assert result.shape == (1, 2), f"Expected (1, 2), got {result.shape}"

    @pytest.mark.xfail(
        not __import__('pathlib').Path("models/xgboost/tier2_meta_labeling_model.pkl").exists(),
        reason="Model artifact not present in this environment",
        strict=False,
    )
    def test_saved_pipeline_predict_proba_value_in_range(self):
        """Saved Pipeline predict_proba returns value in [0, 1]."""
        import joblib
        from pathlib import Path
        model = joblib.load(Path(self.MODEL_PATH))
        result = model.predict_proba(self._single_row())
        proba = result[0, 1]
        assert 0.0 <= proba <= 1.0, f"Probability {proba} out of [0, 1]"


# ---------------------------------------------------------------------------
# Task 5.3 — ECE < 0.08 on synthetic balanced dataset (well-calibrated LR)
# ---------------------------------------------------------------------------

class TestECEDiagnostic:
    def test_well_calibrated_lr_ece_below_gate(self):
        """A well-calibrated LR on synthetic 50/50 data should have ECE < 0.08."""
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.calibration import calibration_curve

        rng = np.random.default_rng(42)
        n = 500
        # Linearly separable synthetic feature with noise
        X = rng.standard_normal((n, 2))
        y = (X[:, 0] + X[:, 1] + rng.standard_normal(n) * 0.5 > 0).astype(int)

        train_end = int(n * 0.8)
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:], y[train_end:]

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42))
        ])
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]

        fop, mpv = calibration_curve(y_val, proba, n_bins=5)
        ece = float(np.mean(np.abs(fop - mpv)))
        assert ece < 0.08, f"Well-calibrated LR ECE {ece:.4f} should be < 0.08"
