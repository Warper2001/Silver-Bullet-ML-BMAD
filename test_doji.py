import sys
import asyncio
from datetime import datetime, timezone
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.research.tier2_streaming_working import MetaLabelingFilter, Tier2StreamingTrader, DollarBar, ML_MODEL_PATH
from sklearn.base import ClassifierMixin, BaseEstimator

class XGBClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model
        self.classes_ = model.classes_
        
    def fit(self, X, y=None):
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

async def test_doji():
    trader = Tier2StreamingTrader()
    
    # Mock data to reach extraction
    trader.dollar_bars = [
        DollarBar(timestamp=datetime.now(timezone.utc), open=100.0, high=101.0, low=99.0, close=100.5, volume=10, up_volume=5, down_volume=5, up_ticks=5, down_ticks=5, total_ticks=10, notional_value=1000)
        for _ in range(20)
    ]
    
    # Create doji
    doji_bar = DollarBar(timestamp=datetime.now(timezone.utc), open=100.0, high=100.0, low=100.0, close=100.0, volume=10, up_volume=5, down_volume=5, up_ticks=5, down_ticks=5, total_ticks=10, notional_value=1000)
    
    trader._daily_ranges = [100.0, 100.0]
    trader._session_open_price = 100.0
    trader._session_high = 110.0
    trader._session_low = 90.0
    trader._h1_slope = 0.5
    
    fvg = {"top": 105.0, "bottom": 102.0}
    
    try:
        features = trader._extract_features(trader.dollar_bars, doji_bar, fvg, "bullish")
        print("SUCCESS! Extracted features for doji bar:")
        print(f"bar_body_ratio = {features['bar_body_ratio']}")
    except Exception as e:
        print(f"FAILED! {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_doji())