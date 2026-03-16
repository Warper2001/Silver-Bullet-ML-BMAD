"""Pattern detection components for ICT Silver Bullet setups.

This package contains pattern detection algorithms for identifying:
- Market Structure Shifts (MSS)
- Fair Value Gaps (FVG)
- Liquidity Sweeps
- Silver Bullet Setups
"""

from src.detection.fvg_detection import (
    check_fvg_fill,
    detect_bearish_fvg,
    detect_bullish_fvg,
)
from src.detection.fvg_detector import FVGDetector
from src.detection.liquidity_sweep_detection import (
    check_bearish_sweep,
    check_bullish_sweep,
    detect_bearish_liquidity_sweep,
    detect_bullish_liquidity_sweep,
)
from src.detection.liquidity_sweep_detector import LiquiditySweepDetector
from src.detection.mss_detector import MSSDetector
from src.detection.silver_bullet_detection import (
    check_silver_bullet_setup,
    detect_silver_bullet_setup,
)
from src.detection.silver_bullet_detector import SilverBulletDetector
from src.detection.swing_detection import (
    RollingVolumeAverage,
    detect_bearish_mss,
    detect_bullish_mss,
    detect_swing_high,
    detect_swing_low,
)
from src.detection.confidence_scorer import (
    calculate_confidence_score,
    score_setup,
)
from src.detection.time_window_filter import (
    DEFAULT_TRADING_WINDOWS,
    LONDON_AM,
    NY_AM,
    NY_PM,
    check_time_window,
    filter_setups_by_time_window,
    is_within_trading_hours,
)

__all__ = [
    "MSSDetector",
    "FVGDetector",
    "LiquiditySweepDetector",
    "SilverBulletDetector",
    "detect_swing_high",
    "detect_swing_low",
    "detect_bullish_mss",
    "detect_bearish_mss",
    "RollingVolumeAverage",
    "detect_bullish_fvg",
    "detect_bearish_fvg",
    "check_fvg_fill",
    "detect_bullish_liquidity_sweep",
    "detect_bearish_liquidity_sweep",
    "check_bullish_sweep",
    "check_bearish_sweep",
    "check_silver_bullet_setup",
    "detect_silver_bullet_setup",
    "calculate_confidence_score",
    "score_setup",
    "is_within_trading_hours",
    "check_time_window",
    "filter_setups_by_time_window",
    "DEFAULT_TRADING_WINDOWS",
    "LONDON_AM",
    "NY_AM",
    "NY_PM",
]
