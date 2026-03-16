"""Pattern detection components for ICT Silver Bullet setups.

This package contains pattern detection algorithms for identifying:
- Market Structure Shifts (MSS)
- Fair Value Gaps (FVG)
- Liquidity Sweeps
- Silver Bullet Setups
"""

from src.detection.mss_detector import MSSDetector
from src.detection.swing_detection import (
    RollingVolumeAverage,
    detect_bearish_mss,
    detect_bullish_mss,
    detect_swing_high,
    detect_swing_low,
)

__all__ = [
    "MSSDetector",
    "detect_swing_high",
    "detect_swing_low",
    "detect_bullish_mss",
    "detect_bearish_mss",
    "RollingVolumeAverage",
]
