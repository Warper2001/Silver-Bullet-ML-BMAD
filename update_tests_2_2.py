import re
import sys

def update_file(path):
    with open(path, 'r') as f:
        content = f.read()

    # Replace attribute access: weights.strategy_name -> weights.strategies["strategy_name"]
    strategies = [
        "triple_confluence_scaler", "wolf_pack_3_edge", "adaptive_ema_momentum",
        "vwap_bounce", "opening_range_breakout"
    ]
    for s in strategies:
        content = content.replace(f"weights.{s}", f'weights.strategies["{s}"]')

    # Replace constructor calls StrategyWeights(triple_confluence_scaler=0.2, ...)
    # This is harder because of multiline. We'll use a regex for common patterns.
    
    # Pattern for equal weights
    equal_weights_old = """StrategyWeights(
            triple_confluence_scaler=0.20,
            wolf_pack_3_edge=0.20,
            adaptive_ema_momentum=0.20,
            vwap_bounce=0.20,
            opening_range_breakout=0.20,
        )"""
    equal_weights_new = """StrategyWeights(
            strategies={
                "triple_confluence_scaler": 0.20,
                "wolf_pack_3_edge": 0.20,
                "adaptive_ema_momentum": 0.20,
                "vwap_bounce": 0.20,
                "opening_range_breakout": 0.20,
            }
        )"""
    content = content.replace(equal_weights_old, equal_weights_new)

    # Manual fix for other common patterns in unit tests
    content = re.sub(r'StrategyWeights\.model_construct\(\s+triple_confluence_scaler=([\d\.]+),\s+wolf_pack_3_edge=([\d\.]+),\s+adaptive_ema_momentum=([\d\.]+),\s+vwap_bounce=([\d\.]+),\s+opening_range_breakout=([\d\.]+),?\s+\)',
                     r'StrategyWeights.model_construct(strategies={"triple_confluence_scaler": \1, "wolf_pack_3_edge": \2, "adaptive_ema_momentum": \3, "vwap_bounce": \4, "opening_range_breakout": \5})',
                     content, flags=re.MULTILINE)

    # Generic StrategyWeights(...) with 5 fields
    content = re.sub(r'StrategyWeights\(\s+triple_confluence_scaler=([\d\.]+),\s+wolf_pack_3_edge=([\d\.]+),\s+adaptive_ema_momentum=([\d\.]+),\s+vwap_bounce=([\d\.]+),\s+opening_range_breakout=([\d\.]+),?\s+\)',
                     r'StrategyWeights(strategies={"triple_confluence_scaler": \1, "wolf_pack_3_edge": \2, "adaptive_ema_momentum": \3, "vwap_bounce": \4, "opening_range_breakout": \5})',
                     content, flags=re.MULTILINE)

    with open(path, 'w') as f:
        f.write(content)

update_file("tests/unit/test_weighted_scorer.py")
update_file("tests/integration/test_weighted_scoring_integration.py")
