"""Unit tests for src/research/config_loader.py (Story 8.2)."""

from datetime import time
from pathlib import Path

import pytest
import yaml

from src.research.strategy_core import StrategyConfig


def write_yaml(tmp_path: Path, content: dict, filename: str = "cfg.yaml") -> Path:
    p = tmp_path / filename
    p.write_text(yaml.dump(content))
    return p


class TestLoadStrategyConfig:
    def test_load_returns_strategyconfig_instance(self, tmp_path):
        from src.research.config_loader import load_strategy_config

        p = write_yaml(tmp_path, {})
        result = load_strategy_config(p)
        assert isinstance(result, StrategyConfig)

    def test_load_empty_yaml_returns_defaults(self, tmp_path):
        from src.research.config_loader import load_strategy_config

        p = tmp_path / "empty.yaml"
        p.write_text("")
        result = load_strategy_config(p)
        assert result == StrategyConfig()

    def test_load_default_yaml_matches_strategyconfig_defaults(self, tmp_path):
        from src.research.config_loader import load_strategy_config

        content = {
            "sl_multiplier": 5.0,
            "tp_multiplier": 6.0,
            "entry_pct": 0.5,
            "atr_threshold": 0.5,
            "max_gap_dollars": 60.0,
            "max_hold_bars": 60,
            "max_pending_bars": 240,
            "contracts_per_trade": 5,
            "max_daily_loss": -750.0,
            "vol_regime_lookback": 120,
            "vol_regime_threshold": 0.75,
            "min_gap_atr_ratio": 0.25,
            "ml_threshold": 0.0,
            "bearish_only": True,
            "h1_sweep_lookback": 6,
            "kill_zone_start_et": "09:30",
            "kill_zone_end_et": "11:00",
            "commission_per_roundtrip": 4.0,
            "enable_kill_zone_filter": False,
            "m15_confirmation": False,
            "tuesday_exclusion": True,
        }
        p = write_yaml(tmp_path, content)
        result = load_strategy_config(p)
        assert result == StrategyConfig()

    def test_load_override_single_field(self, tmp_path):
        from src.research.config_loader import load_strategy_config

        p = write_yaml(tmp_path, {"min_gap_atr_ratio": 0.30})
        result = load_strategy_config(p)
        assert result.min_gap_atr_ratio == 0.30
        assert result.bearish_only is True  # unchanged default

    def test_load_partial_yaml_uses_defaults_for_missing(self, tmp_path):
        from src.research.config_loader import load_strategy_config

        p = write_yaml(tmp_path, {"sl_multiplier": 3.0})
        result = load_strategy_config(p)
        assert result.sl_multiplier == 3.0
        assert result.tp_multiplier == 6.0   # default
        assert result.bearish_only is True    # default
        assert result.tuesday_exclusion is True  # default

    def test_load_unknown_keys_silently_ignored(self, tmp_path):
        from src.research.config_loader import load_strategy_config

        p = write_yaml(tmp_path, {"nonexistent_key": 999, "sl_multiplier": 4.0})
        result = load_strategy_config(p)
        assert result.sl_multiplier == 4.0
        assert not hasattr(result, "nonexistent_key")

    def test_load_time_field_string_parsed_start(self, tmp_path):
        from src.research.config_loader import load_strategy_config

        p = write_yaml(tmp_path, {"kill_zone_start_et": "10:00"})
        result = load_strategy_config(p)
        assert result.kill_zone_start_et == time(10, 0)

    def test_load_time_field_string_parsed_end(self, tmp_path):
        from src.research.config_loader import load_strategy_config

        p = write_yaml(tmp_path, {"kill_zone_end_et": "12:30"})
        result = load_strategy_config(p)
        assert result.kill_zone_end_et == time(12, 30)

    def test_load_missing_file_raises(self, tmp_path):
        from src.research.config_loader import load_strategy_config

        with pytest.raises(FileNotFoundError):
            load_strategy_config(tmp_path / "nonexistent.yaml")

    def test_load_invalid_yaml_raises(self, tmp_path):
        from src.research.config_loader import load_strategy_config

        p = tmp_path / "bad.yaml"
        p.write_text("key: [unclosed bracket")
        with pytest.raises(yaml.YAMLError):
            load_strategy_config(p)

    def test_load_repo_strategy_config_yaml(self):
        """The actual repo-root strategy_config.yaml loads without error."""
        from src.research.config_loader import load_strategy_config

        repo_yaml = Path("strategy_config.yaml")
        if not repo_yaml.exists():
            pytest.skip("strategy_config.yaml not found at repo root")
        result = load_strategy_config(repo_yaml)
        assert isinstance(result, StrategyConfig)
        assert result.min_gap_atr_ratio == 0.25
        assert result.bearish_only is True
        assert result.tuesday_exclusion is True
        assert result.sl_multiplier == 5.0
        assert result.tp_multiplier == 6.0
