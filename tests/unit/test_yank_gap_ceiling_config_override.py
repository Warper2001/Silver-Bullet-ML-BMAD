"""YANK-scoped max_gap_atr_ratio override (ship-A, 2026-08-19).

strategy_config.yaml is SHARED with s26/s26-combine (same default-YAML resolution
order in _build_strategy_config-equivalent loaders). Setting max_gap_atr_ratio in
that YAML would silently re-denominate their gap ceiling too — never authorized by
the gap-ceiling prereg (PR #37/#44, YANK-only). So the fix is applied via an env
var (YANK_MAX_GAP_ATR_RATIO) read only inside yank_streaming_working.py, layered
on top of the shared YAML rather than editing it.
"""
import os

import pytest

from src.research.yank_streaming_working import _build_strategy_config


class TestGapCeilingEnvOverride:
    def test_no_env_var_leaves_default_off(self, monkeypatch):
        monkeypatch.delenv("YANK_MAX_GAP_ATR_RATIO", raising=False)
        monkeypatch.delenv("STRATEGY_CONFIG_PATH", raising=False)
        cfg = _build_strategy_config()
        assert cfg.max_gap_atr_ratio == 0.0

    def test_env_var_sets_the_ratio(self, monkeypatch):
        monkeypatch.setenv("YANK_MAX_GAP_ATR_RATIO", "0.426")
        monkeypatch.delenv("STRATEGY_CONFIG_PATH", raising=False)
        cfg = _build_strategy_config()
        assert cfg.max_gap_atr_ratio == pytest.approx(0.426)

    def test_env_var_does_not_disturb_other_fields(self, monkeypatch):
        """The override must be additive — every other field still comes from
        the shared YAML/dataclass defaults, unchanged."""
        monkeypatch.delenv("YANK_MAX_GAP_ATR_RATIO", raising=False)
        monkeypatch.delenv("STRATEGY_CONFIG_PATH", raising=False)
        baseline = _build_strategy_config()

        monkeypatch.setenv("YANK_MAX_GAP_ATR_RATIO", "0.426")
        overridden = _build_strategy_config()

        assert overridden.max_gap_dollars == baseline.max_gap_dollars
        assert overridden.min_gap_atr_ratio == baseline.min_gap_atr_ratio
        assert overridden.sl_multiplier == baseline.sl_multiplier
        assert overridden.tp_multiplier == baseline.tp_multiplier
        assert overridden.ml_threshold == baseline.ml_threshold
