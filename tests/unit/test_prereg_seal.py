"""Unit tests for prereg_seal.py (Story 3.2)."""

import dataclasses
import hashlib
import json
import os
from datetime import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.research.strategy_core import StrategyConfig


def make_protected_csv(
    tmp_path: Path,
    name: str = "mnq_1min_holdout_20260301_plus.csv",
    mode: int = 0o444,
) -> Path:
    p = tmp_path / name
    p.write_text("timestamp,open,high,low,close,volume\n")
    os.chmod(p, mode)
    return p


def make_access_log(tmp_path: Path) -> Path:
    p = tmp_path / "ACCESS_LOG.md"
    p.write_text("# Access Log\n")
    return p


class TestConfigToJson:
    def test_deterministic_same_config(self):
        from prereg_seal import _config_to_json

        config = StrategyConfig()
        assert _config_to_json(config) == _config_to_json(config)

    def test_time_fields_serialized_as_hhmm(self):
        from prereg_seal import _config_to_json

        config = StrategyConfig()
        j = _config_to_json(config)
        d = json.loads(j)
        assert d["kill_zone_start_et"] == "09:30"
        assert d["kill_zone_end_et"] == "11:00"

    def test_sorted_keys(self):
        from prereg_seal import _config_to_json

        config = StrategyConfig()
        j = _config_to_json(config)
        d = json.loads(j)
        keys = list(d.keys())
        assert keys == sorted(keys)

    def test_no_whitespace(self):
        from prereg_seal import _config_to_json

        config = StrategyConfig()
        j = _config_to_json(config)
        assert " " not in j

    def test_different_configs_produce_different_hashes(self):
        from prereg_seal import _config_to_json

        c1 = StrategyConfig(bearish_only=True)
        c2 = StrategyConfig(bearish_only=False)
        assert _config_to_json(c1) != _config_to_json(c2)


class TestExtractHoldoutDates:
    def test_single_csv_extracts_date(self, tmp_path):
        from prereg_seal import _extract_holdout_dates

        make_protected_csv(tmp_path, "mnq_1min_holdout_20260301_plus.csv")
        start, end = _extract_holdout_dates(tmp_path)
        assert start == "2026-03-01"
        assert end == "2026-03-01"

    def test_multiple_csvs_returns_min_max(self, tmp_path):
        from prereg_seal import _extract_holdout_dates

        make_protected_csv(tmp_path, "mnq_1min_holdout_20260301_a.csv")
        make_protected_csv(tmp_path, "mnq_1min_holdout_20260601_b.csv")
        start, end = _extract_holdout_dates(tmp_path)
        assert start == "2026-03-01"
        assert end == "2026-06-01"

    def test_no_csvs_returns_unknown(self, tmp_path):
        from prereg_seal import _extract_holdout_dates

        start, end = _extract_holdout_dates(tmp_path)
        assert start == "unknown"
        assert end == "unknown"

    def test_no_date_in_filename_returns_unknown(self, tmp_path):
        from prereg_seal import _extract_holdout_dates

        p = tmp_path / "holdout.csv"
        p.write_text("data\n")
        os.chmod(p, 0o444)
        start, end = _extract_holdout_dates(tmp_path)
        assert start == "unknown"
        assert end == "unknown"


class TestSeal:
    def test_seal_returns_0_on_success(self, tmp_path):
        from prereg_seal import seal

        make_protected_csv(tmp_path)
        make_access_log(tmp_path)
        output = tmp_path / "prereg.md"
        with patch("prereg_seal._git_head", return_value="abc123"), \
             patch("prereg_seal._git_is_dirty", return_value=False):
            rc = seal(
                StrategyConfig(), output, "test-exp",
                Path("src/research/strategy_core.py"), tmp_path,
            )
        assert rc == 0
        assert output.exists()

    def test_seal_document_contains_all_config_fields(self, tmp_path):
        from prereg_seal import seal

        make_protected_csv(tmp_path)
        output = tmp_path / "prereg.md"
        with patch("prereg_seal._git_head", return_value="abc123"), \
             patch("prereg_seal._git_is_dirty", return_value=False):
            seal(StrategyConfig(), output, "test-exp",
                 Path("src/research/strategy_core.py"), tmp_path)
        content = output.read_text()
        for field in dataclasses.fields(StrategyConfig()):
            assert field.name in content, f"Field {field.name!r} missing from document"

    def test_seal_document_contains_three_hashes(self, tmp_path):
        from prereg_seal import seal

        make_protected_csv(tmp_path)
        output = tmp_path / "prereg.md"
        with patch("prereg_seal._git_head", return_value="deadbeef"), \
             patch("prereg_seal._git_is_dirty", return_value=False):
            seal(StrategyConfig(), output, "test-exp",
                 Path("src/research/strategy_core.py"), tmp_path)
        content = output.read_text()
        assert "(a) StrategyConfig SHA-256" in content
        assert "(b) strategy_core.py SHA-256" in content
        assert "(c) Git HEAD commit" in content
        assert "deadbeef" in content

    def test_seal_document_contains_success_metrics(self, tmp_path):
        from prereg_seal import seal

        make_protected_csv(tmp_path)
        output = tmp_path / "prereg.md"
        with patch("prereg_seal._git_head", return_value="abc123"), \
             patch("prereg_seal._git_is_dirty", return_value=False):
            seal(StrategyConfig(), output, "test-exp",
                 Path("src/research/strategy_core.py"), tmp_path)
        content = output.read_text()
        assert "PF" in content
        assert "2.0" in content
        assert "Sharpe" in content
        assert "1.5" in content
        assert "200" in content  # min sample size
        assert "100" in content  # stopping rule trades

    def test_seal_document_contains_holdout_date(self, tmp_path):
        from prereg_seal import seal

        make_protected_csv(tmp_path, "mnq_1min_holdout_20260301_plus.csv")
        output = tmp_path / "prereg.md"
        with patch("prereg_seal._git_head", return_value="abc123"), \
             patch("prereg_seal._git_is_dirty", return_value=False):
            seal(StrategyConfig(), output, "test-exp",
                 Path("src/research/strategy_core.py"), tmp_path)
        content = output.read_text()
        assert "2026-03-01" in content

    def test_seal_exits_1_if_holdout_unprotected(self, tmp_path, capsys):
        from prereg_seal import seal

        make_protected_csv(tmp_path, mode=0o644)  # writable — unprotected
        output = tmp_path / "prereg.md"
        with patch("prereg_seal._git_head", return_value="abc123"), \
             patch("prereg_seal._git_is_dirty", return_value=False):
            rc = seal(StrategyConfig(), output, "test-exp",
                      Path("src/research/strategy_core.py"), tmp_path)
        assert rc == 1
        assert not output.exists()
        assert "ERROR" in capsys.readouterr().out

    def test_seal_warns_on_dirty_tree(self, tmp_path, capsys):
        from prereg_seal import seal

        make_protected_csv(tmp_path)
        output = tmp_path / "prereg.md"
        with patch("prereg_seal._git_head", return_value="abc123"), \
             patch("prereg_seal._git_is_dirty", return_value=True):
            rc = seal(StrategyConfig(), output, "test-exp",
                      Path("src/research/strategy_core.py"), tmp_path)
        assert rc == 0  # warning, not error
        out = capsys.readouterr().out
        assert "WARNING" in out
        assert "dirty" in out

    def test_seal_hash_a_deterministic(self, tmp_path):
        from prereg_seal import seal, _config_to_json

        config = StrategyConfig()
        j = _config_to_json(config)
        h1 = hashlib.sha256(j.encode()).hexdigest()
        h2 = hashlib.sha256(j.encode()).hexdigest()
        assert h1 == h2

    def test_seal_creates_parent_dirs(self, tmp_path):
        from prereg_seal import seal

        make_protected_csv(tmp_path)
        output = tmp_path / "nested" / "deep" / "prereg.md"
        with patch("prereg_seal._git_head", return_value="abc123"), \
             patch("prereg_seal._git_is_dirty", return_value=False):
            rc = seal(StrategyConfig(), output, "test-exp",
                      Path("src/research/strategy_core.py"), tmp_path)
        assert rc == 0
        assert output.exists()


class TestConfigJsonOverride:
    def test_build_config_defaults(self):
        from prereg_seal import _build_config

        config = _build_config(None)
        assert config == StrategyConfig()

    def test_build_config_override_scalar(self):
        from prereg_seal import _build_config

        config = _build_config('{"bearish_only": false}')
        assert config.bearish_only is False
        assert config.sl_multiplier == 5.0  # unchanged

    def test_build_config_override_time_field(self):
        from prereg_seal import _build_config

        config = _build_config('{"kill_zone_start_et": "10:00"}')
        assert config.kill_zone_start_et == time(10, 0)

    def test_build_config_invalid_json_raises(self):
        from prereg_seal import _build_config

        with pytest.raises((json.JSONDecodeError, ValueError)):
            _build_config("{not valid json}")
