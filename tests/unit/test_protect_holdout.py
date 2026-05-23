"""Unit tests for protect_holdout.py (Story 3.1)."""

import os
import stat
from pathlib import Path

import pytest

from protect_holdout import init, verify


def make_csv(tmp_path: Path, name: str = "mnq_1min_holdout_20260301_plus.csv", mode: int = 0o444) -> Path:
    p = tmp_path / name
    p.write_text("timestamp,open,high,low,close,volume\n")
    os.chmod(p, mode)
    return p


class TestVerifyPass:
    def test_verify_pass_all_444(self, tmp_path):
        make_csv(tmp_path)
        assert verify(tmp_path) == 0

    def test_verify_pass_multiple_files(self, tmp_path):
        make_csv(tmp_path, name="mnq_1min_holdout_20260301_plus.csv")
        make_csv(tmp_path, name="mnq_1min_holdout_20260401_plus.csv")
        assert verify(tmp_path) == 0

    def test_verify_no_csvs_fails(self, tmp_path):
        assert verify(tmp_path) == 1

    def test_verify_pass_prints_count(self, tmp_path, capsys):
        make_csv(tmp_path)
        verify(tmp_path)
        out = capsys.readouterr().out
        assert "VERIFY PASS" in out
        assert "1 file(s)" in out


class TestVerifyFail:
    def test_verify_fail_one_writable(self, tmp_path, capsys):
        make_csv(tmp_path, mode=0o644)
        rc = verify(tmp_path)
        assert rc == 1
        assert "VERIFY FAIL" in capsys.readouterr().out

    def test_verify_fail_prints_filename(self, tmp_path, capsys):
        make_csv(tmp_path, name="mnq_1min_holdout_20260301_plus.csv", mode=0o644)
        verify(tmp_path)
        out = capsys.readouterr().out
        assert "mnq_1min_holdout_20260301_plus.csv" in out

    def test_verify_fail_mixed_modes(self, tmp_path, capsys):
        make_csv(tmp_path, name="mnq_1min_holdout_20260301_a.csv", mode=0o444)
        make_csv(tmp_path, name="mnq_1min_holdout_20260301_b.csv", mode=0o644)
        rc = verify(tmp_path)
        assert rc == 1


class TestVerifyDateValidation:
    def test_verify_date_valid(self, tmp_path):
        make_csv(tmp_path, name="mnq_1min_holdout_20260301_plus.csv")
        assert verify(tmp_path) == 0

    def test_verify_date_future_valid(self, tmp_path):
        make_csv(tmp_path, name="data_20260601.csv")
        assert verify(tmp_path) == 0

    def test_verify_date_invalid(self, tmp_path, capsys):
        make_csv(tmp_path, name="data_20251231.csv")
        rc = verify(tmp_path)
        assert rc == 1
        assert "predates cutoff" in capsys.readouterr().out

    def test_verify_date_invalid_exact_cutoff_minus_one(self, tmp_path, capsys):
        make_csv(tmp_path, name="data_20260229.csv")
        rc = verify(tmp_path)
        assert rc == 1
        assert "predates cutoff" in capsys.readouterr().out

    def test_verify_no_date_in_filename_passes(self, tmp_path):
        # File with no parseable date in name — date check skipped, permission check applies
        p = tmp_path / "holdout_data.csv"
        p.write_text("timestamp,open,high,low,close,volume\n")
        os.chmod(p, 0o444)
        assert verify(tmp_path) == 0


class TestInit:
    def test_init_idempotent(self, tmp_path):
        make_csv(tmp_path)  # already 444
        (tmp_path / "ACCESS_LOG.md").write_text("# Log\n")
        rc = init(tmp_path)
        assert rc == 0
        mode = stat.S_IMODE(os.stat(tmp_path / "mnq_1min_holdout_20260301_plus.csv").st_mode)
        assert mode == 0o444

    def test_init_protects_writable_csv(self, tmp_path):
        make_csv(tmp_path, mode=0o644)
        init(tmp_path)
        mode = stat.S_IMODE(os.stat(tmp_path / "mnq_1min_holdout_20260301_plus.csv").st_mode)
        assert mode == 0o444

    def test_init_creates_access_log(self, tmp_path):
        make_csv(tmp_path)
        init(tmp_path)
        log = tmp_path / "ACCESS_LOG.md"
        assert log.exists()

    def test_init_appends_to_existing_access_log(self, tmp_path):
        make_csv(tmp_path)
        log = tmp_path / "ACCESS_LOG.md"
        log.write_text("# Existing log\n")
        init(tmp_path)
        content = log.read_text()
        assert "Existing log" in content
        assert "Init —" in content

    def test_init_does_not_chmod_access_log(self, tmp_path):
        make_csv(tmp_path)
        log = tmp_path / "ACCESS_LOG.md"
        log.write_text("# Log\n")
        os.chmod(log, 0o644)
        init(tmp_path)
        mode = stat.S_IMODE(os.stat(log).st_mode)
        assert mode == 0o644  # access log must stay writable

    def test_init_exits_0(self, tmp_path):
        make_csv(tmp_path)
        assert init(tmp_path) == 0

    def test_init_run_twice_idempotent(self, tmp_path):
        make_csv(tmp_path)
        assert init(tmp_path) == 0
        assert init(tmp_path) == 0
        mode = stat.S_IMODE(os.stat(tmp_path / "mnq_1min_holdout_20260301_plus.csv").st_mode)
        assert mode == 0o444

    def test_init_verify_roundtrip(self, tmp_path):
        make_csv(tmp_path, mode=0o644)
        init(tmp_path)
        assert verify(tmp_path) == 0
