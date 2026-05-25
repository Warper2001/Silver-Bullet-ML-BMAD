"""Unit tests for oos_verdict.py (Story 3.4).

Tests never touch real data/sealed_holdout/ or real reports.
All holdout access, BacktestEngine, and checkpoint_or_abort are mocked.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def make_trades(n_wins: int, n_losses: int, win_pnl: float = 200.0, loss_pnl: float = -100.0):
    """Return a list of MagicMock trade objects with pnl_usd and timestamp_exit."""
    trades = []
    base = pd.Timestamp("2026-03-05", tz="America/New_York")
    for i in range(n_wins + n_losses):
        t = MagicMock()
        t.pnl_usd = win_pnl if i < n_wins else loss_pnl
        # Spread across distinct days so each trade is a separate daily return
        t.timestamp_exit = base + pd.Timedelta(days=i)
        trades.append(t)
    return trades


def make_prereg_doc(hash_a: str = "a" * 64, hash_b: str = "b" * 64, hash_c: str = "c" * 40) -> str:
    return f"""# Pre-Registration: test

## Integrity Hashes

| Hash | Value |
|---|---|
| (a) StrategyConfig SHA-256 | `{hash_a}` |
| (b) strategy_core.py SHA-256 | `{hash_b}` |
| (c) Git HEAD commit | `{hash_c}` |
"""


def make_prereg_file(tmp_path: Path, hash_c: str = "c" * 40) -> Path:
    p = tmp_path / "prereg.md"
    p.write_text(make_prereg_doc(hash_c=hash_c))
    return p


def make_tmp_dirs(tmp_path: Path):
    """Return (access_log_path, reports_dir) with initialized access log."""
    access_log = tmp_path / "ACCESS_LOG.md"
    # Minimal pre-existing log so appended row is valid
    access_log.write_text(
        "| Date | SHA (pre-registration) | Accessor | Purpose | Result |\n"
        "|---|---|---|---|---|\n"
    )
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    return access_log, reports_dir


# ---------------------------------------------------------------------------
# AC #1 — checkpoint_or_abort called first; abort on failure
# ---------------------------------------------------------------------------

def test_checkpoint_called_first_on_failure(tmp_path):
    """If checkpoint_or_abort raises SystemExit(1), BacktestEngine is never called."""
    prereg = make_prereg_file(tmp_path)
    access_log, reports_dir = make_tmp_dirs(tmp_path)

    from oos_verdict import verdict

    with (
        patch("oos_verdict.checkpoint_or_abort", side_effect=SystemExit(1)),
        patch("oos_verdict.BacktestEngine") as mock_engine,
        pytest.raises(SystemExit) as exc_info,
    ):
        verdict(prereg, holdout_csv=tmp_path / "fake.csv", access_log=access_log, reports_dir=reports_dir)

    assert exc_info.value.code == 1
    mock_engine.assert_not_called()


# ---------------------------------------------------------------------------
# AC #4 — verdict logic: GO / NO-GO / INCONCLUSIVE
# ---------------------------------------------------------------------------

def test_verdict_go(tmp_path):
    """N=260, small losses after big win run → PF=50, MaxDD=2%, Sharpe high → GO."""
    prereg = make_prereg_file(tmp_path)
    access_log, reports_dir = make_tmp_dirs(tmp_path)
    # 250 wins x $200 then 10 losses x $100: PF=50, MaxDD=2%, Sharpe very high
    trades = make_trades(250, 10, win_pnl=200.0, loss_pnl=-100.0)

    from oos_verdict import verdict

    with (
        patch("oos_verdict.checkpoint_or_abort"),
        patch("oos_verdict.BacktestEngine") as mock_engine,
    ):
        mock_engine.return_value.run.return_value = trades
        rc = verdict(prereg, holdout_csv=tmp_path / "fake.csv", access_log=access_log, reports_dir=reports_dir)

    assert rc == 0
    log_text = access_log.read_text()
    assert "| GO:" in log_text  # "| GO:" does not appear in "| NO-GO:" so this is non-vacuous


def test_verdict_no_go_pf(tmp_path):
    """N=300, PF=1.143 (above stopping rule 1.1 but below threshold 2.0) → NO-GO."""
    prereg = make_prereg_file(tmp_path)
    access_log, reports_dir = make_tmp_dirs(tmp_path)
    # 160 wins x $100, 140 losses x $100: PF=16000/14000=1.143 (> 1.1, < 2.0)
    trades = make_trades(160, 140, win_pnl=100.0, loss_pnl=-100.0)

    from oos_verdict import verdict

    with (
        patch("oos_verdict.checkpoint_or_abort"),
        patch("oos_verdict.BacktestEngine") as mock_engine,
    ):
        mock_engine.return_value.run.return_value = trades
        rc = verdict(prereg, holdout_csv=tmp_path / "fake.csv", access_log=access_log, reports_dir=reports_dir)

    assert rc == 1
    assert "NO-GO" in access_log.read_text()


def test_verdict_no_go_sharpe(tmp_path):
    """Sharpe < 1.5 at N=300, PF >= 2.0 → NO-GO (mock _compute_metrics for isolation)."""
    prereg = make_prereg_file(tmp_path)
    access_log, reports_dir = make_tmp_dirs(tmp_path)
    trades = make_trades(200, 10, win_pnl=200.0, loss_pnl=-100.0)  # placeholder

    from oos_verdict import verdict

    with (
        patch("oos_verdict.checkpoint_or_abort"),
        patch("oos_verdict.BacktestEngine") as mock_engine,
        patch("oos_verdict._compute_metrics", return_value=(2.5, 1.0, 0.05, 300)),
    ):
        mock_engine.return_value.run.return_value = trades
        rc = verdict(prereg, holdout_csv=tmp_path / "fake.csv", access_log=access_log, reports_dir=reports_dir)

    assert rc == 1
    assert "NO-GO" in access_log.read_text()


def test_verdict_no_go_maxdd(tmp_path):
    """N=300, MaxDD=12.5% > 10% → NO-GO (200 wins then 100 losses creates positive-equity drawdown)."""
    prereg = make_prereg_file(tmp_path)
    access_log, reports_dir = make_tmp_dirs(tmp_path)

    # 200 wins x $400 then 100 losses x $100:
    # PF=80000/10000=8.0 (passes), equity peaks at $80000 then drops to $70000
    # MaxDD = (80000-70000)/80000 = 12.5% > 10% → NO-GO
    trades = make_trades(200, 100, win_pnl=400.0, loss_pnl=-100.0)

    from oos_verdict import verdict

    with (
        patch("oos_verdict.checkpoint_or_abort"),
        patch("oos_verdict.BacktestEngine") as mock_engine,
    ):
        mock_engine.return_value.run.return_value = trades
        rc = verdict(prereg, holdout_csv=tmp_path / "fake.csv", access_log=access_log, reports_dir=reports_dir)

    assert rc == 1
    assert "NO-GO" in access_log.read_text()


def test_verdict_inconclusive(tmp_path):
    """N=150 → INCONCLUSIVE regardless of metrics."""
    prereg = make_prereg_file(tmp_path)
    access_log, reports_dir = make_tmp_dirs(tmp_path)
    trades = make_trades(100, 50, win_pnl=300.0, loss_pnl=-50.0)  # PF=6.0, great metrics

    from oos_verdict import verdict

    with (
        patch("oos_verdict.checkpoint_or_abort"),
        patch("oos_verdict.BacktestEngine") as mock_engine,
    ):
        mock_engine.return_value.run.return_value = trades
        rc = verdict(prereg, holdout_csv=tmp_path / "fake.csv", access_log=access_log, reports_dir=reports_dir)

    assert rc == 1
    assert "INCONCLUSIVE" in access_log.read_text()


# ---------------------------------------------------------------------------
# AC #5 — stopping rule
# ---------------------------------------------------------------------------

def test_verdict_stopping_rule_at_105_pf_1_04(tmp_path):
    """AC #5 exact: N=105, PF=1.04 → STOPPING_RULE_TRIGGERED (mock metrics for precision)."""
    prereg = make_prereg_file(tmp_path)
    access_log, reports_dir = make_tmp_dirs(tmp_path)
    trades = make_trades(60, 45)  # placeholder — metrics overridden below

    from oos_verdict import verdict

    with (
        patch("oos_verdict.checkpoint_or_abort"),
        patch("oos_verdict.BacktestEngine") as mock_engine,
        patch("oos_verdict._compute_metrics", return_value=(1.04, 2.0, 0.05, 105)),
    ):
        mock_engine.return_value.run.return_value = trades
        rc = verdict(prereg, holdout_csv=tmp_path / "fake.csv", access_log=access_log, reports_dir=reports_dir)

    assert rc == 1
    assert "STOPPING_RULE_TRIGGERED" in access_log.read_text()


def test_stopping_rule_overrides_inconclusive(tmp_path):
    """N=105, PF<1.1 → STOPPING_RULE_TRIGGERED, not INCONCLUSIVE."""
    prereg = make_prereg_file(tmp_path)
    access_log, reports_dir = make_tmp_dirs(tmp_path)
    trades = make_trades(50, 55, win_pnl=100.0, loss_pnl=-100.0)  # N=105, PF<1.0

    from oos_verdict import verdict

    with (
        patch("oos_verdict.checkpoint_or_abort"),
        patch("oos_verdict.BacktestEngine") as mock_engine,
    ):
        mock_engine.return_value.run.return_value = trades
        rc = verdict(prereg, holdout_csv=tmp_path / "fake.csv", access_log=access_log, reports_dir=reports_dir)

    assert rc == 1
    log = access_log.read_text()
    assert "STOPPING_RULE_TRIGGERED" in log
    assert "INCONCLUSIVE" not in log


def test_stopping_rule_not_triggered_below_100(tmp_path):
    """N=99, PF<1.1 → INCONCLUSIVE (stopping rule requires N≥100)."""
    prereg = make_prereg_file(tmp_path)
    access_log, reports_dir = make_tmp_dirs(tmp_path)
    trades = make_trades(45, 54, win_pnl=100.0, loss_pnl=-100.0)  # N=99, PF<1.0

    from oos_verdict import verdict

    with (
        patch("oos_verdict.checkpoint_or_abort"),
        patch("oos_verdict.BacktestEngine") as mock_engine,
    ):
        mock_engine.return_value.run.return_value = trades
        rc = verdict(prereg, holdout_csv=tmp_path / "fake.csv", access_log=access_log, reports_dir=reports_dir)

    assert rc == 1
    assert "INCONCLUSIVE" in access_log.read_text()


# ---------------------------------------------------------------------------
# AC #2 — ACCESS_LOG appended
# ---------------------------------------------------------------------------

def test_access_log_appended(tmp_path):
    """ACCESS_LOG gets a new row containing the verdict string."""
    prereg = make_prereg_file(tmp_path)
    access_log, reports_dir = make_tmp_dirs(tmp_path)
    trades = make_trades(100, 50, win_pnl=300.0, loss_pnl=-100.0)

    from oos_verdict import verdict

    with (
        patch("oos_verdict.checkpoint_or_abort"),
        patch("oos_verdict.BacktestEngine") as mock_engine,
    ):
        mock_engine.return_value.run.return_value = trades
        verdict(prereg, holdout_csv=tmp_path / "fake.csv", access_log=access_log, reports_dir=reports_dir)

    log_text = access_log.read_text()
    assert "oos_verdict.py" in log_text
    assert "OOS verdict run" in log_text
    # The log row should contain the verdict keyword
    assert any(v in log_text for v in ("GO", "NO-GO", "INCONCLUSIVE", "STOPPING_RULE_TRIGGERED"))


# ---------------------------------------------------------------------------
# AC #6 — report file created
# ---------------------------------------------------------------------------

def test_report_file_created(tmp_path):
    """Verdict report file is written to reports_dir and contains required sections."""
    prereg = make_prereg_file(tmp_path, hash_c="d" * 40)
    access_log, reports_dir = make_tmp_dirs(tmp_path)
    trades = make_trades(200, 100, win_pnl=300.0, loss_pnl=-100.0)

    from oos_verdict import verdict

    with (
        patch("oos_verdict.checkpoint_or_abort"),
        patch("oos_verdict.BacktestEngine") as mock_engine,
    ):
        mock_engine.return_value.run.return_value = trades
        verdict(prereg, holdout_csv=tmp_path / "fake.csv", access_log=access_log, reports_dir=reports_dir)

    report_files = list(reports_dir.glob("oos_verdict_*.md"))
    assert len(report_files) == 1
    content = report_files[0].read_text()
    assert "Profit Factor" in content
    assert "VERDICT" in content
    assert "d" * 40 in content  # hash_c appears in report


# ---------------------------------------------------------------------------
# AC #3 — metric computation details
# ---------------------------------------------------------------------------

def test_compute_metrics_daily_grouping(tmp_path):
    """Two trades on the same date → one daily return entry, not two."""
    from oos_verdict import _compute_metrics

    base = pd.Timestamp("2026-03-05", tz="America/New_York")
    # Both trades on the same day
    t1, t2 = MagicMock(), MagicMock()
    t1.pnl_usd = 100.0
    t1.timestamp_exit = base
    t2.pnl_usd = 200.0
    t2.timestamp_exit = base  # same day

    pf, sharpe, max_dd_pct, n = _compute_metrics([t1, t2])

    # N should be 2 (trade count, not day count)
    assert n == 2
    # With only 1 daily return value, sharpe must be 0.0 (< 2 samples)
    assert sharpe == 0.0
    # PF: gross_win=300, gross_loss=0 → inf
    assert pf == float("inf")


def test_zero_trades_inconclusive(tmp_path):
    """Empty trade list → INCONCLUSIVE verdict, N=0, no crash."""
    prereg = make_prereg_file(tmp_path)
    access_log, reports_dir = make_tmp_dirs(tmp_path)

    from oos_verdict import verdict

    with (
        patch("oos_verdict.checkpoint_or_abort"),
        patch("oos_verdict.BacktestEngine") as mock_engine,
    ):
        mock_engine.return_value.run.return_value = []
        rc = verdict(prereg, holdout_csv=tmp_path / "fake.csv", access_log=access_log, reports_dir=reports_dir)

    assert rc == 1
    assert "INCONCLUSIVE" in access_log.read_text()


# ---------------------------------------------------------------------------
# AR8 ordering — ACCESS_LOG written before report file
# ---------------------------------------------------------------------------

def test_access_log_appended_before_report(tmp_path, monkeypatch):
    """ACCESS_LOG append happens before the verdict report is written (AR8 ordering)."""
    prereg = make_prereg_file(tmp_path)
    access_log, reports_dir = make_tmp_dirs(tmp_path)
    trades = make_trades(200, 100, win_pnl=300.0, loss_pnl=-100.0)

    call_order = []

    original_open = open

    def tracking_open(path, mode="r", *args, **kwargs):
        p = Path(str(path))
        if mode == "a" and p == access_log:
            call_order.append("log")
        elif mode == "w" and str(p).startswith(str(reports_dir)):
            call_order.append("report")
        return original_open(path, mode, *args, **kwargs)

    from oos_verdict import verdict

    with (
        patch("oos_verdict.checkpoint_or_abort"),
        patch("oos_verdict.BacktestEngine") as mock_engine,
        patch("builtins.open", side_effect=tracking_open),
    ):
        mock_engine.return_value.run.return_value = trades
        verdict(prereg, holdout_csv=tmp_path / "fake.csv", access_log=access_log, reports_dir=reports_dir)

    # ACCESS_LOG append must appear before report write
    assert "log" in call_order
    assert "report" in call_order
    assert call_order.index("log") < call_order.index("report")


# ---------------------------------------------------------------------------
# _determine_verdict unit tests
# ---------------------------------------------------------------------------

def test_verdict_no_go_initial_loss_streak(tmp_path):
    """MaxDD gate catches initial losing streak that calc_max_drawdown_pct would miss."""
    prereg = make_prereg_file(tmp_path)
    access_log, reports_dir = make_tmp_dirs(tmp_path)

    # 50 losses × $1000 first → equity dips to -$50K
    # then 250 wins × $300 → equity peaks at -$50K + $75K = $25K
    # Supplemental MaxDD = abs(-50K) / 25K = 200% >> 10% → NO-GO
    losses = make_trades(0, 50, loss_pnl=-1000.0)
    wins = make_trades(250, 0, win_pnl=300.0)
    trades = losses + wins

    from oos_verdict import verdict

    with (
        patch("oos_verdict.checkpoint_or_abort"),
        patch("oos_verdict.BacktestEngine") as mock_engine,
    ):
        mock_engine.return_value.run.return_value = trades
        rc = verdict(prereg, holdout_csv=tmp_path / "fake.csv", access_log=access_log, reports_dir=reports_dir)

    assert rc == 1
    assert "NO-GO" in access_log.read_text()


def test_engine_run_exception_logs_access(tmp_path):
    """If engine.run() raises, ACCESS_LOG is still written with ERROR entry (AR8)."""
    prereg = make_prereg_file(tmp_path)
    access_log, reports_dir = make_tmp_dirs(tmp_path)

    from oos_verdict import verdict

    with (
        patch("oos_verdict.checkpoint_or_abort"),
        patch("oos_verdict.BacktestEngine") as mock_engine,
        pytest.raises(RuntimeError),
    ):
        mock_engine.return_value.run.side_effect = RuntimeError("disk read error")
        verdict(prereg, holdout_csv=tmp_path / "fake.csv", access_log=access_log, reports_dir=reports_dir)

    log_text = access_log.read_text()
    assert "oos_verdict.py" in log_text
    assert "ERROR" in log_text


def test_determine_verdict_logic():
    """Direct unit test of _determine_verdict for all branches."""
    from oos_verdict import _determine_verdict

    # Stopping rule
    assert _determine_verdict(1.04, 3.0, 0.05, 105) == "STOPPING_RULE_TRIGGERED"
    assert _determine_verdict(0.5, 0.0, 0.50, 100) == "STOPPING_RULE_TRIGGERED"
    assert _determine_verdict(1.09, 2.0, 0.05, 100) == "STOPPING_RULE_TRIGGERED"
    # Exactly at boundary: PF=1.1 → NOT triggered
    assert _determine_verdict(1.10, 0.5, 0.50, 100) != "STOPPING_RULE_TRIGGERED"

    # INCONCLUSIVE (stopping rule not triggered)
    assert _determine_verdict(5.0, 5.0, 0.01, 199) == "INCONCLUSIVE"
    assert _determine_verdict(0.0, 0.0, 0.0, 0) == "INCONCLUSIVE"

    # GO
    assert _determine_verdict(2.0, 1.5, 0.10, 200) == "GO"
    assert _determine_verdict(3.0, 2.0, 0.05, 300) == "GO"

    # NO-GO (various failures at N>=200)
    assert _determine_verdict(1.9, 2.0, 0.05, 200) == "NO-GO"   # PF fails
    assert _determine_verdict(2.0, 1.4, 0.05, 200) == "NO-GO"   # Sharpe fails
    assert _determine_verdict(2.0, 2.0, 0.11, 200) == "NO-GO"   # MaxDD fails


def test_verdict_creates_reports_dir_if_missing(tmp_path):
    """verdict() auto-creates reports_dir if it does not exist (no FileNotFoundError)."""
    prereg = make_prereg_file(tmp_path)
    access_log = tmp_path / "ACCESS_LOG.md"
    access_log.write_text("| Date | SHA | Accessor | Purpose | Result |\n|---|---|---|---|---|\n")
    # reports_dir intentionally NOT created
    reports_dir = tmp_path / "new_reports_dir"
    assert not reports_dir.exists()

    trades = make_trades(200, 100, win_pnl=300.0, loss_pnl=-100.0)

    from oos_verdict import verdict

    with (
        patch("oos_verdict.checkpoint_or_abort"),
        patch("oos_verdict.BacktestEngine") as mock_engine,
    ):
        mock_engine.return_value.run.return_value = trades
        rc = verdict(prereg, holdout_csv=tmp_path / "fake.csv", access_log=access_log, reports_dir=reports_dir)

    assert reports_dir.exists()
    assert list(reports_dir.glob("oos_verdict_*.md"))  # report was written
