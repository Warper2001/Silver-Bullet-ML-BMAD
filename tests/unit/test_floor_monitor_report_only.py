"""Floor monitor report-only mode.

Alex removed floor-derived braking from both bots on 2026-07-29 and this service was
stopped, which also blinded the account readout: nothing tracked the high-water mark, so
when the balance made new highs on 08-04 the recorded floor was stale and the reported
cushion overstated by ~$746.

Report-only restores the tracking without restoring the brake. These tests pin that the
brake really is gone — a "report-only" monitor that can still stop the bots, or that
writes the HALT file (MIM-NB SystemExits at startup if it exists), would be a kill path
smuggled in under a safe-sounding name.
"""
import importlib
import inspect

import pytest


def _reload(monkeypatch, value):
    """Reimport the monitor with FLOOR_MONITOR_REPORT_ONLY set to `value` (None = unset)."""
    if value is None:
        monkeypatch.delenv("FLOOR_MONITOR_REPORT_ONLY", raising=False)
    else:
        monkeypatch.setenv("FLOOR_MONITOR_REPORT_ONLY", value)
    import src.research.combine_floor_monitor as M
    return importlib.reload(M)


class TestModeSwitch:
    def test_report_only_is_the_default(self, monkeypatch):
        """Halting must be opt-IN. With opt-out, a stray `systemctl start` of the old
        unit would re-arm a kill path the owner explicitly removed."""
        assert _reload(monkeypatch, None).REPORT_ONLY is True

    @pytest.mark.parametrize("val", ["1", "yes", "true", "anything"])
    def test_any_non_zero_value_is_report_only(self, monkeypatch, val):
        assert _reload(monkeypatch, val).REPORT_ONLY is True

    def test_explicit_zero_arms_halting(self, monkeypatch):
        assert _reload(monkeypatch, "0").REPORT_ONLY is False


class TestReportOnlyCannotHalt:
    def test_trigger_branch_neither_halts_nor_writes_flag(self, monkeypatch):
        """Structural: on the REPORT_ONLY path the loop may only log."""
        M = _reload(monkeypatch, "1")
        src = inspect.getsource(M.main)
        i = src.index("if reason and REPORT_ONLY:")
        branch = src[i:src.index("elif reason", i)]
        assert "do_halt" not in branch, "report-only must never halt"
        assert "HALT_FILE" not in branch, "report-only must never write the HALT flag"
        assert "logger.warning" in branch

    def test_halt_still_reachable_when_armed(self, monkeypatch):
        """The brake must still exist for the armed mode — this is a mode switch, not a
        deletion of the kill path."""
        M = _reload(monkeypatch, "0")
        src = inspect.getsource(M.main)
        assert "await do_halt(px, reason)" in src
        assert "systemctl" in inspect.getsource(M.do_halt)


class TestTrackingUnaffected:
    """The whole point is that HWM/floor accounting keeps working in either mode."""

    def test_floor_ratchets_up_with_hwm(self, monkeypatch):
        M = _reload(monkeypatch, "1")
        assert M.update_floor(48_000.0, 50_217.86) == pytest.approx(50_217.86 - M.TRAIL)

    def test_floor_never_ratchets_down(self, monkeypatch):
        M = _reload(monkeypatch, "1")
        assert M.update_floor(48_500.0, 49_000.0) == 48_500.0

    def test_triggers_still_evaluated_in_report_only(self, monkeypatch):
        """The reason string must still be produced — report-only silences the ACTION,
        not the detection."""
        M = _reload(monkeypatch, "1")
        reason = M.evaluate_triggers(equity=48_050.0, floor=48_000.0,
                                     combined_pf=1.5, n_trades=5)
        assert reason and "DISTANCE_TO_FLOOR" in reason

    def test_no_trigger_when_clear(self, monkeypatch):
        M = _reload(monkeypatch, "1")
        assert M.evaluate_triggers(equity=50_217.86, floor=48_217.86,
                                   combined_pf=1.5, n_trades=5) is None
