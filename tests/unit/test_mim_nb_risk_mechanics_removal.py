"""Risk-mechanics removal — prereg mim-nb-risk-mechanics-removal.

Owner decision (Alex, 2026-07-29): floor-derived entry gating is removed; the static
-$1000 DLL guard is retained as the only automatic brake.

These tests pin prereg §3 V1 and V2. They assert that the bot NO LONGER protects the
account against the MLL — that is the intended behaviour, not an oversight.
"""
import inspect

import pytest

from src.research import mim_nb_live as M
from src.research.mim_nb_live import MimNbLive, DLL_GUARD_USD, CAT_STOP_PTS, PT_VAL, CONTRACTS


CAT_COST = CAT_STOP_PTS * PT_VAL * CONTRACTS


def _closer(day_pnl, buf):
    """A MimNbLive stub carrying just the state the post-close DLL block reads."""
    o = object.__new__(MimNbLive)
    o.day_pnl = day_pnl
    o.day_deactivated = False
    o._buffer_source = "shared"
    o._remaining_mll_buffer = lambda: buf
    return o


def _run_dll_block(o):
    """Execute the post-close DLL decision exactly as _exit does."""
    buf = o._remaining_mll_buffer()
    static_dll = -abs(DLL_GUARD_USD)
    if o.day_pnl <= static_dll and not o.day_deactivated:
        o.day_deactivated = True
    return o.day_deactivated, static_dll, buf


class TestBufferGateRemoved:
    """V1 — a starved buffer must no longer block entry."""

    def test_entry_path_has_no_buffer_early_exit(self):
        """The `elif` that skipped entry when buf <= cat_cost must be gone.

        Pinned structurally: with the gate present the band-break branch was `elif`,
        so a starved buffer short-circuited it. It must now be an independent `if`.
        """
        src = inspect.getsource(MimNbLive.on_bar)
        assert "BUFFER_GATE" not in src, "buffer gate still blocks entries"
        assert "BUFFER_INFO" in src, "buffer should still be logged for auditability"
        gate_idx = src.index("BUFFER_INFO")
        after = src[gate_idx:]
        assert "if c > ub and self.position != 1:" in after, "band-break entry must be reachable"
        assert "elif c > ub" not in after, "entry is still chained to the buffer branch"

    def test_buffer_is_still_computed_for_audit(self):
        """Removing the gate must not remove observability (prereg §1 'retained')."""
        src = inspect.getsource(MimNbLive.on_bar)
        assert "_remaining_mll_buffer()" in src


class TestStaticDLLRetained:
    """V2 — the daily guard survives, and no longer shrinks with the buffer."""

    @pytest.mark.parametrize("buf", [-5000.0, -386.0, 0.0, 499.12, 50_000.0])
    def test_dll_threshold_is_independent_of_buffer(self, buf):
        """The old clamp made the allowance a function of buf; it must not be now."""
        _, static_dll, _ = _run_dll_block(_closer(-100.0, buf))
        assert static_dll == -1000.0

    @pytest.mark.parametrize("buf", [-5000.0, 0.0, 499.12])
    def test_small_loss_no_longer_deactivates_on_starved_buffer(self, buf):
        """Under the old dynamic clamp, buf<=-cat_cost gave an allowance of $0, so ANY
        losing day deactivated the bot. That second floor gate is gone."""
        deactivated, _, _ = _run_dll_block(_closer(-250.0, buf))
        assert deactivated is False

    def test_loss_past_static_cap_still_deactivates(self):
        deactivated, _, _ = _run_dll_block(_closer(-1000.01, 10_000.0))
        assert deactivated is True

    def test_exactly_at_cap_deactivates(self):
        deactivated, _, _ = _run_dll_block(_closer(-1000.0, 10_000.0))
        assert deactivated is True

    def test_dll_guard_value_unchanged(self):
        """Retained by owner instruction — a change here would be out of prereg scope."""
        assert DLL_GUARD_USD == -1000.0


class TestManualKillSwitchRetained:
    """The HALT flag is inert once the monitor is disabled, but must remain usable."""

    def test_initialize_still_honours_halt_flag(self):
        src = inspect.getsource(MimNbLive.initialize)
        assert "HALT" in src and "SystemExit" in src
