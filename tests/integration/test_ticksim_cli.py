"""Integration test: ``cli.main(["simulate", ...])`` over the real 2026-06-22
MBO window.

``@pytest.mark.integration`` -- folds the uncommitted GLBX MDP3 capture
``data/tick/_test/glbx-mdp3-20260622.mbo.dbn.zst``. Skips when the capture is
absent unless ``TICKSIM_REQUIRE_FIXTURE`` is set (same convention as
``tests/integration/test_ticksim_parity_run_part_a_integration.py``).

A smoke assertion, not a parity verdict: two marketable orders are pinned into
the window discovered from the capture's own first slice, run through the real
``simulate`` over the real book, and the outputs are checked to exist, parse as
``OrderOutcome``\\ s, and cover both submitted orders. A marketable order over a
real book should fill, so a ``filled`` terminal state is checked *if any*
outcome reports one (it is not asserted unconditionally -- the discovered anchor
tick could land in a thin patch).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.ticksim import cli
from src.ticksim.events import DbnMboSource
from src.ticksim.orders import (
    IntentAction,
    Leg,
    OrderIntent,
    OrderKind,
    OrderOutcome,
    Side,
)

pytestmark = pytest.mark.integration

WINDOW = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "tick"
    / "_test"
    / "glbx-mdp3-20260622.mbo.dbn.zst"
)

_NS_PER_MINUTE = 60 * 1_000_000_000
_SCAN = 400_000


def _window_or_skip() -> Path:
    if WINDOW.is_file():
        return WINDOW
    if os.environ.get("TICKSIM_REQUIRE_FIXTURE"):
        pytest.fail(f"TICKSIM_REQUIRE_FIXTURE set but window missing: {WINDOW}")
    pytest.skip(f"2026-06-22 MBO window not present: {WINDOW}")
    raise AssertionError("unreachable")  # pragma: no cover


def _front_month_and_start(path: Path) -> tuple[int, int]:
    counts: dict[int, int] = {}
    first_ts: int | None = None
    for i, ev in enumerate(DbnMboSource(path)):
        counts[ev.instrument_id] = counts.get(ev.instrument_id, 0) + 1
        if first_ts is None:
            first_ts = ev.ts_event
        if i >= _SCAN:
            break
    assert first_ts is not None, "capture yielded no records"
    return max(counts, key=lambda k: counts[k]), first_ts


def test_cli_simulate_over_the_2026_06_22_window(tmp_path: Path) -> None:
    path = _window_or_skip()
    iid, start_ts = _front_month_and_start(path)
    entry_ts = start_ts + 5 * _NS_PER_MINUTE
    exit_ts = entry_ts + 5 * _NS_PER_MINUTE

    intents = [
        OrderIntent(
            action=IntentAction.SUBMIT,
            order_id="rt-e",
            trade_id="rt",
            leg=Leg.ENTRY,
            kind=OrderKind.MARKETABLE,
            side=Side.BUY,
            size=1,
            submit_ts_ns=entry_ts,
        ),
        OrderIntent(
            action=IntentAction.SUBMIT,
            order_id="rt-x",
            trade_id="rt",
            leg=Leg.EXIT,
            kind=OrderKind.MARKETABLE,
            side=Side.SELL,
            size=1,
            submit_ts_ns=exit_ts,
        ),
    ]
    intents_path = tmp_path / "intents.jsonl"
    intents_path.write_text("".join(i.model_dump_json() + "\n" for i in intents))

    out_outcomes = tmp_path / "outcomes.jsonl"
    out_manifest = tmp_path / "manifest.json"
    rc = cli.main(
        [
            "simulate",
            "--dbn",
            str(path),
            "--intents",
            str(intents_path),
            "--instrument-id",
            str(iid),
            "--out-outcomes",
            str(out_outcomes),
            "--out-manifest",
            str(out_manifest),
        ]
    )
    assert rc == 0

    lines = out_outcomes.read_text().splitlines()
    assert lines, "outcome log is empty"
    outcomes = [OrderOutcome.model_validate_json(line) for line in lines]
    assert {o.order_id for o in outcomes} == {"rt-e", "rt-x"}
    for outcome in outcomes:
        if outcome.terminal_state.value == "filled":
            assert outcome.fills, "a filled outcome must carry fills"

    manifest = json.loads(out_manifest.read_text())
    assert "seed" in manifest
    assert manifest["event_count"] > 0
