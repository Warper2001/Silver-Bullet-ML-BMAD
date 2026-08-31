"""Unit tests for ``src/ticksim/parity/gate.py`` (prereg §A8.2 / spine AD-26).

Covers: :func:`evaluate` on all four PASS/FAIL combinations + the AD-26
asymmetry wording + verdict validation; :func:`frozen_sha` against this repo's
real ``HEAD`` and its failure branches; :func:`_find_repo_root`;
:func:`build_amendment_stub` on a passing and a failing pair, its ``GateError``
guards, per-trader grouping, and byte-for-byte determinism.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path

import pytest

from src.ticksim.config import (
    PART_A_MIN_N,
    PART_B_MIN_ORDERS,
    PARITY_MAE_MAX_TICKS,
)
from src.ticksim.orders import Leg
from src.ticksim.parity import gate
from src.ticksim.parity.gate import (
    GateError,
    GateVerdict,
    _find_repo_root,
    build_amendment_stub,
    evaluate,
    frozen_sha,
)
from src.ticksim.parity.part_a import FillError, PartAResult, PartAStats
from src.ticksim.parity.part_b import PART_B_COVERAGE_NOTE, PartBResult, Violation

# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #

SHA40 = "0123456789abcdef0123456789abcdef01234567"


def _fill_error(
    *,
    trade_id: str = "t1",
    order_id: str = "o1",
    leg: Leg = Leg.ENTRY,
    signed: float | None = 0.2,
    fidelity: str = "broker_fill",
) -> FillError:
    return FillError(
        trade_id=trade_id,
        order_id=order_id,
        leg=leg,
        real_dbn=20_000_000_000_000,
        real_ts_ns=1_700_000_000_000_000_000,
        sim_vwap_dbn=None if signed is None else 20_000_000_000_000,
        signed_error_ticks=signed,
        miss_reason="leg_unfilled" if signed is None else None,
        sim_terminal_state="expired" if signed is None else None,
        fidelity=fidelity,  # type: ignore[arg-type]
    )


def _stats(
    *, n: int = 30, mae: float = 0.5, p90: float = 1.0, bias: float = 0.1
) -> PartAStats:
    return PartAStats(n=n, mae_ticks=mae, p90_ticks=p90, signed_bias_ticks=bias)


def _part_a(
    *,
    verdict: str = "PASS",
    n: int = 30,
    mae: float = 0.5,
    p90: float = 1.0,
    bias: float = 0.1,
    bf: PartAStats | None = None,
    reason: str = "PASS: all within tolerance",
    warning: str | None = None,
    errors: tuple[FillError, ...] = (),
) -> PartAResult:
    stats = _stats(n=n, mae=mae, p90=p90, bias=bias)
    return PartAResult(
        stats=stats,
        broker_fill_stats=bf if bf is not None else stats,
        verdict=verdict,  # type: ignore[arg-type]
        reason=reason,
        warning=warning,
        unresolved_misses=sum(1 for e in errors if e.signed_error_ticks is None),
        errors=errors,
    )


def _part_b(
    *,
    verdict: str = "PASS",
    n_orders: int = 1200,
    n_fill_events: int = 800,
    violations: tuple[Violation, ...] = (),
    reason: str = "all invariants held across 1200 orders",
) -> PartBResult:
    return PartBResult(
        n_orders=n_orders,
        n_fill_events=n_fill_events,
        violations=violations,
        verdict=verdict,
        reason=reason,
        coverage_note=PART_B_COVERAGE_NOTE,
    )


def _stub(pa: PartAResult, pb: PartBResult, **kw: object) -> str:
    defaults: dict[str, object] = dict(
        amendment_number=7, cycle_number=1, sha=SHA40, date="2026-09-01"
    )
    defaults.update(kw)
    return build_amendment_stub(pa, pb, **defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #


class TestEvaluate:
    def test_both_pass(self) -> None:
        v = evaluate(_part_a(), _part_b())
        assert isinstance(v, GateVerdict)
        assert v.verdict == "PASS"
        assert v.part_a_pass and v.part_b_pass
        # names the §A8.2 rule; embeds neither asymmetry sentence
        assert "§A8.2" in v.reason
        assert "PASS requires Part A AND Part B" in v.reason
        assert "structurally broken" not in v.reason
        assert "miscalibrated" not in v.reason

    def test_part_a_pass_part_b_fail_is_structurally_broken(self) -> None:
        pb = _part_b(verdict="FAIL", reason="3 invariant violation(s) [2=3]")
        v = evaluate(_part_a(), pb)
        assert v.verdict == "FAIL"
        assert v.part_a_pass and not v.part_b_pass
        assert "structurally broken" in v.reason
        assert "3 invariant violation(s) [2=3]" in v.reason  # verbatim

    def test_part_b_pass_part_a_fail_is_miscalibrated(self) -> None:
        pa = _part_a(verdict="FAIL", reason="FAIL: MAE 1.5000 > 1.0 ticks")
        v = evaluate(pa, _part_b())
        assert v.verdict == "FAIL"
        assert not v.part_a_pass and v.part_b_pass
        assert "miscalibrated" in v.reason
        assert "FAIL: MAE 1.5000 > 1.0 ticks" in v.reason  # verbatim

    def test_both_fail_names_both(self) -> None:
        pa = _part_a(verdict="FAIL", reason="FAIL: p90 too wide")
        pb = _part_b(verdict="FAIL", reason="order count 10 < PART_B_MIN_ORDERS")
        v = evaluate(pa, pb)
        assert v.verdict == "FAIL"
        assert "FAIL: p90 too wide" in v.reason
        assert "order count 10 < PART_B_MIN_ORDERS" in v.reason

    @pytest.mark.parametrize("bad", ["pass", "", "PENDING", "Fail"])
    def test_unrecognised_verdict_raises(self, bad: str) -> None:
        with pytest.raises(GateError, match="unrecognised"):
            evaluate(_part_a(verdict=bad), _part_b())
        with pytest.raises(GateError, match="unrecognised"):
            evaluate(_part_a(), _part_b(verdict=bad))


# --------------------------------------------------------------------------- #
# frozen_sha / _find_repo_root
# --------------------------------------------------------------------------- #


class TestFrozenSha:
    def test_returns_this_repos_head(self) -> None:
        sha = frozen_sha()
        assert re.fullmatch(r"[0-9a-f]{40}", sha), sha
        root = _find_repo_root()
        expected = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
        ).stdout.strip()
        assert sha == expected

    def test_deterministic(self) -> None:
        assert frozen_sha() == frozen_sha()

    def test_find_repo_root_locates_dot_git(self) -> None:
        root = _find_repo_root()
        assert isinstance(root, Path)
        assert (root / ".git").exists()

    def test_find_repo_root_raises_when_no_dot_git(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gate.Path, "exists", lambda self: False)
        with pytest.raises(GateError, match="no .git entry"):
            _find_repo_root()

    def test_no_git_on_path_raises_gate_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PATH", "")
        with pytest.raises(GateError):
            frozen_sha()

    def test_called_process_error_raises_gate_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*a: object, **k: object) -> object:
            raise subprocess.CalledProcessError(128, ["git"], stderr="fatal")

        monkeypatch.setattr(gate.subprocess, "run", boom)
        with pytest.raises(GateError):
            frozen_sha()

    def test_non_hex_stdout_raises_gate_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Fake:
            stdout = "not-a-sha\n"

        monkeypatch.setattr(gate.subprocess, "run", lambda *a, **k: _Fake())
        with pytest.raises(GateError, match="not a 40-char hex"):
            frozen_sha()


# --------------------------------------------------------------------------- #
# build_amendment_stub -- passing pair
# --------------------------------------------------------------------------- #

SECTION_HEADERS = [
    "# Amendment 7 -- Parity gate result (cycle 1)",
    "## Verdict",
    "## Frozen SHA",
    "## Part A -- real-fill calibration",
    "## Part A -- per-trader breakdown",
    "## Part B -- synthetic invariant battery",
    "## Integrity",
    "## Cycle / kill criterion",
]


class TestBuildAmendmentStubPassing:
    def test_all_sections_present_in_order(self) -> None:
        text = _stub(_part_a(), _part_b())
        idx = [text.index(h) for h in SECTION_HEADERS]
        assert idx == sorted(idx), text

    def test_header_and_date(self) -> None:
        text = _stub(_part_a(), _part_b())
        assert "# Amendment 7 -- Parity gate result (cycle 1)" in text
        assert "_date: 2026-09-01_" in text

    def test_date_default_placeholder(self) -> None:
        assert "_date: TBD (fill on append)_" in _stub(_part_a(), _part_b(), date=None)

    def test_empty_date_and_integrity_coalesce_to_placeholder(self) -> None:
        text = _stub(_part_a(), _part_b(), date="", integrity="")
        assert "_date: TBD (fill on append)_" in text
        assert "integrity: pending (gate.py slice 2" in text
        # no blank line where the value would go
        assert "\n\n\n" not in text

    def test_sha_line(self) -> None:
        assert f"simulator commit: {SHA40}" in _stub(_part_a(), _part_b())

    def test_verdict_section_text(self) -> None:
        text = _stub(_part_a(), _part_b())
        assert "## Verdict\n\n**PASS**" in text
        assert "- part_a_pass: True" in text
        assert "- part_b_pass: True" in text
        assert "- reason: Part A PASS and Part B PASS" in text

    def test_unresolved_misses_line(self) -> None:
        assert "- unresolved misses: 0" in _stub(_part_a(), _part_b())
        errs = (_fill_error(signed=None),)
        assert "- unresolved misses: 1" in _stub(_part_a(errors=errs), _part_b())

    def test_part_b_none_when_clean(self) -> None:
        assert "-- none --" in _stub(_part_a(), _part_b())

    def test_coverage_note_verbatim(self) -> None:
        assert PART_B_COVERAGE_NOTE in _stub(_part_a(), _part_b())

    def test_pending_integrity_line_when_omitted(self) -> None:
        assert "integrity: pending (gate.py slice 2" in _stub(
            _part_a(), _part_b(), integrity=None
        )

    def test_supplied_integrity_verbatim(self) -> None:
        assert "ts-monotonic OK; 0 persistent cross" in _stub(
            _part_a(), _part_b(), integrity="ts-monotonic OK; 0 persistent cross"
        )

    def test_min_n_met(self) -> None:
        assert f"PART_A_MIN_N: {PART_A_MIN_N} -- met" in _stub(_part_a(), _part_b())

    def test_cycle_and_kill_criterion(self) -> None:
        text = _stub(_part_a(), _part_b())
        assert "cycle 1 of 3" in text
        assert "out of code" in text

    def test_no_sha_note_when_sha_supplied(self) -> None:
        assert "taken as-is" not in _stub(_part_a(), _part_b())

    def test_determinism_byte_identical(self) -> None:
        assert _stub(_part_a(), _part_b()) == _stub(_part_a(), _part_b())

    def test_no_em_dash_in_generated_text(self) -> None:
        # patch 14: no em-dash anywhere in the rendered document (the verbatim
        # coverage note / prereg refs may still carry a section sign).
        assert "—" not in _stub(_part_a(), _part_b())

    def test_no_git_note_present_when_sha_derived(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gate, "frozen_sha", lambda: SHA40)
        text = build_amendment_stub(
            _part_a(),
            _part_b(),
            amendment_number=7,
            cycle_number=1,
            sha=None,
            date="2026-09-01",
        )
        assert "taken as-is" in text
        assert f"simulator commit: {SHA40}" in text

    def test_distinct_full_and_broker_subset_tables(self) -> None:
        # verdict FAIL so the PASS-only display guard does not fire on the
        # deliberately out-of-tolerance broker subset.
        pa = _part_a(
            verdict="FAIL",
            mae=0.5,
            bf=_stats(n=12, mae=1.4, p90=1.9, bias=0.2),
            reason="FAIL: broker_fill subset MAE 1.4 > 1.0",
        )
        text = _stub(pa, _part_b())
        full = text.split("`broker_fill`-only subset:")[0]
        subset = text.split("`broker_fill`-only subset:")[1]
        assert "| MAE | 0.5000 |" in full
        assert "| MAE | 1.4000 |" in subset

    def test_part_a_min_n_not_met(self) -> None:
        text = _stub(_part_a(n=PART_A_MIN_N - 1), _part_b())
        assert f"PART_A_MIN_N: {PART_A_MIN_N} -- NOT met" in text

    def test_part_b_min_orders_not_met(self) -> None:
        pb = _part_b(
            verdict="FAIL",
            n_orders=10,
            reason="order count 10 < PART_B_MIN_ORDERS (1000)",
        )
        text = _stub(_part_a(), pb)
        assert f"PART_B_MIN_ORDERS: {PART_B_MIN_ORDERS} -- NOT met" in text

    def test_empty_broker_fill_subset_renders_na_not_pass(self) -> None:
        pa = _part_a(bf=_stats(n=0, mae=0.0, p90=0.0, bias=0.0))
        text = _stub(pa, _part_b())
        subset = text.split("`broker_fill`-only subset:")[1].split(
            "## Part A -- per-trader"
        )[0]
        assert "(empty subset -- no broker_fill rows)" in subset
        assert "| MAE | n/a |" in subset
        assert "PASS" not in subset and "FAIL" not in subset


# --------------------------------------------------------------------------- #
# build_amendment_stub -- failing pair
# --------------------------------------------------------------------------- #


class TestBuildAmendmentStubFailing:
    def test_part_a_row_fail_and_part_b_table(self) -> None:
        pa = _part_a(
            verdict="FAIL",
            mae=1.5,
            p90=3.0,
            bias=0.4,
            reason="FAIL: MAE 1.5000 > 1.0 ticks",
        )
        pb = _part_b(
            verdict="FAIL",
            reason="3 invariant violation(s) [2=2, 4=1]",
            violations=(
                Violation("o2", "2", "invariant 2 ..."),
                Violation("o5", "2", "invariant 2 ..."),
                Violation("o9", "4", "invariant 4 ..."),
            ),
        )
        text = build_amendment_stub(
            pa,
            pb,
            amendment_number=8,
            cycle_number=2,
            sha=SHA40,
            date="2026-09-02",
        )
        assert "**FAIL**" in text
        assert f"| MAE | 1.5000 | {PARITY_MAE_MAX_TICKS} | FAIL |" in text
        assert "| 2 | 2 |" in text
        assert "| 4 | 1 |" in text
        assert text.index("| 2 | 2 |") < text.index("| 4 | 1 |")

    def test_warning_rendered(self) -> None:
        pa = _part_a(warning="broker_fill subset is empty")
        text = _stub(pa, _part_b())
        assert "- warning: broker_fill subset is empty" in text


# --------------------------------------------------------------------------- #
# GateError guards
# --------------------------------------------------------------------------- #


class TestGateErrors:
    def test_sha_less_and_no_git(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", "")
        with pytest.raises(GateError):
            build_amendment_stub(
                _part_a(),
                _part_b(),
                amendment_number=1,
                cycle_number=1,
                sha=None,
            )

    def test_display_verdict_contradiction_full_sample(self) -> None:
        pa = _part_a(verdict="PASS", mae=PARITY_MAE_MAX_TICKS + 1.0)
        with pytest.raises(GateError, match="inconsistent with src.ticksim.config"):
            _stub(pa, _part_b())

    def test_display_verdict_contradiction_broker_subset(self) -> None:
        pa = _part_a(
            verdict="PASS",
            bf=_stats(n=3, mae=PARITY_MAE_MAX_TICKS + 1.0, p90=1.0, bias=0.0),
        )
        with pytest.raises(GateError, match="inconsistent with src.ticksim.config"):
            _stub(pa, _part_b())

    @pytest.mark.parametrize(
        "bad_sha", ["", "abc", SHA40 + "00", SHA40.upper(), "g" * 40]
    )
    def test_bad_caller_sha_rejected(self, bad_sha: str) -> None:
        with pytest.raises(GateError):
            _stub(_part_a(), _part_b(), sha=bad_sha)

    @pytest.mark.parametrize("bad_num", [0, -1, -99])
    def test_non_positive_amendment_number_rejected(self, bad_num: int) -> None:
        with pytest.raises(GateError, match="amendment_number"):
            _stub(_part_a(), _part_b(), amendment_number=bad_num)

    @pytest.mark.parametrize("field", ["date", "integrity"])
    @pytest.mark.parametrize("payload", ["a\nb", "x ## H", "line\n## H"])
    def test_template_break_in_caller_field_rejected(
        self, field: str, payload: str
    ) -> None:
        with pytest.raises(GateError, match="fixed template"):
            _stub(_part_a(), _part_b(), **{field: payload})

    def test_newline_in_sha_rejected_before_hex_check(self) -> None:
        with pytest.raises(GateError, match="fixed template"):
            _stub(_part_a(), _part_b(), sha=SHA40[:20] + "\n" + SHA40[21:])


# --------------------------------------------------------------------------- #
# per-trader grouping
# --------------------------------------------------------------------------- #


class TestPerTraderGrouping:
    def test_groups_by_fidelity_when_map_omitted(self) -> None:
        errors = (
            _fill_error(
                trade_id="ta", order_id="a", signed=0.2, fidelity="broker_fill"
            ),
            _fill_error(
                trade_id="tb", order_id="b", signed=None, fidelity="bar_reconstructed"
            ),
        )
        text = _stub(_part_a(errors=errors), _part_b())
        assert "Grouped by `err.fidelity`" in text
        assert "broker_fill" in text and "bar_reconstructed" in text
        assert "mim-nb" in text and "yank" in text  # caveat present
        # misses column: the bar_reconstructed group has 1 miss, n/a means
        assert "| bar_reconstructed | 1 | n/a | n/a | 1 |" in text
        assert "| broker_fill | 1 | 0.2000 | 0.2000 | 0 |" in text

    def test_groups_by_trader_when_map_given(self) -> None:
        errors = (
            _fill_error(trade_id="ta", order_id="a", signed=0.2),
            _fill_error(trade_id="tb", order_id="b", signed=0.6),
        )
        text = _stub(
            _part_a(errors=errors),
            _part_b(),
            trader_by_trade_id={"ta": "trader-mim-nb", "tb": "trader-yank"},
        )
        assert "trader_by_trade_id" in text
        assert "| trader-mim-nb | 1 | 0.2000 | 0.2000 | 0 |" in text
        assert "| trader-yank | 1 | 0.6000 | 0.6000 | 0 |" in text

    def test_unmapped_trade_id_renders_placeholder(self) -> None:
        errors = (_fill_error(trade_id="ta", order_id="a", signed=0.2),)
        text = _stub(
            _part_a(errors=errors),
            _part_b(),
            trader_by_trade_id={"other": "trader-yank"},
        )
        assert "| <unmapped:ta> | 1 |" in text


# --------------------------------------------------------------------------- #
# source-level guard: exactly one subprocess.run call (spine AD-26)
# --------------------------------------------------------------------------- #


def test_one_subprocess_run_call_and_no_wallclock_or_network_refs() -> None:
    tree = ast.parse(Path(gate.__file__).read_text())
    run_calls = 0
    attr_chains: list[str] = []
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            fn = node.func
            if (
                isinstance(fn.value, ast.Name)
                and fn.value.id == "subprocess"
                and fn.attr == "run"
            ):
                run_calls += 1
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            attr_chains.append(f"{node.value.id}.{node.attr}")

    assert run_calls == 1, "AD-26: exactly one subprocess.run call"
    for banned_import in ("datetime", "time", "socket", "urllib", "requests", "os"):
        assert banned_import not in imported, f"AD-1/AD-4/AD-11: no {banned_import}"
    for banned_attr in ("time.time", "os.system", "datetime.now"):
        assert banned_attr not in attr_chains
