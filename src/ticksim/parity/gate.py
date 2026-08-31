"""Parity-gate output contract -- the two-part verdict + amendment stub (AD-26).

Pre-registration §A8.2 defines the gate verdict as **Part A PASS AND Part B
PASS**; §4 requires the passing simulator commit SHA frozen into an append-only
amendment. Spine AD-26 pins that output contract here.

Three public entry points:

* :func:`evaluate` -- fold a :class:`~src.ticksim.parity.part_a.PartAResult` and a
  :class:`~src.ticksim.parity.part_b.PartBResult` into a :class:`GateVerdict`.
  ``verdict == "PASS"`` iff **both** parts pass. The ``reason`` spells out the
  AD-26 asymmetry: a Part A pass with a Part B fail means *the fill model is
  structurally broken*; a Part B pass with a Part A fail means *the model runs
  but is miscalibrated*.
* :func:`frozen_sha` -- ``git rev-parse HEAD``. This is the **one** sanctioned
  ``subprocess`` call anywhere in ``src/ticksim`` (spine AD-26; AD-4 / AD-11
  otherwise stand -- no other subprocess, no shell string, no wall-clock).
* :func:`build_amendment_stub` -- the fixed-template Markdown block an analyst
  appends to the seal. The function only **returns** text; it never writes a
  file. The §5 integrity preflight is ``gate.py`` slice 2 -- until then the stub
  carries a ``pending`` placeholder line (or an ``integrity`` string the caller
  supplies).

Imports (spine AD-7, ``PERMITTED_INTERNAL_EDGES["gate"]``): ``config``,
``part_a``, ``part_b`` -- relative form only. Standard library: ``subprocess``,
``collections``, ``dataclasses``, ``typing``, ``pathlib`` (``collections`` / ``re``
newly on the AD-4 allowlist, review round 1). No wall-clock module, no network
(AD-1 / AD-4 / AD-11). Generated text is ASCII-only for byte-for-byte
determinism.
"""

from __future__ import annotations

import subprocess
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..config import (
    PART_A_MIN_N,
    PART_B_MIN_ORDERS,
    PARITY_MAE_MAX_TICKS,
    PARITY_P90_MAX_TICKS,
    PARITY_SIGNED_BIAS_MAX_TICKS,
)
from .part_a import FillError, PartAResult, PartAStats
from .part_b import PartBResult

__all__ = [
    "GateVerdict",
    "GateError",
    "evaluate",
    "frozen_sha",
    "build_amendment_stub",
]

_HEX = frozenset("0123456789abcdef")
_VALID_VERDICTS = ("PASS", "FAIL")
_DATE_PLACEHOLDER = "TBD (fill on append)"
_INTEGRITY_PLACEHOLDER = (
    "integrity: pending (gate.py slice 2 -- §5 preflight not yet wired)"
)
_KILL_CRITERION_SENTENCE = (
    "The 15-working-day / 3-revision-cycle kill clock (§4) is tracked "
    "**out of code** by the analyst -- neither the cycle number nor the clock "
    "is derived here (spine AD-26)."
)
_GIT_SHA_TIMEOUT_S = 15


class GateError(Exception):
    """The parity gate cannot produce a trustworthy amendment.

    Raised for: ``git`` missing / ``cwd`` outside a repo / ``git rev-parse``
    failing / a non-SHA result (the amendment MUST carry a real 40-hex SHA,
    never a guess -- spine AD-26 / prereg §4); a caller-supplied ``sha`` /
    ``date`` / ``integrity`` that is malformed or contains a newline / heading
    marker (breaks the fixed template); a non-positive ``amendment_number``; an
    unrecognised ``part_a`` / ``part_b`` verdict; and a ``PartAResult`` whose
    display rows are inconsistent with the ``config`` tolerances imported here.
    """


@dataclass(frozen=True)
class GateVerdict:
    """The combined two-part parity verdict (prereg §A8.2, spine AD-26).

    ``verdict == "PASS"`` iff ``part_a_pass and part_b_pass``. ``reason`` names
    the failing side(s), quotes the failing result's own ``reason`` verbatim,
    and states the AD-26 asymmetry.
    """

    verdict: Literal["PASS", "FAIL"]
    part_a_pass: bool
    part_b_pass: bool
    reason: str


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _is_sha40(value: str) -> bool:
    return len(value) == 40 and all(ch in _HEX for ch in value)


def _reject_template_break(name: str, value: str) -> None:
    """A caller field that would break the fixed template -> :class:`GateError`.

    A newline or a ``"## "`` heading marker in ``sha`` / ``date`` / ``integrity``
    breaks determinism and the append-only contract.
    """
    if "\n" in value or "## " in value:
        raise GateError(
            f"caller field {name!r} contains a newline / heading marker -- "
            "breaks the fixed template"
        )


def _fmt(value: float | None) -> str:
    """A tick figure as a fixed 4-dp cell; ``None`` -> ``n/a``; ``-0.0`` -> ``0.0000``."""
    if value is None:
        return "n/a"
    formatted = f"{value:.4f}"
    return "0.0000" if formatted == "-0.0000" else formatted


# --------------------------------------------------------------------------- #
# evaluate -- the one place PASS/FAIL is derived
# --------------------------------------------------------------------------- #


def evaluate(part_a: PartAResult, part_b: PartBResult) -> GateVerdict:
    """Combine the two part results into the gate verdict (prereg §A8.2).

    ``part_a_pass = (part_a.verdict == "PASS")``,
    ``part_b_pass = (part_b.verdict == "PASS")``, ``verdict == "PASS"`` iff
    **both**. No other module derives PASS/FAIL -- ``build_amendment_stub`` calls
    this, it does not re-derive it.

    Raises:
        GateError: ``part_a.verdict`` or ``part_b.verdict`` is not exactly
            ``"PASS"`` or ``"FAIL"`` -- a corrupt verdict must not silently
            collapse to a confident FAIL.
    """
    for name, value in (("part_a", part_a.verdict), ("part_b", part_b.verdict)):
        if value not in _VALID_VERDICTS:
            raise GateError(
                f"unrecognised {name} verdict {value!r} -- expected "
                f"{_VALID_VERDICTS[0]!r} or {_VALID_VERDICTS[1]!r}"
            )

    part_a_pass = part_a.verdict == "PASS"
    part_b_pass = part_b.verdict == "PASS"

    if part_a_pass and part_b_pass:
        return GateVerdict(
            verdict="PASS",
            part_a_pass=True,
            part_b_pass=True,
            reason=(
                "Part A PASS and Part B PASS -- parity gate PASS "
                "(prereg §A8.2: PASS requires Part A AND Part B). "
                f"Part A: {part_a.reason} | Part B: {part_b.reason}"
            ),
        )

    if part_a_pass and not part_b_pass:
        reason = (
            "Part A PASS but Part B FAIL -- the fill model is structurally "
            f"broken (prereg §A8.2). Part B: {part_b.reason}"
        )
    elif part_b_pass and not part_a_pass:
        reason = (
            "Part B PASS but Part A FAIL -- the model runs but is miscalibrated "
            f"(prereg §A8.2). Part A: {part_a.reason}"
        )
    else:
        reason = (
            "Part A FAIL and Part B FAIL. "
            f"Part A: {part_a.reason} | Part B: {part_b.reason}"
        )

    return GateVerdict(
        verdict="FAIL",
        part_a_pass=part_a_pass,
        part_b_pass=part_b_pass,
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# frozen_sha -- the one sanctioned subprocess call (spine AD-26)
# --------------------------------------------------------------------------- #


def _find_repo_root() -> Path:
    """Walk parent dirs from this file to the first containing a ``.git`` entry.

    ``.git`` is a directory in a normal checkout and a file in a git worktree --
    :meth:`Path.exists` covers both. :class:`GateError` if no ancestor has one
    (no extra subprocess -- resolved at spec CHECKPOINT 1, 2026-08-31).
    """
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    raise GateError(
        f"no .git entry found walking up from {here} -- frozen_sha() needs a "
        "git checkout (spine AD-26 / prereg §4)"
    )


def frozen_sha() -> str:
    """Return this repo's ``HEAD`` commit SHA -- the frozen simulator commit.

    The **only** ``subprocess`` call anywhere in ``src/ticksim`` (spine AD-26).
    ``git rev-parse HEAD`` run with ``cwd`` at the :func:`_find_repo_root`
    walk-up result. A dirty working tree is **not** an error here -- the analyst
    runs the gate from a committed state by discipline, and
    :func:`build_amendment_stub` notes ``git rev-parse`` was taken as-is.

    Raises:
        GateError: ``git`` is not on ``PATH``, ``cwd`` is outside a repo,
            ``git rev-parse`` exits non-zero / times out, or the result is not a
            40-char hex SHA -- the amendment must carry a real SHA, never a
            guess (spine AD-26 / prereg §4).
    """
    root = _find_repo_root()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=root,
            timeout=_GIT_SHA_TIMEOUT_S,
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        OSError,
        subprocess.TimeoutExpired,
    ) as exc:
        raise GateError(
            f"`git rev-parse HEAD` could not be resolved in {root}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    sha = completed.stdout.strip()
    if not _is_sha40(sha):
        raise GateError(f"`git rev-parse HEAD` returned {sha!r}, not a 40-char hex SHA")
    return sha


# --------------------------------------------------------------------------- #
# build_amendment_stub -- the fixed-template Markdown block (spine AD-26 / §4)
# --------------------------------------------------------------------------- #


def _part_a_rows(stats: PartAStats) -> list[tuple[str, str, str, bool]]:
    """The 3 display rows: (label, value-cell, tolerance-cell, within-tolerance).

    The signed-bias row shows ``|bias|`` in the value cell so the reader sees
    the PASS/FAIL is on magnitude vs the ``±`` tolerance.
    """
    bias = stats.signed_bias_ticks
    return [
        (
            "MAE",
            _fmt(stats.mae_ticks),
            f"{PARITY_MAE_MAX_TICKS}",
            stats.mae_ticks <= PARITY_MAE_MAX_TICKS,
        ),
        (
            "p90",
            _fmt(stats.p90_ticks),
            f"{PARITY_P90_MAX_TICKS}",
            stats.p90_ticks <= PARITY_P90_MAX_TICKS,
        ),
        (
            "signed bias (abs)",
            _fmt(abs(bias)),
            f"+/-{PARITY_SIGNED_BIAS_MAX_TICKS}",
            abs(bias) <= PARITY_SIGNED_BIAS_MAX_TICKS,
        ),
    ]


def _stat_table(stats: PartAStats, *, empty_message: str | None = None) -> list[str]:
    """A 4-column metric table. ``empty_message`` + ``stats.n == 0`` -> an
    all-``n/a`` table and the note, no PASS/FAIL (spec: an empty ``broker_fill``
    subset must not read as three green PASSes)."""
    lines = [
        "| metric | value (ticks) | tolerance (ticks) | result |",
        "|---|---|---|---|",
    ]
    if empty_message is not None and stats.n == 0:
        lines.append(f"| MAE | n/a | {PARITY_MAE_MAX_TICKS} | n/a |")
        lines.append(f"| p90 | n/a | {PARITY_P90_MAX_TICKS} | n/a |")
        lines.append(
            f"| signed bias (abs) | n/a | +/-{PARITY_SIGNED_BIAS_MAX_TICKS} | n/a |"
        )
        lines.append("")
        lines.append(f"_{empty_message}_")
        return lines
    for label, value_cell, tol, ok in _part_a_rows(stats):
        lines.append(f"| {label} | {value_cell} | {tol} | {'PASS' if ok else 'FAIL'} |")
    return lines


def _group_key(err: FillError, trader_by_trade_id: Mapping[str, str] | None) -> str:
    if trader_by_trade_id is not None:
        return trader_by_trade_id.get(err.trade_id, f"<unmapped:{err.trade_id}>")
    return err.fidelity


def _per_trader_section(
    part_a: PartAResult, trader_by_trade_id: Mapping[str, str] | None
) -> list[str]:
    lines = ["## Part A -- per-trader breakdown", ""]
    if trader_by_trade_id is not None:
        lines.append("Grouped by `trader_by_trade_id[trade_id]`.")
    else:
        lines.append(
            "Grouped by `err.fidelity` (no `trader_by_trade_id` map supplied). "
            "**Caveat:** for this sample `broker_fill` ~ trader-mim-nb and "
            "`bar_reconstructed` ~ trader-yank -- an approximation, not a join."
        )
    lines.append("")

    groups: dict[str, list[FillError]] = {}
    for err in part_a.errors:
        groups.setdefault(_group_key(err, trader_by_trade_id), []).append(err)

    if not groups:
        lines.append("_(no per-fill errors recorded)_")
        return lines

    lines.append(
        "| group | count | mean abs err ticks | mean signed err ticks | misses |"
    )
    lines.append("|---|---|---|---|---|")
    for key in sorted(groups):
        errs = groups[key]
        resolved = [
            e.signed_error_ticks for e in errs if e.signed_error_ticks is not None
        ]
        n_miss = sum(1 for e in errs if e.signed_error_ticks is None)
        mean_abs = sum(abs(x) for x in resolved) / len(resolved) if resolved else None
        mean_signed = sum(resolved) / len(resolved) if resolved else None
        lines.append(
            f"| {key} | {len(errs)} | {_fmt(mean_abs)} | {_fmt(mean_signed)} "
            f"| {n_miss} |"
        )
    return lines


def _part_b_section(part_b: PartBResult) -> list[str]:
    met = "met" if part_b.n_orders >= PART_B_MIN_ORDERS else "NOT met"
    lines = [
        "## Part B -- synthetic invariant battery",
        "",
        f"- orders: {part_b.n_orders}",
        f"- fill events: {part_b.n_fill_events}",
        f"- PART_B_MIN_ORDERS: {PART_B_MIN_ORDERS} -- {met}",
        "",
        "Per-label violation counts:",
        "",
    ]
    counts = Counter(v.invariant for v in part_b.violations)
    if not counts:
        lines.append("-- none --")
    else:
        lines.append("| label | count |")
        lines.append("|---|---|")
        for label in sorted(counts):
            lines.append(f"| {label} | {counts[label]} |")
    lines.extend(["", "Coverage note (verbatim):", "", part_b.coverage_note])
    return lines


def build_amendment_stub(
    part_a: PartAResult,
    part_b: PartBResult,
    *,
    amendment_number: int,
    cycle_number: int,
    sha: str | None = None,
    integrity: str | None = None,
    date: str | None = None,
    trader_by_trade_id: Mapping[str, str] | None = None,
) -> str:
    """Return the fixed-template parity-gate amendment as Markdown text.

    The function **only returns text** -- it never writes a file (append-only in
    spirit: the analyst appends to the seal). Section order is fixed (spine
    AD-26): header, verdict, frozen SHA, Part A, Part A per-trader, Part B,
    integrity, cycle / kill criterion. Generated text is ASCII-only.

    Args:
        part_a: the Part A calibration result.
        part_b: the Part B invariant-battery result.
        amendment_number: the amendment's number in the seal (analyst-supplied;
            resolved at spec CHECKPOINT 1). Must be ``> 0``. Rendered verbatim.
        cycle_number: the revision-cycle number (analyst-supplied). Rendered as
            ``cycle {cycle_number} of 3`` -- out-of-range values render as-is;
            the analyst owns the clock (spine AD-26).
        sha: the frozen simulator SHA (40-char lowercase hex). ``None`` ->
            :func:`frozen_sha` is invoked once; on failure the whole call raises
            :class:`GateError` (never a SHA-less stub). A non-``None`` value is
            validated the same way.
        integrity: the §5 integrity-preflight summary. Falsy (``None`` / ``""``)
            -> a ``pending`` placeholder line (slice 2 fills it in).
        date: the append date. Falsy (``None`` / ``""``) -> ``"TBD (fill on
            append)"`` -- no wall-clock in ``src/ticksim`` (spine AD-1).
        trader_by_trade_id: optional ``trade_id -> trader`` map for the
            per-trader breakdown. Omitted -> group by ``fidelity`` with the
            mim-nb / yank caveat.

    Raises:
        GateError: ``amendment_number <= 0``; ``sha`` / ``date`` / ``integrity``
            contains a newline or ``"## "``; ``sha`` is a non-40-hex string;
            ``sha is None`` and :func:`frozen_sha` failed; ``evaluate`` rejected
            a verdict; or ``part_a.verdict == "PASS"`` while a display row (full
            sample or ``broker_fill`` subset) is out of ``config`` tolerance.
    """
    if amendment_number <= 0:
        raise GateError(f"amendment_number must be > 0, got {amendment_number}")
    for field_name, field_value in (
        ("sha", sha),
        ("date", date),
        ("integrity", integrity),
    ):
        if field_value is not None:
            _reject_template_break(field_name, field_value)
    if sha is not None and not _is_sha40(sha):
        raise GateError(
            f"caller-supplied sha {sha!r} is not a 40-char lowercase hex SHA -- "
            "a truncated / typo SHA must never be sealed"
        )

    verdict = evaluate(part_a, part_b)

    # PASS-only display/verdict consistency guard: a FAIL's cause may be the
    # N-floor or an unresolved miss, which the 3 tolerance rows do not show --
    # only a PASS carrying an out-of-tolerance row (full sample OR broker_fill
    # subset) is a red flag (spine AD-26).
    if part_a.verdict == "PASS":
        out_of_tolerance = [
            label
            for stats in (part_a.stats, part_a.broker_fill_stats)
            for (label, _cell, _tol, ok) in _part_a_rows(stats)
            if not ok
        ]
        if out_of_tolerance:
            raise GateError(
                "part_a display rows are inconsistent with src.ticksim.config "
                "-- part_a was built against different constants, or there is "
                "an aggregate bug"
            )

    sha_from_git = sha is None
    resolved_sha = frozen_sha() if sha is None else sha

    n = part_a.stats.n
    a_min = "met" if n >= PART_A_MIN_N else "NOT met"
    date_line = date if date else _DATE_PLACEHOLDER
    integrity_line = integrity if integrity else _INTEGRITY_PLACEHOLDER

    lines: list[str] = [
        f"# Amendment {amendment_number} -- Parity gate result "
        f"(cycle {cycle_number})",
        "",
        f"_date: {date_line}_",
        "",
        "## Verdict",
        "",
        f"**{verdict.verdict}**",
        "",
        f"- part_a_pass: {verdict.part_a_pass}",
        f"- part_b_pass: {verdict.part_b_pass}",
        f"- reason: {verdict.reason}",
        "",
        "## Frozen SHA",
        "",
        f"simulator commit: {resolved_sha}",
    ]
    if sha_from_git:
        lines.append(
            "_`git rev-parse HEAD` taken as-is; run the gate from a committed "
            "state by discipline (a dirty tree is not checked here)._"
        )
    lines.extend(
        [
            "",
            "## Part A -- real-fill calibration",
            "",
            f"- sample N: {n}",
            f"- PART_A_MIN_N: {PART_A_MIN_N} -- {a_min}",
            f"- unresolved misses: {part_a.unresolved_misses}",
            "",
            "Full sample:",
            "",
            *_stat_table(part_a.stats),
            "",
            "`broker_fill`-only subset:",
            "",
            *_stat_table(
                part_a.broker_fill_stats,
                empty_message="(empty subset -- no broker_fill rows)",
            ),
        ]
    )
    if part_a.warning is not None:
        lines.extend(["", f"- warning: {part_a.warning}"])
    lines.append("")
    lines.extend(_per_trader_section(part_a, trader_by_trade_id))
    lines.append("")
    lines.extend(_part_b_section(part_b))
    lines.extend(
        [
            "",
            "## Integrity",
            "",
            integrity_line,
            "",
            "## Cycle / kill criterion",
            "",
            f"cycle {cycle_number} of 3",
            "",
            _KILL_CRITERION_SENTENCE,
            "",
        ]
    )
    return "\n".join(lines)
