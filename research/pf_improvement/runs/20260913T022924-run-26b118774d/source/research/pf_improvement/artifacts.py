"""Immutable artifacts and independent verification for the PF shortlist."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import platform
import re
import shutil
import sys
import uuid

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
ORIGINAL = Path("/root/Silver-Bullet-ML-BMAD")
RUNS = BASE / "runs"
DEFAULT_MIM_DATA = ORIGINAL / "data/mim_nb"
DEFAULT_GAP_DATA = ORIGINAL / "data/gap_fade"
DEFAULT_DIAGNOSTIC_RUN = (
    ORIGINAL / "research/mim_robustness/runs/20260912T151842-run-f8608e71fb"
)
DEFAULT_CARRY_DATA = ORIGINAL / "data/commodity_curve"

# These are the approved bytes observed when the workflow was implemented.  The
# operational ledgers are mutable, so drift must force a new reviewed inventory.
PINNED_INPUTS = {
    "research/mim_robustness/runs/20260912T151842-run-f8608e71fb/completion.json": "b747ceff679c3874a6236d71eb960c4a68890aa5267aaa0575070b34b097a9a7",
    "research/mim_robustness/runs/20260912T151842-run-f8608e71fb/manifest.json": "0d0e7bd2e0d5db740f7dc8c231e70a43d6f4e517a24f8b77e2f1040191ee91fa",
    "research/mim_robustness/runs/20260912T151842-run-f8608e71fb/decisions.csv": "8366972fa95ae95b971452f9e2156d0a1dc7ff72fe00ac7f569c034d661b5aea",
    "research/mim_robustness/runs/20260912T151842-run-f8608e71fb/trades.csv": "17adbd5e6640af4c05c684516c40a47deb80c460f10e4d0609b996315fd04d0d",
    "research/mim_robustness/runs/20260912T151842-run-f8608e71fb/daily.csv": "42f93b0fa583c28f0316da3d41e6aaad6ce9bc89dda234d4c9349efdf1799efd",
    "data/mim_x/mnq_1min_by_contract.csv": "ff76aefca405dd94359b15223c57710f4e7f01f245880426a60d0f934c6f5bea",
    "data/mim_nb/decisions.csv": "acb8e244ae52d5306b96ac2576b69af5fff32d3260ff6fe220d0bd632ae35426",
    "data/mim_nb/orders.csv": "5e1cca8cc9f64ff85fb7f9e56aaf0081f1236c2228c1e1ac2321b7441a1a6b46",
    "data/mim_nb/trades.csv": "f2e7444cfc323042f7bead9a794f57e5ab0437ea46bb69e247e7c5efe1836a3c",
    "data/mim_nb/projectx_fills.json": "d98556116dc89a44ff5b8595b6ca4f5e6d82ac67f25436cdfba93a049e1446fd",
    "data/gap_fade/decisions.csv": "1c4f5a0f116629b9c2f8ed86d6c013f2f62450eebcbcf97ac2d4b28f386548c4",
    "data/gap_fade/fills.csv": "90093f444f6c4dfc9d70b5eab95f1821031623c314efb24dd974403178145a8f",
    "data/gap_fade/trades.csv": "968e83a64cfeaea2e651b0ca1a156373139d1137eb0d1933f0aa4decdc1378a7",
    "_bmad-output/specs/spec-commodity-curve-carry/SPEC.md": "f6a73b47598c33391f5015c94aaf2dbd49d2fd656e435407a7d2f1326158c4ff",
    "_bmad-output/specs/spec-commodity-curve-carry/data-contract.md": "26328f8fdc0fc9b0d6d50ae495edd0a3a2ef265d023be94c5acdc8f4044c9c5c",
    "_bmad-output/specs/spec-commodity-curve-carry/experiment-protocol.md": "8ffd36a9efb9dc0d49d6fea5af099ead1332ebd412f351b88dc353bed83ae72e",
    "_bmad-output/specs/spec-commodity-curve-carry/validation-plan.md": "869beacb3999a6b17c6fe9227e6dd67e7daf516123059ee8f3b15b0c701d82f3",
    "docs/commodity-curve-databento-fields-inquiry-20260907.md": "0bb3b22b8256bcfa1359ba57bf9636c16a142d08e51a83f1018bb062a829f255",
    "docs/commodity-curve-pilot-acceptance-20260907.md": "31b3ca700694a58180f80dfca17b21ebc18dcce5f25efc72c93feed09c759fa8",
    "docs/commodity-curve-send-time-audit-20260907.md": "eb8a8f06177fa4ddbc4fcce0fe2bfcf6b6318f0a6c8200bce4083fb43ccefa2f",
    "data/commodity_curve/coverage-pilot-2025-20260906-v3/contracts.csv": "2b626abc97e5839d2f44c4b93a4a84d1d2d25fb93e3b420b6b40dd0cc45390b0",
    "data/commodity_curve/development-jan-aug-2025-monthly-v2/report.json": "f841dcab9b5bcc078453b50c24e302124d24332dbab1b033e68a413aa0c22368",
    "data/commodity_curve/calendar-audit-2025-20260907/source-facts.json": "483bf6ebd05cbf7583db6b5e498086ff3a9e8c4f9cfe31c4c763609b83d65161",
    "data/commodity_curve/unit-audit-2025-20260907/exchange-unit-facts.json": "7021c723a57fdb0eb3ca0166591a2e6c3d20fe27f0b2ca804aa3b0854d70ecef",
    "tools/portfolio_decay_shadow.py": "09e76f04aab359c9252f47596a3fbbd60ee2797c00442a65e4a57b0d0297cc12",
    "logs/portfolio_decay_shadow.csv": "3b88d7221e1055ed0dbeec43de63447beb1575bd34c054e6a32113a08e6e827d",
    "research/mim_comparison/evidence/reference-audit.md": "1301bd3fca03ad7c8d5dd494abe01138251f61b595e57afda589ec8495bfb7e2",
    "research/mim_comparison/RESULTS.md": "6e1f970ab74a278da6efb8fb9d16bbab61265585849d352872a705991f1e4da6",
    "research/mim_lifecycle/RESULTS.md": "7747aae95682408f277954d4b318dbfa385e64c2a4d7ca76ba579e259fda2b2c",
    "_bmad-output/preregistration_mim_noise_bands.md": "dced42ae563d4e3cf62e19c2ae6386a9550ee6175ecf1a683fc116a85c62af28",
}

AUDIT_REQUIRED = {
    "manifest.json",
    "audit.json",
    "input_inventory.csv",
    "decay_monitor_coverage.json",
    "definitions.md",
    "completion.json",
}
RUN_REQUIRED = AUDIT_REQUIRED | {
    "execution_events.csv",
    "execution_round_trips.csv",
    "execution_coverage.csv",
    "execution_gate.json",
    "mim_trades.csv",
    "mim_decision_marks.csv",
    "mim_lineage_audit.csv",
    "baseline_summary.json",
    "carry_matrix.csv",
    "mim_path_distributions.csv",
    "mim_path_verdict.json",
    "mim_hypothesis_specification.md",
    "carry_evidence.csv",
    "carry_questions.md",
    "carry_verdict.json",
    "report.md",
    "report.html",
}
STOP_REQUIRED = AUDIT_REQUIRED | {
    "execution_events.csv",
    "execution_round_trips.csv",
    "execution_coverage.csv",
    "execution_gate.json",
    "repair_specification.md",
    "report.md",
    "report.html",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def readable(path: Path | str) -> Path:
    resolved = Path(path).resolve()
    if "sealed_holdout" in resolved.parts:
        raise ValueError("Sealed holdout access prohibited")
    return resolved


def digest(path: Path | str) -> str:
    path = readable(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def input_path(relative: str) -> Path:
    if relative not in PINNED_INPUTS:
        raise ValueError("Input is outside the approved inventory")
    path = readable(ORIGINAL / relative)
    if not path.is_relative_to(ORIGINAL.resolve()):
        raise ValueError("Input escaped the original checkout")
    return path


def validate_inputs() -> list[dict[str, object]]:
    rows = []
    for relative, expected in sorted(PINNED_INPUTS.items()):
        actual = digest(input_path(relative))
        if actual != expected:
            raise ValueError(f"Pinned input drift: {relative}")
        rows.append(
            {
                "path": relative,
                "sha256": actual,
                "bytes": input_path(relative).stat().st_size,
            }
        )
    return rows


def validate_requested_paths(
    mim_data: Path, gap_data: Path, diagnostic_run: Path, carry_data: Path
) -> None:
    """Keep the public path interface bound to the approved immutable inventory."""
    requested = {
        "mim-data": (mim_data, DEFAULT_MIM_DATA),
        "gap-data": (gap_data, DEFAULT_GAP_DATA),
        "diagnostic-run": (diagnostic_run, DEFAULT_DIAGNOSTIC_RUN),
        "carry-data": (carry_data, DEFAULT_CARRY_DATA),
    }
    for label, (actual, expected) in requested.items():
        if readable(actual) != readable(expected):
            raise ValueError(f"{label} is not the manifest-bound approved input")


def safe_run(path: Path | str) -> Path:
    raw = Path(path)
    if raw.is_symlink() or RUNS.is_symlink():
        raise ValueError("Symlink output prohibited")
    resolved = raw.resolve()
    if resolved.parent != RUNS.resolve():
        raise ValueError("Output must be a fresh direct runs child")
    return resolved


def create(command: str, output: Path | None = None) -> Path:
    if RUNS.is_symlink():
        raise ValueError("Symlink runs directory")
    RUNS.mkdir(exist_ok=True)
    path = safe_run(
        output
        or RUNS
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            + f"-{command}-{uuid.uuid4().hex[:10]}"
        )
    )
    path.mkdir()
    return path


def json_write(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def inventory(path: Path | str) -> dict[str, str]:
    path = readable(path)
    result = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("Symlink in run inventory")
        if item.is_file() and item != path / "completion.json":
            result[str(item.relative_to(path))] = digest(item)
    return result


def freeze(path: Path, command: str, input_rows: list[dict[str, object]]) -> None:
    source_files = sorted(BASE.glob("*.py")) + [
        BASE / "definitions.md",
        BASE / "official_venue_evidence.json",
    ]
    source = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    manifest = {
        "command": command,
        "created_at": now(),
        "deployment_authorized": False,
        "historical_only": True,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "machine": platform.machine(),
        },
        "defaults": {
            "arm": "A",
            "delay": 2,
            "quantity": 1,
            "round_trip_cost": 2.24,
            "entry_cost": 1.12,
            "point_value": 2.0,
            "mim_data": str(DEFAULT_MIM_DATA),
            "gap_data": str(DEFAULT_GAP_DATA),
            "diagnostic_run": str(DEFAULT_DIAGNOSTIC_RUN),
            "carry_data": str(DEFAULT_CARRY_DATA),
        },
        "inputs": {str(row["path"]): str(row["sha256"]) for row in input_rows},
        "source": source,
    }
    for source_file in source_files:
        target = path / "source" / source_file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
    shutil.copyfile(BASE / "definitions.md", path / "definitions.md")
    json_write(path / "manifest.json", manifest)


def seal(path: Path) -> None:
    # Completion is the final file: a failure before every payload is protected
    # leaves no apparently complete run.
    if (path / "completion.json").exists():
        raise FileExistsError("Run is already sealed")
    recorded = {"sha256": inventory(path)}
    for item in path.rglob("*"):
        if item.is_file():
            item.chmod(0o444)
    for directory in sorted(
        (p for p in path.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
    json_write(path / "completion.json", recorded)
    (path / "completion.json").chmod(0o444)
    path.chmod(0o555)


def verify_inventory(path: Path) -> None:
    if path.stat().st_mode & 0o222:
        raise ValueError("Sealed run child is writable")
    recorded = json.loads((path / "completion.json").read_text())["sha256"]
    for relative in recorded:
        candidate = (path / relative).resolve()
        if not candidate.is_relative_to(path.resolve()):
            raise ValueError("Inventory path escaped run")
    if inventory(path) != recorded:
        raise ValueError("Run inventory changed")


def _privacy_check(path: Path) -> None:
    raw = json.loads(input_path("data/mim_nb/projectx_fills.json").read_text())
    globally_forbidden = {
        str(row["accountId"]) for row in raw if row.get("accountId") is not None
    }
    globally_forbidden |= {str(row[key]) for row in raw for key in ("id", "orderId")}
    local_orders = pd.read_csv(input_path("data/mim_nb/orders.csv"), dtype=str)
    identifier_values = globally_forbidden | set(
        local_orders.loc[local_orders.order_id.ne("FAIL"), "order_id"].dropna()
    )
    gap_fills = pd.read_csv(input_path("data/gap_fade/fills.csv"), dtype=str)
    identifier_values |= set(gap_fills.entry_id.dropna()) | set(
        gap_fills.exit_id.dropna()
    )
    for filename, columns in {
        "execution_events.csv": ("order_id", "fill_id"),
        "execution_round_trips.csv": (
            "entry_order_id",
            "exit_order_id",
            "entry_fill_id",
            "exit_fill_id",
        ),
    }.items():
        candidate = path / filename
        if candidate.is_file():
            frame = pd.read_csv(candidate, dtype=str)
            observed = {
                value
                for column in columns
                for value in frame[column].dropna().astype(str)
            }
            if observed & identifier_values:
                raise ValueError(f"Raw identifier in {filename}")
    for item in path.rglob("*"):
        if (
            not item.is_file()
            or item.name == "completion.json"
            or "source" in item.relative_to(path).parts
        ):
            continue
        text = item.read_text(errors="ignore")
        if any(
            re.search(rf"(?<!\d){re.escape(value)}(?!\d)", text)
            for value in globally_forbidden
        ):
            raise ValueError(f"Privacy drift in {item.relative_to(path)}")


def _agree(left, right, label: str, tolerance: float = 1e-8) -> None:
    if not np.allclose(left, right, atol=tolerance, rtol=0, equal_nan=False):
        raise ValueError("Independent verification: " + label)


def _verify_audit(path: Path, manifest: dict[str, object]) -> None:
    audit = json.loads((path / "audit.json").read_text())
    expected_flags = {
        "valid": True,
        "raw_account_evidence_copied": False,
        "sealed_holdout_accessed": False,
        "strategy_simulator_invoked": False,
        "alternative_strategy_returns_calculated": False,
    }
    if any(audit.get(key) != value for key, value in expected_flags.items()):
        raise ValueError("Audit semantics changed")
    input_inventory = pd.read_csv(path / "input_inventory.csv").sort_values("path")
    if len(input_inventory) != len(PINNED_INPUTS) or audit.get("input_files") != len(
        input_inventory
    ):
        raise ValueError("Audit input coverage changed")
    if dict(zip(input_inventory.path, input_inventory.sha256)) != manifest["inputs"]:
        raise ValueError("Audit inventory differs from manifest")
    expected_bytes = {
        relative: input_path(relative).stat().st_size for relative in PINNED_INPUTS
    }
    if dict(zip(input_inventory.path, input_inventory.bytes)) != expected_bytes:
        raise ValueError("Audit input sizes changed")
    observed = json.loads((path / "decay_monitor_coverage.json").read_text())
    raw = pd.read_csv(input_path("logs/portfolio_decay_shadow.csv"))
    expected_timestamps = sorted(raw.run_at.dropna().astype(str).unique().tolist())
    if (
        observed.get("rows") != len(raw)
        or observed.get("observation_timestamps") != expected_timestamps
        or observed.get("efficacy_interpreted")
        or observed.get("monitor_invoked")
        or observed.get("strategy_rows")
        != {str(k): int(v) for k, v in raw.strategy.value_counts().sort_index().items()}
    ):
        raise ValueError("Decay monitor coverage semantics changed")


def _gate_verdict(gate: dict[str, object]) -> str:
    # Use explicit names rather than trusting the authored verdict.
    complete = bool(gate.get("complete_current_coverage"))
    causal = bool(gate.get("exact_causality"))
    unchanged = bool(gate.get("unchanged_intended_behavior"))
    defect = bool(gate.get("recurring_current_economic_defect"))
    if complete and causal and unchanged and defect:
        return "CURRENT_REPAIR_CANDIDATE"
    if complete and causal and unchanged and not defect:
        return "NO_REPAIRABLE_MECHANISM"
    return "INSUFFICIENT_CAUSAL_EVIDENCE"


def _verify_stopped_inventory(present: set[str]) -> None:
    forbidden = RUN_REQUIRED - STOP_REQUIRED
    if present & forbidden:
        raise ValueError("Stopped run contains forbidden later-stage artifacts")


def _verify_report_claims(
    report_md: str,
    report_html: str,
    markdown_claims: list[str],
    html_claims: list[str],
) -> None:
    if any(claim not in report_md for claim in markdown_claims) or any(
        claim not in report_html for claim in html_claims
    ):
        raise ValueError("Report claims conflict with ledgers")


def _verify_distributions(distributions: pd.DataFrame, marks: pd.DataFrame) -> None:
    required_dimensions = {
        "all",
        "final_outcome",
        "direction",
        "final_exit_reason",
        "top_5pct_trade",
    }
    if set(distributions.group_dimension) != required_dimensions:
        raise ValueError("Incomplete fixed MIM distribution partitions")
    if (
        distributions.candidate_threshold.notna().any()
        or distributions.candidate_return_calculated.any()
    ):
        raise ValueError("MIM distributions crossed the descriptive boundary")
    for row in distributions.itertuples(index=False):
        if row.group_dimension == "all":
            group = marks
        else:
            column = marks[row.group_dimension]
            group = marks[column.astype(str).eq(str(row.group_value))]
        series = group[row.metric].dropna().astype(float)
        if len(group) != row.total_marks or len(series) != row.observed_values:
            raise ValueError("MIM distribution coverage mismatch")
        expected_mean = float(series.mean()) if len(series) else np.nan
        expected_std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
        if not np.isclose(
            row.mean, expected_mean, atol=1e-8, rtol=0, equal_nan=True
        ) or not np.isclose(
            row.sample_std, expected_std, atol=1e-8, rtol=0, equal_nan=True
        ):
            raise ValueError("MIM distribution moments mismatch")
        for q in range(0, 101, 10):
            expected = float(series.quantile(q / 100))
            if not np.isclose(getattr(row, f"q{q:03d}"), expected, atol=1e-8, rtol=0):
                raise ValueError("MIM distribution quantile mismatch")


def _pseudo(kind: str, value: object) -> str:
    return hashlib.sha256(f"pf-improvement-v1|{kind}|{value}".encode()).hexdigest()[:20]


def _verify_execution_sources(events: pd.DataFrame, trips: pd.DataFrame) -> None:
    raw_fills = sorted(
        json.loads(input_path("data/mim_nb/projectx_fills.json").read_text()),
        key=lambda row: row["creationTimestamp"],
    )
    orders = pd.read_csv(input_path("data/mim_nb/orders.csv"), dtype={"order_id": str})
    local_fill_ids = set(orders.loc[orders.event.eq("FILL"), "order_id"].dropna())
    broker_events = events[events.source.eq("saved_broker_export")]
    if len(broker_events) != len(raw_fills):
        raise ValueError("Saved broker event coverage changed")
    for raw in raw_fills:
        order_token = _pseudo("projectx-order", raw["orderId"])
        fill_token = _pseudo("projectx-fill", raw["id"])
        matched = broker_events[
            broker_events.order_id.eq(order_token)
            & broker_events.fill_id.eq(fill_token)
        ]
        if len(matched) != 1:
            raise ValueError("Pseudonymous broker identifier join failed")
        row = matched.iloc[0]
        expected_side = "BUY" if int(raw["side"]) == 0 else "SELL"
        if (
            row.side != expected_side
            or int(row["size"]) != int(raw["size"])
            or row.event_timestamp_utc
            != pd.Timestamp(raw["creationTimestamp"]).isoformat()
        ):
            raise ValueError("Saved broker event identity conflict")
        _agree(
            [row.price, row.costs],
            [float(raw["price"]), float(raw["fees"]) + float(raw["commissions"])],
            "broker event fields",
        )
    for opening, closing in zip(raw_fills[::2], raw_fills[1::2]):
        matched = trips[
            trips.entry_order_id.eq(_pseudo("projectx-order", opening["orderId"]))
            & trips.exit_order_id.eq(_pseudo("projectx-order", closing["orderId"]))
        ]
        if len(matched) != 1:
            raise ValueError("Saved broker pair join failed")
        row = matched.iloc[0]
        direction = 1 if int(opening["side"]) == 0 else -1
        gross = (
            (float(closing["price"]) - float(opening["price"]))
            * direction
            * 2.0
            * int(opening["size"])
        )
        costs = sum(
            float(x["fees"]) + float(x["commissions"]) for x in (opening, closing)
        )
        _agree(
            [
                row.entry_price,
                row.exit_price,
                row.broker_gross,
                row.costs,
                row.broker_net,
            ],
            [opening["price"], closing["price"], gross, costs, gross - costs],
            "broker pair fields",
        )
        exact = (
            str(opening["orderId"]) in local_fill_ids
            and str(closing["orderId"]) in local_fill_ids
        )
        if (row.strategy == "MIM") != exact or bool(
            row.strategy_side_size_validated
        ) != exact:
            raise ValueError("Broker pair attribution conflicts with exact source IDs")
    gap_raw = pd.read_csv(
        input_path("data/gap_fade/fills.csv"), dtype={"entry_id": str, "exit_id": str}
    )
    gap = trips[trips.strategy.eq("GAP")]
    if len(gap) != len(gap_raw):
        raise ValueError("GAP source coverage changed")
    for raw in gap_raw.itertuples(index=False):
        matched = gap[
            gap.entry_order_id.eq(_pseudo("tradestation-order", raw.entry_id))
            & gap.exit_order_id.eq(_pseudo("tradestation-order", raw.exit_id))
        ]
        if len(matched) != 1:
            raise ValueError("GAP pseudonymous ID join failed")
        row = matched.iloc[0]
        direction = 1 if raw.dir == "L" else -1
        gross = (
            (float(raw.exit_exec) - float(raw.entry_exec))
            * direction
            * 2.0
            * int(raw.qty)
        )
        _agree(
            [
                row.entry_price,
                row.exit_price,
                row.broker_gross,
                row.modeled_gross,
                row.signed_difference,
            ],
            [raw.entry_exec, raw.exit_exec, gross, raw.modeled_pnl_usd, raw.delta_usd],
            "GAP source fields",
        )


def _verify_gate_evidence(gate: dict[str, object], trips: pd.DataFrame) -> None:
    current = trips[
        trips.strategy.isin(["MIM", "GAP"])
        & trips.configuration_era.isin(["current_risk_mechanics", "ledger_hardened"])
    ]
    complete = bool(
        len(current)
        and current.join_method.eq("exact_order_id").all()
        and current.complete_costs.all()
        and current.entry_timestamp_utc.notna().all()
        and current.exit_timestamp_utc.notna().all()
    )
    causal = bool(len(current) and current.causal_decomposition_available.all())
    unchanged = bool(
        len(current)
        and current.strategy_side_size_validated.all()
        and current.attribution_status.eq("exact_orders_and_unique_local_trade").all()
    )
    eligible = current[
        current.causal_decomposition_available
        & current.complete_costs
        & current.signed_difference.notna()
    ]
    defect = bool(len(eligible) >= 2 and (eligible.signed_difference < 0).all())
    evidence = {
        "complete_current_coverage": complete,
        "exact_causality": causal,
        "unchanged_intended_behavior": unchanged,
        "recurring_current_economic_defect": defect,
    }
    if any(gate.get(key) != value for key, value in evidence.items()) or gate.get(
        "verdict"
    ) != _gate_verdict(gate):
        raise ValueError("Execution gate conflicts with emitted evidence")


def _verify_mark_sources(marks: pd.DataFrame) -> None:
    bars = pd.read_csv(
        input_path("data/mim_x/mnq_1min_by_contract.csv"),
        usecols=["contract", "timestamp", "high", "low", "close"],
    )
    bars["timestamp"] = pd.to_datetime(bars.timestamp, utc=True).dt.tz_convert(
        "America/New_York"
    )
    bars["day"] = bars.timestamp.dt.strftime("%Y-%m-%d")
    keys = set(zip(marks.day, marks.contract))
    bars = bars[pd.MultiIndex.from_frame(bars[["day", "contract"]]).isin(keys)]
    sessions = {
        key: frame.sort_values("timestamp").reset_index(drop=True)
        for key, frame in bars.groupby(["day", "contract"], sort=False)
    }
    for row in marks.itertuples(index=False):
        entry_time = pd.Timestamp(row.entry_fill_timestamp).tz_convert(
            "America/New_York"
        )
        mark_time = pd.Timestamp(row.mark_timestamp).tz_convert("America/New_York")
        held = sessions[row.day, row.contract]
        held = held[(held.timestamp > entry_time) & (held.timestamp <= mark_time)]
        expected = pd.date_range(
            entry_time + pd.Timedelta(minutes=1), mark_time, freq="min"
        )
        if held.timestamp.duplicated().any() or not pd.DatetimeIndex(
            held.timestamp
        ).equals(expected):
            raise ValueError("Independent held-minute grid mismatch")
        if len(held) != int(row.completed_bar_coverage) or not np.isclose(
            held.close.iloc[-1], row.signal_price, atol=1e-8, rtol=0
        ):
            raise ValueError("Independent mark coverage mismatch")
        favorable = np.maximum(
            row.direction * (held.high.to_numpy(float) - row.entry_fill) * 2.0,
            row.direction * (held.low.to_numpy(float) - row.entry_fill) * 2.0,
        )
        adverse = np.minimum(
            row.direction * (held.high.to_numpy(float) - row.entry_fill) * 2.0,
            row.direction * (held.low.to_numpy(float) - row.entry_fill) * 2.0,
        )
        _agree(
            [row.mfe, row.mae],
            [max(0.0, favorable.max()), max(0.0, -adverse.min())],
            "held-mark excursions",
        )


def verify(path: Path | str, sealed: bool = True) -> dict[str, object]:
    path = safe_run(path)
    if sealed:
        verify_inventory(path)
    if (path / "failure.json").exists():
        raise ValueError("Failed invocation is preserved, not a successful run")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("command") == "audit":
        required = AUDIT_REQUIRED
    elif (path / "execution_gate.json").is_file() and json.loads(
        (path / "execution_gate.json").read_text()
    ).get("verdict") == "CURRENT_REPAIR_CANDIDATE":
        required = STOP_REQUIRED
    else:
        required = RUN_REQUIRED
    present = set(inventory(path)) | {"completion.json"}
    if not required <= present:
        raise ValueError("Missing mandatory inventory")
    expected_inputs = {r["path"]: r["sha256"] for r in validate_inputs()}
    if manifest["inputs"] != expected_inputs:
        raise ValueError("Manifest input inventory drift")
    for relative, expected in manifest["source"].items():
        if digest(path / "source" / relative) != expected:
            raise ValueError("Source snapshot drift")
    _privacy_check(path)
    _verify_audit(path, manifest)
    if manifest["command"] == "audit":
        forbidden = (RUN_REQUIRED | STOP_REQUIRED) - AUDIT_REQUIRED
        if present & forbidden:
            raise ValueError("Audit run contains analytical artifacts")
        return {"verified": True, "command": "audit"}

    gate = json.loads((path / "execution_gate.json").read_text())
    if gate.get("verdict") != _gate_verdict(gate):
        raise ValueError("Execution gate conflicts with recorded evidence")
    if gate["verdict"] == "CURRENT_REPAIR_CANDIDATE":
        _verify_stopped_inventory(present)
    execution_events = pd.read_csv(
        path / "execution_events.csv", dtype={"order_id": str, "fill_id": str}
    )
    execution = pd.read_csv(
        path / "execution_round_trips.csv",
        dtype={
            "entry_order_id": str,
            "exit_order_id": str,
            "entry_fill_id": str,
            "exit_fill_id": str,
        },
    )
    _verify_execution_sources(execution_events, execution)
    _verify_gate_evidence(gate, execution)
    if gate["verdict"] == "CURRENT_REPAIR_CANDIDATE":
        if (
            not gate.get("stops_later_stages")
            or gate.get("recoverable_dollars") is None
        ):
            raise ValueError(
                "Repair candidate lacks stop or measured economic evidence"
            )
        return {
            "verified": True,
            "command": "run",
            "stopped": "CURRENT_REPAIR_CANDIDATE",
        }

    if execution.strategy.value_counts().to_dict() != {
        "MIM": 25,
        "GAP": 21,
        "UNATTRIBUTED_SHARED_ACCOUNT": 2,
    }:
        raise ValueError("Execution attribution counts changed")
    exact_mim = execution[
        (execution.strategy == "MIM") & (execution.join_method == "exact_order_id")
    ]
    if len(exact_mim) != 5 or not exact_mim.strategy_side_size_validated.all():
        raise ValueError("Exact MIM attribution changed")
    _agree(
        exact_mim.broker_gross - exact_mim.costs, exact_mim.broker_net, "MIM broker net"
    )
    _agree(
        exact_mim.broker_gross - exact_mim.modeled_gross,
        exact_mim.signed_difference,
        "MIM signed execution difference",
    )
    gap = execution[execution.strategy == "GAP"]
    _agree(
        (gap.exit_price - gap.entry_price) * gap.direction * gap["size"] * 2.0,
        gap.broker_gross,
        "GAP broker gross",
    )
    _agree(
        gap.broker_gross - gap.modeled_gross,
        gap.signed_difference,
        "GAP signed difference",
    )
    if gap.broker_net.notna().any() or gap.complete_costs.any():
        raise ValueError("GAP incomplete costs became broker net")
    unattributed = execution[execution.strategy == "UNATTRIBUTED_SHARED_ACCOUNT"]
    if len(unattributed) != 2 or unattributed.modeled_gross.notna().any():
        raise ValueError("Shared-account evidence was attributed")
    coverage = pd.read_csv(path / "execution_coverage.csv").set_index("source")
    expected_breaks = {"mim_decisions": 128, "mim_orders": 79, "gap_decisions": 30}
    for source, row in expected_breaks.items():
        if int(coverage.loc[source, "first_chain_break_row"]) != row:
            raise ValueError("Execution chain boundary changed")
    if (
        gate["verdict"] == "INSUFFICIENT_CAUSAL_EVIDENCE"
        and gate["recoverable_dollars"] is not None
    ):
        raise ValueError("Missing causality produced recoverable dollars")

    trades = pd.read_csv(path / "mim_trades.csv")
    marks = pd.read_csv(path / "mim_decision_marks.csv")
    summary = json.loads((path / "baseline_summary.json").read_text())
    if len(trades) != 801 or len(marks) != 6673 or marks.mark_id.duplicated().any():
        raise ValueError("Pinned MIM count mismatch")
    _verify_mark_sources(marks)
    if trades.exit_reason.value_counts().to_dict() != {
        "EOD_CLOSE_PROXY": 723,
        "CAT_STOP": 71,
        "REVERSAL": 7,
    }:
        raise ValueError("Pinned exit labels changed")
    _agree(
        (trades.exit_fill - trades.entry_fill) * trades.direction * 2.0,
        trades.gross,
        "trade gross",
    )
    _agree(trades.gross - 2.24, trades.net, "trade net")
    _agree(trades.net.sum(), 21889.76, "pinned net")
    _agree(marks.current_gross - 1.12, marks.current_net, "open-mark incurred costs")
    _agree(
        (marks.signal_price - marks.entry_fill) * marks.direction * 2.0,
        marks.current_gross,
        "open-mark gross",
    )
    _agree(marks.mfe - marks.current_gross, marks.giveback, "causal giveback")
    expected_change = marks.groupby("trade_id", sort=False).current_net.diff()
    if not np.allclose(
        marks.prior_mark_net_change,
        expected_change,
        atol=1e-8,
        rtol=0,
        equal_nan=True,
    ):
        raise ValueError("Prior-mark change mismatch")
    if (marks.mark_timestamp.str[11:16] < "10:00").any() or (
        marks.mark_timestamp.str[11:16] > "15:30"
    ).any():
        raise ValueError("Scheduled mark outside declared clock")
    mark_exits = marks[["trade_id", "mark_timestamp"]].merge(
        trades[["trade_id", "exit_event_timestamp"]],
        on="trade_id",
        validate="many_to_one",
    )
    if not (
        pd.to_datetime(mark_exits.mark_timestamp, utc=True)
        < pd.to_datetime(mark_exits.exit_event_timestamp, utc=True)
    ).all():
        raise ValueError("Post-exit mark included")
    if (
        int(marks.reversal_mark_exiting_leg.sum()) != 7
        or marks.candidate_return.notna().any()
    ):
        raise ValueError("Reversal ownership or research boundary changed")
    if (marks.mfe < 0).any() or (marks.mae < 0).any() or (marks.giveback < -1e-8).any():
        raise ValueError("Invalid excursion fields")
    if (
        not marks.groupby("trade_id")
        .mfe.apply(lambda s: s.is_monotonic_increasing)
        .all()
    ):
        raise ValueError("MFE is not causal running state")
    if (
        not marks.groupby("trade_id")
        .mae.apply(lambda s: s.is_monotonic_increasing)
        .all()
    ):
        raise ValueError("MAE is not causal running state")
    if (
        summary["sessions"] != 1323
        or summary["trades"] != 801
        or abs(summary["net_profit"] - 21889.76) > 1e-8
    ):
        raise ValueError("Baseline summary mismatch")
    source_daily = pd.read_csv(
        input_path(
            "research/mim_robustness/runs/20260912T151842-run-f8608e71fb/daily.csv"
        )
    )
    source_daily = (
        source_daily[
            source_daily.arm.eq("A")
            & source_daily.delay.eq(2)
            & np.isclose(source_daily.cost, 2.24)
        ]
        .sort_values("day")
        .reset_index(drop=True)
    )
    if source_daily.day.duplicated().any():
        raise ValueError("Duplicate selected daily source row")
    for field in ("gross", "costs", "net"):
        expected_daily = (
            trades.groupby("day")[field].sum().reindex(source_daily.day, fill_value=0)
        )
        _agree(expected_daily, source_daily[field], "daily-to-trade " + field)
    winning = float(trades.loc[trades.net > 0, "net"].sum())
    losing = float(-trades.loc[trades.net < 0, "net"].sum())
    entries = pd.to_datetime(trades.entry_fill_timestamp, utc=True)
    events = pd.to_datetime(trades.exit_event_timestamp, utc=True)
    fills = pd.to_datetime(trades.exit_fill_timestamp, utc=True, errors="coerce")
    upper = np.where(
        trades.exit_reason.eq("CAT_STOP"),
        (events - entries).dt.total_seconds() / 60,
        (fills - entries).dt.total_seconds() / 60,
    )
    lower = np.where(trades.exit_reason.eq("CAT_STOP"), upper - 1, upper)
    equity = source_daily.net.cumsum().to_numpy(float)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])[-len(equity) :]
    top_count = math.ceil(0.05 * len(trades))
    top_trade_net = float(trades.nlargest(top_count, "net").net.sum())
    top_day_net = float(source_daily.nlargest(67, "net").net.sum())
    expected_summary = {
        "closed_trade_net_pf": winning / losing,
        "net_profit": float(trades.net.sum()),
        "trades": len(trades),
        "sessions": len(source_daily),
        "winners": int((trades.net > 0).sum()),
        "losers": int((trades.net < 0).sum()),
        "winning_dollars": winning,
        "losing_dollars": losing,
        "exposure_contract_minutes_lower": float(np.sum(lower)),
        "exposure_contract_minutes_upper": float(np.sum(upper)),
        "daily_max_drawdown": float(-(equity - peaks).min()),
        "top_5pct_trade_count": top_count,
        "top_5pct_trade_net": top_trade_net,
        "top_5pct_trade_fraction_of_net": top_trade_net / float(trades.net.sum()),
        "top_67_day_net": top_day_net,
        "top_67_day_fraction_of_net": top_day_net / float(trades.net.sum()),
    }
    for field, expected in expected_summary.items():
        actual = summary.get(field)
        if isinstance(expected, int):
            if actual != expected:
                raise ValueError("Baseline summary mismatch: " + field)
        else:
            _agree([actual], [expected], "baseline summary " + field)
    carry = pd.read_csv(path / "carry_matrix.csv")
    roots = {"CL", "NG", "RB", "HO", "HG", "ZC", "ZW", "ZS", "ZM", "ZL", "LE", "HE"}
    venues = {"Topstep", "TradeStation_SIM", "future_self_funded"}
    expected_pairs = {(root, venue) for root in roots for venue in venues}
    if (
        set(zip(carry.root, carry.venue)) != expected_pairs
        or carry.duplicated(["root", "venue"]).any()
    ):
        raise ValueError("Carry matrix is not 12 roots by 3 venues")
    if not carry.park_reasons.eq("PARK_ACCOUNT|PARK_DATA|PARK_POWER").all():
        raise ValueError("Carry row missing park reason")
    allowed_status = {"usable", "requires verification/acquisition", "unavailable"}
    for column in (
        "product_permission_status",
        "overnight_status",
        "integer_sizing_status",
        "margin_status",
        "cost_quote_status",
        "calendar_status",
    ):
        if not set(carry[column]) <= allowed_status:
            raise ValueError("Unknown carry feasibility status")
    mappings = {
        "Topstep": ("UNAVAILABLE_OVERNIGHT", "usable", "unavailable"),
        "TradeStation_SIM": (
            "VALIDATION_ONLY",
            "requires verification/acquisition",
            "requires verification/acquisition",
        ),
        "future_self_funded": (
            "REQUIREMENTS_ONLY",
            "requires verification/acquisition",
            "requires verification/acquisition",
        ),
    }
    for venue, expected in mappings.items():
        frame = carry[carry.venue.eq(venue)]
        if not (
            frame.path_classification.eq(expected[0]).all()
            and frame.product_permission_status.eq(expected[1]).all()
            and frame.overnight_status.eq(expected[2]).all()
        ):
            raise ValueError("Carry venue mapping conflicts with bound facts")
    development = json.loads(
        input_path(
            "data/commodity_curve/development-jan-aug-2025-monthly-v2/report.json"
        ).read_text()
    )
    if (
        set(development["roots"]) != roots
        or development["rows"] != 95537
        or development["official_publication_rows"] != 0
        or development["research_status"] != "HOLD-DATA"
    ):
        raise ValueError("Carry development facts changed")

    carry_verdict = json.loads((path / "carry_verdict.json").read_text())
    if (
        set(carry_verdict.get("overall", []))
        != {"PARK_ACCOUNT", "PARK_DATA", "PARK_POWER", "SOURCE_CLARIFICATION_REQUIRED"}
        or carry_verdict.get("matrix_rows") != len(carry)
        or carry_verdict.get("roots") != len(roots)
        or carry_verdict.get("venues") != len(venues)
        or not carry_verdict.get("questions_prepared_not_sent")
        or carry_verdict.get("purchase_authorized")
        or carry_verdict.get("strategy_returns_calculated")
    ):
        raise ValueError("Carry verdict conflicts with matrix")
    evidence = pd.read_csv(path / "carry_evidence.csv")
    official = json.loads(
        (
            path / "source/research/pf_improvement/official_venue_evidence.json"
        ).read_text()
    )
    if not {row["url"] for row in official["sources"]} <= set(evidence.source):
        raise ValueError("Carry venue evidence inventory incomplete")
    if (
        set(evidence[evidence.category.eq("official_exchange_unit_rule")].root_scope)
        != roots
    ):
        raise ValueError("Carry official root evidence incomplete")
    questions = (path / "carry_questions.md").read_text()
    if "not sent" not in questions or not all(
        section in questions
        for section in ("## Data source", "## Account paths", "## Power and seal")
    ):
        raise ValueError("Carry questions conflict with verdict")
    path_verdict = json.loads((path / "mim_path_verdict.json").read_text())
    if path_verdict["verdict"] not in {
        "DISTINCT_HYPOTHESIS_REMAINS",
        "DUPLICATES_PRIOR_WORK",
        "NO_DESCRIPTIVE_SEPARATION",
        "INSUFFICIENT_PATH_COVERAGE",
    }:
        raise ValueError("Unknown MIM path verdict")
    if (
        path_verdict["candidate_returns_calculated"]
        or path_verdict["threshold_selected"]
    ):
        raise ValueError("MIM feasibility crossed the research boundary")
    distributions = pd.read_csv(path / "mim_path_distributions.csv")
    _verify_distributions(distributions, marks)
    lineage = pd.read_csv(path / "mim_lineage_audit.csv")
    if set(lineage.reference) != {
        "MIM V1",
        "MIM V2",
        "published neutral exit",
        "MIM lifecycle",
    }:
        raise ValueError("MIM lineage inventory changed")
    for row in lineage.itertuples(index=False):
        if (
            row.source_path not in PINNED_INPUTS
            or row.source_marker not in input_path(row.source_path).read_text()
        ):
            raise ValueError("MIM lineage is not supported by pinned prior work")
    duplicate = lineage.distinctness.str.contains(
        "duplicates scheduled-mark giveback", case=False
    ).any()
    expected_path_verdict = (
        "DUPLICATES_PRIOR_WORK" if duplicate else "DISTINCT_HYPOTHESIS_REMAINS"
    )
    if path_verdict["verdict"] != expected_path_verdict or path_verdict[
        "prior_rule_duplicate"
    ] != bool(duplicate):
        raise ValueError("MIM path verdict conflicts with lineage")
    report_md = (path / "report.md").read_text()
    report_html = (path / "report.html").read_text()
    claims = [
        gate["verdict"],
        path_verdict["verdict"],
        f'{summary["closed_trade_net_pf"]:.8f}',
        "6,673",
        ", ".join(carry_verdict["overall"]),
    ]
    _verify_report_claims(
        report_md,
        report_html,
        claims,
        [gate["verdict"], path_verdict["verdict"], "6,673"],
    )
    return {"verified": True, "command": "run", "marks": 6673, "trades": 801}
