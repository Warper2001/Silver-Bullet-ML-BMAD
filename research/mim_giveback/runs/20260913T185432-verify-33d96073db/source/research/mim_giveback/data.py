from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.mim_comparison.data import audit_select, load
from research.mim_robustness.features import features

from .artifacts import digest, readable
from .config import CONFIG, DEFAULT_DIAGNOSTIC_RUN


def verify_external_run(path: Path | str) -> dict[str, object]:
    path = readable(path)
    completion = json.loads((path / "completion.json").read_text())
    observed = {
        str(item.relative_to(path)): digest(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.name != "completion.json"
    }
    if observed != completion.get("sha256"):
        raise ValueError("Source run inventory integrity failure")
    return json.loads((path / "manifest.json").read_text())


def load_source(source_run: Path, data_path: Path) -> dict[str, object]:
    source_run = readable(source_run)
    manifest = verify_external_run(source_run)
    if manifest.get("command") != "run":
        raise ValueError("Completed PF shortlist run required")
    marks = pd.read_csv(source_run / "mim_decision_marks.csv")
    trades = pd.read_csv(source_run / "mim_trades.csv")
    summary = json.loads((source_run / "baseline_summary.json").read_text())
    verdict = json.loads((source_run / "mim_path_verdict.json").read_text())
    if verdict.get("verdict") != "DISTINCT_HYPOTHESIS_REMAINS":
        raise ValueError("Source hypothesis verdict changed")
    if len(marks) != 6673 or marks.mark_id.duplicated().any():
        raise ValueError("Source mark grid changed")
    if marks.candidate_return.notna().any():
        raise ValueError("Source run already contains candidate returns")
    expected = manifest["inputs"].get("data/mim_x/mnq_1min_by_contract.csv")
    if expected is None or digest(data_path) != expected:
        raise ValueError("Market data differs from the source manifest")
    if (
        summary.get("sessions") != CONFIG["baseline_sessions"]
        or summary.get("trades") != CONFIG["baseline_trades"]
        or not np.isclose(
            summary.get("net_profit"), CONFIG["baseline_net"], atol=1e-8, rtol=0
        )
        or not np.isclose(
            summary.get("closed_trade_net_pf"),
            CONFIG["baseline_pf"],
            atol=1e-12,
            rtol=0,
        )
        or summary.get("exit_labels") != CONFIG["baseline_exit_labels"]
    ):
        raise ValueError("Source baseline reconciliation changed")
    diagnostic = Path(
        manifest.get("defaults", {}).get("diagnostic_run", DEFAULT_DIAGNOSTIC_RUN)
    )
    expected_diagnostic = manifest["inputs"].get(
        "research/mim_robustness/runs/20260912T151842-run-f8608e71fb/completion.json"
    )
    if (
        expected_diagnostic is None
        or digest(diagnostic / "completion.json") != expected_diagnostic
    ):
        raise ValueError("Diagnostic parent differs from the source manifest")
    verify_external_run(diagnostic)
    daily = pd.read_csv(diagnostic / "daily.csv", float_precision="round_trip")
    daily = (
        daily[
            daily.arm.eq("A")
            & daily.delay.eq(2)
            & np.isclose(daily.cost, CONFIG["round_trip_cost"])
        ]
        .sort_values("day")
        .reset_index(drop=True)
    )
    if len(daily) != CONFIG["baseline_sessions"] or not np.isclose(
        daily.net.sum(), CONFIG["baseline_net"], atol=1e-8, rtol=0
    ):
        raise ValueError("Diagnostic daily baseline changed")
    return {
        "manifest": manifest,
        "marks": marks,
        "trades": trades,
        "summary": summary,
        "daily": daily,
        "diagnostic_run": diagnostic,
    }


def prepare_sessions(
    data_path: Path,
) -> list[tuple[object, np.ndarray, dict[str, np.ndarray]]]:
    consumed = readable(data_path).read_bytes()
    sessions, _ = audit_select(load(io.BytesIO(consumed), "end"))
    moves = [
        np.abs(s.bars.close.to_numpy(float) / float(s.bars.open.iloc[0]) - 1)
        for s in sessions
    ]
    prepared = []
    for index, session in enumerate(sessions):
        if index >= 14:
            sigma = np.mean(moves[index - 14 : index], axis=0)
            prepared.append((session, sigma, features(session, sigma)))
    if len(prepared) != CONFIG["baseline_sessions"]:
        raise ValueError("Prepared baseline session grid changed")
    return prepared


def threshold_family(marks: pd.DataFrame) -> pd.DataFrame:
    eligible = marks.loc[marks.mfe.gt(0)].copy()
    eligible["giveback_ratio"] = eligible.giveback / eligible.mfe
    if not np.isfinite(eligible.giveback_ratio).all():
        raise ValueError("Nonfinite giveback ratio")
    values = eligible.giveback_ratio.quantile(CONFIG["threshold_quantiles"])
    rows = [
        {"quantile": float(q), "threshold": float(value)} for q, value in values.items()
    ]
    family = pd.DataFrame(rows).drop_duplicates("threshold").reset_index(drop=True)
    if len(family) != 9:
        raise ValueError("Threshold quantiles are not nine distinct candidates")
    return family


def file_sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
