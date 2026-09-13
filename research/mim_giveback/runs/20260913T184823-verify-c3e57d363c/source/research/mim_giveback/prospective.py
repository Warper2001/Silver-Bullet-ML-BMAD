from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from research.mim_comparison.data import audit_select, load
from research.mim_robustness.features import features

from .artifacts import digest, resolve_path, verify_run
from .config import ROOT
from .statistics import one_sided_ci, performance
from .study import simulate_sessions

BAR_COLUMNS = ["contract", "timestamp", "open", "high", "low", "close", "volume"]


def protocol_committed(protocol_run: Path) -> tuple[str, pd.Timestamp]:
    """Return the commit and time whose protocol completion bytes are in use."""
    relative = (protocol_run / "completion.json").resolve().relative_to(ROOT.resolve())
    # AGENTS.md requires these repository checks before Git operations.
    subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    for revision_range in ("origin/main..HEAD", "HEAD..origin/main"):
        subprocess.run(
            ["git", "-C", str(ROOT), "rev-list", "--count", revision_range],
            check=True,
            capture_output=True,
            text=True,
        )
    subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--error-unmatch", str(relative)],
        check=True,
        capture_output=True,
        text=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", str(relative)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not commit:
        raise ValueError("Frozen protocol is tracked but not committed")
    committed = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{relative}"],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != digest(
        protocol_run / "completion.json"
    ):
        raise ValueError("Frozen protocol completion differs from committed Git bytes")
    committed_at = subprocess.run(
        ["git", "-C", str(ROOT), "show", "-s", "--format=%cI", commit],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, pd.Timestamp(committed_at)


def _read_future(path: Path, as_of: datetime | None = None) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {*BAR_COLUMNS, "received_at"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Future observations missing fields: {sorted(missing)}")
    for column in ("timestamp", "received_at"):
        if not raw[column].astype(str).str.contains(r"(?:Z|[+-]\d\d:\d\d)$").all():
            raise ValueError(f"{column} must declare timezone offset")
    raw["timestamp_parsed"] = pd.to_datetime(raw.timestamp, utc=True, format="ISO8601")
    raw["received_parsed"] = pd.to_datetime(raw.received_at, utc=True, format="ISO8601")
    if raw[["contract", "timestamp_parsed"]].duplicated().any():
        raise ValueError("Duplicate future contract-minute")
    values = raw[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if (
        not np.isfinite(values).all().all()
        or (values.iloc[:, :4] <= 0).any().any()
        or (values.volume < 0).any()
    ):
        raise ValueError("Invalid future numeric values")
    if (
        (values.high < values[["open", "low", "close"]].max(axis=1))
        | (values.low > values[["open", "high", "close"]].min(axis=1))
    ).any():
        raise ValueError("Invalid future OHLC ordering")
    if (raw.received_parsed < raw.timestamp_parsed).any():
        raise ValueError("Observation received before its completed timestamp")
    now = pd.Timestamp(as_of or datetime.now(timezone.utc))
    if (raw.received_parsed > now).any():
        raise ValueError("Future-dated receipt")
    raw["timely"] = raw.received_parsed <= raw.timestamp_parsed + pd.Timedelta(
        minutes=1
    )
    if not raw.timely.all():
        raise ValueError("Observation received after execution deadline")
    return raw


def _protocol_rules(protocol: dict[str, object]) -> dict[str, object]:
    required = {
        "delay",
        "round_trip_cost",
        "quantity",
        "target_pf",
        "profit_retention",
        "endpoint_sessions",
        "bootstrap_blocks",
        "bootstrap_draws",
        "seed",
        "alpha",
    }
    rules = protocol.get("rules")
    if not isinstance(rules, dict) or required - set(rules):
        raise ValueError("Frozen protocol rules are incomplete")
    return rules


def _history(protocol: dict[str, object]) -> Path:
    path = resolve_path(str(protocol["history_data"]))
    if digest(path) != protocol["history_data_hash"]:
        raise ValueError("Frozen historical market data changed")
    return path


def _require_frozen_source(manifest: dict[str, object]) -> None:
    for relative, expected in manifest.get("source", {}).items():
        current = ROOT / relative
        if not current.is_file() or digest(current) != expected:
            raise ValueError("Current evaluator source differs from frozen protocol")


def collect(
    protocol_run: Path,
    data_path: Path,
    prior_run: Path | None = None,
    as_of: datetime | None = None,
):
    protocol_manifest = verify_run(protocol_run)
    if protocol_manifest["command"] != "freeze":
        raise ValueError("Collection requires a freeze run")
    _require_frozen_source(protocol_manifest)
    protocol = json.loads((protocol_run / "protocol.json").read_text())
    rules = _protocol_rules(protocol)
    history = _history(protocol)
    commit, committed_at = protocol_committed(protocol_run)
    raw = _read_future(data_path, as_of=as_of)
    boundary = max(pd.Timestamp(protocol["frozen_at"]), committed_at)
    deadline = pd.Timestamp(protocol["deadline"])
    if raw.timestamp_parsed.le(boundary).any():
        raise ValueError("Pre-commit or pre-freeze observation")
    if raw.timestamp_parsed.gt(deadline).any():
        raise ValueError("Observation is after the frozen execution deadline")
    columns = [*BAR_COLUMNS, "received_at", "timely"]
    chain = raw[columns].copy()
    prior_hash = None
    if prior_run is not None:
        prior_manifest = verify_run(prior_run)
        if prior_manifest["command"] != "collect":
            raise ValueError("Prior run is not a collection")
        prior_status = json.loads((prior_run / "collection_status.json").read_text())
        if prior_status["protocol_completion_hash"] != digest(
            protocol_run / "completion.json"
        ):
            raise ValueError("Collection chain protocol mismatch")
        if int(prior_status["eligible_sessions"]) >= int(rules["endpoint_sessions"]):
            raise ValueError("Prospective endpoint already reached")
        prior = pd.read_csv(prior_run / "observations.csv")
        prior_times = pd.to_datetime(prior.timestamp, utc=True)
        if not prior.empty and raw.timestamp_parsed.min() <= prior_times.max():
            raise ValueError("Collection batch is not a strict append")
        chain = pd.concat([prior, chain], ignore_index=True)
        prior_hash = digest(prior_run / "completion.json")
    chain = chain.sort_values(["timestamp", "contract"]).reset_index(drop=True)
    eligible = eligibility(chain, history)
    eligible_count = int(eligible.eligible.sum()) if len(eligible) else 0
    if eligible_count > int(rules["endpoint_sessions"]):
        raise ValueError("Collection batch extends beyond the frozen endpoint")
    status = {
        "protocol_completion_hash": digest(protocol_run / "completion.json"),
        "protocol_git_commit": commit,
        "protocol_git_committed_at": committed_at.isoformat(),
        "history_data_hash": digest(history),
        "prior_collection_hash": prior_hash,
        "observation_rows": len(chain),
        "eligible_sessions": eligible_count,
        "efficacy_calculated": False,
        "interim_pf_calculated": False,
        "endpoint_sessions": int(rules["endpoint_sessions"]),
    }
    return chain, eligible, status


def eligibility(observations: pd.DataFrame, history_path: Path) -> pd.DataFrame:
    future = observations.copy()
    future["timestamp_parsed"] = pd.to_datetime(future.timestamp, utc=True)
    future["day"] = future.timestamp_parsed.dt.tz_convert(
        "America/New_York"
    ).dt.strftime("%Y-%m-%d")
    raw_history = pd.read_csv(history_path)
    combined = pd.concat(
        [raw_history[BAR_COLUMNS], future[BAR_COLUMNS]], ignore_index=True
    )
    buffer = io.StringIO()
    combined.to_csv(buffer, index=False)
    selected, _ = audit_select(load(io.StringIO(buffer.getvalue()), "end"))
    future_days = set(future.day)
    selected_map = {
        session.day: session.contract
        for session in selected
        if session.day in future_days
    }
    rows = []
    for day in sorted(future_days):
        contract = selected_map.get(day)
        if contract is None:
            rows.append(
                {
                    "day": day,
                    "contract": None,
                    "eligible": False,
                    "exclusion": "causal_contract_or_session_unavailable",
                }
            )
            continue
        observed = future[(future.day == day) & (future.contract == contract)]
        complete = len(observed) == 390
        timely = complete and observed.timely.astype(str).str.lower().eq("true").all()
        rows.append(
            {
                "day": day,
                "contract": contract,
                "eligible": bool(timely),
                "exclusion": None if timely else "incomplete_or_late_selected_contract",
            }
        )
    return pd.DataFrame(rows, columns=["day", "contract", "eligible", "exclusion"])


def prepare_future(
    observations: pd.DataFrame,
    eligible: pd.DataFrame,
    history_path: Path,
    endpoint_sessions: int,
):
    raw_history = pd.read_csv(history_path)
    combined = pd.concat(
        [raw_history[BAR_COLUMNS], observations[BAR_COLUMNS]], ignore_index=True
    )
    buffer = io.StringIO()
    combined.to_csv(buffer, index=False)
    sessions, _ = audit_select(load(io.StringIO(buffer.getvalue()), "end"))
    moves = [
        np.abs(s.bars.close.to_numpy(float) / float(s.bars.open.iloc[0]) - 1)
        for s in sessions
    ]
    allowed = set(
        eligible.loc[eligible.eligible, "day"].astype(str).head(endpoint_sessions)
    )
    prepared = []
    for index, session in enumerate(sessions):
        if index >= 14 and session.day in allowed:
            sigma = np.mean(moves[index - 14 : index], axis=0)
            prepared.append((session, sigma, features(session, sigma)))
    if len(prepared) != min(endpoint_sessions, len(allowed)):
        raise ValueError("Prospective eligible session preparation mismatch")
    return prepared


def evaluate(protocol_run: Path, collection_run: Path, as_of: datetime | None = None):
    protocol_manifest = verify_run(protocol_run)
    collection_manifest = verify_run(collection_run)
    if protocol_manifest["command"] != "freeze":
        raise ValueError("Evaluation requires a freeze run")
    if collection_manifest["command"] != "collect":
        raise ValueError("Evaluation requires a collection run")
    _require_frozen_source(protocol_manifest)
    protocol = json.loads((protocol_run / "protocol.json").read_text())
    rules = _protocol_rules(protocol)
    history = _history(protocol)
    status = json.loads((collection_run / "collection_status.json").read_text())
    if status["protocol_completion_hash"] != digest(protocol_run / "completion.json"):
        raise ValueError("Evaluation protocol lineage mismatch")
    if status["history_data_hash"] != digest(history):
        raise ValueError("Evaluation historical-data lineage mismatch")
    observations = pd.read_csv(collection_run / "observations.csv")
    eligible = pd.read_csv(collection_run / "eligibility.csv")
    eligible_days = eligible.loc[eligible.eligible, "day"].astype(str).tolist()
    endpoint = int(rules["endpoint_sessions"])
    count = len(eligible_days)
    if count > endpoint:
        raise ValueError("Collection exceeds the frozen prospective endpoint")
    now = pd.Timestamp(as_of or datetime.now(timezone.utc))
    deadline = pd.Timestamp(protocol["deadline"])
    endpoint_timely = False
    if count >= endpoint:
        endpoint_day = eligible_days[endpoint - 1]
        endpoint_close = pd.to_datetime(
            observations.loc[
                pd.to_datetime(observations.timestamp, utc=True)
                .dt.tz_convert("America/New_York")
                .dt.strftime("%Y-%m-%d")
                .eq(endpoint_day),
                "timestamp",
            ],
            utc=True,
        ).max()
        endpoint_timely = bool(endpoint_close <= deadline)
    if count < endpoint or not endpoint_timely:
        verdict = (
            "INCONCLUSIVE" if now >= deadline or count >= endpoint else "COLLECTING"
        )
        return {
            "verdict": verdict,
            "eligible_sessions": count,
            "efficacy_calculated": False,
        }, None
    prepared = prepare_future(observations, eligible, history, endpoint)
    baseline_daily, _, _, baseline_trades = simulate_sessions(prepared, rules=rules)
    candidate_daily, ledger, decisions, candidate_trades = simulate_sessions(
        prepared, threshold=float(protocol["selected_threshold"]), rules=rules
    )
    baseline_stats = performance(baseline_trades, baseline_daily)
    candidate_stats = performance(candidate_trades, candidate_daily)
    delta = candidate_daily.net.to_numpy(float) - baseline_daily.net.to_numpy(float)
    blocks = [int(value) for value in rules["bootstrap_blocks"]]
    cis = {
        str(block): one_sided_ci(
            delta,
            block,
            int(rules["seed"]) + block,
            draws=int(rules["bootstrap_draws"]),
            alpha=float(rules["alpha"]),
        )
        for block in blocks
    }
    gates = {
        "pf": candidate_stats["pf_infinite"]
        or (candidate_stats["pf"] or 0) >= float(rules["target_pf"]),
        "net_retention": candidate_stats["net"]
        >= float(rules["profit_retention"]) * baseline_stats["net"],
        "paired_lower95": cis[str(blocks[0])]["lower95"] > 0,
        "sensitivity_consistent": all(
            cis[str(block)]["lower95"] > 0 for block in blocks[1:]
        ),
    }
    verdict = "SUPPORT" if all(gates.values()) else "FAIL"
    result = {
        "verdict": verdict,
        "eligible_sessions": count,
        "efficacy_calculated": True,
        "baseline": baseline_stats,
        "candidate": candidate_stats,
        "paired_ci": cis,
        "gates": gates,
    }
    return result, (
        baseline_daily,
        candidate_daily,
        ledger,
        decisions,
        candidate_trades,
    )
