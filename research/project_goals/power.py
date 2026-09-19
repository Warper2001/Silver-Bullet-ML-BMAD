"""Exposed calibration and strictly gated, paired-session GAP evaluation."""

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
import numpy as np
from .common import digest, input_path, read_csv, timestamp

REQUIRED = (
    "session",
    "mim_net_unit",
    "yank_net_unit",
    "gap_net_unit",
    "mim_exposure_unit",
    "yank_exposure_unit",
    "gap_exposure_unit",
    "mim_turnover_unit",
    "yank_turnover_unit",
    "gap_turnover_unit",
)


def paired(rows):
    if not rows:
        raise ValueError("no paired corrected sessions")
    dates = []
    matrix = []
    for row in rows:
        if any(row.get(k) in ("", None) for k in REQUIRED):
            raise ValueError("missing paired session/cost/exposure fields")
        if (
            str(row.get("corrected_contract")).lower() != "true"
            or str(row.get("complete_costs")).lower() != "true"
        ):
            raise ValueError("corrected contracts and complete costs must be attested")
        day = datetime.strptime(row["session"], "%Y-%m-%d").date()
        dates.append(day)
        values = [float(row[k]) for k in REQUIRED[1:]]
        if not all(math.isfinite(v) for v in values) or any(v < 0 for v in values[3:]):
            raise ValueError("invalid numeric evidence")
        matrix.append(values)
    if dates != sorted(set(dates)):
        raise ValueError("sessions must be sorted and unique")
    return dates, np.array(matrix)


DECISION_RULE = (
    "positive_three_endpoints_all_costs_blocks_bonferroni_paired_bootstrap_v1"
)


def calibration_design(blocks, costs, alpha, draws, seed):
    return dict(
        blocks=list(blocks),
        cost_stress=list(costs),
        alpha=alpha,
        draws=draws,
        seed=seed,
        allocations={"MIM": 1, "YANK": 2, "GAP": 1},
        decision_rule=DECISION_RULE,
    )


def endpoint_statistics(values, cost, indices):
    """Same three endpoints for power design and confirmatory paired resampling."""
    net = values[:, :3] - values[:, 6:9] * cost
    base = net[:, 0] + 2 * net[:, 1]
    extra = base + net[:, 2]
    base_exp = values[:, 3] + 2 * values[:, 4]
    extra_exp = base_exp + values[:, 5]
    if base_exp.sum() <= 0 or extra_exp.sum() <= 0:
        raise ValueError("zero aggregate exposure")

    def sharpe(arr):
        sd = arr.std(axis=-1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return math.sqrt(252) * arr.mean(axis=-1) / sd

    sampled_base = base[indices]
    sampled_extra = extra[indices]
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = base_exp[indices].sum(axis=-1) / extra_exp[indices].sum(axis=-1)
    stats = np.stack(
        (
            net[indices, 2].mean(axis=-1),
            sharpe(sampled_extra) - sharpe(sampled_base),
            sharpe(sampled_extra * scale[..., None]) - sharpe(sampled_base),
        ),
        axis=-1,
    )
    if not np.isfinite(stats).all():
        raise ValueError("undefined endpoint in paired bootstrap")
    return stats


def bootstrap_statistics(values, cost, block, draws, rng):
    n = len(values)
    starts = rng.integers(0, n, size=(draws, math.ceil(n / block)))
    indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    return endpoint_statistics(values, cost, indices)


def sweep(
    path,
    effects=(5.0, 10.0, 20.0, 40.0),
    horizons=(60, 120, 240, 480),
    blocks=(5, 10, 20),
    alpha=0.05,
    target_power=0.8,
    cost_stress=(0.0, 1.0),
    draws=400,
    seed=7,
):
    """Exposed design approximation for the identical simultaneous decision family.

    Joint power uses a conservative union bound over every endpoint, cost and block.
    Bootstrap endpoint SE is projected with sqrt(N); this is explicitly approximate.
    """
    design = calibration_design(blocks, cost_stress, alpha, draws, seed)
    validate_design(design)
    if not 0 < target_power < 1:
        raise ValueError("invalid target power")
    artifact = dict(
        status="INSUFFICIENT_DATA",
        confirmatory=False,
        sweep=[],
        portfolio_sweep=[],
        design=design,
        target_power=target_power,
        input_sha256=None,
        horizon_formula="ceil(required_eligible_sessions / eligible_session_fraction); calendar weeks ~= /5",
        missing="paired corrected MIM/YANK/GAP sessions with exposure and costs",
    )
    if path is None or not input_path(path).exists():
        return artifact
    before = digest(path)
    dates, values = paired(read_csv(path))
    if before != digest(path):
        raise ValueError("calibration changed during read")
    artifact["input_sha256"] = before
    rows = portfolio_power_sweep(
        values, effects, horizons, blocks, alpha, target_power, draws, cost_stress, seed
    )
    artifact["portfolio_sweep"] = rows
    artifact["sweep"] = [
        dict(row, required_sessions=row["mean_required_sessions"]) for row in rows
    ]
    artifact.update(
        status="CALIBRATION_ONLY" if rows else "UNDERPOWERED",
        calibration_sessions=len(dates),
        first_session=str(dates[0]),
        last_session=str(dates[-1]),
        alpha=alpha,
        limitation="Joint simultaneous endpoint/cost/block power is a conservative normal approximation using paired-bootstrap SE; not observed efficacy.",
    )
    return artifact


def git_read(repo, args):
    """Project-mandated preflight before every substantive git read."""
    subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    for span in ("origin/main..HEAD", "HEAD..origin/main"):
        subprocess.run(
            ["git", "rev-list", "--count", span],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def committed_registration(path, repo):
    path, repo = input_path(path), input_path(repo)
    relative = str(path.relative_to(repo))
    content = git_read(repo, ["show", "HEAD:" + relative])
    if content != path.read_text().strip():
        raise ValueError("registration is missing, uncommitted, or modified")
    commit = git_read(repo, ["log", "-1", "--format=%H", "--", relative])
    commit_date = git_read(repo, ["show", "-s", "--format=%cI", commit])
    return json.loads(content), commit, timestamp(commit_date)


def evaluate(data, registration, power_path, calibration, repo):
    """No efficacy statistics until all identity, freshness and power gates pass."""
    denied = lambda reason: dict(
        status="INSUFFICIENT_DATA", confirmatory=False, reason=reason
    )
    try:
        reg, commit, committed_at = committed_registration(registration, repo)
        if reg.get("status") != "registered":
            return denied("registration not approved")
        validate_registration(reg)
        power = json.loads(input_path(power_path).read_text())
        if reg.get("power_sha256") != digest(power_path) or power.get(
            "input_sha256"
        ) != digest(calibration):
            return denied("power/calibration inputs changed")
        if reg.get("code_sha256") != digest(__file__):
            return denied("evaluation code changed after registration")
        if reg.get("power_status") != "ADEQUATE_BOTH_ENDPOINTS":
            return dict(
                status="UNDERPOWERED",
                confirmatory=False,
                reason="mean and portfolio-risk endpoint power both required",
            )
        expected_design = calibration_design(
            reg["blocks"], reg["cost_stress"], reg["alpha"], reg["draws"], reg["seed"]
        )
        if power.get("design") != expected_design:
            return denied("registration differs from calibrated simultaneous design")
        risk_rows = [
            r
            for r in power.get("portfolio_sweep", [])
            if r["effect_usd"] == reg["effect_usd"]
            and r["sessions"] == reg["required_sessions"]
        ]
        expected = {
            (cost, block) for cost in reg["cost_stress"] for block in reg["blocks"]
        }
        if (
            len(risk_rows) != len(expected)
            or {(r["cost"], r["block"]) for r in risk_rows} != expected
            or any(
                r["required_sessions"] is None
                or reg["required_sessions"] < r["required_sessions"]
                for r in risk_rows
            )
        ):
            return dict(
                status="UNDERPOWERED",
                confirmatory=False,
                reason="selected horizon lacks joint endpoint power for every calibrated cost/block",
            )
        calibration_commit = reg.get("calibration_commit", "")
        if len(calibration_commit) != 40 or any(
            c not in "0123456789abcdef" for c in calibration_commit
        ):
            return denied("full committed calibration artifact revision required")
        power_relative = str(input_path(power_path).relative_to(input_path(repo)))
        if (
            git_read(repo, ["show", calibration_commit + ":" + power_relative])
            != input_path(power_path).read_text().strip()
        ):
            return denied("power artifact differs from committed calibration")
        git_read(repo, ["merge-base", "--is-ancestor", calibration_commit, commit])
        if (
            timestamp(
                git_read(repo, ["show", "-s", "--format=%cI", calibration_commit])
            )
            >= committed_at
        ):
            return denied("calibration must be committed before registration")
        before = digest(data)
        dates, values = paired(read_csv(data))
        if before != digest(data):
            return denied("evaluation input changed during read")
        cutoff = max(
            timestamp(reg["freshness_after"]),
            timestamp(reg["registered_at"]),
            committed_at,
        ).date()
        if any(d > datetime.now(timezone.utc).date() for d in dates):
            return denied("future-dated prospective sessions")
        if any(d <= cutoff for d in dates):
            return denied("evaluation includes exposed or pre-registration sessions")
        if len(dates) < reg["required_sessions"]:
            return dict(
                status="UNDERPOWERED",
                confirmatory=False,
                fresh_sessions=len(dates),
                required_sessions=reg["required_sessions"],
            )
        if len(dates) != reg["required_sessions"]:
            return denied(
                "evaluate exactly the registered session count; no post-hoc horizon choice"
            )
    except (
        ValueError,
        KeyError,
        FileNotFoundError,
        subprocess.CalledProcessError,
        TypeError,
    ) as exc:
        return denied(str(exc))
    rng = np.random.default_rng(reg["seed"])
    results = []
    tail = reg["alpha"] / (3 * len(reg["cost_stress"]) * len(reg["blocks"]))
    try:
        for cost in reg["cost_stress"]:
            for block in reg["blocks"]:
                if len(values) < 2 * block:
                    return dict(
                        status="UNDERPOWERED",
                        confirmatory=False,
                        reason="too few independent paired blocks",
                    )
                stats = bootstrap_statistics(values, cost, block, reg["draws"], rng)
                lower = np.quantile(stats, tail, axis=0)
                observed = endpoint_statistics(
                    values, cost, np.arange(len(values))[None, :]
                )[0]
                results.append(
                    dict(
                        cost=cost,
                        block=block,
                        standalone_net_expectancy=float(observed[0]),
                        standalone_lower=float(lower[0]),
                        delta_sharpe_lower=float(lower[1]),
                        same_exposure_delta_sharpe_lower=float(lower[2]),
                        pass_gate=bool(np.all(lower > 0)),
                    )
                )
    except ValueError as exc:
        return denied(str(exc))
    return dict(
        status="PASS" if all(r["pass_gate"] for r in results) else "FAIL",
        confirmatory=True,
        registration_commit=commit,
        input_sha256=before,
        results=results,
        baseline="MIM1/YANK2",
        candidate="MIM1/YANK2/GAP1 and same gross exposure",
        deployment_authorized=False,
    )


def standalone_sweep(
    path,
    effects=(5.0, 10.0, 20.0, 40.0),
    horizons=(30, 60, 120, 240, 480),
    blocks=(1, 3, 5),
):
    """Exposed trade-level calibration, cannot establish paired-session portfolio power."""
    rows = read_csv(path)
    values = np.array([float(r["pnl_usd"]) for r in rows])
    if len(values) < 2 or not np.isfinite(values).all():
        return dict(status="UNDERPOWERED", sweep=[])
    z = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.8)
    output = []
    for block in blocks:
        n = len(values) // block
        if n < 2:
            continue
        variance = float(
            np.var(values[: n * block].reshape(n, block).mean(axis=1), ddof=1) * block
        )
        for effect in effects:
            for horizon in horizons:
                required = math.ceil(z * z * variance / effect**2)
                output.append(
                    dict(
                        block_trades=block,
                        effect_usd=effect,
                        future_trades=horizon,
                        required_future_trades=required,
                        adequate=horizon >= required,
                        minimum_detectable_usd=z * math.sqrt(variance / horizon),
                    )
                )
    return dict(
        status="EXPOSED_CALIBRATION_ONLY",
        input_sha256=digest(path),
        trades=len(values),
        cost_provenance="Uses supplied pnl_usd; fee/slippage decomposition unresolved, not treated as verified net.",
        sweep=output,
        joint_portfolio_status="INSUFFICIENT_DATA",
        collection_horizon="required_future_trades / prospective_trade_rate_per_eligible_session; rate must be observed",
        confirmatory=False,
    )


def portfolio_power_sweep(
    matrix,
    effects,
    horizons,
    blocks,
    alpha=0.05,
    target_power=0.8,
    draws=400,
    cost_stress=(0.0, 1.0),
    seed=7,
):
    rng = np.random.default_rng(seed)
    output = []
    n = len(matrix)
    family = 3 * len(blocks) * len(cost_stress)
    tail = alpha / family
    z = NormalDist().inv_cdf(1 - tail) + NormalDist().inv_cdf(
        1 - (1 - target_power) / family
    )
    for effect in effects:
        if not math.isfinite(effect) or effect <= 0:
            raise ValueError("positive finite detectable effects required")
        alternative = matrix.copy()
        alternative[:, 2] = matrix[:, 2] - matrix[:, 2].mean() + effect
        for cost in cost_stress:
            for block in blocks:
                if n < 2 * block:
                    continue
                try:
                    observed = endpoint_statistics(
                        alternative, cost, np.arange(n)[None, :]
                    )[0]
                    stats = bootstrap_statistics(alternative, cost, block, draws, rng)
                    se = stats.std(axis=0, ddof=1)
                    required_endpoints = [
                        (
                            max(2 * block, math.ceil(n * (z * s / value) ** 2))
                            if value > 0
                            else None
                        )
                        for value, s in zip(observed, se)
                    ]
                except ValueError:
                    observed = np.full(3, float("nan"))
                    required_endpoints = [None] * 3
                required = (
                    max(required_endpoints)
                    if all(v is not None for v in required_endpoints)
                    else None
                )
                for horizon in horizons:
                    if type(horizon) is not int or horizon < 2:
                        raise ValueError("invalid horizon")
                    output.append(
                        dict(
                            effect_usd=effect,
                            cost=cost,
                            block=block,
                            sessions=horizon,
                            required_sessions=required,
                            mean_required_sessions=required_endpoints[0],
                            endpoint_required_sessions=required_endpoints,
                            tail_probability=tail,
                            adequate=required is not None and horizon >= required,
                        )
                    )
    return output


def validate_design(reg):
    for name in ("blocks", "cost_stress"):
        if not isinstance(reg.get(name), list) or not reg[name]:
            raise ValueError("nonempty " + name + " required")
        if len(set(reg[name])) != len(reg[name]):
            raise ValueError("duplicate design choices")
    if any(type(b) is not int or b <= 0 for b in reg["blocks"]):
        raise ValueError("positive integer blocks required")
    if (
        any(
            type(c) not in (int, float) or not math.isfinite(c) or c < 0
            for c in reg["cost_stress"]
        )
        or 0 not in reg["cost_stress"]
        or not any(c > 0 for c in reg["cost_stress"])
    ):
        raise ValueError("baseline and positive finite cost stress required")
    if type(reg.get("seed")) is not int or reg["seed"] < 0:
        raise ValueError("nonnegative integer seed required")
    if type(reg.get("draws")) is not int or reg["draws"] < 2:
        raise ValueError("positive draw count required")
    if not 0 < float(reg["alpha"]) < 1 or not math.isfinite(float(reg["alpha"])):
        raise ValueError("invalid alpha")
    tail = reg["alpha"] / (3 * len(reg["cost_stress"]) * len(reg["blocks"]))
    if reg["draws"] < math.ceil(1 / tail):
        raise ValueError("bootstrap draws do not resolve simultaneous tail")
    if reg.get("allocations") != {"MIM": 1, "YANK": 2, "GAP": 1}:
        raise ValueError("frozen MIM1/YANK2 plus GAP1 required")


def validate_registration(reg):
    validate_design(reg)
    if type(reg.get("required_sessions")) is not int or reg["required_sessions"] < 2:
        raise ValueError("positive session count required")
    if reg.get("source_hashes") != source_hashes():
        raise ValueError("computational sources changed")


def source_hashes():
    root = Path(__file__).parent
    return {name: digest(root / name) for name in ("power.py", "common.py")}
