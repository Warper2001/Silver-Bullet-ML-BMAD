"""Verify the documentary packet and synthetic arithmetic offline."""

import hashlib
import json
import math
from pathlib import Path
from statistics import covariance, variance

from calculate import (
    ar1_mean_variance,
    calculate,
    paired_variance,
    required_n,
    variance_upper,
)

ROOT = Path(__file__).resolve().parent


def check(condition: bool, description: str) -> None:
    if not condition:
        raise AssertionError(description)


def close(actual: float, expected: float) -> None:
    check(
        math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-11),
        f"{actual} != {expected}",
    )


def rejects(function, *args) -> None:
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError(
        f"accepted invalid assumptions: {function.__name__} {args}"
    )


def main() -> None:
    # Independent sample formula, including perfect correlations.
    k = [1.0, 3.0, -2.0, 7.0, 4.0]
    for m in ([2.0, -1.0, 5.0, 3.0, 6.0], k, [-x for x in k]):
        close(
            paired_variance(variance(k), variance(m), covariance(k, m)),
            variance([x - y for x, y in zip(k, m)]),
        )
    close(paired_variance(10000, 6400, 4000), 8400)
    close(paired_variance(10000, 6400, -8000), 32400)
    rejects(paired_variance, 1, 1, 1.01)
    rejects(paired_variance, -1, 1, 0)
    rejects(paired_variance, math.nan, 1, 0)

    # Direct covariance sum, independent of the lag-sum implementation.
    for n in (1, 5, 20):
        for rho in (-0.5, 0, 0.5, 0.9):
            direct = (
                sum(rho ** abs(i - j) for i in range(n) for j in range(n))
                / n**2
            )
            close(ar1_mean_variance(n, rho), direct)
    check(
        math.sqrt(20 * ar1_mean_variance(20, 0.9)) > 2,
        "hypothetical 2x SE is not a universal bound",
    )
    rejects(ar1_mean_variance, 20, 1)
    rejects(ar1_mean_variance, 0, 0.5)

    # chi-square(df=2) has exact quantile -2 log(1-p).
    close(variance_upper(3, 7, 0.05), 14 / (-2 * math.log(0.95)))
    check(
        variance_upper(20, 8400, 0.025) > variance_upper(20, 8400, 0.05),
        "more coverage must widen variance bound",
    )
    rejects(variance_upper, 1, 1, 0.05)
    rejects(variance_upper, 20, 0, 0.05)
    rejects(variance_upper, 20, 1, 1)
    rejects(required_n, 0, 8400, 1)
    rejects(required_n, -20, 8400, 1)
    rejects(required_n, 20, 8400, 0.5)

    result = calculate()
    check(
        result == json.loads((ROOT / "calculation-results.json").read_text()),
        "stored calculation output differs",
    )
    close(result["zsum"], 3.241515550084654)
    pair = result["synthetic_pair"]
    close(pair["sd_ucl"] ** 2, pair["variance_ucl"])
    check(
        pair["iid_ucl_required_n"] > pair["iid_plugin_required_n"],
        "nuisance uncertainty must increase synthetic requirement",
    )
    close(
        result["uncertainty_accounting_example"][
            "unconditional_assurance_lower_bound"
        ],
        0.76,
    )
    close(result["rare_tail_counterexample"]["true_variance"], 999)
    check(
        result["rare_tail_counterexample"][
            "probability_zero_observed_variance_from_all_zero_pilot"
        ]
        > 0.98,
        "rare tail counterexample",
    )
    perm = result["permutation_counterexamples"]
    check(
        perm["K_equals_M_actual_variance"] == 0
        and perm["K_equals_negative_M_actual_variance"] == 4
        and perm["independent_mismatch_variance"] == 2,
        "mismatch need not be conservative",
    )
    permutation = perm["nonidentity_pairing_with_fixed_points"]
    check(
        permutation != list(range(4))
        and any(i == j for i, j in enumerate(permutation)),
        "whole-identity rejection does not exclude all aligned indices",
    )

    ledger = json.loads((ROOT / "inputs/06-scenarios.json").read_text())
    expected = {
        (r["scenario"], r["months"]): r["conditional_evaluation_sessions"]
        for r in ledger["horizons"]
    }
    check(len(expected) == 9, "nine inherited scenarios")
    for row in result["horizons"]:
        check(
            row["conditional_evaluation_sessions"]
            == expected[(row["scenario"], row["months"])],
            "inherited capacity changed",
        )

    decision = json.loads((ROOT / "decision.json").read_text())
    check(decision["decision"] == "PARK_PENDING_EVIDENCE", "decision changed")
    for obj in (decision, result):
        check(
            obj["strategy_test_permitted"] is False
            and obj["trading_authorized"] is False,
            "permission flags must be false",
        )
    check(
        all(not r["admission_gate_met"] for r in decision["routes"]),
        "route unexpectedly admitted",
    )

    register = json.loads((ROOT / "input-register.json").read_text())
    for row in register["inputs"]:
        check(
            hashlib.sha256((ROOT / row["snapshot"]).read_bytes()).hexdigest()
            == row["sha256"],
            f"input identity: {row['snapshot']}",
        )
    for row in json.loads((ROOT / "sources/retrieval.json").read_text()):
        if "file" in row:
            check(
                hashlib.sha256(
                    (ROOT / "sources" / row["file"]).read_bytes()
                ).hexdigest()
                == row["sha256"],
                f"source identity: {row['id']}",
            )
    manifest = json.loads((ROOT / "COMPLETE.json").read_text())
    actual_files = {
        str(p.relative_to(ROOT))
        for p in ROOT.rglob("*")
        if p.is_file()
        and p.name != "COMPLETE.json"
        and "__pycache__" not in p.parts
    }
    check(
        actual_files == set(manifest["sha256"]), "manifest membership mismatch"
    )
    for name, digest in manifest["sha256"].items():
        check(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest,
            f"hash mismatch: {name}",
        )
    check(
        manifest["strategy_test_permitted"] is False
        and manifest["trading_authorized"] is False,
        "manifest permission flags",
    )
    print(
        "PASS: synthetic arithmetic, counterexamples, nine horizon rows, "
        "permission flags, "
        f"source identities and {len(actual_files)} artifact hashes"
    )


if __name__ == "__main__":
    main()
