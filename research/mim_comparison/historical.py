import json
import numpy as np
import pandas as pd
from .data import load, audit_select
from .engine import ARMS, simulate, sizing_quantity
from .references import author_sigma, author_vectorized, log_reconciliation
from .statistics import summary, minimum_detectable
from .artifacts import write_json, ROOT, verify_run, paired_daily
from pathlib import Path
import hashlib, io


def run(data, labels, path, audit_only=False):
    verify_run(path, data)
    consumed = Path(data).read_bytes()
    expected = json.loads((path / "manifest.json").read_text())["inputs"][
        str(Path(data).resolve())
    ]
    if hashlib.sha256(consumed).hexdigest() != expected:
        raise ValueError("Consumed input does not match frozen hash")
    sessions, exclusions = audit_select(load(io.BytesIO(consumed), labels))
    del consumed
    write_json(
        path / "audit.json",
        {
            "selected_sessions": len(sessions),
            "exclusions": exclusions,
            "provenance": "CSV explicit MNQ quarterly contracts; expiry inferred, not exchange metadata",
            "timestamp_declaration": labels,
            "historical_only": True,
            "missing_sessions": "Absent weekdays explicitly excluded as missing_or_exchange_closed_unverified; no verified exchange calendar",
            "first": sessions[0].day if sessions else None,
            "last": sessions[-1].day if sessions else None,
        },
    )
    if audit_only:
        verify_run(path, data)
        return
    moves = [
        np.abs(s.bars.close.to_numpy() / float(s.bars.open.iloc[0]) - 1)
        for s in sessions
    ]
    author = author_sigma(moves)
    daily = []
    ledger = []
    decisions = []
    sizing = []
    reference = []
    returns = [float(s.bars.close.iloc[-1]) / s.previous_close - 1 for s in sessions]
    if returns:
        returns[0] = float(
            "nan"
        )  # Author initializes first daily return missing; loop starts at day1.
    equities = {"dynamic": 100000.0, "constant_notional": 100000.0}
    for d, s in enumerate(sessions):
        if d < 14:
            exclusions.append(
                {
                    "day": s.day,
                    "contract": s.contract,
                    "eligible": False,
                    "exclusion": "14_complete_prior_selected_sessions_required",
                }
            )
            continue
        sigma = np.mean(moves[d - 14 : d], axis=0)
        ref = author_vectorized(s.bars, s.previous_close, author[d])
        reference.append(
            {
                "day": s.day,
                "contract": s.contract,
                "author_vectorized_gross_points": ref["gross_points"],
                "diagnostic_only": True,
            }
        )
        for arm in ARMS:
            chosen = author[d] if arm.name in ("B", "D") else sigma
            for delay in (2, 1):
                row, events, signals = simulate(s, chosen, arm, delay, 2.24)
                for cost in (2.24, 3.24, 6.24):
                    daily.append(
                        dict(
                            row,
                            cost=cost,
                            costs=row["turnover"] * cost / 2,
                            net=row["gross"] - row["turnover"] * cost / 2,
                        )
                    )
                ledger.extend(dict(e, delay=delay, cost_scenario=2.24) for e in events)
                decisions.extend(dict(e, delay=delay) for e in signals)
        for mode in equities:
            equity = equities[mode]
            # Both author comparators compound their own equity; constant notional means 1x leverage.
            q, vol, lev = sizing_quantity(
                equity, float(s.bars.open.iloc[0]), returns, d, mode == "dynamic"
            )
            row, _, _ = simulate(s, author[d], ARMS[3], 2, 2.24, q)
            equities[mode] += row["net"]
            sizing.append(
                dict(
                    row,
                    mode=mode,
                    equity_before=equity,
                    equity_after=equities[mode],
                    return_on_equity=row["net"] / equity if equity > 0 else 0,
                    lagged_volatility=vol,
                    leverage=lev,
                    realized_notional_leverage=(
                        q * float(s.bars.open.iloc[0]) * 2 / equity if equity > 0 else 0
                    ),
                )
            )
    pd.DataFrame(daily).to_csv(path / "daily.csv", index=False)
    paired_daily(pd.DataFrame(daily)).to_csv(path / "paired-daily.csv", index=False)
    pd.DataFrame(ledger).to_csv(path / "ledger.csv", index=False)
    pd.DataFrame(decisions).to_csv(path / "decisions.csv", index=False)
    pd.DataFrame(sizing).to_csv(path / "sizing.csv", index=False)
    pd.DataFrame(reference).to_csv(path / "author-reference.csv", index=False)
    write_json(path / "eligibility.json", exclusions)
    frame = pd.DataFrame(daily)
    report = {
        "historical_diagnosis_only": True,
        "promotion_allowed": False,
        "limitations": [
            "All history exposed",
            "No contract identity in deployed logs",
            "Reference logs rounded; exact operational parity unavailable",
            "No verified contemporaneous historical receipts",
            "Sizing4x cap applies before integer rounding; realized notional leverage can exceed4x",
            "Expiry inferred; exchange holiday calendar unverified and absent weekdays excluded",
        ],
    }
    if len(frame):
        report["arms"] = {
            f"{arm}/delay{delay}/cost{cost}": summary(g)
            for (arm, delay, cost), g in frame.groupby(["arm", "delay", "cost"])
        }
        primary = frame[(frame.delay == 2) & (frame.cost == 2.24)].pivot(
            index="day", columns="arm", values="net"
        )
        diff = primary.B - primary.A
        report["scenario_contrasts"] = {}
        for (delay, cost), group in frame.groupby(["delay", "cost"]):
            paired = group.pivot(index="day", columns="arm", values="net")
            difference = paired.B - paired.A
            singles = {
                name: float((paired[name] - paired.A).mean())
                for name in ("gap", "confirmation", "exit")
            }
            report["scenario_contrasts"][f"delay{delay}/cost{cost}"] = {
                "paired_B_minus_A": {
                    "mean": float(difference.mean()),
                    "sessions": len(difference),
                },
                "mechanisms": dict(
                    singles,
                    interaction_combined_minus_singles=float(
                        difference.mean() - sum(singles.values())
                    ),
                ),
            }
        report["paired_B_minus_A"] = {"mean": float(diff.mean()), "sessions": len(diff)}
        report["mde"] = minimum_detectable(diff)
        report["mechanisms"] = {
            name: float((primary[name] - primary.A).mean())
            for name in ("gap", "confirmation", "exit")
        }
        report["mechanisms"]["interaction_combined_minus_singles"] = float(
            diff.mean() - sum(report["mechanisms"].values())
        )
        report["mechanisms"][
            "caveat"
        ] = "B also preserves author first-day rolling min13 semantics; first eligible session differs in sigma depth."
    else:
        report["mde"] = {"unavailable": "no_eligible_pairs"}
    if sizing:
        size = pd.DataFrame(sizing)
        report["sizing"] = {
            mode: dict(
                summary(g),
                realized_annualized_volatility=(
                    float(g.return_on_equity.std(ddof=1) * np.sqrt(252))
                    if len(g) > 1
                    else None
                ),
                total_return=float(g.equity_after.iloc[-1] / 100000 - 1),
                cannot_promote=True,
            )
            for mode, g in size.groupby("mode")
        }
    ops = path / "operational-decisions-snapshot.csv"
    if ops.exists():
        # A frozen byte snapshot, never read operational input twice.
        snapshot = ops.read_bytes()
        write_json(
            path / "operational-reconciliation.json",
            log_reconciliation(pd.read_csv(io.BytesIO(snapshot))),
        )
    verify_run(path, data)
    write_json(path / "report.json", report)
    (path / "report.md").write_text(
        "# Historical mechanism diagnosis\n\nAll history is exposed. No result authorizes promotion or deployment.\n\n"
        + json.dumps(report, indent=2)
        + "\n"
    )
