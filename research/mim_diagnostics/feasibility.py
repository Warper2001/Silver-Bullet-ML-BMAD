"""Read-only local provenance inventory; no data acquisition or efficacy analysis."""

from pathlib import Path
import sqlite3
import pandas as pd
from .artifacts import ORIGINAL, now, permitted, read_csv

CALENDARS = {
    "economic_calendar": (
        "data/macro/econ_calendar_2025_2026.csv",
        "date",
        "requires verification/acquisition",
        "Time is ET but source URL, original release schedule vintage, first publication timestamp and revision trail are absent. Official spot checks contradict three dates: May 1 2025 FOMC versus May 6–7; July 11 2025 CPI versus July 15; January 15 2026 CPI versus January 13.",
    ),
    "policy_shock_dates": (
        "data/macro/policy_shock_calendar_2025_2026.csv",
        "date",
        "unavailable",
        "Retrospective realized-move labels have no contemporaneous publication or decision-time evidence; usable only as retrospective documentation. Prior retrospective says 70 dates; actual file has 69.",
    ),
    "policy_shock_windows": (
        "data/macro/policy_shock_windows_2025_2026.csv",
        "start",
        "unavailable",
        "Retrospective windows have no event clock or first-known time; overlapping April 7 boundary. Unavailable as causal information inputs.",
    ),
    "forward_fomc": (
        "data/macro/fomc_calendar_forward.csv",
        "date",
        "requires verification/acquisition",
        "All 11 dates have current official schedule support; usable current schedule snapshot, but original availability, vintage and reschedule history require verification. Fed future dates remain tentative until the preceding meeting.",
    ),
}
DOCS = [
    "_bmad-output/innovation-strategy-2026-09-12.md",
    "_bmad-output/option_c_retrospective_20260703.md",
    "_bmad-output/option_b_gate0_verdict_20260703.md",
    "_bmad-output/preregistration_evfade_fomc_prospective.md",
    "research/mim_comparison/FEED_STATUS.md",
]
PROTOCOLS = [
    "research/mim_comparison/runs/20260910T214803-shadow-47055eba62/freeze-snapshot.json",
    "research/mim_comparison/runs/20260910T214803-shadow-47055eba62/protocol.json",
]
REQUIRED_CATEGORIES = set(CALENDARS) | {
    "nasdaq_options_positioning",
    "leveraged_etf_aum_leverage",
    "observed_ohlcv",
    "price_derived_flow_proxy",
    "mim_prospective",
    "fomc_prospective",
}
OFFICIAL = [
    dict(
        url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        checked="2026-09-12",
        support="Current FOMC dates; page updated August 19 2026. Does not establish historical first availability.",
    ),
    dict(
        url="https://www.bls.gov/schedule/2025/07_sched.htm",
        checked="2026-09-12",
        support="July 15 2025 CPI; schedule states Eastern Time and is mutable.",
    ),
    dict(
        url="https://www.bls.gov/schedule/2026/01_sched.htm",
        checked="2026-09-12",
        support="January 13 2026 CPI; current retrieval is not an archival vintage.",
    ),
]


def build_inventory():
    observed = now()
    rows = []
    evidence = []
    for category, (rel, date_col, status, missing) in CALENDARS.items():
        path = permitted(ORIGINAL / rel)
        if not path.is_file():
            raise ValueError("Missing required calendar evidence: " + rel)
        df = read_csv(permitted(path))
        if date_col not in df:
            raise ValueError("Invalid calendar date schema")
        dates = pd.to_datetime(df[date_col], errors="raise")
        evidence.append(path)
        rows.append(
            dict(
                category=category,
                status=status,
                path=rel,
                rows=len(df),
                columns=list(df),
                coverage_start=dates.min().date().isoformat(),
                coverage_end=(
                    pd.to_datetime(df["end"]).max() if "end" in df else dates.max()
                )
                .date()
                .isoformat(),
                exact_duplicates=int(df.duplicated().sum()),
                date_duplicates=int(df[date_col].duplicated().sum()),
                timezone=(
                    "America/New_York (ET) declared by time_et field"
                    if "time_et" in df
                    else "Unestablished event timezone and clock; ET is only a possible research-session interpretation"
                ),
                observation_type=(
                    "retrospective label"
                    if "policy_shock" in category
                    else "schedule snapshot"
                ),
                official_support=(
                    "No contemporaneous official event-time support for retrospective labels"
                    if "policy_shock" in category
                    else "Current URL spot checks in official_checks; historical availability unverified"
                ),
                missing_evidence=missing,
            )
        )
    rows.extend(
        [
            dict(
                category="nasdaq_options_positioning",
                status="unavailable",
                path=None,
                coverage_start=None,
                coverage_end=None,
                observation_type="no local observed dealer inventory archive",
                missing_evidence="Nasdaq-linked options open interest, contract terms, signed dealer inventory or defensible ownership/sign assumptions, publication/receipt timestamps, revisions, historical vintages and aggregation methodology. SPX is not Nasdaq inventory.",
            ),
            dict(
                category="leveraged_etf_aum_leverage",
                status="unavailable",
                path=None,
                coverage_start=None,
                coverage_end=None,
                observation_type="no local observed AUM/leverage archive",
                missing_evidence="Historical Nasdaq leveraged ETF shares/AUM, fund leverage objective and effective dates, prices/NAV, publication delays, first-available timestamps, creation/redemption flows and methodology. Future use requires acquisition.",
            ),
            dict(
                category="observed_ohlcv",
                status="usable",
                path="data/mim_x/mnq_1min_by_contract.csv",
                coverage_start="2020-12-18",
                coverage_end="2026-08-28",
                observation_type="observed contract OHLCV; frozen baseline uses only eligible source grid",
                missing_evidence="Does not directly observe closing demand, dealer positions, ETF rebalance orders or participant identity.",
            ),
            dict(
                category="price_derived_flow_proxy",
                status="requires verification/acquisition",
                path=None,
                coverage_start=None,
                coverage_end=None,
                observation_type="unbuilt proxy, not observed flow",
                missing_evidence="Specify estimator and assumptions prospectively; identify volume/price ambiguity, publication timing and independent validation. Price/volume delta fields do not establish dealer options inventory.",
            ),
        ]
    )
    for rel in DOCS + [
        "research/mim_comparison/runs/20260910T214803-shadow-47055eba62/freeze-snapshot.json",
        "research/mim_comparison/runs/20260910T214803-shadow-47055eba62/protocol.json",
    ]:
        p = permitted(ORIGINAL / rel)
        if not p.is_file():
            raise ValueError("Missing required protocol evidence: " + rel)
        evidence.append(p)
    journal = (
        ORIGINAL
        / "research/mim_comparison/runs/20260910T214803-shadow-47055eba62/collector/journal.sqlite"
    )
    mim = dict(
        category="mim_prospective",
        status="requires verification/acquisition",
        path=str(journal.relative_to(ORIGINAL)),
        coverage_start=None,
        coverage_end=None,
        observed_at=observed,
        missing_evidence="120 eligible forward sessions not accumulated; no efficacy analysis permitted.",
    )
    journal = permitted(journal)
    if journal.is_file():
        mim.update(observe_journal(journal))
        mim["collection_note"] = (
            "Read-only count/coverage queries; current counts do not measure strategy efficacy. Mutable journal is observed, not a reproducible historical outcome input."
        )
    else:
        mim["collection_note"] = (
            "Journal unavailable at observation; no eligibility inferred."
        )
    feed = permitted(
        ORIGINAL / "research/mim_comparison/runs/20260910-contract-feed/feed.csv"
    )
    mim["feed"] = {
        "path": str(feed.relative_to(ORIGINAL)),
        "exists": feed.exists(),
        "observed_at": observed,
    }
    if feed.exists():
        columns = list(pd.read_csv(permitted(feed), nrows=0))
        time_columns = [
            c for c in ("timestamp", "received_at", "observed_at") if c in columns
        ]
        if not time_columns:
            raise ValueError("Feed lacks coverage timestamp schema")
        content = pd.read_csv(permitted(feed), usecols=time_columns)
        mim["feed"].update(
            rows=len(content),
            coverage={
                c: {"first": str(content[c].min()), "last": str(content[c].max())}
                for c in time_columns
            },
        )
    set_eligibility_status(mim)
    rows.append(mim)
    event_path = permitted(ORIGINAL / "data/evfade_fomc/prospective_events.csv")
    fomc = dict(
        category="fomc_prospective",
        status="requires verification/acquisition",
        path=str(event_path.relative_to(ORIGINAL)),
        observed_at=observed,
        exists=event_path.exists(),
        coverage_start=None,
        coverage_end=None,
        missing_evidence="Prospective event observations; first scheduled event September 16 2026. Preserve K3/M30 sealed fade, only N=15/N=30 efficacy looks, stop December 31 2030.",
    )
    if event_path.exists():
        ef = read_csv(permitted(event_path))
        fomc["rows"] = len(ef)
        datecol = next((c for c in ("date", "event_date") if c in ef), None)
        if datecol and len(ef):
            fomc["coverage_start"], fomc["coverage_end"] = str(ef[datecol].min()), str(
                ef[datecol].max()
            )
    else:
        fomc["rows"] = 0
    log = permitted(ORIGINAL / "logs/evfade_fomc_prospective.log")
    if log.exists():
        from datetime import datetime, timezone

        fomc["log_last_modified"] = datetime.fromtimestamp(
            permitted(log).stat().st_mtime, timezone.utc
        ).isoformat()
    rows.append(fomc)
    scan = archive_scan()
    result = dict(
        archive_scan=scan,
        observed_at=observed,
        categories=rows,
        official_checks=OFFICIAL,
        local_inventory_scope="Read-only filename/header reconnaissance of data excluding sealed_holdout: 325 CSV and 19 parquet observed 2026-09-12. No Nasdaq inventory or ETF AUM/leverage archive identified; absence applies to inspected local evidence, not all vendors.",
        documents=DOCS,
        correction=dict(
            policy_throttle="H-C1 REJECTED; policy-shock throttle DEAD. Preserve Option C rejection.",
            impulse_following="Option B Gate 0 OOS FAIL; CLOSED, no resweeps. Preserve verdict.",
            fomc_fade="Separate EVFADE FOMC K3/M30 prospective fade remains sealed; it is not the failed impulse-following branch.",
            mim="Preserve frozen prospective A/B protocol; no efficacy analysis, filters or resets. The historical diagnostics do not promote a candidate.",
        ),
    )
    validate_inventory(result)
    return result, evidence


def validate_inventory(value):
    def text(v):
        return isinstance(v, str) and bool(v.strip())

    rows = value.get("categories", [])
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError("Invalid category rows")
    categories = [r.get("category") for r in rows]
    if set(categories) != REQUIRED_CATEGORIES or len(categories) != len(
        set(categories)
    ):
        raise ValueError("Incomplete or duplicate feasibility inventory")
    for r in rows:
        category = r["category"]
        required = {
            "path",
            "coverage_start",
            "coverage_end",
            "status",
            "missing_evidence",
        }
        if category in CALENDARS:
            required |= {
                "rows",
                "columns",
                "exact_duplicates",
                "date_duplicates",
                "timezone",
                "official_support",
                "observation_type",
            }
        elif category == "mim_prospective":
            required |= {"observed_at", "collection_note", "eligibility_status"}
        elif category == "fomc_prospective":
            required |= {"observed_at", "exists", "rows"}
        else:
            required |= {"observation_type"}
        if (
            not required <= set(r)
            or r["status"]
            not in {"usable", "requires verification/acquisition", "unavailable"}
            or not text(r["missing_evidence"])
        ):
            raise ValueError("Missing category evidence")
        for key in ("path", "coverage_start", "coverage_end"):
            if r[key] is not None and not text(r[key]):
                raise ValueError("Invalid provenance field")
        if category in CALENDARS:
            if r["path"] != CALENDARS[category][0] or not all(
                text(r[k])
                for k in (
                    "coverage_start",
                    "coverage_end",
                    "timezone",
                    "official_support",
                    "observation_type",
                )
            ):
                raise ValueError("Missing calendar provenance")
            if (
                not isinstance(r["columns"], list)
                or not r["columns"]
                or not all(text(c) for c in r["columns"])
            ):
                raise ValueError("Invalid calendar columns")
            if any(
                type(r[k]) is not int or r[k] < 0
                for k in ("rows", "exact_duplicates", "date_duplicates")
            ):
                raise ValueError("Invalid calendar counts")
        if category == "mim_prospective":
            if not text(r["observed_at"]) or not text(r["collection_note"]):
                raise ValueError("Missing journal observation")
            if "eligible_sessions" in r:
                if (
                    type(r["eligible_sessions"]) is not int
                    or r["eligible_sessions"] < 0
                ):
                    raise ValueError("Invalid eligibility count")
                for key in (
                    "table_counts",
                    "available_observations",
                    "collection_start",
                    "collection_end",
                ):
                    if key not in r:
                        raise ValueError("Missing journal snapshot evidence")
            probe = dict(r)
            set_eligibility_status(probe)
            if r["eligibility_status"] != probe["eligibility_status"]:
                raise ValueError("Incorrect eligibility status")
        if category == "fomc_prospective" and (
            type(r["exists"]) is not bool or type(r["rows"]) is not int or r["rows"] < 0
        ):
            raise ValueError("Invalid FOMC status")
    if not text(value.get("observed_at")):
        raise ValueError("Missing observation timestamp")
    checks = value.get("official_checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("Invalid official checks")
    for check in checks:
        if (
            not isinstance(check, dict)
            or set(check) != {"url", "checked", "support"}
            or not all(text(v) for v in check.values())
            or not check["url"].startswith("https://")
        ):
            raise ValueError("Invalid official support record")
        try:
            pd.Timestamp(check["checked"])
        except ValueError as exc:
            raise ValueError("Invalid official retrieval date") from exc
    correction = value.get("correction")
    if (
        not isinstance(correction, dict)
        or set(correction)
        != {"policy_throttle", "impulse_following", "fomc_fade", "mim"}
        or not all(text(v) for v in correction.values())
    ):
        raise ValueError("Invalid correction structure")
    if value.get("documents") != DOCS or not text(value.get("local_inventory_scope")):
        raise ValueError("Missing documentary provenance")
    scan = value.get("archive_scan")
    if (
        not isinstance(scan, dict)
        or not text(scan.get("observed_at"))
        or not text(scan.get("scope"))
        or not isinstance(scan.get("entries"), list)
    ):
        raise ValueError("Missing archive scan")
    paths = []
    for row in scan["entries"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "kind", "columns", "error"}
            or not text(row["path"])
            or row["kind"] not in ("csv", "parquet")
        ):
            raise ValueError("Invalid archive evidence")
        if (
            Path(row["path"]).is_absolute()
            or ".." in Path(row["path"]).parts
            or "sealed_holdout" in Path(row["path"]).parts
        ):
            raise ValueError("Unsafe archive evidence path")
        if (
            row["kind"] == "csv"
            and row["error"] is None
            and (
                not isinstance(row["columns"], list)
                or not all(text(c) for c in row["columns"])
            )
        ):
            raise ValueError("Invalid archive header evidence")
        paths.append(row["path"])
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate archive evidence")


def render_inventory(value):
    lines = [
        "# Data feasibility and collection status",
        "",
        f"Observation: {value['observed_at']}. Dates alone are insufficient causal provenance.",
        "",
    ]
    for r in value["categories"]:
        lines.extend(
            [
                f"## {r['category']} — {r['status']}",
                "",
                f"Evidence: `{r.get('path')}`. Coverage: {r.get('coverage_start')} through {r.get('coverage_end')}. Rows: {r.get('rows',r.get('table_counts','not applicable'))}.",
                "",
                r["missing_evidence"],
                "",
            ]
        )
        if r["category"] == "mim_prospective":
            lines.extend(
                [
                    f"Observed at {r['observed_at']}; eligible sessions: {r.get('eligible_sessions', 'unknown')}; eligibility status: {r['eligibility_status']}. Journal collection coverage: {r.get('collection_start')} through {r.get('collection_end')}. Available observations: {r.get('available_observations', 'unknown')}. Feed observation: {r.get('feed')}. Feed timestamps describe input freshness, not valid prospective sessions or efficacy.",
                    "",
                ]
            )
        if "exact_duplicates" in r:
            lines.extend(
                [
                    f"Duplicates: {r['exact_duplicates']} exact; {r['date_duplicates']} repeated dates. Timezone: {r['timezone']}.",
                    "",
                ]
            )
    lines.extend(
        [
            "## September 12 innovation assessment correction",
            "",
            "This linked erratum corrects the assessment while retaining its original file and every sealed protocol.",
        ]
    )
    for v in value["correction"].values():
        lines.extend(["", v])
    lines.extend(["", "Evidence documents:"])
    for rel in DOCS:
        lines.append(f"- [{Path(rel).name}]({ORIGINAL/rel})")
    lines.extend(["", "Official support checked September 12 2026:"])
    for r in value["official_checks"]:
        lines.append(f"- [{r['support']}]({r['url']})")
    lines.extend(["", value["local_inventory_scope"], ""])
    return "\n".join(lines)


def observe_journal(journal):
    result = {}
    with sqlite3.connect(
        "file:" + str(permitted(journal)) + "?mode=ro", uri=True
    ) as db:
        db.execute("BEGIN")
        tables = {
            r[0]
            for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        result["table_counts"] = {
            t: db.execute('SELECT count(*) FROM "' + t + '"').fetchone()[0]
            for t in (
                "observations",
                "sessions",
                "outcomes",
                "fills",
                "decisions",
                "invalid_rows",
                "session_flags",
            )
            if t in tables
        }
        columns = {r[1] for r in db.execute("PRAGMA table_info(observations)")}
        if "available" in columns:
            result["available_observations"] = db.execute(
                "SELECT count(*) FROM observations WHERE available=1"
            ).fetchone()[0]
        timecol = next(
            (c for c in ("event", "event_timestamp", "timestamp") if c in columns),
            None,
        )
        if timecol:
            result["coverage_start"], result["coverage_end"] = db.execute(
                f"SELECT min({timecol}),max({timecol}) FROM observations"
            ).fetchone()
        if "collected" in columns:
            result["collection_start"], result["collection_end"] = db.execute(
                "SELECT min(collected),max(collected) FROM observations"
            ).fetchone()
        if "sessions" in tables:
            result["eligible_sessions"] = db.execute(
                "SELECT count(*) FROM sessions WHERE eligible=1"
            ).fetchone()[0]
    return result


def set_eligibility_status(row):
    count = row.get("eligible_sessions")
    row["eligibility_status"] = (
        "unknown"
        if count is None
        else "target_reached" if count >= 120 else "accumulating"
    )
    row["missing_evidence"] = (
        "Eligibility count unavailable; no efficacy analysis permitted."
        if count is None
        else f"{count} eligible forward sessions observed against the frozen 120-session target; no efficacy analysis performed or authorized by this inventory."
    )


def archive_scan():
    import os

    entries = []
    for directory, dirs, files in os.walk(
        permitted(ORIGINAL / "data"), followlinks=False
    ):
        dirs[:] = sorted(
            d
            for d in dirs
            if d != "sealed_holdout" and not (Path(directory) / d).is_symlink()
        )
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink() or path.suffix not in (".csv", ".parquet"):
                continue
            path = permitted(path)
            item = dict(
                path=str(path.relative_to(ORIGINAL)),
                kind=path.suffix[1:],
                columns=None,
                error=None,
            )
            if path.suffix == ".csv":
                try:
                    item["columns"] = list(pd.read_csv(permitted(path), nrows=0))
                except (
                    ValueError,
                    OSError,
                    pd.errors.ParserError,
                    pd.errors.EmptyDataError,
                ) as exc:
                    item["error"] = type(exc).__name__ + ": " + str(exc)
            entries.append(item)
    return dict(
        observed_at=now(),
        scope="Original data tree; sealed_holdout and symlinks excluded before traversal/read; CSV headers and parquet filenames only.",
        entries=entries,
    )
