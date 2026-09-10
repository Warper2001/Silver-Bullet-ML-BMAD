"""Finite shadow collection in a Linux bubblewrap sandbox; never launches a service."""

import json, os, sqlite3, subprocess, sys, shutil
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from .artifacts import BASE, RUNS, digest, write_json
from .data import load, audit_select, Session
from .engine import ARMS, simulate


def launch(run_path, data, warmup, labels, protocol, historical_run, state=None):
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap unavailable: shadow denied, no unsafe fallback")
    history = Path(historical_run).resolve()
    manifest = json.loads((history / "manifest.json").read_text())
    completion = json.loads((history / "completion.json").read_text())
    for name, value in completion["sha256"].items():
        if digest(history / name) != value:
            raise ValueError("Historical run integrity failure")
    if manifest["command"] != "historical":
        raise ValueError("Historical run required")
    historical_source = {Path(k).name: v for k, v in manifest["source"].items()}
    if historical_source != {p.name: digest(p) for p in BASE.glob("*.py")}:
        raise ValueError("Historical implementation incompatible with collection")
    if manifest["config"].get("labels") != labels:
        raise ValueError("Historical timestamp configuration incompatible")
    historical_protocol = json.loads((history / "protocol.json").read_text())
    for key in (
        "primary_delay",
        "primary_cost",
        "costs",
        "quantity",
        "stop_points",
        "gross_reference_guard_usd",
    ):
        if historical_protocol.get(key) != protocol.get(key):
            raise ValueError("Historical execution configuration incompatible")
    report = json.loads((history / "report.json").read_text())
    mde = report.get("mde", {})
    if (
        mde.get("sessions") != 120
        or mde.get("power") != 0.8
        or mde.get("two_sided_alpha") != 0.05
        or not np.isfinite(mde.get("minimum_detectable_increment_usd", float("nan")))
    ):
        raise ValueError(
            "Historical 120-session power estimate required before collection"
        )
    if str(Path(warmup).resolve()) not in manifest["inputs"] or manifest["inputs"][
        str(Path(warmup).resolve())
    ] != digest(warmup):
        raise ValueError("Warmup must match audited historical source")
    state_path = Path(state).resolve() if state else run_path / "collector"
    if not state_path.is_relative_to(RUNS.resolve()):
        raise ValueError("Shadow state must be under isolated research runs")
    state_path.mkdir(exist_ok=True)
    if not (state_path / "freeze.json").exists():
        write_json(
            state_path / "freeze.json",
            {
                "protocol": historical_protocol,
                "warmup_hash": digest(warmup),
                "historical_manifest_hash": digest(history / "manifest.json"),
                "source": {p.name: digest(p) for p in BASE.glob("*.py")},
                "mde": report["mde"],
                "labels": labels,
            },
        )
    freeze = json.loads((state_path / "freeze.json").read_text())
    if freeze["historical_manifest_hash"] != digest(history / "manifest.json"):
        raise ValueError("Original historical manifest binding changed")
    if freeze["warmup_hash"] != digest(warmup) or freeze["labels"] != labels:
        raise ValueError("Frozen warmup/labels mismatch")
    if freeze["source"] != {p.name: digest(p) for p in BASE.glob("*.py")}:
        raise ValueError("Frozen implementation changed; collection denied")
    config = {
        "data": "/inputs/data.csv",
        "warmup": "/inputs/warmup.csv",
        "labels": labels,
        "state": "/state",
        "result": "/output/status.json",
    }
    write_json(run_path / "worker-config.json", config)
    cmd = sandbox_command(run_path, data, warmup, state_path)
    completed = subprocess.run(cmd, env={}, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError("Sandboxed shadow failed: " + completed.stderr[-3000:])
    return json.loads((run_path / "status.json").read_text())


def sandbox_command(run_path, data, warmup, state_path, worker_argv=None):
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap unavailable")
    # Namespace isolation, clean environment, no host home, no network, read-only source/runtime.
    cmd = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "PYTHONPATH",
        "/code",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/code",
        "--dir",
        "/code/research/mim_comparison",
    ]
    for directory in ["/usr", "/lib", "/lib64"]:
        if Path(directory).exists():
            cmd += ["--ro-bind", directory, directory]
    # venv symlinks/interpreter prefix work at original absolute path; no general /root access.
    venv = Path(sys.prefix).resolve()
    if str(venv) not in ("/usr", "/usr/local"):
        cmd += ["--ro-bind", str(venv), str(venv)]
    for module in sorted(BASE.glob("*.py")):
        cmd += [
            "--ro-bind",
            str(module),
            "/code/research/mim_comparison/" + module.name,
        ]
    cmd += [
        "--ro-bind",
        str(Path(data).resolve()),
        "/inputs/data.csv",
        "--ro-bind",
        str(Path(warmup).resolve()),
        "/inputs/warmup.csv",
        "--bind",
        str(state_path),
        "/state",
        "--bind",
        str(run_path),
        "/output",
        "--remount-ro",
        "/code",
        "--chdir",
        "/code",
    ]
    cmd += worker_argv or [
        sys.executable,
        "-m",
        "research.mim_comparison.shadow",
        "/output/worker-config.json",
    ]
    return cmd


def _normalize(raw, labels):
    from .data import expiry

    required = {
        "contract",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "received_at",
    }
    if not required <= set(raw):
        raise ValueError("Explicit contract OHLCV and receipt required")
    expiry(raw["contract"])
    stamp = pd.Timestamp(raw["timestamp"])
    if stamp.tzinfo is None or pd.isna(stamp):
        raise ValueError("Timestamp timezone required")
    if stamp != stamp.floor("min"):
        raise ValueError("Nonminute timestamp")
    stamp = stamp.tz_convert("America/New_York")
    if stamp.weekday() >= 5:
        raise ValueError("Weekend RTH closed")
    if labels == "start":
        stamp += pd.Timedelta(minutes=1)
    result = {k: raw[k] for k in required}
    for name in ("open", "high", "low", "close", "volume"):
        result[name] = float(result[name])
    receipt = pd.Timestamp(result["received_at"])
    if pd.isna(receipt) or receipt.tzinfo is None:
        raise ValueError("Valid receipt timezone required")
    numeric = np.asarray(
        [result[k] for k in ("open", "high", "low", "close", "volume")]
    )
    if not np.isfinite(numeric).all() or min(numeric[:4]) <= 0 or numeric[4] < 0:
        raise ValueError("Invalid numeric OHLCV")
    if result["high"] < max(result["open"], result["low"], result["close"]) or result[
        "low"
    ] > min(result["open"], result["high"], result["close"]):
        raise ValueError("Invalid OHLC")
    result.update(
        timestamp=stamp.isoformat(),
        day=stamp.strftime("%Y-%m-%d"),
        minute=stamp.hour * 60 + stamp.minute,
    )
    return result


def _frame(records):
    result = pd.DataFrame(records)
    if len(result):
        result["timestamp"] = pd.to_datetime(result.timestamp, utc=True).dt.tz_convert(
            "America/New_York"
        )
    return result


def _valid_complete(frame):
    if len(frame) and (frame.timestamp.dt.weekday >= 5).any():
        return False
    if (
        len(frame) != 390
        or frame.timestamp.duplicated().any()
        or set(frame.minute) != set(range(571, 961))
    ):
        return False
    numeric = frame[["open", "high", "low", "close", "volume"]].to_numpy(float)
    return bool(
        np.isfinite(numeric).all()
        and (frame[["open", "high", "low", "close"]] > 0).all().all()
        and (frame.volume >= 0).all()
        and (frame.high >= frame[["open", "low", "close"]].max(axis=1)).all()
        and (frame.low <= frame[["open", "high", "close"]].min(axis=1)).all()
        and frame.timestamp.eq(frame.timestamp.dt.floor("min")).all()
    )


def context_for(day, warmframe, db, cache=None):
    from .data import expiry

    prior = db.execute(
        "SELECT payload,available FROM observations WHERE day<? ORDER BY event,contract",
        (day,),
    ).fetchall()
    observed = _frame([json.loads(r[0]) for r in prior if r[1]])
    if cache is None:
        cache = {}
    if "warm_sessions" not in cache:
        cache["warm_sessions"] = audit_select(warmframe)[0]
        cache["warm_days"] = {
            day: g
            for day, g in warmframe.groupby("day")
            if pd.Timestamp(day).weekday() < 5
        }
    history = [session for session in cache["warm_sessions"] if session.day < day]
    # Static multi-year warmup is audited once. Re-audit only the short prospective tail and its boundary day.
    warm_days = cache["warm_days"]
    boundary = warmframe.iloc[0:0] if not warm_days else warm_days[max(warm_days)]
    if len(observed):
        tail, _ = audit_select(pd.concat([boundary, observed], ignore_index=True))
        history.extend(tail)
    sessions = history
    combined = pd.concat(
        [
            warmframe[
                warmframe.day
                == ((pd.Timestamp(day) - pd.offsets.BDay(1)).strftime("%Y-%m-%d"))
            ],
            observed,
        ],
        ignore_index=True,
    )
    prior_day = (pd.Timestamp(day) - pd.offsets.BDay(1)).strftime("%Y-%m-%d")
    candidates = []
    for contract, g in combined[combined.day == prior_day].groupby("contract"):
        if _valid_complete(g) and pd.Timestamp(day).date() < expiry(contract):
            candidates.append(
                (
                    -float(g.volume.sum()),
                    expiry(contract),
                    contract,
                    float(g.sort_values("timestamp").close.iloc[-1]),
                )
            )
    if not candidates:
        return None, None, None, None, "missing_previous_session_contract_volume"
    _, _, contract, prev = min(candidates)
    if len(sessions) < 14:
        return contract, prev, None, None, "warmup_incomplete"
    moves = [
        np.abs(s.bars.close.to_numpy() / float(s.bars.open.iloc[0]) - 1)
        for s in sessions[-14:]
    ]
    sigma = np.mean(moves, axis=0)
    # At collection the first history day is far outside the 14-day window, so author min13 equals deployed14.
    return contract, prev, sigma, sigma, ""


def _now():
    return pd.Timestamp.now(tz="UTC")


def _raw_identity(raw, labels):
    try:
        stamp = pd.Timestamp(raw.get("timestamp"))
        if pd.isna(stamp) or stamp.tzinfo is None:
            return None, None, None
        stamp = stamp.tz_convert("America/New_York")
        if labels == "start":
            stamp += pd.Timedelta(minutes=1)
        return stamp.isoformat(), raw.get("contract"), stamp.strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None, None, None


def _marks_complete(db, day):
    rows = db.execute(
        "SELECT event,arm,payload FROM decisions WHERE substr(event,1,10)=?", (day,)
    ).fetchall()
    actual = {
        (event, arm)
        for event, arm, payload in rows
        if json.loads(payload).get("eligible") is True
    }
    expected = {
        (
            pd.Timestamp(day + f" {h:02d}:{m:02d}", tz="America/New_York").isoformat(),
            arm,
        )
        for h in range(10, 16)
        for m in (0, 30)
        for arm in ("A", "B")
    }
    return expected <= actual


def finalize_sessions(db, start, deadline, now, warmframe, cache):
    for date in pd.bdate_range(
        start.tz_convert("America/New_York").date(),
        min(now, deadline).tz_convert("America/New_York").date(),
    ):
        if (
            db.execute("SELECT COUNT(*) FROM sessions WHERE eligible=1").fetchone()[0]
            >= 120
        ):
            break
        day = date.strftime("%Y-%m-%d")
        opened = pd.Timestamp(day + " 09:30", tz="America/New_York")
        closed = pd.Timestamp(day + " 16:00", tz="America/New_York")
        if opened <= start or closed > now or closed > deadline:
            continue
        if db.execute("SELECT 1 FROM sessions WHERE day=?", (day,)).fetchone():
            continue
        # Silent dates do not need an expensive strategy context to establish unavailability.
        if not db.execute(
            "SELECT 1 FROM observations WHERE day=? AND available=1 LIMIT 1", (day,)
        ).fetchone():
            if closed + pd.Timedelta(seconds=60) > now:
                continue
            db.execute(
                "INSERT INTO sessions VALUES (?,?,0,?)",
                (day, None, "missing_late_or_exchange_closed_unverified"),
            )
            continue
        selected, prev, sigma, _, reason = context_for(day, warmframe, db, cache)
        records = db.execute(
            "SELECT payload,available FROM observations WHERE day=? AND contract=? ORDER BY event",
            (day, selected),
        ).fetchall()
        frame = _frame([json.loads(r[0]) for r in records])
        flags = db.execute(
            "SELECT reason FROM session_flags WHERE day IN (?, '*')", (day,)
        ).fetchall()
        eligible = (
            not reason
            and not flags
            and bool(records)
            and all(r[1] for r in records)
            and _valid_complete(frame)
            and _marks_complete(db, day)
        )
        if not eligible and closed + pd.Timedelta(seconds=60) > now:
            continue
        exclusion = reason or (
            flags[0][0]
            if flags
            else "missing_late_invalid_or_unavailable_durable_marks"
        )
        db.execute(
            "INSERT INTO sessions VALUES (?,?,?,?)",
            (day, selected, int(eligible), "" if eligible else exclusion),
        )
        if eligible:
            session = Session(day, selected, frame, prev)
            for arm in ARMS[:2]:
                for delay in (2, 1):
                    outcome, fills, _ = simulate(session, sigma, arm, delay, 2.24)
                    for cost in (2.24, 3.24, 6.24):
                        adjusted = dict(
                            outcome,
                            cost=cost,
                            costs=outcome["turnover"] * cost / 2,
                            net=outcome["gross"] - outcome["turnover"] * cost / 2,
                        )
                        db.execute(
                            "INSERT OR IGNORE INTO outcomes VALUES (?,?,?,?,?)",
                            (day, arm.name, delay, cost, json.dumps(adjusted)),
                        )
                    for index, fill in enumerate(fills):
                        db.execute(
                            "INSERT OR IGNORE INTO fills VALUES (?,?,?,?,?)",
                            (
                                day,
                                arm.name,
                                delay,
                                index,
                                json.dumps(dict(fill, delay=delay)),
                            ),
                        )
    db.commit()


def collect(config):
    import csv

    state = Path(config["state"])
    freeze = json.loads((state / "freeze.json").read_text())
    protocol = freeze["protocol"]
    start = pd.Timestamp(protocol["freeze"])
    deadline = start + pd.DateOffset(months=9)
    import hashlib, io

    consumed = Path(config["warmup"]).read_bytes()
    if hashlib.sha256(consumed).hexdigest() != freeze["warmup_hash"]:
        raise ValueError("Consumed warmup differs from freeze")
    if config["labels"] != freeze["labels"] or freeze["source"] != {
        p.name: digest(p) for p in BASE.glob("*.py")
    }:
        raise ValueError("Frozen collector configuration/source changed")
    warmframe = load(io.BytesIO(consumed), config["labels"])
    del consumed
    warmframe = warmframe[
        pd.to_datetime(warmframe.day + " 16:00").dt.tz_localize("America/New_York")
        < start
    ]
    install_socket_filter()
    db = sqlite3.connect(state / "journal.sqlite")
    db.execute("PRAGMA journal_mode=DELETE")
    db.execute("PRAGMA synchronous=FULL")
    db.execute(
        "CREATE TABLE IF NOT EXISTS observations (event TEXT,contract TEXT,day TEXT,payload TEXT NOT NULL,collected TEXT NOT NULL,available INTEGER NOT NULL,reason TEXT NOT NULL,PRIMARY KEY(event,contract))"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS decisions (event TEXT,arm TEXT,payload TEXT NOT NULL,PRIMARY KEY(event,arm))"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS sessions (day TEXT PRIMARY KEY,contract TEXT,eligible INTEGER NOT NULL,reason TEXT NOT NULL)"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS invalid_rows (identity TEXT PRIMARY KEY,source TEXT,line INTEGER,event TEXT,contract TEXT,day TEXT,raw TEXT,error TEXT,collected TEXT)"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS session_flags (day TEXT,reason TEXT,PRIMARY KEY(day,reason))"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS outcomes(day TEXT,arm TEXT,delay INTEGER,cost REAL,payload TEXT,PRIMARY KEY(day,arm,delay,cost))"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS fills(day TEXT,arm TEXT,delay INTEGER,ordinal INTEGER,payload TEXT,PRIMARY KEY(day,arm,delay,ordinal))"
    )
    corrections = 0
    contexts = {}
    cache = {}
    ended = (
        db.execute("SELECT COUNT(*) FROM sessions WHERE eligible=1").fetchone()[0]
        >= 120
        or _now() > deadline
    )
    if not ended:
        with open(config["data"], newline="") as source:
            for line, raw in enumerate(csv.DictReader(source), start=2):
                try:
                    row = _normalize(raw, config["labels"])
                    serialized = json.dumps(row, sort_keys=True, allow_nan=False)
                except (ValueError, TypeError, OverflowError) as error:
                    event, contract, day = _raw_identity(raw, config["labels"])
                    if (
                        event
                        and contract
                        and db.execute(
                            "SELECT 1 FROM observations WHERE event=? AND contract=?",
                            (event, contract),
                        ).fetchone()
                    ):
                        corrections += 1
                        continue
                    raw_json = json.dumps(raw, sort_keys=True)
                    identity = (
                        event + "|" + contract
                        if event and contract
                        else config["data"]
                        + "|"
                        + str(line)
                        + "|"
                        + hashlib.sha256(raw_json.encode()).hexdigest()
                    )
                    db.execute(
                        "INSERT OR IGNORE INTO invalid_rows VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            identity,
                            config["data"],
                            line,
                            event,
                            contract,
                            day,
                            raw_json,
                            str(error),
                            _now().isoformat(),
                        ),
                    )
                    db.execute(
                        "INSERT OR IGNORE INTO session_flags VALUES (?,?)",
                        (
                            day if event and contract else "*",
                            (
                                "invalid_first_observation"
                                if event and contract
                                else "unidentified_input_row"
                            ),
                        ),
                    )
                    db.commit()
                    continue
                minute = row["minute"]
                if not 571 <= minute <= 960:
                    continue
                event = row["timestamp"]
                contract = row["contract"]
                day = row["day"]
                stamp = pd.Timestamp(event)
                if db.execute(
                    "SELECT 1 FROM invalid_rows WHERE event=? AND (contract=? OR contract IS NULL OR contract='')",
                    (event, contract),
                ).fetchone():
                    corrections += 1
                    continue
                old = db.execute(
                    "SELECT payload FROM observations WHERE event=? AND contract=?",
                    (event, contract),
                ).fetchone()
                if old:
                    if old[0] != serialized:
                        corrections += 1
                    continue
                now = _now()
                receipt = pd.Timestamp(row["received_at"])
                opened = pd.Timestamp(day + " 09:30", tz="America/New_York")
                reason = ""
                if db.execute("SELECT 1 FROM session_flags WHERE day='*'").fetchone():
                    reason = "unidentified_input_row"
                elif opened <= start or stamp > deadline or now > deadline:
                    reason = "outside_frozen_horizon"
                elif not (
                    receipt <= now
                    and 0 <= (receipt - stamp).total_seconds() <= 60
                    and 0 <= (now - stamp).total_seconds() <= 60
                ):
                    reason = "late_backfilled_or_future_receipt"
                db.execute(
                    "INSERT INTO observations VALUES (?,?,?,?,?,?,?)",
                    (
                        event,
                        contract,
                        day,
                        serialized,
                        now.isoformat(),
                        int(not reason),
                        reason,
                    ),
                )
                if reason:
                    for arm in ARMS[:2]:
                        db.execute(
                            "INSERT OR IGNORE INTO decisions VALUES (?,?,?)",
                            (
                                event,
                                arm.name,
                                json.dumps(
                                    {
                                        "event_timestamp": event,
                                        "arm": arm.name,
                                        "eligible": False,
                                        "exclusion": reason,
                                    }
                                ),
                            ),
                        )
                    db.commit()
                    if now > deadline:
                        break
                    continue  # irrevocably unavailable rows never need strategy/warmup reconstruction
                if day not in contexts:
                    contexts[day] = context_for(day, warmframe, db, cache)
                selected, prev, sigma, author_sigma, context_reason = contexts[day]
                if contract == selected:
                    records = db.execute(
                        "SELECT payload,available FROM observations WHERE day=? AND contract=? ORDER BY event",
                        (day, contract),
                    ).fetchall()
                    today = _frame([json.loads(r[0]) for r in records])
                    prefix = len(today) == minute - 570 and set(today.minute) == set(
                        range(571, minute + 1)
                    )
                    valid = not context_reason and all(r[1] for r in records) and prefix
                    if valid and minute % 30 == 0:
                        session = Session(day, contract, today, prev)
                        for arm, full_sigma in zip(ARMS[:2], (sigma, author_sigma)):
                            _, _, decisions = simulate(
                                session,
                                full_sigma[today.minute.to_numpy(int) - 571],
                                arm,
                                2,
                                2.24,
                            )
                            for dec in decisions:
                                db.execute(
                                    "INSERT OR IGNORE INTO decisions VALUES (?,?,?)",
                                    (
                                        dec["event_timestamp"],
                                        arm.name,
                                        json.dumps(dec, allow_nan=False),
                                    ),
                                )
                    elif not valid:
                        db.execute(
                            "INSERT OR IGNORE INTO session_flags VALUES (?,?)",
                            (
                                day,
                                context_reason or "missing_prefix_at_first_observation",
                            ),
                        )
                        for arm in ARMS[:2]:
                            db.execute(
                                "INSERT OR IGNORE INTO decisions VALUES (?,?,?)",
                                (
                                    event,
                                    arm.name,
                                    json.dumps(
                                        {
                                            "event_timestamp": event,
                                            "arm": arm.name,
                                            "eligible": False,
                                            "exclusion": context_reason
                                            or "missing_prefix",
                                        }
                                    ),
                                ),
                            )
                db.commit()  # durability before consuming the next csv record
                if minute == 960:
                    finalize_sessions(db, start, deadline, now, warmframe, cache)
                    if (
                        db.execute(
                            "SELECT COUNT(*) FROM sessions WHERE eligible=1"
                        ).fetchone()[0]
                        >= 120
                    ):
                        break
    now = _now()
    finalize_sessions(db, start, deadline, now, warmframe, cache)
    eligible = db.execute("SELECT COUNT(*) FROM sessions WHERE eligible=1").fetchone()[
        0
    ]
    status = {
        "eligible_sessions": eligible,
        "required": 120,
        "corrections_ignored": corrections,
        "deadline": deadline.isoformat(),
        "no_interim_efficacy": True,
        "collection_status": (
            "complete"
            if eligible >= 120
            else "incomplete/inconclusive" if now > deadline else "collecting"
        ),
        "prospective_ready": True,
        "deployment_authorized": False,
        "freeze_hash": digest(state / "freeze.json"),
        "invalid_first_rows": db.execute(
            "SELECT COUNT(*) FROM invalid_rows"
        ).fetchone()[0],
        "limitations": [
            "Unverified closed weekdays excluded; no automatic extension",
            "Contractless operational logs refused",
            "All days require actual collection within60 seconds",
            "Unidentified malformed rows permanently disqualify this collector state",
        ],
    }
    snapshot = Path(config["result"]).parent / "journal-snapshot.sqlite"
    output = Path(config["result"]).parent
    for table, filename in [
        ("decisions", "decisions.csv"),
        ("outcomes", "daily.csv"),
        ("fills", "ledger.csv"),
    ]:
        records = [
            json.loads(row[0])
            for row in db.execute("SELECT payload FROM " + table + " ORDER BY rowid")
        ]
        if records:
            pd.DataFrame(records).to_csv(output / filename, index=False)
    eligibility = pd.read_sql_query("SELECT * FROM sessions ORDER BY day", db)
    eligibility.to_csv(output / "eligibility.csv", index=False)
    backup = sqlite3.connect(snapshot)
    db.backup(backup)
    backup.close()
    db.close()
    (snapshot.parent / "freeze-snapshot.json").write_bytes(
        (state / "freeze.json").read_bytes()
    )
    status["journal_hash"] = digest(snapshot)
    if digest(config["warmup"]) != freeze["warmup_hash"] or freeze["source"] != {
        p.name: digest(p) for p in BASE.glob("*.py")
    }:
        raise ValueError("Frozen inputs/source changed during collection")
    write_json(Path(config["result"]), status)


def evaluate(shadow_run, path):
    from .statistics import decision

    source = Path(shadow_run)
    completion = json.loads((source / "completion.json").read_text())
    for name, value in completion["sha256"].items():
        if digest(source / name) != value:
            raise ValueError("Shadow artifact integrity failure")
    status = json.loads((source / "status.json").read_text())
    if status["eligible_sessions"] < 120:
        write_json(
            path / "decision.json",
            {
                "decision": "incomplete/inconclusive",
                "eligible_sessions": status["eligible_sessions"],
                "deployment_authorized": False,
                "reason": "No interim efficacy; fewer than120 eligible sessions",
            },
        )
        (path / "report.md").write_text(
            "# Prospective evaluation\n\nIncomplete/inconclusive: "
            + str(status["eligible_sessions"])
            + " of120 eligible sessions. No interim efficacy or deployment authorization.\n"
        )
        return
    freeze = json.loads((source / "freeze-snapshot.json").read_text())
    if freeze["source"] != {p.name: digest(p) for p in BASE.glob("*.py")}:
        raise ValueError("Frozen implementation changed; evaluate denied")
    if (
        digest(source / "freeze-snapshot.json") != status["freeze_hash"]
        or digest(source / "journal-snapshot.sqlite") != status["journal_hash"]
    ):
        raise ValueError("Snapshot integrity mismatch")
    manifest = json.loads((source / "manifest.json").read_text())
    warmup = manifest["config"]["warmup"]
    if digest(warmup) != freeze["warmup_hash"]:
        raise ValueError("Warmup integrity mismatch")
    import hashlib, io

    consumed = Path(warmup).read_bytes()
    if hashlib.sha256(consumed).hexdigest() != freeze["warmup_hash"]:
        raise ValueError("Consumed evaluation warmup differs from freeze")
    warm = load(io.BytesIO(consumed), freeze["labels"])
    del consumed
    start = pd.Timestamp(freeze["protocol"]["freeze"])
    deadline = start + pd.DateOffset(months=9)
    warm = warm[
        pd.to_datetime(warm.day + " 16:00").dt.tz_localize("America/New_York") < start
    ]
    db = sqlite3.connect(
        "file:" + str((source / "journal-snapshot.sqlite").resolve()) + "?mode=ro",
        uri=True,
    )
    sessions = db.execute(
        "SELECT day,contract FROM sessions WHERE eligible=1 ORDER BY day LIMIT120".replace(
            "LIMIT120", "LIMIT 120"
        )
    ).fetchall()
    daily = []
    ledger = []
    cache = {}
    for day, contract in sessions:
        if pd.Timestamp(day + " 16:00", tz="America/New_York") > deadline:
            raise ValueError("Session outside horizon")
        selected, prev, sigma, authors, reason = context_for(day, warm, db, cache)
        if pd.Timestamp(day + " 09:30", tz="America/New_York") <= start:
            raise ValueError("Session predates freeze")
        if selected != contract or reason:
            raise ValueError("Session provenance changed on replay")
        rows = db.execute(
            "SELECT payload,available FROM observations WHERE day=? AND contract=? ORDER BY event",
            (day, contract),
        ).fetchall()
        bars = _frame([json.loads(r[0]) for r in rows])
        collected = db.execute(
            "SELECT collected FROM observations WHERE day=? AND contract=? ORDER BY event",
            (day, contract),
        ).fetchall()
        for (_, bar), (seen,) in zip(bars.iterrows(), collected):
            receipt = pd.Timestamp(bar.received_at)
            observed = pd.Timestamp(seen)
            if (
                receipt.tzinfo is None
                or observed.tzinfo is None
                or not (
                    receipt <= observed
                    and 0 <= (receipt - bar.timestamp).total_seconds() <= 60
                    and 0 <= (observed - bar.timestamp).total_seconds() <= 60
                )
            ):
                raise ValueError("Unauthenticated timeliness")
        if not all(r[1] for r in rows) or not _valid_complete(bars):
            raise ValueError("Invalid authenticated session")
        session = Session(day, contract, bars, prev)
        for arm, sig in zip(ARMS[:2], (sigma, authors)):
            for delay in (2, 1):
                row, events, decisions = simulate(session, sig, arm, delay, 2.24)
                if delay == 2:
                    for dec in decisions:
                        saved = db.execute(
                            "SELECT payload FROM decisions WHERE event=? AND arm=?",
                            (dec["event_timestamp"], arm.name),
                        ).fetchone()
                        if saved is None or json.dumps(
                            json.loads(saved[0]), sort_keys=True, allow_nan=True
                        ) != json.dumps(dec, sort_keys=True, allow_nan=True):
                            raise ValueError("Durable decision reconciliation failed")
                for cost in (2.24, 3.24, 6.24):
                    daily.append(
                        dict(
                            row,
                            cost=cost,
                            costs=row["turnover"] * cost / 2,
                            net=row["gross"] - row["turnover"] * cost / 2,
                        )
                    )
                ledger.extend(dict(e, delay=delay) for e in events)
    db.close()
    if digest(warmup) != freeze["warmup_hash"] or freeze["source"] != {
        p.name: digest(p) for p in BASE.glob("*.py")
    }:
        raise ValueError("Frozen inputs/source changed during evaluation")
    frame = pd.DataFrame(daily)
    frame.to_csv(path / "daily.csv", index=False)
    pd.DataFrame(ledger).to_csv(path / "ledger.csv", index=False)
    from .artifacts import paired_daily

    paired_daily(frame).to_csv(path / "paired-daily.csv", index=False)
    result = decision(frame, complete=len(sessions) == 120)
    write_json(path / "decision.json", result)
    (path / "report.md").write_text(
        "# Final prospective evaluation\n\n"
        + result["decision"]
        + ". "
        + str(result["eligible_sessions"])
        + " eligible paired sessions.\n\n"
        + json.dumps(result, indent=2)
        + "\n\nThis result never authorizes deployment.\n"
    )


def install_socket_filter():
    import ctypes, platform

    if platform.machine() != "x86_64":
        raise RuntimeError("Socket sandbox currently supports Linux x86_64 only")

    class Filter(ctypes.Structure):
        _fields_ = [
            ("code", ctypes.c_ushort),
            ("jt", ctypes.c_ubyte),
            ("jf", ctypes.c_ubyte),
            ("k", ctypes.c_uint),
        ]

    class Program(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ushort), ("filter", ctypes.POINTER(Filter))]

    # Validate AUDIT_ARCH_X86_64 before interpreting syscall numbers; reject compat and x32 ABIs.
    rules = (Filter * 10)(
        Filter(0x20, 0, 0, 4),
        Filter(0x15, 1, 0, 0xC000003E),
        Filter(0x06, 0, 0, 0x00050001),
        Filter(0x20, 0, 0, 0),
        Filter(0x35, 0, 1, 0x40000000),
        Filter(0x06, 0, 0, 0x00050001),
        Filter(0x15, 1, 0, 41),
        Filter(0x15, 0, 1, 53),
        Filter(0x06, 0, 0, 0x00050001),
        Filter(0x06, 0, 0, 0x7FFF0000),
    )
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) or libc.prctl(
        22, 2, ctypes.byref(Program(10, rules))
    ):
        raise OSError(ctypes.get_errno(), "seccomp failed")


if __name__ == "__main__":
    collect(json.loads(Path(sys.argv[1]).read_text()))
