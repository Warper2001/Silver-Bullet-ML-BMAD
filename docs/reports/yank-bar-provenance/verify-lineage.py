"""Independent lineage-to-source AND original-CSV check of fixed local evidence."""

import csv
import json
import sys
from datetime import datetime, timezone
from decimal import (
    Context,
    Decimal as D,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    ROUND_HALF_EVEN,
    localcontext,
)
from itertools import zip_longest
from pathlib import Path

ROOT = Path("/root/Silver-Bullet-ML-BMAD")
RESEARCH = "_bmad-output/planning-artifacts/research/technical-yank-bar-provenance-and-pilot-evidence-g-2026-09-07"
CSV = "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
FIELDS = ["open", "high", "low", "close", "volume", "notional"]
ARITHMETIC = Context(
    prec=200,
    rounding=ROUND_HALF_EVEN,
    Emin=-999999,
    Emax=999999,
    capitals=1,
    clamp=0,
    flags=[],
    traps=[InvalidOperation, DivisionByZero, Overflow],
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def instant(label):
    value = datetime.fromisoformat(label.replace("Z", "+00:00"))
    require(value.tzinfo is not None, "timestamp must include timezone")
    return value.astimezone(timezone.utc)


def verify(run, *, root=ROOT, expected=(289230, 29520, 5583)):
    raw = {}
    for i, line in enumerate(
        (root / RESEARCH / "imports/provenance-raw-2025.jsonl").open(), 1
    ):
        r = json.loads(line)
        require(r["source_line"] not in raw, "duplicate raw source line")
        require(instant(r["bar"]["TimeStamp"]).year == 2025, "source outside 2025")
        raw[r["source_line"]] = (i, r["bar"])
    seen = set()
    total = multi = mixed = 0
    previous = None
    with (
        localcontext(ARITHMETIC),
        (run / "lineage.jsonl").open() as lineage,
        (root / CSV).open(newline="") as original,
    ):
        csv_rows = csv.DictReader(original)
        require(csv_rows.fieldnames == ["timestamp", *FIELDS], "unexpected CSV schema")
        for i, (line, csv_row) in enumerate(zip_longest(lineage, csv_rows), 1):
            require(
                line is not None and csv_row is not None,
                f"CSV/lineage row count divergence at {i}",
            )
            r = json.loads(line)
            require(
                (r["csv_ordinal"], r["csv_line"]) == (i, i + 1),
                f"CSV ordinal divergence at {i}",
            )
            group = []
            for ref in r["constituents"]:
                n = ref["raw_line"]
                require(n not in seen, f"repeated raw line {n}")
                seen.add(n)
                el, bar = raw[n]
                require(
                    el == ref["extract_line"] and bar["Contract"] == ref["contract"],
                    f"source reference divergence at {i}",
                )
                require(
                    instant(bar["TimeStamp"]) == instant(ref["label"]),
                    f"source label divergence at {i}",
                )
                t = instant(bar["TimeStamp"])
                require(
                    previous is None or previous < t, f"source order divergence at {i}"
                )
                previous = t
                group.append(bar)
            require(bool(group), f"empty constituent group at {i}")
            value = [
                D(group[0]["Open"]),
                max(D(x["High"]) for x in group),
                min(D(x["Low"]) for x in group),
                D(group[-1]["Close"]),
                sum(D(x["TotalVolume"]) for x in group),
                sum(D(x["Close"]) * D(x["TotalVolume"]) * 20 for x in group),
            ]
            require(
                value == [D(r["reconstructed"][k]) for k in FIELDS],
                f"reconstructed value divergence at {i}",
            )
            require(
                value == [D(csv_row[k]) for k in FIELDS],
                f"original CSV numeric divergence at {i}",
            )
            label = instant(csv_row["timestamp"])
            require(
                label.year == 2025
                and label == instant(r["label"]) == instant(group[-1]["TimeStamp"]),
                f"original CSV timestamp divergence at {i}",
            )
            require(value[-1] >= 50000000, f"threshold not reached at {i}")
            require(
                sum(D(x["Close"]) * D(x["TotalVolume"]) * 20 for x in group[:-1])
                < 50000000,
                f"emission delayed at {i}",
            )
            require(not r["differences"], f"published differences at {i}")
            require(
                r["constituent_count"] == len(group),
                f"constituent count divergence at {i}",
            )
            require(
                r["contracts"] == sorted({x["Contract"] for x in group}),
                f"contract divergence at {i}",
            )
            total += 1
            multi += len(group) > 1
            mixed += len(r["contracts"]) > 1
    require(seen == set(raw), "missing source constituents")
    require((total, multi, mixed) == expected, "baseline counts differ")
    result = {
        "independent_full_lineage_check": "PASS",
        "rows": total,
        "constituents_once_each": len(seen),
        "multi": multi,
        "mixed": mixed,
        "original_csv_all_fields": "PASS",
    }
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: verify-lineage.py <local-output-directory>")
    verify(Path(sys.argv[1]))
