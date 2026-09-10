"""Independent small timelines plus evidence/failure boundary coverage."""

import csv
import json
from datetime import datetime, timezone
from decimal import Decimal as D
from pathlib import Path

import pytest

from src.cli.check_yank_bar_provenance import main
from src.research.yank_bar_provenance import validator as v


def source(minute, close="100", volume="25000", contract="A", open_=None):
    return dict(
        TimeStamp=f"2025-01-02T00:{minute:02d}:00Z",
        Open=open_ or close,
        High=str(max(D(close), D(open_ or close))),
        Low=str(min(D(close), D(open_ or close))),
        Close=close,
        TotalVolume=volume,
        Contract=contract,
    )


def evidence(tmp_path, bars=None):
    root = tmp_path / "inputs"
    root.mkdir()
    bars = bars or [source(1)]
    text = "[\n" + ",\n".join(json.dumps(b, indent=2) for b in bars) + "\n]\n"
    (root / v.RAW).write_text(text)
    raw = list(v.raw_2025(root / v.RAW))
    (root / v.EXTRACT).parent.mkdir(parents=True)
    (root / v.EXTRACT).write_text("".join(json.dumps(x) + "\n" for _, x in raw))
    records = v.verified_sources(root)
    generated = list(v.reconstruct(records))
    (root / v.CSV).parent.mkdir(parents=True)
    with (root / v.CSV).open("w") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", *v.FIELDS])
        for constituents, values in generated:
            w.writerow([v.iso(constituents[-1]["time"]), *map(str, values)])
    rows = []
    for constituents, values in generated:
        label = constituents[-1]["time"]
        for clock in ("capture", "exchange"):
            rows.append(
                dict(
                    clock=clock,
                    convention="end",
                    original_label=v.iso(label),
                    native_minute_start=v.iso(label - v.timedelta(minutes=1)),
                    original_ohlc=list(map(str, values[:4])),
                    original_volume=str(values[4]),
                    native_ohlc=list(map(str, values[:4])),
                    native_T_volume=str(values[4]),
                    ohlc_delta=["0"] * 4,
                    volume_delta="0",
                    coverage_qualification="fixture",
                    gap_reasons=[],
                )
            )
    (root / v.RECON).parent.mkdir(parents=True)
    (root / v.RECON).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def pins(root):
    return {n: v.digest(root / n) for n in (v.RAW, v.EXTRACT, v.CSV, v.RECON)}


def test_exact_threshold_sorting_overshoot_mixing_and_timing(tmp_path):
    # First two together equal threshold, third overshoots, fourth starts fresh.
    root = evidence(
        tmp_path,
        [
            source(4, volume="25000"),
            source(3, volume="30000"),
            source(2, volume="12500", contract="B"),
            source(0, volume="12500"),
        ],
    )
    records = v.verified_sources(root)
    rows = list(v.reconstruct(records))
    assert [str(x[1][-1]) for x in rows] == ["50000000", "60000000", "50000000"]
    assert [len(x[0]) for x in rows] == [2, 1, 1]
    assert [x["source_ordinal"] for x in rows[0][0]] == [4, 3]
    result = v.lineage_row(1, *rows[0], {}, pins(root))
    assert result["contract_class"] == "mixed"
    assert result["timing"]["conditional_end"] == [
        "2025-01-01T23:59:00Z",
        "2025-01-02T00:02:00Z",
    ]
    assert result["timing"]["conditional_start"] == [
        "2025-01-02T00:00:00Z",
        "2025-01-02T00:03:00Z",
    ]
    assert result["timing"]["gaps"][0]["seconds"] == 120
    assert result["timing"]["availability"] is None
    refs = [r["source_ordinal"] for sources, _ in rows for r in sources]
    assert sorted(refs) == [1, 2, 3, 4]


def test_exact_decimal_and_leftovers():
    r = dict(
        time=datetime(2025, 1, 1, tzinfo=timezone.utc),
        values=(D("0.1"),) * 4 + (D(25000000),),
        contract="A",
        source_line=2,
        source_ordinal=1,
        extract_line=1,
    )
    assert list(v.reconstruct([r]))[0][1][-1] == D(50000000)
    r["values"] = (D("0.1"),) * 4 + (D(1),)
    with pytest.raises(v.EvidenceError, match="leftover"):
        list(v.reconstruct([r]))


@pytest.mark.parametrize("value", ["NaN", "Infinity", "bad", True])
def test_bad_numbers(value):
    with pytest.raises(v.EvidenceError):
        v.number(value, "test")


@pytest.mark.parametrize(
    "value",
    [
        "2025-01-01",
        "2025-13-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        "2025-01-01T00:00:00.000000001Z",
    ],
)
def test_bad_timestamps(value):
    with pytest.raises(v.EvidenceError):
        v.stamp(value, "test")


@pytest.mark.parametrize(
    "vals", [[1, 0, 2, 1, 1], [1, 1, 1, 1, -1], [1, 1, 1, 1, "1.5"]]
)
def test_bad_ohlcv(vals):
    with pytest.raises(v.EvidenceError):
        v.ohlcv(vals, "test")


def test_deterministic_runs_and_collisions(tmp_path):
    root = evidence(tmp_path)
    p = pins(root)
    for name in ("a", "b"):
        report = v.run(tmp_path / name, root=root, pins=p, baselines={})
        assert report["status"] == "PASS_PROVENANCE_CHECKS"
        assert report["data_suitability"] == "BLOCKED"
        assert report["research_status"] == "HOLD_VALIDATION"
    files = [
        "report.md",
        "report.json",
        "manifest.json",
        "lineage.jsonl",
        "pilot-differences.jsonl",
    ]
    assert all(
        (tmp_path / "a" / n).read_bytes() == (tmp_path / "b" / n).read_bytes()
        for n in files
    )
    with pytest.raises(v.EvidenceError, match="already exists"):
        v.run(tmp_path / "a", root=root, pins=p, baselines={})
    with pytest.raises(v.EvidenceError, match="overlaps"):
        v.run((root / v.CSV).parent / "new", root=root, pins=p, baselines={})
    (tmp_path / "alias").symlink_to((root / v.CSV).parent, target_is_directory=True)
    with pytest.raises(v.EvidenceError, match="overlaps"):
        v.run(tmp_path / "alias" / "new", root=root, pins=p, baselines={})


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "changed",
        "extract_ref",
        "extract_extra",
        "extract_wrong_year",
        "malformed",
        "csv_diff",
        "csv_extra",
        "csv_missing",
        "csv_nonfinite",
        "csv_year",
        "csv_malformed",
        "pilot_missing",
        "pilot_duplicate",
        "pilot_delta",
    ],
)
def test_failures_never_publish_success(tmp_path, fault):
    root = evidence(tmp_path)
    p = pins(root)
    if fault == "missing":
        (root / v.RAW).unlink()
    elif fault == "changed":
        (root / v.RAW).write_text((root / v.RAW).read_text() + " ")
    elif fault.startswith("extract_"):
        r = json.loads((root / v.EXTRACT).read_text())
        if fault == "extract_ref":
            r["source_line"] += 1
        if fault == "extract_wrong_year":
            r["bar"]["TimeStamp"] = "2026-01-02T00:01:00Z"
        (root / v.EXTRACT).write_text(json.dumps(r) + "\n")
        if fault == "extract_extra":
            (root / v.EXTRACT).write_text((root / v.EXTRACT).read_text() * 2)
    elif fault == "malformed":
        (root / v.EXTRACT).write_text("{bad}\n")
    elif fault == "csv_diff":
        (root / v.CSV).write_text(
            (root / v.CSV).read_text().replace("50000000", "50000001")
        )
    elif fault == "csv_missing":
        (root / v.CSV).write_text((root / v.CSV).read_text().splitlines()[0] + "\n")
    elif fault == "csv_nonfinite":
        (root / v.CSV).write_text((root / v.CSV).read_text().replace("50000000", "NaN"))
    elif fault == "csv_year":
        (root / v.CSV).write_text((root / v.CSV).read_text().replace("2025-", "2026-"))
    elif fault == "csv_malformed":
        (root / v.CSV).write_text((root / v.CSV).read_text().rstrip() + ",extra\n")
    elif fault == "csv_extra":
        lines = (root / v.CSV).read_text().splitlines()
        (root / v.CSV).write_text("\n".join(lines + [lines[1]]) + "\n")
    else:
        rows = [json.loads(x) for x in (root / v.RECON).read_text().splitlines()]
        if fault == "pilot_missing":
            rows.pop()
        elif fault == "pilot_duplicate":
            rows.append(rows[0])
        else:
            rows[0]["ohlc_delta"][0] = "1"
        (root / v.RECON).write_text("".join(json.dumps(r) + "\n" for r in rows))
    if fault not in ("missing", "changed"):
        p = pins(root)
    with pytest.raises(Exception):
        v.run(tmp_path / "out", root=root, pins=p, baselines={})
    result = json.loads((tmp_path / "out" / "report.json").read_text())
    assert result["status"] == "FAIL_PROVENANCE_CHECKS"
    assert not (tmp_path / "out" / "manifest.json").exists()


def test_duplicate_source_timestamps(tmp_path):
    root = evidence(tmp_path)
    text = (root / v.RAW).read_text()
    b = source(1)
    (root / v.RAW).write_text(
        "[\n" + json.dumps(b, indent=2) + ",\n" + json.dumps(b, indent=2) + "\n]\n"
    )
    (root / v.EXTRACT).write_text(
        "".join(json.dumps(x) + "\n" for _, x in v.raw_2025(root / v.RAW))
    )
    with pytest.raises(v.EvidenceError, match="duplicate source timestamp"):
        v.verified_sources(root)


def test_other_year_values_are_not_decoded(tmp_path):
    p = tmp_path / "raw"
    p.write_text('[\n{\n"TimeStamp":"2026-01-01T00:00:00Z",\n"Close": NOT_JSON\n}\n]\n')
    assert list(v.raw_2025(p)) == []


def test_pilot_overlapping_differences(tmp_path):
    root = evidence(tmp_path, [source(1), source(2), source(3)])
    rows = [json.loads(x) for x in (root / v.RECON).read_text().splitlines()]
    # First both clocks differ, second only capture, third only exchange.
    for i in (0, 1, 2, 5):
        rows[i]["native_ohlc"] = ["101"] * 4
        rows[i]["ohlc_delta"] = ["1"] * 4
    (root / v.RECON).write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out"
    c = v.run(out, root=root, pins=pins(root), baselines={})["counts"]
    assert c["capture_different"] == c["exchange_different"] == 2
    assert (
        c["common_differences"]
        == c["resolved_by_exchange"]
        == c["introduced_by_exchange"]
        == 1
    )
    assert c["capture_different_single"] == 2


def test_cli_failure_is_nonzero(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.cli.check_yank_bar_provenance.run",
        lambda _: (_ for _ in ()).throw(v.EvidenceError("fixture")),
    )
    assert main(["--output-dir", str(tmp_path / "x")]) == 1


def test_missing_start_clock_is_qualified_and_exchange_name_preserved(tmp_path):
    root = evidence(tmp_path)
    rows = [json.loads(x) for x in (root / v.RECON).read_text().splitlines()]
    rows[1]["clock"] = "exchange_diagnostic"
    missing = dict(
        rows[0],
        convention="start",
        native_minute_start=rows[0]["original_label"],
        native_ohlc=None,
        native_T_volume=None,
        ohlc_delta=None,
        volume_delta=None,
        coverage_qualification="missing_native_T",
    )
    rows.append(missing)
    (root / v.RECON).write_text("".join(json.dumps(r) + "\n" for r in rows))
    pilots = v.read_pilots(root)
    assert (
        next(iter(pilots.values()))["exchange"]["source_clock"] == "exchange_diagnostic"
    )
    text = (root / v.CSV).read_text().replace("2025-01-02T", "2025-01-02 ")
    (root / v.CSV).write_text(text)
    assert (
        v.run(tmp_path / "out", root=root, pins=pins(root), baselines={})["counts"][
            "matched_rows"
        ]
        == 1
    )


def test_ambient_decimal_precision_does_not_change_pilot_arithmetic(tmp_path):
    from decimal import localcontext

    root = evidence(tmp_path, [source(1, close="12345.25", volume="300")])
    rows = [json.loads(x) for x in (root / v.RECON).read_text().splitlines()]
    for row in rows:
        row["native_ohlc"] = ["12345.50"] * 4
        row["ohlc_delta"] = ["0.25"] * 4
    (root / v.RECON).write_text("".join(json.dumps(r) + "\n" for r in rows))
    with localcontext() as ctx:
        ctx.prec = 2
        assert (
            next(iter(v.read_pilots(root).values()))["capture"]["delta"]
            == [D("0.25")] * 4
        )
        report = v.run(tmp_path / "out", root=root, pins=pins(root), baselines={})
    assert report["counts"]["capture_different"] == 1


def test_decimal_context_exponents_traps_and_generator_isolation(tmp_path):
    from decimal import getcontext, localcontext, ROUND_DOWN

    root = evidence(tmp_path, [source(1, close="12345.25", volume="300")])
    records = v.verified_sources(root)
    with localcontext() as ambient:
        ambient.prec = 2
        ambient.Emax = 2
        ambient.Emin = -2
        ambient.clamp = 1
        ambient.rounding = ROUND_DOWN
        for signal in ambient.traps:
            ambient.traps[signal] = True
        settings = str(ambient)
        gen = v.reconstruct(records)
        assert next(gen)[1][-1] == D("74071500.00")
        assert getcontext() is ambient and str(getcontext()) == settings
        gen.close()
        assert getcontext() is ambient and str(getcontext()) == settings
        v.read_pilots(root)
        report = v.run(
            tmp_path / "out", root=root, pins=pins(root), baselines={"matched_rows": 1}
        )
        assert report["counts"]["matched_rows"] == 1
        assert getcontext() is ambient and str(getcontext()) == settings


def test_fractional_gap_duration(tmp_path):
    root = evidence(tmp_path, [source(0, volume="12500"), source(2, volume="12500")])
    records = v.verified_sources(root)
    records[1]["time"] = records[0]["time"] + v.timedelta(
        seconds=60, microseconds=250000
    )
    constituents, values = next(v.reconstruct(records))
    row = v.lineage_row(1, constituents, values, {}, pins(root))
    assert row["timing"]["gaps"][0]["seconds"] == 60.25


def test_interrupt_after_report_publication_removes_success(tmp_path, monkeypatch):
    root = evidence(tmp_path)
    original_write = v.write_json

    def interrupted(path, value):
        original_write(path, value)
        if path.name == "report.json" and value["status"] == "PASS_PROVENANCE_CHECKS":
            raise KeyboardInterrupt("injected publication interrupt")

    monkeypatch.setattr(v, "write_json", interrupted)
    with pytest.raises(KeyboardInterrupt):
        v.run(tmp_path / "out", root=root, pins=pins(root), baselines={})
    assert not (tmp_path / "out" / "manifest.json").exists()
    for name in ("report.md", "report.json"):
        assert "PASS_PROVENANCE_CHECKS" not in (tmp_path / "out" / name).read_text()
    assert (
        json.loads((tmp_path / "out" / "report.json").read_text())["status"]
        == "FAIL_PROVENANCE_CHECKS"
    )


def test_published_pilot_semantics_and_baselines(tmp_path):
    root = evidence(tmp_path)
    rows = [json.loads(line) for line in (root / v.RECON).read_text().splitlines()]
    rows[1].update(
        clock="exchange_diagnostic",
        native_ohlc=["99", "103", "98", "102"],
        native_T_volume="25007",
        ohlc_delta=["-1", "3", "-2", "2"],
        volume_delta="7",
        coverage_qualification="contaminated_raw_T",
        gap_reasons=["fixture-gap"],
    )
    (root / v.RECON).write_text("".join(json.dumps(row) + "\n" for row in rows))
    p = pins(root)
    report = v.run(
        tmp_path / "good",
        root=root,
        pins=p,
        baselines={"matched_rows": 1, "capture_exact": 1, "exchange_different": 1},
    )
    assert report["expected_baselines"] == {
        "matched_rows": 1,
        "capture_exact": 1,
        "exchange_different": 1,
    }
    published = json.loads((tmp_path / "good" / "pilot-differences.jsonl").read_text())
    exchange = published["clocks"]["exchange"]
    assert exchange == dict(
        source_clock="exchange_diagnostic",
        reconciliation_path=v.RECON,
        reconciliation_sha256=p[v.RECON],
        reconciliation_line=2,
        original_ohlcv=["100", "100", "100", "100", "25000"],
        native_ohlcv=["99", "103", "98", "102", "25007"],
        ohlc_delta=["-1", "3", "-2", "2"],
        volume_delta="7",
        ohlc_class="different",
        coverage_qualification="contaminated_raw_T",
        gap_reasons=["fixture-gap"],
    )
    capture = published["clocks"]["capture"]
    assert capture["source_clock"] == "capture" and capture["reconciliation_line"] == 1
    assert (
        capture["native_ohlcv"]
        == capture["original_ohlcv"]
        == ["100", "100", "100", "100", "25000"]
    )
    assert capture["ohlc_delta"] == ["0"] * 4 and capture["volume_delta"] == "0"
    assert published["fixed_end_interval"] == [
        "2025-01-02T00:00:00Z",
        "2025-01-02T00:01:00Z",
    ]
    with pytest.raises(v.EvidenceError, match="baseline matched_rows"):
        v.run(tmp_path / "bad", root=root, pins=p, baselines={"matched_rows": 2})
    assert (
        json.loads((tmp_path / "bad" / "report.json").read_text())["status"]
        == "FAIL_PROVENANCE_CHECKS"
    )
    assert not (tmp_path / "bad" / "manifest.json").exists()


@pytest.mark.parametrize(
    "tamper", ["lineage", "csv_numeric", "csv_timestamp", "csv_extra"]
)
def test_independent_checker_rejects_tampering_under_optimized_python(tmp_path, tamper):
    import subprocess
    import sys

    root = evidence(tmp_path)
    out = tmp_path / "out"
    v.run(out, root=root, pins=pins(root), baselines={})
    checker = (
        Path(__file__).resolve().parents[3]
        / "docs/reports/yank-bar-provenance/verify-lineage.py"
    )
    invocation = 'import runpy,sys; from pathlib import Path; runpy.run_path(sys.argv[1])["verify"](Path(sys.argv[2]), root=Path(sys.argv[3]), expected=(1,0,0))'
    command = [
        sys.executable,
        "-O",
        "-c",
        invocation,
        str(checker),
        str(out),
        str(root),
    ]
    good = subprocess.run(command, capture_output=True, text=True)
    assert good.returncode == 0 and "PASS" in good.stdout
    if tamper == "lineage":
        row = json.loads((out / "lineage.jsonl").read_text())
        row["reconstructed"]["notional"] = "1"
        (out / "lineage.jsonl").write_text(json.dumps(row) + "\n")
    else:
        text = (root / v.CSV).read_text()
        if tamper == "csv_numeric":
            text = text.replace("50000000", "50000001")
        elif tamper == "csv_timestamp":
            text = text.replace("00:01:00", "00:02:00")
        else:
            text += text.splitlines()[1] + "\n"
        (root / v.CSV).write_text(text)
    bad = subprocess.run(command, capture_output=True, text=True)
    assert bad.returncode != 0 and "PASS" not in bad.stdout


def test_cli_import_isolated_in_fresh_process():
    import subprocess
    import sys

    checkout = Path(__file__).resolve().parents[3]
    script = r"""
import importlib.abc
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])


def forbidden(name):
    return (
        any(name == prefix or name.startswith(prefix + '.')
            for prefix in ('src.research', 'src.data', 'src.detection'))
        or any(token in name.lower() for token in ('auth', 'trader', 'backtest'))
    )


class BlockLegacyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            raise RuntimeError('Forbidden dependency import: ' + fullname)
        return None


sys.meta_path.insert(0, BlockLegacyImports())
from src.cli import check_yank_bar_provenance as cli
loaded = [name for name in sys.modules if forbidden(name)]
if loaded:
    raise RuntimeError('Legacy modules loaded: ' + repr(loaded))
expected = Path(sys.argv[1]) / 'src/research/yank_bar_provenance/validator.py'
if Path(cli.run.__globals__['__file__']).resolve() != expected.resolve():
    raise RuntimeError('Validator file identity changed')
if not expected.with_name('pins.json').is_file():
    raise RuntimeError('Fixed sibling pins unavailable')
# Preserve the public CLI callable seam without loading a research package.
calls = []
def fake_run(output):
    calls.append(output)
    return {'status': 'PASS_PROVENANCE_CHECKS'}
cli.run = fake_run
if cli.main(['--output-dir', 'fixture-only']) != 0 or calls != ['fixture-only']:
    raise RuntimeError('CLI run callable is no longer replaceable')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(checkout)],
        capture_output=True,
        text=True,
        cwd=checkout,
    )
    assert result.returncode == 0, result.stderr
    assert "PASS_PROVENANCE_CHECKS" in result.stdout
