"""Small artifact checks for independent verifier failures under optimized Python."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from test_native_minute import ROOT, b, r, runner, synthetic_native_bars

REPORTS = ROOT / "docs/reports/yank-native-minute"


@pytest.fixture(scope="module")
def published():
    bars = synthetic_native_bars()
    for bar in bars:
        bar["incomplete_event"] = False
    result = r.replay(bars, json.loads((runner.HERE / "pins.json").read_text()), set())
    return bars, result


def invoke(script, path):
    return subprocess.run(
        [sys.executable, "-O", str(REPORTS / script), str(path)],
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "gate_availability",
        "event_interval",
        "event_availability",
        "terminal_availability",
        "deleted_marks",
    ],
)
def test_published_verifier_independently_checks_timing_and_marks(
    tmp_path, published, mutation
):
    bars, original = published
    result = copy.deepcopy(original)
    arm = result["arms"]["no-ml"]
    if mutation == "gate_availability":
        arm["gates"][0]["decision_available_ns"] -= 1
    elif mutation == "event_interval":
        arm["events"][0]["interval_start_ns"] -= 1
    elif mutation == "event_availability":
        arm["events"][0]["bar_available_ns"] -= 1
    elif mutation == "terminal_availability":
        arm["terminal"]["mark_available_ns"] -= 1
    elif mutation == "deleted_marks":
        arm["events"] = [e for e in arm["events"] if e["kind"] != "MARK"]
        for i, event in enumerate(arm["events"]):
            event["sequence"] = i
    (tmp_path / "bars.jsonl").write_bytes(
        b"".join(runner.canonical(bar) for bar in bars)
    )
    path = tmp_path / "replay.json"
    path.write_bytes(runner.canonical(result))
    checked = invoke("verify-published-replay.py", path)
    assert (checked.returncode == 0) == (mutation is None), (
        checked.stdout + checked.stderr
    )


@pytest.mark.parametrize("mutation", ["availability", "incomplete"])
def test_native_oracle_rejects_invalid_completion_under_optimized_python(mutation):
    code = """
import importlib.util,sys
spec=importlib.util.spec_from_file_location('oracle',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
bar={'start_ns':0,'end_ns':m.M,'availability_ns':m.M,'incomplete_event':False}
if sys.argv[2]=='availability':bar['availability_ns']-=1
else:bar['incomplete_event']=True
m.validate_capture_bar(bar)
"""
    checked = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            code,
            str(REPORTS / "verify-native-oracle.py"),
            mutation,
        ],
        text=True,
        capture_output=True,
    )
    assert checked.returncode != 0 and "ValueError" in checked.stderr


@pytest.mark.parametrize(
    "transition,expected", [(0, "NONTRADING"), (30, "MIXED"), (60, "TRADING")]
)
def test_independent_official_status_interval_boundaries(transition, expected):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "oracle", REPORTS / "verify-native-oracle.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        module.interval_status([(-1, True), (transition, False)], 0, 60)[2] == expected
    )
