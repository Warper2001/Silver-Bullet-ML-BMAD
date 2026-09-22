"""Readiness is conservative and all network/credentials below are
synthetic."""

from datetime import datetime, timedelta, timezone
import json
from statistics import NormalDist
import math

import pytest

from research.kronos_readiness import evidence, power, probe, timing
from research.kronos_replay.providers import StubProvider


class Clock:
    def __init__(self, now=None):
        self.seconds = 0.0
        self.now = now or datetime(2026, 9, 22, 14, tzinfo=timezone.utc)

    def mono(self):
        return self.seconds

    def utc(self):
        return self.now + timedelta(seconds=self.seconds)

    def sleep(self, seconds):
        self.seconds += seconds


@pytest.fixture
def setup_probe(tmp_path):
    source = tmp_path / "source.md"
    source.write_text(
        "Reviewed dated session and current explicit successful MNQZ26"
        " request."
    )
    sha = evidence.sha(source.read_bytes())
    plan = {
        "sources": [
            {
                "path": str(source),
                "sha256": sha,
                "date": "2026-09-22",
                "url": "https://example.test/source",
                "claim": "fixture",
            }
        ],
        "session": {
            "verified": True,
            "date": "2026-09-22",
            "open": "2026-09-22T09:30:00-04:00",
            "close": "2026-09-22T16:00:00-04:00",
            "source_sha256": sha,
        },
        "contract": {
            "verified": True,
            "symbol": "MNQZ26",
            "observed_at": "2026-09-22T09:40:00-04:00",
            "source_sha256": sha,
        },
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    token = tmp_path / ".access_token"
    token.write_text("fixture-secret")
    return plan, plan_path, token


def metadata():
    return {
        "Symbols": [
            {
                "Symbol": "MNQZ26",
                "Root": "MNQ",
                "AssetType": "FUTURE",
                "Exchange": "CME",
                "Currency": "USD",
                "ExpirationDate": "2026-12-18T00:00:00Z",
            }
        ]
    }


def call_probe(tmp_path, setup, get, clock=None):
    _, plan, token = setup
    clock = clock or Clock()
    return probe.run(
        plan,
        token,
        tmp_path / "result",
        get=get,
        utc=clock.utc,
        monotonic=clock.mono,
        sleep=clock.sleep,
    )


def test_probe_bounds_order_and_immutable_token(tmp_path, setup_probe):
    calls = []
    clock = Clock()

    def get(url, token, timeout):
        calls.append((url, clock.mono()))
        clock.seconds += 0.1
        result = (
            metadata()
            if url == probe.METADATA
            else {
                "Bars": [
                    {
                        "TimeStamp": "2026-09-22T13:59:00Z",
                        "BarStatus": "Open",
                        "Close": len(calls),
                    }
                ]
            }
        )
        return 200, json.dumps(result).encode()

    token_before = setup_probe[2].read_bytes()
    report = call_probe(tmp_path, setup_probe, get, clock)
    assert report["bar_requests"] == 180
    assert calls[0][0] == probe.METADATA
    assert all(b[1] - a[1] >= 5 - 1e-9 for a, b in zip(calls[1:], calls[2:]))
    assert report["elapsed_seconds"] <= 900
    assert setup_probe[2].read_bytes() == token_before
    assert len(report["description"]["revisions"]) == 179
    assert len(report["description"]["explicitly_open_bars"]) == 180
    manifest = json.loads((tmp_path / "result/COMPLETE.json").read_text())
    assert all(
        evidence.sha((tmp_path / "result" / name).read_bytes()) == sha
        for name, sha in manifest["sha256"].items()
    )
    with pytest.raises(FileExistsError):
        evidence.write_json(tmp_path / "result/observation-0000.json", {})


@pytest.mark.parametrize(
    "change", ["late", "missing_source", "naive", "unverified", "bad_close"]
)
def test_pending_never_reads_token(tmp_path, setup_probe, change):
    plan, path, token = setup_probe
    clock = Clock()
    if change == "late":
        clock.now = datetime(2026, 9, 22, 19, 50, tzinfo=timezone.utc)
    if change == "missing_source":
        plan["sources"][0]["sha256"] = "0" * 64
    if change == "naive":
        plan["session"]["open"] = "2026-09-22T09:30:00"
    if change == "unverified":
        plan["session"]["verified"] = False
    if change == "bad_close":
        plan["session"]["close"] = "2026-09-22T16:59:00-04:00"
    path.write_text(json.dumps(plan))
    token.unlink()
    report = call_probe(
        tmp_path, setup_probe, lambda *a: pytest.fail("network"), clock
    )
    assert report["status"] == "PENDING" and report["token_read"] is False


@pytest.mark.parametrize(
    "mode",
    ["expired", "mismatch", "401", "429", "timeout", "malformed", "badbars"],
)
def test_probe_stops_safely(tmp_path, setup_probe, mode):
    calls = []

    def get(url, token, timeout):
        calls.append(url)
        if mode == "timeout":
            raise TimeoutError(token)
        if mode in ("401", "429"):
            return int(mode), token.encode()
        if mode == "malformed":
            return 200, b"[]"
        result = metadata()
        if mode == "expired":
            result["Symbols"][0]["ExpirationDate"] = "2025-01-01T00:00:00Z"
        if mode == "mismatch":
            result["Symbols"][0]["Symbol"] = "OTHER"
        if url == probe.BARS:
            result = {"Bars": [[], {"TimeStamp": []}]}
        return 200, json.dumps(result).encode()

    report = call_probe(tmp_path, setup_probe, get)
    assert report["status"] == "STOPPED"
    assert len(calls) == (2 if mode == "badbars" else 1)
    assert all(
        "fixture-secret" not in p.read_text()
        for p in (tmp_path / "result").glob("*.json")
    )


@pytest.mark.parametrize(
    "url,method",
    [
        ("https://api.tradestation.com/v3/marketdata/symbols/MNQZ26", "GET"),
        (probe.METADATA + "?extra=1", "GET"),
        (probe.BARS, "POST"),
        ("https://sim-api.tradestation.com/v3/orderexecution/orders", "GET"),
    ],
)
def test_endpoint_guard(url, method):
    with pytest.raises(ValueError):
        probe.guarded_get(url, "unused", 1, method)


def test_redaction_and_redirect():
    body = b'{"access_token":"other", "Bars":[],"note":"fixture-secret"}'
    clean = probe.sanitize(body, "fixture-secret")
    assert b"other" not in clean and b"fixture-secret" not in clean
    with pytest.raises(ValueError):
        probe.NoRedirect().redirect_request(
            None, None, 302, None, None, probe.BARS
        )


def test_descriptive_ambiguity_gap():
    rows = [
        {"TimeStamp": "2026-09-22T14:00:00Z", "BarStatus": "Closed"},
        {"TimeStamp": "2026-09-22T14:02:00Z"},
        {"TimeStamp": "2026-09-22T14:03:00", "BarStatus": "Unknown"},
    ]
    report = probe.describe(
        [
            {
                "url": probe.BARS,
                "sequence": 0,
                "receipt_utc": "2026-09-22T14:04:00Z",
                "response": {"Bars": rows},
            }
        ]
    )
    assert (
        report["ambiguities"] and len(report["observed_timestamp_gaps"]) == 1
    )
    assert report["completion_inferred_from_age"] is False


def test_power_and_paired_covariance():
    row = power.standardized(100, 1.5)
    assert row["detectable_standardized_effect"] == pytest.approx(
        (NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.9)) * 1.5 / 10
    )
    refs = {
        k: "independent-document"
        for k in (
            "k_effect",
            "incremental_effect",
            "k_variance",
            "m_variance",
            "covariance",
            "dependence",
        )
    }
    r = power.dollar_scenario(
        100,
        1,
        k_effect=1,
        incremental_effect=1,
        k_variance=4,
        m_variance=9,
        covariance=3,
        independent_evidence=refs,
    )
    assert r["paired_variance"] == 7 and r["strategy_test_permitted"] is False
    assert r["actual_power"] == "UNASSESSABLE"
    for n, i in ((0, 1), (True, 1), (100, 0.9), (100, math.inf)):
        with pytest.raises(ValueError):
            power.standardized(n, i)
    with pytest.raises(ValueError):
        power.dollar_scenario(
            100,
            1,
            k_effect=1,
            incremental_effect=1,
            k_variance=4,
            m_variance=9,
            covariance=3,
            independent_evidence={},
        )


def test_assess_missing_evidence_holds(tmp_path):
    pack = tmp_path / "pack.json"
    pack.write_text('{"sources":[]}')
    report = evidence.assess(tmp_path, pack, tmp_path / "out")
    assert (
        report["status"] == "HOLD_EVALUATION"
        and report["power_verdict"] == "UNASSESSABLE"
    )
    assert len(report["blockers"]) == 8 and report["admitted_sessions"] == 0
    assert all(
        r["verification"] == "UNRESOLVED"
        for r in report["evidence"]["fixed_reports"]
    )
    with pytest.raises(ValueError):
        evidence.documentary(tmp_path / "data" / "prices.json")


def test_timing_synthetic_only(tmp_path):
    report = timing.run(
        tmp_path / "out",
        tmp_path / "cache",
        2,
        factory=lambda c: StubProvider(),
    )
    assert report["status"] == "TIMING_COMPLETED"
    assert len(report["three_seed_decision_seconds"]) == 2
    assert report["latency_adopted_seconds"] is None


def test_total_deadline_restores_alarm():
    import signal
    import time

    before = signal.getsignal(signal.SIGALRM)
    with pytest.raises(TimeoutError):
        with probe.hard_timeout(0.01):
            time.sleep(0.1)
    assert signal.getsignal(signal.SIGALRM) == before
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_guard_disables_proxy_and_caps_body(monkeypatch):
    captured = {}

    class Response:
        code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            captured["limit"] = limit
            return b"x" * limit

    class Opener:
        def open(self, request, timeout):
            captured["method"] = request.get_method()
            return Response()

    def build(*handlers):
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(probe.urllib.request, "build_opener", build)
    with pytest.raises(ValueError, match="size"):
        probe.guarded_get(probe.BARS, "test-token", 1)
    assert captured["handlers"][0].proxies == {}
    assert isinstance(captured["handlers"][1], probe.NoRedirect)
    assert captured["method"] == "GET"
    assert captured["limit"] == probe.MAX_BYTES + 1


@pytest.mark.parametrize(
    "day,offset,close",
    [
        ("2026-03-06", "-05:00", "16:00:00"),
        ("2026-03-09", "-04:00", "13:00:00"),
    ],
)
def test_dated_dst_and_early_close(setup_probe, day, offset, close):
    plan, _, _ = setup_probe
    plan["session"].update(
        date=day,
        open=f"{day}T09:30:00{offset}",
        close=f"{day}T{close}{offset}",
    )
    plan["contract"]["observed_at"] = f"{day}T09:35:00{offset}"
    now = probe.aware(f"{day}T10:00:00{offset}")
    register = [
        {
            "sha256": plan["session"]["source_sha256"],
            "verification": "HASH_VERIFIED_ONLY",
        }
    ]
    opening, end = probe.preconditions(plan, now, register)
    assert opening.hour == (14 if offset == "-05:00" else 13)
    assert end.hour == (21 if offset == "-05:00" else 17)
    with pytest.raises(ValueError):
        probe.preconditions(plan, end - timedelta(minutes=14), register)


def test_slow_requests_never_catch_up(tmp_path, setup_probe):
    clock = Clock()
    starts = []

    def get(url, token, timeout):
        starts.append(clock.mono())
        clock.seconds += 11
        return (
            200,
            json.dumps(
                metadata() if url == probe.METADATA else {"Bars": []}
            ).encode(),
        )

    report = call_probe(tmp_path, setup_probe, get, clock)
    assert report["bar_requests"] < 180
    assert all(b - a >= 11 for a, b in zip(starts, starts[1:]))
    # The fake transport ignores its supplied hard timeout; real guarded_get
    # enforces it.
    assert max(starts) < 900


def test_source_tampering_is_unresolved(tmp_path):
    path = tmp_path / "source.md"
    path.write_text("original")
    entry = {
        "path": str(path),
        "sha256": evidence.sha(path.read_bytes()),
        "date": "2026-09-22",
        "url": "https://example.test",
        "claim": "claim",
    }
    path.write_text("tampered")
    assert (
        evidence.sources({"sources": [entry]}, tmp_path)[0]["verification"]
        == "UNRESOLVED"
    )


def test_timing_context_is_eligible_and_separate(tmp_path):
    clock = Clock()

    class Provider(StubProvider):
        def forecast(self, context, future):
            assert len(context) == 128
            assert context.index[-1].isoformat() == "2026-03-09T09:45:00-04:00"
            assert future[-1].isoformat() == "2026-03-09T10:45:00-04:00"
            clock.seconds += 9
            return super().forecast(context, future)

    def factory(cache):
        clock.seconds += 12
        return Provider()

    report = timing.run(
        tmp_path / "out",
        tmp_path / "cache",
        2,
        factory=factory,
        clock=clock.mono,
    )
    assert report["startup_seconds"] == 12
    assert report["three_seed_decision_seconds"] == [9, 9]
    assert report["targets"] == [1, 1]


def test_remaining_timeout_and_duration_cap(tmp_path, setup_probe):
    clock = Clock()
    timeouts = []

    def get(url, token, timeout):
        assert timeout <= min(15.0, 900 - clock.mono())
        timeouts.append(timeout)
        clock.seconds += min(11, timeout)
        if timeout < 11:
            raise TimeoutError("bounded")
        return (
            200,
            json.dumps(
                metadata() if url == probe.METADATA else {"Bars": []}
            ).encode(),
        )

    report = call_probe(tmp_path, setup_probe, get, clock)
    assert report["status"] == "STOPPED"
    assert report["elapsed_seconds"] == 900
    assert report["bar_requests"] < 180
    assert timeouts[-1] < 11


def test_frozen_hash_drift_refuses(tmp_path, monkeypatch):
    monkeypatch.setitem(
        evidence.FROZEN, "research/kronos_replay/engine.py", "0" * 64
    )
    with pytest.raises(ValueError, match="frozen"):
        evidence.new_output(tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_paired_power_reference_values():
    refs = {
        k: "independent-reviewed-document"
        for k in (
            "k_effect",
            "incremental_effect",
            "k_variance",
            "m_variance",
            "covariance",
            "dependence",
        )
    }
    report = power.dollar_scenario(
        100,
        1,
        k_effect=1,
        incremental_effect=1,
        k_variance=4,
        m_variance=9,
        covariance=3,
        independent_evidence=refs,
    )
    # Independently checked with scipy.stats.norm, z(.975)=1.959963984540054.
    assert report["marginal_powers"] == pytest.approx(
        [0.9988172507018026, 0.9655961814120477]
    )
    assert report["joint_power_lower_bound"] == pytest.approx(
        0.9644134321138504
    )
    dependent = power.dollar_scenario(
        100,
        2,
        k_effect=1,
        incremental_effect=1,
        k_variance=4,
        m_variance=9,
        covariance=3,
        independent_evidence=refs,
    )
    assert all(
        a > b
        for a, b in zip(
            report["marginal_powers"], dependent["marginal_powers"]
        )
    )


def test_cli_failure_never_echoes_payload(tmp_path, capsys):
    from research.kronos_readiness.__main__ import main

    path = tmp_path / "invalid.json"
    path.write_text("secret credential invalid document")
    assert (
        main(
            [
                "probe",
                "--plan",
                str(path),
                "--token-path",
                str(tmp_path / ".access_token"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
    text = capsys.readouterr().out
    assert "secret credential" not in text
    assert json.loads(text)["strategy_test_permitted"] is False


def test_receipt_precedes_expensive_processing(
    tmp_path, setup_probe, monkeypatch
):
    clock = Clock()
    original = probe.sanitize

    def sanitize(body, token):
        clock.seconds += 100
        return original(body, token)

    def get(url, token, timeout):
        clock.seconds += 1
        return 401, b"{}"

    monkeypatch.setattr(probe, "sanitize", sanitize)
    call_probe(tmp_path, setup_probe, get, clock)
    record = json.loads(
        (tmp_path / "result/observation-0000.json").read_text()
    )
    assert record["receipt_elapsed_seconds"] == 1
    assert (
        record["receipt_utc"] == (clock.now + timedelta(seconds=1)).isoformat()
    )


def test_unicode_escaped_secret_keys_and_values():
    body = b'{"\\u0073ecret-key":"echo \\u0073ecret value"}'
    clean = probe.sanitize(body, "secret")
    result = json.loads(clean)
    assert result == {"[REDACTED]-key": "echo [REDACTED] value"}


@pytest.mark.parametrize("folder", ["data", "logs", "cache", ".venv-research"])
def test_forbidden_token_path_never_opened(
    tmp_path, setup_probe, monkeypatch, folder
):
    from pathlib import Path

    plan, plan_path, _ = setup_probe
    forbidden = tmp_path / folder / ".access_token"
    monkeypatch.setattr(
        Path, "read_text", lambda *a, **k: pytest.fail("credential opened")
    )
    clock = Clock()
    report = probe.run(
        plan_path,
        forbidden,
        tmp_path / "result",
        get=lambda *a: pytest.fail("network"),
        utc=clock.utc,
        monotonic=clock.mono,
        sleep=clock.sleep,
    )
    assert report["status"] == "BLOCKED"
    assert report["token_read"] is False


def test_token_alias_refused(tmp_path, setup_probe):
    _, plan, token = setup_probe
    alias = tmp_path / "alias" / ".access_token"
    alias.parent.mkdir()
    alias.symlink_to(token)
    clock = Clock()
    report = probe.run(
        plan,
        alias,
        tmp_path / "result",
        utc=clock.utc,
        get=lambda *a: pytest.fail("network"),
    )
    assert report["token_read"] is False


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonstandard_json_preserved_and_stopped(tmp_path, setup_probe, number):
    body = ('{"n":' + number + "}").encode()
    report = call_probe(tmp_path, setup_probe, lambda *a: (200, body))
    assert report["status"] == "STOPPED"
    record = json.loads(
        (tmp_path / "result/observation-0000.json").read_text()
    )
    assert record["error_category"] == "invalid-response"
    assert "raw_response_base64" in record
    assert (tmp_path / "result/COMPLETE.json").exists()


def test_nonfinite_write_creates_no_partial_file(tmp_path):
    with pytest.raises(ValueError):
        evidence.write_json(tmp_path / "bad.json", {"n": math.inf})
    assert not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize(
    "kind",
    [
        "timeout",
        "auth",
        "throttled",
        "redirect",
        "size",
        "transport",
        "invalid-response",
    ],
)
def test_transport_failure_categories(tmp_path, setup_probe, kind):
    def get(*args):
        if kind == "auth":
            return 401, b"{}"
        if kind == "throttled":
            return 429, b"{}"
        if kind == "invalid-response":
            return 200, b"not json"
        if kind == "timeout":
            raise TimeoutError("secret text")
        if kind == "transport":
            raise OSError("secret text")
        raise probe.ProbeFailure(kind)

    call_probe(tmp_path, setup_probe, get)
    record = json.loads(
        (tmp_path / "result/observation-0000.json").read_text()
    )
    assert record["error_category"] == kind
    assert "secret text" not in json.dumps(record)


@pytest.mark.parametrize(
    "deltas,target", [([3, -1, 1], 1), ([-3, 1, -1], -1), ([-1, 0, 1], 0)]
)
def test_timing_mean_terminal_policy(tmp_path, deltas, target):
    class Provider(StubProvider):
        def forecast(self, context, future):
            paths = super().forecast(context, future)
            for path, delta in zip(paths, deltas):
                close = context.close.iloc[-1] + delta
                path.loc[:, ["open", "close"]] = close
                path.loc[:, "high"] = close + 1
                path.loc[:, "low"] = close - 1
            return paths

    report = timing.run(
        tmp_path / "out", tmp_path / "cache", 1, factory=lambda c: Provider()
    )
    assert report["targets"] == [target]


@pytest.mark.parametrize("failure", ["partial", "schema", "geometry"])
def test_timing_retains_invalid_forecast_paths(tmp_path, failure):
    from research.kronos_replay.providers import ForecastFailure

    class Provider(StubProvider):
        def forecast(self, context, future):
            paths = super().forecast(context, future)
            if failure == "partial":
                raise ForecastFailure("safe-test", paths[:1])
            if failure == "schema":
                paths[0] = paths[0].drop(columns=["amount"])
            if failure == "geometry":
                paths[0].loc[:, "high"] = 1
            return paths

    report = timing.run(
        tmp_path / "out", tmp_path / "cache", 1, factory=lambda c: Provider()
    )
    assert report["status"] != "TIMING_COMPLETED"
    assert report["error_category"] in ("inference", "invalid-output")
    assert len(list((tmp_path / "out").glob("decision-*.csv"))) == (
        1 if failure == "partial" else 3
    )


def test_timing_invalid_cli_nonzero(tmp_path, monkeypatch):
    from research.kronos_readiness.__main__ import main

    monkeypatch.setattr(
        timing, "run", lambda *a: {"status": "TIMING_INVALID_OUTPUT"}
    )
    assert (
        main(
            [
                "timing",
                "--cache",
                str(tmp_path / "cache"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    "dependency",
    [
        "tools/kronos_inference_pilot.py",
        "research/kronos_replay/fixtures.py",
        "tools/trading_model_readiness.py",
    ],
)
def test_frozen_dependency_drift(tmp_path, monkeypatch, dependency):
    monkeypatch.setitem(evidence.FROZEN, dependency, "0" * 64)
    with pytest.raises(ValueError, match="frozen"):
        evidence.new_output(tmp_path / "out")


def test_assessment_snapshots_exact_category_citation(tmp_path, monkeypatch):
    report_path = tmp_path / evidence.AUDIT
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps({"data_gaps": ["SOURCE_PROVENANCE_NOT_ADMITTED"]})
    )
    digest = evidence.sha(report_path.read_bytes())
    monkeypatch.setattr(evidence, "EVIDENCE", {evidence.AUDIT: digest})
    source = tmp_path / "source.md"
    source.write_text("reviewer assertion")
    pack = tmp_path / "pack.json"
    pack.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "path": str(source),
                        "sha256": evidence.sha(source.read_bytes()),
                        "date": "2026-09-22",
                        "url": "https://example.test",
                        "claim": "assertion",
                        "category": "costs",
                    }
                ]
            }
        )
    )
    result = evidence.assess(tmp_path, pack, tmp_path / "out")
    categories = result["evidence"]["categories"]
    citation = categories["historical_acquisition"]["citations"][0]
    assert citation["json_pointer"] == "/data_gaps/0"
    assert categories["calendar"]["citations"] == []
    assert categories["costs"]["status"] == "UNRESOLVED"
    assert categories["costs"]["reviewer_assertions"]
    assert (
        evidence.sha((tmp_path / "out" / citation["path"]).read_bytes())
        == digest
    )
    assert len(list((tmp_path / "out/sources").iterdir())) == 2


def test_snapshot_rechecks_source_bytes(tmp_path, monkeypatch):
    source = tmp_path / "source.md"
    source.write_text("original")
    digest = evidence.sha(source.read_bytes())
    pack = tmp_path / "pack.json"
    pack.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "path": str(source),
                        "sha256": digest,
                        "date": "2026-09-22",
                        "url": "https://example.test",
                        "claim": "claim",
                    }
                ]
            }
        )
    )
    original = evidence.documentary
    reads = 0

    def changed(path):
        nonlocal reads
        data = original(path)
        if path == source:
            reads += 1
            if reads == 1:
                source.write_text("changed")
        return data

    monkeypatch.setattr(evidence, "documentary", changed)
    with pytest.raises(ValueError, match="snapshot"):
        evidence.assess(tmp_path, pack, tmp_path / "out")


def test_extreme_power_inputs_refused():
    for n, inflation in ((1, 1e308), (10**1000, 1)):
        with pytest.raises(ValueError):
            power.standardized(n, inflation)
    refs = {
        k: "reviewed"
        for k in (
            "k_effect",
            "incremental_effect",
            "k_variance",
            "m_variance",
            "covariance",
            "dependence",
        )
    }
    with pytest.raises(ValueError):
        power.dollar_scenario(
            100,
            1,
            k_effect=1e308,
            incremental_effect=1,
            k_variance=1e-300,
            m_variance=1,
            covariance=0,
            independent_evidence=refs,
        )


def test_nested_response_preserves_receipt_without_secret(
    tmp_path, setup_probe
):
    clock = Clock()
    secret = setup_probe[2].read_text()
    body = b"[" * 1100 + json.dumps(secret).encode() + b"]" * 1100

    def get(*args):
        clock.seconds += 1
        return 200, body

    report = call_probe(tmp_path, setup_probe, get, clock)
    assert report["status"] == "STOPPED"
    assert report["error_category"] == "invalid-response"
    record = json.loads(
        (tmp_path / "result/observation-0000.json").read_text()
    )
    assert record["request_elapsed_seconds"] == 0
    assert record["receipt_elapsed_seconds"] == 1
    assert record["http_status"] == 200
    assert record["error_category"] == "invalid-response"
    assert record["body_omitted_reason"]
    assert "raw_response_base64" not in record
    assert "response" not in record
    assert (tmp_path / "result/COMPLETE.json").exists()
    for path in (tmp_path / "result").rglob("*"):
        if path.is_file():
            assert secret not in path.read_text()
