"""Readiness is conservative and all network/credentials below are synthetic."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
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
        "Reviewed dated session and current explicit successful MNQZ26 request."
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
    report = call_probe(tmp_path, setup_probe, lambda *a: pytest.fail("network"), clock)
    assert report["status"] == "PENDING" and report["token_read"] is False


@pytest.mark.parametrize(
    "mode", ["expired", "mismatch", "401", "429", "timeout", "malformed", "badbars"]
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
        probe.NoRedirect().redirect_request(None, None, 302, None, None, probe.BARS)


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
    assert report["ambiguities"] and len(report["observed_timestamp_gaps"]) == 1
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
        r["verification"] == "UNRESOLVED" for r in report["evidence"]["fixed_reports"]
    )
    with pytest.raises(ValueError):
        evidence.documentary(tmp_path / "data" / "prices.json")


def test_timing_synthetic_only(tmp_path):
    report = timing.run(
        tmp_path / "out", tmp_path / "cache", 2, factory=lambda c: StubProvider()
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
    with pytest.raises(ValueError, match="size cap"):
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
        date=day, open=f"{day}T09:30:00{offset}", close=f"{day}T{close}{offset}"
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
            json.dumps(metadata() if url == probe.METADATA else {"Bars": []}).encode(),
        )

    report = call_probe(tmp_path, setup_probe, get, clock)
    assert report["bar_requests"] < 180
    assert all(b - a >= 11 for a, b in zip(starts, starts[1:]))
    # The fake transport ignores its supplied hard timeout; real guarded_get enforces it.
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
        tmp_path / "out", tmp_path / "cache", 2, factory=factory, clock=clock.mono
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
            json.dumps(metadata() if url == probe.METADATA else {"Bars": []}).encode(),
        )

    report = call_probe(tmp_path, setup_probe, get, clock)
    assert report["status"] == "STOPPED"
    assert report["elapsed_seconds"] == 900
    assert report["bar_requests"] < 180
    assert timeouts[-1] < 11


def test_frozen_hash_drift_refuses(tmp_path, monkeypatch):
    monkeypatch.setitem(evidence.FROZEN, "research/kronos_replay/engine.py", "0" * 64)
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
    assert report["joint_power_lower_bound"] == pytest.approx(0.9644134321138504)
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
        a > b for a, b in zip(report["marginal_powers"], dependent["marginal_powers"])
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
