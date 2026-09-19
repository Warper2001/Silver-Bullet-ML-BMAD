import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
import pytest
from research.project_goals.capture import ReadOnlyBroker, immutable, audit_health
from research.project_goals.daily import (
    economics,
    flat_reasons,
    observation,
    reconcile_day,
    scheduled_days,
    session_bounds,
)
from src.research.broker_fill_evidence import matching_fills, persist_fill


def fill(**kwargs):
    return dict(
        dict(
            id=1,
            accountId=42,
            contractId="MNQ",
            orderId=5,
            side=0,
            size=1,
            price=20000,
            fees=0.5,
            commissions=0.2,
            profitAndLoss=10,
            voided=False,
            creationTimestamp="2026-09-18T14:00:00+00:00",
        ),
        **kwargs,
    )


def submission():
    return dict(accountId=42, contractId="MNQ", orderId=5, side=0, size=2)


def test_exact_individual_partial_fills_and_conflicts(tmp_path):
    one, two = fill(), fill(id=2)
    rows, issues = matching_fills([fill(orderId=99), one, two, one], submission())
    assert rows == [one, two] and not issues
    assert persist_fill(tmp_path, submission(), one) == "new"
    assert persist_fill(tmp_path, submission(), one) == "duplicate"
    assert persist_fill(tmp_path, submission(), fill(price=20001)) == "conflict"
    assert matching_fills([one, fill(price=20001)], submission())[0] == []
    assert matching_fills([fill(voided=True)], submission())[0] == []
    assert matching_fills([one, two, fill(id=3)], submission())[0] == []


@pytest.mark.asyncio
async def test_adapter_refuses_execution_before_auth():
    broker = ReadOnlyBroker(None, None)
    with pytest.raises(ValueError, match="allowlisted"):
        await broker.request("Order/place", {})


def test_unknown_costs_and_no_double_slippage():
    assert economics([fill(), fill()])["net"] == 9.3
    assert economics([fill(fees=None)])["net"] is None
    assert economics([fill(), fill(price=19000)])["net"] is None
    assert economics([fill(voided=True)])["net"] == 0


def snapshot():
    local = {}
    for strategy, state in [
        (
            "MIM",
            dict(position=0, cat_stop_id=None, symbol="MNQ", saved_at="2026-09-18"),
        ),
        ("YANK", dict(daily_pnl=0, daily_halted=False, last_trading_date="2026-09-18")),
    ]:
        local[strategy] = dict(
            state=state,
            stable=True,
            mtime_ns=1789682400000000000,
            producer=dict(
                identity={"pid": 1},
                account="42",
                matches_source=True,
                started_at="2026-09-17T00:00:00+00:00",
            ),
        )
    return dict(
        account="42",
        started_at="2026-09-18T12:00:00+00:00",
        observed_at="2026-09-18T12:00:01+00:00",
        local_before=deepcopy(local),
        local_after=deepcopy(local),
        requests=[
            dict(
                endpoint="Account/search",
                ok=True,
                response={"accounts": [{"id": 42, "balance": 50000}]},
            ),
            dict(endpoint="Order/searchOpen", ok=True, response={"orders": []}),
            dict(endpoint="Position/searchOpen", ok=True, response={"positions": []}),
        ],
    )


def test_flat_requires_individual_stable_account_provenance():
    data = snapshot()
    assert not flat_reasons(data, 42)
    assert flat_reasons(data, 43)
    data["local_before"]["MIM"]["state"]["position"] = 1
    data["local_after"]["MIM"]["state"]["position"] = 1
    data["local_before"]["YANK"]["state"]["direction"] = "short"
    data["local_after"]["YANK"]["state"]["direction"] = "short"
    assert flat_reasons(data, 42)
    data = snapshot()
    data["local_after"]["MIM"]["producer"]["account"] = "99"
    assert flat_reasons(data, 42)


def test_observation_requires_repeated_flatness_and_no_reset_claim():
    one, two = snapshot(), snapshot()
    two.update(
        started_at="2026-09-18T12:01:00+00:00", observed_at="2026-09-18T12:01:01+00:00"
    )
    assert observation([("a", one)], 42)["status"] == "PENDING_FLAT_EVIDENCE"
    obs = observation([("a", one), ("b", two)], 42)
    assert obs["status"] == "STARTED" and obs["account_reset_epoch"] is None


def test_deterministic_reports_corrections_and_unknown_coverage(tmp_path):
    data = snapshot()
    immutable(tmp_path / "snapshots", data)
    first = reconcile_day(tmp_path, 42, "2026-09-18")
    assert first["net_total"] is None and not first["zero_trade_session"]
    assert reconcile_day(tmp_path, 42, "2026-09-18") == first
    assert len(list((tmp_path / "reports/2026-09-18").glob("*.json"))) == 1
    corrected = deepcopy(data)
    corrected["observed_at"] = "2026-09-18T12:02:00+00:00"
    corrected["requests"][0]["response"]["accounts"][0]["balance"] = 49999
    immutable(tmp_path / "snapshots", corrected)
    reconcile_day(tmp_path, 42, "2026-09-18")
    assert len(list((tmp_path / "reports/2026-09-18").glob("*.json"))) == 2


def test_schedule_dst_and_weekends():
    assert scheduled_days(datetime(2026, 9, 21, 12, 30, tzinfo=timezone.utc)) == [
        "2026-09-18"
    ]
    assert "2026-09-21" in scheduled_days(
        datetime(2026, 9, 21, 20, 10, tzinfo=timezone.utc)
    )
    assert session_bounds("2026-12-01")[1].utcoffset().total_seconds() == -18000
    assert audit_health("/nonexistent")["status"] == "MISSING"


def test_cost_unknown_retains_execution_and_known_components():
    result = economics([fill(commissions=None)])
    assert len(result["fills"]) == 1
    assert result["gross"] == 10
    assert result["fees"] == 0.5
    assert result["commissions"] is None and result["net"] is None


def test_schedule_has_exact_due_boundaries():
    from research.project_goals.daily import scheduled_events

    before = scheduled_events(datetime(2026, 9, 21, 12, 29, tzinfo=timezone.utc))
    after = scheduled_events(datetime(2026, 9, 21, 12, 30, tzinfo=timezone.utc))
    assert ("2026-09-21-refresh", "2026-09-18") not in before
    assert ("2026-09-21-refresh", "2026-09-18") in after
    assert not any(event == "2026-09-21-close" for event, _ in after)


def test_complete_zero_session_requires_full_bracketed_evidence(tmp_path):
    def make(at):
        result = snapshot()
        result["started_at"] = at
        result["observed_at"] = at
        for endpoint, field in [("Trade/search", "trades"), ("Order/search", "orders")]:
            result["requests"].append(
                dict(
                    endpoint=endpoint,
                    ok=True,
                    request=dict(
                        startTimestamp="2026-09-17T00:00:00+00:00", endTimestamp=at
                    ),
                    response={field: []},
                )
            )
        return result

    for at in [
        "2026-09-17T21:57:00+00:00",
        "2026-09-17T21:58:00+00:00",
        "2026-09-17T21:59:00+00:00",
        "2026-09-18T21:00:01+00:00",
    ]:
        immutable(tmp_path / "snapshots", make(at))
    report = reconcile_day(tmp_path, 42, "2026-09-18")
    assert report["status"] == "COMPLETE", report["issues"]
    assert report["zero_trade_session"] and report["net_total"] == 0


def test_interruption_does_not_heal_on_return_to_balance(tmp_path):
    for index, value in enumerate([50000, 50000, 50001, 50000]):
        result = snapshot()
        at = f"2026-09-18T12:0{index}:00+00:00"
        result["started_at"] = result["observed_at"] = at
        result["requests"][0]["response"]["accounts"][0]["balance"] = value
        result["requests"].append(
            dict(
                endpoint="Trade/search",
                ok=True,
                request=dict(
                    startTimestamp="2026-09-18T00:00:00+00:00", endTimestamp=at
                ),
                response={"trades": []},
            )
        )
        immutable(tmp_path / "snapshots", result)
    report = reconcile_day(tmp_path, 42, "2026-09-18")
    assert report["observation"]["status"] == "INTERRUPTED"
    assert report["balance_bridge"]["residuals"]


@pytest.mark.asyncio
async def test_real_capture_windows_cover_balance_and_persist_observation(
    tmp_path, monkeypatch
):
    from research.project_goals import capture
    from research.project_goals.daily import account_observed, covering

    monkeypatch.setattr(
        capture, "local_evidence", lambda root: snapshot()["local_after"]
    )

    class Broker:
        async def request(self, endpoint, payload):
            at = datetime.now(timezone.utc).isoformat()
            field = capture.ENDPOINTS[endpoint]
            rows = [{"id": 42, "balance": 50000}] if field == "accounts" else []
            return dict(
                endpoint=endpoint,
                request=payload,
                observed_at=at,
                started_at=at,
                ok=True,
                response={field: rows, "success": True},
            )

    first, _ = await capture.capture_once(Broker(), tmp_path, tmp_path, "42")
    second, _ = await capture.capture_once(Broker(), tmp_path, tmp_path, "42")
    assert covering(
        second, "Trade/search", account_observed(first), account_observed(second)
    )
    observed = json.loads((tmp_path / "observation.json").read_text())
    assert observed["status"] == "STARTED"
    assert capture.audit_health(tmp_path)["observation"]["status"] == "STARTED"
    assert len(list((tmp_path / "sources").glob("*.json"))) == 2


def traded_session(tmp_path, *, missing_exit=False):
    fills = [
        fill(id=1, orderId=11, side=0, size="1", profitAndLoss=0),
        fill(id=2, orderId=12, side=1, creationTimestamp="2026-09-18T15:00:00+00:00"),
        fill(id=3, orderId=13, side=1, contractId="MES", profitAndLoss=0),
        fill(
            id=4,
            orderId=14,
            side=0,
            contractId="MES",
            creationTimestamp="2026-09-18T15:00:00+00:00",
        ),
    ]
    orders = [
        dict(
            id=f["orderId"],
            accountId=42,
            contractId=f["contractId"],
            side=f["side"],
            fillVolume=1,
            creationTimestamp=f["creationTimestamp"],
        )
        for f in fills
    ]

    def make(at, value, active=False):
        data = snapshot()
        data.update(started_at=at, observed_at=at)
        data["requests"][0].update(started_at=at, observed_at=at)
        data["requests"][0]["response"]["accounts"][0]["balance"] = value
        data["local_after"]["mim_submissions"] = [
            dict(accountId=42, orderId=oid, contractId="MNQ") for oid in (11, 12)
        ]
        if active:
            for where in ("local_before", "local_after"):
                data[where]["YANK"]["state"].update(
                    direction="short", sim_entry_order_id=13, sim_tp_order_id=14
                )
        rows = [
            f
            for f in fills
            if f["creationTimestamp"] < at and (not missing_exit or f["id"] != 4)
        ]
        for endpoint, field, records in [
            ("Trade/search", "trades", rows),
            (
                "Order/search",
                "orders",
                [o for o in orders if o["creationTimestamp"] < at],
            ),
        ]:
            data["requests"].append(
                dict(
                    endpoint=endpoint,
                    ok=True,
                    request=dict(
                        startTimestamp="2026-09-17T00:00:00+00:00", endTimestamp=at
                    ),
                    response={field: deepcopy(records)},
                )
            )
        return data

    captures = [
        make("2026-09-17T21:57:00+00:00", 50000),
        make("2026-09-17T21:58:00+00:00", 50000),
        make("2026-09-17T21:59:00+00:00", 50000),
        make("2026-09-18T14:30:00+00:00", 49998.6, True),
        make("2026-09-18T21:00:01+00:00", 50017.2),
    ]
    for item in captures:
        immutable(tmp_path / "snapshots", item)
    return captures, fills, orders


def test_traded_both_strategies_exact_costs_balances_and_inventory(tmp_path):
    traded_session(tmp_path)
    report = reconcile_day(tmp_path, "42", "2026-09-18")
    assert report["status"] == "COMPLETE", report["issues"]
    assert report["net_total"] == pytest.approx(17.2)
    assert report["signed_fill_inventory"] == {"MNQ": 0, "MES": 0}
    assert {f["strategy_attribution"] for f in report["fills"]} == {"MIM", "YANK"}
    assert all(f["attribution_claims"] for f in report["fills"])
    assert all(type(f["size"]) is int for f in report["fills"])


@pytest.mark.parametrize(
    "field,value,issue",
    [
        ("fillVolume", None, "quantity"),
        ("fillVolume", 2, "quantity"),
        ("side", 1, "side"),
    ],
)
def test_order_quantity_and_side_fail_closed(tmp_path, field, value, issue):
    captures, _, _ = traded_session(tmp_path)
    correction = deepcopy(captures[-1])
    correction["observed_at"] = "2026-09-19T12:00:00+00:00"
    correction["requests"][-1]["response"]["orders"][0][field] = value
    immutable(tmp_path / "snapshots", correction)
    report = reconcile_day(tmp_path, "42", "2026-09-18")
    assert report["net_total"] is None
    assert any(issue in item for item in report["issues"])


def test_ambiguous_claim_cannot_be_overwritten_by_later_mim(tmp_path):
    captures, _, _ = traded_session(tmp_path)
    evidence = deepcopy(captures[-2])
    evidence["observed_at"] = "2026-09-18T14:31:00+00:00"
    evidence["local_after"]["mim_submissions"].append(
        dict(accountId=42, orderId=13, contractId="MES")
    )
    immutable(tmp_path / "snapshots", evidence)
    later = deepcopy(evidence)
    later["observed_at"] = "2026-09-18T14:32:00+00:00"
    later["local_after"]["YANK"]["state"] = snapshot()["local_after"]["YANK"]["state"]
    immutable(tmp_path / "snapshots", later)
    report = reconcile_day(tmp_path, "42", "2026-09-18")
    entry = next(row for row in report["fills"] if row["id"] == 3)
    assert entry["strategy_attribution"] == "ambiguous"
    assert set(entry["attribution_claims"]) == {"MIM", "YANK"}


@pytest.mark.parametrize(
    "mutation,expected",
    [("account", "account"), ("pnl", "profitAndLoss"), ("yank", "malformed_yank")],
)
def test_malformed_identity_economics_state_are_incomplete(
    tmp_path, mutation, expected
):
    captures, _, _ = traded_session(tmp_path)
    bad = deepcopy(captures[-1])
    bad["observed_at"] = "2026-09-18T21:00:02+00:00"
    if mutation == "account":
        del bad["requests"][-2]["response"]["trades"][0]["accountId"]
    elif mutation == "pnl":
        del bad["requests"][-2]["response"]["trades"][0]["profitAndLoss"]
    else:
        # Active-scope evidence before closing snapshot, so it is considered attribution evidence.
        bad["started_at"] = bad["observed_at"] = "2026-09-18T14:35:00+00:00"
        bad["local_after"]["YANK"]["state"] = []
    immutable(tmp_path / "snapshots", bad)
    report = reconcile_day(tmp_path, "42", "2026-09-18")
    assert report["net_total"] is None
    assert any(expected in issue for issue in report["issues"])
    if mutation == "pnl":
        assert len(report["fills"]) == 4 and report["gross_reported"] is None


def test_later_account_transition_does_not_invalidate_closed_day(tmp_path):
    captures, _, _ = traded_session(tmp_path)
    first = reconcile_day(tmp_path, "42", "2026-09-18")
    later = deepcopy(captures[-1])
    later.update(
        account="99",
        started_at="2026-09-19T12:00:00+00:00",
        observed_at="2026-09-19T12:00:01+00:00",
    )
    later["requests"][0]["response"]["accounts"] = [{"id": 99, "balance": 49000}]
    immutable(tmp_path / "snapshots", later)
    report = reconcile_day(tmp_path, "42", "2026-09-18")
    assert report == first


def test_late_corrected_fill_repairs_prior_bridge_and_refreshes_once(tmp_path):
    from research.project_goals.daily import refresh_corrections

    captures, fills, _ = traded_session(tmp_path, missing_exit=True)
    first = reconcile_day(tmp_path, "42", "2026-09-18")
    assert first["status"] == "INCOMPLETE"
    refresh_corrections(tmp_path, "42", captures[-1])
    corrected = deepcopy(captures[-1])
    corrected.update(
        started_at="2026-09-19T12:00:00+00:00", observed_at="2026-09-19T12:00:01+00:00"
    )
    corrected["requests"][-2]["response"]["trades"] = fills
    corrected["requests"][-2]["request"]["endTimestamp"] = corrected["observed_at"]
    immutable(tmp_path / "snapshots", corrected)
    assert refresh_corrections(tmp_path, "42", corrected) == ["2026-09-18"]
    assert refresh_corrections(tmp_path, "42", corrected) == []
    latest = json.loads((tmp_path / "latest_report.json").read_text())
    assert latest["status"] == "COMPLETE"
    assert len(list((tmp_path / "reports/2026-09-18").glob("*.json"))) >= 2
    assert (
        json.loads((tmp_path / "observation.json").read_text())["status"] == "STARTED"
    )


def test_balance_http_boundary_fill_is_uncertain_not_interrupted(tmp_path):
    captures, _, _ = traded_session(tmp_path)
    near = deepcopy(captures[-2])
    near.update(
        started_at="2026-09-18T13:59:59+00:00", observed_at="2026-09-18T14:00:02+00:00"
    )
    near["requests"][0].update(
        started_at="2026-09-18T13:59:59+00:00", observed_at="2026-09-18T14:00:01+00:00"
    )
    near["requests"][0]["response"]["accounts"][0]["balance"] = 50000
    immutable(tmp_path / "snapshots", near)
    report = reconcile_day(tmp_path, "42", "2026-09-18")
    assert report["observation"]["status"] == "STARTED"
    assert report["balance_bridge"]["uncertain_boundaries"]
    assert "balance_bridge_incomplete" in report["issues"]


def test_state_before_producer_start_is_unproven():
    data = snapshot()
    for field in ("local_before", "local_after"):
        data[field]["YANK"]["producer"]["started_at"] = "2026-09-19T00:00:00+00:00"
    assert "YANK:state_predates_producer" in flat_reasons(data, "42")


@pytest.mark.asyncio
async def test_real_adapter_refreshes_rejected_cached_token_and_api_failure(
    monkeypatch,
):
    import httpx
    from research.project_goals import capture

    async def no_sleep(_):
        pass

    monkeypatch.setattr(capture.asyncio, "sleep", no_sleep)

    class Auth:
        _token = "rejected"
        _token_expires_at = None

        async def authenticate(self):
            if self._token is None:
                self._token = "renewed"
            return self._token

    seen = []

    def handler(request):
        seen.append(request.headers["Authorization"])
        if len(seen) == 1:
            return httpx.Response(401)
        return httpx.Response(200, json={"success": True, "accounts": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ReadOnlyBroker(Auth(), client).request("Account/search", {})
    assert result["ok"] and seen == ["Bearer rejected", "Bearer renewed"]

    def failed(request):
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(failed)) as client:
        result = await ReadOnlyBroker(Auth(), client).request("Account/search", {})
    assert result["ok"] is False and result["attempts"] == 3
    assert result["error"] == "HTTPStatusError"


def test_consistent_wrong_sides_still_fail_signed_inventory(tmp_path):
    captures, _, _ = traded_session(tmp_path)
    bad = deepcopy(captures[-1])
    bad["observed_at"] = "2026-09-19T12:00:00+00:00"
    bad["requests"][-2]["response"]["trades"][1]["side"] = 0
    bad["requests"][-1]["response"]["orders"][1]["side"] = 0
    immutable(tmp_path / "snapshots", bad)
    report = reconcile_day(tmp_path, "42", "2026-09-18")
    assert "signed_inventory_mismatch:MNQ" in report["issues"]


def test_account_disappearance_updates_current_observation(tmp_path):
    captures, _, _ = traded_session(tmp_path)
    reconcile_day(tmp_path, "42", "2026-09-18")
    missing = deepcopy(captures[-2])
    missing.update(
        started_at="2026-09-18T16:00:00+00:00", observed_at="2026-09-18T16:00:01+00:00"
    )
    missing["requests"][0]["response"]["accounts"] = []
    immutable(tmp_path / "snapshots", missing)
    report = reconcile_day(tmp_path, "42", "2026-09-18")
    assert report["observation"]["reason"] == "account_disappeared_or_invalid"
    current = json.loads((tmp_path / "observation.json").read_text())
    assert current["status"] == "INTERRUPTED"
    assert current["account_reset_epoch"] is None


@pytest.mark.asyncio
async def test_capture_account_transition_and_new_account_schedule(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace
    from research.project_goals import capture, daily
    from src.research.projectx_auth import ProjectXAuth

    calls = []

    async def cleanup():
        pass

    monkeypatch.setattr(
        ProjectXAuth, "from_file", lambda *args: SimpleNamespace(cleanup=cleanup)
    )

    async def fake_capture(*args):
        return {"observed_at": "2026-09-18T21:00:00+00:00", "requests": []}, "fixture"

    monkeypatch.setattr(capture, "capture_once", fake_capture)
    monkeypatch.setattr(
        daily, "scheduled_events", lambda now: [("2026-09-18-close", "2026-09-18")]
    )
    monkeypatch.setattr(
        daily,
        "reconcile_day",
        lambda output, account, day: calls.append(account) or {"status": "INCOMPLETE"},
    )
    for account in ("42", "99", "42"):
        await capture.serve(
            tmp_path, tmp_path, account, tmp_path / "credentials", once=True
        )
    assert calls == ["42", "99"]


def test_prospective_yank_order_source_is_verified_and_used(tmp_path):
    captures, _, _ = traded_session(tmp_path)
    # Remove active-state attribution; the prospective log becomes the exact source.
    active = captures[-2]
    original_key = immutable(tmp_path / "snapshots", active)
    (tmp_path / "snapshots" / (original_key + ".json")).unlink()
    source = dict(
        text="ProjectX entry limit #13\nProjectX market close #14\n",
        producer={"account": "42"},
    )
    key = immutable(tmp_path / "order-sources", source)
    active["local_after"]["YANK"]["state"] = snapshot()["local_after"]["YANK"]["state"]
    active["local_after"]["yank_orders"] = {
        "orders": [
            dict(accountId="42", orderId=oid, source_hash=key, line=line)
            for oid, line in [("13", 1), ("14", 2)]
        ]
    }
    immutable(tmp_path / "snapshots", active)
    report = reconcile_day(tmp_path, "42", "2026-09-18")
    assert report["status"] == "COMPLETE", report["issues"]
    assert key in report["source_hashes"]
    assert any(
        key + ":1" in f["attribution_claims"].get("YANK", []) for f in report["fills"]
    )


def test_late_order_update_retains_prior_session_order(tmp_path):
    captures, _, _ = traded_session(tmp_path)
    correction = deepcopy(captures[-1])
    correction["observed_at"] = "2026-09-19T12:00:00+00:00"
    for order in correction["requests"][-1]["response"]["orders"]:
        order["updateTimestamp"] = "2026-09-19T11:00:00+00:00"
    immutable(tmp_path / "snapshots", correction)
    report = reconcile_day(tmp_path, "42", "2026-09-18")
    assert report["status"] == "COMPLETE", report["issues"]


@pytest.mark.parametrize("field", ["local_before", "local_after"])
def test_intermediate_producer_account_transition_interrupts(tmp_path, field):
    captures, _, _ = traded_session(tmp_path)
    changed = deepcopy(captures[-2])
    changed["observed_at"] = "2026-09-18T14:32:00+00:00"
    changed[field]["MIM"]["producer"]["account"] = "99"
    immutable(tmp_path / "snapshots", changed)
    report = reconcile_day(tmp_path, "42", "2026-09-18")
    assert report["observation"]["status"] == "INTERRUPTED"
    assert "producer_account_identity_transition:MIM" in report["issues"]
    current = json.loads((tmp_path / "observation.json").read_text())
    assert current["reason"] == "producer_account_identity_transition"


def test_intermediate_unknown_producer_identity_incomplete(tmp_path):
    captures, _, _ = traded_session(tmp_path)
    changed = deepcopy(captures[-2])
    changed["observed_at"] = "2026-09-18T14:32:00+00:00"
    changed["local_after"]["MIM"]["producer"] = {}
    immutable(tmp_path / "snapshots", changed)
    report = reconcile_day(tmp_path, "42", "2026-09-18")
    assert report["status"] == "INCOMPLETE"
    assert "producer_identity_unknown:MIM" in report["issues"]
