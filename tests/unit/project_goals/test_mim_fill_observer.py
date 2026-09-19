"""Execute real MIM observer methods without importing live log-writer globals."""

import ast
import asyncio
import csv
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


@pytest.mark.asyncio
async def test_submission_identity_survives_await_and_observer_persists_partials(
    tmp_path, monkeypatch
):
    source = Path(__file__).resolve().parents[3] / "src/research/mim_nb_live.py"
    tree = ast.parse(source.read_text())
    writer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ChainedCsv"
    )
    bot_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(method, ast.AsyncFunctionDef) and method.name == "_log_fill"
            for method in node.body
        )
    )
    methods = [
        method
        for method in bot_class.body
        if isinstance(method, ast.AsyncFunctionDef)
        and method.name in ("_order", "_log_fill")
    ]
    isolated = ast.Module(
        body=[
            writer,
            ast.ClassDef(
                name="Bot", bases=[], keywords=[], body=methods, decorator_list=[]
            ),
        ],
        type_ignores=[],
    )
    namespace = dict(
        Path=Path,
        csv=csv,
        hashlib=hashlib,
        json=json,
        asyncio=asyncio,
        datetime=datetime,
        timedelta=timedelta,
        timezone=timezone,
        logger=logging.getLogger("mim_fixture"),
        DATA_DIR=tmp_path,
        _BASE_URL="https://example.test",
        _TYPE_MARKET=2,
        _TYPE_STOP=4,
        CONTRACTS=2,
    )
    exec(compile(ast.fix_missing_locations(isolated), str(source), "exec"), namespace)
    headers = [
        "ts_utc",
        "event",
        "order_id",
        "otype",
        "side",
        "size",
        "price",
        "outcome",
        "detail",
    ]
    log = namespace["ChainedCsv"](tmp_path / "orders.csv", headers)
    namespace["orders_log"] = log
    bot = namespace["Bot"]()
    bot.account_id, bot.contract_id = 42, "MNQ"

    async def place(payload):
        assert payload == dict(accountId=42, contractId="MNQ", type=2, side=0, size=2)
        bot.account_id, bot.contract_id = 99, "OTHER"
        return 7

    async def auth_headers():
        return {}

    bot.px = SimpleNamespace(_place_order=place, _headers=auth_headers)
    base = dict(
        accountId=42,
        contractId="MNQ",
        orderId=7,
        side=0,
        size=1,
        price=100,
        fees=0.5,
        commissions=None,
        voided=False,
    )
    records = [
        dict(base, id=1, orderId=8),
        dict(base, id=2, accountId=99),
        dict(base, id=3, contractId="OTHER"),
        dict(base, id=4),
        dict(base, id=5, price=101),
        dict(base, id=4),
    ]

    async def post(url, *, json, headers):
        assert json["accountId"] == 42
        return httpx.Response(
            200,
            json=dict(success=True, trades=records),
            request=httpx.Request("POST", url),
        )

    bot.http = SimpleNamespace(post=post)

    async def no_sleep(seconds):
        return None

    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    tasks = []
    loop = asyncio.get_running_loop()

    def create_task(coro):
        task = loop.create_task(coro)
        tasks.append(task)
        return task

    monkeypatch.setattr(
        asyncio, "get_event_loop", lambda: SimpleNamespace(create_task=create_task)
    )
    assert await bot._order(2, 0, price=99.5) == 7
    assert len(tasks) == 1
    await asyncio.gather(*tasks)
    with (tmp_path / "orders.csv").open() as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames == headers + ["chain"]
    assert [row["event"] for row in rows] == ["PLACE", "FILL", "FILL"]
    assert [row["price"] for row in rows[1:]] == ["100", "101"]
    assert all(
        json.loads(row["detail"])["strategy_reference"] == 99.5 for row in rows[1:]
    )
    assert len(list((tmp_path / "broker_fill_evidence").glob("*.json"))) == 2
    # A restarted observer reuses durable broker IDs, without appending duplicate CSV rows.
    submitted = json.loads(rows[0]["detail"])
    submitted.pop("orderId")
    await bot._log_fill(7, submitted)
    with (tmp_path / "orders.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 3
    records[:] = [
        dict(base, id=6, voided=True),
        dict(base, id=7),
        dict(base, id=7, price=102),
    ]
    await bot._log_fill(7, submitted)
    with (tmp_path / "orders.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 3


@pytest.mark.parametrize(
    "bad", [dict(id=""), dict(id=True), dict(size=True), dict(price=True)]
)
def test_invalid_broker_scalar_cannot_be_logged(bad):
    from src.research.broker_fill_evidence import matching_fills

    identity = dict(accountId=42, contractId="MNQ", orderId=7, side=0, size=1)
    record = dict(identity, id=1, price=100, voided=False)
    record.update(bad)
    rows, issues = matching_fills([record], identity)
    assert not rows and issues
