import json
from research.project_goals import order_evidence
from research.project_goals.scheduler import atomic


def test_prospective_short_trade_and_restart_do_not_bless_old_accounts(
    tmp_path, monkeypatch
):
    root, out = tmp_path / "root", tmp_path / "out"
    (root / "logs").mkdir(parents=True)
    out.mkdir()
    path = root / "logs/yank_streaming_working.log"
    path.write_text("ProjectX entry limit #1\n")
    identity = dict(pid=10, start_ticks="1", boot_id="boot")
    producer = dict(identity=identity, account="42", matches_source=True)
    monkeypatch.setattr(order_evidence, "process_identity", lambda pid: identity)
    evidence, cursor = order_evidence.prospective_yank_orders(root, out, producer)
    assert not evidence["orders"] and evidence["status"] == "BASELINE_ONLY"
    atomic(out / "yank-log-cursor.json", cursor)
    with path.open("a") as stream:
        stream.write(
            "ProjectX entry limit #2\nProjectX TP limit #3\nProjectX SL stop #4\nProjectX market close #5\n"
        )
    evidence, cursor = order_evidence.prospective_yank_orders(root, out, producer)
    assert [o["orderId"] for o in evidence["orders"]] == ["2", "3", "4", "5"]
    assert all(o["accountId"] == "42" for o in evidence["orders"])
    # Crash before cursor publication replays the same immutable evidence safely.
    again, _ = order_evidence.prospective_yank_orders(root, out, producer)
    assert again == evidence
    atomic(out / "yank-log-cursor.json", cursor)
    assert not order_evidence.prospective_yank_orders(root, out, producer)[0]["orders"]
    changed = dict(producer, account="99")
    with path.open("a") as stream:
        stream.write("ProjectX market close #6\n")
    assert not order_evidence.prospective_yank_orders(root, out, changed)[0]["orders"]
    source = json.loads(next((out / "order-sources").glob("*.json")).read_text())
    assert source["start_offset"] > 0 and "limit #1" not in source["text"]


def test_log_overflow_and_partial_line_are_explicit(tmp_path, monkeypatch):
    root, out = tmp_path / "root", tmp_path / "out"
    (root / "logs").mkdir(parents=True)
    out.mkdir()
    path = root / "logs/yank_streaming_working.log"
    path.write_text("")
    identity = dict(pid=1)
    producer = dict(identity=identity, account="42", matches_source=True)
    monkeypatch.setattr(order_evidence, "process_identity", lambda pid: identity)
    _, cursor = order_evidence.prospective_yank_orders(root, out, producer)
    atomic(out / "yank-log-cursor.json", cursor)
    path.write_text("ProjectX market close #9")
    data, cursor = order_evidence.prospective_yank_orders(root, out, producer)
    assert not data["orders"] and cursor["offset"] == 0
    with path.open("a") as stream:
        stream.write("\n")
    assert (
        order_evidence.prospective_yank_orders(root, out, producer)[0]["orders"][0][
            "orderId"
        ]
        == "9"
    )
    monkeypatch.setattr(order_evidence, "MAX_BYTES", 1)
    assert (
        order_evidence.prospective_yank_orders(root, out, producer)[0]["status"]
        == "LOG_CAPTURE_GAP"
    )
