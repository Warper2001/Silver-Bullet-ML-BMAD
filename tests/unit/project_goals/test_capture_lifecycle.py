"""Capture singleton and scheduler persistence across process recreation."""

import fcntl
from types import SimpleNamespace

import pytest
from research.project_goals import capture, daily
from src.research.projectx_auth import ProjectXAuth


@pytest.mark.asyncio
async def test_singleton_refuses_second_collector_before_auth(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("overlapping collector attempted authentication")

    monkeypatch.setattr(ProjectXAuth, "from_file", forbidden)
    with (tmp_path / "capture.lock").open("a") as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            await capture.serve(
                tmp_path, tmp_path, "42", tmp_path / "credentials", once=True
            )


@pytest.mark.asyncio
async def test_service_recreation_retains_scheduled_receipts(tmp_path, monkeypatch):
    closes = []
    reports = []
    captures = []

    async def cleanup():
        closes.append(True)

    monkeypatch.setattr(
        ProjectXAuth, "from_file", lambda *args: SimpleNamespace(cleanup=cleanup)
    )

    async def record(*args):
        captures.append(True)
        return {"observed_at": "2026-09-18T21:00:00+00:00", "requests": []}, "fixture"

    def reconcile(output, account, day):
        reports.append((account, day))
        return dict(status="INCOMPLETE")

    monkeypatch.setattr(capture, "capture_once", record)
    monkeypatch.setattr(
        daily, "scheduled_events", lambda now: [("2026-09-18-close", "2026-09-18")]
    )
    monkeypatch.setattr(daily, "reconcile_day", reconcile)
    await capture.serve(tmp_path, tmp_path, "42", tmp_path / "credentials", once=True)
    await capture.serve(tmp_path, tmp_path, "42", tmp_path / "credentials", once=True)
    assert len(captures) == 2 and len(closes) == 2
    assert reports == [("42", "2026-09-18")]
    assert list((tmp_path / "schedule").rglob("*.json"))
