"""Receipt ownership, freshness and finalization without sending any signals."""

from datetime import datetime, timedelta, timezone

import pytest
from research.project_goals import scheduler


@pytest.mark.parametrize(
    "alive,supervisor,elapsed,age,status",
    [
        (True, True, 2, 0, "SUPERVISED_POLLING"),
        (True, True, 62, 0, "OVERDUE_EXECUTION"),
        (True, True, 2, 12, "OVERDUE_EXECUTION"),
        (False, True, 62, 0, "FINALIZING"),
        (False, True, 62, 65, "OVERDUE_FINALIZATION"),
        (True, False, 2, 0, "ORPHANED_CHILD"),
        (False, False, 2, 0, "OPERATOR_RECOVERY_REQUIRED"),
    ],
)
def test_lifecycle(tmp_path, monkeypatch, alive, supervisor, elapsed, age, status):
    now = datetime.now(timezone.utc)
    owner = dict(pid=10, start_ticks="10", boot_id="boot")
    child = dict(pid=11, start_ticks="11", boot_id="boot")

    def identity(pid):
        if pid == 10 and supervisor:
            return owner
        if pid == 11 and alive:
            return child
        return None

    monkeypatch.setattr(scheduler, "process_identity", identity)
    start = (now - timedelta(seconds=elapsed)).isoformat()
    scheduler.atomic(
        tmp_path / "child.json",
        dict(
            identity=child,
            supervisor=owner,
            started_at=start,
            poll_started_at=start,
            stale_after_seconds=60,
        ),
    )
    scheduler.atomic(
        tmp_path / "heartbeat.json",
        dict(
            supervisor=owner,
            started_at=start,
            last_observed_at=(now - timedelta(seconds=age)).isoformat(),
            status="FINALIZING" if not alive else "SUPERVISED_POLLING",
        ),
    )
    result = scheduler.child_health(tmp_path)
    assert result["status"] == status
    assert result["operator_recovery_required"] == (not supervisor)


def test_reused_supervisor_pid_and_malformed_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        scheduler, "process_identity", lambda pid: dict(pid=pid, start_ticks="new")
    )
    now = datetime.now(timezone.utc).isoformat()
    scheduler.atomic(
        tmp_path / "child.json",
        dict(identity=None, supervisor=dict(pid=10, start_ticks="old"), started_at=now),
    )
    assert scheduler.child_health(tmp_path)["status"] == "OPERATOR_RECOVERY_REQUIRED"
    (tmp_path / "child.json").write_text("{broken")
    assert scheduler.child_health(tmp_path)["operator_recovery_required"]
