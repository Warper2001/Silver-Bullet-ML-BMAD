"""thursday_short.main() must not swallow a fatal.

This is the whole of the 2026-07-27 incident. The bot crashed on a shared-.env write
race that wiped TRADESTATION_CLIENT_ID/SECRET. The race was fixed 15 minutes later by
commit 017521c -- but main() caught the exception, logged "Fatal: ...", and returned
normally. Python exited 0. systemd's `Restart=on-failure` (RestartSec=60, see
deploy/systemd/trader-thursday-short.service) correctly read that as a clean shutdown
and did not restart it.

The bot stayed dead for 24 days. Four Thursdays (2026-07-30, 08-06, 08-13, 08-20) passed
with zero log output and zero ledger rows, silently voiding four observations of a
pre-registered prospective accrual. The bot's own in-loop absence alarm cannot fire when
the process is not running.

A clean SIGTERM shutdown must still exit 0, or systemd would restart the bot every time
it was deliberately stopped -- so the two paths are tested separately.
"""
from __future__ import annotations

import asyncio

import pytest

import thursday_short


class _Boom(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _no_signal_handlers(monkeypatch):
    """asyncio signal handlers need a real loop signal setup; not what we're testing."""
    monkeypatch.setattr(
        asyncio.unix_events._UnixSelectorEventLoop, "add_signal_handler",
        lambda self, sig, cb, *a: None, raising=False,
    )


def test_fatal_propagates_so_systemd_sees_failure(monkeypatch):
    """The regression guard: a fatal must reach asyncio.run, not be swallowed."""
    async def _raise(self):
        raise _Boom("credentials missing")

    monkeypatch.setattr(thursday_short.ThursdayShortTrader, "run", _raise)
    monkeypatch.setattr(thursday_short.ThursdayShortTrader, "stop", lambda self: None)

    with pytest.raises(_Boom):
        asyncio.run(thursday_short.main())


def test_clean_return_still_exits_zero(monkeypatch):
    """A normal shutdown (SIGTERM -> stop() -> _run returns) must NOT look like failure.

    Otherwise systemd would restart the bot every time it was deliberately stopped.
    """
    async def _clean(self):
        return None

    monkeypatch.setattr(thursday_short.ThursdayShortTrader, "run", _clean)
    monkeypatch.setattr(thursday_short.ThursdayShortTrader, "stop", lambda self: None)

    asyncio.run(thursday_short.main())  # must not raise


def test_stop_is_called_even_on_fatal(monkeypatch):
    """The finally: block must still run — the fix must not skip cleanup."""
    called = []

    async def _raise(self):
        raise _Boom("boom")

    monkeypatch.setattr(thursday_short.ThursdayShortTrader, "run", _raise)
    monkeypatch.setattr(thursday_short.ThursdayShortTrader, "stop",
                        lambda self: called.append(True))

    with pytest.raises(_Boom):
        asyncio.run(thursday_short.main())
    assert called == [True]
