"""ProjectX/TopstepX market-data adapter — emits TradeStation-shaped bar dicts.

Lets the YANK + MIM-NB live bots source 1-min bars from ProjectX instead of
TradeStation by returning the SAME dict shape they already parse:

    {"TimeStamp": "2026-06-18T02:15:00Z", "Open", "High", "Low", "Close", "TotalVolume"}

so each bot's existing parse / dedup / finality / pipeline code runs unchanged
(YANK `_parse_bar`; MIM `on_bar` / `_ts_get_bars`).

CRITICAL labeling fact (validated across 3 live MNQU26 sessions via
tools/bar_parity_probe.py): ProjectX labels a 1-min bar by its OPEN time;
TradeStation by its CLOSE time — ProjectX is +1 min behind. This adapter ADDS
PX_LABEL_OFFSET_MIN to each ProjectX bar so the emitted TimeStamp matches the
close-time convention the strategies + meta-model were calibrated on. Forgetting
this would shift every signal by a minute.

Read-only against POST /api/History/retrieveBars. Takes the caller's already-held
ProjectXAuth + httpx.AsyncClient (no new login / connection).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.research.projectx_client import _to_contract_id  # re-exported for callers

__all__ = ["fetch_px_ts_shaped", "ProjectXBarFetchError", "PX_LABEL_OFFSET_MIN", "_to_contract_id"]

PX_HISTORY_URL = "https://api.topstepx.com/api/History/retrieveBars"
PX_LABEL_OFFSET_MIN = 1          # ProjectX open-time + 1 min == TradeStation close-time
_PAGE_LIMIT = 20000              # retrieveBars per-call cap
_MAX_PAGES = 8                   # safety bound (8 * 20000 = 160k bars ≈ months)


class ProjectXBarFetchError(RuntimeError):
    """retrieveBars returned a transport error or success:false. Caller skips the poll."""


def _parse_t(t: str) -> datetime:
    """ProjectX 't' (ISO, e.g. '2026-06-18T02:14:00+00:00') -> UTC, floored to the minute."""
    return (datetime.fromisoformat(t.replace("Z", "+00:00"))
            .astimezone(timezone.utc).replace(second=0, microsecond=0))


def _to_ts_shaped(b: dict, now_utc: datetime) -> dict | None:
    """One ProjectX bar -> TradeStation-shaped dict with +1-min offset. None if future/partial."""
    ts = _parse_t(b["t"]) + timedelta(minutes=PX_LABEL_OFFSET_MIN)
    if ts > now_utc:                                  # drop forming/future bar (mirrors yank:1066)
        return None
    return {
        "TimeStamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Open": float(b["o"]), "High": float(b["h"]),
        "Low": float(b["l"]), "Close": float(b["c"]),
        "TotalVolume": int(b["v"]),
    }


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


async def _retrieve(http, token, contract_id, start_utc, end_utc, live) -> list[dict]:
    payload = {
        "contractId": contract_id, "live": live,
        "startTime": _iso_z(start_utc), "endTime": _iso_z(end_utc),
        "unit": 2, "unitNumber": 1, "limit": _PAGE_LIMIT, "includePartialBar": False,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
               "Accept": "application/json"}
    resp = await http.post(PX_HISTORY_URL, json=payload, headers=headers)
    if resp.status_code != 200:
        raise ProjectXBarFetchError(f"retrieveBars HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    if not data.get("success", False):
        raise ProjectXBarFetchError(
            f"retrieveBars success=false errorCode={data.get('errorCode')} "
            f"msg={data.get('errorMessage')!r}")
    bars = data.get("bars") or []
    if bars and not all(k in bars[0] for k in ("t", "o", "h", "l", "c", "v")):
        raise ProjectXBarFetchError(f"unexpected ProjectX bar schema: {bars[0]!r}")
    return bars


async def fetch_px_ts_shaped(http, px_auth, contract_id, *, now_utc, live,
                             since_utc=None, barsback=None) -> list[dict]:
    """Return TradeStation-shaped 1-min bar dicts from ProjectX: time-ordered ascending,
    +1-min aligned to the close-time convention, completed bars only.

    Window selection (one of):
      since_utc  — incremental fetch [since_utc, now]  (YANK poll)
      barsback   — last N bars (count); over-fetch a 2x-minute window then trim  (MIM)
      neither    — last 48h backfill default
    retrieveBars caps at 20000 bars/call; when the window exceeds that, pages backward
    by shrinking the end toward the start. Raises ProjectXBarFetchError on any fetch error.
    """
    token = await px_auth.authenticate()
    if since_utc is not None:
        start = since_utc.astimezone(timezone.utc)
    elif barsback is not None:
        start = now_utc - timedelta(minutes=int(barsback) * 2)
    else:
        start = now_utc - timedelta(hours=48)

    # Page backward (retrieveBars returns the most-recent <=limit bars in the window;
    # if truncated, refetch [start, oldest_seen] for the next-older batch).
    raw: list[dict] = []
    cursor_end = now_utc
    for _ in range(_MAX_PAGES):
        page = await _retrieve(http, token, contract_id, start, cursor_end, live)
        if not page:
            break
        raw.extend(page)
        if len(page) < _PAGE_LIMIT:
            break
        oldest = min(_parse_t(b["t"]) for b in page)
        if oldest <= start:
            break
        if oldest >= cursor_end:                      # no progress guard
            break
        cursor_end = oldest

    out, seen = [], set()
    for b in raw:
        row = _to_ts_shaped(b, now_utc)
        if row is None or row["TimeStamp"] in seen:   # drop future bars + paging overlap
            continue
        seen.add(row["TimeStamp"])
        out.append(row)
    out.sort(key=lambda r: r["TimeStamp"])
    if barsback is not None and len(out) > int(barsback):
        out = out[-int(barsback):]
    return out
