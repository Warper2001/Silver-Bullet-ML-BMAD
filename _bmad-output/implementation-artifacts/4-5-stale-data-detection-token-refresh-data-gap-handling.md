# Story 4.5: Stale Data Detection, Token Refresh, and Data Gap Handling

Status: done

## Story

As Alex,
I want the system to detect stale bars, auto-refresh expired API tokens, and handle polling gaps without generating phantom trades,
So that the system never silently trades on outdated data or fails due to an expired credential.

## Acceptance Criteria

**AC#1 — Stale data detection sets flag and blocks entries:**
Given the polling loop receives a bar during RTH (09:30–16:00 ET),
When the most recent bar's timestamp is more than 5 minutes older than system wall-clock time,
Then the system logs `STALE_DATA: last bar at <timestamp>, system time <now> — halting entries` and sets `self._data_stale = True` blocking all new entries (FR36).

**AC#2 — Stale flag clears on fresh bar:**
Given `_data_stale` is True,
When the next poll returns a bar with timestamp within 5 minutes of wall-clock,
Then `_data_stale` is cleared to False and trading resumes automatically.

**AC#3 — Token refresh: auto-refresh already handles this:**
Given the existing `auth.start_auto_refresh()` background loop + `authenticate()` called each poll,
Then token expiry is handled automatically — Story 4-5 adds explicit logging when `authenticate()` triggers a refresh (the token was stale before the call) by checking `_is_token_valid()` before and after (FR39).

**AC#4 — DATA_GAP on empty response:**
Given a polling cycle where the TradeStation API returns an empty bars list,
When `_poll_and_process()` processes this,
Then it logs `DATA_GAP: no bars returned at <timestamp>` and skips all detection for this cycle — no stale FVG or sweep signal carried forward as if bars were received (NFR13).

**AC#5 — API timeout is caught and loop continues:**
Given `httpx.TimeoutException` (or `asyncio.TimeoutError`) is raised during any API call,
When the exception propagates to `_poll_and_process()`,
Then it is caught separately from the generic `Exception` handler, logged as `API_TIMEOUT: request timed out — skipping bar`, and the loop continues on the next 60-second cycle without crashing (NFR2).

**AC#6 — Stale check is only active during RTH:**
Given a bar arrives outside RTH (before 09:30 or after 16:00 ET),
When timestamp check runs,
Then `_data_stale` is NOT set — stale detection is suppressed outside regular trading hours.

**AC#7 — Unit tests:**
Given `tests/unit/test_stale_data_gap_handling.py`,
When pytest runs it,
Then all tests pass covering:
- Stale flag set when bar >5 min old during RTH
- Stale flag not set outside RTH
- Stale flag clears on fresh bar
- DATA_GAP logged when bars list empty
- API_TIMEOUT caught and loop continues
- `_detect_and_enter` blocked when `_data_stale = True`

## Tasks / Subtasks

- [x] Task 1: Add stale data detection state and check (AC: #1, #2, #6)
  - [x] `self._data_stale: bool = False` added to `__init__()`
  - [x] `_is_rth(now_et)` and `_check_stale(bar)` methods added
  - [x] `_check_stale` called after each new bar appended in `_poll_and_process`
  - [x] `_detect_and_enter` returns early if `self._data_stale`

- [x] Task 2: DATA_GAP handling (AC: #4)
  - [x] Empty `bars_data` → log `DATA_GAP: no bars returned` and return

- [x] Task 3: API_TIMEOUT handling (AC: #5)
  - [x] `except (httpx.TimeoutException, asyncio.TimeoutError)` before generic `except Exception`
  - [x] Logs `API_TIMEOUT: request timed out — skipping bar`

- [x] Task 4: 11 unit tests written and passing (AC: #7)

- [x] Task 5: 54/54 story tests pass, no regressions

## Dev Notes

### RTH Definition

RTH is 09:30–16:00 ET (America/New_York). In UTC: 13:30–20:00 (no DST) or 13:30–21:00 (DST). Simplest approach: convert wall-clock to ET for RTH check.

```python
def _is_rth(self, now_et: datetime) -> bool:
    """Return True if now is within 09:30–16:00 ET (RTH)."""
    h, m = now_et.hour, now_et.minute
    return (h == 9 and m >= 30) or (10 <= h <= 15) or (h == 16 and m == 0)
```

### Stale Check

```python
def _check_stale(self, bar: DollarBar) -> bool:
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(ET_TZ)
    if not self._is_rth(now_et):
        return False
    age_seconds = (now_utc - bar.timestamp.replace(tzinfo=timezone.utc) 
                   if bar.timestamp.tzinfo is None 
                   else now_utc - bar.timestamp).total_seconds()
    return age_seconds > 300  # 5 minutes
```

### Token Refresh

`authenticate()` is already called at the start of each poll cycle (line 971) and already calls `_refresh_token_flow()` when the token is within 5 minutes of expiry. No additional code needed — add a log line when the refresh is triggered. Check `self.auth._is_token_valid()` before `authenticate()`, then after — if it changed from False to True, a refresh occurred.

### Timeout Handling in _poll_and_process

The current code structure:
```python
async def _poll_and_process(self):
    try:
        ...
    except Exception as e:
        logger.error(...)
```

Add `except (httpx.TimeoutException, asyncio.TimeoutError)` BEFORE the generic `except Exception`:
```python
    except (httpx.TimeoutException, asyncio.TimeoutError):
        logger.warning("API_TIMEOUT: request timed out — skipping bar")
    except Exception as e:
        logger.error(...)
```

### Key File Locations

- `src/research/tier2_streaming_working.py` — all changes here
- `tests/unit/test_stale_data_gap_handling.py` — new test file

## Dev Agent Record

### File List

- `src/research/tier2_streaming_working.py` — modified
- `tests/unit/test_stale_data_gap_handling.py` — new

### Change Log

- 2026-05-25: Story created
