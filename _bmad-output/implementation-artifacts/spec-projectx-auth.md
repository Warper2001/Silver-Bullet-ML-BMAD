---
title: 'ProjectX API Authentication Module'
type: 'feature'
created: '2026-06-07'
status: 'done'
route: 'one-shot'
---

## Intent

**Problem:** The live trader (`tier2_streaming_working.py`) uses `TradeStationAuthV3` for authentication, but Topstep account execution requires the ProjectX/TopstepX API, which uses API key → JWT authentication — a completely different flow with no OAuth refresh tokens.

**Approach:** Create `src/research/projectx_auth.py` with a `ProjectXAuth` class that mirrors the `TradeStationAuthV3` public interface (`from_file`, `authenticate`, `start_auto_refresh`, `is_authenticated`, `cleanup`) so it can be swapped in as the auth provider for `ProjectXClient`.

## Suggested Review Order

1. [`src/research/projectx_auth.py:60`](../../src/research/projectx_auth.py#L60) — `from_file()`: credential loading, encoding, line count validation
2. [`src/research/projectx_auth.py:96`](../../src/research/projectx_auth.py#L96) — `authenticate()`: double-checked locking pattern with `asyncio.Lock`
3. [`src/research/projectx_auth.py:148`](../../src/research/projectx_auth.py#L148) — `_login()`: HTTP retry loop, 4xx short-circuit, no credential echo in logs
4. [`src/research/projectx_auth.py:115`](../../src/research/projectx_auth.py#L115) — `start_auto_refresh()` + `_refresh_loop()`: initial stale-token check before first sleep
5. [`src/research/projectx_auth.py:132`](../../src/research/projectx_auth.py#L132) — `cleanup()`: sets `_refresh_task=None` before awaiting, `_closed` guard prevents post-cleanup client creation
6. [`src/research/projectx_auth.py:205`](../../src/research/projectx_auth.py#L205) — `_extract_token()`: multi-shape response handling
7. [`src/research/projectx_auth.py:218`](../../src/research/projectx_auth.py#L218) — `_decode_jwt_exp()`: graceful JWT decode with debug-level fallback log

## Spec Change Log

<!-- empty — no review loops required -->
