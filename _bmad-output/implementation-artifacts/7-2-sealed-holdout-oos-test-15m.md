# Story 7.2: Sealed Holdout OOS Test (15m)

Status: done

## Story

As Alex (researcher),
I want to execute the pre-registered Phase 2 15m OOS test against the sealed holdout,
so that I have auditable out-of-sample evidence for the Phase 2 verdict.

## Acceptance Criteria

1. `holdout_15m_oos_test.py` executed with pre-registration SHA `5b581f4d` gating access.
2. ACCESS_LOG.md updated with result before any results printed.
3. Verdict report `_bmad-output/s_phase2_15m_verdict_<date>.md` produced.
4. No modifications to strategy_core.py, backtest_engine.py, or tier2_streaming_working.py.

## Tasks / Subtasks

- [x] Task 1 — Run holdout_15m_oos_test.py (AC #1, #2, #3)
  - [x] Execute via nohup
  - [x] Verify ACCESS_LOG.md updated
  - [x] Verify verdict report written

- [x] Task 2 — Verify no strategy code modified (AC #4)

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6 (2026-05-23)

### Completion Notes List

1. Ran via `nohup bash -c 'PYTHONPATH=. .venv/bin/python src/research/holdout_15m_oos_test.py > /tmp/phase2_15m_holdout.log 2>&1' &`
2. Holdout: 75,081 1m bars → 5,171 15m bars (2026-03-01 to 2026-05-19).
3. **Result: 6 trades | PF=2.586 | WR=0.667 | Sharpe=7.684 | Exits: TP=4, SL=2, TIME_STOP=0**
4. **VERDICT: PASS — H₁ SUPPORTED.** PF=2.586 >> threshold 1.1.
5. Caveat: N=6 is an extremely small sample. Expected ~13; got 6. The StrategyConfig defaults generate fewer trades at 15m in the holdout window than training. Result is directional, not conclusive.
6. ACCESS_LOG.md appended before any results printed. Pre-reg SHA `5b581f4d` recorded.
7. No strategy code modifications.

### File List

- `_bmad-output/s_phase2_15m_verdict_20260523.md` — NEW (produced by script)
- `data/sealed_holdout/ACCESS_LOG.md` — UPDATED (new entry appended)
