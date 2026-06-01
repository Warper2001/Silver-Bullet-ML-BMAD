# Story 5.3: Dynamic Contract Limit and XFA Scaling Plan

Status: review

## Story

As Alex,
I want the system to read a configurable scaling plan and automatically cap position size based on accumulated profit milestones,
So that my contract count scales safely with the XFA rules (2 → 3 → 5 MNQ) as profit milestones are reached.

## Background

TopStep XFA (Express Funded Account) enforces a dynamic contract limit that scales with accumulated profit: start at 2 MNQ contracts, scale to 3 at +$1,500, scale to 5 at +$2,000. This is a hard rule — exceeding the current contract limit is a compliance violation.

The `AccountConfig.scaling_plan` is a list of milestone dicts. The pure function `calc_contract_limit()` is the sole logic owner — it is side-effect-free and testable in isolation. The result is applied as a cap in `_enter_trade()` before order submission.

## Acceptance Criteria

**Given** `AccountConfig.scaling_plan = [{"milestone_usd": 0, "max_contracts": 2}, {"milestone_usd": 1500, "max_contracts": 3}, {"milestone_usd": 2000, "max_contracts": 5}]` and `accumulated_profit = $1,600`,
**When** `calc_contract_limit(accumulated_profit: float, scaling_plan: list[dict]) -> int` is called,
**Then** it returns `3` (the `$1,500` milestone is passed, the `$2,000` milestone is not) (FR20)

**Given** `accumulated_profit = $2,100`,
**When** `calc_contract_limit()` is called,
**Then** it returns `5`

**Given** `accumulated_profit = $900` (below first paid milestone),
**When** `calc_contract_limit()` is called,
**Then** it returns `2`

**Given** `dynamic_contract_limit = 2` and `config.contracts_per_trade = 5`,
**When** `_enter_trade()` resolves the final contract count,
**Then** `contracts = min(dynamic_contract_limit, config.contracts_per_trade) = 2` — the scaling limit caps the config (FR20)

**Given** a different `AccountConfig` with a different `scaling_plan`,
**When** `calc_contract_limit()` is called,
**Then** it uses the new plan without any code changes — fully configurable (FR14 / FR15 principle applied)

**Given** `scaling_plan = []` (empty — no scaling plan configured),
**When** `calc_contract_limit()` is called,
**Then** it returns `config.contracts_per_trade` (no cap applied — scaling plan is opt-in)

## Tasks / Subtasks

- [x] Task 1: Add `scaling_plan` to `AccountConfig` (AC: #5, #6)
  - [x] Add `scaling_plan: list[dict] = field(default_factory=list)` to `AccountConfig` dataclass
  - [x] Each entry: `{"milestone_usd": float, "max_contracts": int}` (validated by caller, no runtime schema enforcement needed)
  - [x] Default is empty list — no cap by default, backward compatible

- [x] Task 2: Implement `calc_contract_limit()` pure function (AC: #1–#3, #6)
  - [x] `def calc_contract_limit(accumulated_profit: float, scaling_plan: list[dict]) -> Optional[int]` — add to `strategy_core.py`
  - [x] Sort plan by `milestone_usd` descending, find highest milestone whose `milestone_usd ≤ accumulated_profit`
  - [x] Return `max_contracts` for that tier
  - [x] If `scaling_plan` is empty or no milestone matches: return `None` (no cap)
  - [x] The base tier `{"milestone_usd": 0, ...}` always matches if present

- [x] Task 3: Track `accumulated_profit` in `RiskManager` (AC: #1–#4)
  - [x] Add `_accumulated_profit: float = 0.0` field to `RiskManager`
  - [x] Update in `register_close(pnl)` — accumulates lifetime (not reset daily)
  - [x] Persist in `to_state_dict()` as `accumulated_profit`
  - [x] Restore in `restore_from_state()` — persists across days (unlike daily_pnl)

- [x] Task 4: Wire contract cap into `_enter_trade()` (AC: #4)
  - [x] After `make_entry_decision()`, call `calc_contract_limit(accumulated_profit, account_config.scaling_plan)`
  - [x] If result is not None: `effective_contracts = min(entry_dec.contracts, contract_limit)`
  - [x] Log `SCALING_LIMIT: capping contracts to X (accumulated=$Y.ZZ, plan tier=$A→B contracts)`
  - [x] Combine with consistency-rule reduction from Story 5-2: apply both caps; final = `min(consistency_cap, scaling_cap, strategy_config_cap)`

- [x] Task 5: Unit tests (all ACs)
  - [x] `test_calc_contract_limit_first_tier` — accumulated below first paid milestone → base limit
  - [x] `test_calc_contract_limit_middle_tier` — accumulated past $1,500 → 3 contracts
  - [x] `test_calc_contract_limit_top_tier` — accumulated past $2,000 → 5 contracts
  - [x] `test_calc_contract_limit_empty_plan` → returns None (no cap)
  - [x] `test_calc_contract_limit_exactly_at_milestone` — accumulated = milestone exactly → that tier
  - [x] `test_contracts_capped_by_scaling_limit_in_enter_trade` — entry resolves to 2 when limit=2
  - [x] `test_accumulated_profit_persists_across_days` — `to_state_dict` / `restore_from_state` round-trip
  - [x] `test_scaling_and_consistency_both_applied` — min of both caps wins

## Dev Notes

### `calc_contract_limit()` Implementation

Add to `src/research/strategy_core.py`:

```python
def calc_contract_limit(
    accumulated_profit: float,
    scaling_plan: list[dict],
) -> Optional[int]:
    """Return max contracts allowed by XFA scaling plan, or None if no plan configured.

    Plan format: [{"milestone_usd": 0, "max_contracts": 2}, {"milestone_usd": 1500, "max_contracts": 3}, ...]
    Sorted descending by milestone — first entry whose milestone ≤ accumulated_profit wins.
    """
    if not scaling_plan:
        return None
    sorted_plan = sorted(scaling_plan, key=lambda t: t["milestone_usd"], reverse=True)
    for tier in sorted_plan:
        if accumulated_profit >= tier["milestone_usd"]:
            return tier["max_contracts"]
    return None
```

### `accumulated_profit` vs `daily_pnl`

`_daily_pnl` resets each calendar day. `_accumulated_profit` is the running lifetime total — it only increases (and decreases if there are losses). This reflects the TopStep XFA interpretation: milestones are based on cumulative profit since account inception or last payout. Initialize to 0.0 on first run; restore from state on crash recovery.

### Contract Resolution Order in `_enter_trade()`

All three caps must be applied together:

```python
effective_contracts = entry_dec.contracts  # from strategy_core (= contracts_per_trade)

# Cap 1: Consistency rule (Story 5-2)
if self._risk_manager.consistency_size_reduced:
    effective_contracts = min(effective_contracts, 1)

# Cap 2: XFA scaling plan (Story 5-3)
scaling_limit = calc_contract_limit(
    self._risk_manager.accumulated_profit,
    self._account_config.scaling_plan,
)
if scaling_limit is not None:
    effective_contracts = min(effective_contracts, scaling_limit)
```

### Accumulated Profit State Key

```python
# In to_state_dict():
{
    "accumulated_profit": self._accumulated_profit,  # float, never reset
    # existing keys: daily_pnl, daily_halted, last_trading_date, ...
}
```

### References

- FR20: `_bmad-output/planning-artifacts/prd.md:579`
- Epic 5 story spec: `_bmad-output/planning-artifacts/epics.md:1004`
- Journey 4 narrative: `_bmad-output/planning-artifacts/prd.md:198`
- `AccountConfig`: `src/research/tier2_streaming_working.py:81`
- `RiskManager.register_close()`: `src/research/tier2_streaming_working.py:306`
- `make_entry_decision` + `EntryDecision`: `src/research/strategy_core.py:613`
- `_enter_trade`: `src/research/tier2_streaming_working.py:1624`

## Dev Agent Record

### Agent Model Used

claude-sonnet-4-6

### Debug Log References

None — clean implementation.

### Completion Notes List

- Task 1: `scaling_plan: list[dict] = field(default_factory=list)` added to `AccountConfig`. Import of `field` from `dataclasses` added.
- Task 2: `calc_contract_limit()` added to `strategy_core.py`. Sorts descending and returns first matching tier. Returns None for empty plan.
- Task 3: `_accumulated_profit` field added to `RiskManager`. `register_close()` now increments both `_daily_pnl` and `_accumulated_profit`. Lifetime state — restored regardless of day match in `restore_from_state`.
- Task 4: Two-cap logic in `_enter_trade()` — consistency first (min to 1), then scaling limit (min to plan result). Both caps combine cleanly.
- Task 5: 6 pure function tests in `test_strategy_core_scaling.py` + 3 RiskManager tests in `test_risk_manager.py`. All pass.

### Review Findings

### File List

- `src/research/strategy_core.py` — add `calc_contract_limit()`
- `src/research/tier2_streaming_working.py` — AccountConfig (`scaling_plan` field, `field` import), RiskManager (`_accumulated_profit` + `accumulated_profit` property, `register_close` update, `to_state_dict`/`restore_from_state` extensions), `_enter_trade` (dual-cap logic)
- `tests/unit/test_risk_manager.py` — extended with 3 accumulated profit tests
- `tests/unit/test_strategy_core_scaling.py` — new, 6 pure function tests

## Change Log

- 2026-06-01: Story implemented — calc_contract_limit, AccountConfig.scaling_plan, RiskManager._accumulated_profit, dual-cap in _enter_trade. 9 new tests passing.
