# Tier2/YANK back-month replay: plan

**Written:** 2026-09-14, after `tag_census.py` and before any replay. It is committed before the replays run.

## Question

Does YANK's pre-cutoff census result (2025-05-19 → 2026-02-28, the replay behind the June seal) depend on Jan–Feb 2026 bars that came from the thin deferred contract instead of the front month?

## Evidence so far (`tag_results.json`)

**Roll-week splices:**
- **No census signal** (ML0.50 or no-ML) falls in an interleaved roll-week session.
- None has a contract switch in its prior 24 hours.

**Jan–Feb 2026** (`mnq_1min_2026_ytd.csv`, the deferred MNQM26, 1–6 contracts a minute):
- It holds **40 of 68 ML0.50 trades (+$5,411 of +$6,297.50)** and 54 of 91 no-ML trades.
- January 2026 alone has 31 ML trades (+$6,387). Every 2025 month has 1–7 trades.
- The 25 clean-2025 ML trades net −$1,329.50.

## Runs

All runs use the unmodified `tools/tier2_census.py` with the 2025-05-19 → 2026-02-28 window and `--pin max_daily_loss=-750` (the June seal replay), with nice 19.

**Setup:**
- Each run uses its own replay root: a symlink farm to this worktree's code, with its own `data/` and `logs/`.
- `models/xgboost/tier2_threshold.json` and `lr_regime_config.json` are linked from the main checkout; the model pickle is tracked and byte-identical.

| Run | 2025 bars | Jan–Feb 2026 bars | Arm |
|---|---|---|---|
| **G0** | `mnq_1min_2025.csv` | `mnq_1min_2026_ytd.csv` (original) | ML0.50 |
| **C-ml** | `mnq_1min_2025.csv` | raw MNQH26 front-month 1-minute bars (`build_front_month_2026.py`) | ML0.50 |
| **C-noml** | same as C-ml | same as C-ml | no-ML |

**Gates:**
- **G0 must reproduce** `census_trades_ml050.csv` (2026-09-13) row for row. If it does not, the corrected runs are not interpreted.
- **Continuity:** trades before 2026-01-01 should match between G0 and C-ml. The bars are identical before then; state carried from late December is the only source of difference. Any mismatch is reported.

## Reported (descriptive; no adoption threshold)

- Trades, P&L, PF and wins per month for G0, C-ml and C-noml.
- Jan–Feb 2026 totals, and whole-window totals.
- 1-minute bar count per day, Jan–Feb, original vs front month.

## What the result can and cannot do

- **It can say** whether Jan–Feb 2026 evidence for YANK came from the deferred contract's data.
- **It changes no strategy parameter, threshold or live setting.** It spends no holdout data, since the cutoff is enforced by the tool.
- If the corrected Jan–Feb result differs materially, any claim that relied on it is marked affected. That includes the June seal's in-sample numbers and the 2026-09-13 entry-mechanics diagnostics. Re-validating YANK is a separate decision.
