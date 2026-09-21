# YANK sizing and provenance follow-up — 2026-09-07

**The saved P&L reconciles exactly with historical dynamic sizing. Research status remains HOLD_VALIDATION.** The initial fixed-five-contract check exposed an undocumented account-policy effect, not an unexplained arithmetic error.

## Correction to the initial audit

The initial audit inspected `src/research/yank_streaming_working.py` at the preregistration commit. The actual historical runner imports `src/research/tier2_streaming_working.py` (runner lines 43–44). These similarly named implementations have different sizing behavior. The initial module reference was insufficient; this follow-up corrects it.

At commit `138cab1b31d064555ede4c9c07503399a743893f`, the imported implementation:

- Sets default MNQ size to five contracts (line 76).
- Defines a 45% concentration threshold (line 101).
- Calculates best positive session P&L divided by the sum of positive session P&Ls; this describes the repository algorithm, not a verified account-provider rule (`strategy_core.py`, `calc_consistency_ratio`).
- Latches a size reduction for the risk day once the threshold is reached (lines 483–507).
- Caps the entry to one contract while that flag is active (lines 1982–1986), stores that quantity on the active trade (line 2027), and uses it in closing P&L (line 1629).
- Applies the same flat $4 commission deduction to either size. This reproduces the saved calculation; actual account costs are not established.

## Independent saved-ledger reconstruction

[trace_sizing.py](trace_sizing.py) reconstructs quantities from prior reconstructed session P&L, then computes each trade's P&L from its exported entry/exit prices. Reported P&L is used only for comparison, never as the input determining subsequent size. No trading module is imported, no model is deserialized, and no bars or broker services are accessed.

| Export | Trades | One contract | Five contracts | Reconstructed net | Maximum P&L residual |
|---|---:|---:|---:|---:|---:|
| ML-associated 181838 | 82 | 23 | 59 | $7,804.00 | $0.00 |
| Baseline-associated 185354 | 107 | 64 | 43 | $1,748.00 | $0.00 |
| Duplicate ML 214013 | 82 | 23 | 59 | $7,804.00 | $0.00 |

All 189 trades in the two distinct exports reconcile. The duplicate ML run reconciles too. Detailed quantities, prior concentration ratios, risk dates, residuals and source hashes are in [sizing-trace.json](sizing-trace.json).

The risk-day convention matters. The replay advances/exits an existing trade before detecting a new entry. Detection skips risk-day updates while a position is active; closure registers P&L before the subsequent risk-day reset. The reconstruction therefore assigns the trade's P&L to its entry risk day. Three ML trades and five baseline trades cross Eastern calendar dates. A simpler exit-date attribution check disagreed with nine baseline quantities; entry-risk-day attribution reconciles all quantities. This is a reproduction of historical behavior, not approval of that convention for future account controls. The exported trades are non-overlapping, which the script verifies.

## What the result means

The mixed sizes are consistent with the known historical policy. Retiring the exports solely for arithmetic inconsistency is unwarranted. They support an internally coherent saved-ledger result for a dynamic-size policy.

They do not establish a fixed-five-contract YANK result, isolate ML's predictive contribution, or establish executable profitability. ML and baseline take different trades and accumulate different account states, so size reductions differ substantially between arms. Do not rescale the saved trades to manufacture a new backtest: changing size changes subsequent concentration and daily-loss decisions.

## Provenance still incomplete

The preregistration SHA identifies an available source snapshot. The historical runner's `verify_preregistration` only checks that the supplied commit exists and names a preregistration document; it does not check out that commit or prove the executing working tree matches it. The report export does not embed configuration, model hash, source hash, quantity or fee breakdown.

The model path is tracked at the historical commit. Its blob is hashed without unpickling in this audit. That establishes the available snapshot's identity, not which model file the original running process actually loaded. Model training-range provenance is still based on the results note's assertion.

The constructor creates a fresh RiskManager, and the inspected replay does not call `initialize()`, which contains state restoration. This reduces the hypothesis of restored account state for that source snapshot. It does not prove the original process had an identical working tree or loaded model.

The runner mocks broker methods and state saving but does not suppress all logging helpers. No replay was run here; a future isolated harness needs to contain those side effects as well as enforce fresh state.

## Next useful work

Build a research-only replay harness with explicit dynamic sizing policy, risk-session convention, quantities, costs and input/configuration/model/code identities in its output. First verify it with synthetic entry/exit sequences and an independent cash reconciliation. Then reproduce an already-consumed development window under fixed assumptions. Fresh validation should be specified only after this passes.

Existing limitations remain: reused strategy-selection data, concentrated returns, unmeasured execution friction, and weak transfer evidence. The audit does not authorize a fresh holdout look, parameter sweep, deployment or recurring automation.

## Verification

Both saved-ledger scripts run successfully. Repeated execution produces byte-identical JSON. The sizing reconstruction verifies ordering, non-overlap, every P&L residual, the independently recorded full-period totals, and equality of the duplicate ML records. A deliberately corrupted reported P&L is detected by the residual check. This is forensic validation of saved output, not a strategy backtest.
