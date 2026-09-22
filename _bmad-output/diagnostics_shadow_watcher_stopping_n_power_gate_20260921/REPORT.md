# Power gate — stopping N for the YANK bullish shadow watcher (2026-09-21)

**Verdict: UNDERPOWERED.** No stopping N that can answer the pre-registration's own question (bullish PF > 1.3) is reachable within a 2-year leash.
Outcome-blind: `logs/yank_shadow_bullish_trades.csv` was never opened (Amendment 4 §4.7 forbids using it as evidence, and a stopping N does not need it);
`data/sealed_holdout/` not touched. Files: `power_gate.py` (decision rule pre-committed in its docstring), `power_gate_output.md`, `power_verdict.json`.

## What the watcher is for, and what the gate measured
The watcher (Amendment 3, deployed 2026-08-19) accrues unseen bullish shadow trades so the bullish leg can be judged on new data. The decision it feeds is
§5's bar, **bullish PF > 1.3**. So the read-out has to be able to *see the mean per trade that PF 1.3 implies*, and that effect is the primary anchor.
Everything comes from the Amendment 4 corrected-bar derivation trades (N=19 bullish, mean $27.18, SD $707.68, PF 1.105; reproduced exactly before use) and two rates.

- **PF 1.3 implies +$77.87/trade** (0.3 x the $259.58 gross loss per trade), i.e. **d = 0.110** of one trade's SD. The derivation window's own point estimate is d = 0.038.
- **Rates:** 0.31 bullish trades/week in the derivation window (19 in 424 days); 1.0/week for the watcher (4 completed in 28 days, taken from the prereg text).

## Result (one-sided 5%, 80% power, DEFF 1.0)
| | trades needed | years at 0.31/wk | years at 1.0/wk |
|---|---|---|---|
| **PF-1.3 effect (primary)** | **511** | **31.3** | **9.8** |
| same, SD x0.75 / x1.25 | 287 / 798 | 17.6 / 48.9 | 5.5 / 15.3 |
| 1.5 x the bar (a much bigger edge), SD x0.75 | 128 | 7.8 | 2.5 |
| 0.5 x the bar | 2,042 | 125 | 39 |
| derivation point estimate (+$27.18) | 4,190 | 257 | 81 |

- **Rate uncertainty does not rescue it.** Four trades is a noisy rate (exact Poisson 95%: 0.27 to 2.56/week). At the *upper* rate the primary N still takes **3.8 years**
  (2.2 at SD x0.75). Only the corner "1.5x the bar with SD x0.75 at the upper rate" (1.0 year) fits the leash.
- **DEFF 1.5** (clustering) makes every cell worse (766 trades for the primary anchor).
- **Simulated power** (pooled bullish+bearish trade shape, n=60; size at d=0 is 0.043): at the PF-1.3 effect, **0.82 at 511 trades** (so 511 is the right N *if* it could be reached), but
  **0.10 at 33 trades and 0.27 at 104 trades**, the counts a 2-year leash actually delivers at the two rates.

## What a leash-bounded read-out could and could not say
Inside 2 years the watcher delivers about 33 trades (0.31/wk) to 104 (1.0/wk) (266 at the upper Poisson rate). The smallest mean it could then detect at 80% power is
**$308, $173 or $108 per trade, i.e. 3.96x, 2.22x or 1.38x the PF-1.3 effect.** By the same gross-loss arithmetic, $173/trade is roughly PF 1.67.
So a 2-year read-out could confirm only an edge well above the pre-committed bar, and **a null would mean "not that large", not "below PF 1.3".** It could never confirm the bar itself.

## What this means for keeping the watcher (Alex's call)
- **Keep it as instrumentation only.** It cannot decide anything by itself; there is no N at which a read-out settles the PF-1.3 question in any reasonable time.
- **If a stopping rule is wanted anyway,** the only defensible one is leash-based, with its detection limit written in: stop at the earlier of about 104 completed unseen shadow trades
  or 2 years from 2026-09-17, read out only "mean >= $173/trade?", and say in advance that a null does not refute the bar. It needs an Amendment 5 authorized by Alex, citing this artifact and its commit.
- **Or retire it** (cost to retire is nil; it is read-only and carries no capital).
- The four shadow trades to 09-16 stay excluded, as §4.7 requires; the accrual window for any read-out starts 2026-09-17.

## Caveats
- SD and shape come from 19 bullish (60 pooled) trades on a window already mined for three hypotheses (Amendment 2); both are noisy.
- The shadow rate is 4 trades; the live watcher runs under conditions the backtest did not (ML filter disabled since 09-15, hardcoded Tuesday exclusion), so its rate may differ.
- The 2.0-year leash is the operator convention of the earlier gates, not a derived number; the grid lets any other leash be read off.
- The PF-1.3 mean is derived by holding gross loss per trade fixed at the observed value, an approximation of "the effect the bar implies".
- The gate says nothing about whether the shadow trades faithfully match the sealed spec.
