# MIM profit-giveback result

## Verdict

**UNDERPOWERED — terminal.** The approved 500-session / 30-month program cannot distinguish the minimum acceptable payoff change with adequate power. No aligned historical candidate return, threshold sweep, prospective freeze, or strategy recommendation was produced.

The definitive run is `runs/20260913T184305-power-de068376a1`; full independent replay verification is `runs/20260913T184823-verify-c3e57d363c`. The terminal inventory denial is preserved in `runs/20260913T185428-inventory-1934561818` and independently verified by `runs/20260913T185432-verify-33d96073db`. Three earlier full runs produced the identical verdict and numerically identical per-threshold table.

## Reconciliation and power

The firewalled engine reproduced arm A at 1,323 sessions, 801 trades, $21,889.76 net, PF 1.2856688, and 71 catastrophe-stop / 723 EOD / 7 reversal exits. It evaluated nine mechanically derived giveback-ratio quantiles using five non-identity session shifts. Identity pairings and aligned candidate returns both remained at zero.

PF 1.40 with 90% winner-dollar retention implies a conservative power effect of $2.6024 per session; the prospective rule separately requires at least 90% of baseline net. At 500 sessions, the family-adjusted 80% minimum detectable effect ranged from $6.1163 to $22.1912 per session; achieved power ranged from 13.55% down to 1.61%. Under square-root sample-size scaling, individual thresholds would require roughly 2,760 to 36,350 sessions to detect the minimum effect, with the complete-family gate governed by the upper end.

## Decision

Do not spend exposed history on a giveback threshold sweep and do not start the prospective collection program under the approved economics and horizon. Reconsideration would require a separately justified, preregistered effect size large enough to be detectable or a much larger independent sample; changing the hurdle after this result would not rescue this experiment.

No live code, strategy parameter, service, broker, collector, or sealed holdout was accessed or changed.
