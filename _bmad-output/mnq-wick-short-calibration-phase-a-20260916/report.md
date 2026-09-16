# MNQ wick-short Phase A historical calibration

Calibration only. Original gate: **POWER_UNDETERMINED**; `evaluation_allowed=false`.

Sample role: `calibration-development-only`. Completed: 2026-09-16T18:24:57.334094+00:00.

Counts reconciled to the original gate **before** aligned outcomes: {"eligible_sessions": 515, "excluded_sessions": 61, "exclusion_reasons": {"INCOMPLETE_RTH_MINUTES": 23, "MIXED_CONTRACT": 38}, "post_cutoff_records_skipped": 62277, "rth_sessions_seen": 576, "sessions_with_signal": 351, "sessions_without_signal": 164, "signals_per_eligible_session": 1.1359223300970873, "total_signals": 585}

All eligible zero-signal sessions are included in session estimands; excluded sessions are separate.

| Estimand | N | Total | Mean | Sample SD | Session G / df / SE / 95% CI | Week G / df / SE / 95% CI | Envelope |
|---|---:|---:|---:|---:|---|---|---|
| Gross $/signal | 585 | -2390.0 | -4.085470085470085 | 50.9390874237179 | 351 / 350 / 2.03725749633022 / [-8.092276831073987, -0.07866333986618379] | 112 / 111 / 2.2042377059350566 / [-8.453314248242034, 0.2823740773018635] | [-8.453314248242034, 0.2823740773018635] |
| Net $/signal (cost $1.22) | 585 | -3103.7 | -5.305470085470085 | 50.9390874237179 | 351 / 350 / 2.03725749633022 / [-9.312276831073987, -1.2986633398661835] | 112 / 111 / 2.2042377059350566 / [-9.673314248242033, -0.9376259226981363] | [-9.673314248242033, -0.9376259226981363] |
| Net $/signal (cost $2.22) | 585 | -3688.7 | -6.305470085470085 | 50.9390874237179 | 351 / 350 / 2.03725749633022 / [-10.312276831073987, -2.2986633398661835] | 112 / 111 / 2.2042377059350566 / [-10.673314248242033, -1.9376259226981363] | [-10.673314248242033, -1.9376259226981363] |
| Net $/signal (cost $3.22) | 585 | -4273.7 | -7.305470085470085 | 50.9390874237179 | 351 / 350 / 2.03725749633022 / [-11.312276831073987, -3.2986633398661835] | 112 / 111 / 2.2042377059350566 / [-11.673314248242033, -2.9376259226981363] | [-11.673314248242033, -2.9376259226981363] |
| Signals/eligible session | 515 | 585.0 | 1.1359223300970873 | 1.1040123470496785 | 515 / 514 / 0.0486485965520263 / [1.0403477837768662, 1.2314968764173084] | 113 / 112 / 0.049473197510384696 / [1.0378975309582026, 1.233947129235972] | [1.0378975309582026, 1.233947129235972] |
| Gross $/eligible session | 515 | -2390.0 | -4.640776699029126 | 52.463365830447614 | 515 / 514 / 2.311812114119373 / [-9.182539661260916, -0.09901373679733538] | 113 / 112 / 2.5412183946558455 / [-9.675875201645429, 0.39432180358717783] | [-9.675875201645429, 0.39432180358717783] |
| Net $/eligible session (cost $1.22) | 515 | -3103.7 | -6.026601941747573 | 52.529684769761396 | 515 / 514 / 2.314734475749708 / [-10.57410614641591, -1.4790977370792362] | 113 / 112 / 2.5554042567658435 / [-11.08980791171835, -0.963395971776797] | [-11.08980791171835, -0.963395971776797] |
| Net $/eligible session (cost $2.22) | 515 | -3688.7 | -7.16252427184466 | 52.609704559606854 | 515 / 514 / 2.318260568988405 / [-11.716955804030263, -2.6080927396590567] | 113 / 112 / 2.5680317103139756 / [-12.250749922369364, -2.074298621319956] | [-12.250749922369364, -2.074298621319956] |
| Net $/eligible session (cost $3.22) | 515 | -4273.7 | -8.298446601941746 | 52.71273026710541 | 515 / 514 / 2.322800424083303 / [-12.86179708805011, -3.7350961158333833] | 113 / 112 / 2.581545662854534 / [-13.41344841713789, -3.1834447867456035] | [-13.41344841713789, -3.1834447867456035] |

Distributions, ECDFs, cluster sizes/concentration, null reasons and complete provenance are in `report.json`; individual observations are in the three JSONL ledgers.

- Calibration only: all historical rows were previously exposed; none are untouched confirmation evidence. No profitability or future-edge verdict.
- Historical next-open/close prices do not establish timely signal availability, executable fills, routing latency or measured costs.
- Costs are the frozen $1.22 fee plus $0/$1/$2 adverse execution assumptions; they are not reconstructed historical charges or measured future costs.
- Approximate Student-t intervals assume finite variance, adequate cluster information, dependence represented within clusters and independence across clusters. Weekly clustering does not resolve cross-week dependence, regime change, feed errors or prior design exposure.
- The outer interval is a sensitivity envelope, not a third confidence procedure; neither clustering nor the envelope guarantees 95% coverage.
- Calendar/contract transition flags are descriptive observations only; they introduce no eligibility or trading filters. Calendar closures are not inferred from absent dates in this input.
- Original POWER_UNDETERMINED and evaluation_allowed=false remain binding. No fresh power verdict, Phase B, confirmation, or trading is performed.

Output byte hashes are in `manifest.json`, including this report. The manifest's own hash is in `COMPLETE.json`; its final Git commit binds both without a self-referential hash.
