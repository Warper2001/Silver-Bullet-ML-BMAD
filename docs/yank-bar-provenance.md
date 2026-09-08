# Frozen YANK bar provenance

Run from the repository root with Python 3.11 or later:

```sh
python -m src.cli.check_yank_bar_provenance --output-dir .local-provenance/run-1
# Installed Poetry entry point:
check_yank_bar_provenance --output-dir .local-provenance/run-2
```

The directory must be fresh. Existing directories, symlink collisions and overlap
with evidence directories are rejected. The interface accepts only the output
path. A conventional linked Git worktree can read fixed inputs from its common
repository root. No credentials, network calls, strategy modules or native DBN
processing are involved.

The CLI privately loads the fixed sibling `validator.py` with `importlib`, retaining
its real file path for pins and code hashes. This small loader avoids the eager
`src.research` initializer. Normal research-package imports still load legacy
backtester, data/authentication and detection dependencies; those existing APIs
remain unchanged. The isolated CLI does not import them.

The fixed identities live in `src/research/yank_bar_provenance/pins.json`.
Supporting manifest expectations were copied from the adopted research's
`artifact-hashes.json`; its own bytes are separately pinned. The four primary
inputs are the frozen annual CSV, mixed-year raw JSON, research's 2025 extract,
and immutable audit reconciliation imported from commit `84f2382`. Every required
file is hash checked before and after analysis. The scanner reads timestamp
strings in the mixed-year source, decoding only 2025 objects. Each extracted
object and source line must agree with that scan.

The exact Decimal reconstruction consumes whole source records ordered by UTC
instant and original ordinal. It emits at accumulated `Close * TotalVolume * 20
>= 50000000`, resets only after emission, and compares timestamp plus all six
numeric fields against every CSV row. It rejects duplicate source timestamps,
leftovers, malformed values, and missing/extra records. The observed multiplier
and threshold express mathematical compatibility, not their economic validity.

All five canonical outputs have schema version 1 and stable content:

- `report.json`: result, observed and expected aggregate counts, fixed assumptions,
  verification results and evidence limitations.
- `report.md`: human readable counts, suitability blockers and audit qualifications.
- `lineage.jsonl`: one object per CSV row, with one-based ordinal and physical CSV
  line, every raw ordinal/line and extract line with file hashes, constituent
  labels/contracts, exact reconstructed numeric strings and any differences.
  Aggregation and contract mixing are independent classifications. Timing retains
  inherited last labels, unknown actual start/arrival/availability, half-open
  conditional start/end bounds and explicitly enumerated gaps over 60 seconds.
- `pilot-differences.jsonl`: one object per fixed covered pilot label, paired
  capture/exchange native OHLCV, deltas, original reconciliation line/hash,
  source clock name and coverage qualification. Native comparisons are reused
  research evidence; native `MNQM5`/`42009475` and raw contract labels remain
  separately attributed. Category overlap is not causal resolution.
- `manifest.json`: expected/observed input hashes, actual package/CLI/pins code
  hashes and hashes of the other four files. A manifest cannot hash itself;
  verification across runs must compare all five files including the manifest.

Outputs omit wall-clock execution time and destination names. Compare two fresh
runs byte for byte to verify reproducibility. Large lineage files belong in the
ignored `.local-provenance/` directory; compact measured reports and the external
five-file comparison are in `docs/reports/yank-bar-provenance/`.

A successful command exits zero with `PASS_PROVENANCE_CHECKS`, while explicitly
retaining `data_suitability: BLOCKED` and `research_status: HOLD_VALIDATION`.
Failures exit nonzero and write `FAIL_PROVENANCE_CHECKS` diagnostics where safe;
partial lineage may remain for diagnosis but no success manifest is published.
No existing evidence is overwritten.

Neither numeric agreement nor conditional interval bounds authenticates origin,
continuous observations, timing, availability, economic multipliers or a roll
policy. These diagnostics create no replacement data or strategy results and
does not establish data readiness for a new backtest. All frozen execution qualifications remain:
five cases, eight arm orders, 30 timing scenarios, 11 supported/19 unassessable
outcomes, 0/100/500 ms delays and 240 scheduled opportunities. The May 28 case 4
ambiguity and conditional evidence remain; pauses do not settle the 11,819
locked/crossed events. The 174-contract boundary example permits no adjustment.
