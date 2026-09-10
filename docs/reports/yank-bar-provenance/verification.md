# Provenance implementation verification

The reviewed implementation passed **44 new tests** and **133 existing replay/accounting/policy tests**. The legacy suite ran at sibling revision `b6626622e3468c932722409530120d60f29c6ce6`; both JUnit files and exact commands are retained here.

Final runs `.local-provenance/verified-run1` and `verified-run2` exited zero. All five canonical files match byte for byte and by SHA256. `verification.json` records sizes and hashes; the compact report and manifest here are unchanged copies from verified-run1. Full lineage and pilot extracts remain in those ignored local directories.

The independent checker directly compared the original CSV timestamp and six numeric fields on every row, recomputed threshold emissions and verified all 351,628 constituents exactly once. It passed under `python3 -O`; optimized-interpreter tamper tests confirm checks remain active. The checker is a fixed local research utility using `/root/Silver-Bullet-ML-BMAD` for evidence.

```sh
python3 -O docs/reports/yank-bar-provenance/verify-lineage.py .local-provenance/verified-run1
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m src.cli.check_yank_bar_provenance --output-dir .local-provenance/new-run1
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m pytest tests/unit/yank_bar_provenance -q
```

Run from the provenance worktree; choose an unused output directory for each rerun. All 18 original inputs/archive files matched their before-run hashes. Details are in `parent-verification.json`. The CLI's fresh-process import test blocks legacy research, data/authentication and detection dependencies; existing package APIs remain unchanged.

Review fixes covered required pin packaging, complete Decimal context isolation, fractional gap durations, interruption cleanup, independent CSV verification, published pilot attribution and baseline rejection. No actionable review findings remain. The tests emit two pre-existing Pydantic deprecation warnings through test fixtures; the CLI avoids those legacy imports.

**PASS_PROVENANCE_CHECKS** retains **BLOCKED** suitability and **HOLD_VALIDATION**. No replacement dataset, signal rerun or changed execution outcome was produced.
