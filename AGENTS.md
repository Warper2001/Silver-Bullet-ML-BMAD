<!-- bmad:context -->
<!-- Verified 2026-09-11 against 4946c69. Managed by bmad-project-context; edits inside this block are replaced on refresh. Keep anything you want preserved outside the markers. -->

## Silver-Bullet-ML-BMAD

Quant research plus live trading bots for one operator: MNQ on a Topstep combine via ProjectX, TradeStation SIM paper accounts, and Kraken crypto. Python 3.12 in `.venv`; dependencies declared in `pyproject.toml` (Poetry, no lock file). The live bots are systemd `trader-*` units running straight from this checkout. Pre-registrations, results, and runbooks live in `_bmad-output/`.

## Policy

- Never change a strategy parameter — in `strategy_config.yaml`, an `Environment=` line of a `trader-*` unit, or a constant hardcoded in a trader file — before committing a pre-registration to `_bmad-output/preregistration_*.md`; the YAML header names the current seal.
- Never read `data/sealed_holdout/` without the committed pre-registration and log entry that `data/sealed_holdout/ACCESS_LOG.md` requires.
- Never adopt a filter or restriction chosen on favorable past data without retesting it on data the choice did not see; treat ambiguous evidence (50th–90th percentile vs random) as a FAIL.
- Never put a hand-set threshold in a sealed doc — derive it from a sweep and cite artifact and commit; until then monitors observe and report only. Change one parameter per experiment.
- Run a power gate before any new strategy test; UNDERPOWERED is a valid verdict. Template: `tools/xsmom1_power_gate.py`.
- Never edit code the live bots run in this checkout — a crash restart (`Restart=on-failure`) loads whatever is on disk. Work in a worktree under `.claude/worktrees/`, merge into `main` here, verify, then restart.
- Never deploy by copying files from a worktree or branch; merge and diff instead — this checkout has held fixes the branches lacked.
- After a verified merge you may `systemctl restart` the affected `trader-*` unit; any other trader start/stop, and killing any process, needs asking first.
- Never `pip`/`poetry install` into `.venv` — the live bots run on it; research libraries go in `.venv-research` (see `pyproject.toml`).
- Never commit credentials (`.env`, `.access_token`).

## Where things are

- Live bots: `systemctl list-units 'trader-*'`; `systemctl cat <unit>` shows entry file and env. Logs are mostly `logs/<entry-file-stem>.log`.
- YANK, the live Tier2 FVG strategy: `src/research/yank_streaming_working.py`. `tier2_streaming_working.py` is no longer a service, but YANK imports from it.
- Live bots also import `src/research/strategy_core.py`, `src/data/auth_v3.py`, `src/data/models.py`, `src/monitoring/trade_db.py`, `src/execution/kraken/`, `src/ml/regime_detection/` — edits there reach them on restart.
- YANK's ML filter loads `models/xgboost/tier2_meta_labeling_model.pkl` and `tier2_threshold.json`; the presence of `models/xgboost/lr_regime_config.json` switches its LR regime filter on.
- The async pipeline (`src/data/orchestrator.py`, `src/detection/`, `src/ml/pipeline.py`, `config.yaml`) is wired to no live bot; when working there, read `docs/async-pipeline.md`.
- Trade ledger: `data/trades.db` (gitignored, live-appended). 1-min bars: `data/processed/dollar_bars/1_minute/`.
- Tier2-engine replays: `backtest_tier2_1year_validation.py` (1 year), `backtest_tier2_today.py` (today's session); they read `strategy_config.yaml` but not YANK's unit-file overrides.

## Running and verifying

- Run Python as `.venv/bin/python` — bare `python` is not on PATH. `tools/validation/` scripts run under `.venv-research/bin/python`.
- Tests: `.venv/bin/python -m pytest tests/<path>`; `make test` is a stub that runs nothing. Examples: `.venv/bin/python -m pytest tests/integration/test_ml_pipeline_integration.py -v`; single test `.venv/bin/python -m pytest tests/integration/test_orchestrator_integration.py::TestPipelineEndToEnd::test_pipeline_initialization -v`; coverage `--cov=src` needs `pytest-cov`, which `.venv` lacks.
- Format and lint with `.venv/bin/black src/ tests/`, `.venv/bin/flake8 src/ tests/`, `.venv/bin/mypy src/`; `make format`/`make lint` call `poetry`, which is not on PATH.
- Seal a config change: `PYTHONPATH=. .venv/bin/python prereg_seal.py --name <id> --config strategy_config.yaml --output _bmad-output/preregistration_<id>.md`, then `git add -f` it (`_bmad-output/` is gitignored) and commit before editing config.
- Weekly check: `PYTHONPATH=. .venv/bin/python tools/weekly_backtest.py --weeks 4` (needs fresh post-holdout TradeStation data). Before any holdout access: `PYTHONPATH=. .venv/bin/python oos_checkpoint.py --prereg <doc> --config strategy_config.yaml`.
- Launch anything over ~30s as `nohup … > <log> 2>&1 &` and tail the log; the CPU is slow and replays run 1h+. Time a Tier2/YANK replay on a slice before sizing a job — its wall-clock is not reproducible.

## Conventions that differ from defaults

- Don't add mypy `ignore_errors` overrides to `pyproject.toml` (the `src.data.*`/`src.ml.*` ones are legacy); type new code instead.
- Tests mock `TradeStationAuth` — never require live credentials. Async tests need `@pytest.mark.asyncio` (strict mode).

## Known pitfalls

- Before any git operation here, run `git status --porcelain` and count both `git log origin/main..HEAD` and `HEAD..origin/main` — uncommitted live hotfixes and one-sided divergence have each silently lost fixes.
- `data/trades.db` `pnl` is mostly backfilled backtest rows — filter `write_mode = 'realtime'` for live trades, and parse `timestamp` with `format="ISO8601"` (mixed formats).
- Replaying bars through a trader class writes real trade-log rows under `logs/`, and YANK's also writes `data/trades.db` outside backfill mode; only `tier2_streaming_working.TradeLogger(persist=False)` is sandboxed — use it or a standalone pandas engine.
- Tuesday exclusion and the M15 CHoCH constants (`SWING_R`, `CHOCH_ATR_MULT`) are hardcoded in the trader files; `tuesday_exclusion` in the YAML has no effect.
- On a combine account reset, update `COMBINE_EPOCH_START_FALLBACK` in `src/research/mim_nb_live.py` and `PROJECTX_ACCOUNT_ID` in every unit that sets it — a stale epoch leaks the retired account's trades into the new balance.

<!-- /bmad:context -->

## Methodology history (kept from CLAUDE.md, Program C as of 2026-05-24)

Program C evidence chain, preserved at the maintainer's request. Treat it as history: the 2026-05-20 methodology reset marks all earlier performance claims as tentative.

- Phase 1 (S12 + S13): S12 AMBIGUOUS (1m 70th pct of random null) → PIVOT → P1 (15m). S13 PATTERNS SURVIVE (15m PF=1.179, TIME_STOP 65%→11%).
- Phase 2 OOS (holdout 2026-03-01 to 2026-05-19): N=6, PF=2.586 → PASS (weak, N=6 caution).
- Epic 2 enhancements (BIDIR, KZ, M15CONF, VOL): all H₀ — baseline wins.
