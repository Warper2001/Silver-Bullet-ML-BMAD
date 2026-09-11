# Deploy runbook — 2026-09-11 fleet fixes

**Branch:** `feat/research-validation-stack` (pushed; commits `1552ed8`, `700bed5`, `3e8780e`)
**What is already live:** the ledger and database changes (see §0). **Only the code is undeployed.**
**Why a human:** the session that made these changes was worktree-isolated and its sandbox
refuses git operations against the shared checkout. Every command below must be run in
`/root/Silver-Bullet-ML-BMAD`.

---

## 0. Already applied to live data — do NOT redo

| Change | State |
|---|---|
| `data/trades.db` — s26-combine relabelled `paper` (81 rows) | done |
| `data/trades.db` — `write_mode` backfilled, 0 NULL | done |
| `data/trades.db` — `execution_mode` backfilled, 0 NULL | done |
| `data/thursday_ts/*.csv` — 08-27 + 09-03 rows reconstructed | done, chains verify 7/14/12 |

Backups: `/root/backups/fleet_fixes_20260911/` plus in-place
`data/trades.db.pre-fix-*.bak` snapshots taken by the script itself.

## 1. Pre-flight

```bash
cd /root/Silver-Bullet-ML-BMAD
git status --porcelain          # MUST be inspected before anything else
git log --oneline -1
git rev-parse HEAD > /root/backups/fleet_fixes_20260911/HEAD-before.txt
```

**Anything uncommitted must be committed to a branch first.** This checkout has
previously held fixes that existed *only* in its working tree — on 2026-09-10 a naive
file copy would have reinstated the bug that blocked every YANK entry for 2.5 days.
Untracked `data/reports/*.csv` are backtest output and can be ignored.

## 2. Merge

```bash
git fetch origin
git log --oneline origin/main..HEAD | wc -l     # expect ~21 (local ahead)
git log --oneline HEAD..origin/main | wc -l     # expect ~15 (local behind)
git merge origin/main
git merge feat/research-validation-stack        # this session's fixes
```

Resolve any conflict **by inspection**, never by taking one side wholesale.

## 3. Verify before restarting anything

```bash
# The hotfix that must survive: renamed attribute, expect 2 hits
grep -c "_shadow_trade_logger" src/research/yank_streaming_working.py     # expect 2

# The two things the merge is FOR
grep -n "data/thursday_ts" .gitignore                                     # must exist now
grep -n "write_mode" src/monitoring/trade_db.py | head -3                 # must exist now

# Nothing unintended moved
md5sum -c /root/backups/fleet_fixes_20260911/MD5SUMS-live-before 2>&1 | grep -v ': OK'

PYTHONPATH=. .venv/bin/python -m pytest \
  tests/unit/test_decision_log.py \
  tests/unit/test_trade_db_provenance.py \
  tests/unit/test_thursday_short_exit_code.py -q     # expect 20 passed
```

## 4. Restart, one at a time

Check each is clean before starting the next.

```bash
for svc in trader-yank trader-mim-nb trader-gap-fade trader-s26 \
           trader-s27 trader-s26-combine trader-thursday-short trader-btc-carry; do
  echo "=== $svc ==="
  systemctl restart "$svc"
  sleep 15
  systemctl is-active "$svc"
  journalctl -u "$svc" --since "1 min ago" | grep -iE "error|traceback" | head -5
  read -p "continue? " _
done
```

`trader-yank` first and deliberately: it is one of the two bots on real money, and it is
currently flat (no entry since 2026-08-17), so a restart cannot orphan a position.
`trader-mim-nb` is also live — confirm it is flat before restarting it, or wait for its
16:00 ET EOD exit.

## 5. Post-deploy checks

```bash
# The decision log rotates to the new schema on the first live bar, then:
head -1 logs/tier2_bar_decisions.csv       # expect trader_id + rejection_reason columns
ls logs/archive/*_preschema.csv            # the old-schema rows, preserved

# Funnel now answers the question it could not answer before
.venv/bin/python analyze_filter_funnel.py --days 3 --trader trader-yank

# Thursday-short is now monitored
PYTHONPATH=. .venv/bin/python tools/combine_ops_healthcheck.py | grep thursday
```

**Next live trade from any bot** should land with non-NULL `write_mode='realtime'` and a
correct `execution_mode`:

```bash
.venv/bin/python -c "
import sqlite3; c=sqlite3.connect('file:data/trades.db?mode=ro',uri=True)
for r in c.execute('select trader_id,timestamp,write_mode,execution_mode from trades order by id desc limit 5'): print(r)"
```

## 6. Deadline

The Thursday-short restart seal (`_bmad-output/preregistration_kraken_thursday_short_restart.md`)
declares the first counted Thursday as **2026-09-17**. Commit it before then:

```bash
git add -f _bmad-output/preregistration_kraken_thursday_short_restart.md \
           _bmad-output/reconstruction_note_thursday_20260911.md
git commit -m "seal thursday-short restart pre-registration (N=0 from 2026-09-17)"
```

(Both are already committed on `feat/research-validation-stack`, so the merge in §2
brings them; this is only needed if you deploy some other way.)

## 7. Deliberately NOT deployed

`_bmad-output/preregistration_yank_bidirectional_DRAFT.md` is a **draft, uncommitted and
unsealed**, per the 2026-09-11 decision. Its forward evidence is N=2, both losses. It
becomes sealable only if the shadow bullish ledger reaches N ≥ 15 with PF ≥ 1.00; if it
reaches N ≥ 15 below that, discard it. Nothing about YANK's trading config was changed —
S25 remains frozen.

## 8. Rollback

```bash
git reset --hard $(cat /root/backups/fleet_fixes_20260911/HEAD-before.txt)
cp /root/backups/fleet_fixes_20260911/trades.db.bak data/trades.db
cp -r /root/backups/fleet_fixes_20260911/thursday_ts/. data/thursday_ts/
# then restart the services as in §4
```
