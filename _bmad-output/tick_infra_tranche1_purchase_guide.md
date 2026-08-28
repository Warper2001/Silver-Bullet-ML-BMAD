# How to buy TICK-INFRA Tranche 1 in Databento

**Goal:** the minimum MBO data to run the §3 parity gate (replay 129 known live fills,
check the simulator reproduces them). Companion: `preregistration_tick_data_infrastructure.md`
(seal + Amendments 1–7). Machine: KVM 4, 162 GB free — **done**.

**Cost:** ~$2 test download, then **~$30–90** for the re-scoped Tranche 1 (Amendment 7) —
or $453 for the full contiguous slice as a fallback. All fit the KVM 4.

---

## Step 1 — account + billing

1. **databento.com → sign up**, verify email. **$125 historical credit** auto-applied
   (expires 6 months after signup).
2. **Portal → Settings → Billing → add a card** (or prepay credits), and **set a spending
   limit** — Tranche 2 later is a ~$1,200 fat-finger risk.
3. **Portal → API Keys → create a key.** Store it in the repo's **gitignored** `.env` or a
   password manager. Never commit it. (The key pasted earlier in chat is already rotated/dead.)

---

## Step 2 — the ~$2 test download (do this before anything else)

One mid-session hour of one contract. It answers three questions at once.

```python
pip install databento          # once, in the venv

import databento as db
c = db.Historical("YOUR_KEY")

print(c.metadata.get_cost(dataset="GLBX.MDP3", symbols=["MNQM6"], stype_in="raw_symbol",
      schema="mbo", start="2026-06-16T14:00", end="2026-06-16T15:00"))   # ~$1-2

job = c.batch.submit_job(
    dataset="GLBX.MDP3", symbols=["MNQM6"], stype_in="raw_symbol", schema="mbo",
    start="2026-06-16T14:00", end="2026-06-16T15:00",
    encoding="dbn", compression="zstd", split_duration="day")
print(job["id"])
# wait for email / poll c.batch.list_jobs(), then:
c.batch.download(job_id=job["id"], output_dir="data/tick/_test/")
```

**Record these three things (they become Amendment 8):**

1. **zstd ratio** — `.dbn.zst` file bytes ÷ (`record_count` × 56). Multiply by 270 GB → the
   real full-slice disk size. (Expect ~2.5–3.5× → ~80–110 GB, which fits.)
2. **Book completeness at an intraday start** — read the file:
   ```python
   store = db.DBNStore.from_file("data/tick/_test/<file>.mbo.dbn.zst")
   df = store.to_df()
   print(df["action"].head(50).value_counts())
   ```
   If the first records are `R` (clear) then a burst of `A` (add) → Databento rebuilds the
   full book at an arbitrary start → **use Mode C**. If it starts mid-stream with scattered
   `A`/`C`/`M` and no initial rebuild → **use Mode C'**.
3. **Pipeline works** — the download, the `DBNStore` read, and the seal §5 integrity checks
   (timestamps monotonic; reconstruct BBO, `bid ≤ ask` on 100%; trades within session range)
   all succeed on this tiny sample.

---

## Step 3 — pick the acquisition mode

| From the test | Mode | What to buy | Est. cost | Est. disk |
|---|---|---|---|---|
| Book rebuilds at intraday start | **C — targeted windows** | `timeseries.get_range`, ±90 min around each of the 129 trades (87 merged windows), `MNQM6` before ~2026-06-18 / `MNQU6` after | **~$30–90** | ~3–8 GB |
| Book does NOT rebuild intraday | **C' — parent days** | full sessions for the ~13–20 distinct calendar days the windows land on | **~$25–55** | ~5–12 GB |
| C/C' too fiddly, want simplicity | **A — full slice** | `MNQM6,MNQU6,MNQZ6`, 2026-05-01 → 2026-08-28, one batch job | **$453.12** (−$125 = $328) | ~80–110 GB |

All three fit the KVM 4. C/C' save ~$350–420 and leave the box clear for Tranche 2.

---

## Step 4 — buy Tranche 1

### Mode C — targeted windows

Build the window list from `trades.db` (the 129 parity trades → ±90 min → merge overlaps),
then pull each:

```python
import sqlite3, pandas as pd, datetime as dt, pathlib
con = sqlite3.connect("data/trades.db")
q = """select timestamp from trades
       where trader_id in ('trader-mim-nb','trader-gap-fade','trader-s26-combine')
          or (trader_id='trader-yank' and timestamp >= '2026-06-01')
       order by timestamp"""
ts = [dt.datetime.fromisoformat(r[0].replace("Z","+00:00")) for r in con.execute(q)]
wins = []
for t in ts:
    s, e = t - dt.timedelta(minutes=90), t + dt.timedelta(minutes=90)
    if wins and s <= wins[-1][1]:
        wins[-1] = (wins[-1][0], max(wins[-1][1], e))
    else:
        wins.append((s, e))
print(len(wins), "windows")

out = pathlib.Path("data/tick/mnq_mbo_parity/"); out.mkdir(parents=True, exist_ok=True)
for i, (s, e) in enumerate(wins):
    sym = "MNQM6" if s < dt.datetime(2026, 6, 18, tzinfo=dt.timezone.utc) else "MNQU6"
    data = c.timeseries.get_range(
        dataset="GLBX.MDP3", symbols=[sym], stype_in="raw_symbol", schema="mbo",
        start=s.isoformat(), end=e.isoformat())
    data.to_file(out / f"win{i:03d}_{sym}.mbo.dbn.zst")
```

(Cross-check the running spend with `c.metadata.get_cost(...)` summed over the windows
before you start — expect ~$30–90 total.)

### Mode C' — parent days

Same, but expand each window to its full calendar day(s), dedupe the day list, and pull one
batch job per day (or one job covering the deduped set).

### Mode A — full slice

```python
c.metadata.get_cost(dataset="GLBX.MDP3", symbols=["MNQM6","MNQU6","MNQZ6"],
    stype_in="raw_symbol", schema="mbo", start="2026-05-01", end="2026-08-28")  # ~453.12

job = c.batch.submit_job(
    dataset="GLBX.MDP3", symbols=["MNQM6","MNQU6","MNQZ6"], stype_in="raw_symbol",
    schema="mbo", start="2026-05-01", end="2026-08-28",
    encoding="dbn", compression="zstd", split_duration="day")
```

Data availability ends **2026-08-28 22:30 UTC** — do not set `end` later.

> Every job / `get_range` call **bills immediately** (credit first, then card). No undo.
> Confirm `get_cost` before each.

---

## Step 5 — integrity checks (seal §1 / §5, before any parity run)

For each file (or a sample):

```python
store = db.DBNStore.from_file(path)
df = store.to_df()
assert df.index.is_monotonic_increasing
# reconstruct BBO from the MBO stream; assert bid <= ask on 100.000% of book states
# assert trade prices within [session low, session high]
```

Report the exact pass rate. Databento flags **2026-05-24** and **2026-07-30** as `degraded`
(still `available`) — note them, don't exclude.

---

## Step 6 — licensing

First CME request shows a click-through license. Internal research / backtesting
(non-display, non-redistribution) is covered. Do not redistribute the raw data or a derived
real-time feed.

---

## After Tranche 1

Build the §2 simulator → run the §3 parity gate → append the result as **Amendment 9**.
Buy Tranche 2 (RTH bulk, ~$1,150–1,250) only on a PASS. If parity fails within the §4
window (3 cycles / 15 working days), R3 closes — total spend under $100.
