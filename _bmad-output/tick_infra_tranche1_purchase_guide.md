# How to buy TICK-INFRA Tranche 1 in Databento

**What you are buying:** `mbo` (full order book) for the MNQ front-month contracts,
**2026-05-01 → 2026-08-28**, ~270 GB uncompressed / **~80–110 GB on disk** as `.dbn.zst`,
**$453.12** before the $125 credit (**$328.12 charged**). Parity-gate slice only (Amendment 2/4/6). Not the bulk.

**Before you spend $328:** do Step 2.5 — a ~$2 one-day test download — to measure the real
compression ratio and validate the whole pipeline. And upgrade the VPS to KVM 4 first
(Step 1.5): Tranche 1 does **not** fit the current KVM 2's 64 GB free (Amendment 6).

Companion doc: `_bmad-output/preregistration_tick_data_infrastructure.md` (the seal + Amendments 1–5).

---

## Step 0 — decide the symbology (pick one)

| Option | Symbols / stype_in | Cost | Billable | Get |
|---|---|---|---|---|
| **Raw contracts (recommended)** | `MNQM6,MNQU6,MNQZ6` / `raw_symbol` | **$453.12** | 270 GB | Complete books for **every** contract — both sides of the June→Sept roll. Honors the seal's "per-contract, no stitching" exactly. MNQZ6 is ~$15 of insurance. |
| Two contracts | `MNQM6,MNQU6` / `raw_symbol` | $437.80 | 260 GB | Same, without the Dec contract (no live trade rolled that early — safe to drop). |
| Volume-rolled continuous | `MNQ.v.0` / `continuous` | $412.65 | 246 GB | A single stream that is the volume-front contract at each moment; every record still carries `instrument_id`, so you can split by contract. Cheapest. Does not include the non-front contract's book around the roll. |

The seal §1 gives the front-month definition as **"greater cumulative volume on the prior
session"** — that is the `.v` (volume) roll, *not* `.c` (calendar). The "`MNQ.c.0`" string in
§1 is imprecise shorthand; Amendment 5 records the resolution. Use the raw contracts or
`MNQ.v.0`.

Contract codes: `MNQM6` = MNQ **Jun** 2026, `MNQU6` = **Sep** 2026, `MNQZ6` = **Dec** 2026
(GLBX uses a single-digit year, so `MNQU26` in `trades.db` = `MNQU6` here).

---

## Step 1 — account + billing

1. **databento.com → Sign up** (email + password), verify the email. A **$125 historical
   credit** is applied automatically (expires 6 months after signup).
2. **Portal → Settings → Billing → add a payment method.** Tranche 1 is ~$288–328 over the
   credit, so a card (or prepaid credits) must be on file. While you are there, set a
   **spending limit / billing alert** — a fat-finger on Tranche 2 is a ~$1,200 mistake.
3. **Portal → Settings → API Keys → Create key.** Copy it into a password manager or the
   repo's **gitignored** `.env` (never commit it). **Delete / rotate** the key
   `db-sny5wf…` that was pasted into the working session — treat it as burned.

---

## Step 1.5 — upgrade the VPS to KVM 4

Tranche 1 is ~80–110 GB on disk; the current KVM 2 has 64 GB free. Upgrade **KVM 2 → KVM 4**
(4 vCPU / 16 GB / 200 GB) in hPanel first — in-place, keeps your data. ~+$6/mo on the
current promo. This also fixes the box's RAM shortage (it idles at ~0.2 GB free). See
Amendment 3 / Amendment 6 §A6.3.

---

## Step 2 — confirm the exact cost (free, do this first, every time)

```bash
pip install databento     # once, in a venv
```

```python
import databento as db
c = db.Historical("YOUR_NEW_KEY")

print(c.metadata.get_cost(
    dataset="GLBX.MDP3",
    symbols=["MNQM6", "MNQU6", "MNQZ6"],
    stype_in="raw_symbol",
    schema="mbo",
    start="2026-05-01",
    end="2026-08-28",
))                                   # expect ~453.12

print(c.metadata.get_billable_size(  # expect ~2.70e11  (270 GB uncompressed)
    dataset="GLBX.MDP3", symbols=["MNQM6","MNQU6","MNQZ6"], stype_in="raw_symbol",
    schema="mbo", start="2026-05-01", end="2026-08-28"))
```

If the number does not match what you expect, **stop** and work out why before spending.

> Note: data availability ends **2026-08-28 22:30 UTC**. Do not set `end` past `2026-08-28`
> or the request 422s.

---

## Step 2.5 — one-day test download first (~$2, mandatory)

Before the $328 job, submit one day of one contract. It measures the real MNQ MBO zstd
ratio (so you know the true Tranche 1 disk size) and exercises the entire path — billing,
job, download, `DBNStore` read, integrity checks — on a trivial sample.

```python
job = c.batch.submit_job(
    dataset="GLBX.MDP3", symbols=["MNQM6"], stype_in="raw_symbol",
    schema="mbo", start="2026-06-16", end="2026-06-17",
    encoding="dbn", compression="zstd", split_duration="day",
)
print(job["id"])
# when ready:
c.batch.download(job_id=job["id"], output_dir="data/tick/_test/")
```

Then: `ls -la data/tick/_test/` → compare the `.dbn.zst` size to ~0.9 GB uncompressed
(one day, one contract) → that ratio × 270 GB = the real Tranche 1 disk footprint.
Read it back with `db.DBNStore.from_file(...)`, run the Step 5 checks. Record the ratio
and outcome in the seal as Amendment 7.

If the ratio is far worse than ~3× (i.e. Tranche 1 would be >150 GB on disk), reconsider:
KVM 8 instead of KVM 4, or `MNQ.v.0` (single stream, $412.65, ~10% less data).

---

## Step 3 — submit the batch job

### Option A — web portal (simplest)

Portal → **Download** (batch) → new request:

| Field | Value |
|---|---|
| Dataset | CME Globex MDP 3.0 — `GLBX.MDP3` |
| Symbology | Raw symbol → `MNQM6, MNQU6, MNQZ6` |
| Schema | **MBO** (Market by order) |
| Date range | 2026-05-01 → 2026-08-28 |
| Encoding | **DBN** |
| Compression | **Zstd** |
| Split | **By day** (resumable downloads; one file per contract per day) |

Review the cost (**$453.12**, minus $125 credit → **$328.12 charged**), confirm, submit.

### Option B — Python

```python
job = c.batch.submit_job(
    dataset="GLBX.MDP3",
    symbols=["MNQM6", "MNQU6", "MNQZ6"],
    stype_in="raw_symbol",
    schema="mbo",
    start="2026-05-01",
    end="2026-08-28",
    encoding="dbn",
    compression="zstd",
    split_duration="day",
)
print(job["id"])          # save this
```

**The job bills your account the moment it is submitted** (credit first, then card).
There is no undo. This is why Step 2 is mandatory.

---

## Step 4 — wait, then download

- The job runs server-side. ~270 GB uncompressed → **minutes to ~2 hours**. You get an
  email when it is ready; poll with `c.batch.list_jobs()` or the portal.
- Download into the repo's tick directory:

```python
c.batch.download(job_id="YOUR_JOB_ID", output_dir="data/tick/mnq_mbo_tranche1/")
```

or the CLI: `databento download YOUR_JOB_ID --output-dir data/tick/mnq_mbo_tranche1/`
or the HTTPS links from the portal.

- **On-disk size: ~80–110 GB** of `.dbn.zst` (the 270 GB is uncompressed; MBO zstd is only
  ~2.5–3.5× — Step 2.5 gives the exact figure). Fits KVM 4 (200 GB), **not** KVM 2. Do
  **not** decompress it all — the Databento reader streams the compressed files.

---

## Step 5 — integrity checks (seal §1 requires these before use)

```python
import databento as db
store = db.DBNStore.from_file("data/tick/mnq_mbo_tranche1/glbx-mdp3-YYYYMMDD.mbo.dbn.zst")
df = store.to_df()
# per the seal §1:
assert df.index.is_monotonic_increasing                     # timestamps non-decreasing
# reconstruct BBO and check bid <= ask on 100.000% of states
# check trade prices fall within [session low, session high]
```

Record the exact pass rate (the seal wants "report the exact pass rate", same discipline as
OFI-1's `upvol+downvol==volume` check). A `degraded` day appears in the availability list for
**2026-05-24** and **2026-07-30** — note those in the integrity report; they are still
`available`, just flagged by Databento.

---

## Step 6 — licensing

On the first CME request Databento shows a click-through license. For **internal research /
backtesting** (non-display, non-redistribution) the standard terms cover you. Do not
redistribute the raw data or any derived real-time feed.

---

## After Tranche 1

Build the §2 simulator → run the §3 parity gate against the ~129 live `trades.db` fills →
append the result to the seal as **Amendment 6**. Only on a PASS do you buy Tranche 2
(the bulk, RTH-only, ~$1,150–1,250 — see Amendment 4 §A4.4).
