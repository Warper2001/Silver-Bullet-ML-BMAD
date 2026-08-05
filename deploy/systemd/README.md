# systemd units

Vendored copies of the units installed in `/etc/systemd/system/` on the trading host.
These are the source of truth for **deployment configuration** — the flags in
`Environment=` lines are load-bearing, and several encode sealed risk decisions that are
not recoverable from the Python source alone.

## Install / update

```bash
sudo cp deploy/systemd/<unit> /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now <unit>
```

To capture drift the other way (host → repo), diff before copying:

```bash
for u in deploy/systemd/*.service deploy/systemd/*.timer; do
  diff -q "$u" "/etc/systemd/system/$(basename "$u")" || echo "DRIFT: $u"
done
```

## Flags that carry a decision

| Unit | Setting | Why it matters |
|---|---|---|
| `combine-floor-monitor.service` | `FLOOR_MONITOR_REPORT_ONLY=1` | Tracks HWM/floor and publishes `floor_state.json`, but **never halts or flattens**. Per prereg `mim-nb-risk-mechanics-removal` Amendment 2 (2026-08-04). Setting `0` re-arms a kill path the owner explicitly removed — do not do it casually. The code defaults to report-only, so this line is belt-and-braces, not the only guard. |
| `trader-mim-nb.service` | `MIM_NB_AUTOROLL=1` (default) | Front month follows the broker's `activeContract` flag. `0` pins `MIM_NB_SYMBOL` for a manual roll. |
| `trader-mim-nb.service` / `trader-yank.service` | `SIM_INVVOL=1`, `*_MIRROR_TS_SIM=1` | Mirror live combine orders to the TradeStation SIM account for the scaling rehearsal. Isolation-by-construction; never affects the primary order. |
| `trader-yank.service` | `YANK_CONTRACTS=2` | YANK trades 2ct against MIM-NB's 1ct on the **same** combine account (23884932). |
| `mim-parity-check.timer` | `OnCalendar=` | One-shot parity gate check. `Persistent=true` so a reboot does not skip a fire. |

## Which file each bot actually runs

Check `ExecStart`, not the docs. In particular:

* **YANK runs `src/research/yank_streaming_working.py`** — *not*
  `src/research/tier2_streaming_working.py`. The two are separate files that have
  diverged. `CLAUDE.md` describes `tier2_streaming_working.py` as "the deployed system",
  which is true of the S25 lineage but **not** of the YANK service. Editing the wrong one
  produces a change that tests green, merges, deploys, restarts cleanly — and does
  nothing. This cost a no-op deployment on 2026-07-29 (PR #21).

Verify what a running service actually opened:

```bash
tr '\0' '\n' < /proc/$(systemctl show trader-yank -p MainPID --value)/cmdline
```
