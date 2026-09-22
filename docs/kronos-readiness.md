# Kronos readiness

This milestone produces documentary HOLD_EVALUATION, with actual power UNASSESSABLE. No command permits historical scoring, strategy testing, or trading. Existing replay/preflight are unchanged. All generated reports, statuses, protocols and manifests carry both authorization flags as false.

Run from the readiness worktree. Every output directory must be fresh and outside data, logs, model, credential and environment paths. Sources must be documentary JSON/Markdown/text/HTML/PDF with no symlink or data-path aliases; no historical price input is accepted. The fixed preflight documentary reports are allowlisted by their original hashes; `--documentary-root` locates these in the main checkout.

```bash
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m research.kronos_readiness assess --documentary-root /root/Silver-Bullet-ML-BMAD --source-pack docs/reports/kronos-readiness/sources-20260922/source-pack.json --output-dir docs/reports/kronos-readiness/assessment-001
/root/Silver-Bullet-ML-BMAD/.venv/bin/python -m research.kronos_readiness probe --plan docs/reports/kronos-readiness/sources-20260922/probe-plan.json --token-path /root/Silver-Bullet-ML-BMAD/.access_token --output-dir docs/reports/kronos-readiness/probe-001
/root/Silver-Bullet-ML-BMAD/.venv-research/bin/python -m research.kronos_readiness timing --cache /root/Silver-Bullet-ML-BMAD/.venv-research/kronos --decisions 3 --output-dir docs/reports/kronos-readiness/timing-001
```

Long runs should be launched with `nohup ... > <dedicated-report-log> 2>&1 &` and observed through the log. Exit code 2 means HOLD, pending, blocked or refused; it does not mean permission to evaluate. Timing succeeds only on valid outputs. Missing cached pinned assets fail offline without downloading.

A source pack contains `sources`, an array of entries with `path`, `sha256`, `date`, `url`, and a narrowly scoped `claim`. Paths may be absolute or relative to the pack. Optional `category` matches one blocker key in the assessment. Hash verification establishes source identity, not historical admissibility or the truth of arbitrary claims. An empty pack is valid and leaves all gaps unresolved. Reports manifest every output; each observation is written exclusively once with read-only permissions and ordered sequence IDs. These permissions plus hashes detect ordinary changes; they are not a privileged-user tamper-proof store.

A probe plan includes the same `sources` array and the following documentary reviewer assessments:

```json
{
  "sources": [],
  "session": {
    "verified": true,
    "date": "2026-09-22",
    "open": "2026-09-22T09:30:00-04:00",
    "close": "2026-09-22T16:00:00-04:00",
    "source_sha256": "HASH_OF_DATED_CALENDAR_JUSTIFICATION"
  },
  "contract": {
    "verified": true,
    "symbol": "MNQZ26",
    "observed_at": "2026-09-22T10:00:00-04:00",
    "source_sha256": "HASH_OF_ARCHIVED_RECENT_SUCCESSFUL_REQUEST"
  }
}
```

A reviewer must verify the dated session against official calendar evidence, including early closes and DST, and the explicit contract against a successful GET from the current NY date. `verified` records this documentary reviewer assessment; the software does not infer calendar truth from an archive hash. Both referenced hashes must pass source verification. Only MNQZ26 is allowed for this preregistered capture. If a full 900-second window is unavailable, the command records PENDING without reading the token. An early close must be supplied explicitly; aware timestamps convert to UTC through the NY calendar checks.

The probe reads the existing plain token once, never imports live auth, and never refreshes or writes shared state. It first requests SIM metadata and checks symbol, root, asset, exchange, currency and explicit future expiration. It then requests only latest-three minute bars, at least five seconds between starts, at most 180 bar requests and 900 seconds including metadata. Each HTTP call has a hard Linux main-thread alarm deadline of at most 15 seconds and the remaining capture/session budget. Redirects, proxies and other endpoints/methods are refused; bodies are capped at 1 MiB. Auth errors, throttle, timeout, malformed responses or metadata failures stop without retries. Non-success responses are recorded with secrets redacted. There is no background or ongoing collector launched by these modules.

Receipt timestamps are captured immediately after transport returns, before response processing. Credential echoes are redacted in raw bytes and decoded JSON keys/strings. Nonstandard or nonfinite JSON numbers stop safely with sanitized raw evidence retained.

Ordered raw response bytes (base64 after credential redaction), provider fields, request/receipt UTC and monotonic times are retained. Repeated timestamps are never discarded from observations; the descriptive report identifies revisions, absent status/timezone and observed timestamp gaps. Receipt minus timestamp is a descriptive difference, not completion or first-arrival latency. Bar age never proves completion. Current observations cannot authenticate historical files.

Timing uses the bundled synthetic 128-bar context and pinned CPU assets only. Startup and each complete three-seed decision duration are separate. No latency value is adopted. Three seeds remain one ensemble, and repeated timing decisions supply no additional statistical evidence.

The candidate protocol preserves the engine's sign of mean terminal close versus observed close, four-bar momentum and flat control. Per-session one-contract outcomes are K net, M net and paired K-M. Both K and K-M means must exceed zero; Sharpe and drawdown are separate outcomes. Commission, exchange/regulatory, slippage and latency remain separate and unadopted; SIM instant fills cannot establish slippage.

Conditional planning uses positive integer session N, SE inflation >=1, one-sided alpha .025 per test and marginal target power .90, giving joint power at least .80 by union bound. Detectable standardized effect is `(z(.975)+z(.90))*inflation/sqrt(N)`. Illustrative N/inflation grids are sensitivity assumptions, not thresholds or admitted samples. Dollar scenarios require independent evidence references for both useful effects, arm variances, covariance and dependence; references do not themselves authenticate evidence. Paired variance is `Var(K)+Var(M)-2Cov(K,M)`. No arm/seed/overlapping forecast multiplies N. Actual power stays UNASSESSABLE.

Next evidence needed: acquisition records binding each historical file; historical timestamp/completion/arrival provenance; dated calendars; causal same-contract selection and roll boundaries; untouched eligible sessions and research-exposure audit; independently supported execution costs, economic effects and variance/dependence. If history remains inadmissible, commit a separate prospective collection preregistration, archive immutable ordered observations and selection/calendar evidence, segregate research exposure, freeze the eligible population and execution assumptions, and run a power gate before scoring. This document specifies that future collection; it does not start or authorize it.

The expanded [prospective collection specification](kronos-prospective-collection.md) records the parent-reviewed collection design and remaining evidence requirements.
