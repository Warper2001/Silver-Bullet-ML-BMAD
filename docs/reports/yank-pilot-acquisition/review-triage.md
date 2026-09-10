# Review triage

Three fresh-context agents reviewed the change: blind, edge-case and verification-gap. A concurrency limit delayed the third launch until a slot opened; all findings were collected before triage. Corrections remain within the approved acquisition/evidence boundaries.

| ID | Verdict | Evidence and resolution |
|---|---|---|
| B1 | high | Nonfinite parsed numbers fail JSON publication after a reply arrives; preserve raw bytes with explicit representation diagnostics. Confirmed independently with a NaN response. |
| B2 | medium | Lone surrogate strings fail the credential scanner's UTF8 encoding; handle them explicitly without mislabeling a response as transport failure. |
| B3 | medium | File fsync without directory fsync does not support the durable-before-network claim; durably publish directory entries. |
| B4 | high | Cost evidence is account-specific while a mutable token provider can change credentials; require an approved credential digest and compare it before every attempt. |
| B5 | medium | A future review timestamp parses and passes; reject future reviews with a bounded clock-skew allowance. |
| B6 | medium | Retry/control decisions were not retained; persist safe derived control flags and retry timing, excluding unrestricted response headers. |
| B7 | medium | HTTPX inactivity timeout is not a total response deadline; add a cancellable total deadline. |
| B8 | medium | Unbounded Retry-After can wait effectively forever; stop when its required delay exceeds the remaining one-hour run budget, never retry early. |
| B9 | medium | Reviewed diff preceded benchmark completion and lacked the final report; retain the final actual log/metrics and findings before completion. |
| E1 | medium | Same total-deadline defect as B7; same correction, separate finding retained. |
| E2 | medium | Same durable-directory publication defect as B3; same correction. |
| E3 | high | Overflowing JSON numbers share B1's raw-publication defect; include exponent-overflow coverage. |
| V1 | medium | Existing tests do not advance a real token provider through expiry after successful acquisition/retry; add both paths with evidence retention and unchanged token-file assertions. |
| R1 | high | Root reproduced NaN response as transport_failure with only a partial .tmp result; include exact raw-byte retention assertion in B1 correction. |

All listed corrections are complete. Final verification:58 focused acquisition cases pass,633 broad regression cases pass with the unchanged known configuration failure, and the full startup benchmark passes. Actual benchmark artifacts and final outcomes are retained in README.md. No finding authorizes subscription changes, acquisition with unverified cost, installation or live collection.
