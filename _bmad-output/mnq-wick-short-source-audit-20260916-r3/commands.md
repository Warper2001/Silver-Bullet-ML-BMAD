# Commands

```bash
/root/Silver-Bullet-ML-BMAD/.venv/bin/python tools/audit_mnq_wick_short_source.py --output _bmad-output/mnq-wick-short-source-audit-YYYYMMDD-new-run
```

The audit is read-only: it hashes the registered source before decoding it and does not call the calibration runner.
