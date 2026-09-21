"""Audit three saved trade exports only; no strategy imports or raw-data access."""
import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal, getcontext
from pathlib import Path

getcontext().prec = 34
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
D = Decimal
FILES = {
    'ml': 'data/reports/backtest_1year_20260615_181838.csv',
    'baseline': 'data/reports/backtest_1year_20260615_185354.csv',
    'repeat': 'data/reports/backtest_1year_20260615_214013.csv',
}

def stats(rows):
    values = [D(r['pnl']) for r in rows]
    gains = sum((p for p in values if p > 0), D(0))
    losses = -sum((p for p in values if p < 0), D(0))
    return {'n': len(rows), 'net_usd': str(sum(values, D(0))),
            'profit_factor': str(gains / losses) if losses else None}

def key(row):
    return row['entry_time'], row['direction']

runs = {}
report = {'status': 'HOLD_VALIDATION', 'scope': 'Saved reports only; no fresh historical replay',
          'assumptions': {'point_value_usd': '2', 'documented_contracts': '5', 'roundtrip_fee_usd': '4'},
          'runs': {}, 'source_sha256': {}}
for name, path in FILES.items():
    raw = (ROOT / path).read_bytes()
    report['source_sha256'][path] = hashlib.sha256(raw).hexdigest()
    rows = list(csv.DictReader(raw.decode().splitlines()))
    assert len({key(r) for r in rows}) == len(rows), 'Duplicate trade keys'
    runs[name] = rows
    discrepancies = []
    implied = Counter()
    for row in rows:
        assert row['direction'] in ('LONG', 'SHORT')
        move = (D(row['exit_price']) - D(row['entry_price'])) * (1 if row['direction'] == 'LONG' else -1) * 2
        expected = move * 5 - 4
        quantity = str((D(row['pnl']) + 4) / move) if move else 'indeterminate'
        implied[quantity] += 1
        if expected != D(row['pnl']):
            discrepancies.append({'entry_time': row['entry_time'], 'reported_pnl': row['pnl'],
                                  'documented_size_pnl': str(expected), 'implied_quantity': quantity})
    subset = [r for r in rows if r['entry_time'].startswith('2026-')]
    report['runs'][name] = {'full': stats(rows), '2026': stats(subset),
        'implied_quantities_not_verified_fills': dict(sorted(implied.items())),
        'fixed_five_contract_discrepancies': discrepancies}

ml = {key(r): r for r in runs['ml'] if r['entry_time'].startswith('2026-')}
base = {key(r): r for r in runs['baseline'] if r['entry_time'].startswith('2026-')}
shared = ml.keys() & base.keys()
report['comparison_2026'] = {'shared_entries': len(shared), 'ml_only_entries': len(ml.keys() - base.keys()),
    'baseline_only_entries': len(base.keys() - ml.keys()),
    'shared_entries_with_changed_records': sum(ml[k] != base[k] for k in shared),
    'simple_removed_trade_interpretation_supported': False}
months = sorted({r['entry_time'][:7] for r in ml.values()})
report['ml_monthly_2026'] = {m: stats([r for r in ml.values() if r['entry_time'][:7] == m]) for m in months}
report['repeat_rows_identical'] = runs['ml'] == runs['repeat']
# Hand-recorded checks against the saved report and direct set comparison.
checks = [stats(runs['ml'])['net_usd'] == '7804.0', stats(runs['baseline'])['net_usd'] == '1748.0',
          len(ml) == 54, len(base) == 70, len(shared) == 39, report['repeat_rows_identical']]
report['audit_computation_checks_pass'] = all(checks)
report['source_sha256']['docs/reports/strategy-evidence-audit-20260907/reconcile.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(OUT / 'ledger-audit.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
assert all(checks), 'Audit reproduction failed'
print(json.dumps({'checks_pass': all(checks), 'runs': {n: {'full': r['full'], '2026': r['2026'], 'implied_quantities': r['implied_quantities_not_verified_fills']} for n, r in report['runs'].items()}}, indent=2))
