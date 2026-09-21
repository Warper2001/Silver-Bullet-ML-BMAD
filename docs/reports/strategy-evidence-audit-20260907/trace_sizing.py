"""Reconstruct historical sizing from saved prices, without importing trading code.

Entry-day P&L attribution reproduces the historical close-before-day-reset order.
This is a forensic reconstruction, not endorsement of that session convention.
"""
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, getcontext
from pathlib import Path
from zoneinfo import ZoneInfo

getcontext().prec = 34
D = Decimal
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SEAL = '138cab1b31d064555ede4c9c07503399a743893f'
ET = ZoneInfo('America/New_York')

def session(timestamp):
    value = datetime.fromisoformat(timestamp)
    assert value.tzinfo is not None
    return value.astimezone(ET).date()

def reconstruct(rows):
    daily = defaultdict(lambda: D(0))
    reduced = {}
    records = []
    previous_exit = None
    for r in rows:
        entry = datetime.fromisoformat(r['entry_time'])
        end = datetime.fromisoformat(r['exit_time'])
        assert entry <= end
        assert previous_exit is None or previous_exit <= entry, 'Overlapping positions'
        previous_exit = end
        day = session(r['entry_time'])
        profitable = [p for p in daily.values() if p > 0]
        ratio = max(profitable) / sum(profitable) if profitable else D(0)
        reduced[day] = reduced.get(day, False) or ratio >= D('0.45')
        quantity = 1 if reduced[day] else 5
        assert r['direction'] in ('LONG', 'SHORT')
        sign = 1 if r['direction'] == 'LONG' else -1
        pnl = (D(r['exit_price']) - D(r['entry_price'])) * sign * 2 * quantity - 4
        # Use reconstructed P&L, never reported P&L, to determine subsequent sizing.
        daily[day] += pnl
        records.append({'entry_time': r['entry_time'], 'exit_time': r['exit_time'],
            'risk_session': str(day), 'exit_session': str(session(r['exit_time'])),
            'prior_profit_concentration': str(ratio), 'predicted_contracts': quantity,
            'reconstructed_pnl': str(pnl), 'reported_pnl': r['pnl'],
            'residual': str(D(r['pnl']) - pnl)})
    return records

result = {'status': 'PASS_SAVED_LEDGER_RECONSTRUCTION', 'research_status': 'HOLD_VALIDATION',
          'historical_commit': SEAL, 'source_sha256': {}, 'runs': {},
          'limitations': ['No signal, bar-path or fill replay',
                         'No proof of original working-tree or loaded-model identity',
                         'Historical risk-session attribution reproduced, not endorsed']}
for name, stamp in [('ml', '181838'), ('baseline', '185354'), ('repeat', '214013')]:
    path = f'data/reports/backtest_1year_20260615_{stamp}.csv'
    raw = (ROOT / path).read_bytes()
    result['source_sha256'][path] = hashlib.sha256(raw).hexdigest()
    records = reconstruct(list(csv.DictReader(raw.decode().splitlines())))
    assert all(D(r['residual']) == 0 for r in records), name
    result['runs'][name] = {'n': len(records),
        'quantity_counts': dict(sorted(Counter(str(r['predicted_contracts']) for r in records).items())),
        'net': str(sum((D(r['reconstructed_pnl']) for r in records), D(0))),
        'max_absolute_residual': str(max(abs(D(r['residual'])) for r in records)),
        'cross_session_trades': sum(r['risk_session'] != r['exit_session'] for r in records),
        'records': records}
# Hash historical source blobs without importing them or loading the model pickle.
for path in ['backtest_tier2_1year_validation.py', 'src/research/tier2_streaming_working.py',
             'src/research/strategy_core.py', 'models/xgboost/tier2_meta_labeling_model.pkl']:
    raw = subprocess.run(['git', 'show', f'{SEAL}:{path}'], cwd=ROOT, check=True, capture_output=True).stdout
    result['source_sha256'][f'{SEAL}:{path}'] = hashlib.sha256(raw).hexdigest()
result['source_sha256']['trace_sizing.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
assert D(result['runs']['ml']['net']) == D('7804.0')
assert D(result['runs']['baseline']['net']) == D('1748.0')
assert result['runs']['ml']['records'] == result['runs']['repeat']['records']
(OUT / 'sizing-trace.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({k: {f: v for f, v in run.items() if f != 'records'} for k, run in result['runs'].items()}, indent=2))
