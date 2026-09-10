"""Read-only operational indicator reconciliation; never imports the live bot.

This evidence script is deliberately separate from comparison implementation.
Duplicate records use first observation here, and are disclosed. This does not
qualify any session for performance evaluation or prove broker-event agreement.
"""
import csv
import hashlib
import io
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np


def audit(root):
    paths = [root / 'data/mim_nb/bars_raw.csv', root / 'data/mim_nb/decisions.csv']
    raw, logged = [p.read_bytes() for p in paths]
    sessions = defaultdict(dict)
    duplicates = defaultdict(int)
    for row in csv.DictReader(io.StringIO(raw.decode())):
        ts = datetime.fromisoformat(row['ts_utc'].replace('Z', '+00:00')).astimezone(ZoneInfo('America/New_York'))
        hm = ts.strftime('%H:%M')
        if not '09:31' <= hm <= '16:00':
            continue
        day = str(ts.date())
        if hm in sessions[day]:
            duplicates[day] += 1
            continue
        sessions[day][hm] = row
    decisions = list(csv.DictReader(io.StringIO(logged.decode())))
    target = max(r['ts_et'][:10] for r in decisions)
    prior = sorted(d for d, bars in sessions.items() if d < target and len(bars) == 390)[-14:]
    result = {'target': target, 'prior_days': prior, 'duplicate_counts': dict(duplicates),
              'inputs': {str(p.relative_to(root)): hashlib.sha256(b).hexdigest() for p, b in zip(paths, [raw, logged])},
              'scope': 'first-observed indicator reconciliation only; no contract identity or broker-action parity claimed', 'marks': []}
    if len(prior) != 14 or len(sessions[target]) != 390:
        result['status'] = 'unavailable: insufficient complete recorded sessions'
        return result
    today = sessions[target]
    opening = float(today['09:31']['open'])
    previous = float(sessions[prior[-1]]['16:00']['close'])
    for row in decisions:
        if row['ts_et'][:10] != target or not row['sigma']:
            continue
        hm = row['mark']
        sig = float(np.mean([abs(float(sessions[d][hm]['close']) / float(sessions[d]['09:31']['open']) - 1) for d in prior]))
        ub = opening * (1 + sig) + max(previous - opening, 0)
        lb = opening * (1 - sig) - max(opening - previous, 0)
        through = [today[k] for k in sorted(today) if k <= hm]
        vol = sum(float(r['volume']) for r in through)
        vwap = sum(float(r['close']) * float(r['volume']) for r in through) / vol if vol else float(today[hm]['close'])
        values = {'sigma': sig, 'ub': ub, 'lb': lb, 'vwap': vwap}
        agrees = all(f'{v:.6f}' == row[k] if k == 'sigma' else f'{v:.2f}' == row[k] for k, v in values.items())
        result['marks'].append({'mark': hm, **values, 'serialization_agrees': agrees,
                                'old_sigma_tolerance_would_fail': abs(sig-float(row['sigma'])) >= 1e-9})
    result['status'] = 'agrees at serialized precision' if result['marks'] and all(r['serialization_agrees'] for r in result['marks']) else 'disagreement requires provenance diagnosis'
    return result


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[3]
    print(json.dumps(audit(root), indent=2))
