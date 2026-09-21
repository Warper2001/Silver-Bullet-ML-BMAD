"""Reproduce this descriptive report: PYTHONPATH=. .venv/bin/python <script> --output <new-directory>."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from statistics import median

from src.data.tradestation_curve_batch import ROOTS, utc, verify_run


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(source):
    integrity = verify_run(source)
    manifest = json.loads((source / 'manifest.json').read_text())
    assert manifest['identity']['year'] == 2025
    assert manifest['report_generation'] == 'complete'
    contracts = list(csv.DictReader((source / 'contracts.csv').open()))
    bars = list(csv.DictReader((source / 'contract_bars.csv').open()))
    info = {c['Symbol']: c for c in contracts}
    assert len(info) == len(contracts)
    expiry = {s: utc(c['ExpirationDate']).date() for s, c in info.items()}
    groups = defaultdict(dict)
    by_contract = defaultdict(list)
    for b in bars:
        root, t, symbol = b['canonical_root'], b['TimeStamp'], b['contract_code']
        assert root == info[symbol]['canonical_root'] and utc(t).year == 2025
        assert symbol not in groups[root, t]
        groups[root, t][symbol] = b
        by_contract[symbol].append(b)
    clocks = {root: sorted(t for r, t in groups if r == root) for root in ROOTS}
    indices = {root: {t: i for i, t in enumerate(ts)} for root, ts in clocks.items()}
    repeats, contract_stats = set(), []
    for symbol, observations in sorted(by_contract.items()):
        observations.sort(key=lambda b: b['TimeStamp'])
        root = info[symbol]['canonical_root']
        index = indices[root]
        run, longest, repeat_count = 1, 1, 0
        previous = None
        for b in observations:
            consecutive = previous is not None and index[b['TimeStamp']] == index[previous['TimeStamp']] + 1
            if consecutive and Decimal(b['Close']) == Decimal(previous['Close']):
                run += 1
                repeat_count += 1
            else:
                run = 1
            if run >= 3:
                repeats.add((symbol, b['TimeStamp']))
            longest = max(longest, run)
            previous = b
        span = index[observations[-1]['TimeStamp']] - index[observations[0]['TimeStamp']] + 1
        contract_stats.append(dict(root=root, symbol=symbol, observations=len(observations),
            first=observations[0]['TimeStamp'], last=observations[-1]['TimeStamp'],
            internal_missing=span-len(observations), zero_volume=sum(Decimal(b['TotalVolume']) == 0 for b in observations),
            zero_oi=sum(Decimal(b['OpenInterest']) == 0 for b in observations),
            repeated_close_steps=repeat_count, longest_equal_close_run=longest,
            repeat_flag_rows=sum((symbol,b['TimeStamp']) in repeats for b in observations),
            after_broker_expiry=sum(utc(b['TimeStamp']).date() > expiry[symbol] for b in observations)))
    daily, transitions = [], []
    for root in ROOTS:
        previous_pair = None
        for t in clocks[root]:
            observations = groups[root,t]
            # Date-only exclusion is a diagnostic, not an exchange delivery-safety rule.
            ordered = sorted((s for s in observations if utc(t).date() <= expiry[s]), key=lambda s:(expiry[s],s))
            pairs = [(a,b) for a,b in combinations(ordered,2) if 90 <= (expiry[b]-expiry[a]).days <= 180]
            def active(symbol, field):
                return Decimal(observations[symbol][field]) > 0
            def passes(pair, fields):
                return all(active(s,f) for s in pair for f in fields)
            both = [p for p in pairs if passes(p, ('TotalVolume','OpenInterest'))]
            clean = [p for p in both if all((s,t) not in repeats for s in p)]
            near = ordered[0] if ordered else None
            fixed = next((p for p in pairs if p[0] == near), None)
            fixed_ok = bool(fixed and passes(fixed, ('TotalVolume','OpenInterest')))
            fixed_clean = bool(fixed_ok and all((s,t) not in repeats for s in fixed))
            if previous_pair is not None and fixed != previous_pair:
                transitions.append(dict(root=root, timestamp=t,
                    previous_near=previous_pair[0], previous_far=previous_pair[1],
                    next_near=fixed[0] if fixed else '', next_far=fixed[1] if fixed else ''))
            previous_pair = fixed
            daily.append(dict(root=root, timestamp=t, observed_contracts=len(observations),
                pair_count=len(pairs), volume_pair_count=sum(passes(p,('TotalVolume',)) for p in pairs),
                oi_pair_count=sum(passes(p,('OpenInterest',)) for p in pairs),
                active_pair_count=len(both), repeat_filtered_pair_count=len(clean),
                fixed_near=fixed[0] if fixed else '', fixed_far=fixed[1] if fixed else '',
                fixed_active=int(fixed_ok), fixed_repeat_filtered=int(fixed_clean)))
    root_stats = []
    for root in ROOTS:
        d = [r for r in daily if r['root'] == root]
        c = [r for r in contract_stats if r['root'] == root]
        root_stats.append(dict(root=root, timestamps=len(d), contracts=len(c), bars=sum(r['observations'] for r in c),
            pair_days=sum(r['pair_count']>0 for r in d),
            volume_days=sum(r['volume_pair_count']>0 for r in d), oi_days=sum(r['oi_pair_count']>0 for r in d),
            active_days=sum(r['active_pair_count']>0 for r in d), clean_days=sum(r['repeat_filtered_pair_count']>0 for r in d),
            fixed_days=sum(bool(r['fixed_near']) for r in d), fixed_active_days=sum(r['fixed_active'] for r in d),
            fixed_clean_days=sum(r['fixed_repeat_filtered'] for r in d),
            min_pairs=min(r['pair_count'] for r in d), median_pairs=median(r['pair_count'] for r in d), max_pairs=max(r['pair_count'] for r in d),
            internal_missing=sum(r['internal_missing'] for r in c), zero_volume=sum(r['zero_volume'] for r in c),
            zero_oi=sum(r['zero_oi'] for r in c), repeated_close_steps=sum(r['repeated_close_steps'] for r in c),
            longest_equal_close_run=max(r['longest_equal_close_run'] for r in c),
            repeat_flag_rows=sum(r['repeat_flag_rows'] for r in c), after_broker_expiry=sum(r['after_broker_expiry'] for r in c),
            fixed_pair_changes=sum(r['root']==root for r in transitions)))
    monthly = []
    for month in sorted({r['timestamp'][:7] for r in daily}):
        latest = {root:max((r for r in daily if r['root']==root and r['timestamp'].startswith(month)),key=lambda r:r['timestamp']) for root in ROOTS}
        monthly.append(dict(month=month, roots_with_pairs=sum(r['pair_count']>0 for r in latest.values()),
            roots_with_active_pair=sum(r['active_pair_count']>0 for r in latest.values()),
            roots_with_clean_pair=sum(r['repeat_filtered_pair_count']>0 for r in latest.values()),
            roots_with_fixed_active_pair=sum(r['fixed_active'] for r in latest.values()),
            fixed_inactive_roots=','.join(root for root,r in latest.items() if not r['fixed_active']),
            observation_dates=','.join(sorted({r['timestamp'][:10] for r in latest.values()}))))
    totals = {k:sum(r[k] for r in root_stats) for k in ('bars','contracts','timestamps','pair_days','active_days','clean_days','fixed_active_days','fixed_clean_days','internal_missing','zero_volume','zero_oi','repeat_flag_rows','after_broker_expiry')}
    assert totals['bars']==len(bars) and totals['contracts']==len(contracts)
    assert all(r['repeat_filtered_pair_count'] <= r['active_pair_count'] <= r['pair_count'] for r in daily)
    assert all(r['active_pair_count'] <= r['volume_pair_count'] and r['active_pair_count'] <= r['oi_pair_count'] for r in daily)
    return dict(scope='2025 descriptive broker-bar feasibility; no signals, returns or P&L', verdict='HOLD-DATA',
        source=str(source), source_hashes={name:checksum(source/name) for name in ('manifest.json','contracts.csv','contract_bars.csv','acquisition-report.json')},
        analyzer_sha256=checksum(Path(__file__)), integrity=integrity, totals=totals,
        roots=root_stats, months=monthly), daily, contract_stats, transitions


def report(result):
    total = result['totals']
    lines = ['# 2025 commodity curve coverage and reported activity', '',
        'Scope: descriptive development evidence. Verdict: **HOLD-DATA**. No strategy signals, rankings, returns or P&L were calculated.', '',
        f"Verified source: {total['contracts']} contracts, {total['bars']:,} bars, {total['timestamps']:,} root/timestamp observations. All 3,059 source requests and six reports passed integrity verification.", '',
        '## Definitions and limits', '',
        '- A candidate pair has the same root and exact broker timestamp, with broker expirations 90–180 calendar days apart. Rows after the broker expiry date are excluded from pair counting. This does not establish notice/last-trade safety.',
        '- Active means both contracts report volume > 0 AND open interest > 0. It is a minimal activity screen, not evidence of fillability, spreads, executable liquidity, or timely open-interest publication.',
        '- Repeat-filtered additionally excludes a leg at its third or later identical close on consecutive root-observed timestamps. Missing observations reset the run. This deliberately conservative sensitivity check does not prove staleness; legitimate unchanged prices can fail it.',
        '- Fixed pair uses the earliest-expiring observed unexpired contract and its earliest 90–180-day deferred contract (symbol breaks expiry ties), selected BEFORE the activity screens. No alternative pair is substituted after a failure. This is an availability-based diagnostic, not the protocol’s calendar-safe nearby selection.',
        '- Internal missing counts absent contract observations between its first and last bar against the root’s union of observed timestamps. Outside-span absence is excluded because listing dates are unknown. These are not missing exchange sessions; absence shared by every contract is invisible to this measure.',
        '- Monthly snapshots use each root’s last observed timestamp in the month. Different sector clocks and missing publication timestamps prevent interpreting them as a synchronized, causal rebalance.',
        '- All filters are descriptive sensitivity checks, not optimized thresholds or modifications to the frozen universe. Source artifacts and the research protocol remain unchanged.', '',
        '## How much pair coverage survives?', '',
        'Counts are root/timestamp observations, not independent samples.', '',
        '| Root | Observed timestamps | Any pair | Both volume > 0 | Both OI > 0 | Both screens | + repeat filter | Fixed pair: both screens | Fixed pair: + repeat filter |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['roots']:
        lines.append('| '+ ' | '.join(str(r[k]) for k in ('root','timestamps','pair_days','volume_days','oi_days','active_days','clean_days','fixed_active_days','fixed_clean_days'))+' |')
    lines += ['', f"Any-pair activity coverage: {total['active_days']:,}/{total['timestamps']:,}; with the repeated-close sensitivity filter: {total['clean_days']:,}/{total['timestamps']:,}. Fixed-pair activity coverage: {total['fixed_active_days']:,}/{total['timestamps']:,}; with repeat filtering: {total['fixed_clean_days']:,}/{total['timestamps']:,}.", '',
        '## Pair depth and data-quality concentrations', '',
        '| Root | Pairs per timestamp min / median / max | Zero-volume rows | Zero-OI rows | Internal missing | Repeat-flag rows | Longest equal-close run | Fixed-pair changes |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in result['roots']:
        lines.append(f"| {r['root']} | {r['min_pairs']} / {r['median_pairs']} / {r['max_pairs']} | {r['zero_volume']} | {r['zero_oi']} | {r['internal_missing']} | {r['repeat_flag_rows']} | {r['longest_equal_close_run']} | {r['fixed_pair_changes']} |")
    lines += ['', f"Total zero-volume rows: {total['zero_volume']:,}; zero-OI rows: {total['zero_oi']:,}; internal missing observations: {total['internal_missing']:,}; rows after broker-expiry date: {total['after_broker_expiry']:,}. These categories can overlap. Fixed-pair changes compare consecutive observations; they are not trade or roll counts.", '',
        '## Last-observation monthly snapshots', '',
        '| Month | Roots: any pair | Roots: active pair | Roots: repeat-filtered pair | Roots: fixed active pair | Fixed inactive roots | Observed UTC dates |',
        '|---|---:|---:|---:|---:|---|---|']
    for r in result['months']:
        lines.append('| '+' | '.join(str(r[k]) or 'none' for k in ('month','roots_with_pairs','roots_with_active_pair','roots_with_clean_pair','roots_with_fixed_active_pair','fixed_inactive_roots','observation_dates'))+' |')
    lines += ['', '## Files and reproducibility', '',
        '- `summary.json`: metrics, source hashes, analyzer hash and integrity results.',
        '- `daily.csv`: every root/timestamp, pair counts and fixed-pair identities/activity outcomes.',
        '- `contracts.csv`: contract-level missingness, reported activity and repeated-close diagnostics.',
        '- `transitions.csv`: changes in the fixed diagnostic pair. No executions are implied.',
        '- `analyze.py`: run from the repository root with `PYTHONPATH=. .venv/bin/python docs/reports/commodity-curve-feasibility-2025/analyze.py --output /tmp/curve-feasibility-new`. Use an output directory without existing report files; writes fail rather than overwrite.', '',
        'Settlement provenance, publication/revision history, exchange calendars, delivery safety, account costs and executable quotes remain unresolved. Broad activity coverage does not change HOLD-DATA.']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    source = Path('data/commodity_curve/coverage-pilot-2025-20260906-v3')
    result, daily, contracts, transitions = analyze(source)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ('summary.json','daily.csv','contracts.csv','transitions.csv','report.md'):
        if (args.output/name).exists():
            raise FileExistsError(args.output/name)
    write_csv(args.output/'daily.csv', daily)
    write_csv(args.output/'contracts.csv', contracts)
    write_csv(args.output/'transitions.csv', transitions)
    with (args.output/'summary.json').open('x') as f:
        json.dump(result,f,indent=2,sort_keys=True)
        f.write('\n')
    with (args.output/'report.md').open('x') as f:
        f.write(report(result))
    print(json.dumps(result['totals'],indent=2))
    print(args.output/'report.md')
