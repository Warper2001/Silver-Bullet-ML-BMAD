"""Separate diagnostics preserving profile entries from private module reloads.

cProfile's default snapshot overwrites distinct code objects sharing a source
label. Aggregate those entries, since the CLI intentionally reloads collectors.
This runner is never used for latency measurements.
"""
import argparse
import cProfile
import json
import hashlib
import marshal
from pathlib import Path
import subprocess
import sys

import observer_overhead as harness


class AggregateProfile(cProfile.Profile):
    def dump_stats(self, filename):
        self.disable()
        super().snapshot_stats()
        with open(str(filename)+'.standard-lossy', 'wb') as stream:
            marshal.dump(self.stats, stream)
        self.snapshot_stats()
        with open(filename, 'wb') as stream:
            marshal.dump(self.stats, stream)

    def snapshot_stats(self):
        entries = self.getstats()
        values = {}
        for entry in entries:
            label = cProfile.label(entry.code)
            row = values.setdefault(label, [0, 0, 0., 0., {}])
            row[0] += entry.callcount-entry.reccallcount
            row[1] += entry.callcount
            row[2] += entry.inlinetime
            row[3] += entry.totaltime
        # Attribution needs top-level totals only. Caller edges can refer to C
        # calls outside getstats' boundary and are deliberately not reconstructed.
        self.stats = {key:tuple(value) for key,value in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workload', choices=('startup','steady'))
    args = parser.parse_args()
    if args.workload:
        cProfile.Profile = AggregateProfile
        driver_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result = harness.cell(args.output, args.workload, 'guarded', diagnostic=True)
        assert driver_hash == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result['profile_driver_sha256'] = driver_hash
        result['profile_aggregation'] = 'SUM_DISTINCT_CODE_OBJECTS_SHARING_SOURCE_LABEL_AFTER_DISABLE'
        (args.output/'result.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
        return
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for workload in ('startup','steady'):
        target = args.output/(workload+'-0-guarded')
        command = [sys.executable, str(Path(__file__).resolve()), '--output', str(target), '--workload', workload]
        print(' '.join(command), flush=True)
        subprocess.run(command, check=True, cwd=harness.ROOT)
        rows.append(json.loads((target/'result.json').read_text()))
    (args.output/'results.json').write_text(json.dumps(rows, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__': main()
