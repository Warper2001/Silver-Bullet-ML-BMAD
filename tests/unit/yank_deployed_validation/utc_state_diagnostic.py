"""Component diagnostic for explicit built-in UTC timestamps, not poll latency.

Loads the original state method from its recorded git revision and compares it
with the optimized method on identical synthetic state. No strategy poll, live
transport, account observation, or production claim is involved.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from test_evidence import prepared_runtime
from test_observation_benchmark import startup_bars

BASELINE = 'a0ff173522ee7a8029f53ada7be16ee0f1356e7f'
ADAPTER = 'src/research/yank_deployed_validation/adapter.py'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'fresh output required'
    source = subprocess.check_output(['git', 'show', BASELINE+':'+ADAPTER], cwd=ROOT, text=True)
    original = types.ModuleType('_yank_overhead_original_adapter')
    original.__file__ = str(ROOT/ADAPTER)+'@'+BASELINE
    sys.modules[original.__name__] = original
    exec(compile(source, original.__file__, 'exec'), original.__dict__)
    runtime, trader, view, *_ = prepared_runtime(args.output.parent/'unused-fixture')
    adapter = sys.modules[type(view).__module__]
    rows = startup_bars(7500, datetime(2025, 5, 19, 13, 59, tzinfo=timezone.utc))
    parsed_types = set()
    for raw in rows:
        bar = trader._parse_bar(raw)
        assert bar is not None
        parsed_types.add(type(bar.timestamp).__qualname__)
        bar.timestamp = datetime.fromisoformat(raw['TimeStamp'])
        assert type(bar.timestamp) is datetime and bar.timestamp.tzinfo is timezone.utc
        trader.dollar_bars.append(bar)
    trader._last_processed_timestamp = trader.dollar_bars[-1].timestamp
    state = original.Adapter.state(view)
    expected_hash = hashlib.sha256(original.canonical(state).encode()).hexdigest()
    adapter._utc_timestamp_parts.cache_clear()
    records = []
    def measure(mode):
        view._buffer_cache_key = None
        function = original.Adapter.state if mode == 'original' else type(view).state
        wall, cpu = time.perf_counter(), time.process_time()
        result = function(view)
        record = dict(mode=mode, wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)
        assert result == state, 'complete state mismatch'
        records.append(record)
    measure('optimized')
    cold = records.pop()
    for repeat in range(6):
        for mode in (('original','optimized') if repeat % 2 == 0 else ('optimized','original')):
            measure(mode)
    info = adapter._utc_timestamp_parts.cache_info()
    assert info.maxsize == 8192 and info.currsize == info.misses == 7500
    assert info.hits == 6*7500
    report = dict(scope='COMPONENT_DIAGNOSTIC_EXPLICIT_BUILTIN_UTC_NOT_POLL_LATENCY',
        baseline_commit=BASELINE, source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        original_adapter_sha256=hashlib.sha256(source.encode()).hexdigest(),
        optimized_adapter_sha256=hashlib.sha256((ROOT/ADAPTER).read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256(original.canonical(rows).encode()).hexdigest(),
        fixture_bar_count=7500, parsed_timestamp_types_before_explicit_conversion=sorted(parsed_types),
        compared_timestamp_type='datetime.datetime with tzinfo is datetime.timezone.utc',
        complete_state_equal=True, complete_state_sha256=expected_hash,
        cold_optimized=cold, counterbalanced_component_samples=records,
        median_seconds={mode:statistics.median(r['wall_seconds'] for r in records if r['mode']==mode)
                        for mode in ('original','optimized')},
        cache_info=info._asdict(), decision='HOLD_VALIDATION', limitations=[
            'Synthetic state projection only; no trading poll or full strategy warmup is measured.',
            'Explicitly converts parsed private Clock subclasses to built-in UTC datetime for this diagnostic.',
            'This cache path is not exercised by the unchanged private Clock latency fixtures.',
            'No extrapolation to production timestamp types, live latency, or complete evidence capacity.'])
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')


if __name__ == '__main__': main()
