"""Offline subprocess benchmark. Run with --output DIR [--diagnostic].

Latency cells have no profiler, tracing, or filesystem instrumentation. Each cell
loads fresh private fixtures; warmup and signed publication are timed separately.
"""
import argparse
import asyncio
import cProfile
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from test_evidence import prepared_runtime, startup, e
from test_observation_benchmark import startup_bars
from test_startup_benchmark import with_shadows

LIMITS = dict(capacity=1, max_bytes=64_000_000, max_nodes=4_000_000)
MODES = ('baseline', 'disabled', 'guarded')


def schedule():
    return [(workload, repeat, mode) for workload in ('startup', 'steady')
            for repeat in range(3) for mode in MODES[repeat:] + MODES[:repeat]]


def pins():
    paths = list((ROOT / 'src/research/yank_deployed_validation').glob('*.py'))
    paths += [Path(__file__)] + [Path(__file__).with_name(n) for n in
        ('test_evidence.py', 'test_observation_benchmark.py', 'test_startup_benchmark.py')]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def measured(call):
    wall, cpu = time.perf_counter(), time.process_time()
    result = call()
    return result, dict(wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)


def cell(output, workload, mode, diagnostic=False):
    output.mkdir(parents=True, exist_ok=False)
    code = pins()
    end = datetime(2025, 5, 19, 13, 59, tzinfo=timezone.utc)
    bars = startup_bars(2880 if workload == 'startup' else 7500, end)
    setup = with_shadows(prepared_runtime(output / 'fixture', bars=bars), list(bars[-15:]))
    runtime, trader, view, expected, key = setup
    warmup = None
    if workload == 'steady':
        _, warmup = measured(lambda: asyncio.run(trader._poll_and_process()))
    expected = dict(expected, runtime=startup.runtime_identity(trader), checkpoint_sha256=e.sha(view.checkpoint()))
    session = None
    def prepare():
        nonlocal session
        if mode != 'baseline':
            session = startup.prepare_observation(trader, enabled=mode == 'guarded', expected_release=expected,
                output_dir=output/'capture', signer=key, key_id='benchmark', pseudonym_salt=b'test', capture_limits=LIMITS)
            if mode == 'guarded':
                assert isinstance(session, startup.ObservationSession), session
    _, preparation = measured(prepare)
    feeds = [bars.copy()] if workload == 'startup' else [startup_bars(15, end + timedelta(minutes=i+1)) for i in range(30)]
    inputs = [hashlib.sha256(e.canonical(feed).encode()).hexdigest() for feed in feeds]
    async def polls():
        for i, feed in enumerate(feeds):
            if workload == 'steady':
                now = end + timedelta(minutes=i+1)
                runtime.clock = now + timedelta(minutes=1)
                bars[:] = feed
            await trader._poll_and_process()
    # Feed generation and fixture hashing finish before timing starts.
    profile = cProfile.Profile() if diagnostic else None
    assert sys.getprofile() is None and sys.gettrace() is None
    if profile: profile.enable()
    try:
        _, polling = measured(lambda: asyncio.run(polls()))
    finally:
        if profile:
            profile.disable(); profile.dump_stats(str(output/'diagnostic.pstats'))
    coverage, close = measured(session.close) if mode == 'guarded' else (None, None)
    assert not runtime.logger.errors, runtime.logger.errors
    assert code == pins()
    capture_path = output/'capture/capture.jsonl'
    report = dict(workload=workload, mode=mode, diagnostic=diagnostic, pid=os.getpid(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        code_sha256=code, fixture_sha256=inputs, limits=LIMITS, warmup=warmup,
        prepare=preparation, poll=polling, close=close, coverage=coverage,
        capture_bytes=capture_path.stat().st_size if capture_path.exists() else 0,
        process_high_water_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        final_state_sha256=e.sha(view.state()), final_bar_count=len(trader.dollar_bars),
        parity_rows=len(trader._shadow_logger.rows),
        environment=dict(python=sys.version, platform=platform.platform(), cpu_count=os.cpu_count()),
        release_expectation='SELF_DERIVED_PRIVATE_SYNTHETIC_NOT_INDEPENDENT_RELEASE_EVIDENCE',
        shadow_feed='FIXED_SYNTHETIC_STARTUP_TAIL_NOT_LIVE_PARITY_EVIDENCE',
        decision='HOLD_VALIDATION')
    (output/'result.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cell', choices=MODES)
    parser.add_argument('--workload', choices=('startup','steady'))
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    if args.cell:
        cell(args.output, args.workload, args.cell, args.diagnostic)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    jobs = [('startup', 0, 'guarded'), ('steady', 0, 'guarded')] if args.diagnostic else schedule()
    for workload, repeat, mode in jobs:
        target = args.output/f'{workload}-{repeat}-{mode}'
        command = [sys.executable, str(Path(__file__).resolve()), '--output', str(target), '--cell', mode, '--workload', workload]
        if args.diagnostic: command.append('--diagnostic')
        print(' '.join(command), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)
        results.append(json.loads((target/'result.json').read_text()))
    (args.output/'results.json').write_text(json.dumps(results, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    main()
