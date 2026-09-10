#!/usr/bin/env bash
# Finite operational reader. Embedded Python uses only the standard library.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
exec "$repo_dir/.venv/bin/python" - "$script_dir/poll.sh" "$@" <<'PY'
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

SCRIPT = Path(sys.argv.pop(1)).resolve()
ROOT = SCRIPT.parents[2]
RUNS = ROOT / 'research/mim_comparison/runs'
ADAPTER = RUNS / '20260910-contract-feed'
HISTORY = RUNS / '20260910T210224-historical-f3950efb68'
COLLECTOR = RUNS / '20260910T214803-shadow-47055eba62/collector'
LIVE_STATE = RUNS / '20260910-contract-feed-poll'
PREPARE_STATE = RUNS / '20260910-contract-feed-poll-prepare'
WARMUP = ROOT / 'data/mim_x/mnq_1min_by_contract.csv'


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path, content):
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare(feed, state):
    # Hold the same lock as the adapter while verifying/copying its committed prefix.
    with open(feed.parent / 'lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        db = sqlite3.connect((feed.parent / 'journal.sqlite').as_uri() + '?mode=ro', uri=True)
        try:
            record = db.execute("SELECT value FROM meta WHERE key='feed'").fetchone()
            if not record:
                raise ValueError('adapter has no committed feed metadata')
            committed = json.loads(record[0])
        finally:
            db.close()
        with open(feed, 'rb') as source:
            size = committed['size']
            initial = os.fstat(source.fileno())
            if initial.st_size < size:
                raise ValueError('feed shorter than committed prefix')
            digest = hashlib.sha256()
            remaining = size
            while remaining:
                block = source.read(min(1048576, remaining))
                if not block:
                    raise ValueError('feed truncated while verifying')
                digest.update(block)
                remaining -= len(block)
            if digest.hexdigest() != committed['hash']:
                raise ValueError('feed committed prefix hash mismatch')
            # An interrupted adapter append can leave a partial uncommitted line.
            # Leave it pending. Complete uncommitted lines require adapter recovery.
            suffix = source.read(1048577)
            if b'\n' in suffix or len(suffix) > 1048576:
                raise ValueError('uncommitted feed records require adapter recovery')
            source.seek(0)
            header = source.readline(min(size, 1048576))
            if not header.endswith(b'\n'):
                raise ValueError('feed header is not a complete line')
            fields = next(csv.reader([header.decode('utf-8')], strict=True))
            required = {'contract', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'received_at'}
            if len(fields) != len(set(fields)) or not required <= set(fields):
                raise ValueError('unexpected identified feed header')
            header_end = len(header)
            position = size
            chunks = []
            newlines = 0
            while position > header_end and newlines < 501:
                length = min(65536, position - header_end)
                position -= length
                source.seek(position)
                block = source.read(length)
                if len(block) != length:
                    raise ValueError('feed truncated during tail scan')
                chunks.append(block)
                newlines += block.count(b'\n')
            tail = b''.join(reversed(chunks))
            if position > header_end:
                tail = tail.split(b'\n', 1)[1]
            records = tail.splitlines(keepends=True)[-500:]
            for record in records:
                if not record.endswith(b'\n'):
                    raise ValueError('committed feed ends in a partial row')
                values = next(csv.reader([record.decode('utf-8')], strict=True))
                if len(values) != len(fields):
                    raise ValueError('feed row is not a complete single-line record')
            final = os.fstat(source.fileno())
            current = feed.stat()
            if (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns) != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns) or (current.st_dev, current.st_ino) != (initial.st_dev, initial.st_ino):
                raise ValueError('feed changed during preparation')
        # Independently atomic derived files can transiently disagree after a crash.
        # They are never authoritative; dispatch validates the pair below.
        snapshot = header + b''.join(records)
        atomic_bytes(state / 'window.csv', snapshot)
        manifest = {
            'kind': 'derived_last_500_complete_records_not_authoritative',
            'prepared_at': datetime.now(timezone.utc).isoformat(),
            'source': str(feed), 'committed_feed': committed,
            'pending_partial_bytes': len(suffix), 'rows': len(records),
            'window_sha256': hashlib.sha256(snapshot).hexdigest(),
            'wrapper_sha256': hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            'timestamps_and_columns_preserved': True,
        }
        atomic_bytes(state / 'window-manifest.json', (json.dumps(manifest, sort_keys=True, indent=2) + '\n').encode())
        return manifest


def validate_window(state, expected):
    manifest = json.loads((state / 'window-manifest.json').read_text())
    if manifest != expected or hashlib.sha256((state / 'window.csv').read_bytes()).hexdigest() != manifest['window_sha256']:
        raise ValueError('derived window/manifest mismatch; dispatch denied')


def index_collector(journal):
    if not journal.is_relative_to(RUNS.resolve()):
        raise ValueError('collector index writes must remain under isolated research runs')
    db = sqlite3.connect(journal.as_uri() + '?mode=rw', uri=True, timeout=1)
    try:
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='invalid_rows'").fetchone()
        named = db.execute("SELECT tbl_name FROM sqlite_master WHERE type='index' AND name='mim_feed_event_contract'").fetchone()
        if named:
            columns = [row[2] for row in db.execute('PRAGMA index_info(mim_feed_event_contract)')]
            partial = next(row[4] for row in db.execute("SELECT * FROM pragma_index_list(?)", (named[0],)) if row[1] == 'mim_feed_event_contract')
            if named[0] != 'invalid_rows' or columns != ['event', 'contract'] or partial:
                raise ValueError('mim_feed_event_contract index has incompatible table/columns or predicate')
        if exists:
            db.execute('CREATE INDEX IF NOT EXISTS mim_feed_event_contract ON invalid_rows(event,contract)')
            db.commit()
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description='Finite adapter plus bounded shadow-reader poll')
    parser.add_argument('--prepare-only', action='store_true', help='Verify and prepare a fixture/research window; invoke no collectors')
    parser.add_argument('--state', type=Path, help='Isolated wrapper state; prepare-only defaults to a separate fixture state')
    parser.add_argument('--feed', type=Path, help='Prepare-only identified feed.csv under research runs')
    parser.add_argument('--collector-journal', type=Path, help='Prepare-only fixture collector index preflight under research runs')
    args = parser.parse_args()
    state = (args.state or (PREPARE_STATE if args.prepare_only else LIVE_STATE)).resolve()
    if args.prepare_only and state == LIVE_STATE.resolve():
        parser.error('prepare-only cannot use live wrapper state')
    feed = (args.feed or ADAPTER / 'feed.csv').resolve()
    if not args.prepare_only and state != LIVE_STATE.resolve():
        parser.error('custom --state is allowed only with --prepare-only; live polls share one lock')
    if (args.feed or args.collector_journal) and not args.prepare_only:
        parser.error('fixture overrides are allowed only with --prepare-only')
    if not state.is_relative_to(RUNS.resolve()) or state == RUNS.resolve():
        parser.error('all wrapper writes must be under isolated research runs')
    if not feed.is_relative_to(RUNS.resolve()) or feed.name != 'feed.csv':
        parser.error('feed must be an identified feed.csv under isolated research runs')
    if state == feed.parent or state.is_relative_to(feed.parent) or feed.is_relative_to(state):
        parser.error('wrapper state must be separate from authoritative adapter state')
    journal = args.collector_journal.resolve() if args.collector_journal else (COLLECTOR / 'journal.sqlite').resolve()
    if not journal.is_relative_to(RUNS.resolve()):
        parser.error('collector index writes must remain under isolated research runs')
    state.mkdir(parents=True, exist_ok=True)
    with open(state / 'lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze = {
            'wrapper_sha256': hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
            'feed': str(feed), 'window_rows': 500,
            'historical_run': str(HISTORY), 'collector': str(COLLECTOR),
            'warmup': str(WARMUP), 'labels': 'end',
            'prepare_only': args.prepare_only, 'collector_journal': str(journal),
        }
        freeze_path = state / 'freeze.json'
        if freeze_path.exists():
            if json.loads(freeze_path.read_text()) != freeze:
                raise ValueError('frozen wrapper source/configuration drift')
        else:
            atomic_bytes(freeze_path, (json.dumps(freeze, sort_keys=True, indent=2) + '\n').encode())
        if not args.prepare_only:
            subprocess.run([sys.executable, '-m', 'research.mim_comparison.feed_adapter', '--bars', str(ROOT / 'data/mim_nb/bars_raw.csv'), '--log', str(ROOT / 'logs/mim_nb_live.log'), '--log-timezone', 'UTC', '--state', str(ADAPTER)], cwd=ROOT, check=True)
        if not args.prepare_only or args.collector_journal:
            index_collector(journal)
        manifest = prepare(feed, state)
        print(json.dumps(dict(manifest, window=str(state / 'window.csv')), sort_keys=True), flush=True)
        validate_window(state, manifest)
        if not args.prepare_only:
            subprocess.run([sys.executable, '-m', 'research.mim_comparison', 'shadow', '--data', str(state / 'window.csv'), '--labels', 'end', '--warmup', str(WARMUP), '--historical-run', str(HISTORY), '--state', str(COLLECTOR)], cwd=ROOT, check=True)


try:
    main()
except Exception as error:
    print('Finite poll failed: ' + str(error), file=sys.stderr)
    raise SystemExit(1)
PY
