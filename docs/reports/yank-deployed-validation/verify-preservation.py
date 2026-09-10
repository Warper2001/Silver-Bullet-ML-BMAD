"""Read-only verification of the task's retained source/artifact pins."""
import hashlib
import json
from pathlib import Path


def verify():
    evidence = json.loads(Path(__file__).with_name('preservation.json').read_text())
    mismatches = []
    for name, expected in evidence['sha256'].items():
        checksum = hashlib.sha256()
        with Path(name).open('rb') as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                checksum.update(block)
        if checksum.hexdigest() != expected:
            mismatches.append(name)
    print(json.dumps({'decision': 'HOLD_VALIDATION', 'checked': len(evidence['sha256']), 'mismatches': mismatches}, sort_keys=True))
    if mismatches:
        raise SystemExit(1)


if __name__ == '__main__':
    verify()
