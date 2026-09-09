"""Rebuild disabled preparation overlay from current HEAD; run only on clean approved inputs."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

root = Path(__file__).resolve().parents[3]
out = root / 'docs/reports/yank-evidence-preparation'
paths = sorted({
    *[str(p.relative_to(root)) for p in (root/'src/research/yank_deployed_validation').glob('*.py')],
    'src/cli/check_yank_deployed_replay.py',
    'src/cli/compare_yank_decision_capture.py',
    'deploy/yank-observation.disabled.json',
    'docs/yank-evidence-pilot-preparation.md',
    'docs/yank-tradestation-semantics.md',
})
contents = {p: (root/p).read_bytes() for p in paths}
config = json.loads(contents['deploy/yank-observation.disabled.json'])
assert config['enabled'] is False and config['historical_acquisition_enabled'] is False
assert config['authorized_incremental_spend_usd'] == 0
assert not config['sessions']
sha = lambda b: hashlib.sha256(b).hexdigest()
manifest = {
    'schema_version': 1,
    'decision': 'HOLD_VALIDATION',
    'purpose': 'Disabled observation preparation overlay; not an installation authorization',
    'code_revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
    'base_revision': 'a478d674e8d507928cec843b139f086d14d85296',
    'required_snapshot_manifest_sha256': sha((root/'docs/yank-validation/snapshot/v1/manifest.json').read_bytes()),
    'files': {p: {'sha256': sha(data), 'bytes': len(data)} for p, data in contents.items()},
    'operational_gates': {
        'installation_authorized': False,
        'restart_authorized': False,
        'live_collection_authorized': False,
        'acquisition_authorized': False,
        'incremental_spend_usd': 0,
        'production_expected_release': None,
        'production_signer': None,
        'production_trusted_public_keys': None,
        'actual_observation_sessions': [],
    },
    'requirements': [
        'Existing repository and pinned snapshot/dependencies; this overlay does not replace the installed strategy.',
        'Follow docs/yank-evidence-pilot-preparation.md after concrete maintenance approval.',
        'Supply independently approved runtime/checkpoint/process/key inputs outside captures.',
    ],
}
manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode()
archive_members = dict(contents, **{'release-manifest.json': manifest_bytes})
stream = io.BytesIO()
with gzip.GzipFile(fileobj=stream, mode='wb', filename='', mtime=0) as zipped:
    with tarfile.open(fileobj=zipped, mode='w') as tar:
        for name, data in sorted(archive_members.items()):
            info = tarfile.TarInfo('yank-observation-preparation/'+name)
            info.size = len(data); info.mode = 0o644; info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
archive = stream.getvalue()
with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
    for member in tar.getmembers():
        assert member.isfile()
        key = member.name.removeprefix('yank-observation-preparation/')
        assert tar.extractfile(member).read() == archive_members[key]
for name, data in contents.items():
    assert (root/name).read_bytes() == data, 'release input changed: '+name
(out/'release-manifest.json').write_bytes(manifest_bytes)
(out/'yank-observation-preparation.tar.gz').write_bytes(archive)
report = {'decision':'HOLD_VALIDATION', 'package_sha256':sha(archive), 'manifest_sha256':sha(manifest_bytes),
          'package_bytes':len(archive), 'verified_members':len(archive_members), 'all_members_match_sources':True,
          'enabled':False, 'code_revision':manifest['code_revision']}
(out/'release-package-verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps(report, sort_keys=True))
