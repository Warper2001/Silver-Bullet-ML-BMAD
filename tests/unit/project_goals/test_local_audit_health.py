"""Audit advisory output is excluded from optional external alerting."""

import importlib.util
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def test_external_filter_excludes_local_audit_even_with_warn_text():
    script = (ROOT / "tools/combine_ops_alert.sh").read_text()
    line = next(line for line in script.splitlines() if line.startswith("push_out="))
    result = subprocess.run(
        [
            "bash",
            "-c",
            "out=$'[LOCAL AUDIT] WARN audit failure\\n[ WARN] real existing alert'\n"
            + line
            + '\nprintf "%s" "$push_out"',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == "[ WARN] real existing alert"
    assert '"$push_out" | grep' in script


def test_unavailable_audit_health_stays_explicit(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "ops_fixture", ROOT / "tools/combine_ops_healthcheck.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "BASE", tmp_path)
    result = module.broker_audit_local_status()
    assert isinstance(result, dict) and result["status"] != "HEALTHY"
