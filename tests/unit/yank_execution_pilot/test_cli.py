"""Exercise the command in a fresh interpreter with eager/live imports forbidden."""
import subprocess
import sys
from pathlib import Path


def test_command_loads_real_audit_without_research_initializer_or_live_dependencies():
    root = Path(__file__).resolve().parents[3]
    code = r'''
import importlib.abc
import sys
class BlockEager(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "src.research" or fullname.startswith(("src.live", "src.execution", "src.broker", "src.trading")):
            raise AssertionError("Forbidden import: " + fullname)
sys.meta_path.insert(0, BlockEager())
from src.cli import check_yank_execution_pilot as command
real_load = command.load_audit
calls = []
def load():
    audit = real_load()
    assert audit.Scenario.__module__ == "_yank_execution_pilot.audit"
    audit.run = lambda output: calls.append(output)
    return audit
command.load_audit = load
assert command.main(["--output-dir", "unused-fresh-output"]) == 0
assert calls == ["unused-fresh-output"]
assert "src.research" not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=root, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'PASS_AUDIT_CHECKS; HOLD_VALIDATION' in result.stdout
