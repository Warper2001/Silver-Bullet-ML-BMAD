"""Independent disk-failure verification for disabled prospective capture."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('capture_matrix_cli',ROOT/'src/cli/check_yank_deployed_replay.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)

def test_real_capture_writer_failure_invalidates_coverage(tmp_path,monkeypatch):
    capture=cli.load_tool('capture')
    original=Path.open
    def failing_open(path,*args,**kwargs):
        if path.name=='capture.jsonl':raise OSError('injected disk failure')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',failing_open)
    c=capture.DecisionCapture(enabled=True,output_dir=tmp_path/'capture')
    c.emit({'kind':'test'})
    summary=c.close()
    assert summary['valid_coverage'] is False
    assert 'writer_failure' in summary['invalid_reasons']
    assert summary['written']==0
