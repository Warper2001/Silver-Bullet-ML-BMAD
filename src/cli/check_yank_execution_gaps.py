"""Offline incremental execution-gap assessment, with no research-package import."""
import importlib.util
from pathlib import Path


def main(argv=None):
    path = Path(__file__).resolve().parents[1] / 'research/yank_deployed_validation/gaps.py'
    spec = importlib.util.spec_from_file_location('_yank_execution_gaps', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
