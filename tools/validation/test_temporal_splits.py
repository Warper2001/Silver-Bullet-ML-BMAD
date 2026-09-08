"""Regression test for the time-respecting splits in scripts/.

Each patched script defines a local `temporal_split` (and most a `ts_cv`).
This asserts the properties that make them leak-free, by feeding a
DELIBERATELY SHUFFLED index -- a helper that trusts input order instead of
sorting will fail here, which is exactly the regression worth catching.

Standalone (not part of the pytest suite) because loading these scripts
pulls heavy optional deps such as hmmlearn:

    .venv-research/bin/python tools/validation/test_temporal_splits.py
"""
import importlib.util
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except SystemExit:
        pass
    return m


PATHS = [
    "scripts/tune_regime_1_quick.py",
    "scripts/train_regime_models_real_labels.py",
    "scripts/tune_regime_1_model.py",
    "scripts/train_regime_specific_models.py",
]

rng = np.random.default_rng(0)
fails = 0
for path in PATHS:
    m = load(path, path.replace("/", "_").replace(".py", ""))
    # deliberately shuffled index: temporal_split must restore chronological order
    idx = rng.permutation(100)
    X = pd.DataFrame({"f": np.arange(100.0)}, index=idx)
    y = pd.Series(np.arange(100) % 2, index=idx)
    Xtr, Xte, ytr, yte = m.temporal_split(X, y, test_size=0.2)

    checks = {
        "sizes 80/20": (len(Xtr), len(Xte)) == (80, 20),
        "train strictly precedes test": Xtr.index.max() < Xte.index.min(),
        "disjoint": not (set(Xtr.index) & set(Xte.index)),
        "X/y aligned": (ytr.index == Xtr.index).all() and (yte.index == Xte.index).all(),
    }
    bad = [k for k, v in checks.items() if not v]
    fails += len(bad)
    print(f"{path:46s} {'PASS' if not bad else 'FAIL ' + str(bad)}")

    if hasattr(m, "ts_cv"):
        folds = list(m.ts_cv(5).split(np.zeros((100, 1))))
        ok = all(tr.max() < te.min() for tr, te in folds)
        fails += not ok
        print(f"{'  ts_cv: %d folds, train always precedes test' % len(folds):46s} "
              f"{'PASS' if ok else 'FAIL'}")

print("\nALL HELPER TESTS PASSED" if not fails else f"\n{fails} FAILURE(S)")
sys.exit(1 if fails else 0)
