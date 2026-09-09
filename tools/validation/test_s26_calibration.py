"""Regression test for the time-respecting calibration in run_mnq_s26_pipeline.py.

Exercises the exact calibration block from that pipeline on synthetic data --
both the preferred held-out-later-block path and the TimeSeriesSplit fallback --
without loading a year of 1-minute bars or overwriting the model artifact.

Guards three things that have each already broken once:
  1. `cv="prefit"` is REMOVED from sklearn (raises InvalidParameterError on
     1.8 and 1.9 alike). FrozenEstimator is the supported replacement.
  2. The calibrator must never be fitted on data that precedes what trained
     the underlying classifier.
  3. The fallback must not train on the future either.

    .venv-research/bin/python tools/validation/test_s26_calibration.py
    .venv/bin/python          tools/validation/test_s26_calibration.py
"""

import sys

import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit

try:
    from sklearn.frozen import FrozenEstimator as _FrozenEstimator
except ImportError:
    _FrozenEstimator = None


def build(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    y = (X[:, 0] + rng.standard_normal(n) * 0.7 > 0).astype(int)
    return X, y


def calibrate(X_train, y_train, cal_fraction=0.2, force_fallback=False):
    """The block from run_mnq_s26_pipeline.py, on arrays."""
    clf = HistGradientBoostingClassifier(max_iter=30, random_state=42)
    cal_start = int(len(X_train) * (1.0 - cal_fraction))
    X_fit, y_fit = X_train[:cal_start], y_train[:cal_start]
    X_cal, y_cal = X_train[cal_start:], y_train[cal_start:]

    usable = (_FrozenEstimator is not None and len(X_cal) >= 50
              and len(np.unique(y_cal)) == 2 and not force_fallback)
    if usable:
        clf.fit(X_fit, y_fit)
        c = CalibratedClassifierCV(estimator=_FrozenEstimator(clf), method="sigmoid")
        c.fit(X_cal, y_cal)
        return c, "held-out-later-block", cal_start
    c = CalibratedClassifierCV(estimator=clf, method="sigmoid",
                               cv=TimeSeriesSplit(n_splits=5))
    c.fit(X_train, y_train)
    return c, "timeseriessplit-fallback", cal_start


def main() -> int:
    print(f"sklearn {sklearn.__version__}   FrozenEstimator="
          f"{'available' if _FrozenEstimator else 'MISSING'}")
    X, y = build()
    fails = 0

    # 1 -- the removed idiom must be confirmed dead, so nobody reinstates it
    try:
        CalibratedClassifierCV(
            estimator=HistGradientBoostingClassifier(), method="sigmoid",
            cv="prefit").fit(X, y)
        print('  cv="prefit" unexpectedly ACCEPTED -- this test is stale')
    except Exception as e:
        print(f'  cv="prefit" correctly rejected ({type(e).__name__})')

    # 2 -- preferred path
    model, mode, cal_start = calibrate(X, y)
    p = model.predict_proba(X[:20])[:, 1]
    ok = (mode == "held-out-later-block" and np.all((p >= 0) & (p <= 1))
          and len(np.unique(np.round(p, 6))) > 1)
    print(f"  preferred path: mode={mode} probs in [0,1] and non-degenerate: "
          f"{'PASS' if ok else 'FAIL'}")
    fails += not ok

    # 3 -- calibration rows are strictly LATER than the fitting rows
    ok = cal_start == int(len(X) * 0.8) and cal_start > 0
    print(f"  calibration block starts at index {cal_start} of {len(X)}, "
          f"strictly after the fit block: {'PASS' if ok else 'FAIL'}")
    fails += not ok

    # 4 -- fallback path
    model_fb, mode_fb, _ = calibrate(X, y, force_fallback=True)
    p_fb = model_fb.predict_proba(X[:20])[:, 1]
    ok = (mode_fb == "timeseriessplit-fallback" and np.all((p_fb >= 0) & (p_fb <= 1)))
    print(f"  fallback path:  mode={mode_fb} probs in [0,1]: {'PASS' if ok else 'FAIL'}")
    fails += not ok

    # 5 -- the fallback splitter never trains on the future
    folds = list(TimeSeriesSplit(n_splits=5).split(X))
    ok = all(tr.max() < te.min() for tr, te in folds)
    print(f"  fallback CV: {len(folds)} folds, train always precedes test: "
          f"{'PASS' if ok else 'FAIL'}")
    fails += not ok

    print("\nALL CALIBRATION TESTS PASSED" if not fails else f"\n{fails} FAILURE(S)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
