"""Independent exact-rational ledger audit; never imports the calibration runner."""

from collections import Counter, defaultdict
from fractions import Fraction
from math import isclose, sqrt

from scipy.stats import t

if not __debug__:
    raise SystemExit("Audit refuses optimized execution: assertions are required.")


def reference_summary(values, groups):
    """Reconstruct the registered estimand from persisted values and labels."""
    numbers = [Fraction(str(value)) for value in values]
    assert len(numbers) == len(groups)
    n = len(numbers)
    sizes = Counter(groups)
    g = len(sizes)
    mean = sum(numbers, Fraction()) / n if n else None
    answer = {
        "n": n,
        "g": g,
        "mean": float(mean) if mean is not None else None,
        "se": None,
        "t_critical": None,
    }
    if n < 2:
        return dict(answer, variance=None, sd=None, ci=None)
    sample_variance = sum((value - mean) ** 2 for value in numbers) / (n - 1)
    answer["sd"] = sqrt(float(sample_variance))
    if g < 2:
        return dict(answer, variance=None, ci=None)
    residuals = defaultdict(Fraction)
    for value, group in zip(numbers, groups):
        residuals[group] += value - mean
    variance = Fraction(g, g - 1) * sum(r * r for r in residuals.values()) / n**2
    if variance <= 0:
        return dict(answer, variance=None, ci=None)
    se = sqrt(float(variance))
    critical = float(t.ppf(0.975, g - 1))
    radius = critical * se
    return dict(
        answer,
        variance=float(variance),
        se=se,
        t_critical=critical,
        ci=[float(mean) - radius, float(mean) + radius],
    )


def assert_number(actual, expected, label):
    if expected is None:
        assert actual is None, (label, actual, expected)
    else:
        assert isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9), (
            label,
            actual,
            expected,
        )


if __name__ == "__main__":
    import json
    from pathlib import Path

    cases = json.loads(Path(__file__).with_name("independent-oracles.json").read_text())
    for case in cases:
        actual = reference_summary(case["values"], case["labels"])
        assert_number(actual["variance"], case["variance"], "variance")
        assert_number(actual["mean"], float(Fraction(case["mean_exact"])), "mean")
    assert reference_summary([0.1] * 3, ["a", "b", "c"])["variance"] is None
    assert reference_summary([1, 3, 1, 3], ["a", "a", "b", "b"])["variance"] is None
    print(
        "Independent rational audit helpers verified on four cluster oracles "
        "and degenerate cases."
    )
