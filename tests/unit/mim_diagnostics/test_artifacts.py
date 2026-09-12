import pytest
from research.mim_diagnostics import artifacts as a
from research.mim_diagnostics.feasibility import validate_inventory, REQUIRED_CATEGORIES


def test_output_refuses_existing_and_escape(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "RUNS", tmp_path / "runs")
    run = a.create("audit", tmp_path / "runs" / "one")
    with pytest.raises(FileExistsError):
        a.create("audit", run)
    with pytest.raises(ValueError):
        a.create("audit", tmp_path / "escape")
    (tmp_path / "runs" / "link").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError):
        a.create("audit", tmp_path / "runs" / "link")


def test_holdout_path_refused_before_read(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "ORIGINAL", tmp_path)
    for name in ("data/sealed_holdout/bars.csv", "sealed_holdout/run"):
        with pytest.raises(ValueError, match="Holdout"):
            a.permitted(tmp_path / name)
    (tmp_path / "alias").symlink_to(tmp_path / "data/sealed_holdout/bars.csv")
    with pytest.raises(ValueError, match="Holdout"):
        a.permitted(tmp_path / "alias")


def test_inventory_detects_addition_and_missing_required_even_resealed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(a, "RUNS", tmp_path / "runs")
    run = a.create("audit")
    (run / "small").write_text("ok")
    a.seal(run)
    with pytest.raises(ValueError, match="mandatory"):
        a.verify(run)
    (run / "unexpected").write_text("bad")
    with pytest.raises(ValueError, match="inventory"):
        a.verify_inventory(run)


def test_missing_or_duplicate_feasibility_category():
    with pytest.raises(ValueError):
        validate_inventory({"categories": []})
    with pytest.raises(ValueError):
        validate_inventory(
            {
                "categories": [
                    dict(category=c, status="unavailable", missing_evidence="missing")
                    for c in sorted(REQUIRED_CATEGORIES)
                ],
                "observed_at": "2026-09-12",
                "official_checks": [1],
                "correction": {"x": 1},
            }
        )


def test_cli_preserves_failed_run(tmp_path, monkeypatch):
    import sys
    from research.mim_diagnostics.__main__ import main

    monkeypatch.setattr(a, "RUNS", tmp_path / "runs")
    output = tmp_path / "runs" / "failed"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnostics",
            "audit",
            "--data",
            str(tmp_path / "missing.csv"),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError):
        main()
    assert (output / "failure.json").is_file()
    a.verify_inventory(output)
    with pytest.raises(ValueError, match="Failed invocation"):
        a.verify(output)
