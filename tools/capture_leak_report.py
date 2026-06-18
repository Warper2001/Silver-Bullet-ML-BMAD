"""capture_leak_report.py — CLI for the offline capture leak-detector.

Usage:
    PYTHONPATH=. .venv/bin/python tools/capture_leak_report.py --bot mim [--verify-chains]
    PYTHONPATH=. .venv/bin/python tools/capture_leak_report.py --bot yank [--since 2026-06-17] [--ml-threshold 0.50]

Reconciles a bot's realized live fills against its backtest-theoretical fills and
prints a markdown ΔMaxGiveback report (sliced by regime x drawdown-state, with
fat-tail days isolated). Self-marks UNTRUSTED if the conservation/parity gates
fail. Measurement-only: touches no live trading code.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from src.research.capture_recon import ReconConfig, render_markdown, run_recon

REPO_ROOT = Path("/root/Silver-Bullet-ML-BMAD")


def _parse_since(s: str | None) -> datetime | None:
    if not s:
        return None
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser(description="Capture leak-detector report")
    ap.add_argument("--bot", choices=["yank", "mim"], required=True)
    ap.add_argument("--since", default=None, help="ISO date; only reconcile trades at/after")
    ap.add_argument("--ml-threshold", type=float, default=None, help="YANK only")
    ap.add_argument("--verify-chains", action="store_true", default=False, help="MIM hash-chain check")
    ap.add_argument("--repo-root", default=str(REPO_ROOT))
    args = ap.parse_args()

    repo_root = Path(args.repo_root)
    since = _parse_since(args.since)

    from src.research import capture_loaders as L

    extra_untrusted: list[str] = []

    if args.bot == "yank":
        real = L.load_yank_realized(repo_root, since=since)
        theo = L.load_yank_theoretical(repo_root, ml_threshold=args.ml_threshold, since=since)
        cfg = ReconConfig(
            bot_id="YANK",
            label_offset_min=0,
            confound_note="bar-source — theoretical replays processed CSV, not YANK's "
                          "live-seen bars (YANK logs none). Treat leak figures as upper-bounded "
                          "by feed divergence; MIM is the faithful reference.",
        )
    else:  # mim
        real, chain_ok = L.load_mim_realized(repo_root, since=since, verify_chains=args.verify_chains)
        real = L.tag_mim_regime(real, repo_root)
        theo = L.load_mim_theoretical(repo_root)
        if args.verify_chains and not chain_ok:
            extra_untrusted.append("MIM hash-chain verification FAILED (structural break)")
        cfg = ReconConfig(bot_id="MIM-NB", label_offset_min=0)

    report = run_recon(theo, real, cfg)
    if extra_untrusted:
        report.attribution.trusted = False
        report.attribution.untrusted_reasons.extend(extra_untrusted)

    print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
