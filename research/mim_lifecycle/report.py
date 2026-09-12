"""Portable scientific plots and an observation-only narrative."""

from html import escape
import json
from io import StringIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

CAUTIONS = """Run files are made read-only and their hashes are verified to detect tampering. This does not prevent replacement through writable directories.

This is a descriptive study of exposed history for unchanged A, delay2, one contract. It tests no alternative stop, filter, entry, or holding rule and makes no superiority or deployability inference. A future strategy test requires a power gate and committed preregistration.

Gross MFE/MAE use only experienced completed-bar highs/lows and known entry/exit fills. They are nonnegative dollar excursions relative to the actual fill ($2/point). Stop-bar OHLC is excluded; stop excursions are observed lower bounds and stop duration/fill-derived event times are minute intervals. Net marks reserve the full $2.24 roundtrip friction at every hypothetical liquidation; the entry mark is not counted as a negative completed close.

Observation checkpoints are not trading thresholds. At-risk means an exact completed close at entry opening time plus the checkpoint, observed before the trade exits (a completed close immediately preceding an opening exit is included). Exited trades are outside the denominator; ambiguous stop-minute checkpoints without a surviving close are separately identified as a subset of already-exited trades. Recovery means a nonnegative mark at a strictly later valid close or the later terminal fill, after a negative close. It is not a counterfactual return. Green includes exactly zero.

Time to first positive is censored when no positive completed close or terminal fill was observed. Summaries retain never-positive counts; quantiles among observed times do not impute censored trades. Eventual-winner cohorts and the top 5% trades / top 67 daily dates are hindsight selections, not causal entry information. Excursion and first-profit plots avoid aligned-path survivorship; first-profit distributions still condition on an observed positive mark, with censored counts displayed. These observations do not establish that any pattern will persist out of sample."""


def fmt(value):
    return "—" if value is None else f"{value:,.2f}"


def markdown(summary):
    out = [
        "# MIM-NB trade lifecycle diagnostic",
        "",
        CAUTIONS,
        "",
        f"Accounting: {summary['trade_count']} trades across {summary['sessions']} daily sessions ({summary['flat_sessions']} flat), net ${summary['total_net']:,.2f}; daily reconciliation ${summary['daily_net']:,.2f}.",
        "",
        "## Observed lifecycle by final outcome",
        "",
        "| Final outcome | N | Net $ | MFE median $ | MAE median $ | Never positive | Negative observed | Later recovery |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in summary["groups"]["final_winner"].items():
        out.append(
            f"| {'Winner' if key == 'True' else 'Nonwinner'} | {row['count']} | {fmt(row['net'])} | {fmt(row['quantiles']['mfe']['0.5'])} | {fmt(row['quantiles']['mae']['0.5'])} | {row['never_positive']} | {row['had_negative']} | {row['recovered_after_first_negative']} |"
        )
    out.extend(
        [
            "",
            "## Landmark cohorts",
            "",
            "| Elapsed min | At risk | Already exited | Stop interval unknown | Cohort | N | Final winner rate | Mean final $ | Mean remaining change $ | Recovered after red |",
            "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for mark in summary["landmarks"]:
        for key, row in mark["cohorts"].items():
            out.append(
                f"| {mark['checkpoint_min']} | {mark['at_risk']} | {mark['already_exited']} | {mark.get('stop_interval_unobserved', 0)} | {key} | {row['count']} | {fmt(row['eventual_winner_rate'])} | {fmt(row['mean_final_net'])} | {fmt(row['mean_remaining_net_change'])} | {row['recovered_after_red'] if key == 'red' else '—'} |"
            )
    out.extend(
        [
            "",
            "## Positive-close history at each landmark",
            "",
            "| Elapsed min | Observed history | N at risk | Final winner rate | Mean final $ | Mean remaining change $ |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for mark in summary["landmarks"]:
        for label, row in mark["positive_history_cohorts"].items():
            out.append(
                f"| {mark['checkpoint_min']} | {label} | {row['count']} | {fmt(row['eventual_winner_rate'])} | {fmt(row['mean_final_net'])} | {fmt(row['mean_remaining_net_change'])} |"
            )
    for field, title in (
        ("direction", "Direction (1 long, -1 short)"),
        ("year", "Year"),
        ("exit_reason", "Exit reason"),
        (
            "hindsight_top_5pct_trade",
            "Hindsight: largest ceil(5% × N) final-net trades",
        ),
        ("hindsight_top_67_day", "Hindsight: baseline top 67 daily-net dates"),
    ):
        out.extend(
            [
                "",
                "## " + title,
                "",
                "| Group | N | Net $ | Contribution to total net | MAE median $ | Negative observed | Later recovery | Never positive |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key, row in summary["groups"][field].items():
            out.append(
                f"| {key} | {row['count']} | {fmt(row['net'])} | {fmt(row['net_contribution'])} | {fmt(row['quantiles']['mae']['0.5'])} | {row['had_negative']} | {row['recovered_after_first_negative']} | {row['never_positive']} |"
            )
    overall = summary["overall"]
    out.extend(
        [
            "",
            "## Time to first positive",
            "",
            f"Never observed positive: {overall['never_positive']} of {overall['count']} trades. Observed first-positive elapsed-minute quantiles (lower endpoints): {json.dumps(overall['quantiles']['first_positive_min_lower'], sort_keys=True)}. Upper endpoints and full group quantiles are in summary.json. Close-only first-positive timings and never-positive-close counts are also recorded separately in lifecycle.csv and summary.json; first_positive includes terminal fills with its basis stated. First-negative and subsequent-recovery times are recorded in lifecycle.csv.",
            "",
            "## Future questions, untested",
            "",
            "Do the early-red recovery patterns recur in unseen history? Does dependence on a small set of large winners persist across independent periods? No rule, cutoff, or trade recommendation is inferred here. Any later intervention is a separate preregistered experiment, one parameter at a time.",
            "",
            "Artifacts: paths.csv contains all observed points and cumulative excursions; lifecycle.csv contains per-trade timings and hindsight cohort labels; landmarks.csv retains every trade/checkpoint including excluded observations; summary.json contains counts, quantiles, and contribution accounting. report.html embeds standalone SVG plots with no external assets.",
        ]
    )
    return "\n".join(out) + "\n"


def figure_svg(fig):
    buffer = StringIO()
    with matplotlib.rc_context(
        {"svg.hashsalt": "mim-lifecycle", "font.family": "DejaVu Sans"}
    ):
        fig.savefig(buffer, format="svg", metadata={"Date": None}, bbox_inches="tight")
    plt.close(fig)
    svg = buffer.getvalue()
    return svg[svg.index("<svg") :]


def scatter_svg(life):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for winner, color, label in (
        (True, "#167549", "Eventual winner"),
        (False, "#a63333", "Nonwinner"),
    ):
        frame = life[life.final_winner == winner]
        ax.scatter(
            frame.mae,
            frame.mfe,
            color=color,
            alpha=0.5,
            s=18,
            label=f"{label} (N={len(frame)})",
        )
    stops = life[life.stop_time_uncertain]
    ax.scatter(
        stops.mae,
        stops.mfe,
        facecolors="none",
        edgecolors="#111111",
        s=32,
        label=f"Stop lower bounds (N={len(stops)})",
    )
    ax.set(
        xlabel="MAE ($), gross adverse excursion",
        ylabel="MFE ($), gross favorable excursion",
        title="Observed gross excursions by eventual outcome (hindsight)",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.text(
        0.12,
        -0.02,
        "Stop-minute OHLC excluded; known stop fills included. Stop excursions are observed lower bounds.",
        fontsize=9,
    )
    return figure_svg(fig)


def profit_svg(life):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for winner, color, label in (
        (True, "#167549", "Eventual winner"),
        (False, "#a63333", "Nonwinner"),
    ):
        frame = life[life.final_winner == winner]
        observed = np.sort(frame.first_positive_min_upper.dropna().to_numpy(float))
        x = np.r_[0, observed]
        y = np.r_[0, np.arange(1, len(observed) + 1) / max(len(observed), 1)]
        ax.step(
            x,
            y,
            where="post",
            color=color,
            label=f"{label}: observed {len(observed)}/{len(frame)}, never positive {int(frame.never_positive.sum())}",
        )
    ax.set(
        xlabel="Elapsed minutes (upper interval endpoint for stop fills)",
        ylabel="Empirical CDF among observed positives",
        title="Time to first observed positive net close or terminal fill",
        ylim=(0, 1.02),
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.text(
        0.12,
        -0.02,
        "Censored trades are counted, not imputed. Eventual outcome is hindsight; no aligned-path survivor conditioning.",
        fontsize=9,
    )
    return figure_svg(fig)


def write_report(path, summary, life):
    md = markdown(summary)
    with open(path / "report.md", "x") as stream:
        stream.write(md)
    plots = [
        ("excursions.svg", scatter_svg(life)),
        ("first_positive.svg", profit_svg(life)),
    ]
    for name, svg in plots:
        with open(path / name, "x") as stream:
            stream.write(svg)
    with open(path / "report.html", "x") as stream:
        stream.write(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>MIM-NB lifecycle diagnostic</title><style>body{max-width:1100px;margin:2rem auto;font-family:system-ui,sans-serif;padding:1rem;color:#20252a}svg{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit;line-height:1.5}</style><main><h1>MIM-NB lifecycle diagnostic</h1>'
            + "\n".join(svg for _, svg in plots)
            + "<pre>"
            + escape(md)
            + "</pre></main></html>"
        )
