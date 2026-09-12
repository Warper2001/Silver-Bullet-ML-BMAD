"""Full fixed-batch reporting. No historical result authorizes deployment."""

from html import escape
import json

import numpy as np
import pandas as pd

from .artifacts import write_json, CONFIG
from .statistics import ARMS, metrics, paired_matrix, stationary_bootstrap, classify

NAMES = dict(
    A="Unchanged MIM-NB",
    R="Regression agreement",
    E="Path efficiency",
    F="Fresh-breakout reset",
    P="Breakout persistence",
)


def number(value):
    return "undefined" if value is None else f"{value:,.4f}"


def diagnostics(daily, trades):
    rows = []
    for (arm, delay, outcome), frame in trades.assign(
        outcome=np.where(trades.net > 0, "winner", "nonwinner")
    ).groupby(["arm", "delay", "outcome"]):
        item = dict(
            arm=arm,
            delay=int(delay),
            outcome=outcome,
            trades=len(frame),
            net=float(frame.net.sum()),
            gross=float(frame.gross.sum()),
        )
        for col in (
            "entry_slope",
            "entry_r2",
            "entry_efficiency",
            "entry_displacement",
        ):
            values = frame[col].dropna()
            item[col] = (
                dict(
                    mean=float(values.mean()),
                    median=float(values.median()),
                    q25=float(values.quantile(0.25)),
                    q75=float(values.quantile(0.75)),
                )
                if len(values)
                else None
            )
        rows.append(item)
    exits = [
        dict(
            arm=arm,
            delay=int(delay),
            exit_reason=reason,
            trades=len(g),
            gross=float(g.gross.sum()),
            net=float(g.net.sum()),
            costs=float(g.costs.sum()),
        )
        for (arm, delay, reason), g in trades.groupby(["arm", "delay", "exit_reason"])
    ]
    winner_rows = []
    for (delay, cost), group in daily.groupby(["delay", "cost"]):
        base = group[group.arm == "A"].sort_values(
            ["net", "day"], ascending=[False, True]
        )
        k = max(1, int(np.ceil(0.05 * len(base))))
        days = base[base.net > 0].head(k).day
        baseline_net = float(base[base.day.isin(days)].net.sum())
        for arm in ARMS:
            candidate = group[group.arm == arm]
            on_days = candidate[candidate.day.isin(days)]
            winner_rows.append(
                dict(
                    arm=arm,
                    delay=int(delay),
                    cost=float(cost),
                    baseline_top_days=len(days),
                    active_on_top_days=int((on_days.turnover > 0).sum()),
                    net_on_top_days=float(on_days.net.sum()),
                    top_day_net_retention=(
                        float(on_days.net.sum() / baseline_net)
                        if baseline_net
                        else None
                    ),
                    net_outside_baseline_top_days=float(
                        candidate[~candidate.day.isin(days)].net.sum()
                    ),
                )
            )
    return dict(
        feature_distributions=rows,
        exit_attribution=exits,
        winner_retention=winner_rows,
        limitation="Descriptive historical association only; no threshold fitting. Nonwinner includes zero-net trades. Execution rejection/latency losses cannot be inferred from modeled history alone.",
    )


def svg_paths(matrix, drawdown=False):
    colors = ["#243b53", "#c34e36", "#278064", "#8064a2", "#b57914"]
    paths = []
    series = []
    for arm in ARMS:
        equity = np.r_[0, matrix[arm].to_numpy().cumsum()]
        series.append(equity - np.maximum.accumulate(equity) if drawdown else equity)
    lo, hi = min(map(np.min, series)), max(map(np.max, series))
    span = max(hi - lo, 1)
    for arm, color, values in zip(ARMS, colors, series):
        pts = " ".join(
            f"{35+900*i/(len(values)-1):.1f},{235-205*(v-lo)/span:.1f}"
            for i, v in enumerate(values)
        )
        paths.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="1.4" points="{pts}"><title>{arm}: {NAMES[arm]}</title></polyline>'
        )
    label = "Daily drawdown" if drawdown else "Cumulative net P&L"
    return (
        f'<svg viewBox="0 0 1000 270" role="img" aria-label="{label} in dollars by eligible-session index">'
        f'<text x="35" y="18">{label}: ${hi:,.0f} top / ${lo:,.0f} bottom</text>'
        + "".join(paths)
        + f'<text x="35" y="261">Start</text><text x="770" y="261">{len(matrix)} eligible sessions</text></svg>'
    )


def generate(path, daily, trades):
    scenarios = {}
    metric_rows, yearly = [], []
    for (arm, delay, cost), frame in daily.groupby(["arm", "delay", "cost"], sort=True):
        result = metrics(frame)
        scenarios[f"{arm}/delay{delay}/cost{cost}"] = result
        metric_rows.append(dict(arm=arm, delay=int(delay), cost=float(cost), **result))
        for year, g in frame.groupby(frame.day.str[:4]):
            yearly.append(
                dict(
                    arm=arm, delay=int(delay), cost=float(cost), year=year, **metrics(g)
                )
            )
    delay = CONFIG["delays"][0]
    cost = CONFIG["costs"][0]
    primary = {a: scenarios[f"{a}/delay{delay}/cost{cost}"] for a in ARMS}
    costly = {a: scenarios[f"{a}/delay{delay}/cost{CONFIG['costs'][-1]}"] for a in ARMS}
    matrix = paired_matrix(daily, delay=delay, cost=cost)
    uncertainty = {}
    for block in CONFIG["blocks"]:
        print(
            f"Bootstrap: {CONFIG['draws']} paired resamples, block {block}", flush=True
        )
        uncertainty[str(block)] = stationary_bootstrap(
            matrix.to_numpy(), block, draws=CONFIG["draws"], seed=CONFIG["seed"]
        )
    evaluation = classify(primary, costly, uncertainty)
    diagnostic = diagnostics(daily, trades)
    result = dict(
        historical_only=True,
        deployment_authorized=False,
        history_exposed=True,
        metrics=scenarios,
        primary=primary,
        uncertainty=uncertainty,
        evaluation=evaluation,
        conventions=dict(
            capital=CONFIG["capital"],
            risk_free=CONFIG["risk_free"],
            annualization=CONFIG["annualization"],
            std_ddof=1,
            daily_grid="Common eligible sessions, flat outcomes zero; unavailable dates excluded",
            bootstrap_draws=CONFIG["draws"],
            seed=CONFIG["seed"],
            blocks=CONFIG["blocks"],
            adjusted_confidence=CONFIG["adjusted_confidence"],
        ),
        thresholds=dict(
            minimum_sharpe=(
                primary["A"]["sharpe"] + CONFIG["minimum_delta_sharpe"]
                if primary["A"]["sharpe"] is not None
                else None
            ),
            maximum_drawdown=primary["A"]["max_drawdown"] * CONFIG["drawdown_ratio"],
            minimum_total_net=primary["A"]["total_net"] * CONFIG["profit_retention"],
        ),
    )
    write_json(path / "results.json", result)
    write_json(path / "diagnostics.json", diagnostic)
    for name, rows in [
        ("metrics.csv", metric_rows),
        ("yearly.csv", yearly),
        ("winner-retention.csv", diagnostic["winner_retention"]),
    ]:
        with open(path / name, "x") as out:
            pd.DataFrame(rows).to_csv(out, index=False)
    lines = [
        "# MIM-NB fixed entry-filter experiments",
        "",
        "Historical diagnosis only. Existing history was previously exposed. No deployment or sizing authorization.",
        "",
        "Recommendation: "
        + (
            "Consider separate prospective validation for "
            + ", ".join(evaluation["ranked_shortlist"])
            + "."
            if evaluation["ranked_shortlist"]
            else "No candidate qualifies for the historical shortlist. Retain the baseline while interpreting the diagnostics."
        ),
        "",
        "| Arm | Net daily Sharpe | Total net | Daily max drawdown | Profit retained | Classification |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for arm in ARMS:
        m = primary[arm]
        retained = (
            m["total_net"] / primary["A"]["total_net"]
            if primary["A"]["total_net"]
            else None
        )
        label = (
            "Baseline"
            if arm == "A"
            else evaluation["candidates"][arm]["classification"]
        )
        lines.append(
            f"| {arm} — {NAMES[arm]} | {number(m['sharpe'])} | ${m['total_net']:,.2f} | ${m['max_drawdown']:,.2f} | {number(retained)} | {label} |"
        )
    lines += [
        "",
        "Screen: delta Sharpe ≥0.20, drawdown ≤80% of baseline, net profit ≥75% of baseline. Highest-cost positive expectancy/delta Sharpe and no-worse drawdown are additionally required for shortlisting.",
        "",
        "## Paired Sharpe uncertainty",
        "",
        "| Arm | Block | 95% interval | Adjusted 98.75% interval | Undefined draws |",
        "|---|---:|---|---|---:|",
    ]
    for block, arms in uncertainty.items():
        for arm, u in arms.items():
            lines.append(
                f"| {arm} | {block} | {u['ci95']} | {u['ci98_75']} | {u['undefined_draws']} |"
            )
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Baseline compatibility preserves queued reversals: an accepted reversal can fill after a catastrophe stop during the modeled latency, unless the daily guard deactivates trading. This inherited behavior is not an execution improvement.",
        "- One contract, fixed $10,000 return denominator, zero risk-free return and sqrt(252) annualization. Sample daily standard deviation; flat eligible sessions included. Missing dates and unverified closures remain excluded.",
        "- Primary delay2 uses the open one full minute after decision close. Delay1 is optimistic only; costs are scenarios, not measured execution costs.",
        "- Adjusted intervals use four-comparison Bonferroni coverage. Stationary resampling is conditional on dependence assumptions; all three block lengths must support the shortlist.",
        "- Maximum drawdown is daily closing-equity drawdown, not an intraday capital guarantee. Unresolved drawdowns are censored in underwater-duration reporting.",
        "- Entry features and exit-attribution distributions are descriptive. Candidate thresholds were fixed before returns; there is no parameter search or combined filter.",
        "- The original 120-session A/B protocol remains unchanged. New candidate prospective validation would require a separate protocol.",
        "",
        "Detailed artifacts: metrics.csv, yearly.csv, winner-retention.csv, diagnostics.json, decisions.csv, features.csv, ledger.csv, trades.csv and paired-daily.csv.",
        "",
    ]
    with open(path / "report.md", "x") as out:
        out.write("\n".join(lines))
    with open(path / "diagnostics.md", "x") as out:
        out.write(
            "# Descriptive loss attribution\n\n"
            + diagnostic["limitation"]
            + "\n\n| Arm | Delay | Exit | Trades | Gross | Costs | Net |\n|---|---:|---|---:|---:|---:|---:|\n"
        )
        for d in diagnostic["exit_attribution"]:
            out.write(
                f"| {d['arm']} | {d['delay']} | {d['exit_reason']} | {d['trades']} | {d['gross']:.2f} | {d['costs']:.2f} | {d['net']:.2f} |\n"
            )
    primary_table = pd.DataFrame(
        [
            dict(
                arm=a,
                name=NAMES[a],
                **primary[a],
                classification=(
                    "baseline"
                    if a == "A"
                    else evaluation["candidates"][a]["classification"]
                ),
            )
            for a in ARMS
        ]
    )
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>MIM-NB experiments</title><style>body{font:16px system-ui;margin:3vw;color:#203040}table{border-collapse:collapse;font-size:14px}th,td{padding:8px;border-bottom:1px solid #ddd}section{overflow:auto;margin:24px 0}svg{width:100%;max-width:1000px}summary{cursor:pointer}pre{white-space:pre-wrap}</style><h1>MIM-NB: fixed entry-filter study</h1><p>Exposed historical evidence; no deployment authorization. A baseline · R regression · E efficiency · F reset · P persistence.</p>'
    html += (
        "<p>"
        + escape(lines[4])
        + "</p><section>"
        + primary_table.to_html(index=False, escape=True)
        + "</section>"
    )
    html += svg_paths(matrix) + svg_paths(matrix, True)
    html += (
        "<h2>Uncertainty and screen</h2><pre>"
        + escape(
            json.dumps(dict(evaluation=evaluation, uncertainty=uncertainty), indent=2)
        )
        + "</pre>"
    )
    html += (
        "<details><summary>All timing and cost scenarios</summary><section>"
        + pd.DataFrame(metric_rows).to_html(index=False, escape=True)
        + "</section></details>"
    )
    html += (
        "<details><summary>Yearly outcomes</summary><section>"
        + pd.DataFrame(yearly).to_html(index=False, escape=True)
        + "</section></details>"
    )
    html += (
        "<details><summary>Baseline large-winner participation</summary><section>"
        + pd.DataFrame(diagnostic["winner_retention"]).to_html(index=False, escape=True)
        + "</section></details>"
    )
    html += (
        "<h2>Conventions and limitations</h2><pre>"
        + escape("\n".join(lines[lines.index("## Interpretation limits") + 1 :]))
        + "</pre></html>"
    )
    with open(path / "report.html", "x") as out:
        out.write(html)
