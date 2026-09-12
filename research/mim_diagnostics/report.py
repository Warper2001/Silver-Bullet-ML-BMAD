"""Self-contained readable reports with inline SVG, no external dependencies."""

import html
import re
import numpy as np
from .feasibility import render_inventory

PROPOSALS = """# Research specifications for future simpler comparisons

These are definitions and falsifiable proposals only. No return is computed, no candidate is promoted and no trade is rescored. Windows below are session/instrument conventions, not selected from attribution. The historical best 67 and profitable half-hours are not candidate definitions.

1. **Cash-session exposure reference.** Define one unit of long MNQ exposure from the cash-session opening modeled fill to the cash-session close proxy on every eligible session. Retain the frozen contract/session selection, one-contract point value and explicit execution costs. Prerequisites: committed preregistration, enough untouched future sessions under a paired power gate, tradable opening/closing timestamp definitions and unavailable-fill treatment. Falsifier: fail to improve the preregistered risk-adjusted objective relative to frozen A under the planned paired uncertainty procedure. This definition is not a backtest instruction.
2. **Once-per-session baseline entry.** Prospectively define retaining only the first qualifying entry under the unchanged frozen baseline decision rule, with its already-defined protective exit and cash-session close, without later entries. Prerequisites: justify this hypothesis independently of the displayed first/subsequent attribution, power gate and preregistration before new data, explicit paired estimand and cost assumptions. Falsifier: fail the preregistered improvement criterion on previously unseen sessions; ambiguous evidence is FAIL. No such returns are produced here.
3. **Scheduled-information demand hypothesis.** Specify whether genuinely preannounced Nasdaq-relevant information changes the timing of baseline payoff using contemporaneously available schedule vintages. Prerequisites: repair incorrect economic dates, retain original publication/receipt evidence and reschedules, predefine event overlap handling and enough independent events. Falsifier: no stable out-of-sample incremental explanation beyond the preregistered controls. No policy-shock throttle revival; no resweep of the failed impulse-following branch. FOMC fade remains governed exclusively by its existing sealed protocol.
4. **Closing-demand measurement hypothesis.** Specify a measurement model linking observed Nasdaq options exposure and leveraged ETF rebalance demand to closing prices, with explicit signed inventory assumptions, leverage/AUM vintage, lag and sensitivity uncertainty. Prerequisites: acquire eligible archives with point-in-time publication evidence and Nasdaq basis; separate observed positions from inferred proxies. Falsifier: claimed demand direction or explanatory increment fails on independent observations under preregistered measurement checks. SPX inventory and price-derived delta cannot substitute silently.

No numerical hurdle is invented here. A future test's decision rule and suitable power gate must be preregistered before its independent observations; these proposals grant no permission to access sealed holdout data, change parameters, collect datasets or alter prospective collection.
"""


def table(frame):
    cols = list(frame)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| "
            + " | ".join(f"{x:,.2f}" if isinstance(x, float) else str(x) for x in row)
            + " |"
        )
    return "\n".join(lines)


def svg_bar(frame, title):
    values = frame.net.tolist()
    labels = frame["group"].tolist()
    scale = max([abs(x) for x in values] + [1])
    width = 840
    height = 50 + len(values) * 28
    bars = []
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 35 + i * 28
        length = abs(value) / scale * 255
        x = 490 - length if value < 0 else 490
        bars.append(
            f'<text x="5" y="{y+14}" font-size="12">{html.escape(str(label))}</text><rect x="{x}" y="{y}" width="{length}" height="18" fill="{"#ad3939" if value<0 else "#187469"}"/><text x="760" y="{y+14}" font-size="11">{value:,.2f}</text>'
        )
    return (
        f'<svg role="img" aria-label="{html.escape(title)}" viewBox="0 0 {width} {height}"><title>{html.escape(title)}</title><text x="5" y="18">{html.escape(title)}</text>'
        + "".join(bars)
        + "</svg>"
    )


def write_reports(path, result, feasibility):
    s = result["summary"]
    parts = result["partitions"]
    entry = parts[
        (parts.basis == "entry_cohort") & (parts.dimension == "entry_half_hour")
    ]
    accrual = parts[
        (parts.basis == "minute_accrual") & (parts.dimension == "accrual_half_hour")
    ]
    exits = parts[(parts.basis == "entry_cohort") & (parts.dimension == "exit_reason")]
    text = f"""# MIM-NB frozen baseline payoff diagnostics

Exposed development evidence only. Primary arm A, recorded delay 2, one MNQ, $2/point and $2.24 roundtrip. Entry at the open one full minute after decision. No strategy simulation or new return test.

{ s['sessions']:,} sessions, {s['trades']} trades, {s['events']} execution events and {s['flat_sessions']} flat sessions reconcile to gross ${s['gross']:,.2f}, costs ${s['costs']:,.2f}, net **${s['net']:,.2f}**. Best 67 sessions contributed ${s['best_67_net']:,.2f}; the remainder lost ${-s['remainder_net']:,.2f}. This is hindsight concentration, not a selection rule.

The sampled minute-close maximum drawdown is ${s['drawdown_minute']['depth']:,.2f}; sampled daily-close maximum drawdown is ${s['drawdown_daily']['depth']:,.2f}. Longest underwater durations are {s['drawdown_minute']['longest_underwater_samples']:,} minute samples and {s['drawdown_daily']['longest_underwater_samples']} daily samples. Full peak/trough/recovery labels and unresolved episodes are in summary.json. These clocks omit overnight/non-session minutes; unobserved intraminute portfolio extremes are unknown.

Exposure is bounded by {s['exposure_minutes_lower']:,.0f}–{s['exposure_minutes_upper']:,.0f} contract-minutes; {s['uncertain_stop_trades']} stops have uncertain intraminute time and incomplete excursion coverage. Source execution labels are retained; actual transitions: {s['transitions']}.

## Entry time answers when trades began

{table(entry[['group','count','gross','costs','net','losing_trade_dollars','losing_trade_share','exposure_minutes_upper']])}

## P&L accrual answers when marked payoff changed

{table(accrual[['group','count','gross','costs','net','exposure_minutes_upper']])}

Counts in accrual partitions count entries in each bucket, rather than trades touching it. Costs belong to execution event labels. Actual modeled fill timestamps remain in events.csv and trades.csv. Signed eventual-loser contributions in accrual.csv describe actual minute changes of trades that eventually lost; favorable periods can have positive contributions.

## Original exit labels

{table(exits[['group','count','gross','costs','net','losing_trade_dollars','losing_trade_share']])}

Every complete direction/year/entry-order/entry-half-hour/exit-label partition is in partitions.csv. Event ledger realized P&L, close-marked equity, position and exposure are in events.csv/minute.csv. trades.csv contains close/fill sampled and fully-held range excursions. All source exclusions are copied unchanged.

## Already recorded timing/cost alternatives

{table(result['scenarios'])}

These rows display source arm A scenarios separately; they are not new benchmark results. No filter, restriction, threshold or strategy parameter is selected here. The primary conclusion is accounting and concentration, not evidence that a new candidate works.

""" + render_inventory(feasibility) + "\n" + PROPOSALS
    (path / "report.md").write_text(text)
    body = "".join(
        svg_bar(f, title)
        for f, title in (
            (entry, "Entry half-hour net payoff"),
            (accrual, "Accrual half-hour net payoff"),
            (exits, "Original exit-label net payoff"),
        )
    )
    for dimension in ("direction", "year", "entry_order"):
        group = parts[(parts.basis == "entry_cohort") & (parts.dimension == dimension)]
        text += (
            "\n## "
            + dimension
            + " payoff\n\n"
            + table(
                group[
                    [
                        "group",
                        "count",
                        "gross",
                        "net",
                        "losing_trade_dollars",
                        "losing_trade_share",
                        "exposure_minutes_upper",
                    ]
                ]
            )
            + "\n"
        )
    text += "\nEntry-time and accrual totals describe different clocks: trades entered early can accumulate most of their payoff much later. EOD exits include both winners and losing trades; their positive aggregate does not imply that all EOD trades win. Neither view selects a future trading window.\n"
    (path / "report.md").write_text(text)
    body += svg_curve(
        result["minute"].marked_equity_net, "Sampled minute-close net equity"
    )
    equity = result["minute"].marked_equity_net.to_numpy()
    body += svg_curve(
        equity - np.maximum.accumulate(np.r_[0.0, equity])[1:],
        "Sampled minute-close drawdown",
    )
    losses = exits.copy()
    losses["net"] = losses.losing_trade_dollars
    body += svg_bar(losses, "Losing-trade dollars by original exit label")
    body += markdown_html(text)
    (path / "report.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MIM-NB payoff diagnostics</title><style>body{max-width:1100px;margin:2rem auto;padding:1rem;color:#183039;background:#f7faf9;font:16px system-ui}svg{display:block;width:100%;background:white;margin:1rem 0}p,li{line-height:1.6}table{border-collapse:collapse;display:block;overflow:auto;font-size:13px}th,td{padding:6px;border:1px solid #ccd}a{color:#126}code{overflow-wrap:anywhere}</style><h1>MIM-NB payoff diagnostics</h1>'
        + body
        + "</html>"
    )


def svg_curve(values, title):
    values = np.asarray(values, float)
    # Each consecutive block retains both extrema in original order, plus endpoints.
    indices = {0, len(values) - 1}
    for start in range(0, len(values), max(1, len(values) // 1200)):
        block = values[start : start + max(1, len(values) // 1200)]
        indices.update((start + int(np.argmin(block)), start + int(np.argmax(block))))
    low, high = min(0.0, float(values.min())), max(0.0, float(values.max()))
    span = max(high - low, 1.0)
    points = " ".join(
        f"{60+i/max(len(values)-1,1)*930:.2f},{40+(high-values[i])/span*240:.2f}"
        for i in sorted(indices)
    )
    return f'<svg role="img" aria-label="{html.escape(title)}" viewBox="0 0 1060 325"><title>{html.escape(title)}</title><text x="10" y="20">{html.escape(title)}</text><text x="5" y="45">${high:,.0f}</text><text x="5" y="280">${low:,.0f}</text><polyline points="{points}" fill="none" stroke="#187469" stroke-width="1"/><text x="60" y="310">All {len(values):,} eligible samples; consecutive-block extrema retained for display</text></svg>'


def markdown_html(text):
    def inline(raw):
        value = html.escape(raw)
        value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', value)
        value = re.sub(r"`([^`]+)`", r"<code>\1</code>", value)
        return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", value)

    lines = text.splitlines()
    output = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("| "):
            group = []
            while i < len(lines) and lines[i].startswith("| "):
                group.append(lines[i])
                i += 1
            output.append("<table>")
            for n, row in enumerate(group):
                if n == 1:
                    continue
                tag = "th" if n == 0 else "td"
                output.append(
                    "<tr>"
                    + "".join(
                        f"<{tag}>" + inline(c.strip()) + f"</{tag}>"
                        for c in row.strip("|").split("|")
                    )
                    + "</tr>"
                )
            output.append("</table>")
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            output.append(f"<h{level}>" + inline(line[level:].strip()) + f"</h{level}>")
        elif line.startswith("- "):
            output.append("<p>• " + inline(line[2:]) + "</p>")
        else:
            output.append("<p>" + inline(line) + "</p>")
        i += 1
    return "".join(output)
