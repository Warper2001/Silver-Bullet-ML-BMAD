"""Deterministic standalone Markdown and HTML reports."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def _bar_chart(title: str, labels: list[str], values: list[float], color: str) -> str:
    width, height, left, top = 760, 250, 150, 42
    maximum = max(values) if values else 1
    rows = []
    for index, (label, value) in enumerate(zip(labels, values)):
        y = top + index * 30
        bar = 540 * value / maximum if maximum else 0
        rows.append(
            f'<text x="8" y="{y + 15}" font-size="12">{escape(label)}</text>'
            f'<rect x="{left}" y="{y}" width="{bar:.2f}" height="18" fill="{color}"/>'
            f'<text x="{left + bar + 6:.2f}" y="{y + 14}" font-size="12">{value:.2f}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        f'<rect width="100%" height="100%" fill="white"/><text x="8" y="22" font-size="16" font-weight="bold">{escape(title)}</text>'
        + "".join(rows)
        + "</svg>"
    )


def _signed_bar_chart(title: str, labels: list[str], values: list[float]) -> str:
    width, height, zero = 760, 150, 390
    scale = 300 / max([abs(value) for value in values] or [1])
    rows = []
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 42 + index * 38
        bar = abs(value) * scale
        x = zero if value >= 0 else zero - bar
        color = "#2f855a" if value >= 0 else "#c53030"
        rows.append(
            f'<text x="8" y="{y + 15}" font-size="12">{escape(label)}</text>'
            f'<rect x="{x:.2f}" y="{y}" width="{bar:.2f}" height="18" fill="{color}"/>'
            f'<text x="{zero + 6 if value >= 0 else zero - 72}" y="{y + 15}" font-size="12">{value:.2f}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        f'<rect width="100%" height="100%" fill="white"/><text x="8" y="22" font-size="16" font-weight="bold">{escape(title)}</text>'
        f'<line x1="{zero}" y1="34" x2="{zero}" y2="130" stroke="#333"/>'
        + "".join(rows)
        + "</svg>"
    )


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    head = "| " + " | ".join(headers) + " |\n"
    rule = "| " + " | ".join("---" for _ in headers) + " |\n"
    body = "".join(
        "| " + " | ".join(str(value) for value in row) + " |\n" for row in rows
    )
    return head + rule + body


def _money(value: float) -> str:
    return f"-${abs(value):,.2f}" if value < 0 else f"${value:,.2f}"


def write_reports(
    path: Path,
    execution: dict[str, object],
    marks: dict[str, object],
    carry: dict[str, object],
) -> None:
    summary = marks["summary"]
    gate = execution["gate"]
    mark_frame: pd.DataFrame = marks["marks"]
    carry_matrix: pd.DataFrame = carry["matrix"]
    trips: pd.DataFrame = execution["round_trips"]
    mim_observed = trips[(trips.strategy == "MIM") & trips.signed_difference.notna()]
    gap_observed = trips[(trips.strategy == "GAP") & trips.signed_difference.notna()]
    signed_labels = [
        f"MIM exact ({len(mim_observed)}/25 local trades)",
        f"GAP gross-only ({len(gap_observed)}/21 fill rows)",
    ]
    signed_values = [
        float(mim_observed.signed_difference.sum()),
        float(gap_observed.signed_difference.sum()),
    ]
    by_hour = (
        mark_frame.assign(hour=mark_frame.mark_timestamp.str[11:13])
        .groupby("hour")
        .size()
    )
    by_outcome = mark_frame.groupby("final_outcome").agg(
        median_giveback=("giveback", "median"), marks=("mark_id", "size")
    )
    distributions: pd.DataFrame = marks["distributions"]
    giveback = distributions[
        (distributions.group_dimension == "final_outcome")
        & (distributions.metric == "giveback")
    ].set_index("group_value")
    charts = [
        _signed_bar_chart(
            "Observed fill minus modeled P&L ($); incomplete coverage",
            signed_labels,
            signed_values,
        ),
        _bar_chart(
            "Active scheduled marks by ET hour",
            list(by_hour.index),
            [float(x) for x in by_hour],
            "#3568a8",
        ),
        _bar_chart(
            "Median observed giveback by final outcome ($)",
            list(by_outcome.index),
            [float(x) for x in by_outcome.median_giveback],
            "#a85b35",
        ),
        _bar_chart(
            "Parked carry rows by venue",
            list(carry_matrix.groupby("venue").size().index),
            [float(x) for x in carry_matrix.groupby("venue").size()],
            "#5b7d3a",
        ),
    ]
    metrics = [
        ["Closed-trade net PF", f'{summary["closed_trade_net_pf"]:.8f}'],
        ["Net profit", f'${summary["net_profit"]:,.2f}'],
        ["Trades / sessions", f'{summary["trades"]} / {summary["sessions"]}'],
        [
            "Exposure contract-minutes",
            f'{summary["exposure_contract_minutes_lower"]:,.0f}--{summary["exposure_contract_minutes_upper"]:,.0f}',
        ],
        ["Daily max drawdown", f'${summary["daily_max_drawdown"]:,.2f}'],
        [
            "Top 5% trade net / total net",
            f'{summary["top_5pct_trade_fraction_of_net"]:.2f}x',
        ],
        ["Top 67 day net / total net", f'{summary["top_67_day_fraction_of_net"]:.2f}x'],
    ]
    markdown = f"""# Portfolio PF improvement shortlist execution

## Stage 1 — execution reconciliation

**Gate: `{gate['verdict']}`.** Later feasibility stages were allowed because no current repair candidate was proven. Exact order IDs were used before any descriptive matching. Broker net appears only for complete two-leg MIM saved-export round trips with explicit fees and commissions. GAP remains gross-only because timestamps, executable quotes and complete costs are absent.

{chr(10).join('- ' + reason for reason in gate['reasons'])}

Observed signed fill-minus-model differences are **{_money(signed_values[0])} for five exact MIM pairs** and **{_money(signed_values[1])} for 21 GAP gross-only rows**. Neither is a recoverable-loss estimate; MIM coverage is partial and GAP lacks complete costs and causal timestamps.

## Stage 2 — MIM profit-giveback feasibility

**Verdict: `{marks['verdict']['verdict']}`.** The hypothesis is distinct from the prior exit rules and has complete descriptive coverage, but this does not establish predictive separation or PF improvement. At observed marks, eventual losers had median giveback **${giveback.loc['LOSS', 'q050']:,.2f}** versus **${giveback.loc['WIN', 'q050']:,.2f}** for eventual winners. Their fixed 10th--90th percentile ranges overlap: **${giveback.loc['LOSS', 'q010']:,.2f}--${giveback.loc['LOSS', 'q090']:,.2f}** for losers and **${giveback.loc['WIN', 'q010']:,.2f}--${giveback.loc['WIN', 'q090']:,.2f}** for winners. These groups use retrospective labels.

## Unchanged MIM baseline

{_markdown_table(['Metric', 'Value'], metrics)}

The 6,673-row scheduled-mark ledger contains only completed information available at the existing 10:00--15:30 ET marks. It includes current causal P&L, running MFE/MAE, giveback, prior-mark change and original retrospective outcome labels. The seven reversal marks attach to the exiting leg. It calculates no candidate exit return, threshold, window or classifier.

Exit labels remain 71 `CAT_STOP`, 723 `EOD_CLOSE_PROXY`, and seven `REVERSAL`.

## Stage 3 — commodity carry feasibility

The 12-root by 3-venue matrix contains 36 rows. Overall verdicts are `{', '.join(carry['verdict']['overall'])}`. Topstep is unavailable for the proposed monthly overnight book; TradeStation SIM is validation-only; a future self-funded path remains requirements-only. Account, data and power reasons coexist. Prepared source/account questions remain unsent, and no alternative-strategy return was calculated.

Official-source observations are dated in `carry_evidence.csv`: [Topstep trading hours](https://help.topstep.com/en/articles/8284206-when-and-what-products-can-i-trade), [TradeStation futures margins](https://www.tradestation.com/pricing/futures-margin-requirements/), and [TradeStation physical-delivery policy](https://uploads.tradestation.com/uploads/Futures-Physical-Delivery.pdf).

Prior verdicts remain TSC-1/TSMOM-1/COT `FAIL` and XSMOM-1/VRP-1 `UNDERPOWERED`.

## Portfolio decay monitor boundary

The monitor file's observation timestamp and row coverage are inventoried in `decay_monitor_coverage.json`. Its efficacy is not interpreted; the monitor script and log were not called or edited.

## Artifacts

`execution_events.csv`, `execution_round_trips.csv`, `execution_coverage.csv`, `mim_trades.csv`, `mim_decision_marks.csv`, `mim_lineage_audit.csv`, `mim_path_distributions.csv`, `mim_path_verdict.json`, `mim_hypothesis_specification.md`, `carry_matrix.csv`, `carry_evidence.csv`, `carry_questions.md`, frozen source snapshots and completion hashes are included in this sealed run.
"""
    (path / "report.md").write_text(markdown, encoding="utf-8")
    html_metrics = "".join(
        f"<tr><th>{escape(a)}</th><td>{escape(b)}</td></tr>" for a, b in metrics
    )
    html_reasons = "".join(f"<li>{escape(reason)}</li>" for reason in gate["reasons"])
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Portfolio PF shortlist</title><style>
body{{font-family:system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#17202a}}table{{border-collapse:collapse}}th,td{{border:1px solid #ccd1d1;padding:.45rem;text-align:left}}code{{background:#eef2f3;padding:.1rem .25rem}}svg{{display:block;max-width:100%;margin:1rem 0;border:1px solid #ddd}}
</style></head><body><h1>Portfolio PF improvement shortlist execution</h1>
<h2>Stage 1 — execution reconciliation</h2><p><strong>Gate: <code>{escape(gate['verdict'])}</code>.</strong> No current repair candidate was proven.</p><ul>{html_reasons}</ul>
{charts[0]}<p>MIM exact signed difference: {_money(signed_values[0])}; GAP gross-only signed difference: {_money(signed_values[1])}. Coverage is incomplete and these are not recoverable-loss estimates.</p>
<h2>Stage 2 — MIM profit-giveback feasibility</h2><p><strong>{escape(marks['verdict']['verdict'])}</strong>; no predictive or PF claim.</p>
<h2>Unchanged MIM baseline</h2><table>{html_metrics}</table>{charts[1]}{charts[2]}
<p>The 6,673 scheduled marks use completed information only. Seven reversal marks belong to the exiting leg. No candidate return, threshold, window or classifier was calculated.</p>
<h2>Commodity carry feasibility</h2><p>{escape(', '.join(carry['verdict']['overall']))}; 36 root/venue rows, with prepared questions unsent. Official evidence: <a href="https://help.topstep.com/en/articles/8284206-when-and-what-products-can-i-trade">Topstep trading hours</a>, <a href="https://www.tradestation.com/pricing/futures-margin-requirements/">TradeStation margins</a>, and <a href="https://uploads.tradestation.com/uploads/Futures-Physical-Delivery.pdf">TradeStation delivery policy</a>.</p>{charts[3]}
<h2>Evidence boundary</h2><p>All source hashes, runtime details, normalized ledgers and reports are contained in this run. The portfolio decay monitor was observed for timestamp and coverage only.</p>
</body></html>"""
    (path / "report.html").write_text(html, encoding="utf-8")
