"""Recompute the published ledger and entry timing without importing replay code."""

import json
import sys
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def number(value):
    result = Decimal(str(value))
    require(result.is_finite(), "nonfinite accounting value")
    return result


def check(path):
    result = json.loads(Path(path).read_text())
    require(result["outcome"] == "REPLAY_COMPLETE", "replay unavailable")
    require(set(result["arms"]) == {"no-ml", "ml050"}, "missing arm")
    bars = [
        json.loads(line)
        for line in Path(path).with_name("bars.jsonl").read_text().splitlines()
        if line
    ]
    require(bool(bars), "missing bars")
    by_time, prefix, indices = {}, {}, {}
    watermark = 0
    previous = -1
    for i, bar in enumerate(bars):
        start, end = bar["start_ns"], bar["end_ns"]
        require(
            start > previous
            and start % 60_000_000_000 == 0
            and end == start + 60_000_000_000,
            "bar interval identity",
        )
        require(
            bar["availability_ns"] >= end and bar["incomplete_event"] is False,
            "bar completion availability",
        )
        stamp = datetime.fromtimestamp(start // 1_000_000_000, timezone.utc).isoformat()
        watermark = max(watermark, bar["availability_ns"])
        by_time[stamp], prefix[stamp], indices[stamp] = bar, watermark, i
        previous = start

    def timing(row, gate=False):
        require(row["timestamp"] in by_time, "unknown event/gate bar")
        bar = by_time[row["timestamp"]]
        require(
            row["interval_start_ns"] == bar["start_ns"]
            and row["interval_end_ns"] == bar["end_ns"],
            "event/gate interval identity",
        )
        require(
            row["bar_available_ns"] == bar["availability_ns"],
            "event/gate bar availability",
        )
        if gate:
            require(
                row["decision_available_ns"] == prefix[row["timestamp"]],
                "gate prefix availability",
            )

    verified = {}
    for name, arm in result["arms"].items():
        cash, costs, realized = number(50000), number(0), number(0)
        positions, exits, fills = {}, {}, {}
        gates = {g["timestamp"]: g for g in arm["gates"]}
        require(
            len(gates) == len(arm["gates"]) == len(bars)
            and list(gates) == list(by_time),
            "gate coverage",
        )
        for gate in gates.values():
            timing(gate, True)
            idx = indices[gate["timestamp"]]
            if gate["outcome"] == "ORDER" and idx + 1 < len(bars):
                require(
                    prefix[gate["timestamp"]] <= bars[idx + 1]["start_ns"],
                    "order before next interval availability",
                )
        fill_indices, mark_indices = {}, {}
        marks = 0
        previous_index = -1
        for sequence, e in enumerate(arm["events"]):
            timing(e)
            require(e["sequence"] == sequence, "event sequence")
            index = indices[e["timestamp"]]
            require(index >= previous_index, "event bar ordering")
            previous_index = index
            if e["kind"] == "FILL":
                oid = e["order_id"]
                require(oid not in positions and oid not in exits, "duplicate fill")
                qty = number(e["quantity"])
                require(
                    qty < 0 and qty == qty.to_integral_value(), "short integer quantity"
                )
                price = number(e["price"])
                require(number(e["cost"]) == 0, "unexpected modeled entry fee")
                signal = gates[e["signal_time"]]
                require(signal["outcome"] == "ORDER", "fill has no order")
                require(
                    signal["decision_available_ns"] <= e["interval_start_ns"],
                    "fill before causal availability",
                )
                require(
                    signal["interval_start_ns"] < e["interval_start_ns"],
                    "same-signal-bar fill",
                )
                require(not positions, "overlapping positions")
                fills[oid] = e
                fill_indices[oid], mark_indices[oid] = index, []
                positions[oid] = (qty, price)
                cash -= qty * price * 2
            elif e["kind"] == "MARK":
                require(e["order_id"] in positions, "mark without fill")
                qty, entry = positions[e["order_id"]]
                require(number(e["quantity"]) == qty, "mark quantity")
                require(
                    number(e["unrealized"]) == -qty * (entry - number(e["price"])) * 2,
                    "mark unrealized",
                )
                require(
                    number(e["price"])
                    == number(by_time[e["timestamp"]]["ohlcv"][3]) / 1_000_000_000,
                    "mark close identity",
                )
                mark_indices[e["order_id"]].append(index)
                require(
                    mark_indices[e["order_id"]]
                    == list(range(fill_indices[e["order_id"]], index + 1)),
                    "missing or duplicate mark coverage",
                )
                marks += 1
            elif e["kind"] == "EXIT":
                require(e["order_id"] in positions, "exit without fill")
                require(
                    mark_indices[e["order_id"]]
                    == list(range(fill_indices[e["order_id"]], index + 1)),
                    "missing exit mark coverage",
                )
                require(
                    e["bars_held"] == index - fill_indices[e["order_id"]],
                    "exit observed holding age",
                )
                fill = fills[e["order_id"]]
                require(
                    e["fill_time"] == fill["timestamp"]
                    and e["signal_time"] == fill["signal_time"],
                    "exit fill/signal identity",
                )
                qty, entry = positions.pop(e["order_id"])
                require(number(e["contracts"]) == -qty, "exit quantity")
                require(number(e["entry_price"]) == entry, "exit entry price")
                fee = number(e["cost"])
                require(fee == 4, "frozen modeled fee changed")
                price = number(e["exit_price"])
                pnl = -qty * (entry - price) * 2 - fee
                require(
                    number(e["economic_pnl"]) == pnl == number(e["pnl"]), "exit PNL"
                )
                cash += qty * price * 2 - fee
                costs += fee
                realized += pnl
                exits[e["order_id"]] = e
        trades = {t["order_id"]: t for t in arm["trades"]}
        require(
            len(trades) == len(arm["trades"]) and set(trades) == set(exits),
            "published trade identity",
        )
        for oid, t in trades.items():
            e = exits[oid]
            for key in (
                "contracts",
                "entry_price",
                "exit_price",
                "economic_pnl",
                "pnl",
                "cost",
            ):
                require(
                    number(t[key]) == number(e[key]),
                    "published trade monetary field " + key,
                )
            for key in ("signal_time", "fill_time", "ambiguity"):
                require(t[key] == e[key], "published trade metadata " + key)
            require(t["exit_time"] == e["timestamp"], "published exit time")
            for key in ("signal_time", "fill_time", "exit_time"):
                raw = by_time[t[key]]
                require(
                    t["timing"][key]
                    == dict(
                        interval_start_ns=raw["start_ns"],
                        interval_end_ns=raw["end_ns"],
                        bar_available_ns=raw["availability_ns"],
                        decision_available_ns=prefix[t[key]],
                    ),
                    "trade interval/availability identity",
                )
        terminal = arm["terminal"]
        last_stamp = list(by_time)[-1]
        require(
            terminal["mark_time"] == last_stamp
            and terminal["mark_interval_end_ns"] == bars[-1]["end_ns"],
            "terminal interval identity",
        )
        require(
            terminal["mark_available_ns"] == prefix[last_stamp],
            "terminal prefix availability",
        )
        require(
            number(terminal["mark_price"])
            == number(bars[-1]["ohlcv"][3]) / 1_000_000_000,
            "terminal close identity",
        )
        for oid in positions:
            require(
                mark_indices[oid] == list(range(fill_indices[oid], len(bars))),
                "missing terminal mark coverage",
            )
            require(
                terminal["status"] == "OPEN"
                and terminal["order_id"] == oid
                and terminal["bars_held"] == len(bars) - 1 - fill_indices[oid],
                "terminal open identity",
            )
        qty = sum((q for q, entry in positions.values()), number(0))
        mark = number(terminal["mark_price"])
        inventory = qty * mark * 2
        unrealized = sum(
            (-q * (entry - mark) * 2 for q, entry in positions.values()), number(0)
        )
        equity = cash + inventory
        require(number(terminal["quantity"]) == qty, "terminal quantity")
        require(number(terminal["inventory_value"]) == inventory, "terminal inventory")
        require(number(terminal["unrealized"]) == unrealized, "terminal unrealized")
        require(equity == 50000 + realized + unrealized, "equity identity")
        calculated = dict(
            cash=cash, costs=costs, economic_realized=realized, equity=equity
        )
        for key, value in calculated.items():
            require(number(arm["summary"][key]) == value, "summary " + key)
            require(
                number(arm["independent_reconciliation"][key]) == value,
                "saved reconciliation " + key,
            )
        require(arm["summary"]["trades"] == len(trades), "summary trade count")
        verified[name] = {
            **{k: str(v) for k, v in calculated.items()},
            "trades": len(trades),
            "marks_checked": marks,
            "entry_timing": "PASS",
        }
    return {
        "status": "PASS_PUBLISHED_LEDGER",
        "arms": verified,
        "fee_qualification": "4 USD per closed trade is a modeled assumption, not a verified broker charge",
    }


if __name__ == "__main__":
    print(json.dumps(check(sys.argv[1]), sort_keys=True, indent=2))
