from pathlib import Path
import struct, json
import numpy as np
import zstandard as z

ROOT = Path("/root/Silver-Bullet-ML-BMAD/data/yank/databento-pilot-20260907")
D = np.dtype(
    {
        "names": ["length", "rtype", "id", "event", "flags", "action", "recv"],
        "formats": ["u1", "u1", "<u4", "<u8", "u1", "u1", "<u8"],
        "offsets": [0, 1, 4, 8, 36, 38, 40],
        "itemsize": 56,
    }
)
M = 60_000_000_000
out = {
    "records": 0,
    "trades": 0,
    "snapshot_trades": 0,
    "bad_flag_trades": 0,
    "trade_event_crosses_minute": 0,
    "unterminated_trades": 0,
    "examples": [],
    "daily": {},
}
for p in sorted(ROOT.glob("native/*/*.mbo.dbn.zst")):
    n = 0
    pending = []
    nt = 0
    with p.open("rb") as f, z.ZstdDecompressor().stream_reader(f) as rd:
        head = rd.read(8)
        if head[:4] != b"DBN\x03":
            raise RuntimeError(head)
        meta_len = struct.unpack("<I", head[4:])[0]
        if len(rd.read(meta_len)) != meta_len:
            raise RuntimeError("metadata")
        while chunk := rd.read(56 * 65536):
            if len(chunk) % 56:
                raise RuntimeError("framing")
            a = np.frombuffer(chunk, dtype=D)
            if not np.all(
                (a["length"] == 14) & (a["rtype"] == 160) & (a["id"] == 42009475)
            ):
                raise RuntimeError("schema")
            trades = np.flatnonzero(a["action"] == ord("T"))
            nt += len(trades)
            out["snapshot_trades"] += int(np.count_nonzero(a["flags"][trades] & 32))
            out["bad_flag_trades"] += int(np.count_nonzero(a["flags"][trades] & 8))
            ends = np.flatnonzero(a["flags"] & 128)
            if len(ends) and pending:
                e = int(a["recv"][ends[0]])
                for idx, t in pending:
                    if e // M > t // M:
                        out["trade_event_crosses_minute"] += 1
                        if len(out["examples"]) < 10:
                            out["examples"].append(
                                [p.name, idx, t, n + int(ends[0]), e]
                            )
                pending = []
            positions = np.searchsorted(ends, trades)
            good = positions < len(ends)
            if np.any(good):
                ti = trades[good]
                ei = ends[positions[good]]
                crossed = a["recv"][ei] // M > a["recv"][ti] // M
                out["trade_event_crosses_minute"] += int(crossed.sum())
                for t, e in zip(
                    ti[crossed][: max(0, 10 - len(out["examples"]))], ei[crossed]
                ):
                    out["examples"].append(
                        [
                            p.name,
                            n + int(t),
                            int(a["recv"][t]),
                            n + int(e),
                            int(a["recv"][e]),
                        ]
                    )
            pending.extend((n + int(t), int(a["recv"][t])) for t in trades[~good])
            n += len(a)
    out["unterminated_trades"] += len(pending)
    out["records"] += n
    out["trades"] += nt
    out["daily"][p.name] = {"records": n, "trades": nt}
    print(p.name, n, nt, flush=True)
Path("/tmp/yank-minute-event-probe.json").write_text(
    json.dumps(out, sort_keys=True, indent=2) + "\n"
)
print(json.dumps({k: v for k, v in out.items() if k != "daily"}), flush=True)
