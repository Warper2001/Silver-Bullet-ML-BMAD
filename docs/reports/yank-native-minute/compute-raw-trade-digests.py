"""Independent byte-matrix digest oracle; never uses structured record copies."""

from pathlib import Path
import hashlib, json, struct
import numpy as np
import zstandard as z

ROOT = Path("/root/Silver-Bullet-ML-BMAD/data/yank/databento-pilot-20260907")
M = 60_000_000_000
hashes = {"capture": {}, "exchange_diagnostic": {}}
counts = {"capture": {}, "exchange_diagnostic": {}}
for p in sorted(ROOT.glob("native/*/*.mbo.dbn.zst")):
    with p.open("rb") as f, z.ZstdDecompressor().stream_reader(f) as rd:
        head = rd.read(8)
        if head[:4] != b"DBN\x03":
            raise RuntimeError("DBN version")
        rd.read(struct.unpack("<I", head[4:])[0])
        while raw := rd.read(56 * 65536):
            if len(raw) % 56:
                raise RuntimeError("framing")
            matrix = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 56)
            selected = np.flatnonzero(
                (matrix[:, 38] == ord("T")) & ((matrix[:, 36] & 32) == 0)
            )
            for clock, offset in [("capture", 40), ("exchange_diagnostic", 8)]:
                times = np.ndarray(
                    shape=(len(matrix),),
                    dtype="<u8",
                    buffer=raw,
                    offset=offset,
                    strides=(56,),
                )[selected]
                minutes = times // M * M
                for minute in np.unique(minutes):
                    ix = selected[minutes == minute]
                    key = str(int(minute))
                    h = hashes[clock].setdefault(key, hashlib.sha256())
                    h.update(matrix[ix].tobytes(order="C"))
                    counts[clock][key] = counts[clock].get(key, 0) + len(ix)
    print(p.name, flush=True)
result = {
    clock: {
        k: {"sha256": h.hexdigest(), "count": counts[clock][k]}
        for k, h in values.items()
    }
    for clock, values in hashes.items()
}
Path("/tmp/yank-minute-raw-digests.json").write_text(
    json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
)
print({k: len(v) for k, v in result.items()}, flush=True)
