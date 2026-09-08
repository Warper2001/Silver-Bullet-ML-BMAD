"""Check historical writer compatibility using only pinned 2025 extract objects.

Local research utility; never authenticates the original acquisition or invocation.
"""

import contextlib, hashlib, io, json, os, pathlib, subprocess, tempfile, types, builtins

root = pathlib.Path("/root/Silver-Bullet-ML-BMAD-yank-minute")
rev = "d54b1a4a07a31dabe1f5bfd715793dbce9cfe4b8"
path = "scripts/generate_1min_dollar_bars_2025.py"
source = subprocess.check_output(["git", "show", rev + ":" + path], cwd=root)
extract = pathlib.Path(
    "/root/Silver-Bullet-ML-BMAD/_bmad-output/planning-artifacts/research/technical-yank-bar-provenance-and-pilot-evidence-g-2026-09-07/imports/provenance-raw-2025.jsonl"
)
if (
    hashlib.sha256(source).hexdigest()
    != "8f603a91acce9dfcec199b73e5f1ecb3f2d19904bbafeb0eb0bb018523c8f27e"
):
    raise ValueError("historical source pin")
with extract.open("rb") as stream:
    if (
        hashlib.file_digest(stream, "sha256").hexdigest()
        != "baeb1a060250c6c6071fe658459123604c85595df0688d0af278ecb3b6fa1b40"
    ):
        raise ValueError("2025 extract pin")
with extract.open() as stream:
    bars = [json.loads(line)["bar"] for line in stream]
if not all(b["TimeStamp"].startswith("2025-") for b in bars):
    raise ValueError("scope")
namespace = {"__name__": "closure_candidate"}
exec(compile(source, path, "exec"), namespace)
namespace["json"] = types.SimpleNamespace(load=lambda stream: bars)


def isolated_open(path, *args, **kwargs):
    if str(path) == "/root/mnq_historical.json":
        return io.StringIO("2025 extract injected; mixed-year raw never decoded")
    return builtins.open(path, *args, **kwargs)


namespace["open"] = isolated_open
with tempfile.TemporaryDirectory(prefix="yank-closure-candidate-") as tmp:
    os.chdir(tmp)
    with contextlib.redirect_stdout(io.StringIO()):
        code = namespace["main"]()
    output = pathlib.Path(
        "data/processed/dollar_bars/1_minute/mnq_1min_2025.csv"
    ).read_bytes()
expected = "3f20ec70885cdee6b48e6c5c7ed3254dd4cc8ce7bd8533696c5e461c75fb7822"
result = {
    "revision": rev,
    "path": path,
    "source_sha256": hashlib.sha256(source).hexdigest(),
    "input": "verified 2025 extract in original extract order; json.load replaced with these objects; temporary working directory; input open intercepted",
    "input_records": len(bars),
    "output_sha256": hashlib.sha256(output).hexdigest(),
    "expected_sha256": expected,
    "bytes": len(output),
    "matches_frozen_bytes": hashlib.sha256(output).hexdigest() == expected,
    "exit_code": code,
    "limitation": "Controlled compatibility check, not an original invocation or acquisition receipt.",
}
print(json.dumps(result, indent=2))

if code != 0 or not result["matches_frozen_bytes"]:
    raise ValueError("candidate differs from frozen CSV")
