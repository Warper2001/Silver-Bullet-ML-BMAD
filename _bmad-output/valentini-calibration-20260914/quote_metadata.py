"""One-off free metadata probe; never submits a data request or purchase."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import requests
from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
PLAN = json.loads((HERE / "quote-plan.json").read_text())
KEY = dotenv_values("/root/Silver-Bullet-ML-BMAD/.env").get("DATABENTO_API_KEY")
if not KEY:
    raise SystemExit("Existing Databento credential unavailable")
SESSION = requests.Session()
SESSION.trust_env = False
SESSION.auth = (KEY, "")
BASE = "https://hist.databento.com/v0/"
PARAMS = {"dataset": PLAN["dataset"], "symbols": PLAN["symbol"],
          "stype_in": PLAN["stype_in"], "start": PLAN["start"], "end": PLAN["end"]}

def probe(endpoint, params, index):
    assert endpoint in PLAN["allowed_endpoints"]
    path = HERE / f"metadata-{index:02d}.json"
    assert not path.exists()
    record = {"endpoint": BASE + endpoint, "method": "GET", "params": params,
              "requested_at": datetime.now(timezone.utc).isoformat()}
    try:
        with SESSION.get(BASE + endpoint, params=params, timeout=(10, 20),
                         allow_redirects=False, stream=True) as response:
            record["http_status"] = response.status_code
            if response.status_code != 200:
                record["error"] = "HTTP_REFUSAL_BODY_NOT_RETAINED"
            else:
                chunks = bytearray()
                for chunk in response.iter_content(4096):
                    chunks.extend(chunk)
                    if len(chunks) > 65536:
                        raise ValueError("Oversized metadata")
                value = json.loads(chunks)
                if endpoint.startswith("metadata."):
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError("Unexpected numeric metadata")
                else:
                    if not isinstance(value, dict) or "result" not in value:
                        raise ValueError("Unexpected symbology metadata")
                    value = {k: value[k] for k in ("result", "partial", "not_found") if k in value}
                record["response"] = value
                record["raw_response_sha256"] = hashlib.sha256(chunks).hexdigest()
    except Exception as exc:
        record["error"] = type(exc).__name__
    record["received_at"] = datetime.now(timezone.utc).isoformat()
    with path.open("x") as output:
        json.dump(record, output, indent=2, sort_keys=True)
        output.write("\n")
    print(json.dumps({k: record[k] for k in ("endpoint", "http_status", "response", "error") if k in record}), flush=True)
    if "error" in record:
        raise SystemExit("Metadata probe stopped; no data acquisition attempted")

index = 0
for schema in PLAN["schemas"]:
    for endpoint in ("metadata.get_cost", "metadata.get_record_count", "metadata.get_billable_size"):
        probe(endpoint, dict(PARAMS, schema=schema), index)
        index += 1
probe("symbology.resolve", dict(dataset=PLAN["dataset"], symbols=PLAN["symbol"],
      stype_in="raw_symbol", stype_out="instrument_id", start_date=PLAN["start"][:10],
      end_date=PLAN["end"][:10]), index)
SESSION.close()
