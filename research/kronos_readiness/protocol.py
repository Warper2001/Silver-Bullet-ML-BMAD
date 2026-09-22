"""Frozen documentary candidate protocol; no executable strategy."""

from pathlib import Path
from typing import Any
from . import FLAGS
from .evidence import sha
from tools import kronos_inference_pilot as pilot


def candidate() -> dict[str, Any]:
    engine = Path(__file__).resolve().parents[1] / "kronos_replay/engine.py"
    return {
        **FLAGS,
        "status": "CANDIDATE_NOT_ADMITTED",
        "model_revision": pilot.MODEL_REVISION,
        "tokenizer_revision": pilot.TOKENIZER_REVISION,
        "source_revision": pilot.SOURCE_REVISION,
        "engine_sha256": sha(engine.read_bytes()),
        "input_label_convention": "interval_open",
        "aggregate_label_convention": "interval_close",
        "calendar_feature_timezone": "America/New_York",
        "availability_rule": "aggregate close plus modeled latency",
        "historical_availability_authenticated": False,
        "context_bars": 128,
        "bucket_minutes": 15,
        "horizon_bars": 4,
        "contracts": 1,
        "point_value_usd": 2,
        "tick_size": 0.25,
        "sampling": {
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 0.9,
            "sample_count": 1,
        },
        "checkpoint_hashes": pilot.CHECKPOINT_HASHES,
        "source_hashes": pilot.SOURCE_FILES,
        "invalid_forecast": (
            "flat target; preserve available malformed paths and error"
        ),
        "contract_reset_preconditions": (
            "flat accounts, no pending order, no partial bucket; otherwise"
            " stop incomplete"
        ),
        "flatten_details": (
            "cancel pending orders at scheduled final minute open; missing"
            " minute stops incomplete without invented exit price"
        ),
        "repeat_target": "no trade",
        "reversal": "close then open, two charged sides",
        "seeds": list(pilot.SEEDS),
        "arms": ["kronos", "momentum", "flat"],
        "kronos_rule": (
            "sign(mean(three terminal forecast closes) - observed close)"
        ),
        "momentum_rule": (
            "sign(current close - close four completed bars earlier)"
        ),
        "warmup": (
            "128 completed same-contract 15-minute bars; reset on contract"
            " change"
        ),
        "forecast_bars": 4,
        "fill_rule": "first eligible minute open strictly after availability",
        "flatten": (
            "scheduled final minute open; no forecast crossing session close"
        ),
        "costs": {
            "commission": None,
            "exchange_regulatory": None,
            "slippage": None,
            "latency": None,
        },
        "outcomes_per_session_one_contract": ["K net", "M net", "K-M net"],
        "economic_requirements": ["mean(K net)>0", "mean(K-M net)>0"],
        "other_outcomes": ["Sharpe", "drawdown"],
        "prospective_collection": [
            (
                "Commit separate collection preregistration before launching"
                " any continuing collector."
            ),
            (
                "Archive dated calendar, contract selection, request"
                " identities, ordered raw observations and arrival times."
            ),
            (
                "Preserve completion flags, timezone, revisions and gaps"
                " without deduplication or inferred completion."
            ),
            (
                "Keep observations segregated from research exposure; freeze"
                " eligible sessions and execution assumptions before scoring."
            ),
            (
                "Run a dependence-aware power gate before any future strategy"
                " test."
            ),
        ],
    }
