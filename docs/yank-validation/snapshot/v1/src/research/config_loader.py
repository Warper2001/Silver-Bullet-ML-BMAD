"""config_loader.py — Load StrategyConfig from a YAML file.

This module intentionally lives outside strategy_core.py to preserve AR1
(no I/O in strategy_core). Call load_strategy_config() from the live trader
or CLI tools; never import it from strategy_core itself.
"""

import dataclasses
from datetime import time
from pathlib import Path
from typing import Union

import yaml

from src.research.strategy_core import StrategyConfig


def load_strategy_config(yaml_path: Union[str, Path]) -> StrategyConfig:
    """Load a StrategyConfig from a YAML file.

    Only YAML keys that match StrategyConfig field names are used; unknown
    keys are silently ignored. Missing keys use the StrategyConfig default.
    Time fields are accepted as "HH:MM" strings (e.g. "09:30").

    Parameters
    ----------
    yaml_path:
        Path to the YAML config file.

    Returns
    -------
    StrategyConfig

    Raises
    ------
    FileNotFoundError
        When yaml_path does not exist.
    yaml.YAMLError
        When the file contains invalid YAML.
    """
    path = Path(yaml_path)
    raw = yaml.safe_load(path.read_text())
    if not raw:
        return StrategyConfig()

    defaults = dataclasses.asdict(StrategyConfig())
    merged: dict = dict(defaults)

    for k, v in raw.items():
        if k not in defaults:
            continue
        base_val = defaults[k]
        if isinstance(base_val, time) and isinstance(v, str):
            h, m = v.split(":")
            merged[k] = time(int(h), int(m))
        else:
            merged[k] = v

    return StrategyConfig(**merged)
