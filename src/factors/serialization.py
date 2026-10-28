"""JSON helpers shared by Stage 4 outputs."""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def jsonable(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    if isinstance(value, np.ndarray):
        return [jsonable(item) for item in value.tolist()]
    return value


def json_default(value: Any) -> Any:
    converted = jsonable(value)
    return str(value) if converted is value else converted
