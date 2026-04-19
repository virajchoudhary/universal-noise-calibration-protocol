"""Serialization utilities for NSP profiles, calibration configs, and results."""
from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        return asdict(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def load_json(path: str | Path) -> Any:
    with Path(path).open("r") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)
    return path


def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def timestamped_dir(root: str | Path, prefix: str = "run") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(root) / f"{prefix}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out
