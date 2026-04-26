"""Shared utilities for FraudShield training and evaluation."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch when available."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # pragma: no cover - torch is optional at runtime
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a ``Path``."""

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(payload: Any, path: str | Path) -> None:
    """Write JSON with stable indentation."""

    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """Load JSON from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from model output."""

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("Model output did not contain a JSON object.")
    return json.loads(text[start : end + 1])


def moving_average(values: Sequence[float], window: int = 10) -> list[float]:
    """Compute a simple moving average."""

    if not values:
        return []
    window = max(1, int(window))
    averaged: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        averaged.append(sum(chunk) / len(chunk))
    return averaged


def approximate_token_count(text: str) -> int:
    """Cheap token estimate that works without a tokenizer."""

    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, int(len(stripped.split()) * 1.3))


def flatten_dict_items(mapping: dict[str, Any], prefix: str = "") -> Iterable[tuple[str, Any]]:
    """Flatten nested dictionaries for logging."""

    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from flatten_dict_items(value, prefix=full_key)
        else:
            yield full_key, value
