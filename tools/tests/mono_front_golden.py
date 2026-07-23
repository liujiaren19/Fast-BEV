#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""读取旧版板端实现冻结出的版本化 golden fixture。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" /
    "mono_front_board_legacy_v1.json")


def load_golden() -> Dict[str, Any]:
    with FIXTURE_PATH.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema_version") != 1:
        raise RuntimeError(
            f"不支持的 mono-front golden schema: {value.get('schema_version')}")
    return value


def array_fingerprint(value: np.ndarray) -> Dict[str, Any]:
    """冻结 shape/dtype/连续内存字节，避免浮点容差掩盖重构漂移。"""
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }
