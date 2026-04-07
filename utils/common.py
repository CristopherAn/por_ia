import os
import json
from typing import Any, Dict


def load_config_file(path: str, *, fallback_path: str | None = None) -> Dict[str, Any]:
    if not os.path.exists(path):
        if fallback_path and os.path.exists(fallback_path):
            path = fallback_path
        else:
            return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def normalize_local_path(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/")
