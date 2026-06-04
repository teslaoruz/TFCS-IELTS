from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BENCHMARK = _PROJECT_ROOT / "configs" / "benchmark.yaml"
_DEFAULT_MODELS = _PROJECT_ROOT / "configs" / "models.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_benchmark_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else _DEFAULT_BENCHMARK
    return _load_yaml(path)


def load_models_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else _DEFAULT_MODELS
    return _load_yaml(path)


def get_project_root() -> Path:
    return _PROJECT_ROOT


def resolve_path(path: str | Path, config: dict[str, Any] | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return _PROJECT_ROOT / p


def resolve_torch_device(preferred_device: str | None = "auto") -> str:
    preferred = str(preferred_device or "auto").strip().lower()

    try:
        import torch
    except ImportError:
        return "cpu"

    if preferred == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    if preferred.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"

    return preferred
