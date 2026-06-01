"""Utilities (torch-dependent helpers loaded only on demand)."""

from __future__ import annotations

from typing import Any

from src.utils.config import load_config, merge_configs, project_root
from src.utils.constants import CHANNEL_NAMES, CLASS_NAMES
from src.utils.io import ensure_dir, load_trials_npz, save_trials_npz
from src.utils.logging import setup_logger

__all__ = [
    "CHANNEL_NAMES",
    "CLASS_NAMES",
    "ensure_dir",
    "get_device",
    "load_config",
    "load_trials_npz",
    "merge_configs",
    "project_root",
    "save_trials_npz",
    "set_seed",
    "setup_logger",
]


def get_device(*args: Any, **kwargs: Any) -> Any:
    from src.utils.device import get_device as _get_device

    return _get_device(*args, **kwargs)


def set_seed(*args: Any, **kwargs: Any) -> None:
    from src.utils.seed import set_seed as _set_seed

    _set_seed(*args, **kwargs)
