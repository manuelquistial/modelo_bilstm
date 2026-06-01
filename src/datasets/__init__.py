"""Dataset loaders (lazy imports — avoids loading PyTorch for MOABB import scripts)."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "LowerLimbMIDataset",
    "generate_all_subjects",
    "load_subject_trials",
    "prepare_subject_data",
    "import_bnci2014_001",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "load_subject_trials": ("src.datasets.data_loader", "load_subject_trials"),
    "prepare_subject_data": ("src.datasets.data_loader", "prepare_subject_data"),
    "LowerLimbMIDataset": ("src.datasets.lower_limb_dataset", "LowerLimbMIDataset"),
    "generate_all_subjects": ("src.datasets.synthetic_data", "generate_all_subjects"),
    "import_bnci2014_001": ("src.datasets.bnci2014_001_loader", "import_bnci2014_001"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = _LAZY_EXPORTS[name]
    return getattr(importlib.import_module(module_path), attr)
