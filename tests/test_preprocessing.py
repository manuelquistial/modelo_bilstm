"""Test preprocessing on synthetic raw data."""

from pathlib import Path

import numpy as np
import pytest

from src.datasets.synthetic_data import generate_subject
from src.preprocessing.preprocessing_pipeline import preprocess_subject
from src.utils.config import load_config, project_root


@pytest.fixture
def synthetic_subject(tmp_path):
    generate_subject("S99", tmp_path, n_sessions=1, trials_per_session=10, seed=0)
    return tmp_path


def test_synthetic_to_processed(synthetic_subject):
    root = project_root()
    prep_cfg = load_config(root / "configs" / "preprocessing.yaml")
    dataset_cfg = load_config(root / "configs" / "dataset.yaml")
    out = synthetic_subject / "processed"
    result = preprocess_subject("S99", synthetic_subject, out, prep_cfg, dataset_cfg["channel_names"])
    X = result["X"]
    assert X.shape[1] == 501
    assert X.shape[2] == 15
    assert len(result["y"]) == 10
