"""Test paper model output shapes (Table 1)."""

import torch

from src.models.cnn_se_bilstm import CNNSEBiLSTM
from src.models.paper_cnn_frontend import PAPER_FLATTEN_SIZE


def test_cnn_se_bilstm_output_shape():
    model = CNNSEBiLSTM()
    x = torch.randn(4, 251, 15)
    logits = model(x)
    assert logits.shape == (4, 2)


def test_flatten_target_size():
    model = CNNSEBiLSTM()
    x = torch.randn(2, 251, 15)
    flat = model.cnn_se(x)
    assert flat.shape == (2, PAPER_FLATTEN_SIZE)
