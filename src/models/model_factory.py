"""Model factory — Sun et al. 2026 baselines and proposed model."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from src.models.cnn_lstm import CNNLSTM
from src.models.cnn_se_bilstm import CNNSEBiLSTM
from src.models.convnet import ConvNet
from src.models.eegnet import EEGNet


def build_model(model_name: str, model_cfg: dict[str, Any]) -> nn.Module:
    """Instantiate model by name."""
    name = model_name.lower().replace("-", "_")
    if name in ("proposed", "cnn_se_bilstm"):
        return CNNSEBiLSTM.from_config(model_cfg.get("proposed", model_cfg))
    if name == "cnn_lstm":
        return CNNLSTM.from_config(model_cfg.get("cnn_lstm", model_cfg))
    if name == "eegnet":
        return EEGNet.from_config(model_cfg.get("eegnet", model_cfg))
    if name in ("convnet", "shallow_convnet"):
        return ConvNet.from_config(model_cfg.get("convnet", model_cfg))
    raise ValueError(f"Unknown model: {model_name}")


def build_model_with_params(
    model_name: str,
    base_cfg: dict[str, Any],
    pso_params: dict[str, Any] | None = None,
) -> nn.Module:
    """Build proposed model with PSO-tuned CNN hyperparameters."""
    cfg = dict(base_cfg.get("proposed", base_cfg))
    if pso_params:
        cfg.update({
            "conv1_filters": int(pso_params["conv1_filters"]),
            "conv2_filters": int(pso_params["conv2_filters"]),
            "conv1_kernel": [
                int(pso_params["conv1_kernel_time"]),
                int(pso_params["conv1_kernel_channels"]),
            ],
            "conv2_kernel": [
                int(pso_params["conv2_kernel_time"]),
                int(pso_params["conv2_kernel_channels"]),
            ],
            "conv1_padding": None,
            "conv2_padding": None,
        })
    return CNNSEBiLSTM.from_config(cfg)
