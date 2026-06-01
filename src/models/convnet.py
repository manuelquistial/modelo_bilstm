"""ConvNet baseline — alias to ShallowConvNet [37] per Sun et al. §2.6."""

from __future__ import annotations

from src.models.shallow_convnet import ShallowConvNet

ConvNet = ShallowConvNet

__all__ = ["ConvNet", "ShallowConvNet"]
