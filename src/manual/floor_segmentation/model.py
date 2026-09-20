from __future__ import annotations

import torch
from torch import nn

import segmentation_models_pytorch as smp


class FloorUNet(nn.Module):
    """
    U-Net с pretrained ResNet18 encoder.

    Вход:
        [B, 3, H, W]

    Выход:
        [B, 1, H, W]

    Выход модели = logits.
    Sigmoid здесь НЕ используется,
    потому что train.py использует BCEWithLogitsLoss.
    """

    def __init__(
        self,
        pretrained: bool = True,
    ):
        super().__init__()

        encoder_weights = (
            "imagenet"
            if pretrained
            else None
        )

        self.model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.model(x)