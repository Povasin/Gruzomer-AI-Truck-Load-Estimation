"""
Ceiling / roof segmentation model.

Architecture:
    ResNet18 encoder
    +
    U-Net decoder
    +
    binary segmentation head

Input:
    B x 3 x H x W
    Recommended: 320 x 320 truck ROI

Output:
    B x 1 x H x W raw logits

IMPORTANT:
    Model returns LOGITS, not probabilities.

    During training:
        BCEWithLogitsLoss(logits, mask)

    During inference:
        probability = torch.sigmoid(logits)
        mask = probability >= threshold
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


# ============================================================
# CONV BLOCK
# ============================================================


class ConvBlock(nn.Module):
    """
    Two convolutions:

        Conv 3x3
        BatchNorm
        ReLU
        Conv 3x3
        BatchNorm
        ReLU

    Used in U-Net decoder after concatenating
    decoder features with encoder skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:

        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(
                inplace=True
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.block(x)


# ============================================================
# DECODER BLOCK
# ============================================================


class DecoderBlock(nn.Module):
    """
    U-Net decoder block.

    1. Resize decoder tensor to skip tensor size.
    2. Concatenate:
           decoder + encoder skip
    3. Process with ConvBlock.

    interpolate(size=...) is deliberately used instead of a
    fixed ConvTranspose2d because it is robust to arbitrary
    image sizes and odd feature-map dimensions.
    """

    def __init__(
        self,
        decoder_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:

        super().__init__()

        self.conv = ConvBlock(
            decoder_channels
            + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:

        # Example:
        # x    : B x 512 x 10 x 10
        # skip : B x 256 x 20 x 20
        #
        # resize x -> 20 x 20

        x = F.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat(
            [
                x,
                skip,
            ],
            dim=1,
        )

        x = self.conv(x)

        return x


# ============================================================
# CEILING U-NET + RESNET18
# ============================================================


class CeilingUNetResNet18(nn.Module):
    """
    Binary ceiling segmentation network.

    Encoder:
        ImageNet-pretrained ResNet18

    Decoder:
        U-Net style skip connections.

    For 320x320 input:

        input                3 x 320 x 320

        stem                64 x 160 x 160
        layer1              64 x  80 x  80
        layer2             128 x  40 x  40
        layer3             256 x  20 x  20
        layer4             512 x  10 x  10

        decoder3            256 x  20 x  20
        decoder2            128 x  40 x  40
        decoder1             64 x  80 x  80
        decoder0             32 x 160 x 160

        segmentation head     1 x 320 x 320
    """

    def __init__(
        self,
        pretrained: bool = True,
    ) -> None:

        super().__init__()

        # ====================================================
        # RESNET18 ENCODER
        # ====================================================

        if pretrained:
            weights = (
                ResNet18_Weights.DEFAULT
            )
        else:
            weights = None

        encoder = resnet18(
            weights=weights
        )

        # ----------------------------------------------------
        # Stem
        #
        # Input:
        # 3 x 320 x 320
        #
        # Output:
        # 64 x 160 x 160
        # ----------------------------------------------------

        self.encoder_stem = (
            nn.Sequential(
                encoder.conv1,
                encoder.bn1,
                encoder.relu,
            )
        )

        # maxpool:
        # 160 -> 80
        self.encoder_pool = (
            encoder.maxpool
        )

        # ----------------------------------------------------
        # ResNet stages
        # ----------------------------------------------------

        # 64 x 80 x 80
        self.encoder1 = (
            encoder.layer1
        )

        # 128 x 40 x 40
        self.encoder2 = (
            encoder.layer2
        )

        # 256 x 20 x 20
        self.encoder3 = (
            encoder.layer3
        )

        # 512 x 10 x 10
        self.encoder4 = (
            encoder.layer4
        )

        # ====================================================
        # U-NET DECODER
        # ====================================================

        # 512 @ 10x10
        # +
        # 256 @ 20x20
        #
        # ->
        # 256 @ 20x20
        self.decoder3 = (
            DecoderBlock(
                decoder_channels=512,
                skip_channels=256,
                out_channels=256,
            )
        )

        # 256 @ 20x20
        # +
        # 128 @ 40x40
        #
        # ->
        # 128 @ 40x40
        self.decoder2 = (
            DecoderBlock(
                decoder_channels=256,
                skip_channels=128,
                out_channels=128,
            )
        )

        # 128 @ 40x40
        # +
        # 64 @ 80x80
        #
        # ->
        # 64 @ 80x80
        self.decoder1 = (
            DecoderBlock(
                decoder_channels=128,
                skip_channels=64,
                out_channels=64,
            )
        )

        # 64 @ 80x80
        # +
        # stem 64 @ 160x160
        #
        # ->
        # 32 @ 160x160
        self.decoder0 = (
            DecoderBlock(
                decoder_channels=64,
                skip_channels=64,
                out_channels=32,
            )
        )

        # ====================================================
        # SEGMENTATION HEAD
        # ====================================================

        self.segmentation_head = (
            nn.Sequential(
                nn.Conv2d(
                    32,
                    16,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(
                    16
                ),
                nn.ReLU(
                    inplace=True
                ),

                nn.Conv2d(
                    16,
                    1,
                    kernel_size=1,
                ),
            )
        )

        # Initialize only decoder/head.
        #
        # DO NOT reinitialize the encoder,
        # otherwise ImageNet pretrained weights
        # would be destroyed.
        self._initialize_decoder()


    # ========================================================
    # INITIALIZATION
    # ========================================================

    def _initialize_decoder(
        self,
    ) -> None:

        modules = [
            self.decoder3,
            self.decoder2,
            self.decoder1,
            self.decoder0,
            self.segmentation_head,
        ]

        for parent in modules:

            for module in parent.modules():

                if isinstance(
                    module,
                    nn.Conv2d,
                ):

                    nn.init.kaiming_normal_(
                        module.weight,
                        mode="fan_out",
                        nonlinearity="relu",
                    )

                    if (
                        module.bias
                        is not None
                    ):
                        nn.init.zeros_(
                            module.bias
                        )

                elif isinstance(
                    module,
                    nn.BatchNorm2d,
                ):

                    nn.init.ones_(
                        module.weight
                    )

                    nn.init.zeros_(
                        module.bias
                    )


    # ========================================================
    # ENCODER FREEZE
    # ========================================================

    def freeze_encoder(
        self,
    ) -> None:
        """
        Optional helper.

        Can be used for first training epochs if desired.
        """

        modules = [
            self.encoder_stem,
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4,
        ]

        for module in modules:

            for parameter in (
                module.parameters()
            ):
                parameter.requires_grad = False


    def unfreeze_encoder(
        self,
    ) -> None:
        """
        Re-enable training of ResNet18 encoder.
        """

        modules = [
            self.encoder_stem,
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4,
        ]

        for module in modules:

            for parameter in (
                module.parameters()
            ):
                parameter.requires_grad = True


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        input_size = (
            x.shape[-2:]
        )

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------

        # B x 64 x 160 x 160
        stem = self.encoder_stem(
            x
        )

        # B x 64 x 80 x 80
        pooled = self.encoder_pool(
            stem
        )

        # B x 64 x 80 x 80
        e1 = self.encoder1(
            pooled
        )

        # B x 128 x 40 x 40
        e2 = self.encoder2(
            e1
        )

        # B x 256 x 20 x 20
        e3 = self.encoder3(
            e2
        )

        # B x 512 x 10 x 10
        e4 = self.encoder4(
            e3
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        d3 = self.decoder3(
            e4,
            e3,
        )

        d2 = self.decoder2(
            d3,
            e2,
        )

        d1 = self.decoder1(
            d2,
            e1,
        )

        d0 = self.decoder0(
            d1,
            stem,
        )

        # ----------------------------------------------------
        # SEGMENTATION HEAD
        # ----------------------------------------------------

        logits = (
            self.segmentation_head(
                d0
            )
        )

        # 160x160 -> original input size
        #
        # Normally:
        # 160 -> 320
        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        return logits


# ============================================================
# FACTORY
# ============================================================


def build_model(
    pretrained: bool = True,
) -> CeilingUNetResNet18:
    """
    Single model constructor used by train.py and runtime.py.

    This prevents train and inference from accidentally
    creating slightly different architectures.
    """

    return CeilingUNetResNet18(
        pretrained=pretrained
    )


# ============================================================
# QUICK TEST
# ============================================================


if __name__ == "__main__":

    model = build_model(
        pretrained=False
    )

    model.eval()

    x = torch.randn(
        2,
        3,
        320,
        320,
    )

    with torch.no_grad():

        logits = model(x)

        probabilities = (
            torch.sigmoid(
                logits
            )
        )


    print(
        "Input:",
        tuple(x.shape),
    )

    print(
        "Logits:",
        tuple(logits.shape),
    )

    print(
        "Probabilities:",
        tuple(
            probabilities.shape
        ),
    )

    print(
        "Probability range:",
        float(
            probabilities.min()
        ),
        float(
            probabilities.max()
        ),
    )

    expected_shape = (
        2,
        1,
        320,
        320,
    )

    assert (
        tuple(logits.shape)
        == expected_shape
    )

    print(
        "MODEL TEST PASSED"
    )