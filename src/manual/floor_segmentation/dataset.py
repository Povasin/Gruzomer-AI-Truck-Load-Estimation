from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# =========================================================
# IMAGENET NORMALIZATION
# =========================================================

IMAGENET_MEAN = np.array(
    [
        0.485,
        0.456,
        0.406,
    ],
    dtype=np.float32,
)

IMAGENET_STD = np.array(
    [
        0.229,
        0.224,
        0.225,
    ],
    dtype=np.float32,
)


class FloorSegmentationDataset(Dataset):

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        image_ids: list[str],
        size: int = 320,
        transform: Callable | None = None,
    ):

        self.image_dir = Path(
            image_dir
        )

        self.mask_dir = Path(
            mask_dir
        )

        self.image_ids = list(
            image_ids
        )

        self.size = int(
            size
        )

        self.transform = transform

    def __len__(
        self,
    ) -> int:

        return len(
            self.image_ids
        )

    def __getitem__(
        self,
        index: int,
    ):

        # =====================================================
        # ID
        # =====================================================

        image_id = self.image_ids[
            index
        ]

        image_path = (
            self.image_dir
            / f"{image_id}.jpg"
        )

        mask_path = (
            self.mask_dir
            / f"{image_id}.png"
        )

        # =====================================================
        # IMAGE
        # =====================================================

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            raise FileNotFoundError(
                f"Could not decode image: "
                f"{image_path}"
            )

        # BGR -> RGB

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # =====================================================
        # MASK
        # =====================================================

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:

            raise FileNotFoundError(
                f"Could not decode mask: "
                f"{mask_path}"
            )

        # =====================================================
        # RESIZE
        # =====================================================

        image = cv2.resize(
            image,
            (
                self.size,
                self.size,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        mask = cv2.resize(
            mask,
            (
                self.size,
                self.size,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        # =====================================================
        # AUGMENTATION
        # =====================================================

        if self.transform is not None:

            image, mask = self.transform(
                image,
                mask,
            )

        # =====================================================
        # 0..255 -> 0..1
        # =====================================================

        image = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        # =====================================================
        # IMAGENET NORMALIZATION
        # =====================================================

        image = (
            image
            - IMAGENET_MEAN
        ) / IMAGENET_STD

        # =====================================================
        # MASK -> 0/1
        # =====================================================

        mask = (
            mask > 127
        ).astype(
            np.float32
        )

        # =====================================================
        # HWC -> CHW
        # =====================================================

        image = np.transpose(
            image,
            (
                2,
                0,
                1,
            ),
        )

        # =====================================================
        # NUMPY -> TORCH
        # =====================================================

        image = torch.from_numpy(
            image
        ).float()

        mask = torch.from_numpy(
            mask[
                None,
                :,
                :,
            ]
        ).float()

        return (
            image,
            mask,
            image_id,
        )