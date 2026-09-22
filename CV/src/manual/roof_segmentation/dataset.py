from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


# ============================================================
# CONSTANTS
# ============================================================

IMAGE_SIZE = 320

VALID_IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)


# ImageNet normalization.
#
# Нужна потому, что encoder нашей CeilingUNet
# использует pretrained ResNet18.
IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


# ============================================================
# FILE UTILITIES
# ============================================================


def find_image(
    image_dir: str | Path,
    image_id: str,
) -> Path:
    """
    Ищет изображение по image_id с одним из допустимых расширений.

    Например:

        image_id = img_123

    будет искать:

        img_123.jpg
        img_123.jpeg
        img_123.png
        ...
    """

    image_dir = Path(image_dir)

    for extension in VALID_IMAGE_EXTENSIONS:

        path = (
            image_dir
            / f"{image_id}{extension}"
        )

        if path.exists():
            return path

        # На случай .JPG / .PNG
        path_upper = (
            image_dir
            / f"{image_id}{extension.upper()}"
        )

        if path_upper.exists():
            return path_upper

    raise FileNotFoundError(
        f"Не найдено изображение для image_id={image_id} "
        f"в папке {image_dir}"
    )


def find_mask(
    mask_dir: str | Path,
    image_id: str,
) -> Path:
    """
    Маски ceiling ожидаются в PNG:

        <image_id>.png
    """

    mask_dir = Path(mask_dir)

    path = (
        mask_dir
        / f"{image_id}.png"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Не найдена mask для image_id={image_id}: "
            f"{path}"
        )

    return path


# ============================================================
# DATASET
# ============================================================


class CeilingSegmentationDataset(Dataset):
    """
    Dataset для бинарной сегментации потолка кузова.

    Вход:
        truck ROI

    Выход:
        image:
            torch.float32
            shape = [3, 320, 320]

        mask:
            torch.float32
            shape = [1, 320, 320]
            values = {0, 1}

        image_id:
            str


    Parameters
    ----------
    image_ids:
        Список image_id без расширения.

    image_dir:
        Папка с truck ROI.

    mask_dir:
        Папка с бинарными PNG-масками потолка.

    image_size:
        Размер входа модели.

    augment:
        True для train.
        False для validation.

    normalize:
        ImageNet normalization для pretrained ResNet18.
    """

    def __init__(
        self,
        image_ids: Sequence[str],
        image_dir: str | Path,
        mask_dir: str | Path,
        image_size: int = IMAGE_SIZE,
        augment: bool = False,
        normalize: bool = True,
    ) -> None:

        super().__init__()

        self.image_ids = [
            str(image_id)
            for image_id in image_ids
        ]

        self.image_dir = Path(
            image_dir
        )

        self.mask_dir = Path(
            mask_dir
        )

        self.image_size = int(
            image_size
        )

        self.augment = bool(
            augment
        )

        self.normalize = bool(
            normalize
        )

        if not self.image_dir.exists():

            raise FileNotFoundError(
                f"Папка изображений не существует: "
                f"{self.image_dir}"
            )

        if not self.mask_dir.exists():

            raise FileNotFoundError(
                f"Папка масок не существует: "
                f"{self.mask_dir}"
            )

        if len(self.image_ids) == 0:

            raise ValueError(
                "CeilingSegmentationDataset получил "
                "пустой список image_ids."
            )


    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(
        self,
    ) -> int:

        return len(
            self.image_ids
        )


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    def _load_image(
        self,
        image_id: str,
    ) -> Image.Image:

        path = find_image(
            self.image_dir,
            image_id,
        )

        try:

            image = (
                Image.open(path)
                .convert("RGB")
            )

        except Exception as error:

            raise RuntimeError(
                f"Не удалось прочитать изображение: "
                f"{path}"
            ) from error

        return image


    # ========================================================
    # LOAD MASK
    # ========================================================

    def _load_mask(
        self,
        image_id: str,
    ) -> Image.Image:

        path = find_mask(
            self.mask_dir,
            image_id,
        )

        try:

            # L = grayscale
            mask = (
                Image.open(path)
                .convert("L")
            )

        except Exception as error:

            raise RuntimeError(
                f"Не удалось прочитать mask: "
                f"{path}"
            ) from error

        return mask


    # ========================================================
    # GEOMETRIC TRANSFORMS
    # ========================================================

    def _resize_pair(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[
        Image.Image,
        Image.Image,
    ]:
        """
        Resize изображения и маски до 320x320.

        КРИТИЧНО:

        image:
            bilinear

        mask:
            nearest

        Для mask нельзя использовать bilinear,
        иначе значения 0/255 превратятся в серые пиксели.
        """

        size = [
            self.image_size,
            self.image_size,
        ]

        image = TF.resize(
            image,
            size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        mask = TF.resize(
            mask,
            size,
            interpolation=InterpolationMode.NEAREST,
        )

        return (
            image,
            mask,
        )


    # ========================================================
    # AUGMENTATIONS
    # ========================================================

    def _augment_pair(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> tuple[
        Image.Image,
        Image.Image,
    ]:
        """
        Безопасные аугментации для ceiling segmentation.

        Геометрические операции применяются
        одновременно к image и mask.

        Цветовые операции применяются только к image.

        Нам нельзя сильно ломать перспективу,
        потому что позже из ceiling-mask будут
        извлекаться геометрические признаки.
        """

        # ----------------------------------------------------
        # Horizontal flip
        # ----------------------------------------------------

        if random.random() < 0.5:

            image = TF.hflip(
                image
            )

            mask = TF.hflip(
                mask
            )


        # ----------------------------------------------------
        # Small rotation
        #
        # Только небольшой угол.
        # Не делаем ±15/30 градусов.
        # ----------------------------------------------------

        if random.random() < 0.20:

            angle = random.uniform(
                -3.0,
                3.0,
            )

            image = TF.rotate(
                image,
                angle=angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )

            mask = TF.rotate(
                mask,
                angle=angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )


        # ----------------------------------------------------
        # Brightness
        # ----------------------------------------------------

        if random.random() < 0.40:

            factor = random.uniform(
                0.80,
                1.20,
            )

            image = (
                ImageEnhance.Brightness(
                    image
                )
                .enhance(
                    factor
                )
            )


        # ----------------------------------------------------
        # Contrast
        # ----------------------------------------------------

        if random.random() < 0.40:

            factor = random.uniform(
                0.80,
                1.20,
            )

            image = (
                ImageEnhance.Contrast(
                    image
                )
                .enhance(
                    factor
                )
            )


        # ----------------------------------------------------
        # Saturation
        # ----------------------------------------------------

        if random.random() < 0.20:

            factor = random.uniform(
                0.85,
                1.15,
            )

            image = (
                ImageEnhance.Color(
                    image
                )
                .enhance(
                    factor
                )
            )


        # ----------------------------------------------------
        # Gamma
        #
        # Полезно из-за разного освещения внутри кузовов.
        # ----------------------------------------------------

        if random.random() < 0.20:

            gamma = random.uniform(
                0.80,
                1.20,
            )

            image_array = np.asarray(
                image,
                dtype=np.float32,
            )

            image_array /= 255.0

            image_array = np.power(
                image_array,
                gamma,
            )

            image_array = np.clip(
                image_array * 255.0,
                0,
                255,
            ).astype(
                np.uint8
            )

            image = Image.fromarray(
                image_array,
                mode="RGB",
            )


        return (
            image,
            mask,
        )


    # ========================================================
    # MASK -> TENSOR
    # ========================================================

    @staticmethod
    def _mask_to_tensor(
        mask: Image.Image,
    ) -> torch.Tensor:
        """
        PNG mask:

            0   -> background
            255 -> ceiling

        превращаем в:

            0.0
            1.0

        Shape:

            [1, H, W]
        """

        mask_array = np.asarray(
            mask,
            dtype=np.uint8,
        )

        # Не доверяем тому, что mask строго
        # состоит только из 0/255.
        #
        # Всё >127 считаем ceiling.
        mask_array = (
            mask_array > 127
        ).astype(
            np.float32
        )

        mask_tensor = (
            torch.from_numpy(
                mask_array
            )
            .unsqueeze(0)
        )

        return mask_tensor


    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index: int,
    ) -> dict[
        str,
        torch.Tensor | str,
    ]:

        image_id = (
            self.image_ids[
                index
            ]
        )

        image = self._load_image(
            image_id
        )

        mask = self._load_mask(
            image_id
        )


        # ----------------------------------------------------
        # Проверяем исходные размеры.
        # ----------------------------------------------------

        if image.size != mask.size:

            raise ValueError(
                f"Размер image != mask для {image_id}: "
                f"image={image.size}, mask={mask.size}"
            )


        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image, mask = (
            self._resize_pair(
                image,
                mask,
            )
        )


        # ----------------------------------------------------
        # Train augmentations
        # ----------------------------------------------------

        if self.augment:

            image, mask = (
                self._augment_pair(
                    image,
                    mask,
                )
            )


        # ----------------------------------------------------
        # PIL -> Tensor
        #
        # [H,W,C]
        # ->
        # [C,H,W]
        #
        # values:
        # 0..255 -> 0..1
        # ----------------------------------------------------

        image_tensor = TF.to_tensor(
            image
        )


        # ----------------------------------------------------
        # ImageNet normalization
        # ----------------------------------------------------

        if self.normalize:

            image_tensor = TF.normalize(
                image_tensor,
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            )


        mask_tensor = (
            self._mask_to_tensor(
                mask
            )
        )


        # ----------------------------------------------------
        # Sanity checks
        # ----------------------------------------------------

        expected_image_shape = (
            3,
            self.image_size,
            self.image_size,
        )

        expected_mask_shape = (
            1,
            self.image_size,
            self.image_size,
        )

        if tuple(
            image_tensor.shape
        ) != expected_image_shape:

            raise RuntimeError(
                f"Неверный shape изображения "
                f"{image_id}: "
                f"{tuple(image_tensor.shape)}"
            )

        if tuple(
            mask_tensor.shape
        ) != expected_mask_shape:

            raise RuntimeError(
                f"Неверный shape маски "
                f"{image_id}: "
                f"{tuple(mask_tensor.shape)}"
            )


        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_id": image_id,
        }


# ============================================================
# HELPERS
# ============================================================


def image_ids_from_masks(
    mask_dir: str | Path,
) -> list[str]:
    """
    Возвращает все размеченные image_id,
    для которых существует PNG mask.

    Это удобно, потому что annotate.py уже
    сохраняет одну mask на каждое размеченное фото.
    """

    mask_dir = Path(
        mask_dir
    )

    if not mask_dir.exists():

        raise FileNotFoundError(
            f"Папка masks не существует: "
            f"{mask_dir}"
        )

    ids = sorted(
        path.stem
        for path in mask_dir.glob(
            "*.png"
        )
    )

    return ids


def validate_pairs(
    image_ids: Sequence[str],
    image_dir: str | Path,
    mask_dir: str | Path,
) -> None:
    """
    Проверяет весь dataset до начала обучения.

    Лучше упасть здесь сразу, чем через 3 эпохи
    из-за отсутствующей картинки.
    """

    missing_images = []
    missing_masks = []

    for image_id in image_ids:

        try:

            find_image(
                image_dir,
                str(image_id),
            )

        except FileNotFoundError:

            missing_images.append(
                str(image_id)
            )

        try:

            find_mask(
                mask_dir,
                str(image_id),
            )

        except FileNotFoundError:

            missing_masks.append(
                str(image_id)
            )


    if missing_images:

        preview = (
            missing_images[:10]
        )

        raise FileNotFoundError(
            f"Нет изображений для "
            f"{len(missing_images)} ID. "
            f"Примеры: {preview}"
        )


    if missing_masks:

        preview = (
            missing_masks[:10]
        )

        raise FileNotFoundError(
            f"Нет masks для "
            f"{len(missing_masks)} ID. "
            f"Примеры: {preview}"
        )


# ============================================================
# QUICK TEST
# ============================================================


if __name__ == "__main__":

    # Подстрой пути при необходимости.
    PROJECT_DIR = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    IMAGE_DIR = (
        PROJECT_DIR
        / "DataSet"
        / "train"
        / "floor_images"
    )

    MASK_DIR = (
        PROJECT_DIR
        / "debug_output"
        / "roof_masks"
    )


    ids = image_ids_from_masks(
        MASK_DIR
    )


    print(
        f"Найдено masks: "
        f"{len(ids)}"
    )


    if len(ids) == 0:

        raise RuntimeError(
            "Нет размеченных roof masks."
        )


    validate_pairs(
        ids,
        IMAGE_DIR,
        MASK_DIR,
    )


    dataset = (
        CeilingSegmentationDataset(
            image_ids=ids,
            image_dir=IMAGE_DIR,
            mask_dir=MASK_DIR,
            image_size=320,
            augment=True,
        )
    )


    sample = dataset[0]


    print(
        "image_id:",
        sample["image_id"],
    )

    print(
        "image:",
        sample["image"].shape,
        sample["image"].dtype,
    )

    print(
        "mask:",
        sample["mask"].shape,
        sample["mask"].dtype,
    )

    print(
        "mask min/max:",
        float(
            sample["mask"].min()
        ),
        float(
            sample["mask"].max()
        ),
    )

    print(
        "ceiling pixels:",
        int(
            sample["mask"].sum()
        ),
    )

    print(
        "DATASET TEST PASSED"
    )