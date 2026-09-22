from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


# ============================================================
# LOCAL IMPORTS
# ============================================================

try:
    from .model import build_model

    from .dataset import (
        IMAGENET_MEAN,
        IMAGENET_STD,
    )

except ImportError:

    from model import build_model

    from dataset import (
        IMAGENET_MEAN,
        IMAGENET_STD,
    )


# ============================================================
# DEFAULTS
# ============================================================


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "ceiling_unet_resnet18.best_iou.pt"
)


DEFAULT_IMAGE_SIZE = 320


# ============================================================
# CEILING SEGMENTER
# ============================================================


class CeilingSegmenter:
    """
    Runtime inference для CeilingUNetResNet18.

    Вход:
        truck ROI

        numpy:
            H x W x 3
            uint8
            RGB по умолчанию

        или PIL.Image

    Выход:

        probability:
            H x W
            float32
            диапазон [0, 1]

        mask:
            H x W
            uint8
            значения {0, 1}

    ВАЖНО:

    probability и mask возвращаются в ИСХОДНОМ
    размере truck ROI.

    Модель внутри работает на 320x320.
    """

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_MODEL_PATH,
        device: str = "auto",
        threshold: Optional[float] = None,
    ) -> None:

        self.weights_path = Path(
            weights_path
        )


        if not self.weights_path.exists():

            raise FileNotFoundError(
                f"Ceiling checkpoint не найден:\n"
                f"{self.weights_path}"
            )


        # ====================================================
        # DEVICE
        # ====================================================

        if device == "auto":

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )


        if (
            device == "cuda"
            and
            not torch.cuda.is_available()
        ):

            raise RuntimeError(
                "Выбран CUDA, но "
                "torch.cuda.is_available() == False"
            )


        self.device = torch.device(
            device
        )


        # ====================================================
        # CHECKPOINT
        # ====================================================

        checkpoint = self._load_checkpoint(
            self.weights_path
        )


        # ====================================================
        # CONFIG
        # ====================================================

        self.image_size = int(
            checkpoint.get(
                "image_size",
                DEFAULT_IMAGE_SIZE,
            )
        )


        checkpoint_threshold = float(
            checkpoint.get(
                "threshold",
                0.5,
            )
        )


        if threshold is None:

            self.threshold = (
                checkpoint_threshold
            )

        else:

            self.threshold = float(
                threshold
            )


        if not (
            0.0
            <= self.threshold
            <= 1.0
        ):

            raise ValueError(
                f"Некорректный threshold: "
                f"{self.threshold}"
            )


        # ====================================================
        # MODEL
        # ====================================================

        # pretrained=False:
        #
        # ImageNet weights нам больше не нужны,
        # потому что сейчас загрузятся наши обученные веса.
        self.model = build_model(
            pretrained=False
        )


        state_dict = checkpoint.get(
            "state_dict"
        )


        if state_dict is None:

            raise RuntimeError(
                "В checkpoint нет 'state_dict'."
            )


        self.model.load_state_dict(
            state_dict,
            strict=True,
        )


        self.model.to(
            self.device
        )


        self.model.eval()


        # ====================================================
        # INFO
        # ====================================================

        self.epoch = checkpoint.get(
            "epoch"
        )

        self.val_iou = checkpoint.get(
            "val_iou"
        )

        self.val_dice = checkpoint.get(
            "val_dice"
        )


    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    def _load_checkpoint(
        self,
        path: Path,
    ) -> dict:

        """
        Совместим с разными версиями PyTorch.
        """

        try:

            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

        except TypeError:

            checkpoint = torch.load(
                path,
                map_location="cpu",
            )


        if not isinstance(
            checkpoint,
            dict,
        ):

            raise RuntimeError(
                "Checkpoint должен быть dict."
            )


        return checkpoint


    # ========================================================
    # INPUT CONVERSION
    # ========================================================

    @staticmethod
    def _to_pil(
        image: np.ndarray | Image.Image,
        input_bgr: bool = False,
    ) -> Image.Image:

        """
        Приводит вход к RGB PIL.Image.

        input_bgr=True нужен, если изображение пришло
        напрямую из cv2.imread().
        """

        if isinstance(
            image,
            Image.Image,
        ):

            return image.convert(
                "RGB"
            )


        if not isinstance(
            image,
            np.ndarray,
        ):

            raise TypeError(
                "image должен быть numpy.ndarray "
                "или PIL.Image."
            )


        if image.ndim != 3:

            raise ValueError(
                f"Ожидалось HxWx3, "
                f"получено shape={image.shape}"
            )


        if image.shape[2] != 3:

            raise ValueError(
                f"Ожидалось 3 канала, "
                f"получено {image.shape[2]}"
            )


        array = image


        # ----------------------------------------------------
        # dtype -> uint8
        # ----------------------------------------------------

        if array.dtype != np.uint8:

            array = array.astype(
                np.float32
            )


            # Если RGB находится в диапазоне 0..1.
            if (
                array.size > 0
                and
                array.max() <= 1.0
            ):

                array = (
                    array
                    * 255.0
                )


            array = np.clip(
                array,
                0,
                255,
            ).astype(
                np.uint8
            )


        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        if input_bgr:

            array = array[
                :,
                :,
                ::-1,
            ]


        # После ::-1 могут появиться negative strides.
        array = np.ascontiguousarray(
            array
        )


        return Image.fromarray(
            array,
            mode="RGB",
        )


    # ========================================================
    # PREPROCESS
    # ========================================================

    def preprocess(
        self,
        image: np.ndarray | Image.Image,
        input_bgr: bool = False,
    ) -> tuple[
        torch.Tensor,
        tuple[int, int],
    ]:

        """
        Тот же preprocessing, что в dataset.py:

            RGB
            ↓
            resize 320x320
            ↓
            tensor 0..1
            ↓
            ImageNet normalization

        Возвращает:
            tensor [1, 3, 320, 320]
            original_size = (height, width)
        """

        pil_image = self._to_pil(
            image,
            input_bgr=input_bgr,
        )


        original_width, original_height = (
            pil_image.size
        )


        resized = TF.resize(
            pil_image,
            [
                self.image_size,
                self.image_size,
            ],
            interpolation=(
                InterpolationMode.BILINEAR
            ),
            antialias=True,
        )


        tensor = TF.to_tensor(
            resized
        )


        tensor = TF.normalize(
            tensor,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )


        # C,H,W -> 1,C,H,W
        tensor = tensor.unsqueeze(
            0
        )


        tensor = tensor.to(
            self.device
        )


        return (
            tensor,
            (
                original_height,
                original_width,
            ),
        )


    # ========================================================
    # PREDICT PROBABILITY
    # ========================================================

    @torch.inference_mode()
    def predict_probability(
        self,
        image: np.ndarray | Image.Image,
        input_bgr: bool = False,
    ) -> np.ndarray:

        """
        Возвращает soft probability map.

        Shape:
            original H x original W

        dtype:
            float32

        range:
            0..1
        """

        tensor, original_size = (
            self.preprocess(
                image,
                input_bgr=input_bgr,
            )
        )


        logits = self.model(
            tensor
        )


        probabilities = torch.sigmoid(
            logits
        )


        # ----------------------------------------------------
        # Вернуть prediction в исходный размер ROI
        # ----------------------------------------------------

        probabilities = (
            torch.nn.functional.interpolate(
                probabilities,
                size=original_size,
                mode="bilinear",
                align_corners=False,
            )
        )


        probability = (
            probabilities[
                0,
                0,
            ]
            .float()
            .cpu()
            .numpy()
        )


        probability = np.clip(
            probability,
            0.0,
            1.0,
        ).astype(
            np.float32
        )


        return probability


    # ========================================================
    # PREDICT MASK
    # ========================================================

    @torch.inference_mode()
    def predict_mask(
        self,
        image: np.ndarray | Image.Image,
        input_bgr: bool = False,
        threshold: Optional[float] = None,
    ) -> np.ndarray:

        """
        Возвращает binary mask.

        Значения:
            0 = background
            1 = ceiling
        """

        probability = (
            self.predict_probability(
                image,
                input_bgr=input_bgr,
            )
        )


        if threshold is None:

            threshold = (
                self.threshold
            )


        mask = (
            probability
            >= threshold
        ).astype(
            np.uint8
        )


        return mask


    # ========================================================
    # PREDICT BOTH
    # ========================================================

    @torch.inference_mode()
    def predict(
        self,
        image: np.ndarray | Image.Image,
        input_bgr: bool = False,
        threshold: Optional[float] = None,
    ) -> dict:

        """
        Основной метод runtime.

        Возвращает одновременно:

            probability
            mask
            threshold

        Именно этот метод удобно использовать
        потом в features.py.
        """

        probability = (
            self.predict_probability(
                image,
                input_bgr=input_bgr,
            )
        )


        if threshold is None:

            threshold = (
                self.threshold
            )


        threshold = float(
            threshold
        )


        mask = (
            probability
            >= threshold
        ).astype(
            np.uint8
        )


        return {
            "probability": (
                probability
            ),
            "mask": mask,
            "threshold": (
                threshold
            ),
        }


# ============================================================
# GLOBAL CACHE
# ============================================================

_GLOBAL_SEGMENTERS: dict[
    tuple[str, str],
    CeilingSegmenter,
] = {}


def get_ceiling_segmenter(
    weights_path: str | Path | None = DEFAULT_MODEL_PATH,
    device: str = "auto",
) -> CeilingSegmenter:
    """
    Загружает CeilingUNet только один раз.

    Важно для feature extraction:

        1000 фотографий

    нельзя создавать модель заново
    для каждого изображения.
    """

    path = Path(
        DEFAULT_MODEL_PATH if weights_path is None else weights_path
    ).resolve()


    # auto превращаем в реальный device,
    # чтобы cache key был стабильным.
    if device == "auto":

        resolved_device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        resolved_device = device


    key = (
        str(path),
        resolved_device,
    )


    if key not in _GLOBAL_SEGMENTERS:

        _GLOBAL_SEGMENTERS[
            key
        ] = CeilingSegmenter(
            weights_path=path,
            device=resolved_device,
        )


    return _GLOBAL_SEGMENTERS[
        key
    ]


# ============================================================
# SIMPLE FUNCTIONS
# ============================================================


def predict_ceiling_mask(
    image: np.ndarray | Image.Image,
    weights_path: str | Path | None = DEFAULT_MODEL_PATH,
    device: str = "auto",
    input_bgr: bool = False,
) -> np.ndarray:
    """
    Быстрый внешний API.

    Пример:

        mask = predict_ceiling_mask(
            roi_rgb
        )
    """

    segmenter = get_ceiling_segmenter(
        weights_path=weights_path,
        device=device,
    )


    return segmenter.predict_mask(
        image,
        input_bgr=input_bgr,
    )


def predict_ceiling_probability(
    image: np.ndarray | Image.Image,
    weights_path: str | Path | None = DEFAULT_MODEL_PATH,
    device: str = "auto",
    input_bgr: bool = False,
) -> np.ndarray:
    """
    Soft probability map.

    Нужна позже для soft ceiling features.
    """

    segmenter = get_ceiling_segmenter(
        weights_path=weights_path,
        device=device,
    )


    return (
        segmenter.predict_probability(
            image,
            input_bgr=input_bgr,
        )
    )


# ============================================================
# QUICK TEST
# ============================================================


if __name__ == "__main__":

    import argparse


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "image",
        type=Path,
    )


    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )


    parser.add_argument(
        "--device",
        type=str,
        default="auto",
    )


    args = parser.parse_args()


    if not args.image.exists():

        raise FileNotFoundError(
            args.image
        )


    image = Image.open(
        args.image
    ).convert(
        "RGB"
    )


    segmenter = CeilingSegmenter(
        weights_path=args.weights,
        device=args.device,
    )


    result = segmenter.predict(
        image
    )


    probability = (
        result["probability"]
    )

    mask = (
        result["mask"]
    )


    print()
    print(
        "===== CEILING RUNTIME TEST ====="
    )

    print(
        "input size:",
        image.size,
    )

    print(
        "probability shape:",
        probability.shape,
    )

    print(
        "mask shape:",
        mask.shape,
    )

    print(
        "threshold:",
        result["threshold"],
    )

    print(
        "probability min/max:",
        float(
            probability.min()
        ),
        float(
            probability.max()
        ),
    )

    print(
        "ceiling pixels:",
        int(
            mask.sum()
        ),
    )

    print(
        "ceiling ratio:",
        float(
            mask.mean()
        ),
    )

    print(
        "checkpoint epoch:",
        segmenter.epoch,
    )

    print(
        "checkpoint val IoU:",
        segmenter.val_iou,
    )

    print(
        "checkpoint val Dice:",
        segmenter.val_dice,
    )

    print(
        "RUNTIME TEST PASSED"
    )
