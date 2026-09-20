from pathlib import Path

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


PROJECT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_DIR
    / "src"
    / "segmentation"
    / "models"
    / "best_unet_resnet18.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ------------------------------------------------------------
# Глобальный cache.
#
# КРИТИЧЕСКИ ВАЖНО:
# модель загружается ОДИН раз,
# а не заново для каждой фотографии.
# ------------------------------------------------------------

_MODEL = None
_TRANSFORM = None
_IMAGE_SIZE = None
_MODEL_PATH = None


def _create_model():
    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )


def _load_segmenter(weights_path=None):
    global _MODEL
    global _TRANSFORM
    global _IMAGE_SIZE
    global _MODEL_PATH

    model_path = Path(weights_path or MODEL_PATH).resolve()

    if _MODEL is not None and _MODEL_PATH == model_path:
        return
    if not model_path.is_file():
        raise FileNotFoundError(
        f"Segmentation model not found: {model_path}"
        )

    print(
        f"[SEGMENTATION] loading model "
        f"from {model_path}"
    )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )

    _IMAGE_SIZE = checkpoint.get(
        "image_size",
        256,
    )

    model = _create_model()

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    _MODEL = model
    _MODEL_PATH = model_path

    _TRANSFORM = A.Compose([
        A.Resize(
            _IMAGE_SIZE,
            _IMAGE_SIZE,
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),

        ToTensorV2(),
    ])

    print(
        f"[SEGMENTATION] ready | "
        f"device={DEVICE} | "
        f"size={_IMAGE_SIZE}"
    )


def predict_truck_mask(
    image_rgb,
    threshold=0.5,
    weights_path=None,
):
    """
    image_rgb:
        H x W x 3 RGB uint8

    return:
        H x W uint8

        0 = игнорируем
        1 = внутренняя часть кузова
    """

    _load_segmenter(weights_path)

    height, width = image_rgb.shape[:2]

    transformed = _TRANSFORM(
        image=image_rgb
    )

    tensor = (
        transformed["image"]
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.inference_mode():

        logits = _MODEL(tensor)

        probability = torch.sigmoid(
            logits
        )[0, 0]

    probability = (
        probability
        .cpu()
        .numpy()
    )

    # Возвращаем размер исходной фотографии
    probability = cv2.resize(
        probability,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    mask = (
        probability >= threshold
    ).astype(np.uint8)

    return mask
