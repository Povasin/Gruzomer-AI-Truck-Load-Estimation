from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from floor_segmentation.model import FloorUNet

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
DEFAULT_SIZE = 320

DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "floor_unet_resnet18_lr1e3.best_loss.pt"
)

_MODEL = None
_DEVICE = None
_MODEL_PATH = None


def _load_model(
    weights_path: str | Path | None = None,
):
    """
    Загружает floor segmentation model один раз
    и затем переиспользует её для следующих изображений.

    Если весов нет, возвращает:
        None, None
    """

    global _MODEL
    global _DEVICE
    global _MODEL_PATH

    weights_path = Path(
        weights_path
        or DEFAULT_WEIGHTS_PATH
    ).resolve()

    # Переиспользуем модель только для того же checkpoint.
    if _MODEL is not None and _MODEL_PATH == weights_path:
        return (
            _MODEL,
            _DEVICE,
        )

    import torch

    # Если веса ещё не обучены
    if not weights_path.is_file():
        return (
            None,
            None,
        )

    # CUDA, если доступна
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # Создаём архитектуру
    model = FloorUNet(
    pretrained=False
    )

    # Загружаем checkpoint
    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=True,
    )

    # train.py сохраняет:
    #
    # {
    #     "state_dict": ...,
    #     "epoch": ...,
    #     "valid_loss": ...
    # }
    #
    # Но поддерживаем и вариант,
    # когда сохранён только state_dict.
    if (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]

    else:
        state_dict = checkpoint

    # Загружаем веса
    model.load_state_dict(
        state_dict
    )

    model = model.to(
        device
    )

    # Переключаем BatchNorm и т.д.
    # в inference mode
    model.eval()

    _MODEL = model
    _DEVICE = device
    _MODEL_PATH = weights_path

    return (
        _MODEL,
        _DEVICE,
    )


def heuristic_floor_mask(
    image: np.ndarray,
    truck_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Временная эвристика.

    Используется ТОЛЬКО если ещё нет
    обученной floor segmentation model.

    Идея:
    1. Берём нижнюю центральную область как seed пола
    2. Смотрим её средний цвет в LAB
    3. Ищем похожие пиксели внутри кузова
    4. Убираем сильно текстурные области
    5. Оставляем связные компоненты,
       соединённые с seed

    Это НЕ замена нормальной сегментации.
    Нужно только чтобы pipeline можно было
    запустить до обучения модели.
    """

    # =========================================================
    # 1. INPUT
    # =========================================================

    height, width = image.shape[:2]

    if truck_mask is None:

        truck_mask = np.ones(
            (height, width),
            dtype=np.uint8,
        )

    else:

        truck_mask = (
            truck_mask > 0
        ).astype(np.uint8)

    # =========================================================
    # 2. FLOOR SEED
    # =========================================================

    # Предполагаем, что в нижней центральной
    # части изображения часто находится пол.

    x1 = int(
        width * 0.30
    )

    x2 = int(
        width * 0.70
    )

    y1 = int(
        height * 0.78
    )

    y2 = int(
        height * 0.98
    )

    seed_mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    seed_mask[
        y1:y2,
        x1:x2,
    ] = 1

    # Seed должен находиться только
    # внутри кузова
    seed_mask &= truck_mask

    if (
        np.count_nonzero(
            seed_mask
        )
        == 0
    ):
        return np.zeros(
            (height, width),
            dtype=np.uint8,
        )

    # =========================================================
    # 3. LAB COLOR SPACE
    # =========================================================

    # LAB лучше RGB для сравнения цвета,
    # потому что расстояние между цветами
    # более осмысленное.

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB,
    ).astype(
        np.float32
    )

    seed_pixels = lab[
        seed_mask.astype(bool)
    ]

    seed_mean = (
        seed_pixels.mean(axis=0)
    )

    # Цветовое расстояние каждого пикселя
    # до среднего цвета пола
    color_distance = np.linalg.norm(
        lab
        - seed_mean[
            None,
            None,
            :
        ],
        axis=2,
    )

    # =========================================================
    # 4. TEXTURE
    # =========================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    # Laplacian как грубая оценка текстуры
    gradient = cv2.Laplacian(
        blurred,
        cv2.CV_32F,
    )

    texture = np.abs(
        gradient
    )

    valid_pixels = (
        truck_mask.astype(bool)
    )

    # =========================================================
    # 5. THRESHOLDS
    # =========================================================

    color_threshold = np.percentile(
        color_distance[
            valid_pixels
        ],
        45,
    )

    texture_threshold = np.percentile(
        texture[
            valid_pixels
        ],
        70,
    )

    # =========================================================
    # 6. FLOOR CANDIDATES
    # =========================================================

    floor_mask = (

        (
            color_distance
            < color_threshold
        )

        &

        (
            texture
            < texture_threshold
        )

        &

        (
            truck_mask > 0
        )

    ).astype(
        np.uint8
    )

    # =========================================================
    # 7. MORPHOLOGY
    # =========================================================

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8,
    )

    # Удаляем мелкий шум
    floor_mask = cv2.morphologyEx(
        floor_mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    # Закрываем небольшие дырки
    floor_mask = cv2.morphologyEx(
        floor_mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # =========================================================
    # 8. CONNECTED COMPONENTS
    # =========================================================

    number_of_labels, labels, _, _ = (
        cv2.connectedComponentsWithStats(
            floor_mask,
            connectivity=8,
        )
    )

    # Какие компоненты пересекаются
    # с нашей seed-зоной
    seed_labels = np.unique(
        labels[
            seed_mask.astype(bool)
        ]
    )

    # 0 = background
    seed_labels = seed_labels[
        seed_labels > 0
    ]

    result = np.zeros_like(
        floor_mask
    )

    # Оставляем только области,
    # соединённые с нижним seed.
    for label in seed_labels:

        result[
            labels == label
        ] = 1

    # На всякий случай снова ограничиваем кузовом
    result &= truck_mask

    return result.astype(
        np.uint8
    )


def predict_floor_mask(
    image: np.ndarray,
    truck_mask: np.ndarray | None = None,
    threshold: float = 0.5,
    weights_path: str | Path | None = None,
) -> np.ndarray:
    """
    Главная функция inference.

    Вход:
        image:
            RGB uint8
            shape [H, W, 3]

        truck_mask:
            бинарная маска кузова
            shape [H, W]

        threshold:
            порог вероятности пола

    Выход:
        floor_mask:
            бинарная np.ndarray
            shape [H, W]

            1 = свободный пол
            0 = всё остальное
    """

    # =========================================================
    # 1. VALIDATION
    # =========================================================

    if (
        image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError(
            "Expected RGB image with shape HxWx3"
        )

    # =========================================================
    # 2. LOAD MODEL
    # =========================================================

    model, device = _load_model(
        weights_path=weights_path
    )

    # =========================================================
    # 3. FALLBACK
    # =========================================================

    # Пока модель ещё не обучена,
    # используем временную эвристику.

    if model is None:

        return heuristic_floor_mask(
            image=image,
            truck_mask=truck_mask,
        )

    # =========================================================
    # 4. TORCH
    # =========================================================

    import torch

    original_height, original_width = (
        image.shape[:2]
    )

    # =========================================================
    # 5. RESIZE
    # =========================================================

    resized = cv2.resize(
        image,
        (
            DEFAULT_SIZE,
            DEFAULT_SIZE,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    # =========================================================
    # 6. NORMALIZATION
    # =========================================================

    tensor = (
        resized.astype(
            np.float32
        )
        / 255.0
    )
    tensor = (
        tensor
        - IMAGENET_MEAN
    ) / IMAGENET_STD
    # HWC -> CHW
    tensor = np.transpose(
        tensor,
        (2, 0, 1),
    )

    # Добавляем batch dimension
    #
    # [3, 320, 320]
    # ->
    # [1, 3, 320, 320]

    tensor = tensor[
        None,
        :,
        :,
        :
    ]

    tensor = torch.from_numpy(
        tensor
    )

    tensor = tensor.to(
        device
    )

    # =========================================================
    # 7. INFERENCE
    # =========================================================

    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
        )

    probabilities = (
        probabilities[
            0,
            0,
        ]
        .detach()
        .cpu()
        .numpy()
    )

    # =========================================================
    # 8. RESIZE BACK
    # =========================================================

    probabilities = cv2.resize(
        probabilities,
        (
            original_width,
            original_height,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    # =========================================================
    # 9. THRESHOLD
    # =========================================================

    floor_mask = (
        probabilities
        >= threshold
    ).astype(
        np.uint8
    )

    # =========================================================
    # 10. TRUCK MASK
    # =========================================================

    # Пол физически не может находиться
    # за пределами кузова.

    if truck_mask is not None:

        truck_mask_binary = (
            truck_mask > 0
        ).astype(
            np.uint8
        )

        floor_mask &= (
            truck_mask_binary
        )

    return floor_mask.astype(
        np.uint8
    )
