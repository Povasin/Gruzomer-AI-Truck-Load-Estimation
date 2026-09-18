from pathlib import Path
from segmentation.runtime import predict_truck_mask
import cv2
import numpy as np


BLUE_HSV_LOWER = np.array([95, 50, 25], dtype=np.uint8)
BLUE_HSV_UPPER = np.array([135, 255, 255], dtype=np.uint8)
ROI_SIZE = 320
DETECTION_MAX_SIDE = 640


def read_rgb_image(path):
    """Читаем изображение OpenCV и возвращаем массив RGB с ориентацией EXIF."""
    # IMREAD_COLOR декодирует JPEG в три 8-битных канала BGR. По умолчанию
    # OpenCV также применяет EXIF Orientation, поэтому портретные снимки
    # приводятся к фактической ориентации ещё до вычисления признаков.
    bgr_image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    # OpenCV сигнализирует о повреждённом или неподдерживаемом файле значением
    # None. Явная ошибка сразу указывает на проблемное изображение.
    if bgr_image is None:
        raise ValueError(f"Could not decode image: {path}")

    # Модель опирается на порядок RGB, поэтому меняем порядок каналов BGR,
    # используемый OpenCV, до нормализации и расчёта статистик.
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def blue_ratio_from_rgb(image):
    """Возвращаем долю пикселей синего груза в RGB-изображении."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with three channels")

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    blue_mask = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)
    return float(np.count_nonzero(blue_mask) / blue_mask.size)

def crop_by_mask(image, mask):
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return image, np.ones(
            image.shape[:2],
            dtype=np.uint8,
        )

    x1 = xs.min()
    x2 = xs.max() + 1
    y1 = ys.min()
    y2 = ys.max() + 1

    return (
        image[y1:y2, x1:x2],
        mask[y1:y2, x1:x2],
    )


def masked_spatial_features(image, mask):
    features = []

    for row in range(8):
        for col in range(8):
            y1 = row * 4
            y2 = y1 + 4
            x1 = col * 4
            x2 = x1 + 4

            cell = image[y1:y2, x1:x2]
            cell_mask = mask[y1:y2, x1:x2].astype(bool)

            pixels = cell[cell_mask]

            if len(pixels) == 0:
                mean_rgb = np.zeros(3, dtype=np.float64)
            else:
                mean_rgb = pixels.mean(axis=0)

            features.extend(mean_rgb)

    return np.asarray(
        features,
        dtype=np.float64,
    )


def masked_texture_features(image, mask):
    gray = image.mean(axis=2)

    features = []

    for row in range(4):
        for col in range(4):
            y1 = row * 8
            y2 = y1 + 8
            x1 = col * 8
            x2 = x1 + 8

            cell = gray[y1:y2, x1:x2]
            cell_mask = mask[y1:y2, x1:x2].astype(bool)

            pixels = cell[cell_mask]

            if len(pixels) == 0:
                std = 0.0
            else:
                std = float(pixels.std())

            features.append(std)

    return np.asarray(
        features,
        dtype=np.float64,
    )


def masked_histogram(image, mask):
    valid = mask.astype(bool)

    valid_count = np.count_nonzero(valid)

    if valid_count == 0:
        return np.zeros(
            24,
            dtype=np.float64,
        )

    histograms = []

    for channel in range(3):
        pixels = image[:, :, channel][valid]

        hist, _ = np.histogram(
            pixels,
            bins=8,
            range=(0, 1),
        )

        hist = (
            hist.astype(np.float64)
            / valid_count
        )

        histograms.append(hist)

    return np.concatenate(histograms)


def blue_ratio_from_rgb_masked(image, mask):
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2HSV,
    )

    blue_mask = cv2.inRange(
        hsv,
        BLUE_HSV_LOWER,
        BLUE_HSV_UPPER,
    )

    valid = mask > 0

    denominator = np.count_nonzero(valid)

    if denominator == 0:
        return 0.0

    blue_inside = (
        (blue_mask > 0)
        & valid
    )

    return float(
        np.count_nonzero(blue_inside)
        / denominator
    )

def feature(path):

    # =========================================================
    # 1. IMAGE
    # =========================================================

    image = read_rgb_image(path)

    height, width = image.shape[:2]

    aspect = width / height


    # =========================================================
    # 2. SEGMENTATION
    # =========================================================

    mask = predict_truck_mask(
        image,
        threshold=0.5,
    )


    # =========================================================
    # 3. MASK FEATURE
    # =========================================================

    mask_area_ratio = float(
        mask.mean()
    )


    # =========================================================
    # 4. CROP
    # =========================================================

    roi, roi_mask = crop_by_mask(
        image,
        mask,
    )


    # =========================================================
    # 5. 32x32
    # =========================================================

    small = cv2.resize(
        roi,
        (32, 32),
        interpolation=cv2.INTER_LINEAR,
    )

    small_mask = cv2.resize(
        roi_mask,
        (32, 32),
        interpolation=cv2.INTER_NEAREST,
    )

    small = (
        small.astype(np.float64)
        / 255.0
    )


    # =========================================================
    # 6. SPATIAL
    # =========================================================

    spatial = masked_spatial_features(
        small,
        small_mask,
    )


    # =========================================================
    # 7. TEXTURE
    # =========================================================

    texture = masked_texture_features(
        small,
        small_mask,
    )


    # =========================================================
    # 8. HISTOGRAM
    # =========================================================

    hist = masked_histogram(
        small,
        small_mask,
    )


    # =========================================================
    # 9. BLUE RATIO
    # =========================================================

    blue_ratio = (
        blue_ratio_from_rgb_masked(
            roi,
            roi_mask,
        )
    )


    # =========================================================
    # 10. RESULT
    # =========================================================

    return np.concatenate(
        (
            spatial,          # 192
            texture,          # 16
            hist,             # 24
            [
                aspect,       # 1
                blue_ratio,   # 1
                mask_area_ratio,  # 1
            ]
        )
    )


def features(image_dir, rows):
    """Извлекаем признаки для изображений в порядке строк CSV."""
    image_dir = Path(image_dir)
    result = []

    for index, row in enumerate(rows):
        # image_id в CSV не содержит расширения; путь строится без изменения
        # порядка строк, чтобы матрица признаков совпадала с разметкой и test.csv.
        image_path = image_dir / f"{row['image_id']}.jpg"

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Каждый вектор имеет фиксированную длину 235 и добавляется в том же
        # порядке, в котором изображение указано во входной таблице.
        result.append(feature(image_path))

        if (index + 1) % 200 == 0:
            print(f"Features: {index + 1}/{len(rows)}", flush=True)

    return np.asarray(result, dtype=np.float64)