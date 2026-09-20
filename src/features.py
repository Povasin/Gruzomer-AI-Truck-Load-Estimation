from pathlib import Path

import cv2
import numpy as np

from segmentation.runtime import predict_truck_mask
from floor_segmentation.runtime import predict_floor_mask


BLUE_HSV_LOWER = np.array([95, 50, 25], dtype=np.uint8)
BLUE_HSV_UPPER = np.array([135, 255, 255], dtype=np.uint8)
ROI_SIZE = 320
DETECTION_MAX_SIDE = 640

SMALL_SIZE = 32
FLOOR_PROFILE_BINS = 8
FLOOR_FEATURE_DIM = 42
BASELINE_FEATURE_DIM = 235
TOTAL_FEATURE_DIM = BASELINE_FEATURE_DIM + FLOOR_FEATURE_DIM


def read_rgb_image(path):
    """Читаем изображение OpenCV и возвращаем массив RGB с ориентацией EXIF."""
    bgr_image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError(f"Could not decode image: {path}")
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


def resize_roi(image, mask, size=ROI_SIZE):
    resized_image = cv2.resize(
        image,
        (size, size),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_mask = cv2.resize(
        mask.astype(np.uint8),
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized_image, (resized_mask > 0).astype(np.uint8)


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
        return np.zeros(24, dtype=np.float64)

    histograms = []

    for channel in range(3):
        pixels = image[:, :, channel][valid]
        hist, _ = np.histogram(
            pixels,
            bins=8,
            range=(0, 1),
        )
        hist = hist.astype(np.float64) / valid_count
        histograms.append(hist)

    return np.concatenate(histograms)


def blue_ratio_from_rgb_masked(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    blue_mask = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)
    valid = mask > 0
    denominator = np.count_nonzero(valid)

    if denominator == 0:
        return 0.0

    blue_inside = (blue_mask > 0) & valid
    return float(np.count_nonzero(blue_inside) / denominator)


def safe_ratio(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def clean_floor_mask(floor_mask, truck_mask):
    floor_mask = (floor_mask > 0).astype(np.uint8)
    truck_mask = (truck_mask > 0).astype(np.uint8)
    floor_mask &= truck_mask

    kernel = np.ones((5, 5), dtype=np.uint8)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, kernel)
    floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)
    floor_mask &= truck_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(floor_mask, connectivity=8)
    if num_labels <= 1:
        return floor_mask

    truck_area = int(np.count_nonzero(truck_mask))
    min_area = max(32, int(truck_area * 0.002))
    cleaned = np.zeros_like(floor_mask)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 1

    return cleaned.astype(np.uint8)


def lower75_slice(height):
    start = int(round(height * 0.25))
    return slice(start, height)


def floor_ratio_in_region(floor_mask, truck_mask, row_slice):
    floor_region = floor_mask[row_slice]
    truck_region = truck_mask[row_slice]
    return safe_ratio(
        np.count_nonzero(floor_region),
        np.count_nonzero(truck_region),
    )


def iter_depth_bin_slices(height, bins=FLOOR_PROFILE_BINS):
    region = np.arange(int(round(height * 0.25)), height)
    split = np.array_split(region, bins)

    slices = []
    for rows in split:
        if len(rows) == 0:
            slices.append(slice(height, height))
        else:
            slices.append(slice(int(rows[0]), int(rows[-1]) + 1))
    return slices


def depth_profile_features(floor_mask, truck_mask):
    h, _ = floor_mask.shape
    depth_slices = iter_depth_bin_slices(h)

    ratios = []
    width_means = []

    for row_slice in reversed(depth_slices):
        floor_bin = floor_mask[row_slice]
        truck_bin = truck_mask[row_slice]
        ratios.append(
            safe_ratio(
                np.count_nonzero(floor_bin),
                np.count_nonzero(truck_bin),
            )
        )

        row_widths = []
        for floor_row, truck_row in zip(floor_bin, truck_bin):
            truck_count = np.count_nonzero(truck_row)
            if truck_count == 0:
                continue
            row_widths.append(np.count_nonzero(floor_row) / truck_count)

        if len(row_widths) == 0:
            width_means.append(0.0)
        else:
            width_means.append(float(np.mean(row_widths)))

    groups = np.array_split(np.arange(FLOOR_PROFILE_BINS), 3)
    near_mid_far_ratios = [float(np.mean(np.take(ratios, group))) if len(group) > 0 else 0.0 for group in groups]
    near_mid_far_widths = [float(np.mean(np.take(width_means, group))) if len(group) > 0 else 0.0 for group in groups]

    return (
        np.asarray(ratios, dtype=np.float64),
        np.asarray(near_mid_far_ratios, dtype=np.float64),
        np.asarray(width_means, dtype=np.float64),
        np.asarray(near_mid_far_widths, dtype=np.float64),
    )


def perspective_weighted_floor_ratio(floor_mask, truck_mask):
    h, _ = floor_mask.shape
    weights = np.linspace(4.0, 1.0, h, dtype=np.float64)[:, None]
    weighted_floor = np.sum((floor_mask > 0) * (truck_mask > 0) * weights)
    weighted_truck = np.sum((truck_mask > 0) * weights)
    return safe_ratio(weighted_floor, weighted_truck)


def bottom_connected_depth(floor_mask, truck_mask, x_slice):
    floor_band = (floor_mask[:, x_slice] > 0).astype(np.uint8)
    truck_band = (truck_mask[:, x_slice] > 0).astype(np.uint8)
    floor_band &= truck_band

    if np.count_nonzero(floor_band) == 0:
        return 0.0

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(floor_band, connectivity=8)
    if num_labels <= 1:
        return 0.0

    bottom_indices = np.where(floor_band[-1] > 0)[0]
    if len(bottom_indices) == 0:
        search_rows = floor_band[max(0, floor_band.shape[0] - 10):]
        ys, xs = np.where(search_rows > 0)
        if len(xs) == 0:
            return 0.0
        bottom_indices = xs
        bottom_labels = np.unique(labels[max(0, floor_band.shape[0] - 10) + ys, xs])
    else:
        bottom_labels = np.unique(labels[-1, bottom_indices])

    bottom_labels = bottom_labels[bottom_labels > 0]
    if len(bottom_labels) == 0:
        return 0.0

    mask = np.isin(labels, bottom_labels)
    ys = np.where(mask)[0]
    if len(ys) == 0:
        return 0.0

    top_y = ys.min()
    return float((floor_band.shape[0] - top_y) / floor_band.shape[0])


def free_depth_features(floor_mask, truck_mask):
    _, w = floor_mask.shape
    x_ranges = np.array_split(np.arange(w), 3)
    depths = []

    for xs in x_ranges:
        if len(xs) == 0:
            depths.append(0.0)
            continue
        x_slice = slice(int(xs[0]), int(xs[-1]) + 1)
        depths.append(bottom_connected_depth(floor_mask, truck_mask, x_slice))

    depths = np.asarray(depths, dtype=np.float64)
    aggregated = np.asarray(
        [
            float(depths.mean()),
            float(depths.min()),
            float(depths.max()),
            float(depths.std()),
        ],
        dtype=np.float64,
    )
    return depths, aggregated


def connected_component_features(floor_mask, truck_mask):
    floor_mask = (floor_mask > 0).astype(np.uint8)
    truck_mask = (truck_mask > 0).astype(np.uint8)

    floor_area = int(np.count_nonzero(floor_mask))
    truck_area = int(np.count_nonzero(truck_mask))

    if floor_area == 0 or truck_area == 0:
        return np.zeros(4, dtype=np.float64)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(floor_mask, connectivity=8)
    component_areas = []
    for label in range(1, num_labels):
        component_areas.append(int(stats[label, cv2.CC_STAT_AREA]))

    if len(component_areas) == 0:
        return np.zeros(4, dtype=np.float64)

    largest = max(component_areas)
    fragmentation = 1.0 - safe_ratio(largest, floor_area)

    return np.asarray(
        [
            float(len(component_areas)),
            safe_ratio(largest, floor_area),
            safe_ratio(largest, truck_area),
            fragmentation,
        ],
        dtype=np.float64,
    )


def floor_geometry_features(floor_mask):
    ys, xs = np.where(floor_mask > 0)
    h, w = floor_mask.shape

    if len(xs) == 0 or len(ys) == 0:
        return np.zeros(6, dtype=np.float64)

    x_min = xs.min()
    x_max = xs.max() + 1
    y_min = ys.min()
    y_max = ys.max() + 1

    centroid_x = float(xs.mean() / w)
    centroid_y = float(ys.mean() / h)
    bbox_width = float((x_max - x_min) / w)
    bbox_height = float((y_max - y_min) / h)
    y_min_norm = float(y_min / h)
    y_max_norm = float(y_max / h)

    return np.asarray(
        [
            centroid_x,
            centroid_y,
            bbox_width,
            bbox_height,
            y_min_norm,
            y_max_norm,
        ],
        dtype=np.float64,
    )


def extract_floor_features(floor_mask, truck_mask):
    floor_mask = (floor_mask > 0).astype(np.uint8)
    truck_mask = (truck_mask > 0).astype(np.uint8)
    floor_mask = clean_floor_mask(floor_mask, truck_mask)

    global_floor_ratio = safe_ratio(
        np.count_nonzero(floor_mask),
        np.count_nonzero(truck_mask),
    )
    lower75_ratio = floor_ratio_in_region(
        floor_mask,
        truck_mask,
        lower75_slice(floor_mask.shape[0]),
    )
    weighted_ratio = perspective_weighted_floor_ratio(floor_mask, truck_mask)

    depth_ratios, near_mid_far_ratios, width_profile, near_mid_far_widths = depth_profile_features(
        floor_mask,
        truck_mask,
    )
    free_depths, free_depth_stats = free_depth_features(floor_mask, truck_mask)
    component_features = connected_component_features(floor_mask, truck_mask)
    geometry_features = floor_geometry_features(floor_mask)

    features = np.concatenate(
        [
            np.asarray([global_floor_ratio, lower75_ratio], dtype=np.float64),
            np.asarray([weighted_ratio], dtype=np.float64),
            depth_ratios,
            near_mid_far_ratios,
            width_profile,
            near_mid_far_widths,
            free_depths,
            free_depth_stats,
            component_features,
            geometry_features,
        ]
    )

    if len(features) != FLOOR_FEATURE_DIM:
        raise ValueError(
            f"Expected {FLOOR_FEATURE_DIM} floor features, got {len(features)}"
        )

    return features


def _features_from_truck_roi(
    roi,
    roi_mask,
    *,
    aspect,
    mask_area_ratio,
    floor_weights_path=None,
):
    """Извлекает числовые признаки из уже найденной области кузова."""
    floor_roi, floor_roi_mask = resize_roi(roi, roi_mask, size=ROI_SIZE)
    floor_mask = predict_floor_mask(
        floor_roi,
        truck_mask=floor_roi_mask,
        threshold=0.5,
        weights_path=floor_weights_path,
    )
    floor_features = extract_floor_features(
        floor_mask,
        floor_roi_mask,
    )

    small = cv2.resize(
        roi,
        (SMALL_SIZE, SMALL_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    small_mask = cv2.resize(
        roi_mask,
        (SMALL_SIZE, SMALL_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    small = small.astype(np.float64) / 255.0

    spatial = masked_spatial_features(small, small_mask)
    texture = masked_texture_features(small, small_mask)
    hist = masked_histogram(small, small_mask)
    blue_ratio = blue_ratio_from_rgb_masked(roi, roi_mask)

    result = np.concatenate(
        (
            spatial,          # 192
            texture,          # 16
            hist,             # 24
            [
                aspect,       # 1
                blue_ratio,   # 1
                mask_area_ratio,  # 1
            ],
            floor_features,   # 42
        )
    )

    if len(result) != TOTAL_FEATURE_DIM:
        raise ValueError(
            f"Expected {TOTAL_FEATURE_DIM} total features, got {len(result)}"
        )

    return result


def extract_hybrid_sample(path, *, truck_weights_path=None, floor_weights_path=None):
    """Возвращает признаки и тот же замаскированный ROI для CNN.

    Маска кузова вычисляется один раз. Числовая ветка использует маску при
    расчёте признаков, а CNN получает этот же bbox с нулями за пределами
    кузова. Поэтому фон вне кузова не становится входом CNN.
    """
    image = read_rgb_image(path)
    height, width = image.shape[:2]
    aspect = width / height
    mask = predict_truck_mask(image, threshold=0.5, weights_path=truck_weights_path)
    mask = (mask > 0).astype(np.uint8)
    roi, roi_mask = crop_by_mask(image, mask)
    numeric = _features_from_truck_roi(
        roi,
        roi_mask,
        aspect=aspect,
        mask_area_ratio=float(mask.mean()),
        floor_weights_path=floor_weights_path,
    )
    cnn_roi = roi.copy()
    cnn_roi[roi_mask == 0] = 0
    return numeric, cnn_roi


def feature(path, *, truck_weights_path=None, floor_weights_path=None):
    """Совместимый интерфейс прежней ветки: возвращает только 277 признаков."""
    features, _ = extract_hybrid_sample(
        path,
        truck_weights_path=truck_weights_path,
        floor_weights_path=floor_weights_path,
    )

    return features


def features(image_dir, rows):
    """Извлекаем признаки для изображений в порядке строк CSV."""
    image_dir = Path(image_dir)
    result = []

    for index, row in enumerate(rows):
        image_path = image_dir / f"{row['image_id']}.jpg"

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        result.append(feature(image_path))

        if (index + 1) % 200 == 0:
            print(f"Features: {index + 1}/{len(rows)}", flush=True)

    return np.asarray(result, dtype=np.float64)
