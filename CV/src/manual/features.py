from pathlib import Path

import cv2
import numpy as np

from manual.feature_schema import (
    BASELINE_FEATURE_DIM,
    FLOOR_FEATURE_DIM,
    CEILING_FEATURE_DIM,
    CROSS_FEATURE_DIM,
    TOTAL_FEATURE_DIM,
)
from manual.truck_segmentation.runtime import predict_truck_mask
from manual.floor_segmentation.runtime import predict_floor_mask
from manual.roof_segmentation.runtime import predict_ceiling_mask


BLUE_HSV_LOWER = np.array([95, 50, 25], dtype=np.uint8)
BLUE_HSV_UPPER = np.array([135, 255, 255], dtype=np.uint8)
ROI_SIZE = 320
DETECTION_MAX_SIDE = 640

SMALL_SIZE = 32
FLOOR_PROFILE_BINS = 8
CEILING_PROFILE_BINS = 8


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


def _edge_connected_depth(surface_mask, truck_mask, x_slice, *, from_top):
    """Максимальная относительная глубина компоненты от границы кузова.

    Верх/низ и высота кузова определяются отдельно в каждом столбце.
    Компонента должна пересекать первые/последние 10 строк локального
    кузова (допуск на погрешность сегментации), а не границу bbox ROI.
    Это доля высоты изображения внутри кузова, не метрическая глубина.
    """
    truck = truck_mask[:, x_slice] > 0
    surface = (surface_mask[:, x_slice] > 0) & truck
    if not surface.any():
        return 0.0

    height = truck.shape[0]
    top = truck.argmax(axis=0)
    bottom = height - 1 - truck[::-1].argmax(axis=0)
    rows = np.arange(height)[:, None]
    distance = rows - top if from_top else bottom - rows
    edge = truck & (distance >= 0) & (distance < 10)

    _, labels = cv2.connectedComponents(surface.astype(np.uint8), connectivity=8)
    touching = np.unique(labels[edge & surface])
    if len(touching) == 0:
        return 0.0
    selected = np.isin(labels, touching)
    valid_columns = selected.any(axis=0)
    if from_top:
        farthest = height - 1 - selected[::-1].argmax(axis=0)
        depth = farthest - top + 1
    else:
        farthest = selected.argmax(axis=0)
        depth = bottom - farthest + 1
    local_height = bottom - top + 1
    return float(np.max(depth[valid_columns] / local_height[valid_columns]))


def bottom_connected_depth(floor_mask, truck_mask, x_slice):
    return _edge_connected_depth(floor_mask, truck_mask, x_slice, from_top=False)


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



# ============================================================
# CEILING FEATURES: 42
# ============================================================


def clean_ceiling_mask(ceiling_mask, truck_mask):
    """Очищаем ceiling-mask и ограничиваем её маской кузова."""
    ceiling_mask = (ceiling_mask > 0).astype(np.uint8)
    truck_mask = (truck_mask > 0).astype(np.uint8)
    ceiling_mask &= truck_mask

    kernel = np.ones((5, 5), dtype=np.uint8)
    ceiling_mask = cv2.morphologyEx(ceiling_mask, cv2.MORPH_OPEN, kernel)
    ceiling_mask = cv2.morphologyEx(ceiling_mask, cv2.MORPH_CLOSE, kernel)
    ceiling_mask &= truck_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        ceiling_mask,
        connectivity=8,
    )
    if num_labels <= 1:
        return ceiling_mask

    truck_area = int(np.count_nonzero(truck_mask))
    min_area = max(32, int(truck_area * 0.002))
    cleaned = np.zeros_like(ceiling_mask)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned[labels == label] = 1

    return cleaned.astype(np.uint8)


def upper75_slice(height):
    end = int(round(height * 0.75))
    return slice(0, max(end, 1))


def ceiling_ratio_in_region(ceiling_mask, truck_mask, row_slice):
    ceiling_region = ceiling_mask[row_slice]
    truck_region = truck_mask[row_slice]
    return safe_ratio(
        np.count_nonzero(ceiling_region),
        np.count_nonzero(truck_region),
    )


def iter_ceiling_depth_bin_slices(height, bins=CEILING_PROFILE_BINS):
    """
    Near -> far для ceiling идёт сверху вниз.
    Это зеркально floor-профилю, где near -> far идёт снизу вверх.
    """
    end = max(1, int(round(height * 0.75)))
    region = np.arange(0, end)
    split = np.array_split(region, bins)

    result = []
    for rows in split:
        if len(rows) == 0:
            result.append(slice(0, 0))
        else:
            result.append(slice(int(rows[0]), int(rows[-1]) + 1))
    return result


def ceiling_depth_profile_features(ceiling_mask, truck_mask):
    h, _ = ceiling_mask.shape
    depth_slices = iter_ceiling_depth_bin_slices(h)

    ratios = []
    width_means = []

    for row_slice in depth_slices:
        ceiling_bin = ceiling_mask[row_slice]
        truck_bin = truck_mask[row_slice]

        ratios.append(
            safe_ratio(
                np.count_nonzero(ceiling_bin),
                np.count_nonzero(truck_bin),
            )
        )

        row_widths = []
        for ceiling_row, truck_row in zip(ceiling_bin, truck_bin):
            truck_count = np.count_nonzero(truck_row)
            if truck_count == 0:
                continue
            row_widths.append(
                np.count_nonzero(ceiling_row) / truck_count
            )

        width_means.append(
            float(np.mean(row_widths)) if row_widths else 0.0
        )

    groups = np.array_split(np.arange(CEILING_PROFILE_BINS), 3)

    near_mid_far_ratios = [
        float(np.mean(np.take(ratios, group))) if len(group) > 0 else 0.0
        for group in groups
    ]
    near_mid_far_widths = [
        float(np.mean(np.take(width_means, group))) if len(group) > 0 else 0.0
        for group in groups
    ]

    return (
        np.asarray(ratios, dtype=np.float64),
        np.asarray(near_mid_far_ratios, dtype=np.float64),
        np.asarray(width_means, dtype=np.float64),
        np.asarray(near_mid_far_widths, dtype=np.float64),
    )


def perspective_weighted_ceiling_ratio(ceiling_mask, truck_mask):
    h, _ = ceiling_mask.shape
    weights = np.linspace(1.0, 4.0, h, dtype=np.float64)[:, None]
    weighted_ceiling = np.sum(
        (ceiling_mask > 0) * (truck_mask > 0) * weights
    )
    weighted_truck = np.sum((truck_mask > 0) * weights)
    return safe_ratio(weighted_ceiling, weighted_truck)


def top_connected_depth(ceiling_mask, truck_mask, x_slice):
    return _edge_connected_depth(ceiling_mask, truck_mask, x_slice, from_top=True)


def ceiling_depth_features(ceiling_mask, truck_mask):
    _, w = ceiling_mask.shape
    x_ranges = np.array_split(np.arange(w), 3)
    depths = []

    for xs in x_ranges:
        if len(xs) == 0:
            depths.append(0.0)
            continue

        x_slice = slice(int(xs[0]), int(xs[-1]) + 1)
        depths.append(
            top_connected_depth(
                ceiling_mask,
                truck_mask,
                x_slice,
            )
        )

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


def ceiling_geometry_features(ceiling_mask):
    ys, xs = np.where(ceiling_mask > 0)
    h, w = ceiling_mask.shape

    if len(xs) == 0 or len(ys) == 0:
        return np.zeros(6, dtype=np.float64)

    x_min = xs.min()
    x_max = xs.max() + 1
    y_min = ys.min()
    y_max = ys.max() + 1

    return np.asarray(
        [
            float(xs.mean() / w),
            float(ys.mean() / h),
            float((x_max - x_min) / w),
            float((y_max - y_min) / h),
            float(y_min / h),
            float(y_max / h),
        ],
        dtype=np.float64,
    )


def extract_ceiling_features(ceiling_mask, truck_mask):
    """
    42 ceiling features:
      3 global/weighted
      8 depth bins
      3 near/mid/far depth
      8 width bins
      3 near/mid/far width
      3 left/center/right depth
      4 depth stats
      4 connected-component
      6 geometry
    """
    ceiling_mask = (ceiling_mask > 0).astype(np.uint8)
    truck_mask = (truck_mask > 0).astype(np.uint8)
    ceiling_mask = clean_ceiling_mask(ceiling_mask, truck_mask)

    global_ceiling_ratio = safe_ratio(
        np.count_nonzero(ceiling_mask),
        np.count_nonzero(truck_mask),
    )

    upper75_ratio = ceiling_ratio_in_region(
        ceiling_mask,
        truck_mask,
        upper75_slice(ceiling_mask.shape[0]),
    )

    weighted_ratio = perspective_weighted_ceiling_ratio(
        ceiling_mask,
        truck_mask,
    )

    (
        depth_ratios,
        near_mid_far_ratios,
        width_profile,
        near_mid_far_widths,
    ) = ceiling_depth_profile_features(
        ceiling_mask,
        truck_mask,
    )

    ceiling_depths, ceiling_depth_stats = ceiling_depth_features(
        ceiling_mask,
        truck_mask,
    )

    component_features = connected_component_features(
        ceiling_mask,
        truck_mask,
    )

    geometry_features = ceiling_geometry_features(ceiling_mask)

    features = np.concatenate(
        [
            np.asarray(
                [global_ceiling_ratio, upper75_ratio],
                dtype=np.float64,
            ),
            np.asarray([weighted_ratio], dtype=np.float64),
            depth_ratios,
            near_mid_far_ratios,
            width_profile,
            near_mid_far_widths,
            ceiling_depths,
            ceiling_depth_stats,
            component_features,
            geometry_features,
        ]
    )

    if len(features) != CEILING_FEATURE_DIM:
        raise ValueError(
            f"Expected {CEILING_FEATURE_DIM} ceiling features, got {len(features)}"
        )

    return features


# ============================================================
# FLOOR + CEILING CROSS FEATURES: 30
# ============================================================


def _mean_valid(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    return float(values.mean())


def _std_valid(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    return float(values.std())


def _boundary_profiles(floor_mask, ceiling_mask, truck_mask):
    floor_mask = (floor_mask > 0) & (truck_mask > 0)
    ceiling_mask = (ceiling_mask > 0) & (truck_mask > 0)

    h, w = floor_mask.shape

    floor_top = np.full(w, np.nan, dtype=np.float64)
    ceiling_bottom = np.full(w, np.nan, dtype=np.float64)

    for x in range(w):
        floor_y = np.where(floor_mask[:, x])[0]
        if len(floor_y) > 0:
            floor_top[x] = float(floor_y.min() / h)

        ceiling_y = np.where(ceiling_mask[:, x])[0]
        if len(ceiling_y) > 0:
            ceiling_bottom[x] = float(ceiling_y.max() / h)

    valid = np.isfinite(floor_top) & np.isfinite(ceiling_bottom)

    gap = np.full(w, np.nan, dtype=np.float64)
    gap[valid] = np.clip(
        floor_top[valid] - ceiling_bottom[valid],
        0.0,
        1.0,
    )

    return floor_top, ceiling_bottom, gap


def _three_band_means(values):
    values = np.asarray(values, dtype=np.float64)
    x_groups = np.array_split(np.arange(len(values)), 3)
    result = []

    for xs in x_groups:
        if len(xs) == 0:
            result.append(0.0)
        else:
            result.append(_mean_valid(values[xs]))

    return np.asarray(result, dtype=np.float64)


def _safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if len(a) != len(b) or len(a) < 2:
        return 0.0

    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0

    value = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(value):
        return 0.0

    return float(value)


def _deepest_visible_bin(joint_profile, threshold):
    joint_profile = np.asarray(joint_profile, dtype=np.float64)
    valid = np.where(joint_profile >= threshold)[0]

    if len(valid) == 0:
        return 0.0

    return float((int(valid.max()) + 1) / len(joint_profile))


def _shell_visibility_ratio(floor_mask, ceiling_mask, truck_mask, x_slice):
    truck_region = truck_mask[:, x_slice] > 0
    denominator = np.count_nonzero(truck_region)

    if denominator == 0:
        return 0.0

    shell = (
        ((floor_mask[:, x_slice] > 0) | (ceiling_mask[:, x_slice] > 0))
        & truck_region
    )
    return float(np.count_nonzero(shell) / denominator)


def extract_cross_features(floor_mask, ceiling_mask, truck_mask):
    """
    РОВНО 30 Tier-1 cross features в согласованной ROI.

    Порядок:
      1-5   gap_left, gap_center, gap_right, gap_mean, gap_std
      6-8   joint_depth_min_left/center/right
      9-16  joint_min_depth_bin_0..7
      17-19 deepest_both_visible_10/25/50
      20    floor_ceiling_depth_corr
      21    floor_area / (floor_area + ceiling_area)
      22    central_shell_visible_ratio
      23    far_shell_visible_ratio
      24-27 cargo_front_center/left/right/std
      28    ceiling_boundary_std
      29    ceiling_left_right_asymmetry
      30    central_joint_visibility
    """
    truck_mask = (truck_mask > 0).astype(np.uint8)

    floor_mask = clean_floor_mask(
        (floor_mask > 0).astype(np.uint8),
        truck_mask,
    )
    ceiling_mask = clean_ceiling_mask(
        (ceiling_mask > 0).astype(np.uint8),
        truck_mask,
    )

    # 1-5
    _, ceiling_bottom, gap_profile = _boundary_profiles(
        floor_mask,
        ceiling_mask,
        truck_mask,
    )

    gap_bands = _three_band_means(gap_profile)

    gap_left = float(gap_bands[0])
    gap_center = float(gap_bands[1])
    gap_right = float(gap_bands[2])
    gap_mean = _mean_valid(gap_profile)
    gap_std = _std_valid(gap_profile)

    # 6-8
    floor_depths, _ = free_depth_features(
        floor_mask,
        truck_mask,
    )
    ceiling_depths, _ = ceiling_depth_features(
        ceiling_mask,
        truck_mask,
    )
    joint_depth_min = np.minimum(
        floor_depths,
        ceiling_depths,
    )

    # 9-16
    floor_profile, _, _, _ = depth_profile_features(
        floor_mask,
        truck_mask,
    )
    ceiling_profile, _, _, _ = ceiling_depth_profile_features(
        ceiling_mask,
        truck_mask,
    )
    joint_min_bins = np.minimum(
        floor_profile,
        ceiling_profile,
    )

    # 17-19
    deepest_10 = _deepest_visible_bin(joint_min_bins, 0.10)
    deepest_25 = _deepest_visible_bin(joint_min_bins, 0.25)
    deepest_50 = _deepest_visible_bin(joint_min_bins, 0.50)

    # 20
    depth_corr = _safe_corr(
        floor_profile,
        ceiling_profile,
    )

    # 21
    floor_area = int(np.count_nonzero(floor_mask))
    ceiling_area = int(np.count_nonzero(ceiling_mask))
    floor_share = safe_ratio(
        floor_area,
        floor_area + ceiling_area,
    )

    # 22
    _, w = truck_mask.shape
    center_x1 = int(round(w * 0.25))
    center_x2 = int(round(w * 0.75))
    center_slice = slice(
        center_x1,
        max(center_x1 + 1, center_x2),
    )

    central_shell_visible_ratio = _shell_visibility_ratio(
        floor_mask,
        ceiling_mask,
        truck_mask,
        center_slice,
    )

    # 23
    far_start = max(0, len(floor_profile) - 3)
    far_shell_visible_ratio = float(
        np.mean(
            np.clip(
                (
                    floor_profile[far_start:]
                    + ceiling_profile[far_start:]
                ) / 2.0,
                0.0,
                1.0,
            )
        )
    )

    # 24-27
    cargo_front_bands = (
        floor_depths + ceiling_depths
    ) / 2.0

    cargo_front_left = float(cargo_front_bands[0])
    cargo_front_center = float(cargo_front_bands[1])
    cargo_front_right = float(cargo_front_bands[2])
    cargo_front_std = float(cargo_front_bands.std())

    # 28
    ceiling_boundary_std = _std_valid(
        ceiling_bottom
    )

    # 29
    mid = w // 2

    left_ceiling_ratio = safe_ratio(
        np.count_nonzero(ceiling_mask[:, :mid]),
        np.count_nonzero(truck_mask[:, :mid]),
    )
    right_ceiling_ratio = safe_ratio(
        np.count_nonzero(ceiling_mask[:, mid:]),
        np.count_nonzero(truck_mask[:, mid:]),
    )
    ceiling_left_right_asymmetry = float(
        abs(left_ceiling_ratio - right_ceiling_ratio)
    )

    # 30
    central_floor_depth = bottom_connected_depth(
        floor_mask,
        truck_mask,
        center_slice,
    )
    central_ceiling_depth = top_connected_depth(
        ceiling_mask,
        truck_mask,
        center_slice,
    )
    central_joint_visibility = float(
        min(central_floor_depth, central_ceiling_depth)
    )

    features = np.concatenate(
        [
            np.asarray(
                [
                    gap_left,
                    gap_center,
                    gap_right,
                    gap_mean,
                    gap_std,
                ],
                dtype=np.float64,
            ),
            joint_depth_min.astype(np.float64),
            joint_min_bins.astype(np.float64),
            np.asarray(
                [
                    deepest_10,
                    deepest_25,
                    deepest_50,
                    depth_corr,
                    floor_share,
                    central_shell_visible_ratio,
                    far_shell_visible_ratio,
                    cargo_front_center,
                    cargo_front_left,
                    cargo_front_right,
                    cargo_front_std,
                    ceiling_boundary_std,
                    ceiling_left_right_asymmetry,
                    central_joint_visibility,
                ],
                dtype=np.float64,
            ),
        ]
    )

    if len(features) != CROSS_FEATURE_DIM:
        raise ValueError(
            f"Expected {CROSS_FEATURE_DIM} cross features, got {len(features)}"
        )

    return features

def _features_from_truck_roi(
    roi,
    roi_mask,
    *,
    aspect,
    mask_area_ratio,
    floor_weights_path=None,
    ceiling_weights_path=None,
):
    """
    Извлекает 349 числовых признаков из одной и той же truck ROI:

        235 base
        + 42 floor
        + 42 ceiling
        + 30 cross
        = 349
    """
    segmentation_roi, segmentation_roi_mask = resize_roi(
        roi,
        roi_mask,
        size=ROI_SIZE,
    )

    # --------------------------------------------------------
    # FLOOR
    # --------------------------------------------------------
    floor_mask = predict_floor_mask(
        segmentation_roi,
        truck_mask=segmentation_roi_mask,
        threshold=0.5,
        weights_path=floor_weights_path,
    )

    floor_features = extract_floor_features(
        floor_mask,
        segmentation_roi_mask,
    )

    # --------------------------------------------------------
    # CEILING
    # --------------------------------------------------------
    # Runtime ceiling-модели должен использовать threshold,
    # сохранённый в best_iou checkpoint.
    ceiling_mask = predict_ceiling_mask(
        segmentation_roi,
        weights_path=ceiling_weights_path,
    )

    ceiling_features = extract_ceiling_features(
        ceiling_mask,
        segmentation_roi_mask,
    )

    # --------------------------------------------------------
    # FLOOR + CEILING CROSS
    # --------------------------------------------------------
    cross_features = extract_cross_features(
        floor_mask,
        ceiling_mask,
        segmentation_roi_mask,
    )

    # --------------------------------------------------------
    # LEGACY 235 BASE
    # --------------------------------------------------------
    # Одинаковые веса интерполяции для RGB * mask и mask исключают
    # примесь фона и затемнение на границе кузова.
    valid = (roi_mask > 0).astype(np.float64)
    small_weight = cv2.resize(
        valid,
        (SMALL_SIZE, SMALL_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    weighted_rgb = cv2.resize(
        roi.astype(np.float64) * valid[:, :, None],
        (SMALL_SIZE, SMALL_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )
    small_mask = small_weight > 0
    small = np.zeros_like(weighted_rgb)
    np.divide(
        weighted_rgb,
        small_weight[:, :, None],
        out=small,
        where=small_mask[:, :, None],
    )
    small = np.clip(small / 255.0, 0.0, 1.0)

    spatial = masked_spatial_features(small, small_mask)
    texture = masked_texture_features(small, small_mask)
    hist = masked_histogram(small, small_mask)
    blue_ratio = blue_ratio_from_rgb_masked(roi, roi_mask)

    base_features = np.concatenate(
        (
            spatial,              # 192
            texture,              # 16
            hist,                 # 24
            [
                aspect,           # 1
                blue_ratio,       # 1
                mask_area_ratio,  # 1
            ],
        )
    )

    if len(base_features) != BASELINE_FEATURE_DIM:
        raise ValueError(
            f"Expected {BASELINE_FEATURE_DIM} base features, got {len(base_features)}"
        )

    result = np.concatenate(
        (
            base_features,      #   0..234 = 235
            floor_features,     # 235..276 = 42
            ceiling_features,   # 277..318 = 42
            cross_features,     # 319..348 = 30
        )
    )

    if len(result) != TOTAL_FEATURE_DIM:
        raise ValueError(
            f"Expected {TOTAL_FEATURE_DIM} total features, got {len(result)}"
        )

    return result

def extract_hybrid_sample(
    path,
    *,
    truck_weights_path=None,
    floor_weights_path=None,
    ceiling_weights_path=None,
):
    """
    Возвращает:
      - numeric: 349 признаков
      - cnn_roi: тот же masked truck ROI для CNN

    Floor и ceiling считаются в одной ROI 320x320,
    поэтому cross-признаки геометрически согласованы.
    """
    image = read_rgb_image(path)
    height, width = image.shape[:2]
    aspect = width / height

    mask = predict_truck_mask(
        image,
        threshold=0.5,
        weights_path=truck_weights_path,
    )
    mask = (mask > 0).astype(np.uint8)

    roi, roi_mask = crop_by_mask(
        image,
        mask,
    )

    numeric = _features_from_truck_roi(
        roi,
        roi_mask,
        aspect=aspect,
        mask_area_ratio=float(mask.mean()),
        floor_weights_path=floor_weights_path,
        ceiling_weights_path=ceiling_weights_path,
    )

    cnn_roi = roi.copy()
    cnn_roi[roi_mask == 0] = 0

    return numeric, cnn_roi


def feature(
    path,
    *,
    truck_weights_path=None,
    floor_weights_path=None,
    ceiling_weights_path=None,
):
    """Возвращает полный вектор из 349 числовых признаков."""
    numeric, _ = extract_hybrid_sample(
        path,
        truck_weights_path=truck_weights_path,
        floor_weights_path=floor_weights_path,
        ceiling_weights_path=ceiling_weights_path,
    )
    return numeric

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
