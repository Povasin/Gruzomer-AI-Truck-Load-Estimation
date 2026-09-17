from pathlib import Path

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


def _resize_for_detection(image):
    """Уменьшаем кадр для устойчивой и ограниченной по времени геометрии."""
    height, width = image.shape[:2]
    scale = min(1.0, DETECTION_MAX_SIDE / max(height, width))
    if scale == 1.0:
        return image, scale

    resized = cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _line_coefficients(segment):
    """Строим коэффициенты прямой ax + by + c = 0 по отрезку Hough."""
    x1, y1, x2, y2 = (float(value) for value in segment)
    a = y1 - y2
    b = x2 - x1
    length = np.hypot(a, b)
    if length < 1e-8:
        return None
    return np.array([a / length, b / length, (x1 * y2 - x2 * y1) / length])


def _intersect_lines(first, second):
    """Находим пересечение двух бесконечных прямых или None для параллельных."""
    determinant = first[0] * second[1] - second[0] * first[1]
    if abs(determinant) < 1e-6:
        return None

    x = (first[1] * second[2] - second[1] * first[2]) / determinant
    y = (second[0] * first[2] - first[0] * second[2]) / determinant
    return np.array([x, y], dtype=np.float32)


def _angle_difference(first, second):
    """Возвращаем разницу направлений прямых без учёта направления отрезка."""
    return abs((first - second + np.pi / 2) % np.pi - np.pi / 2)


def _segment_position(segment, orientation, width, height):
    """Вычисляем положение и наклон отрезка относительно центра кадра."""
    x1, y1, x2, y2 = (float(value) for value in segment)
    if orientation == "vertical":
        if abs(y2 - y1) < 1e-8:
            return None
        position = x1 + (x2 - x1) * (height / 2 - y1) / (y2 - y1)
        angle = np.arctan2(x2 - x1, y2 - y1)
    else:
        if abs(x2 - x1) < 1e-8:
            return None
        position = y1 + (y2 - y1) * (width / 2 - x1) / (x2 - x1)
        angle = np.arctan2(y2 - y1, x2 - x1)

    angle = (angle + np.pi / 2) % np.pi - np.pi / 2
    return float(position), float(angle)


def _fit_line_cluster(cluster, orientation, width, height):
    """Аппроксимируем фрагменты одной границы общей бесконечной прямой."""
    points = np.asarray(cluster["points"], dtype=np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    coefficients = np.array(
        [vy, -vx, -vy * x0 + vx * y0],
        dtype=np.float64,
    )
    length = np.hypot(coefficients[0], coefficients[1])
    coefficients /= length

    if orientation == "vertical":
        position = -(coefficients[1] * height / 2 + coefficients[2]) / coefficients[0]
        extent = (float(points[:, 1].min()), float(points[:, 1].max()))
        reference_length = height
    else:
        position = -(coefficients[0] * width / 2 + coefficients[2]) / coefficients[1]
        extent = (float(points[:, 0].min()), float(points[:, 0].max()))
        reference_length = width

    strength = min(
        1.0,
        float(np.log1p(cluster["weight"] / reference_length) / np.log(6.0)),
    )
    return {
        "line": coefficients,
        "position": float(position),
        "angle": float(cluster["angle"]),
        "extent": extent,
        "strength": strength,
        "segments": cluster["segments"],
    }


def _merge_line_segments(segments, orientation, width, height):
    """Объединяем близкие по положению и наклону Hough-отрезки."""
    entries = []
    for segment in segments:
        geometry = _segment_position(segment, orientation, width, height)
        if geometry is None:
            continue
        position, angle = geometry
        length = float(np.hypot(segment[2] - segment[0], segment[3] - segment[1]))
        entries.append((length, segment, position, angle))

    entries.sort(key=lambda item: item[0], reverse=True)
    clusters = []
    position_gap = (width if orientation == "vertical" else height) * 0.04
    angle_gap = np.deg2rad(8)

    for length, segment, position, angle in entries:
        selected = None
        for cluster in clusters:
            if (
                abs(position - cluster["position"]) < position_gap
                and _angle_difference(angle, cluster["angle"]) < angle_gap
            ):
                selected = cluster
                break

        if selected is None:
            selected = {
                "position": position,
                "angle": angle,
                "weight": 0.0,
                "points": [],
                "segments": [],
            }
            clusters.append(selected)

        previous_weight = selected["weight"]
        selected["weight"] += length
        selected["position"] = (
            selected["position"] * previous_weight + position * length
        ) / selected["weight"]
        selected["angle"] = (
            selected["angle"] * previous_weight + angle * length
        ) / selected["weight"]
        selected["points"].extend(
            [[segment[0], segment[1]], [segment[2], segment[3]]]
        )
        selected["segments"].append(segment)

    fitted = [
        _fit_line_cluster(cluster, orientation, width, height)
        for cluster in clusters
        if len(cluster["segments"]) >= 2
    ]
    fitted.sort(key=lambda candidate: candidate["strength"], reverse=True)
    return fitted[:16]


def _line_candidates(edges):
    """Находим и объединяем фрагменты вертикальных и горизонтальных границ."""
    height, width = edges.shape
    min_length = max(25, round(min(height, width) * 0.06))
    segments = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=min_length,
        maxLineGap=max(10, round(min_length * 0.5)),
    )
    if segments is None:
        return [], []

    vertical_segments = []
    horizontal_segments = []
    for segment in segments.reshape(-1, 4):
        x1, y1, x2, y2 = (float(value) for value in segment)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        if dy > 0 and dx / dy <= 0.5:
            vertical_segments.append(segment)
        if dx > 0 and dy / dx <= 0.5:
            horizontal_segments.append(segment)

    vertical = _merge_line_segments(
        vertical_segments,
        "vertical",
        width,
        height,
    )
    horizontal = _merge_line_segments(
        horizontal_segments,
        "horizontal",
        width,
        height,
    )
    return vertical, horizontal


def _boundary_metrics(edge_distance, gray, start, end):
    """Оцениваем поддержку стороны границами и перепад яркости через неё."""
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1:
        return 0.0, 0.0

    inward = np.array([-direction[1], direction[0]]) / length
    offset = max(4.0, min(gray.shape) * 0.015)
    positions = np.linspace(0.1, 0.9, 80)[:, np.newaxis]
    points = start * (1 - positions) + end * positions

    def values_at(sample_points, source):
        x = np.clip(np.rint(sample_points[:, 0]).astype(int), 0, source.shape[1] - 1)
        y = np.clip(np.rint(sample_points[:, 1]).astype(int), 0, source.shape[0] - 1)
        return source[y, x]

    support = values_at(points, edge_distance) <= 5
    inside = values_at(points + inward * offset, gray).astype(np.float64)
    outside = values_at(points - inward * offset, gray).astype(np.float64)
    contrast = np.abs(inside - outside) / 255
    signed_contrast = (inside - outside) / 255
    return (
        float(np.mean(support)),
        float(np.mean(contrast)),
        float(np.mean(signed_contrast)),
    )


def _bottom_texture_transition(texture, line):
    """Оцениваем рост текстуры от пола кузова к наружной рампе."""
    height, width = texture.shape
    a, b, c = line
    if abs(b) < 1e-8:
        return 0.0

    x = np.linspace(width * 0.1, width * 0.9, 100)
    y = -(a * x + c) / b
    x_index = np.clip(np.rint(x).astype(int), 0, width - 1)
    inside = []
    outside = []
    for offset in (10, 20, 30, 40):
        inside_y = np.clip(np.rint(y - offset).astype(int), 0, height - 1)
        outside_y = np.clip(np.rint(y + offset).astype(int), 0, height - 1)
        inside.append(texture[inside_y, x_index])
        outside.append(texture[outside_y, x_index])

    inside_mean = float(np.mean(inside))
    outside_mean = float(np.mean(outside))
    transition = (outside_mean - inside_mean) / (
        outside_mean + inside_mean + 1e-8
    )
    return float(np.clip(transition, 0.0, 1.0))


def _valid_roi(corners, width, height):
    """Проверяем геометрию кандидата проёма до перспективного преобразования."""
    polygon = corners.reshape(-1, 1, 2).astype(np.float32)
    margin_x = width * 0.1
    margin_y = height * 0.1
    if (
        np.any(corners[:, 0] < -margin_x)
        or np.any(corners[:, 0] > width + margin_x)
        or np.any(corners[:, 1] < -margin_y)
        or np.any(corners[:, 1] > height + margin_y)
        or not cv2.isContourConvex(polygon.astype(np.int32))
    ):
        return False

    area = abs(cv2.contourArea(polygon))
    if area < 0.3 * width * height:
        return False

    center = (width / 2, height / 2)
    return cv2.pointPolygonTest(polygon, center, False) >= 0


def _find_container_corners(image):
    """Ищем четырёхугольник проёма по двум стенкам, верху и порогу кузова."""
    detection_image, scale = _resize_for_detection(image)
    gray = cv2.cvtColor(detection_image, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(enhanced, 40, 120, apertureSize=3)
    edge_distance = cv2.distanceTransform(
        (edges == 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    texture = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    vertical, horizontal = _line_candidates(edges)
    height, width = edges.shape
    left_candidates = [
        candidate
        for candidate in vertical
        if width * 0.01 < candidate["position"] < width * 0.35
    ][:8]
    right_candidates = [
        candidate
        for candidate in vertical
        if width * 0.68 < candidate["position"] < width * 0.99
    ][:8]
    top_candidates = [
        candidate
        for candidate in horizontal
        if (
            candidate["position"] < height * 0.42
            and abs(candidate["angle"]) < np.deg2rad(12)
        )
    ][:8]
    bottom_candidates = [
        candidate
        for candidate in horizontal
        if (
            height * 0.55 < candidate["position"] < height * 0.93
            and abs(candidate["angle"]) < np.deg2rad(12)
        )
    ][:8]
    for candidate in bottom_candidates:
        candidate["span"] = (
            candidate["extent"][1] - candidate["extent"][0]
        ) / width
        candidate["texture_transition"] = _bottom_texture_transition(
            texture,
            candidate["line"],
        )

    best = None
    best_score = 0.0
    for left in left_candidates:
        for right in right_candidates:
            for top in top_candidates:
                for bottom in bottom_candidates:
                    intersections = [
                        _intersect_lines(left["line"], top["line"]),
                        _intersect_lines(right["line"], top["line"]),
                        _intersect_lines(right["line"], bottom["line"]),
                        _intersect_lines(left["line"], bottom["line"]),
                    ]
                    if any(point is None for point in intersections):
                        continue

                    corners = np.asarray(intersections, dtype=np.float32)
                    if not _valid_roi(corners, width, height):
                        continue

                    top_width = np.linalg.norm(corners[1] - corners[0])
                    bottom_width = np.linalg.norm(corners[2] - corners[3])
                    left_height = np.linalg.norm(corners[3] - corners[0])
                    right_height = np.linalg.norm(corners[2] - corners[1])
                    if (
                        min(top_width, bottom_width) < width * 0.45
                        or (left_height + right_height) / 2 < height * 0.35
                    ):
                        continue

                    bottom_position = float(np.mean(corners[2:, 1]) / height)
                    if (
                        not 0.65 < bottom_position < 0.93
                        or abs(corners[2, 1] - corners[3, 1]) > height * 0.15
                    ):
                        continue

                    area_ratio = abs(
                        cv2.contourArea(corners.reshape(-1, 1, 2))
                    ) / (width * height)
                    top_position = float(np.mean(corners[:2, 1]) / height)
                    if area_ratio < 0.45 or top_position > 0.28:
                        continue

                    boundary_metrics = [
                        _boundary_metrics(
                            edge_distance,
                            gray,
                            corners[index],
                            corners[(index + 1) % 4],
                        )
                        for index in range(4)
                    ]
                    supports = [item[0] for item in boundary_metrics]
                    contrasts = [item[1] for item in boundary_metrics]
                    side_brightness = np.mean(
                        [
                            max(0.0, boundary_metrics[1][2]),
                            max(0.0, boundary_metrics[3][2]),
                        ]
                    )
                    if np.mean(supports) < 0.35 or min(supports) < 0.1:
                        continue

                    bottom_prior = max(0.0, 1 - abs(bottom_position - 0.78) / 0.22)
                    strengths = [
                        left["strength"],
                        right["strength"],
                        top["strength"],
                        bottom["strength"],
                    ]
                    score = (
                        0.28 * np.mean(supports)
                        + 0.07 * min(supports)
                        + 0.12 * np.mean(contrasts)
                        + 0.11 * np.mean(strengths)
                        + 0.12 * min(bottom["span"], 1.0)
                        + 0.15 * bottom["texture_transition"]
                        + 0.04 * bottom_prior
                        + 0.03 * min(area_ratio, 1.0)
                        + 0.08 * side_brightness
                    )
                    if score > best_score:
                        best = corners
                        best_score = score

    if best is None or best_score < 0.45:
        return None
    return best / scale


def _normalized_container_image(image):
    """Возвращаем выпрямленный проём или полный кадр и признак успеха ROI."""
    corners = _find_container_corners(image)
    if corners is None:
        return cv2.resize(image, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_LINEAR), 0.0

    destination = np.array(
        [[0, 0], [ROI_SIZE - 1, 0], [ROI_SIZE - 1, ROI_SIZE - 1], [0, ROI_SIZE - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    normalized = cv2.warpPerspective(
        image,
        transform,
        (ROI_SIZE, ROI_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return normalized, 1.0


def feature(path):
    """
    Преобразуем одно изображение в 235 признаков.

    Главное изменение:
    spatial, texture, hist и blue_ratio теперь считаются
    по найденному и выпрямленному проёму кузова.

    Если проём найти не удалось, используется fallback:
    всё изображение приводится к размеру ROI_SIZE x ROI_SIZE.
    """

    # ============================================================
    # 1. Читаем исходное изображение
    # ============================================================

    image = read_rgb_image(path)


    # ============================================================
    # 2. Сохраняем aspect исходного изображения
    # ============================================================

    # Aspect пока оставляем от исходного кадра.
    #
    # Если считать его после warpPerspective,
    # он всегда будет:
    #
    # 320 / 320 = 1
    #
    # и перестанет нести информацию.
    height, width = image.shape[:2]
    aspect = width / height


    # ============================================================
    # 3. Находим проём кузова
    # ============================================================

    # Если проём найден:
    #
    # original image
    #       ↓
    # container corners
    #       ↓
    # warpPerspective
    #       ↓
    # 320 × 320 normalized ROI
    #
    # Если проём НЕ найден:
    #
    # весь исходный кадр просто resize -> 320 × 320
    #
    # roi_found:
    #
    # 1.0 -> проём найден
    # 0.0 -> fallback
    normalized_roi, roi_found = _normalized_container_image(image)


    # ============================================================
    # 4. Уменьшаем уже НОРМАЛИЗОВАННЫЙ ПРОЁМ до 32 × 32
    # ============================================================

    # Раньше здесь использовался исходный image.
    #
    # Теперь все основные признаки описывают именно внутреннюю
    # область кузова, а не асфальт, двери машины, фон и т.д.
    small = cv2.resize(
        normalized_roi,
        (32, 32),
        interpolation=cv2.INTER_LINEAR
    )

    # Переводим RGB:
    #
    # 0...255
    #
    # в:
    #
    # 0...1
    small = small.astype(np.float64) / 255.0


    # ============================================================
    # 5. Spatial features
    # ============================================================

    # Делим нормализованный кузов на сетку 8 × 8.
    #
    # В каждой ячейке считаем средний:
    #
    # R
    # G
    # B
    #
    # Получаем:
    #
    # 8 × 8 × 3 = 192 признака
    #
    # Теперь положение цвета относительно кузова
    # становится намного более сопоставимым между фотографиями.
    spatial = (
        small
        .reshape(8, 4, 8, 4, 3)
        .mean(axis=(1, 3))
        .ravel()
    )


    # ============================================================
    # 6. Texture features
    # ============================================================

    # Получаем простое grayscale-представление.
    gray = small.mean(axis=2)

    # Делим кузов на сетку:
    #
    # 4 × 4
    #
    # В каждом регионе считаем standard deviation.
    #
    # Получаем 16 признаков.
    #
    # Теперь texture относится именно к внутренности кузова.
    texture = (
        gray
        .reshape(4, 8, 4, 8)
        .std(axis=(1, 3))
        .ravel()
    )


    # ============================================================
    # 7. RGB histogram
    # ============================================================

    # Строим по 8 bins для каждого RGB-канала.
    #
    # 8 × 3 = 24 признака.
    #
    # Главное отличие:
    # в histogram больше не должны попадать цвета асфальта,
    # наружной части машины и окружающей сцены,
    # если ROI был успешно найден.
    hist = np.concatenate(
        [
            np.histogram(
                small[:, :, channel],
                bins=8,
                range=(0, 1)
            )[0] / 1024

            for channel in range(3)
        ]
    )


    # ============================================================
    # 8. Доля синего груза
    # ============================================================

    # blue_ratio уже раньше считался по normalized_roi.
    #
    # Здесь ничего принципиально не меняем.
    blue_ratio = blue_ratio_from_rgb(
        normalized_roi
    )


    # ============================================================
    # 9. Собираем признаки
    # ============================================================

    # spatial    = 192
    # texture    = 16
    # hist       = 24
    # aspect     = 1
    # blue_ratio = 1
    # roi_found  = 1
    #
    # Итого:
    #
    # 235 признаков
    return np.concatenate(
        (
            spatial,
            texture,
            hist,
            [
                aspect,
                blue_ratio,
                roi_found
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
