from pathlib import Path

import cv2
import numpy as np


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


def feature(path):
    """Преобразуем одно изображение в 233 признака."""
    image = read_rgb_image(path)

    # Соотношение сторон описывает геометрию проёма кузова после EXIF-поворота.
    height, width = image.shape[:2]
    aspect = width / height

    # Приводим снимок к единому небольшому размеру. Билинейная интерполяция
    # подавляет мелкий шум и оставляет признаки сопоставимыми между снимками.
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LINEAR)
    small = small.astype(np.float64) / 255.0

    # Усредняем RGB в сетке 8x8: 64 ячейки по 3 канала дают 192 признака
    # пространственного распределения груза, освещения и стен кузова.
    spatial = small.reshape(8, 4, 8, 4, 3).mean(axis=(1, 3)).ravel()

    # Среднее RGB формирует яркостное изображение. Стандартное отклонение в
    # сетке 4x4 отражает локальную текстуру: границы, коробки и палеты.
    gray = small.mean(axis=2)
    texture = gray.reshape(4, 8, 4, 8).std(axis=(1, 3)).ravel()

    # Три гистограммы по 8 корзин описывают общий цвет и освещённость. Каждая
    # корзина нормирована на число пикселей, поэтому её масштаб стабилен.
    hist = np.concatenate(
        [
            np.histogram(small[:, :, channel], bins=8, range=(0, 1))[0] / 1024
            for channel in range(3)
        ]
    )

    # Итог: 192 пространственных + 16 текстурных + 24 гистограммных + аспект.
    return np.concatenate((spatial, texture, hist, [aspect]))


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

        # Каждый вектор имеет фиксированную длину 233 и добавляется в том же
        # порядке, в котором изображение указано во входной таблице.
        result.append(feature(image_path))

        if (index + 1) % 200 == 0:
            print(f"Features: {index + 1}/{len(rows)}", flush=True)

    return np.asarray(result, dtype=np.float64)
