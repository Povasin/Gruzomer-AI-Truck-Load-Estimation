from pathlib import Path
import random

import cv2
import numpy as np

from manual.truck_segmentation.runtime import predict_truck_mask


# =========================================================
# SETTINGS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT_DIR
    / "DataSet"
    / "train"
    / "images"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "DataSet"
    / "train"
    / "floor_images"
)

# Для проверки качества crop
PREVIEW_DIR = (
    PROJECT_DIR
    / "DataSet"
    / "train"
    / "floor_prepare_previews"
)

# Запоминаем, какие именно 200 фото выбрали
SELECTED_FILE = (
    PROJECT_DIR
    / "DataSet"
    / "train"
    / "floor_selected_images.txt"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# PARAMETERS
# =========================================================

COUNT = 800

ROI_SIZE = 320

RANDOM_SEED = 42

TRUCK_THRESHOLD = 0.5

VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# =========================================================
# READ IMAGE
# =========================================================

def read_rgb_image(path):
    """
    OpenCV читает BGR.
    Для segmentation model нужен RGB.
    """

    bgr = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if bgr is None:
        raise ValueError(
            f"Не удалось открыть изображение: {path}"
        )

    rgb = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2RGB,
    )

    return rgb


# =========================================================
# CROP BY TRUCK MASK
# =========================================================

def crop_by_mask(
    image,
    mask,
):
    """
    Находим bounding box маски кузова
    и вырезаем соответствующий участок изображения.

    Возвращает:
        cropped image
        cropped truck mask
    """

    ys, xs = np.where(
        mask > 0
    )

    # Если segmentation вообще ничего не нашла
    if (
        len(xs) == 0
        or len(ys) == 0
    ):
        return None, None

    x1 = xs.min()
    x2 = xs.max() + 1

    y1 = ys.min()
    y2 = ys.max() + 1

    roi = image[
        y1:y2,
        x1:x2,
    ]

    roi_mask = mask[
        y1:y2,
        x1:x2,
    ]

    return (
        roi,
        roi_mask,
    )


# =========================================================
# SELECT 200 IMAGES
# =========================================================

def get_selected_images():
    """
    Сохраняет уже выбранные изображения
    и при увеличении COUNT добавляет новые.

    Например:
        раньше COUNT = 200
        теперь COUNT = 500

    Тогда старые 200 останутся,
    и будет добавлено ещё 300 новых.
    """

    all_images = sorted([
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in VALID_EXTENSIONS
    ])

    if len(all_images) == 0:
        raise RuntimeError(
            f"В папке нет изображений: {IMAGE_DIR}"
        )

    # =====================================================
    # Читаем уже выбранные изображения
    # =====================================================

    selected = []

    if SELECTED_FILE.exists():

        print(
            f"Найден существующий список:"
            f"\n{SELECTED_FILE}"
        )

        with open(
            SELECTED_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            filenames = [
                line.strip()
                for line in file
                if line.strip()
            ]

        for filename in filenames:

            path = (
                IMAGE_DIR
                / filename
            )

            if path.exists():

                selected.append(
                    path
                )

            else:

                print(
                    f"[WARNING] "
                    f"Не найдено: {filename}"
                )

    # =====================================================
    # Проверяем сколько уже выбрано
    # =====================================================

    print(
        f"Уже выбрано: {len(selected)}"
    )

    if len(selected) >= COUNT:

        print(
            f"Уже есть нужное количество: "
            f"{len(selected)}"
        )

        return sorted(
            selected[:COUNT]
        )

    # =====================================================
    # Ищем изображения,
    # которых ещё нет в selected
    # =====================================================

    selected_names = {
        path.name
        for path in selected
    }

    available = [
        path
        for path in all_images
        if path.name not in selected_names
    ]

    need_count = (
        COUNT
        - len(selected)
    )

    print(
        f"Нужно добавить: {need_count}"
    )

    print(
        f"Доступно новых: {len(available)}"
    )

    # =====================================================
    # Выбираем новые
    # =====================================================

    random.seed(
        RANDOM_SEED
    )

    add_count = min(
        need_count,
        len(available),
    )

    new_images = random.sample(
        available,
        add_count,
    )

    selected.extend(
        new_images
    )

    selected = sorted(
        selected
    )

    # =====================================================
    # Перезаписываем список
    # =====================================================

    with open(
        SELECTED_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        for path in selected:

            file.write(
                path.name + "\n"
            )

    print(
        f"Добавлено новых: {len(new_images)}"
    )

    print(
        f"Теперь всего выбрано: "
        f"{len(selected)}"
    )

    print(
        f"Список обновлён:"
        f"\n{SELECTED_FILE}"
    )

    return selected

    # =====================================================
    # Первый запуск
    # =====================================================

    random.seed(
        RANDOM_SEED
    )

    count = min(
        COUNT,
        len(all_images),
    )

    selected = random.sample(
        all_images,
        count,
    )

    # Сортировка нужна только,
    # чтобы потом было удобно размечать.
    selected = sorted(
        selected
    )

    with open(
        SELECTED_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        for path in selected:

            file.write(
                path.name + "\n"
            )

    print(
        f"Выбрано изображений: {len(selected)}"
    )

    print(
        f"Список сохранён:"
        f"\n{SELECTED_FILE}"
    )

    return selected


# =========================================================
# PREVIEW
# =========================================================

def save_preview(
    original_rgb,
    truck_mask,
    roi_rgb,
    output_path,
):
    """
    Сохраняет картинку для визуальной проверки:

    original с зелёной маской | полученный ROI
    """

    original_bgr = cv2.cvtColor(
        original_rgb,
        cv2.COLOR_RGB2BGR,
    )

    overlay = (
        original_bgr.copy()
    )

    overlay[
        truck_mask > 0
    ] = (
        0,
        255,
        0,
    )

    masked_preview = cv2.addWeighted(
        overlay,
        0.30,
        original_bgr,
        0.70,
        0,
    )

    roi_bgr = cv2.cvtColor(
        roi_rgb,
        cv2.COLOR_RGB2BGR,
    )

    # приводим preview к одинаковой высоте
    preview_height = 500

    scale_original = (
        preview_height
        / masked_preview.shape[0]
    )

    original_width = int(
        masked_preview.shape[1]
        * scale_original
    )

    original_preview = cv2.resize(
        masked_preview,
        (
            original_width,
            preview_height,
        ),
    )

    scale_roi = (
        preview_height
        / roi_bgr.shape[0]
    )

    roi_width = int(
        roi_bgr.shape[1]
        * scale_roi
    )

    roi_preview = cv2.resize(
        roi_bgr,
        (
            roi_width,
            preview_height,
        ),
    )

    preview = np.hstack([
        original_preview,
        roi_preview,
    ])

    cv2.imwrite(
        str(output_path),
        preview,
    )


# =========================================================
# PROCESS ONE IMAGE
# =========================================================

def process_image(
    image_path,
):
    """
    Основной pipeline:

    image
    ↓
    truck segmentation
    ↓
    crop
    ↓
    resize 320x320
    ↓
    save
    """

    # =====================================================
    # 1. READ
    # =====================================================

    image = read_rgb_image(
        image_path
    )

    # =====================================================
    # 2. TRUCK SEGMENTATION
    # =====================================================

    truck_mask = predict_truck_mask(
        image,
        threshold=TRUCK_THRESHOLD,
    )

    truck_mask = (
        truck_mask > 0
    ).astype(
        np.uint8
    )

    # =====================================================
    # 3. CHECK MASK
    # =====================================================

    if np.count_nonzero(
        truck_mask
    ) == 0:

        print(
            f"[ERROR] Кузов не найден: "
            f"{image_path.name}"
        )

        return False

    # =====================================================
    # 4. CROP
    # =====================================================

    roi, roi_mask = crop_by_mask(
        image,
        truck_mask,
    )

    if roi is None:

        print(
            f"[ERROR] Не удалось сделать crop: "
            f"{image_path.name}"
        )

        return False

    # =====================================================
    # 5. RESIZE
    # =====================================================

    roi_320 = cv2.resize(
        roi,
        (
            ROI_SIZE,
            ROI_SIZE,
        ),
        interpolation=cv2.INTER_LINEAR,
    )

    # =====================================================
    # 6. SAVE FLOOR IMAGE
    # =====================================================

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}.jpg"
    )

    # Сейчас roi RGB,
    # а cv2.imwrite ожидает BGR.
    roi_bgr = cv2.cvtColor(
        roi_320,
        cv2.COLOR_RGB2BGR,
    )

    cv2.imwrite(
        str(output_path),
        roi_bgr,
    )

    # =====================================================
    # 7. PREVIEW
    # =====================================================

    preview_path = (
        PREVIEW_DIR
        / f"{image_path.stem}_preview.jpg"
    )

    save_preview(
        original_rgb=image,
        truck_mask=truck_mask,
        roi_rgb=roi_320,
        output_path=preview_path,
    )

    return True


# =========================================================
# MAIN
# =========================================================

def main():

    print()
    print(
        "=" * 60
    )

    print(
        "PREPARE FLOOR SEGMENTATION DATASET"
    )

    print(
        "=" * 60
    )

    print(
        f"Source:"
        f"\n{IMAGE_DIR}"
    )

    print(
        f"\nOutput:"
        f"\n{OUTPUT_DIR}"
    )

    print()

    selected_images = (
        get_selected_images()
    )

    total = len(
        selected_images
    )

    success = 0
    errors = 0

    for index, image_path in enumerate(
        selected_images,
        start=1,
    ):

        output_path = (
            OUTPUT_DIR
            / f"{image_path.stem}.jpg"
        )

        # =================================================
        # Уже обработано
        # =================================================

        if output_path.exists():

            print(
                f"[{index}/{total}] "
                f"[SKIP EXISTING] "
                f"{image_path.name}"
            )

            success += 1

            continue

        # =================================================
        # PROCESS
        # =================================================

        print(
            f"[{index}/{total}] "
            f"{image_path.name}"
        )

        try:

            ok = process_image(
                image_path
            )

            if ok:

                success += 1

            else:

                errors += 1

        except Exception as error:

            errors += 1

            print(
                f"[ERROR] "
                f"{image_path.name}: "
                f"{error}"
            )

    # =====================================================
    # RESULT
    # =====================================================

    print()
    print(
        "=" * 60
    )

    print(
        "ГОТОВО"
    )

    print(
        f"Успешно: {success}"
    )

    print(
        f"Ошибок: {errors}"
    )

    print(
        f"Всего: {total}"
    )

    print()
    print(
        f"Floor images:"
        f"\n{OUTPUT_DIR}"
    )

    print()
    print(
        f"Previews:"
        f"\n{PREVIEW_DIR}"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":

    main()