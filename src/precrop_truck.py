"""Предвычисляет кроп кузова по маске сегментатора для всех фото в папке —
один раз, а не на лету в DataLoader (сегментация на GPU внутри воркеров
DataLoader несовместима с multiprocessing/fork и не даёт выигрыша, так как
маска не зависит от эпохи).

Результат — обычная папка с jpg такой же структуры (image_id.jpg), которую
потом просто передаёте как --images в train_cnn.py / predict_cnn.py вместо
исходной директории — остальной пайплайн не меняется вообще.

Защита от плохой маски (важно для outside-кадров с похожим на кузов фоном):
  1. Берём не весь bbox по mask==1, а bbox САМОЙ БОЛЬШОЙ связной компоненты
     (cv2.connectedComponents) — единичные ложные пиксели маски на стене/
     столбике вне кузова не растягивают кроп, в отличие от features.py::
     crop_by_mask, который берёт bbox по всем ненулевым пикселям сразу.
  2. Если площадь этой компоненты подозрительно мала относительно всего
     кадра (--min-area-ratio) — считаем маску ненадёжной и откатываемся на
     полное изображение без кропа, а не обрезаем непонятно что.
  3. К найденному bbox добавляется отступ (--margin), чтобы не срезать край
     груза, прилегающий к границе маски.

Использование:
    python precrop_truck.py \
        --images ./DataSet/train/images \
        --output ./DataSet/train/images_cropped

    python precrop_truck.py \
        --images ./DataSet/test/images \
        --output ./DataSet/test/images_cropped

Дальше просто:
    python train_cnn.py --images ./DataSet/train/images_cropped ...
    python predict_cnn.py --images ./DataSet/test/images_cropped ...
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from segmentation.runtime import predict_truck_mask


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def largest_component_bbox(mask):
    """bbox самой большой связной компоненты маски; None, если маска пустая."""
    mask_u8 = (mask > 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    if num_labels <= 1:
        return None

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    x = int(stats[largest_label, cv2.CC_STAT_LEFT])
    y = int(stats[largest_label, cv2.CC_STAT_TOP])
    w = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    h = int(stats[largest_label, cv2.CC_STAT_HEIGHT])
    return x, y, w, h


def expand_bbox(x, y, w, h, img_w, img_h, margin_frac):
    margin_x = int(w * margin_frac)
    margin_y = int(h * margin_frac)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)
    return x1, y1, x2, y2


def crop_to_truck(image_rgb, threshold, margin_frac, min_area_ratio):
    """Возвращает (cropped_image_rgb, was_cropped: bool)."""
    height, width = image_rgb.shape[:2]
    mask = predict_truck_mask(image_rgb, threshold=threshold)

    bbox = largest_component_bbox(mask)
    if bbox is None:
        return image_rgb, False

    bx, by, bw, bh = bbox
    area_ratio = (bw * bh) / (width * height)
    if area_ratio < min_area_ratio:
        # Подозрительно маленькая маска — вероятно, сегментатор ошибся;
        # безопаснее отдать полный кадр, чем обрезать непонятно что.
        return image_rgb, False

    x1, y1, x2, y2 = expand_bbox(bx, by, bw, bh, width, height, margin_frac)
    cropped = image_rgb[y1:y2, x1:x2]
    return cropped, True


def collect_images(images_dir):
    return sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in VALID_EXTENSIONS
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="Папка с исходными фото")
    parser.add_argument("--output", required=True, help="Куда сохранить кропы")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--margin", type=float, default=0.04,
        help="Отступ от bbox как доля его ширины/высоты",
    )
    parser.add_argument(
        "--min-area-ratio", type=float, default=0.15,
        help="Если найденная компонента меньше этой доли кадра — fallback на полный кадр",
    )
    parser.add_argument(
        "--save-previews", action="store_true",
        help="Сохранить ORIGINAL|CROPPED превью для визуальной проверки",
    )
    parser.add_argument("--previews-dir", default=None)
    args = parser.parse_args()

    images_dir = Path(args.images)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    previews_dir = Path(args.previews_dir) if args.previews_dir else output_dir.parent / "crop_previews"
    if args.save_previews:
        previews_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(images_dir)
    print(f"Найдено изображений: {len(image_paths)}")

    cropped_count = 0
    fallback_count = 0
    error_count = 0

    for index, image_path in enumerate(image_paths, start=1):
        try:
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cv2 could not decode image")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            cropped_rgb, was_cropped = crop_to_truck(
                rgb, args.threshold, args.margin, args.min_area_ratio
            )

            out_path = output_dir / image_path.name
            cv2.imwrite(str(out_path), cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))

            if was_cropped:
                cropped_count += 1
            else:
                fallback_count += 1

            if args.save_previews:
                original_resized = cv2.resize(bgr, (cropped_rgb.shape[1], cropped_rgb.shape[0]))
                preview = np.hstack(
                    [original_resized, cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR)]
                )
                cv2.imwrite(str(previews_dir / f"{image_path.stem}_preview.jpg"), preview)

        except Exception as error:
            error_count += 1
            print(f"[ERROR] {image_path.name}: {error}")

        if index % 100 == 0:
            print(f"{index}/{len(image_paths)}", flush=True)

    print()
    print("=== Готово ===")
    print(f"Обрезано по маске: {cropped_count}")
    print(f"Fallback на полный кадр (маска ненадёжна/пустая): {fallback_count}")
    print(f"Ошибок чтения: {error_count}")
    print(f"Сохранено в: {output_dir}")
    if args.save_previews:
        print(f"Превью для проверки глазами: {previews_dir}")


if __name__ == "__main__":
    main()
