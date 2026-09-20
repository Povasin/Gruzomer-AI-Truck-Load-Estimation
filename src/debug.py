import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

"""Визуальная проверка truck- и floor-сегментации для одного изображения."""

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from manual.floor_segmentation.runtime import predict_floor_mask
from manual.truck_segmentation.runtime import predict_truck_mask
from manual.features import read_rgb_image

DEFAULT_TRUCK_WEIGHTS = (
    SRC_DIR / "manual" / "truck_segmentation" / "models" / "best_unet_resnet18.pth"
)
DEFAULT_FLOOR_WEIGHTS = (
    SRC_DIR / "manual" / "floor_segmentation" / "models"
    / "floor_unet_resnet18_lr1e3.best_loss.pt"
)

def write_rgb(path, image):
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def mask_overlay(image, mask, color, alpha=0.45):
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0)

def mask_image(mask):
    return (mask.astype(np.uint8) * 255)

def draw_contours(image, mask, color):
    result = image.copy()
    contours, _ = cv2.findContours(mask_image(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 3)
    return result

def process_image(image_path, output_dir, truck_weights, floor_weights):
    image = read_rgb_image(image_path)
    truck_mask = predict_truck_mask(
        image,
        threshold=0.5,
        weights_path=truck_weights,
    )
    floor_mask = predict_floor_mask(
        image,
        truck_mask=truck_mask,
        threshold=0.5,
        weights_path=floor_weights,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    write_rgb(output_dir / f"{stem}_original.jpg", image)
    cv2.imwrite(str(output_dir / f"{stem}_truck_mask.png"), mask_image(truck_mask))
    cv2.imwrite(str(output_dir / f"{stem}_floor_mask.png"), mask_image(floor_mask))
    write_rgb(
        output_dir / f"{stem}_truck_overlay.jpg",
        mask_overlay(image, truck_mask, (0, 255, 0)),
    )
    write_rgb(
        output_dir / f"{stem}_floor_overlay.jpg",
        mask_overlay(image, floor_mask, (0, 0, 255)),
    )

    combined = mask_overlay(image, truck_mask, (0, 255, 0), alpha=0.25)
    combined = mask_overlay(combined, floor_mask, (255, 0, 0), alpha=0.45)
    combined = draw_contours(combined, truck_mask, (0, 255, 0))
    combined = draw_contours(combined, floor_mask, (255, 0, 0))
    write_rgb(output_dir / f"{stem}_combined_overlay.jpg", combined)

    truck_ratio = float(np.mean(truck_mask > 0))
    floor_ratio = float(np.mean(floor_mask > 0))
    print(f"Изображение: {image_path}")
    print(f"Truck mask: {truck_mask.shape}, площадь={truck_ratio:.4%}")
    print(f"Floor mask: {floor_mask.shape}, площадь={floor_ratio:.4%}")
    print(f"Результаты сохранены в: {output_dir.resolve()}")

def main():
    parser = argparse.ArgumentParser(
        description="Показывает результаты truck_segmentation и floor_segmentation."
    )
    parser.add_argument("image", type=Path, help="Путь к исходному изображению")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "debug_output" / "segmentation",
        help="Каталог для масок и наложений",
    )
    parser.add_argument(
        "--truck-weights",
        type=Path,
        default=DEFAULT_TRUCK_WEIGHTS,
        help="Веса сегментации кузова",
    )
    parser.add_argument(
        "--floor-weights",
        type=Path,
        default=DEFAULT_FLOOR_WEIGHTS,
        help="Веса сегментации пола",
    )
    args = parser.parse_args()

    image_path = args.image.resolve()
    truck_weights = args.truck_weights.resolve()
    floor_weights = args.floor_weights.resolve()
    for label, path in (
        ("изображение", image_path),
        ("truck-веса", truck_weights),
        ("floor-веса", floor_weights),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Не найден {label}: {path}")

    process_image(image_path, args.output.resolve(), truck_weights, floor_weights)

if __name__ == "__main__":
    main()
