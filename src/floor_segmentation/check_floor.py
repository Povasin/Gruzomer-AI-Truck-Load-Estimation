from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from runtime import predict_floor_mask


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        help="Путь к ROI изображению 320x320",
    )

    parser.add_argument(
        "--weights",
        required=True,
        help="Путь к обученной модели floor_unet.pt",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Порог сегментации пола",
    )

    parser.add_argument(
        "--output-dir",
        default="../debug_output/floor_check",
        help="Куда сохранить результаты",
    )

    args = parser.parse_args()

    # =========================================================
    # PATHS
    # =========================================================

    image_path = Path(args.image)
    weights_path = Path(args.weights)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Изображение не найдено: {image_path}"
        )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Веса не найдены: {weights_path}"
        )

    # =========================================================
    # LOAD IMAGE
    # =========================================================

    image_pil = (
        Image.open(image_path)
        .convert("RGB")
    )

    image = np.array(
        image_pil
    )

    print(
        f"Image shape: {image.shape}"
    )

    # =========================================================
    # FLOOR SEGMENTATION
    # =========================================================

    floor_mask = predict_floor_mask(
        image=image,
        truck_mask=None,
        threshold=args.threshold,
        weights_path=weights_path,
    )

    print(
        f"Floor pixels: "
        f"{np.count_nonzero(floor_mask)}"
    )

    print(
        f"Floor ratio: "
        f"{floor_mask.mean():.4f}"
    )

    # =========================================================
    # MASK IMAGE
    # =========================================================

    mask_image = (
        floor_mask * 255
    ).astype(
        np.uint8
    )

    mask_pil = Image.fromarray(
        mask_image,
        mode="L",
    )

    # =========================================================
    # OVERLAY
    # =========================================================

    overlay_array = image.copy()

    floor_pixels = (
        floor_mask > 0
    )

    # Зелёным показываем найденный пол
    overlay_array[
        floor_pixels
    ] = [
        0,
        255,
        0,
    ]

    overlay_colored = Image.fromarray(
        overlay_array
    )

    # Прозрачный overlay
    overlay = Image.blend(
        image_pil,
        overlay_colored,
        0.40,
    )

    # =========================================================
    # SIDE BY SIDE
    # =========================================================

    width, height = image_pil.size

    mask_rgb = mask_pil.convert(
        "RGB"
    )

    result = Image.new(
        "RGB",
        (
            width * 3,
            height,
        ),
    )

    result.paste(
        image_pil,
        (0, 0),
    )

    result.paste(
        mask_rgb,
        (width, 0),
    )

    result.paste(
        overlay,
        (width * 2, 0),
    )

    # =========================================================
    # SAVE
    # =========================================================

    stem = image_path.stem

    mask_path = (
        output_dir
        / f"{stem}_mask.png"
    )

    overlay_path = (
        output_dir
        / f"{stem}_overlay.jpg"
    )

    result_path = (
        output_dir
        / f"{stem}_result.jpg"
    )

    mask_pil.save(
        mask_path
    )

    overlay.save(
        overlay_path,
        quality=95,
    )

    result.save(
        result_path,
        quality=95,
    )

    print()
    print(
        f"Mask: {mask_path}"
    )

    print(
        f"Overlay: {overlay_path}"
    )

    print(
        f"Result: {result_path}"
    )

    # =========================================================
    # OPEN RESULT
    # =========================================================

    result.show()


if __name__ == "__main__":
    main()


"""
..\venv\Scripts\python.exe -m floor_segmentation.check_floor `
  --image ..\DataSet\train\floor_images\img_123.jpg `
  --weights .\floor_segmentation\models\floor_unet.pt
"""