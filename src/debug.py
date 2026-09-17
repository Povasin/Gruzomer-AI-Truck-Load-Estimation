import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "src")

from features import (
    BLUE_HSV_LOWER,
    BLUE_HSV_UPPER,
    _find_container_corners,
    _line_candidates,
    _normalized_container_image,
    _resize_for_detection,
    read_rgb_image,
)


def write_rgb(path, image):
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def draw_infinite_line(image, line, color, thickness=2):
    height, width = image.shape[:2]
    a, b, c = line
    if abs(a) >= abs(b):
        start = (int(round(-c / a)), 0)
        end = (int(round(-(b * (height - 1) + c) / a)), height - 1)
    else:
        start = (0, int(round(-c / b)))
        end = (width - 1, int(round(-(a * (width - 1) + c) / b)))
    cv2.line(image, start, end, color, thickness, cv2.LINE_AA)


def process_image(image_id, image_dir, output_dir):
    source = image_dir / f"{image_id}.jpg"
    image = read_rgb_image(source)
    corners = _find_container_corners(image)
    roi, roi_found = _normalized_container_image(image)

    detection_image, _ = _resize_for_detection(image)
    gray = cv2.cvtColor(detection_image, cv2.COLOR_RGB2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(enhanced, 40, 120, apertureSize=3)
    vertical, horizontal = _line_candidates(edges)

    segments_image = detection_image.copy()
    for candidate in vertical:
        for x1, y1, x2, y2 in candidate["segments"]:
            cv2.line(
                segments_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
    for candidate in horizontal:
        for x1, y1, x2, y2 in candidate["segments"]:
            cv2.line(
                segments_image,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 165, 0),
                1,
                cv2.LINE_AA,
            )

    merged_image = detection_image.copy()
    for candidate in vertical:
        draw_infinite_line(merged_image, candidate["line"], (0, 255, 0))
    for candidate in horizontal:
        draw_infinite_line(merged_image, candidate["line"], (255, 165, 0))

    outline = image.copy()
    if corners is not None:
        cv2.polylines(
            outline,
            [corners.astype(np.int32)],
            True,
            (0, 255, 0),
            5,
        )
        status = "ROI найден"
    else:
        cv2.putText(
            outline,
            "ROI not found: full frame",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
        )
        status = "ROI не найден, использован полный кадр"

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, BLUE_HSV_LOWER, BLUE_HSV_UPPER)
    ratio = np.count_nonzero(mask) / mask.size
    overlay = roi.copy()
    overlay[mask > 0] = (255, 0, 0)
    overlay = cv2.addWeighted(roi, 0.55, overlay, 0.45, 0)

    cv2.imwrite(str(output_dir / f"{image_id}_edges.png"), edges)
    write_rgb(output_dir / f"{image_id}_segments.jpg", segments_image)
    write_rgb(output_dir / f"{image_id}_merged_lines.jpg", merged_image)
    write_rgb(output_dir / f"{image_id}_outline.jpg", outline)
    write_rgb(output_dir / f"{image_id}_roi.jpg", roi)
    cv2.imwrite(str(output_dir / f"{image_id}_blue_mask.png"), mask)
    write_rgb(output_dir / f"{image_id}_blue_overlay.jpg", overlay)

    print(
        f"{image_id}: {status}; roi_found={roi_found:.0f}; "
        f"blue_ratio={ratio:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Сохраняет промежуточные изображения поиска проёма кузова."
    )
    parser.add_argument("image_ids", nargs="+", help="image_id без расширения")
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("./DataSet/train/images"),
        help="Каталог исходных JPG",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("debug_output"),
        help="Каталог отладочных изображений",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    for image_id in args.image_ids:
        process_image(image_id, args.images, args.output)

    print(f"Файлы сохранены в: {args.output.resolve()}")


if __name__ == "__main__":
    main()
