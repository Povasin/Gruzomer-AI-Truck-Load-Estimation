"""Визуализация работы алгоритма 20 пространственных сечений с нейросетью глубины MiDaS."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from manual.depth_slicer import analyze_image, SlicingResult


def draw_3d_frames_on_image(
    image: np.ndarray,
    result: SlicingResult,
) -> np.ndarray:
    """Отрисовывает 20 вложенных перспективных сечений от ворот вглубь кузова."""
    vis = image.copy()
    overlay = vis.copy()

    for frame in result.frames:
        k = frame.index
        fill = frame.fill

        if result.stop_index != -1 and k > result.stop_index:
            color = (0, 140, 255)  # синий
            thickness = 2
        elif (result.stop_index != -1 and k == result.stop_index) or fill >= 0.75:
            color = (230, 20, 20)  # ярко-красный
            thickness = 3
        elif fill > 0.15:
            color = (240, 160, 20)  # янтарный
            thickness = 2
        else:
            color = (30, 210, 30)   # зеленый
            thickness = 2

        # Прямоугольное сечение кузова на этой глубине
        cv2.rectangle(
            overlay,
            (frame.x_left, frame.y_ceil),
            (frame.x_right, frame.y_floor),
            color,
            thickness,
        )

        # Вертикальная линия сечения от потолка к полу
        cv2.line(
            overlay,
            (frame.x_center, frame.y_ceil),
            (frame.x_center, frame.y_floor),
            color,
            thickness,
        )

        if k % 4 == 0 or k == result.stop_index:
            lbl = f"S{k+1}:{fill*100:.0f}%"
            cv2.putText(
                overlay,
                lbl,
                (frame.x_left + 6, frame.y_ceil + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                lbl,
                (frame.x_left + 6, frame.y_ceil + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
    return vis


def render_depth_heatmap(
    depth_map: np.ndarray,
    truck_mask: np.ndarray,
    target_h: int = 600,
) -> np.ndarray:
    """Создаёт цветовую тепловую карту нейросетевой глубины MiDaS."""
    z_vis = np.clip(depth_map * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(z_vis, cv2.COLORMAP_INFERNO)
    # Маскируем фон вне кузова темным цветом
    heatmap[truck_mask == 0] = heatmap[truck_mask == 0] // 3

    cv2.putText(
        heatmap,
        "MiDaS Depth Map",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        heatmap,
        "Near (Bright) -> Far (Dark)",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    h, w, _ = heatmap.shape
    new_w = int(w * (target_h / h))
    return cv2.resize(heatmap, (new_w, target_h), interpolation=cv2.INTER_AREA)


def render_chart_panel(
    result: SlicingResult,
    target_pct: float | None,
    panel_width: int = 500,
    panel_height: int = 600,
) -> np.ndarray:
    """Рендерит правую панель со столбчатой диаграммой 20 срезов."""
    chart = np.full((panel_height, panel_width, 3), 248, dtype=np.uint8)

    margin_left = 60
    margin_right = 25
    margin_top = 70
    margin_bottom = 80

    plot_w = panel_width - margin_left - margin_right
    plot_h = panel_height - margin_top - margin_bottom

    cv2.putText(
        chart,
        "20 Slices (MiDaS Depth)",
        (margin_left, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    sub_text = f"Pred: {result.load_pct:.1f}%"
    if target_pct is not None:
        sub_text += f" | GT: {target_pct:.1f}% | Diff: {abs(result.load_pct - target_pct):.1f}%"
    cv2.putText(
        chart,
        sub_text,
        (margin_left, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        chart,
        (margin_left, margin_top),
        (margin_left + plot_w, margin_top + plot_h),
        (200, 200, 200),
        1,
    )

    for pct in [0, 25, 50, 75, 100]:
        y_grid = margin_top + int(plot_h * (1.0 - pct / 100.0))
        cv2.line(chart, (margin_left, y_grid), (margin_left + plot_w, y_grid), (225, 225, 225), 1)
        cv2.putText(chart, f"{pct}%", (margin_left - 48, y_grid + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1)

    y_75 = margin_top + int(plot_h * (1.0 - 0.75))
    for x_dash in range(margin_left, margin_left + plot_w, 12):
        cv2.line(chart, (x_dash, y_75), (min(x_dash + 7, margin_left + plot_w), y_75), (200, 30, 30), 2)
    cv2.putText(chart, "75% Stop Threshold", (margin_left + plot_w - 135, y_75 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 30, 30), 1)

    n = len(result.slice_fills)
    bar_slot = plot_w / n
    bar_width = int(max(4, bar_slot * 0.75))

    for i, fill in enumerate(result.slice_fills):
        x_center = margin_left + int((i + 0.5) * bar_slot)
        x1 = x_center - bar_width // 2
        x2 = x1 + bar_width

        bar_h = int(plot_h * np.clip(fill, 0.0, 1.0))
        y1 = margin_top + plot_h - bar_h
        y2 = margin_top + plot_h

        if result.stop_index != -1 and i > result.stop_index:
            color = (0, 140, 255)
        elif (result.stop_index != -1 and i == result.stop_index) or fill >= 0.75:
            color = (230, 20, 20)
        elif fill > 0.15:
            color = (240, 160, 20)
        else:
            color = (30, 210, 30)

        cv2.rectangle(chart, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(chart, (x1, y1), (x2, y2), (40, 40, 40), 1)
        cv2.putText(chart, str(i + 1), (x_center - (4 if i < 9 else 8), margin_top + plot_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1)

    cv2.putText(chart, "Slice Index (1: Doors -> 20: Cab)", (margin_left + 15, margin_top + plot_h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1)

    y_pred = margin_top + int(plot_h * (1.0 - result.load_pct / 100.0))
    cv2.line(chart, (margin_left, y_pred), (margin_left + plot_w, y_pred), (140, 30, 180), 2)
    cv2.putText(chart, f"Avg: {result.load_pct:.1f}%", (margin_left + 8, y_pred - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 30, 180), 1)

    leg_y = margin_top + plot_h + 65
    legends = [
        ("Empty", (30, 210, 30)),
        ("Partial", (240, 160, 20)),
        ("Stop Trigger", (230, 20, 20)),
        ("Occluded", (0, 140, 255)),
    ]
    cur_x = margin_left - 10
    for leg_text, leg_col in legends:
        cv2.rectangle(chart, (cur_x, leg_y), (cur_x + 12, leg_y + 10), leg_col, -1)
        cv2.rectangle(chart, (cur_x, leg_y), (cur_x + 12, leg_y + 10), (50, 50, 50), 1)
        cv2.putText(chart, leg_text, (cur_x + 16, leg_y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (60, 60, 60), 1)
        cur_x += 105

    return chart


def plot_slices_figure(
    image_path: Path,
    output_path: Path,
    ground_truth_pct: float | None = None,
    truck_weights: Path | None = None,
    floor_weights: Path | None = None,
    ceiling_weights: Path | None = None,
    device: str | None = None,
    *args,
    **kwargs,
):
    """Строит триптих: фото с 3D-сечениями + тепловая карта глубины MiDaS + столбчатый график."""
    if device is None:
        device = kwargs.get("device", None)
    result, image, truck_mask, floor_mask, ceiling_mask = analyze_image(
        image_path=image_path,
        truck_weights=truck_weights,
        floor_weights=floor_weights,
        ceiling_weights=ceiling_weights,
        device=device,
    )

    target_h = 600

    # 1. Фото с 3D сечениями
    vis_image = draw_3d_frames_on_image(image, result)
    img_h, img_w, _ = vis_image.shape
    new_w1 = int(img_w * (target_h / img_h))
    vis_resized = cv2.resize(vis_image, (new_w1, target_h), interpolation=cv2.INTER_AREA)

    # 2. Тепловая карта нейросетевой глубины MiDaS
    depth_heatmap = render_depth_heatmap(result.depth_map, truck_mask, target_h=target_h)

    # 3. График 20 срезов
    chart_panel = render_chart_panel(result, ground_truth_pct, panel_width=500, panel_height=target_h)

    # Склеиваем триптих
    combined = np.hstack([vis_resized, depth_heatmap, chart_panel])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    print(f"Иллюстрация сохранена: {output_path}")
    print(f"  Прогноз load_pct: {result.load_pct:.1f}%")
    if ground_truth_pct is not None:
        print(f"  Эталон target:    {ground_truth_pct:.1f}%")
        print(f"  Ошибка Diff:      {abs(result.load_pct - ground_truth_pct):.1f}%")
    if result.stop_index != -1:
        print(f"  Ранняя остановка: сработала на срезе {result.stop_index + 1}")
    else:
        print("  Ранняя остановка: не сработала")


def main():
    parser = argparse.ArgumentParser(description="Визуализация 20 срезов с MiDaS")
    parser.add_argument("--image", type=Path, required=True, help="Путь к изображению")
    parser.add_argument("--output", type=Path, default=Path("scratch/slice_visualization.png"))
    parser.add_argument("--target", type=float, default=None, help="Эталонный процент load_pct")
    parser.add_argument("--device", type=str, default=None, help="Устройство (cuda или cpu)")
    args = parser.parse_args()

    plot_slices_figure(
        image_path=args.image,
        output_path=args.output,
        ground_truth_pct=args.target,
        device=args.device,
    )


if __name__ == "__main__":
    main()
