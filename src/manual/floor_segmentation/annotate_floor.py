from pathlib import Path
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk


# =========================================================
# SETTINGS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = (
    PROJECT_DIR
    / "DataSet"
    / "train"
    / "floor_images"
)

MASK_DIR = (
    PROJECT_DIR
    / "debug_output"
    / "floor_masks"
)

PREVIEW_DIR = (
    PROJECT_DIR
    / "debug_output"
    / "floor_previews"
)

MASK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# =========================================================
# IMAGES
# =========================================================

def find_images(image_dir: str | Path) -> list[Path]:
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return []
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in VALID_EXTENSIONS
    )


images = find_images(IMAGE_DIR)


# =========================================================
# ANNOTATOR
# =========================================================

class FloorAnnotator:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "Floor Polygon Annotator"
        )

        # =================================================
        # STATE
        # =================================================

        self.image_index = -1

        self.image_path = None

        self.original_image = None

        self.display_image = None

        self.tk_image = None

        self.points = []

        # Масштаб между реальным изображением
        # и изображением на экране
        self.scale = 1.0

        self.offset_x = 0
        self.offset_y = 0

        # =================================================
        # WINDOW
        # =================================================

        self.root.geometry(
            "1100x850"
        )

        # =================================================
        # INFO LABEL
        # =================================================

        self.info_label = tk.Label(
            root,
            text="",
            font=(
                "Arial",
                12,
            ),
        )

        self.info_label.pack(
            pady=5
        )

        # =================================================
        # CANVAS
        # =================================================

        self.canvas = tk.Canvas(
            root,
            bg="black",
            cursor="cross",
        )

        self.canvas.pack(
            fill=tk.BOTH,
            expand=True,
        )

        # =================================================
        # HELP
        # =================================================

        help_text = (
            "ЛКМ: точка | "
            "ПКМ: удалить последнюю | "
            "ENTER: сохранить | "
            "R: очистить | "
            "E: пола нет | "
            "F: весь кадр пол | "
            "S: пропустить | "
            "Q/ESC: выйти"
        )

        self.help_label = tk.Label(
            root,
            text=help_text,
            font=(
                "Arial",
                10,
            ),
        )

        self.help_label.pack(
            pady=5
        )

        # =================================================
        # MOUSE
        # =================================================

        self.canvas.bind(
            "<Button-1>",
            self.left_click,
        )

        self.canvas.bind(
            "<Button-3>",
            self.right_click,
        )

        # =================================================
        # KEYBOARD
        # =================================================

        self.root.bind(
            "<Return>",
            self.save_polygon,
        )

        self.root.bind(
            "<Key-r>",
            self.reset_polygon,
        )

        self.root.bind(
            "<Key-R>",
            self.reset_polygon,
        )

        self.root.bind(
            "<Key-e>",
            self.save_empty_mask,
        )

        self.root.bind(
            "<Key-E>",
            self.save_empty_mask,
        )

        self.root.bind(
            "<Key-f>",
            self.save_full_mask,
        )

        self.root.bind(
            "<Key-F>",
            self.save_full_mask,
        )

        self.root.bind(
            "<Key-s>",
            self.skip_image,
        )

        self.root.bind(
            "<Key-S>",
            self.skip_image,
        )

        self.root.bind(
            "<Key-q>",
            self.exit_program,
        )

        self.root.bind(
            "<Key-Q>",
            self.exit_program,
        )

        self.root.bind(
            "<Escape>",
            self.exit_program,
        )

        # Если окно resize
        self.canvas.bind(
            "<Configure>",
            self.on_resize,
        )

        # =================================================
        # START
        # =================================================

        self.next_image()


    # =====================================================
    # NEXT IMAGE
    # =====================================================

    def next_image(self):

        self.points = []

        while True:

            self.image_index += 1

            if self.image_index >= len(images):

                print(
                    "Разметка закончена."
                )

                self.root.destroy()

                return

            image_path = images[
                self.image_index
            ]

            mask_path = (
                MASK_DIR
                / f"{image_path.stem}.png"
            )

            # Уже размеченные пропускаем
            if mask_path.exists():

                print(
                    f"[SKIP EXISTING] "
                    f"{image_path.name}"
                )

                continue

            self.image_path = image_path

            break

        # =================================================
        # LOAD IMAGE
        # =================================================

        try:

            self.original_image = (
                Image.open(
                    self.image_path
                )
                .convert("RGB")
            )

        except Exception as error:

            print(
                f"Не удалось открыть "
                f"{self.image_path}: "
                f"{error}"
            )

            self.next_image()

            return

        print()

        print(
            f"[{self.image_index + 1}/"
            f"{len(images)}] "
            f"{self.image_path.name}"
        )

        self.update_info()

        self.draw()


    # =====================================================
    # INFO
    # =====================================================

    def update_info(self):

        self.info_label.config(
            text=(
                f"{self.image_index + 1}/"
                f"{len(images)} | "
                f"{self.image_path.name} | "
                f"точек: {len(self.points)}"
            )
        )


    # =====================================================
    # RESIZE
    # =====================================================

    def on_resize(
        self,
        event,
    ):

        if self.original_image is not None:

            self.draw()


    # =====================================================
    # DRAW
    # =====================================================

    def draw(self):

        if self.original_image is None:

            return

        canvas_width = max(
            self.canvas.winfo_width(),
            1,
        )

        canvas_height = max(
            self.canvas.winfo_height(),
            1,
        )

        image_width, image_height = (
            self.original_image.size
        )

        # Оставляем небольшой отступ
        max_width = max(
            canvas_width - 20,
            1,
        )

        max_height = max(
            canvas_height - 20,
            1,
        )

        self.scale = min(
            max_width / image_width,
            max_height / image_height,
        )

        # Не увеличиваем сильно маленькие картинки
        self.scale = min(
            self.scale,
            2.0,
        )

        display_width = max(
            int(
                image_width
                * self.scale
            ),
            1,
        )

        display_height = max(
            int(
                image_height
                * self.scale
            ),
            1,
        )

        display = (
            self.original_image.resize(
                (
                    display_width,
                    display_height,
                ),
                Image.Resampling.LANCZOS,
            )
        )

        # =================================================
        # POLYGON PREVIEW
        # =================================================

        if len(self.points) > 0:

            overlay = display.copy()

            overlay_draw = ImageDraw.Draw(
                overlay,
                "RGBA",
            )

            scaled_points = [
                (
                    int(x * self.scale),
                    int(y * self.scale),
                )
                for x, y in self.points
            ]

            # Заполнение polygon
            if len(scaled_points) >= 3:

                overlay_draw.polygon(
                    scaled_points,
                    fill=(
                        0,
                        255,
                        0,
                        60,
                    ),
                    outline=(
                        0,
                        255,
                        0,
                        255,
                    ),
                )

            # Линии
            elif len(scaled_points) >= 2:

                overlay_draw.line(
                    scaled_points,
                    fill=(
                        0,
                        255,
                        0,
                        255,
                    ),
                    width=2,
                )

            # Точки
            for index, (
                x,
                y,
            ) in enumerate(
                scaled_points
            ):

                radius = 5

                overlay_draw.ellipse(
                    (
                        x - radius,
                        y - radius,
                        x + radius,
                        y + radius,
                    ),
                    fill=(
                        255,
                        0,
                        0,
                        255,
                    ),
                )

                overlay_draw.text(
                    (
                        x + 7,
                        y - 7,
                    ),
                    str(
                        index + 1
                    ),
                    fill=(
                        255,
                        255,
                        0,
                        255,
                    ),
                )

            display = overlay

        # =================================================
        # CANVAS
        # =================================================

        self.tk_image = ImageTk.PhotoImage(
            display
        )

        self.canvas.delete(
            "all"
        )

        self.offset_x = (
            canvas_width
            - display_width
        ) // 2

        self.offset_y = (
            canvas_height
            - display_height
        ) // 2

        self.canvas.create_image(
            self.offset_x,
            self.offset_y,
            anchor=tk.NW,
            image=self.tk_image,
        )

        self.update_info()


    # =====================================================
    # SCREEN -> IMAGE COORDS
    # =====================================================

    def canvas_to_image_coords(
        self,
        canvas_x,
        canvas_y,
    ):

        x = (
            canvas_x
            - self.offset_x
        )

        y = (
            canvas_y
            - self.offset_y
        )

        if self.scale <= 0:

            return None

        x = int(
            x
            / self.scale
        )

        y = int(
            y
            / self.scale
        )

        width, height = (
            self.original_image.size
        )

        # Клик вне изображения
        if (
            x < 0
            or y < 0
            or x >= width
            or y >= height
        ):

            return None

        return (
            x,
            y,
        )


    # =====================================================
    # LEFT CLICK
    # =====================================================

    def left_click(
        self,
        event,
    ):

        coords = (
            self.canvas_to_image_coords(
                event.x,
                event.y,
            )
        )

        if coords is None:

            return

        self.points.append(
            coords
        )

        self.draw()


    # =====================================================
    # RIGHT CLICK
    # =====================================================

    def right_click(
        self,
        event,
    ):

        if self.points:

            self.points.pop()

            self.draw()


    # =====================================================
    # CREATE POLYGON MASK
    # =====================================================

    def create_polygon_mask(
        self,
    ):

        width, height = (
            self.original_image.size
        )

        mask = Image.new(
            "L",
            (
                width,
                height,
            ),
            0,
        )

        draw = ImageDraw.Draw(
            mask
        )

        draw.polygon(
            self.points,
            fill=255,
        )

        return mask


    # =====================================================
    # SAVE POLYGON
    # =====================================================

    def save_polygon(
        self,
        event=None,
    ):

        if len(self.points) < 3:

            print(
                "Нужно минимум 3 точки."
            )

            return

        mask = (
            self.create_polygon_mask()
        )

        self.save_mask(
            mask
        )

        self.next_image()


    # =====================================================
    # SAVE EMPTY
    # =====================================================

    def save_empty_mask(
        self,
        event=None,
    ):

        width, height = (
            self.original_image.size
        )

        mask = Image.new(
            "L",
            (
                width,
                height,
            ),
            0,
        )

        self.save_mask(
            mask
        )

        print(
            "EMPTY FLOOR MASK"
        )

        self.next_image()


    # =====================================================
    # SAVE FULL
    # =====================================================

    def save_full_mask(
        self,
        event=None,
    ):

        width, height = (
            self.original_image.size
        )

        mask = Image.new(
            "L",
            (
                width,
                height,
            ),
            255,
        )

        self.save_mask(
            mask
        )

        print(
            "FULL FLOOR MASK"
        )

        self.next_image()


    # =====================================================
    # SAVE MASK + PREVIEW
    # =====================================================

    def save_mask(
        self,
        mask,
    ):

        mask_path = (
            MASK_DIR
            / f"{self.image_path.stem}.png"
        )

        mask.save(
            mask_path
        )

        # =================================================
        # PREVIEW
        # =================================================

        original = (
            self.original_image.copy()
        )

        overlay = (
            original.copy()
        )

        overlay_array = np.array(
            overlay
        )

        mask_array = np.array(
            mask
        )

        floor_pixels = (
            mask_array > 0
        )

        # RGB зелёный
        overlay_array[
            floor_pixels
        ] = [
            0,
            255,
            0,
        ]

        overlay = Image.fromarray(
            overlay_array
        )

        highlighted = Image.blend(
            original,
            overlay,
            0.35,
        )

        # original | highlighted
        preview_width = (
            original.width
            + highlighted.width
        )

        preview_height = max(
            original.height,
            highlighted.height,
        )

        preview = Image.new(
            "RGB",
            (
                preview_width,
                preview_height,
            ),
        )

        preview.paste(
            original,
            (
                0,
                0,
            ),
        )

        preview.paste(
            highlighted,
            (
                original.width,
                0,
            ),
        )

        preview_path = (
            PREVIEW_DIR
            / (
                f"{self.image_path.stem}"
                f"_preview.jpg"
            )
        )

        preview.save(
            preview_path,
            quality=95,
        )

        print(
            f"Saved: {mask_path}"
        )


    # =====================================================
    # RESET
    # =====================================================

    def reset_polygon(
        self,
        event=None,
    ):

        self.points = []

        self.draw()

        print(
            "Polygon reset"
        )


    # =====================================================
    # SKIP
    # =====================================================

    def skip_image(
        self,
        event=None,
    ):

        print(
            f"Skipped: "
            f"{self.image_path.name}"
        )

        self.next_image()


    # =====================================================
    # EXIT
    # =====================================================

    def exit_program(
        self,
        event=None,
    ):

        print(
            "Разметка остановлена."
        )

        self.root.destroy()


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        """
============================================================

FREE FLOOR POLYGON ANNOTATOR

ЛКМ      -> добавить точку
ПКМ      -> удалить последнюю точку

ENTER    -> сохранить маску
            и перейти к следующему фото

R        -> очистить polygon

E        -> свободного пола НЕТ

F        -> всё изображение является полом

S        -> пропустить изображение

Q / ESC  -> выйти


РАЗМЕЧАЕМ:

ТОЛЬКО ВИДИМЫЙ СВОБОДНЫЙ ПОЛ.


НЕ РАЗМЕЧАЕМ:

- груз
- контейнеры
- коробки
- стены
- потолок
- пространство снаружи кузова

============================================================
"""
    )

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Папка не найдена: "
            f"{IMAGE_DIR}"
        )

    if len(images) == 0:

        raise RuntimeError(
            f"В папке нет изображений: "
            f"{IMAGE_DIR}"
        )

    root = tk.Tk()

    FloorAnnotator(
        root
    )

    root.mainloop()


if __name__ == "__main__":

    main()
