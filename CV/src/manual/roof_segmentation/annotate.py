from __future__ import annotations

from pathlib import Path
import random
import tkinter as tk

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageTk


# =========================================================
# SETTINGS
# =========================================================

# annotate.py находится:
# project/src/manual/roof_segmentation/annotate.py
#
# Поэтому:
# parents[0] = roof_segmentation
# parents[1] = manual
# parents[2] = src
# parents[3] = project
PROJECT_DIR = Path(__file__).resolve().parents[3]


TRAIN_CSV = (
    PROJECT_DIR
    / "DataSet"
    / "train"
    / "train_split.csv"
)

VALIDATION_CSV = (
    PROJECT_DIR
    / "DataSet"
    / "train"
    / "validation_split.csv"
)


# ВАЖНО:
# Floor annotator размечает именно подготовленные ROI.
# Для потолка лучше использовать ТО ЖЕ изображение,
# которое использовалось для floor segmentation.
#
# Если твои ROI действительно лежат здесь:
IMAGE_DIR = (
    PROJECT_DIR
    / "DataSet"
    / "train"
    / "floor_images"
)

# Если хочешь размечать исходные изображения,
# замени выше на:
#
# IMAGE_DIR = (
#     PROJECT_DIR
#     / "DataSet"
#     / "train"
#     / "images"
# )


OUTPUT_DIR = (
    PROJECT_DIR
    / "debug_output"
)

MASK_DIR = (
    OUTPUT_DIR
    / "roof_masks"
)

PREVIEW_DIR = (
    OUTPUT_DIR
    / "roof_previews"
)

SELECTION_CSV = (
    OUTPUT_DIR
    / "roof_selection.csv"
)


MASK_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PREVIEW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRAIN_COUNT = 320
VALIDATION_COUNT = 80

SEED = 42


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


LOAD_BIN_ORDER = [
    "0-10",
    "10-20",
    "20-30",
    "30-40",
    "40-50",
    "50-60",
    "60-70",
    "70-80",
    "80-90",
    "90-100",
]


# =========================================================
# IMAGE SEARCH
# =========================================================


def find_image_by_id(
    image_dir: Path,
    image_id: str,
) -> Path | None:

    for ext in VALID_EXTENSIONS:

        path = (
            image_dir
            / f"{image_id}{ext}"
        )

        if path.exists():
            return path

        upper_path = (
            image_dir
            / f"{image_id}{ext.upper()}"
        )

        if upper_path.exists():
            return upper_path

    return None


# =========================================================
# BALANCED SELECTION
# =========================================================


def balanced_quotas(
    df: pd.DataFrame,
    target: int,
) -> dict[str, int]:

    counts = (
        df["load_bin"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    bins = [
        load_bin
        for load_bin in LOAD_BIN_ORDER
        if load_bin in counts
    ]

    if target > len(df):
        raise ValueError(
            f"Нужно выбрать {target}, "
            f"но в CSV только {len(df)} изображений."
        )

    quotas = {
        load_bin: 0
        for load_bin in bins
    }

    remaining = target

    # По одному добавляем в каждый bin.
    # Если bin закончился, распределяем дальше
    # между оставшимися.
    while remaining > 0:

        changed = False

        minimum = min(
            quotas[b]
            for b in bins
            if quotas[b] < counts[b]
        )

        for load_bin in bins:

            if remaining <= 0:
                break

            if (
                quotas[load_bin] < counts[load_bin]
                and
                quotas[load_bin] == minimum
            ):

                quotas[load_bin] += 1

                remaining -= 1

                changed = True

        if not changed:
            break

    if remaining != 0:
        raise RuntimeError(
            "Не удалось распределить quota."
        )

    return quotas


def select_from_bin(
    df: pd.DataFrame,
    count: int,
    rng: random.Random,
) -> pd.DataFrame:

    if count >= len(df):
        return df.copy()

    df = df.copy()

    # Главное здесь group_id.
    # Сначала пытаемся брать разные группы,
    # а не 10 почти одинаковых фото одной машины.

    groups = []

    for _, group in df.groupby(
        "group_id",
        sort=False,
    ):

        indices = list(group.index)

        rng.shuffle(indices)

        groups.append(indices)

    rng.shuffle(groups)

    selected = []

    depth = 0

    while len(selected) < count:

        added = False

        for group_indices in groups:

            if len(selected) >= count:
                break

            if depth < len(group_indices):

                selected.append(
                    group_indices[depth]
                )

                added = True

        if not added:
            break

        depth += 1

    return df.loc[selected].copy()


def select_split(
    df: pd.DataFrame,
    target: int,
    split_name: str,
    seed: int,
) -> pd.DataFrame:

    quotas = balanced_quotas(
        df,
        target,
    )

    rng = random.Random(seed)

    result_parts = []

    print()
    print(
        f"===== {split_name.upper()} ====="
    )

    for load_bin in LOAD_BIN_ORDER:

        if load_bin not in quotas:
            continue

        subset = df[
            df["load_bin"].astype(str)
            == load_bin
        ].copy()

        wanted = quotas[load_bin]

        selected = select_from_bin(
            subset,
            wanted,
            rng,
        )

        selected["split"] = split_name

        result_parts.append(
            selected
        )

        print(
            f"{load_bin:>6}: "
            f"{len(selected):>3} / "
            f"{len(subset):>3}"
        )

    result = pd.concat(
        result_parts,
        ignore_index=True,
    )

    result = result.sample(
        frac=1,
        random_state=seed,
    ).reset_index(
        drop=True
    )

    return result


def create_selection() -> pd.DataFrame:

    print(
        "Создаю список из 400 изображений..."
    )

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    validation_df = pd.read_csv(
        VALIDATION_CSV
    )

    required = {
        "image_id",
        "load_pct",
        "group_id",
        "load_bin",
    }

    for name, df in [
        ("train", train_df),
        ("validation", validation_df),
    ]:

        missing = (
            required
            - set(df.columns)
        )

        if missing:

            raise ValueError(
                f"{name}: отсутствуют "
                f"колонки {missing}"
            )

    train_selected = select_split(
        train_df,
        TRAIN_COUNT,
        "train",
        SEED,
    )

    validation_selected = select_split(
        validation_df,
        VALIDATION_COUNT,
        "validation",
        SEED + 1,
    )

    selected = pd.concat(
        [
            train_selected,
            validation_selected,
        ],
        ignore_index=True,
    )

    # Перемешиваем train / validation между собой,
    # чтобы не размечать похожие диапазоны подряд.
    selected = selected.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(
        drop=True
    )

    selected.insert(
        0,
        "annotation_order",
        range(len(selected)),
    )

    selected.to_csv(
        SELECTION_CSV,
        index=False,
    )

    print()
    print(
        f"Список сохранён:"
        f"\n{SELECTION_CSV}"
    )

    return selected


def load_selection() -> pd.DataFrame:

    if SELECTION_CSV.exists():

        print(
            f"Использую существующий список:"
            f"\n{SELECTION_CSV}"
        )

        return pd.read_csv(
            SELECTION_CSV
        )

    return create_selection()


# =========================================================
# LOAD SELECTED IMAGES
# =========================================================


selection_df = load_selection()


images: list[Path] = []

metadata_by_stem = {}


for _, row in selection_df.iterrows():

    image_id = str(
        row["image_id"]
    )

    image_path = find_image_by_id(
        IMAGE_DIR,
        image_id,
    )

    if image_path is None:

        print(
            f"[WARNING] Не найдено изображение: "
            f"{image_id}"
        )

        continue

    images.append(
        image_path
    )

    metadata_by_stem[image_path.stem] = {
        "load_pct": row.get(
            "load_pct",
            "",
        ),
        "load_bin": row.get(
            "load_bin",
            "",
        ),
        "split": row.get(
            "split",
            "",
        ),
        "group_id": row.get(
            "group_id",
            "",
        ),
    }


# =========================================================
# ANNOTATOR
# =========================================================


class RoofAnnotator:

    def __init__(
        self,
        root,
    ):

        self.root = root

        self.root.title(
            "Roof / Ceiling Polygon Annotator"
        )

        self.root.geometry(
            "1200x900"
        )


        # =================================================
        # STATE
        # =================================================

        self.image_index = -1

        self.image_path = None

        self.original_image = None

        self.tk_image = None

        # Текущий polygon
        self.points = []

        # Уже законченные части потолка.
        #
        # Нужны, если груз разделяет потолок
        # на несколько видимых областей.
        self.polygons = []

        self.scale = 1.0

        self.offset_x = 0
        self.offset_y = 0


        # =================================================
        # INFO
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
        # BUTTONS
        # =================================================
        #
        # Сделал кнопки специально,
        # чтобы даже если Windows/Tkinter
        # когда-нибудь не поймает клавишу,
        # EMPTY всегда можно нажать мышкой.

        button_frame = tk.Frame(
            root
        )

        button_frame.pack(
            pady=4
        )


        tk.Button(
            button_frame,
            text="Добавить часть [A]",
            command=self.add_polygon,
        ).pack(
            side=tk.LEFT,
            padx=4,
        )


        tk.Button(
            button_frame,
            text="Сохранить [Enter]",
            command=self.save_annotation,
        ).pack(
            side=tk.LEFT,
            padx=4,
        )


        tk.Button(
            button_frame,
            text="Потолка нет [E]",
            command=self.save_empty_mask,
        ).pack(
            side=tk.LEFT,
            padx=4,
        )


        tk.Button(
            button_frame,
            text="Очистить [R]",
            command=self.reset_polygon,
        ).pack(
            side=tk.LEFT,
            padx=4,
        )


        tk.Button(
            button_frame,
            text="Пропустить [S]",
            command=self.skip_image,
        ).pack(
            side=tk.LEFT,
            padx=4,
        )


        # =================================================
        # HELP
        # =================================================

        help_text = (
            "ЛКМ: точка | "
            "ПКМ: удалить точку | "
            "A: добавить часть потолка | "
            "ENTER: сохранить | "
            "E: потолка нет | "
            "U: удалить последнюю часть | "
            "R: очистить | "
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
            self.save_annotation,
        )

        self.root.bind(
            "<Key-a>",
            self.add_polygon,
        )

        self.root.bind(
            "<Key-A>",
            self.add_polygon,
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
            "<Key-r>",
            self.reset_polygon,
        )

        self.root.bind(
            "<Key-R>",
            self.reset_polygon,
        )

        self.root.bind(
            "<Key-u>",
            self.undo_polygon,
        )

        self.root.bind(
            "<Key-U>",
            self.undo_polygon,
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


        self.canvas.bind(
            "<Configure>",
            self.on_resize,
        )


        # Чтобы E / R / A и другие клавиши
        # сразу работали после запуска.
        self.root.after(
            200,
            self.force_keyboard_focus,
        )


        # =================================================
        # START
        # =================================================

        self.next_image()


    # =====================================================
    # KEYBOARD FOCUS
    # =====================================================

    def force_keyboard_focus(
        self,
    ):

        try:

            self.root.focus_force()

            self.canvas.focus_set()

        except tk.TclError:

            pass


    # =====================================================
    # NEXT IMAGE
    # =====================================================

    def next_image(
        self,
    ):

        self.points = []

        self.polygons = []

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


            # Уже сохранённые маски
            # автоматически пропускаем.
            if mask_path.exists():

                print(
                    f"[SKIP EXISTING] "
                    f"{image_path.name}"
                )

                continue


            self.image_path = (
                image_path
            )

            break


        try:

            self.original_image = (
                Image.open(
                    self.image_path
                )
                .convert(
                    "RGB"
                )
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

        self.root.after(
            50,
            self.force_keyboard_focus,
        )


    # =====================================================
    # INFO
    # =====================================================

    def update_info(
        self,
    ):

        if self.image_path is None:
            return

        meta = metadata_by_stem.get(
            self.image_path.stem,
            {},
        )

        text = (
            f"{self.image_index + 1}/"
            f"{len(images)} | "
            f"{self.image_path.name} | "
            f"load={meta.get('load_pct', '')} | "
            f"bin={meta.get('load_bin', '')} | "
            f"{meta.get('split', '')} | "
            f"точек={len(self.points)} | "
            f"частей={len(self.polygons)}"
        )

        self.info_label.config(
            text=text
        )


    # =====================================================
    # RESIZE
    # =====================================================

    def on_resize(
        self,
        event,
    ):

        if (
            self.original_image
            is not None
        ):

            self.draw()


    # =====================================================
    # DRAW
    # =====================================================

    def draw(
        self,
    ):

        if (
            self.original_image
            is None
        ):
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


        overlay = display.copy()

        overlay_draw = ImageDraw.Draw(
            overlay,
            "RGBA",
        )


        # =================================================
        # SAVED PARTS
        # =================================================

        for polygon in self.polygons:

            scaled = [
                (
                    int(x * self.scale),
                    int(y * self.scale),
                )
                for x, y in polygon
            ]

            if len(scaled) >= 3:

                overlay_draw.polygon(
                    scaled,
                    fill=(
                        0,
                        255,
                        0,
                        70,
                    ),
                    outline=(
                        0,
                        255,
                        0,
                        255,
                    ),
                )


        # =================================================
        # CURRENT POLYGON
        # =================================================

        scaled_points = [
            (
                int(x * self.scale),
                int(y * self.scale),
            )
            for x, y in self.points
        ]


        if len(scaled_points) >= 3:

            overlay_draw.polygon(
                scaled_points,
                fill=(
                    255,
                    255,
                    0,
                    45,
                ),
                outline=(
                    255,
                    255,
                    0,
                    255,
                ),
            )

        elif len(
            scaled_points
        ) >= 2:

            overlay_draw.line(
                scaled_points,
                fill=(
                    255,
                    255,
                    0,
                    255,
                ),
                width=2,
            )


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
                    255,
                    255,
                ),
            )


        display = overlay


        self.tk_image = (
            ImageTk.PhotoImage(
                display
            )
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
    # CANVAS -> IMAGE COORDS
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
            x / self.scale
        )

        y = int(
            y / self.scale
        )


        width, height = (
            self.original_image.size
        )


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
    # ADD POLYGON PART
    # =====================================================

    def add_polygon(
        self,
        event=None,
    ):

        if len(
            self.points
        ) < 3:

            print(
                "Для части потолка "
                "нужно минимум 3 точки."
            )

            return


        self.polygons.append(
            self.points.copy()
        )

        self.points = []


        print(
            f"Добавлена часть потолка. "
            f"Всего частей: "
            f"{len(self.polygons)}"
        )


        self.draw()


    # =====================================================
    # CREATE MASK
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


        for polygon in self.polygons:

            if len(polygon) >= 3:

                draw.polygon(
                    polygon,
                    fill=255,
                )


        return mask


    # =====================================================
    # SAVE ANNOTATION
    # =====================================================

    def save_annotation(
        self,
        event=None,
    ):

        # Если текущий polygon ещё не был
        # добавлен через A,
        # автоматически добавляем его.
        if self.points:

            if len(
                self.points
            ) < 3:

                print(
                    "Нужно минимум 3 точки."
                )

                return


            self.polygons.append(
                self.points.copy()
            )

            self.points = []


        if not self.polygons:

            print(
                "Потолок не размечен. "
                "Если его нет, нажми E."
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
    # EMPTY MASK
    # =====================================================

    def save_empty_mask(
        self,
        event=None,
    ):

        if (
            self.original_image
            is None
        ):
            return


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
            "EMPTY ROOF MASK"
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


        roof_pixels = (
            mask_array > 0
        )


        overlay_array[
            roof_pixels
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

        self.polygons = []


        self.draw()


        print(
            "Roof polygons reset"
        )


    # =====================================================
    # UNDO COMPLETED POLYGON
    # =====================================================

    def undo_polygon(
        self,
        event=None,
    ):

        if self.points:

            self.points = []

            print(
                "Текущая часть удалена."
            )

        elif self.polygons:

            self.points = (
                self.polygons.pop()
            )

            print(
                "Последняя часть возвращена "
                "для редактирования."
            )

        self.draw()


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

ROOF / CEILING POLYGON ANNOTATOR

ЛКМ      -> добавить точку
ПКМ      -> удалить последнюю точку

A        -> закончить текущую часть потолка
            и начать следующую

ENTER    -> сохранить всю маску
            и перейти к следующему фото

E        -> потолка НЕ ВИДНО
            сохранить пустую маску

U        -> удалить / вернуть последнюю часть

R        -> очистить всю текущую разметку

S        -> пропустить изображение

Q / ESC  -> выйти


РАЗМЕЧАЕМ:

ТОЛЬКО РЕАЛЬНО ВИДИМУЮ ПОВЕРХНОСТЬ ПОТОЛКА.


НЕ РАЗМЕЧАЕМ:

- груз
- коробки
- контейнеры
- стены
- пол
- скрытый за грузом потолок
- пространство снаружи кузова

ВАЖНО:

Если груз разделяет видимый потолок на две части:

1. размечаем первую
2. нажимаем A
3. размечаем вторую
4. нажимаем ENTER

============================================================
"""
    )


    if not TRAIN_CSV.exists():

        raise FileNotFoundError(
            f"Не найден:"
            f"\n{TRAIN_CSV}"
        )


    if not VALIDATION_CSV.exists():

        raise FileNotFoundError(
            f"Не найден:"
            f"\n{VALIDATION_CSV}"
        )


    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Папка с изображениями "
            f"не найдена:"
            f"\n{IMAGE_DIR}"
        )


    if len(images) == 0:

        raise RuntimeError(
            "Не найдено ни одного "
            "из выбранных изображений."
        )


    print(
        f"Выбрано изображений: "
        f"{len(selection_df)}"
    )

    print(
        f"Найдено изображений: "
        f"{len(images)}"
    )

    print(
        f"Маски:"
        f"\n{MASK_DIR}"
    )

    print(
        f"Preview:"
        f"\n{PREVIEW_DIR}"
    )


    root = tk.Tk()


    RoofAnnotator(
        root
    )


    root.mainloop()


if __name__ == "__main__":

    main()