import cv2
import numpy as np
from pathlib import Path


# =========================
# SETTINGS
# =========================

PROJECT_DIR = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_DIR / "DataSet" / "train" / "images"
MASK_DIR = PROJECT_DIR / "debug_output" / "masks"
PREVIEW_DIR = PROJECT_DIR / "debug_output" / "previews"

MASK_DIR.mkdir(parents=True, exist_ok=True)
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

images = sorted([
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in VALID_EXTENSIONS
])


# =========================
# GLOBAL STATE
# =========================

points = []
current_image = None
display_image = None

WINDOW_NAME = "Truck Polygon Annotator"


# =========================
# DRAW POLYGON
# =========================

def redraw():
    """
    Перерисовывает изображение + выбранные точки + polygon.
    """
    global display_image

    display_image = current_image.copy()

    # Рисуем точки
    for i, point in enumerate(points):
        cv2.circle(
            display_image,
            point,
            6,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            display_image,
            str(i + 1),
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    # Соединяем точки линиями
    if len(points) >= 2:
        pts = np.array(points, dtype=np.int32)

        cv2.polylines(
            display_image,
            [pts],
            False,
            (0, 255, 0),
            2
        )

    # Если >= 3 точек, показываем замкнутый polygon
    if len(points) >= 3:
        pts = np.array(points, dtype=np.int32)

        overlay = display_image.copy()

        cv2.fillPoly(
            overlay,
            [pts],
            (0, 255, 0)
        )

        display_image = cv2.addWeighted(
            overlay,
            0.20,
            display_image,
            0.80,
            0
        )

        cv2.polylines(
            display_image,
            [pts],
            True,
            (0, 255, 0),
            2
        )

    cv2.imshow(WINDOW_NAME, display_image)


# =========================
# MOUSE
# =========================

def mouse_callback(event, x, y, flags, param):
    global points

    # ЛКМ -> добавить точку
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw()

    # ПКМ -> удалить последнюю точку
    elif event == cv2.EVENT_RBUTTONDOWN:

        if points:
            points.pop()
            redraw()


# =========================
# SAVE POLYGON MASK
# =========================

def save_polygon(image_path):
    """
    Создает бинарную маску:
    255 = внутри кузова
    0   = всё снаружи
    """

    if len(points) < 3:
        print("Нужно минимум 3 точки.")
        return False

    h, w = current_image.shape[:2]

    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    polygon = np.array(
        points,
        dtype=np.int32
    )

    cv2.fillPoly(
        mask,
        [polygon],
        255
    )

    mask_path = MASK_DIR / f"{image_path.stem}.png"

    cv2.imwrite(
        str(mask_path),
        mask
    )

    # -------------------------
    # Preview
    # -------------------------

    preview = current_image.copy()

    # Всё вне кузова -> черное
    result = current_image.copy()
    result[mask == 0] = 0

    # polygon для проверки
    cv2.polylines(
        preview,
        [polygon],
        True,
        (0, 255, 0),
        3
    )

    # оригинал + результат
    preview = np.hstack([
        preview,
        result
    ])

    preview_path = PREVIEW_DIR / f"{image_path.stem}_preview.jpg"

    cv2.imwrite(
        str(preview_path),
        preview
    )

    print(f"Saved: {mask_path}")

    return True


# =========================
# FULL IMAGE MASK
# =========================

def save_full_mask(image_path):
    """
    Для INSIDE-фотографии.

    Всё изображение считается полезной областью.
    """

    h, w = current_image.shape[:2]

    mask = np.full(
        (h, w),
        255,
        dtype=np.uint8
    )

    mask_path = MASK_DIR / f"{image_path.stem}.png"

    cv2.imwrite(
        str(mask_path),
        mask
    )

    print(f"FULL MASK: {mask_path}")


# =========================
# MAIN
# =========================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


print("""
========================================
TRUCK POLYGON ANNOTATOR

ЛКМ      -> добавить точку
ПКМ      -> удалить последнюю точку

ENTER    -> сохранить polygon и следующее фото
F        -> INSIDE фото: всё изображение = mask
R        -> очистить polygon
S        -> пропустить фотографию
Q / ESC  -> выйти

========================================
""")


for index, image_path in enumerate(images):

    mask_path = MASK_DIR / f"{image_path.stem}.png"

    # Уже размеченные автоматически пропускаем
    if mask_path.exists():
        print(f"[SKIP EXISTING] {image_path.name}")
        continue

    current_image = cv2.imread(
        str(image_path)
    )

    if current_image is None:
        print(f"Не удалось открыть: {image_path}")
        continue

    points = []

    print()
    print(
        f"[{index + 1}/{len(images)}] "
        f"{image_path.name}"
    )

    redraw()

    while True:

        key = cv2.waitKey(0) & 0xFF

        # ENTER
        if key in [10, 13]:

            if save_polygon(image_path):
                break

        # F -> full image mask
        elif key == ord("f"):

            save_full_mask(image_path)
            break

        # R -> reset
        elif key == ord("r"):

            points = []
            redraw()

        # S -> skip
        elif key == ord("s"):

            print("Skipped")
            break

        # Q / ESC -> exit
        elif key == ord("q") or key == 27:

            cv2.destroyAllWindows()
            exit()


cv2.destroyAllWindows()

print("Разметка закончена.")