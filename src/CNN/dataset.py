"""Dataset и аугментации для end-to-end CNN-регрессии загрузки кузова.

Изображение подаётся в модель целиком (letterbox до квадрата), без
геометрического кропа по углам кузова — corner-detection на реальных фото
регулярно даёт кривой/неполный контур и теряет часть груза из кадра.
Сеть с аугментациями учится сама фокусироваться на полу и грузе.
"""

from pathlib import Path

import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Фиксированный порядок категорий для aux-голов. Порядок должен совпадать
# при обучении и инференсе, поэтому он захардкожен, а не берётся из данных.
LOAD_BIN_CLASSES = [
    "0-10", "10-20", "20-30", "30-40", "40-50",
    "50-60", "60-70", "70-80", "80-90", "90-100",
]
CARGO_TYPE_CLASSES = [
    "blue_full_container",
    "blue_open_containers",
    "boxes",
    "mixed",
    "pallets",
]

LOAD_BIN_TO_IDX = {name: idx for idx, name in enumerate(LOAD_BIN_CLASSES)}
CARGO_TYPE_TO_IDX = {name: idx for idx, name in enumerate(CARGO_TYPE_CLASSES)}


def read_rgb_image(path):
    """Читаем изображение и возвращаем RGB-массив; None -> понятная ошибка."""
    bgr_image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr_image is None:
        raise ValueError(f"Could not decode image: {path}")
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)


def _letterbox_steps(img_size):
    """LongestMaxSize + PadIfNeeded сохраняют аспект без искажения перспективы
    (в отличие от прямого resize до квадрата), что важно для кузова, снятого
    вдоль оси движения."""
    return [
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=(0, 0, 0),
        ),
    ]


def build_transforms(train, img_size):
    if not train:
        return A.Compose(
            _letterbox_steps(img_size)
            + [
                A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                ToTensorV2(),
            ]
        )

    return A.Compose(
        _letterbox_steps(img_size)
        + [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.35, contrast_limit=0.35, p=0.8
            ),
            A.RandomGamma(gamma_limit=(70, 150), p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15, p=0.4
            ),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=7,
                border_mode=cv2.BORDER_CONSTANT,
                fill=(0, 0, 0),
                p=0.5,
            ),
            A.Perspective(scale=(0.02, 0.06), p=0.3),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(1, max(1, int(img_size * 0.12))),
                hole_width_range=(1, max(1, int(img_size * 0.12))),
                fill=0,
                p=0.3,
            ),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


class TruckLoadDataset(Dataset):
    """rows — список dict с колонками image_id,load_pct,load_bin,cargo_type,...

    Формат строк совместим с выводом cv_split.py (train_metadata.csv +
    колонка fold, прочитанная через pandas.DataFrame.to_dict("records")).
    """

    def __init__(self, rows, image_dir, img_size, train):
        self.rows = rows
        self.image_dir = Path(image_dir)
        self.transform = build_transforms(train=train, img_size=img_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self.image_dir / f"{row['image_id']}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = read_rgb_image(image_path)
        image = self.transform(image=image)["image"]

        load_pct = float(row["load_pct"])
        load_bin_idx = LOAD_BIN_TO_IDX[row["load_bin"]]
        cargo_type_idx = CARGO_TYPE_TO_IDX[row["cargo_type"]]

        return {
            "image": image,
            "load_pct": torch.tensor(load_pct, dtype=torch.float32),
            "load_bin": torch.tensor(load_bin_idx, dtype=torch.long),
            "cargo_type": torch.tensor(cargo_type_idx, dtype=torch.long),
            "image_id": row["image_id"],
        }


class InferenceDataset(Dataset):
    """Для test.csv: только image_id, без разметки. hflip=True — TTA-вариант
    с горизонтальным отражением."""

    def __init__(self, rows, image_dir, img_size, hflip=False):
        self.rows = rows
        self.image_dir = Path(image_dir)

        steps = _letterbox_steps(img_size)
        if hflip:
            steps.append(A.HorizontalFlip(p=1.0))
        steps += [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
        self.transform = A.Compose(steps)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = self.image_dir / f"{row['image_id']}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = read_rgb_image(image_path)
        image = self.transform(image=image)["image"]
        return {"image": image, "image_id": row["image_id"]}
