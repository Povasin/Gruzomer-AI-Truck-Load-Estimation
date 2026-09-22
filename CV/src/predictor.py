"""Production-ready inference module for XML API service integration.

Usage:
    from predictor import TruckLoadPredictor, InvalidImageError

    # Initialize once on API startup:
    predictor = TruckLoadPredictor(weights_path="models/cnn_v1")

    # Predict single image (file path, bytes, or PIL Image):
    try:
        result = predictor.predict("test_photo.jpg")
        print(f"Load: {result['load_pct']:.1f}%")
        print(f"Cargo: {result['cargo_type']}")
    except InvalidImageError as e:
        print(f"Error for XML API: {e}")
"""

import io
from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

from CNN.model import TruckLoadNet

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CARGO_TYPES = [
    "blue_open_containers",
    "pallets",
    "blue_full_container",
    "boxes",
    "mixed",
]


class InvalidImageError(ValueError):
    """Исключение при передаче повреждённого или неподдерживаемого файла."""
    pass


class TruckLoadPredictor:
    """Удобный предиктор для сервиса и API."""

    def __init__(
        self,
        weights_path: Union[str, Path] = "models/cnn_v1",
        device: str = "auto",
        use_tta: bool = True,
        use_clahe_tta: bool = True,
        calibrate: bool = True,
    ):
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        self.use_tta = use_tta
        self.use_clahe_tta = use_clahe_tta
        self.calibrate = calibrate
        self.models: List[nn.Module] = []
        self.img_size = 384
        self.backbone_name = "convnext_tiny"

        weights_path = Path(weights_path)
        checkpoints = []
        if weights_path.is_dir():
            checkpoints = sorted(list(weights_path.glob("fold*_best.pt")))
            if not checkpoints:
                checkpoints = sorted(list(weights_path.glob("*.pt")))
        elif weights_path.is_file() and weights_path.suffix.lower() == ".pt":
            checkpoints = [weights_path]

        if not checkpoints:
            raise FileNotFoundError(f"Чекпоинты моделей не найдены по пути: {weights_path}")

        for cp in checkpoints:
            ckpt = torch.load(cp, map_location=self.device, weights_only=False)
            bb = ckpt.get("backbone", "convnext_tiny")
            self.backbone_name = bb
            self.img_size = ckpt.get("img_size", 384)
            state = ckpt.get("model_state_dict", {})
            head_type = ckpt.get("head_type", "distributional" if "dist_head.weight" in state else "scalar")

            model = TruckLoadNet(backbone_name=bb, pretrained=False, head_type=head_type)
            model.load_state_dict(state, strict=False)
            model.to(self.device).eval()
            self.models.append(model)

        self._init_transforms()

    def _init_transforms(self):
        letterbox = [
            A.LongestMaxSize(max_size=self.img_size),
            A.PadIfNeeded(
                min_height=self.img_size,
                min_width=self.img_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            ),
        ]
        norm = [
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
        self.transform_std = A.Compose(letterbox + norm)
        self.transform_clahe = A.Compose(
            [A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0)]
            + letterbox
            + norm
        )

    def _load_image(self, image_input: Union[str, Path, bytes, Image.Image, np.ndarray]) -> np.ndarray:
        try:
            if isinstance(image_input, (str, Path)):
                path = Path(image_input)
                if not path.is_file():
                    raise InvalidImageError(f"Файл не найден: {path}")
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img)
                    return np.array(img.convert("RGB"), dtype=np.uint8)

            elif isinstance(image_input, bytes):
                with Image.open(io.BytesIO(image_input)) as img:
                    img = ImageOps.exif_transpose(img)
                    return np.array(img.convert("RGB"), dtype=np.uint8)

            elif isinstance(image_input, Image.Image):
                img = ImageOps.exif_transpose(image_input)
                return np.array(img.convert("RGB"), dtype=np.uint8)

            elif isinstance(image_input, np.ndarray):
                if image_input.ndim != 3 or image_input.shape[2] != 3:
                    raise InvalidImageError("Массив изображения должен иметь форму [H, W, 3]")
                return image_input.astype(np.uint8)

            else:
                raise InvalidImageError(f"Неподдерживаемый тип входных данных: {type(image_input)}")

        except Exception as e:
            if isinstance(e, InvalidImageError):
                raise e
            raise InvalidImageError(f"Ошибка чтения/декодирования изображения: {e}")

    @torch.no_grad()
    def predict(self, image_input: Union[str, Path, bytes, Image.Image, np.ndarray]) -> Dict[str, Union[float, str]]:
        rgb_img = self._load_image(image_input)
        x = self.transform_std(image=rgb_img)["image"].unsqueeze(0).to(self.device)

        x_clahe = (
            self.transform_clahe(image=rgb_img)["image"].unsqueeze(0).to(self.device)
            if self.use_clahe_tta
            else None
        )

        all_pcts = []
        cargo_logits_list = []

        for model in self.models:
            # Прямой проход
            p1, _, cargo_log = model(x)
            cargo_logits_list.append(cargo_log)

            if self.use_tta:
                x_flip = torch.flip(x, dims=[-1])
                p2 = model(x_flip)[0]

                if x_clahe is not None:
                    p3 = model(x_clahe)[0]
                    p4 = model(torch.flip(x_clahe, dims=[-1]))[0]
                    p_model = 0.30 * p1 + 0.30 * p2 + 0.20 * p3 + 0.20 * p4
                else:
                    p_model = 0.5 * (p1 + p2)
            else:
                p_model = p1

            all_pcts.append(p_model.item())

        mean_pct = float(np.mean(all_pcts))

        # Калибровка экстремальных значений
        if self.calibrate:
            if mean_pct <= 3.5:
                mean_pct = 0.0
            elif mean_pct >= 94.5:
                mean_pct = 100.0
            else:
                mean_pct = float(np.clip(mean_pct, 0.0, 100.0))

        # Определение типа груза из aux-головы
        mean_cargo_logits = torch.mean(torch.stack(cargo_logits_list), dim=0)
        cargo_idx = int(torch.argmax(mean_cargo_logits, dim=-1)[0].item())
        cargo_type_name = CARGO_TYPES[cargo_idx] if cargo_idx < len(CARGO_TYPES) else "mixed"

        return {
            "load_pct": round(mean_pct, 2),
            "cargo_type": cargo_type_name,
            "models_count": len(self.models),
            "backbone": self.backbone_name,
        }
