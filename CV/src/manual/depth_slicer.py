"""Оценка загрузки кузова методом 20 пространственных срезов на базе нейросети глубины MiDaS.

Путь 1: Честная нейросетевая карта глубины (Monocular Depth Estimation).
- Нейросеть MiDaS (torch.hub: intel-isl/MiDaS, MiDaS_small) строит относительную карту глубины Z для каждого пикселя.
- Внутреннее пространство кузова (truck_mask) нарезается на 20 диапазонов расстояний Z (от дверей вглубь к кабине).
- На каждом шаге глубины строится полное сечение от потолка к полу и замеряется высота груза.
- Если на срезе зафиксирован высокий груз (>= 75%), срабатывает правило ранней остановки:
  все последующие срезы в глубине кузова помечаются как 100% заполненные.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import torch

from manual.features import (
    read_rgb_image,
    clean_floor_mask,
    clean_ceiling_mask,
    safe_ratio,
)
from manual.floor_segmentation.runtime import predict_floor_mask
from manual.roof_segmentation.runtime import predict_ceiling_mask
from manual.truck_segmentation.runtime import predict_truck_mask

NUM_SLICES = 20
OCCLUSION_THRESHOLD = 0.75

_MIDAS_MODEL = None
_MIDAS_TRANSFORM = None
_MIDAS_DEVICE = None


def get_midas(device: Optional[str] = None):
    """Загружает и кэширует модель нейросети глубины MiDaS_small."""
    global _MIDAS_MODEL, _MIDAS_TRANSFORM, _MIDAS_DEVICE

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _MIDAS_MODEL is None or _MIDAS_DEVICE != device:
        print(f"[MIDAS] Загрузка нейросети глубины MiDaS_small на {device}...")
        _MIDAS_MODEL = torch.hub.load(
            "intel-isl/MiDaS",
            "MiDaS_small",
            trust_repo=True,
        ).to(device).eval()

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True,
        )
        _MIDAS_TRANSFORM = midas_transforms.small_transform
        _MIDAS_DEVICE = device
        print("[MIDAS] Модель глубины готова к работе.")

    return _MIDAS_MODEL, _MIDAS_TRANSFORM, _MIDAS_DEVICE


def predict_depth_map(image: np.ndarray, device: Optional[str] = None) -> np.ndarray:
    """Вычисляет карту относительной глубины для изображения с помощью нейросети MiDaS."""
    model, transform, dev = get_midas(device)
    input_batch = transform(image).to(dev)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    return prediction.cpu().numpy()


@dataclass
class SliceFrame:
    """Геометрия одного вертикального среза в 2D-координатах кадра."""
    index: int
    y_ceil: int
    y_floor: int
    x_left: int
    x_right: int
    x_center: int
    fill: float


@dataclass
class SlicingResult:
    """Результат анализа кузова методом 20 срезов с нейросетью глубины."""
    load_pct: float
    slice_fills: np.ndarray       # shape (NUM_SLICES,), values in [0.0, 1.0]
    stop_index: int               # индекс среза, где сработал останов (-1 если нет)
    frames: List[SliceFrame]      # геометрия 20 сечений от дверей вглубь
    depth_map: np.ndarray         # нормализованная карта глубины Z (H, W)


def compute_neural_depth_slices(
    image: np.ndarray,
    floor_mask: np.ndarray,
    ceiling_mask: np.ndarray,
    truck_mask: np.ndarray,
    num_slices: int = NUM_SLICES,
    occlusion_threshold: float = OCCLUSION_THRESHOLD,
    device: Optional[str] = None,
) -> SlicingResult:
    """Вычисляет профиль 20 срезов с использованием нейросетевой карты глубины MiDaS."""
    truck = (truck_mask > 0).astype(np.uint8)
    floor = clean_floor_mask((floor_mask > 0).astype(np.uint8), truck)
    ceiling = clean_ceiling_mask((ceiling_mask > 0).astype(np.uint8), truck)

    h, w = truck.shape

    # 1. Получаем предсказание глубины от нейросети MiDaS
    disparity = predict_depth_map(image, device=device)

    # Нормализуем глубину Z внутри кузова в диапазон [0.0 (двери), 1.0 (кабина)]
    truck_disp = disparity[truck > 0]
    if len(truck_disp) == 0:
        return SlicingResult(
            load_pct=0.0,
            slice_fills=np.zeros(num_slices, dtype=np.float64),
            stop_index=-1,
            frames=[],
            depth_map=np.zeros((h, w), dtype=np.float32),
        )

    p2, p98 = np.percentile(truck_disp, 2), np.percentile(truck_disp, 98)
    z_map = np.clip((p98 - disparity) / max(1e-5, p98 - p2), 0.0, 1.0)

    # 2. Нарезаем кузов на 20 равных шагов по нейросетевой глубине Z
    slice_fills = []
    frames = []
    stop_idx = -1

    # Также находим общую геометрию кузова для опорных сечений
    truck_rows, truck_cols = np.where(truck > 0)
    y_top, y_bot = int(truck_rows.min()), int(truck_rows.max())
    x_left, x_right = int(truck_cols.min()), int(truck_cols.max())
    h_truck = max(1, y_bot - y_top)
    x_center = (x_left + x_right) / 2.0
    y_horizon = y_top + 0.45 * h_truck
    z_ratio = 4.0

    for k in range(num_slices):
        z_low = k / num_slices
        z_high = (k + 1) / num_slices

        # Пиксели, отнесённые нейросетью к данному шагу глубины внутри кузова
        slice_mask = (z_map >= z_low) & (z_map < z_high) & (truck > 0)
        slice_px = np.count_nonzero(slice_mask)

        # Геометрия створки кузова в перспективе для этого среза
        u = (k + 0.5) / num_slices
        t_k = 1.0 / (1.0 + (z_ratio - 1.0) * u)
        y_c = int(np.clip(round(y_horizon + (y_top - y_horizon) * t_k), 0, h - 1))
        y_f = int(np.clip(round(y_horizon + (y_bot - y_horizon) * t_k), 0, h - 1))
        x_l = int(np.clip(round(x_center + (x_left - x_center) * t_k), 0, w - 1))
        x_r = int(np.clip(round(x_center + (x_right - x_center) * t_k), 0, w - 1))
        xc = (x_l + x_r) // 2
        slice_h = max(1, y_f - y_c)

        # Если пикселей в этом слое глубины мало, используем геометрический срез
        if slice_px < 30:
            y_floor_band = int(max(y_c, y_f - 0.25 * slice_h))
            floor_sub = floor[y_floor_band : y_f + 1, x_l : x_r + 1]
            truck_sub = truck[y_floor_band : y_f + 1, x_l : x_r + 1]
            free_floor_ratio = safe_ratio(np.count_nonzero(floor_sub), np.count_nonzero(truck_sub))
            is_empty = free_floor_ratio > 0.25
        else:
            # Анализируем пиксели пола именно на этой нейросетевой глубине
            floor_in_slice = np.count_nonzero(slice_mask & (floor > 0))
            is_empty = (floor_in_slice / slice_px) > 0.30

        if is_empty:
            fill_val = 0.0
            slice_fills.append(0.0)
            frames.append(SliceFrame(k, y_c, y_f, x_l, x_r, xc, 0.0))
        else:
            # Пол закрыт: замеряем высоту груза от пола к потолку на этом сечении
            cargo_sub = (
                (floor[y_c : y_f + 1, x_l : x_r + 1] == 0)
                & (ceiling[y_c : y_f + 1, x_l : x_r + 1] == 0)
                & (truck[y_c : y_f + 1, x_l : x_r + 1] > 0)
            )
            cargo_ratio = safe_ratio(
                np.count_nonzero(cargo_sub),
                np.count_nonzero(truck[y_c : y_f + 1, x_l : x_r + 1]),
            )

            # Проверка ранней остановки (окклюзия: стена груза)
            if cargo_ratio >= occlusion_threshold and stop_idx == -1:
                stop_idx = k
                fill_val = 1.0
                slice_fills.append(1.0)
                slice_fills.extend([1.0] * (num_slices - 1 - k))
                frames.append(SliceFrame(k, y_c, y_f, x_l, x_r, xc, 1.0))
                for rem_k in range(k + 1, num_slices):
                    rem_u = (rem_k + 0.5) / num_slices
                    rem_tk = 1.0 / (1.0 + (z_ratio - 1.0) * rem_u)
                    ry_c = int(np.clip(round(y_horizon + (y_top - y_horizon) * rem_tk), 0, h - 1))
                    ry_f = int(np.clip(round(y_horizon + (y_bot - y_horizon) * rem_tk), 0, h - 1))
                    rx_l = int(np.clip(round(x_center + (x_left - x_center) * rem_tk), 0, w - 1))
                    rx_r = int(np.clip(round(x_center + (x_right - x_center) * rem_tk), 0, w - 1))
                    frames.append(SliceFrame(rem_k, ry_c, ry_f, rx_l, rx_r, (rx_l + rx_r) // 2, 1.0))
                break
            else:
                fill_val = float(np.clip(cargo_ratio, 0.0, 1.0))
                slice_fills.append(fill_val)
                frames.append(SliceFrame(k, y_c, y_f, x_l, x_r, xc, fill_val))

    load_pct = float(np.clip(np.mean(slice_fills) * 100.0, 0.0, 100.0))

    return SlicingResult(
        load_pct=load_pct,
        slice_fills=np.array(slice_fills, dtype=np.float64),
        stop_index=stop_idx,
        frames=frames,
        depth_map=z_map,
    )


def analyze_image(
    image_path: Path | str,
    truck_weights: Optional[Path | str] = None,
    floor_weights: Optional[Path | str] = None,
    ceiling_weights: Optional[Path | str] = None,
    occlusion_threshold: float = OCCLUSION_THRESHOLD,
    device: Optional[str] = None,
) -> Tuple[SlicingResult, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Полный пайплайн анализа одного изображения: сегментация + нейросеть глубины MiDaS + 20 срезов."""
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
    ceiling_mask = predict_ceiling_mask(
        image,
        weights_path=ceiling_weights,
    )

    result = compute_neural_depth_slices(
        image=image,
        floor_mask=floor_mask,
        ceiling_mask=ceiling_mask,
        truck_mask=truck_mask,
        num_slices=NUM_SLICES,
        occlusion_threshold=occlusion_threshold,
        device=device,
    )

    return result, image, truck_mask, floor_mask, ceiling_mask
