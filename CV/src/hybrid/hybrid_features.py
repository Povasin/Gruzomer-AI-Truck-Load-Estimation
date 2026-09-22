"""Проверка совместимости и кэш 349 признаков; без импорта нейросетей."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile

import numpy as np

from manual.feature_schema import FEATURE_VERSION, TOTAL_FEATURE_DIM

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIM = TOTAL_FEATURE_DIM
DEFAULT_TRUCK = ROOT / "src/manual/truck_segmentation/models/best_unet_resnet18.pth"
DEFAULT_FLOOR = ROOT / "src/manual/floor_segmentation/models/floor_unet_resnet18_lr1e3.best_loss.pt"
DEFAULT_CEILING = ROOT / "src/manual/roof_segmentation/models/ceiling_unet_resnet18.best_iou.pt"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def resolve_saved_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def get_package_version(name):
    try:
        return version(name)
    except Exception:
        if name == "opencv-python":
            for alt in (
                "opencv-python-headless",
                "opencv-contrib-python",
                "opencv-contrib-python-headless",
            ):
                try:
                    return version(alt)
                except Exception:
                    pass
        raise


def build_contract(
    truck_weights=None,
    floor_weights=None,
    ceiling_weights=None,
):
    """
    Строит контракт полного 349-dimensional feature pipeline.

    Контракт зависит от:
      - весов truck segmentation;
      - весов floor segmentation;
      - весов ceiling segmentation;
      - кода feature extraction/runtime/model;
      - версий ключевых библиотек.

    Поэтому старый кэш 277 признаков автоматически получает другую signature
    и не может быть случайно переиспользован.
    """
    truck = Path(truck_weights or DEFAULT_TRUCK).resolve()
    floor = Path(floor_weights or DEFAULT_FLOOR).resolve()
    ceiling = Path(ceiling_weights or DEFAULT_CEILING).resolve()

    for path in (truck, floor, ceiling):
        if not path.is_file():
            raise FileNotFoundError(
                f"Для гибрида нужны обученные веса сегментации: {path}"
            )

    sources = (
        "manual/feature_schema.py",
        "manual/features.py",
        "manual/truck_segmentation/runtime.py",
        "manual/floor_segmentation/runtime.py",
        "manual/floor_segmentation/model.py",
        "manual/roof_segmentation/runtime.py",
        "manual/roof_segmentation/model.py",
        "hybrid/hybrid_features.py",
        "CNN/dataset.py",
    )

    source_sha256 = {}
    for name in sources:
        source_path = ROOT / "src" / name
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Не найден исходник для контракта: {source_path}"
            )
        source_sha256[name] = sha256_file(source_path)

    fingerprint = {
        "feature_version": FEATURE_VERSION,
        "dimension": FEATURE_DIM,
        "truck_sha256": sha256_file(truck),
        "floor_sha256": sha256_file(floor),
        "ceiling_sha256": sha256_file(ceiling),
        "source_sha256": source_sha256,
        "packages": {
            name: get_package_version(name)
            for name in (
                "numpy",
                "opencv-python",
                "torch",
                "torchvision",
                "albumentations",
                "segmentation_models_pytorch",
                "timm",
            )
        },
    }

    signature = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True).encode()
    ).hexdigest()

    return {
        **fingerprint,
        "signature": signature,
        "truck_weights": portable_path(truck),
        "floor_weights": portable_path(floor),
        "ceiling_weights": portable_path(ceiling),
    }


def verify_contract(saved, current):
    if saved.get("signature") != current.get("signature"):
        raise ValueError("Изменились признаки, зависимости или веса сегментации; нужно исходное окружение либо переобучение гибрида")


def validate_features(values, count):
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (count, FEATURE_DIM) or not np.isfinite(values).all():
        raise ValueError(f"Ожидаются конечные признаки [{count}, {FEATURE_DIM}]")
    return values


def fit_normalization(train_features):
    values = validate_features(train_features, len(train_features))
    if len(values) == 0:
        raise ValueError("Пустая обучающая выборка")
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def cached_features(image_dir, rows, contract, cache_dir=None, extractor=None):
    if extractor is None:
        from manual.features import feature
        extractor = feature
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    values = []
    for index, row in enumerate(rows):
        path = Path(image_dir) / f"{row['image_id']}.jpg"
        image_hash = sha256_file(path)
        key = hashlib.sha256((contract["signature"] + image_hash).encode()).hexdigest()
        entry = cache / f"{key}.npz" if cache else None
        if entry is not None and entry.is_file():
            with np.load(entry, allow_pickle=False) as saved:
                value = saved["features"]
        else:
            kwargs = {}
            for name in ("truck_weights", "floor_weights", "ceiling_weights"):
                if name in contract:
                    kwargs[name + "_path"] = resolve_saved_path(contract[name])
            value = extractor(path, **kwargs)
            value = validate_features(np.asarray(value)[None, :], 1)[0]
            if entry is not None:
                with tempfile.NamedTemporaryFile(dir=cache, suffix=".tmp", delete=False) as file:
                    temp_path = Path(file.name)
                    np.savez_compressed(file, features=value)
                temp_path.replace(entry)
        values.append(validate_features(np.asarray(value)[None, :], 1)[0])
        if (index + 1) % 50 == 0 or index + 1 == len(rows):
            print(f"Признаки гибрида: {index + 1}/{len(rows)}", flush=True)
    return validate_features(np.asarray(values).reshape(len(rows), FEATURE_DIM), len(rows))


def _cache_key(contract, image_path):
    image_hash = sha256_file(image_path)
    return hashlib.sha256((contract["signature"] + image_hash).encode()).hexdigest()


def _validate_roi(roi):
    roi = np.asarray(roi)
    if (
        roi.ndim != 3
        or roi.shape[2] != 3
        or roi.shape[0] == 0
        or roi.shape[1] == 0
        or roi.dtype != np.uint8
    ):
        raise ValueError("ROI для CNN должен быть непустым RGB-массивом uint8")
    return roi


def cached_samples(image_dir, rows, contract, cache_dir, extractor=None):
    """Кэширует пару «349 признаков, замаскированный ROI для CNN».

    Возвращает матрицу признаков и пути к кэшированным NPZ в порядке CSV.
    Одно содержимое cache entry создаётся одной сегментацией кузова, поэтому
    CNN и числовая ветка используют ровно одну область изображения.
    """
    if extractor is None:
        from manual.features import extract_hybrid_sample
        extractor = extract_hybrid_sample

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    values = []
    entries = []
    for index, row in enumerate(rows):
        image_path = Path(image_dir) / f"{row['image_id']}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"Нет изображения: {image_path}")
        entry = cache / f"{_cache_key(contract, image_path)}.npz"
        if entry.is_file():
            with np.load(entry, allow_pickle=False) as saved:
                value = saved["features"]
                roi = saved["roi"]
        else:
            kwargs = {
                name + "_path": resolve_saved_path(contract[name])
                for name in ("truck_weights", "floor_weights", "ceiling_weights")
                if name in contract
            }
            value, roi = extractor(image_path, **kwargs)
            value = validate_features(np.asarray(value)[None, :], 1)[0]
            roi = _validate_roi(roi)
            with tempfile.NamedTemporaryFile(dir=cache, suffix=".tmp", delete=False) as file:
                temporary_path = Path(file.name)
                np.savez_compressed(file, features=value, roi=roi)
            temporary_path.replace(entry)
        values.append(validate_features(np.asarray(value)[None, :], 1)[0])
        _validate_roi(roi)
        entries.append(entry)
        if (index + 1) % 50 == 0 or index + 1 == len(rows):
            print(f"ROI и признаки гибрида: {index + 1}/{len(rows)}", flush=True)
    return (
        validate_features(np.asarray(values).reshape(len(rows), FEATURE_DIM), len(rows)),
        entries,
    )
