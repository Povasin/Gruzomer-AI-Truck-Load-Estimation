"""Image-regression baseline with a FIXED train/validation split.

Python 3.10+, NumPy and Pillow only.

This version does NOT generate train/validation splits.
It uses two already prepared CSV files:
- train_split.csv
- validation_split.csv

Metadata is used only to validate the split and to produce diagnostics.
The Ridge model itself still uses only image features, exactly like the original baseline.
"""

from pathlib import Path
import argparse
import csv
import json
from collections import Counter

import numpy as np
from PIL import Image, ImageOps


# -----------------------------------------------------------------------------
# CSV reading and validation
# -----------------------------------------------------------------------------

# Эти поля должны присутствовать в train_split.csv и validation_split.csv.
# Дополнительные поля, например index или split, разрешены и просто игнорируются.
METADATA_FIELDS = [
    "image_id",
    "load_pct",
    "group_id",
    "load_bin",
    "vehicle_type",
    "viewpoint",
    "lighting",
    "cargo_type",
]

# Эти поля нужны только для отчёта о том, насколько похожи train и validation.
DIAGNOSTIC_COLUMNS = [
    "load_bin",
    "vehicle_type",
    "cargo_type",
    "viewpoint",
    "lighting",
]


def _validate_image_ids(rows, path):
    """Проверяем, что image_id заполнены, уникальны и безопасны как имена файлов."""
    if not rows:
        raise ValueError(f"{path}: table is empty")

    image_ids = [row["image_id"] for row in rows]

    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"{path}: duplicate image_id values")

    for image_id in image_ids:
        if (
            not image_id
            or Path(image_id).name != image_id
            or "/" in image_id
            or "\\" in image_id
        ):
            raise ValueError(f"{path}: invalid image_id: {image_id!r}")


def read_split_table(path):
    """Читаем уже готовый train_split.csv или validation_split.csv.

    Файл открывается только на чтение и НИКОГДА не перезаписывается.
    Дополнительные колонки разрешены.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        missing = [field for field in METADATA_FIELDS if field not in fieldnames]
        if missing:
            raise ValueError(
                f"{path}: missing columns {missing}. Found columns: {fieldnames}"
            )

        rows = list(reader)

    _validate_image_ids(rows, path)

    # Проверяем пустые значения и убираем случайные пробелы только в памяти.
    for row_number, row in enumerate(rows, start=2):
        for field in METADATA_FIELDS:
            value = row.get(field)
            if value is None or not str(value).strip():
                raise ValueError(
                    f"{path}: empty value in column {field!r}, CSV row {row_number}"
                )
            row[field] = str(value).strip()

    # Проверяем target.
    y = np.asarray([float(row["load_pct"]) for row in rows], dtype=np.float64)

    if not np.isfinite(y).all() or ((y < 0) | (y > 100)).any():
        raise ValueError(f"{path}: load_pct must contain finite values in [0, 100]")

    return rows


def read_test_table(path):
    """Читаем test.csv. В нём ожидается только image_id."""
    path = Path(path)

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames != ["image_id"]:
            raise ValueError(
                f"{path}: expected columns ['image_id'], got {reader.fieldnames}"
            )

        rows = list(reader)

    _validate_image_ids(rows, path)
    return rows


def validate_fixed_split(train_rows, val_rows):
    """Главные проверки готового train/validation split."""

    train_ids = {row["image_id"] for row in train_rows}
    val_ids = {row["image_id"] for row in val_rows}

    # Один и тот же image_id не должен быть одновременно в train и validation.
    image_overlap = train_ids & val_ids
    if image_overlap:
        raise ValueError(
            "image_id leakage between train and validation: "
            f"{sorted(image_overlap)[:10]}"
        )

    train_groups = {row["group_id"] for row in train_rows}
    val_groups = {row["group_id"] for row in val_rows}

    # Самая важная проверка: одна group_id должна целиком находиться
    # либо в train, либо в validation.
    group_overlap = train_groups & val_groups
    if group_overlap:
        raise ValueError(
            "group_id leakage between train and validation: "
            f"{sorted(group_overlap)[:10]}"
        )

    return {
        "train_count": len(train_rows),
        "validation_count": len(val_rows),
        "total_count": len(train_rows) + len(val_rows),
        "train_group_count": len(train_groups),
        "validation_group_count": len(val_groups),
        "image_overlap_count": 0,
        "group_overlap_count": 0,
    }


# -----------------------------------------------------------------------------
# Image features: математически те же признаки, что и в исходном baseline
# -----------------------------------------------------------------------------

def feature(path):
    """Преобразуем одно изображение в 233 признака."""
    with Image.open(path) as source:
        im = ImageOps.exif_transpose(source).convert("RGB")

        # 1 признак: отношение ширины к высоте.
        aspect = im.width / im.height

        # Уменьшаем изображение до 32x32 и нормализуем RGB в [0, 1].
        small = np.asarray(
            im.resize((32, 32), Image.Resampling.BILINEAR),
            dtype=np.float64,
        ) / 255.0

        # 8x8 spatial cells x RGB = 192 признака.
        spatial = small.reshape(8, 4, 8, 4, 3).mean(axis=(1, 3)).ravel()

        # 4x4 локальных texture-признака = 16 признаков.
        gray = small.mean(axis=2)
        texture = gray.reshape(4, 8, 4, 8).std(axis=(1, 3)).ravel()

        # 8 bins x 3 RGB-канала = 24 признака.
        hist = np.concatenate(
            [
                np.histogram(small[:, :, channel], bins=8, range=(0, 1))[0] / 1024
                for channel in range(3)
            ]
        )

    # 192 + 16 + 24 + 1 = 233.
    return np.concatenate((spatial, texture, hist, [aspect]))


def features(image_dir, rows):
    """Извлекаем признаки для изображений в порядке строк CSV."""
    image_dir = Path(image_dir)
    result = []

    for index, row in enumerate(rows):
        image_path = image_dir / f"{row['image_id']}.jpg"

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        result.append(feature(image_path))

        if (index + 1) % 200 == 0:
            print(f"Features: {index + 1}/{len(rows)}", flush=True)

    return np.asarray(result, dtype=np.float64)


# -----------------------------------------------------------------------------
# Ridge baseline
# -----------------------------------------------------------------------------

def fit_ridge(x, y, alpha):
    """Обучаем Ridge-регрессию."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)

    # Защита от деления на ноль для константных признаков.
    scale[scale < 1e-8] = 1.0

    # Стандартизация признаков.
    z = (x - mean) / scale

    intercept = float(y.mean())

    weights = np.linalg.solve(
        z.T @ z + alpha * np.eye(z.shape[1]),
        z.T @ (y - intercept),
    )

    return {
        "mean": mean,
        "scale": scale,
        "weights": weights,
        "intercept": np.asarray(intercept),
    }


def predict_ridge(model, x):
    prediction = (
        ((x - model["mean"]) / model["scale"]) @ model["weights"]
        + float(model["intercept"])
    )

    # По условию прогноз должен быть в диапазоне [0, 100].
    return np.clip(prediction, 0, 100)


def metrics(y, prediction):
    errors = np.abs(y - prediction)

    return {
        "mae": float(errors.mean()),
        "within_10_pct": float(100 * (errors <= 10).mean()),
        "count": int(len(y)),
    }


# -----------------------------------------------------------------------------
# Диагностика уже готового split. Ничего здесь не меняется и не перебирается.
# -----------------------------------------------------------------------------

def _distribution(rows, column):
    counts = Counter(row[column] for row in rows)
    total = len(rows)

    return {
        key: counts[key] / total
        for key in sorted(counts)
    }


def _tv_distance(rows_a, rows_b, column):
    """Насколько распределения категории отличаются: 0 = одинаковые, 1 = совсем разные."""
    dist_a = _distribution(rows_a, column)
    dist_b = _distribution(rows_b, column)

    categories = set(dist_a) | set(dist_b)

    return 0.5 * sum(
        abs(dist_a.get(category, 0.0) - dist_b.get(category, 0.0))
        for category in categories
    )


def make_split_diagnostics(train_rows, val_rows):
    diagnostics = {}

    for column in DIAGNOSTIC_COLUMNS:
        diagnostics[column] = {
            "train": _distribution(train_rows, column),
            "validation": _distribution(val_rows, column),
            "tv_distance": float(_tv_distance(train_rows, val_rows, column)),
        }

    return diagnostics


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # 1. Читаем УЖЕ ГОТОВЫЕ split-файлы.
    # Никаких seed, shuffle и перебора 5000 вариантов здесь больше нет.
    train_rows = read_split_table(args.train_split)
    val_rows = read_split_table(args.validation_split)

    # 2. Проверяем, что между train и validation нет утечки
    # ни по image_id, ни по group_id.
    split_info = validate_fixed_split(train_rows, val_rows)

    print(
        "Fixed split loaded: "
        f"train={split_info['train_count']}, "
        f"validation={split_info['validation_count']}, "
        f"train_groups={split_info['train_group_count']}, "
        f"validation_groups={split_info['validation_group_count']}",
        flush=True,
    )

    # 3. Targets для train и validation.
    y_train = np.asarray(
        [float(row["load_pct"]) for row in train_rows],
        dtype=np.float64,
    )

    y_val = np.asarray(
        [float(row["load_pct"]) for row in val_rows],
        dtype=np.float64,
    )

    # 4. Извлекаем признаки ОТДЕЛЬНО для train и validation.
    x_train = features(args.images, train_rows)
    x_val = features(args.images, val_rows)

    # 5. Обучаем Ridge ТОЛЬКО на train_split.csv.
    ridge = fit_ridge(
        x_train,
        y_train,
        args.alpha,
    )

    # 6. Считаем честную метрику ТОЛЬКО на validation_split.csv.
    ridge_prediction = predict_ridge(ridge, x_val)
    ridge_metrics = metrics(y_val, ridge_prediction)

    # 7. Median baseline тоже вычисляется только по train.
    train_median = float(np.median(y_train))
    median_prediction = np.full(len(y_val), train_median)
    median_metrics = metrics(y_val, median_prediction)

    # 8. Если method=auto, выбираем Ridge или median по фиксированной validation.
    selected = args.method

    if selected == "auto":
        selected = (
            "ridge"
            if ridge_metrics["mae"] < median_metrics["mae"]
            else "median"
        )

    # 9. Validation закончена.
    # Теперь объединяем train + validation и обучаем финальную модель
    # на ВСЕХ доступных размеченных изображениях.
    all_rows = train_rows + val_rows
    y_all = np.concatenate([y_train, y_val])
    x_all = np.concatenate([x_train, x_val], axis=0)

    final = fit_ridge(
        x_all,
        y_all,
        args.alpha,
    )

    # Если auto выбрал median, predict() потом будет использовать медиану,
    # а сохранённые Ridge-веса просто останутся в файле и не будут использоваться.
    final.update(
        method=np.asarray(selected),
        median=np.asarray(float(np.median(y_all))),
    )

    # 10. Сохраняем финальную модель.
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(model_path, **final)

    # 11. Диагностика фиксированного split.
    # Это только отчёт, split здесь НЕ меняется.
    diagnostics = make_split_diagnostics(train_rows, val_rows)

    report = {
        "split_type": "fixed_precomputed",
        "train_split": str(args.train_split),
        "validation_split": str(args.validation_split),
        "alpha": float(args.alpha),
        "features": int(x_train.shape[1]),
        "train_count": int(len(train_rows)),
        "validation_count": int(len(val_rows)),
        "total_count": int(len(all_rows)),
        "train_group_count": int(split_info["train_group_count"]),
        "validation_group_count": int(split_info["validation_group_count"]),
        "image_overlap_count": 0,
        "group_overlap_count": 0,
        "median": median_metrics,
        "ridge": ridge_metrics,
        "selected_method": selected,
        "split_diagnostics": diagnostics,
        "note": (
            "The provided train_split.csv and validation_split.csv were used exactly as-is. "
            "No random split generation was performed. Model selection used only the fixed "
            "validation split; the final model was then refitted on train + validation."
        ),
    }

    report_path = model_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


# -----------------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------------

def predict(args):
    rows = read_test_table(args.test_csv)

    with np.load(args.model, allow_pickle=False) as saved:
        model = {key: saved[key] for key in saved.files}

    if str(model["method"]) == "median":
        values = np.full(len(rows), float(model["median"]))
    else:
        x_test = features(args.images, rows)
        values = predict_ridge(model, x_test)

    if not np.isfinite(values).all():
        raise ValueError("Nonfinite predictions")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "load_pct"])
        writer.writerows(
            (row["image_id"], f"{float(value):.6f}")
            for row, value in zip(rows, values)
        )

    print(f"Saved {len(rows)} predictions to {output_path}.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    # TRAIN
    train_parser = sub.add_parser("train")

    train_parser.add_argument(
        "--train-split",
        required=True,
        help="Path to the already prepared train_split.csv",
    )

    train_parser.add_argument(
        "--validation-split",
        required=True,
        help="Path to the already prepared validation_split.csv",
    )

    train_parser.add_argument(
        "--images",
        required=True,
        help="Directory containing all TRAIN jpg images",
    )

    train_parser.add_argument(
        "--model",
        default="../models/model.npz",
    )

    train_parser.add_argument(
        "--alpha",
        type=float,
        default=100.0,
    )

    train_parser.add_argument(
        "--method",
        choices=["auto", "ridge", "median"],
        default="auto",
    )

    train_parser.set_defaults(run=train)

    # PREDICT
    predict_parser = sub.add_parser("predict")

    predict_parser.add_argument(
        "--model",
        default="../models/model.npz",
    )

    predict_parser.add_argument(
        "--test-csv",
        required=True,
    )

    predict_parser.add_argument(
        "--images",
        required=True,
    )

    predict_parser.add_argument(
        "--output",
        default="../output_csv/submission_check.csv",
    )

    predict_parser.set_defaults(run=predict)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
