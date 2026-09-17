from pathlib import Path
import csv
from collections import Counter

import numpy as np


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
    """Читаем уже готовый train_split.csv или validation_split.csv."""
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

    for row_number, row in enumerate(rows, start=2):
        for field in METADATA_FIELDS:
            value = row.get(field)
            if value is None or not str(value).strip():
                raise ValueError(
                    f"{path}: empty value in column {field!r}, CSV row {row_number}"
                )
            row[field] = str(value).strip()

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

    image_overlap = train_ids & val_ids
    if image_overlap:
        raise ValueError(
            "image_id leakage between train and validation: "
            f"{sorted(image_overlap)[:10]}"
        )

    train_groups = {row["group_id"] for row in train_rows}
    val_groups = {row["group_id"] for row in val_rows}

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


def _distribution(rows, column):
    counts = Counter(row[column] for row in rows)
    total = len(rows)

    return {key: counts[key] / total for key in sorted(counts)}


def _tv_distance(rows_a, rows_b, column):
    """Насколько распределения категорий отличаются: 0 = одинаковые, 1 = совсем разные."""
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
