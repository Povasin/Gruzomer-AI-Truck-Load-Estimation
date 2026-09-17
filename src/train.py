from pathlib import Path
import csv
import json

import numpy as np

from features import features
from report import make_split_diagnostics, read_split_table, validate_fixed_split


def fit_ridge(x, y, alpha):
    """Обучаем Ridge-регрессию."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
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
    return np.clip(prediction, 0, 100)


def metrics(y, prediction):
    errors = np.abs(y - prediction)
    return {
        "mae": float(errors.mean()),
        "within_10_pct": float(100 * (errors <= 10).mean()),
        "count": int(len(y)),
    }


def write_validation_errors(path, rows, actual, prediction):
    """Сохраняем ошибки на validation, начиная с наибольшей абсолютной ошибки."""
    if len(rows) != len(actual) or len(rows) != len(prediction):
        raise ValueError("Rows, actual values and predictions must have the same length")

    report_rows = []
    for row, actual_value, prediction_value in zip(rows, actual, prediction):
        actual_value = float(actual_value)
        prediction_value = float(prediction_value)
        report_rows.append(
            {
                "image_id": row["image_id"],
                "actual_load_pct": actual_value,
                "predicted_load_pct": prediction_value,
                "absolute_error_pct": abs(prediction_value - actual_value),
            }
        )

    report_rows.sort(key=lambda row: row["absolute_error_pct"], reverse=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)


def train(args):
    train_rows = read_split_table(args.train_split)
    val_rows = read_split_table(args.validation_split)
    split_info = validate_fixed_split(train_rows, val_rows)

    print(
        "Fixed split loaded: "
        f"train={split_info['train_count']}, "
        f"validation={split_info['validation_count']}, "
        f"train_groups={split_info['train_group_count']}, "
        f"validation_groups={split_info['validation_group_count']}",
        flush=True,
    )

    y_train = np.asarray([float(row["load_pct"]) for row in train_rows], dtype=np.float64)
    y_val = np.asarray([float(row["load_pct"]) for row in val_rows], dtype=np.float64)
    x_train = features(args.images, train_rows)
    x_val = features(args.images, val_rows)
    ridge = fit_ridge(x_train, y_train, args.alpha)
    ridge_prediction = predict_ridge(ridge, x_val)
    ridge_metrics = metrics(y_val, ridge_prediction)
    train_median = float(np.median(y_train))
    median_prediction = np.full(len(y_val), train_median)
    median_metrics = metrics(y_val, median_prediction)

    selected = args.method
    if selected == "auto":
        selected = "ridge" if ridge_metrics["mae"] < median_metrics["mae"] else "median"

    selected_prediction = (
        ridge_prediction if selected == "ridge" else median_prediction
    )
    write_validation_errors(
        args.errors_output,
        val_rows,
        y_val,
        selected_prediction,
    )

    all_rows = train_rows + val_rows
    y_all = np.concatenate([y_train, y_val])
    x_all = np.concatenate([x_train, x_val], axis=0)
    final = fit_ridge(x_all, y_all, args.alpha)
    final.update(
        method=np.asarray(selected),
        median=np.asarray(float(np.median(y_all))),
    )

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_path, **final)

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
        "split_diagnostics": diagnostics
    }

    report_path = model_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Validation error report saved to: {args.errors_output}")
