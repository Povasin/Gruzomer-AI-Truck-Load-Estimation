"""Валидация и формирование прогноза методом 20 срезов (Вариант А).

Команды:
  --mode val:   расчет MAE и доли попаданий в пределах 10 п.п. на validation_split.csv
  --mode test:  генерация файла submission для 307 тестовых изображений
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from manual.depth_slicer import analyze_image


def evaluate_validation(
    val_csv: Path,
    images_dir: Path,
    output_errors_csv: Path | None = None,
) -> dict:
    """Оценивает MAE и метрику within_10_pct на валидационной выборке."""
    df_val = pd.read_csv(val_csv)
    print(f"[VAL] Загружено {len(df_val)} строк из {val_csv}")

    preds = []
    targets = []
    errors = []
    stop_indices = []

    start_time = time.time()
    for idx, row in tqdm(df_val.iterrows(), total=len(df_val), desc="Оценка валидации"):
        image_id = str(row["image_id"]).strip()
        img_path = images_dir / f"{image_id}.jpg"
        target_pct = float(row["load_pct"])

        if not img_path.exists():
            print(f"Внимание: файл {img_path} не найден!")
            preds.append(50.0)
            targets.append(target_pct)
            errors.append(abs(50.0 - target_pct))
            stop_indices.append(-1)
            continue

        result, _, _, _, _ = analyze_image(img_path)
        pred_pct = result.load_pct

        preds.append(pred_pct)
        targets.append(target_pct)
        err = abs(pred_pct - target_pct)
        errors.append(err)
        stop_indices.append(result.stop_index)

    elapsed = time.time() - start_time
    mae = float(np.mean(errors))
    within_10 = float(np.mean(np.array(errors) <= 10.0) * 100.0)

    print("\n" + "=" * 50)
    print(f"РЕЗУЛЬТАТЫ ВАЛИДАЦИИ (20 срезов):")
    print(f"  Количество фото: {len(df_val)}")
    print(f"  MAE:             {mae:.2f} п.п.")
    print(f"  Within 10%:      {within_10:.1f}%")
    print(f"  Время расчёта:   {elapsed:.1f} с ({elapsed / len(df_val):.2f} с/фото)")
    print("=" * 50)

    if output_errors_csv is not None:
        df_out = df_val.copy()
        df_out["pred_load_pct"] = preds
        df_out["abs_error"] = errors
        df_out["stop_index"] = stop_indices
        output_errors_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(output_errors_csv, index=False)
        print(f"Ошибки сохранены в: {output_errors_csv}")

    return {
        "mae": mae,
        "within_10": within_10,
        "num_samples": len(df_val),
    }


def predict_test(
    test_csv: Path,
    images_dir: Path,
    output_submission_csv: Path,
) -> None:
    """Генерирует submission.csv на тестовой выборке (307 фото)."""
    df_test = pd.read_csv(test_csv)
    print(f"[TEST] Загружено {len(df_test)} идентификаторов из {test_csv}")

    preds = []
    start_time = time.time()
    for idx, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Инференс test"):
        image_id = str(row["image_id"]).strip()
        img_path = images_dir / f"{image_id}.jpg"

        if not img_path.exists():
            print(f"Внимание: файл {img_path} не найден! Ставим 50.0")
            preds.append(50.0)
            continue

        result, _, _, _, _ = analyze_image(img_path)
        # Ограничиваем диапазон строго [0, 100]
        pred_pct = float(np.clip(result.load_pct, 0.0, 100.0))
        preds.append(pred_pct)

    elapsed = time.time() - start_time
    df_sub = pd.DataFrame({
        "image_id": df_test["image_id"],
        "load_pct": preds,
    })

    output_submission_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sub.to_csv(output_submission_csv, index=False)

    print("\n" + "=" * 50)
    print(f"САБМИТ СФОРМИРОВАН:")
    print(f"  Файл:            {output_submission_csv}")
    print(f"  Строк:           {len(df_sub)}")
    print(f"  Диапазон:        [{df_sub['load_pct'].min():.1f}%, {df_sub['load_pct'].max():.1f}%]")
    print(f"  Средний прогноз: {df_sub['load_pct'].mean():.1f}%")
    print(f"  Время расчёта:   {elapsed:.1f} с")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Валидация и инференс 20 срезов")
    parser.add_argument("--mode", choices=["val", "test", "all"], default="val")
    parser.add_argument("--val-split", type=Path, default=Path("DataSet/train/validation_split.csv"))
    parser.add_argument("--train-images", type=Path, default=Path("DataSet/train/images"))
    parser.add_argument("--test-csv", type=Path, default=Path("DataSet/test/test.csv"))
    parser.add_argument("--test-images", type=Path, default=Path("DataSet/test/images"))
    parser.add_argument("--output-errors", type=Path, default=Path("models/slices_val_errors.csv"))
    parser.add_argument("--output-submission", type=Path, default=Path("output_csv/submission_slices_v1.csv"))
    args = parser.parse_args()

    if args.mode in ("val", "all"):
        evaluate_validation(
            val_csv=args.val_split,
            images_dir=args.train_images,
            output_errors_csv=args.output_errors,
        )

    if args.mode in ("test", "all"):
        predict_test(
            test_csv=args.test_csv,
            images_dir=args.test_images,
            output_submission_csv=args.output_submission,
        )


if __name__ == "__main__":
    main()
