"""Строит group-aware k-fold разбиение train_metadata.csv для обучения CNN.

В отличие от work_data.py (один фиксированный train/validation split), этот
скрипт делит весь train на K сбалансированных фолдов, где фотографии одной
group_id никогда не попадают в разные фолды. Результат сохраняется в
folds.csv рядом с train_metadata.csv: те же колонки + колонка "fold"
(0..K-1), которую использует train_cnn.py.

Запуск:
    python cv_split.py --metadata ./DataSet/train/train_metadata.csv \
        --output ./DataSet/train/folds.csv --n-splits 4
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    HAS_STRATIFIED_GROUP_KFOLD = False


REQUIRED_COLUMNS = [
    "image_id",
    "load_pct",
    "group_id",
    "load_bin",
    "vehicle_type",
    "viewpoint",
    "lighting",
    "cargo_type",
]


def build_folds(df, n_splits, seed):
    """Возвращает массив fold-индексов (0..n_splits-1), выровненный с df."""
    groups = df["group_id"].to_numpy()

    if HAS_STRATIFIED_GROUP_KFOLD:
        # Стратифицируем по load_bin: внутри каждого фолда стараемся
        # сохранить распределение загрузки близким к общему, при этом
        # ни одна group_id не делится между фолдами.
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        fold = np.full(len(df), -1, dtype=int)
        for fold_index, (_, val_index) in enumerate(
            splitter.split(df, df["load_bin"], groups)
        ):
            fold[val_index] = fold_index
        return fold

    # Фолбэк для старых версий sklearn: обычный GroupKFold без
    # стратификации по load_bin (leakage по группам всё равно исключён).
    splitter = GroupKFold(n_splits=n_splits)
    fold = np.full(len(df), -1, dtype=int)
    for fold_index, (_, val_index) in enumerate(splitter.split(df, groups=groups)):
        fold[val_index] = fold_index
    return fold


def validate_folds(df):
    """Проверяем отсутствие leakage по group_id между фолдами."""
    grouped = df.groupby("group_id")["fold"].nunique()
    leaking_groups = grouped[grouped > 1]
    if len(leaking_groups) > 0:
        raise ValueError(
            "group_id leakage between folds: "
            f"{list(leaking_groups.index[:10])}"
        )


def print_fold_report(df, n_splits):
    print("\nРазмер фолдов:\n")
    for fold_index in range(n_splits):
        fold_df = df[df["fold"] == fold_index]
        print(
            f"Fold {fold_index}: {len(fold_df)} фото, "
            f"{fold_df['group_id'].nunique()} групп"
        )

    print("\nload_bin по фолдам (%):\n")
    table = (
        pd.crosstab(df["load_bin"], df["fold"], normalize="columns") * 100
    ).round(1)
    print(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        default="./DataSet/train/train_metadata.csv",
        help="Путь к train_metadata.csv (image_id,load_pct,group_id,...)",
    )
    parser.add_argument(
        "--output",
        default="./DataSet/train/folds.csv",
        help="Куда сохранить таблицу с колонкой fold",
    )
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"В {args.metadata} нет колонок: {missing}")

    if df[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError(f"В {args.metadata} есть пустые значения в обязательных колонках")

    df["fold"] = build_folds(df, args.n_splits, args.seed)
    validate_folds(df)
    print_fold_report(df, args.n_splits)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nСохранено: {output_path} ({len(df)} строк, {args.n_splits} фолдов)")


if __name__ == "__main__":
    main()
