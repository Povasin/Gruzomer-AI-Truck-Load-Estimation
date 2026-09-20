import pandas as pd
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "DataSet" / "train"

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# 1. Загружаем размеченный датасет
# ============================================================

df = pd.read_csv(TRAIN_DIR /"train_metadata.csv")


# ============================================================
# 2. Проверяем таблицу
# ============================================================

required_columns = [
    "image_id",
    "load_pct",
    "group_id",
    "load_bin",
    "vehicle_type",
    "viewpoint",
    "lighting",
    "cargo_type",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Нет колонок: {missing_columns}"
    )


# Проверяем, что нигде нет пустых значений
if df[required_columns].isna().any().any():
    raise ValueError(
        "В metadata есть пустые значения"
    )


# ============================================================
# 3. Какие параметры хотим балансировать
# ============================================================

# Чем больше вес, тем важнее распределение этой колонки.
#
# load_bin и vehicle_type для нас самые важные.
weights = {
    "load_bin": 3.0,
    "vehicle_type": 3.0,
    "cargo_type": 1.5,
    "viewpoint": 1.0,
    "lighting": 1.0,
}


# ============================================================
# 4. Сохраняем распределение ВСЕГО датасета
# ============================================================

full_distributions = {}

for column in weights:
    full_distributions[column] = (
        df[column]
        .value_counts(normalize=True)
        .sort_index()
    )


# ============================================================
# 5. Функция оценки качества split
# ============================================================

def calculate_split_score(train_df, val_df):

    score = 0.0

    # --------------------------------------------------------
    # Проверяем размер validation
    # --------------------------------------------------------

    validation_ratio = len(val_df) / len(df)

    # Хотим около 20%.
    # Если validation сильно отличается от 20%,
    # добавляем штраф.
    score += abs(validation_ratio - 0.20) * 10


    # --------------------------------------------------------
    # Проверяем распределения категорий
    # --------------------------------------------------------

    for column, weight in weights.items():

        # Распределение validation
        val_distribution = (
            val_df[column]
            .value_counts(normalize=True)
            .reindex(
                full_distributions[column].index,
                fill_value=0
            )
        )

        # Распределение всего датасета
        full_distribution = full_distributions[column]

        # Насколько validation отличается
        # от полного датасета
        difference = np.abs(
            val_distribution.values
            -
            full_distribution.values
        ).mean()

        # Более важные признаки имеют больший вес
        score += difference * weight


        # Если какая-то категория вообще исчезла
        # из validation, добавляем дополнительный штраф
        missing_categories = (
            set(df[column].unique())
            -
            set(val_df[column].unique())
        )

        score += len(missing_categories) * 0.5


    return score


# ============================================================
# 6. Генерируем много вариантов split
# ============================================================

best_score = float("inf")
best_train_indices = None
best_val_indices = None
best_seed = None


# Проверяем 5000 различных вариантов
for seed in range(5000):

    splitter = GroupShuffleSplit(
        n_splits=1,

        # Около 20% групп в validation
        test_size=0.20,

        random_state=seed
    )


    train_indices, val_indices = next(
        splitter.split(
            X=df,

            # ГЛАВНОЕ:
            # одна group_id никогда не делится
            groups=df["group_id"]
        )
    )


    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]


    # Оцениваем качество этого split
    score = calculate_split_score(
        train_df,
        val_df
    )


    # Если этот вариант лучше предыдущих,
    # запоминаем его
    if score < best_score:

        best_score = score

        best_train_indices = train_indices
        best_val_indices = val_indices

        best_seed = seed


# ============================================================
# 7. Получаем лучший split
# ============================================================

train_df = (
    df.iloc[best_train_indices]
    .copy()
    .reset_index(drop=True)
)

val_df = (
    df.iloc[best_val_indices]
    .copy()
    .reset_index(drop=True)
)


# ============================================================
# 8. Проверяем отсутствие leakage по group_id
# ============================================================

train_groups = set(
    train_df["group_id"]
)

val_groups = set(
    val_df["group_id"]
)

intersection = train_groups & val_groups


if intersection:
    raise ValueError(
        "ОШИБКА: group_id попала одновременно "
        "в train и validation"
    )


# ============================================================
# 9. Выводим результат
# ============================================================

print("Best seed:", best_seed)

print(
    "Train:",
    len(train_df)
)

print(
    "Validation:",
    len(val_df)
)

print(
    "Validation percentage:",
    len(val_df) / len(df) * 100
)

print(
    "Group leakage:",
    len(intersection)
)


# ============================================================
# 10. Смотрим распределения
# ============================================================

for column in weights:

    print("\n==============================")
    print(column)
    print("==============================")

    comparison = pd.DataFrame({

        "ALL": (
            df[column]
            .value_counts(normalize=True)
        ),

        "TRAIN": (
            train_df[column]
            .value_counts(normalize=True)
        ),

        "VAL": (
            val_df[column]
            .value_counts(normalize=True)
        ),
    }).fillna(0)

    print(
        (comparison * 100).round(2)
    )


# ============================================================
# 11. Сохраняем split
# ============================================================

train_df.to_csv(
    "train_split.csv",
    index=False
)

val_df.to_csv(
    "validation_split.csv",
    index=False
)

print("\nSplit сохранён.")