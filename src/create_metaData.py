import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "DataSet" / "train"

# Читаем исходные файлы
train = pd.read_csv(TRAIN_DIR / "train.csv")
groups = pd.read_csv(TRAIN_DIR / "train_groups.csv")

# Объединяем их по image_id
df = train.merge(
    groups,
    on="image_id",
    how="left"
)

# Создаём интервалы загрузки
bins = [-0.1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

labels = [
    "0-10",
    "10-20",
    "20-30",
    "30-40",
    "40-50",
    "50-60",
    "60-70",
    "70-80",
    "80-90",
    "90-100"
]

df["load_bin"] = pd.cut(
    df["load_pct"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Добавляем пустые колонки,
# которые потом размечаешь вручную
df["vehicle_type"] = ""
df["viewpoint"] = ""
df["lighting"] = ""
df["cargo_type"] = ""

# Сохраняем отдельный metadata-файл
df.to_csv(
    TRAIN_DIR / "train_metadata.csv",
    index=False
)