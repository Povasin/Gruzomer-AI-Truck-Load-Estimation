from __future__ import annotations

import csv
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
MASK_DIR = PROJECT_DIR / "debug_output" / "floor_masks"
SOURCE_TRAIN_SPLIT = PROJECT_DIR / "DataSet" / "train" / "train_split.csv"
SOURCE_VALID_SPLIT = PROJECT_DIR / "DataSet" / "train" / "validation_split.csv"
TRAIN_IDS_FILE = PROJECT_DIR / "DataSet" / "train" / "train_ids_grouped.txt"
VALID_IDS_FILE = PROJECT_DIR / "DataSet" / "train" / "valid_ids_grouped.txt"


def read_grouped_ids(path: str | Path) -> list[tuple[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"image_id", "group_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain image_id and group_id")
        rows = [(row["image_id"].strip(), row["group_id"].strip()) for row in reader]
    if any(not image_id or not group_id for image_id, group_id in rows):
        raise ValueError(f"{path} contains empty image_id or group_id")
    ids = [image_id for image_id, _ in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate image_id values")
    return rows


def build_grouped_floor_split(
    mask_dir: str | Path,
    train_split_file: str | Path,
    valid_split_file: str | Path,
) -> tuple[list[str], list[str]]:
    mask_ids = {path.stem for path in Path(mask_dir).glob("*.png")}
    if not mask_ids:
        raise RuntimeError(f"No PNG masks found in {mask_dir}")
    train_rows = read_grouped_ids(train_split_file)
    valid_rows = read_grouped_ids(valid_split_file)
    train_groups = {group_id for _, group_id in train_rows}
    valid_groups = {group_id for _, group_id in valid_rows}
    group_overlap = train_groups & valid_groups
    if group_overlap:
        raise RuntimeError(
            f"Train and validation overlap by group_id: {len(group_overlap)} groups"
        )
    train_source_ids = {image_id for image_id, _ in train_rows}
    valid_source_ids = {image_id for image_id, _ in valid_rows}
    id_overlap = train_source_ids & valid_source_ids
    if id_overlap:
        raise RuntimeError(
            f"Train and validation overlap by image_id: {len(id_overlap)} IDs"
        )
    unassigned = mask_ids - train_source_ids - valid_source_ids
    if unassigned:
        examples = ", ".join(sorted(unassigned)[:5])
        raise RuntimeError(
            f"{len(unassigned)} masks are absent from grouped splits: {examples}"
        )
    train_ids = sorted(mask_ids & train_source_ids)
    valid_ids = sorted(mask_ids & valid_source_ids)
    if not train_ids or not valid_ids:
        raise RuntimeError("Grouped floor split produced an empty train or validation set")
    return train_ids, valid_ids


def save_ids(path: str | Path, image_ids: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for image_id in image_ids:
            file.write(f"{image_id}\n")


def main() -> None:
    train_ids, valid_ids = build_grouped_floor_split(
        mask_dir=MASK_DIR,
        train_split_file=SOURCE_TRAIN_SPLIT,
        valid_split_file=SOURCE_VALID_SPLIT,
    )
    save_ids(TRAIN_IDS_FILE, train_ids)
    save_ids(VALID_IDS_FILE, valid_ids)
    print("GROUPED FLOOR SPLIT SAVED")
    print(f"Train: {len(train_ids)}")
    print(f"Validation: {len(valid_ids)}")
    print("Group overlap: 0")
    print(f"Train IDs: {TRAIN_IDS_FILE}")
    print(f"Validation IDs: {VALID_IDS_FILE}")


if __name__ == "__main__":
    main()
