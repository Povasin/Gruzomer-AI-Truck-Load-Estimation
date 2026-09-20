from pathlib import Path
import csv
import math


def write_submission(rows, values, output):
    values = [float(value) for value in values]
    if len(rows) != len(values):
        raise ValueError("Количество прогнозов не совпадает с количеством image_id")
    if any(not math.isfinite(value) or not 0 <= value <= 100 for value in values):
        raise ValueError("Прогнозы должны быть конечными числами в [0, 100]")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "load_pct"])
        writer.writerows(
            (row["image_id"], f"{float(value):.6f}")
            for row, value in zip(rows, values)
        )

    print(f"Saved {len(rows)} predictions to {output_path}.")
