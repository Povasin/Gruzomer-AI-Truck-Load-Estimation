from pathlib import Path
import csv


def write_submission(rows, values, output):
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
