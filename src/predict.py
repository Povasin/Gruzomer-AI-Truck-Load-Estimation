from pathlib import Path
import pickle
import numpy as np

from report import read_test_table
from submission import write_submission


def predict(args):
    model_path = Path(args.model)
    # Загружаем ровно указанный файл: соседний .pkl не подменяет .npz.
    if model_path.suffix.lower() == ".pkl":
        with open(model_path, "rb") as f:
            data = pickle.load(f)
        method = data["method"]
        model = data["model"]
    else:
        with np.load(model_path, allow_pickle=False) as saved:
            model = {key: saved[key] for key in saved.files}
        method = str(model["method"])

    if method == "hybrid":
        from hybrid import predict as predict_hybrid
        return predict_hybrid(args)

    rows = read_test_table(args.test_csv)
    model_path = Path(args.model)

    if method == "median":
        values = np.full(len(rows), float(model["median"]))
    elif method == "boosting":
        from features import features
        x_test = features(args.images, rows)
        values = np.clip(model.predict(x_test), 0, 100)
    elif method == "ridge":
        from train import predict_ridge

        from features import features
        x_test = features(args.images, rows)
        values = predict_ridge(model, x_test)
    else:
        raise ValueError(f"Unknown method: {method}")

    if not np.isfinite(values).all():
        raise ValueError("Nonfinite predictions")

    write_submission(rows, values, args.output)