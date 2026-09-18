from pathlib import Path
import pickle
import numpy as np

from features import features
from report import read_test_table
from submission import write_submission
from train import predict_ridge


def predict(args):
    rows = read_test_table(args.test_csv)
    model_path = Path(args.model)

    # Проверяем наличие .pkl (для бустинга) или .npz (для Ridge)
    pkl_candidate = model_path.with_suffix(".pkl")
    if pkl_candidate.is_file():
        with open(pkl_candidate, "rb") as f:
            data = pickle.load(f)
        method = data["method"]
        model = data["model"]
    else:
        with np.load(model_path, allow_pickle=False) as saved:
            model = {key: saved[key] for key in saved.files}
        method = str(model["method"])

    if method == "median":
        values = np.full(len(rows), float(model["median"]))
    elif method == "boosting":
        x_test = features(args.images, rows)
        values = np.clip(model.predict(x_test), 0, 100)
    else:
        x_test = features(args.images, rows)
        values = predict_ridge(model, x_test)

    if not np.isfinite(values).all():
        raise ValueError("Nonfinite predictions")

    write_submission(rows, values, args.output)