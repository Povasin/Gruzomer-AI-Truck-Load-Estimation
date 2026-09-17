import numpy as np

from features import features
from report import read_test_table
from submission import write_submission
from train import predict_ridge


def predict(args):
    rows = read_test_table(args.test_csv)

    with np.load(args.model, allow_pickle=False) as saved:
        model = {key: saved[key] for key in saved.files}

    if str(model["method"]) == "median":
        values = np.full(len(rows), float(model["median"]))
    else:
        x_test = features(args.images, rows)
        values = predict_ridge(model, x_test)

    if not np.isfinite(values).all():
        raise ValueError("Nonfinite predictions")

    write_submission(rows, values, args.output)
