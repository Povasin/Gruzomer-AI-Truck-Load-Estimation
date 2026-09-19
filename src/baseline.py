"""Image-regression baseline with a fixed train/validation split."""

import argparse

from predict import predict
from train import train


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train")
    train_parser.add_argument(
        "--train-split",
        default="./DataSet/train/train_split.csv",
        help="Path to the already prepared train_split.csv",
    )
    train_parser.add_argument(
        "--validation-split",
        default="./DataSet/train/validation_split.csv",
        help="Path to the already prepared validation_split.csv",
    )
    train_parser.add_argument(
        "--images",
        default="./DataSet/train/images",
        help="Directory containing all TRAIN jpg images",
    )
    train_parser.add_argument("--model", required=True, default="./models/model.npz")
    train_parser.add_argument(
        "--errors-output",
        required=True,
        help=(
            "Path to CSV report with validation errors, sorted by absolute error"
        ),
    )
    train_parser.add_argument("--alpha", type=float, default=70.0)
    train_parser.add_argument(
        "--method",
        choices=["auto", "ridge", "boosting", "median"],
        default="auto",
    )
    train_parser.set_defaults(run=train)

    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--model", required=True, default="./models/model.npz")
    predict_parser.add_argument("--test-csv", default="./DataSet/test/test.csv")
    predict_parser.add_argument("--images", default="./DataSet/test/images")
    predict_parser.add_argument("--output",required=True, default="./output_csv/submission_check.csv")
    predict_parser.set_defaults(run=predict)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
