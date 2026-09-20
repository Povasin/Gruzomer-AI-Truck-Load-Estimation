"""Image-regression baseline with a fixed train/validation split."""

import argparse

from manual.feature_schema import TOTAL_FEATURE_DIM

def train(args):
    if args.method == "hybrid":
        from hybrid.hybrid import train as run
    else:
        from train import train as run
    run(args)


def predict(args):
    from predict import predict as run
    run(args)


def build_parser():
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
        choices=["hybrid", "auto", "ridge", "boosting", "median"],
        default="hybrid",
        help=f"hybrid: CNN + {TOTAL_FEATURE_DIM} признаков (по умолчанию); auto: прежний выбор регрессора",
    )
    train_parser.add_argument("--backbone", default="resnet18")
    train_parser.add_argument("--img-size", type=int, default=320)
    train_parser.add_argument("--phase1-epochs", type=int, default=6)
    train_parser.add_argument("--phase2-epochs", type=int, default=18)
    train_parser.add_argument("--head-lr", type=float, default=1e-3)
    train_parser.add_argument("--backbone-lr", type=float, default=1e-4)
    train_parser.add_argument("--aux-weight", type=float, default=0.2)
    train_parser.add_argument("--patience", type=int, default=6)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--amp", action="store_true")
    train_parser.add_argument("--no-pretrained", action="store_true", help="Не скачивать ImageNet-веса; обучение с нуля")
    train_parser.set_defaults(run=train)

    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("--model", required=True, default="./models/model.npz")
    predict_parser.add_argument("--test-csv", default="./DataSet/test/test.csv")
    predict_parser.add_argument("--images", default="./DataSet/test/images")
    predict_parser.add_argument("--output",required=True, default="./output_csv/submission_check.csv")
    predict_parser.set_defaults(run=predict)

    for command_parser in (train_parser, predict_parser):
        command_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        command_parser.add_argument("--batch-size", type=int, default=16)
        command_parser.add_argument("--num-workers", type=int, default=0)
        command_parser.add_argument("--truck-weights", default="./src/manual/truck_segmentation/models/best_unet_resnet18.pth", help="Веса сегментации кузова для гибрида")
        command_parser.add_argument("--floor-weights", default="./src/manual/floor_segmentation/models/floor_unet_resnet18_lr1e3.best_loss.pt", help="Веса сегментации пола для гибрида")
        command_parser.add_argument("--feature-cache", help=f"Каталог кэша {TOTAL_FEATURE_DIM} признаков")
    return parser


def main():
    args = build_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
