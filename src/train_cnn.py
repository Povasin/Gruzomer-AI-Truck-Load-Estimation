"""Обучение TruckLoadNet с group k-fold кросс-валидацией.

Использование:
    python train_cnn.py \
        --folds ./DataSet/train/folds.csv \
        --images ./DataSet/train/images \
        --output-dir ./models/cnn_v1 \
        --backbone resnet18 --img-size 320 \
        --phase1-epochs 6 --phase2-epochs 18 \
        --batch-size 32 --amp

Двухфазное обучение:
    Фаза 1 (phase1-epochs): backbone заморожен, учится только голова, LR
        повыше (head-lr) — быстро и без риска "сломать" предобученные веса.
    Фаза 2 (phase2-epochs): backbone разморожен целиком, но с более низким
        LR (backbone-lr) через отдельную param-группу; головы продолжают
        учиться с head-lr.

Сохраняет по одному лучшему (по val MAE) чекпоинту на фолд
(models/cnn_v1/fold{K}_best.pt) и общий отчёт metrics.json с MAE по фолдам
и усреднённым MAE.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TruckLoadDataset
from model import TruckLoadNet


def rows_from_dataframe(df):
    return df.to_dict(orient="records")


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def run_epoch(model, loader, optimizer, scaler, device, aux_weight, train):
    model.train(mode=train)
    total_loss = 0.0
    all_targets, all_preds = [], []

    regression_loss_fn = nn.L1Loss()
    classification_loss_fn = nn.CrossEntropyLoss()

    with torch.set_grad_enabled(train):
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            load_pct = batch["load_pct"].to(device, non_blocking=True)
            load_bin = batch["load_bin"].to(device, non_blocking=True)
            cargo_type = batch["cargo_type"].to(device, non_blocking=True)

            if train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", enabled=scaler is not None):
                pred_pct, load_bin_logits, cargo_logits = model(images)
                regression_loss = regression_loss_fn(pred_pct, load_pct)
                aux_loss = classification_loss_fn(
                    load_bin_logits, load_bin
                ) + classification_loss_fn(cargo_logits, cargo_type)
                loss = regression_loss + aux_weight * aux_loss

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += float(loss.detach()) * len(load_pct)
            all_targets.extend(load_pct.detach().cpu().numpy().tolist())
            all_preds.extend(pred_pct.detach().cpu().numpy().tolist())

    return {
        "loss": total_loss / len(all_targets),
        "mae": mae(all_targets, all_preds),
    }


def train_one_fold(args, df, fold_index, device):
    train_df = df[df["fold"] != fold_index]
    val_df = df[df["fold"] == fold_index]

    train_rows = rows_from_dataframe(train_df)
    val_rows = rows_from_dataframe(val_df)

    train_dataset = TruckLoadDataset(
        train_rows, args.images, img_size=args.img_size, train=True
    )
    val_dataset = TruckLoadDataset(
        val_rows, args.images, img_size=args.img_size, train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=len(train_dataset) > args.batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = TruckLoadNet(backbone_name=args.backbone, pretrained=True).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # --- Фаза 1: backbone заморожен, обучаются только головы ---
    model.set_backbone_trainable(False)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.head_lr,
        weight_decay=1e-4,
    )

    best_val_mae = float("inf")
    best_state = None
    epochs_without_improvement = 0
    history = []

    total_epochs = args.phase1_epochs + args.phase2_epochs

    for epoch in range(total_epochs):
        # Переход в фазу 2: размораживаем backbone, ставим отдельную (более
        # низкую) LR через новый оптимизатор с двумя param-группами.
        if epoch == args.phase1_epochs:
            model.set_backbone_trainable(True)
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.backbone.parameters(), "lr": args.backbone_lr},
                    {
                        "params": list(model.regression_head.parameters())
                        + list(model.load_bin_head.parameters())
                        + list(model.cargo_type_head.parameters()),
                        "lr": args.head_lr,
                    },
                ],
                weight_decay=1e-4,
            )

        start = time.time()
        train_metrics = run_epoch(
            model, train_loader, optimizer, scaler if args.amp else None,
            device, args.aux_weight, train=True,
        )
        val_metrics = run_epoch(
            model, val_loader, optimizer=None, scaler=None,
            device=device, aux_weight=args.aux_weight, train=False,
        )
        elapsed = time.time() - start

        history.append(
            {
                "epoch": epoch,
                "phase": 1 if epoch < args.phase1_epochs else 2,
                "train_loss": train_metrics["loss"],
                "train_mae": train_metrics["mae"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "seconds": elapsed,
            }
        )
        print(
            f"[fold {fold_index}] epoch {epoch + 1}/{total_epochs} "
            f"train_mae={train_metrics['mae']:.2f} "
            f"val_mae={val_metrics['mae']:.2f} ({elapsed:.1f}s)",
            flush=True,
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"[fold {fold_index}] early stopping на эпохе {epoch + 1}")
            break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"fold{fold_index}_best.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "backbone": args.backbone,
            "img_size": args.img_size,
            "val_mae": best_val_mae,
        },
        checkpoint_path,
    )
    history_path = output_dir / f"fold{fold_index}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"[fold {fold_index}] лучший val MAE={best_val_mae:.3f} -> {checkpoint_path}")
    return best_val_mae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default="./DataSet/train/folds.csv")
    parser.add_argument("--images", default="./DataSet/train/images")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--phase1-epochs", type=int, default=6)
    parser.add_argument("--phase2-epochs", type=int, default=18)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--amp", action="store_true", help="Mixed precision (только на GPU)")
    parser.add_argument(
        "--folds-to-run",
        type=str,
        default="",
        help="Список фолдов через запятую (например '0,1'); по умолчанию все.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    df = pd.read_csv(args.folds)
    all_folds = sorted(df["fold"].unique().tolist())
    folds_to_run = (
        [int(f) for f in args.folds_to_run.split(",")]
        if args.folds_to_run
        else all_folds
    )

    fold_scores = {}
    for fold_index in folds_to_run:
        fold_scores[fold_index] = train_one_fold(args, df, fold_index, device)

    report = {
        "backbone": args.backbone,
        "img_size": args.img_size,
        "folds": fold_scores,
        "mean_val_mae": float(np.mean(list(fold_scores.values()))),
        "std_val_mae": float(np.std(list(fold_scores.values()))),
    }
    report_path = Path(args.output_dir) / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
