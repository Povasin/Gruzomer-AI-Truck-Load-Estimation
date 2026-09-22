from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================
# LOCAL IMPORTS
# ============================================================

try:
    from .model import build_model
    from .dataset import (
        CeilingSegmentationDataset,
        validate_pairs,
    )
except ImportError:
    from model import build_model
    from dataset import (
        CeilingSegmentationDataset,
        validate_pairs,
    )


# ============================================================
# PROJECT PATHS
# ============================================================

# train.py:
# project/src/manual/roof_segmentation/train.py
PROJECT_DIR = Path(__file__).resolve().parents[3]


DEFAULT_IMAGE_DIR = (
    PROJECT_DIR
    / "DataSet"
    / "train"
    / "floor_images"
)

DEFAULT_MASK_DIR = (
    PROJECT_DIR
    / "debug_output"
    / "roof_masks"
)

DEFAULT_SELECTION_CSV = (
    PROJECT_DIR
    / "debug_output"
    / "roof_selection.csv"
)

DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parent
    / "models"
)


# ============================================================
# SEED
# ============================================================


def set_seed(
    seed: int,
) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOSS
# ============================================================


class DiceLoss(nn.Module):
    """
    Soft Dice Loss.

    logits:
        [B, 1, H, W]

    target:
        [B, 1, H, W]
        values {0, 1}
    """

    def __init__(
        self,
        smooth: float = 1.0,
    ) -> None:

        super().__init__()

        self.smooth = smooth


    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = probabilities.flatten(
            start_dim=1
        )

        target = target.flatten(
            start_dim=1
        )


        intersection = (
            probabilities
            * target
        ).sum(
            dim=1
        )


        denominator = (
            probabilities.sum(
                dim=1
            )
            +
            target.sum(
                dim=1
            )
        )


        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            denominator
            + self.smooth
        )


        return (
            1.0 - dice
        ).mean()


class CombinedLoss(nn.Module):
    """
    BCE + Dice.

    BCE хорошо обучает каждый пиксель.
    Dice помогает при дисбалансе ceiling/background.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:

        super().__init__()

        self.bce = (
            nn.BCEWithLogitsLoss()
        )

        self.dice = (
            DiceLoss()
        )

        self.bce_weight = (
            bce_weight
        )

        self.dice_weight = (
            dice_weight
        )


    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:

        bce_loss = self.bce(
            logits,
            target,
        )

        dice_loss = self.dice(
            logits,
            target,
        )


        return (
            self.bce_weight
            * bce_loss
            +
            self.dice_weight
            * dice_loss
        )


# ============================================================
# METRICS
# ============================================================


@torch.no_grad()
def batch_iou(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    IoU считается отдельно для каждого изображения.

    Возвращает:
        tensor [B]
    """

    predictions = (
        probabilities
        >= threshold
    )

    targets_bool = (
        targets >= 0.5
    )


    intersection = (
        predictions
        & targets_bool
    ).sum(
        dim=(1, 2, 3)
    ).float()


    union = (
        predictions
        | targets_bool
    ).sum(
        dim=(1, 2, 3)
    ).float()


    # Важный случай:
    #
    # target пустой
    # prediction пустой
    #
    # Это правильный prediction -> IoU = 1.
    iou = torch.where(
        union > 0,
        intersection / (union + eps),
        torch.ones_like(union),
    )


    return iou


@torch.no_grad()
def batch_dice(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    threshold: float,
    eps: float = 1e-7,
) -> torch.Tensor:

    predictions = (
        probabilities
        >= threshold
    )

    targets_bool = (
        targets >= 0.5
    )


    intersection = (
        predictions
        & targets_bool
    ).sum(
        dim=(1, 2, 3)
    ).float()


    prediction_area = (
        predictions
    ).sum(
        dim=(1, 2, 3)
    ).float()


    target_area = (
        targets_bool
    ).sum(
        dim=(1, 2, 3)
    ).float()


    denominator = (
        prediction_area
        + target_area
    )


    dice = torch.where(
        denominator > 0,
        (
            2.0 * intersection
            / (denominator + eps)
        ),
        torch.ones_like(
            denominator
        ),
    )


    return dice


# ============================================================
# DATA
# ============================================================


def read_split_ids(
    selection_csv: Path,
    mask_dir: Path,
    allow_partial: bool,
) -> tuple[
    list[str],
    list[str],
]:
    """
    Используем split из roof_selection.csv.

    НЕЛЬЗЯ делать новый random split здесь,
    иначе validation перестанет соответствовать
    заранее выбранным группам.
    """

    if not selection_csv.exists():

        raise FileNotFoundError(
            f"Не найден selection CSV: "
            f"{selection_csv}"
        )


    df = pd.read_csv(
        selection_csv
    )


    required_columns = {
        "image_id",
        "split",
    }


    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"В {selection_csv} отсутствуют "
            f"колонки: {sorted(missing)}"
        )


    df["image_id"] = (
        df["image_id"]
        .astype(str)
    )

    df["split"] = (
        df["split"]
        .astype(str)
        .str.lower()
    )


    train_df = df[
        df["split"] == "train"
    ].copy()


    validation_df = df[
        df["split"].isin(
            [
                "validation",
                "val",
            ]
        )
    ].copy()


    if train_df.empty:

        raise RuntimeError(
            "В selection CSV нет train изображений."
        )


    if validation_df.empty:

        raise RuntimeError(
            "В selection CSV нет validation изображений."
        )


    train_ids = (
        train_df["image_id"]
        .tolist()
    )

    validation_ids = (
        validation_df["image_id"]
        .tolist()
    )


    # --------------------------------------------------------
    # Проверка пересечений
    # --------------------------------------------------------

    overlap = (
        set(train_ids)
        & set(validation_ids)
    )

    if overlap:

        raise RuntimeError(
            f"Train/validation пересекаются: "
            f"{list(overlap)[:10]}"
        )


    # --------------------------------------------------------
    # Проверяем наличие masks
    # --------------------------------------------------------

    def has_mask(
        image_id: str,
    ) -> bool:

        return (
            mask_dir
            / f"{image_id}.png"
        ).exists()


    missing_train = [
        image_id
        for image_id in train_ids
        if not has_mask(image_id)
    ]


    missing_validation = [
        image_id
        for image_id in validation_ids
        if not has_mask(image_id)
    ]


    if (
        missing_train
        or missing_validation
    ):

        print()

        print(
            f"Нет train masks: "
            f"{len(missing_train)}"
        )

        print(
            f"Нет validation masks: "
            f"{len(missing_validation)}"
        )


        if not allow_partial:

            raise RuntimeError(
                "Размечены не все изображения. "
                "Заверши разметку либо запусти "
                "с --allow-partial."
            )


        train_ids = [
            image_id
            for image_id in train_ids
            if has_mask(image_id)
        ]


        validation_ids = [
            image_id
            for image_id in validation_ids
            if has_mask(image_id)
        ]


    if not train_ids:

        raise RuntimeError(
            "После проверки не осталось train данных."
        )


    if not validation_ids:

        raise RuntimeError(
            "После проверки не осталось validation данных."
        )


    return (
        train_ids,
        validation_ids,
    )


# ============================================================
# OPTIMIZER
# ============================================================


def build_optimizer(
    model: nn.Module,
    encoder_lr: float,
    decoder_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """
    Encoder обучаем осторожнее,
    потому что он pretrained на ImageNet.

    Decoder/head могут использовать больший LR.
    """

    encoder_parameters = []

    decoder_parameters = []


    for name, parameter in (
        model.named_parameters()
    ):

        if name.startswith(
            "encoder"
        ):

            encoder_parameters.append(
                parameter
            )

        else:

            decoder_parameters.append(
                parameter
            )


    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": encoder_lr,
            },
            {
                "params": decoder_parameters,
                "lr": decoder_lr,
            },
        ],
        weight_decay=weight_decay,
    )


    return optimizer


# ============================================================
# TRAIN EPOCH
# ============================================================


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler,
    use_amp: bool,
) -> float:

    model.train()


    running_loss = 0.0

    sample_count = 0


    progress = tqdm(
        loader,
        desc="Train",
        leave=False,
    )


    for batch in progress:

        images = (
            batch["image"]
            .to(
                device,
                non_blocking=True,
            )
        )


        masks = (
            batch["mask"]
            .to(
                device,
                non_blocking=True,
            )
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits = model(
                images
            )


            loss = criterion(
                logits,
                masks,
            )


        scaler.scale(
            loss
        ).backward()


        scaler.unscale_(
            optimizer
        )


        # Защита от редких больших gradients.
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )


        scaler.step(
            optimizer
        )


        scaler.update()


        batch_size = (
            images.shape[0]
        )


        running_loss += (
            float(loss.item())
            * batch_size
        )


        sample_count += (
            batch_size
        )


        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )


    return (
        running_loss
        / max(sample_count, 1)
    )


# ============================================================
# VALIDATION
# ============================================================


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> dict:

    model.eval()


    running_loss = 0.0

    sample_count = 0


    all_probabilities = []

    all_targets = []


    progress = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )


    for batch in progress:

        images = (
            batch["image"]
            .to(
                device,
                non_blocking=True,
            )
        )


        masks = (
            batch["mask"]
            .to(
                device,
                non_blocking=True,
            )
        )


        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            logits = model(
                images
            )


            loss = criterion(
                logits,
                masks,
            )


        probabilities = (
            torch.sigmoid(
                logits
            )
        )


        batch_size = (
            images.shape[0]
        )


        running_loss += (
            float(loss.item())
            * batch_size
        )


        sample_count += (
            batch_size
        )


        # CPU:
        # validation всего ~80 изображений,
        # поэтому можем спокойно сохранить probability maps
        # и подобрать хороший threshold.
        all_probabilities.append(
            probabilities
            .float()
            .cpu()
        )


        all_targets.append(
            masks
            .float()
            .cpu()
        )


    probabilities = torch.cat(
        all_probabilities,
        dim=0,
    )


    targets = torch.cat(
        all_targets,
        dim=0,
    )


    # --------------------------------------------------------
    # Threshold calibration
    # --------------------------------------------------------

    thresholds = np.arange(
        0.30,
        0.71,
        0.05,
    )


    best_threshold = 0.5

    best_iou = -1.0

    best_dice = -1.0


    threshold_results = {}


    for threshold in thresholds:

        threshold = float(
            round(
                threshold,
                2,
            )
        )


        iou = float(
            batch_iou(
                probabilities,
                targets,
                threshold,
            )
            .mean()
            .item()
        )


        dice = float(
            batch_dice(
                probabilities,
                targets,
                threshold,
            )
            .mean()
            .item()
        )


        threshold_results[
            str(threshold)
        ] = {
            "iou": iou,
            "dice": dice,
        }


        if iou > best_iou:

            best_iou = iou

            best_dice = dice

            best_threshold = (
                threshold
            )


    # Метрика на стандартном 0.5
    fixed_iou = float(
        batch_iou(
            probabilities,
            targets,
            0.5,
        )
        .mean()
        .item()
    )


    fixed_dice = float(
        batch_dice(
            probabilities,
            targets,
            0.5,
        )
        .mean()
        .item()
    )


    return {
        "loss": (
            running_loss
            / max(sample_count, 1)
        ),
        "iou_05": fixed_iou,
        "dice_05": fixed_dice,
        "best_threshold": best_threshold,
        "best_iou": best_iou,
        "best_dice": best_dice,
        "threshold_results": threshold_results,
    }


# ============================================================
# CHECKPOINT
# ============================================================


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    args: argparse.Namespace,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    checkpoint = {
        "architecture": (
            "CeilingUNetResNet18"
        ),
        "state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "epoch": int(
            epoch
        ),
        "image_size": int(
            args.image_size
        ),
        "threshold": float(
            metrics["best_threshold"]
        ),
        "val_loss": float(
            metrics["loss"]
        ),
        "val_iou": float(
            metrics["best_iou"]
        ),
        "val_dice": float(
            metrics["best_dice"]
        ),
        "normalization": "imagenet",
    }


    torch.save(
        checkpoint,
        path,
    )


# ============================================================
# ARGUMENTS
# ============================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Train Ceiling U-Net ResNet18"
        )
    )


    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
    )


    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=DEFAULT_MASK_DIR,
    )


    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=DEFAULT_SELECTION_CSV,
    )


    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
    )


    parser.add_argument(
        "--image-size",
        type=int,
        default=320,
    )


    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )


    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )


    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=1e-4,
    )


    parser.add_argument(
        "--decoder-lr",
        type=float,
        default=3e-4,
    )


    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )


    parser.add_argument(
        "--bce-weight",
        type=float,
        default=0.5,
    )


    parser.add_argument(
        "--dice-weight",
        type=float,
        default=0.5,
    )


    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=2,
        help=(
            "Первые N эпох encoder ResNet18 заморожен."
        ),
    )


    parser.add_argument(
        "--patience",
        type=int,
        default=7,
    )


    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )


    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
        default="auto",
    )


    parser.add_argument(
        "--no-pretrained",
        action="store_true",
    )


    parser.add_argument(
        "--no-amp",
        action="store_true",
    )


    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help=(
            "Разрешить обучение, если размечены "
            "не все выбранные 400 изображений."
        ),
    )


    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    args = parse_args()


    set_seed(
        args.seed
    )


    # ========================================================
    # DEVICE
    # ========================================================

    if args.device == "auto":

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:

        device = torch.device(
            args.device
        )


    if (
        device.type == "cuda"
        and
        not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "Указан --device cuda, "
            "но CUDA недоступна."
        )


    use_amp = (
        device.type == "cuda"
        and
        not args.no_amp
    )


    if device.type == "cuda":

        torch.backends.cudnn.benchmark = True


    print()
    print(
        "=========================================="
    )

    print(
        "CEILING SEGMENTATION TRAINING"
    )

    print(
        "=========================================="
    )

    print(
        f"Device: {device}"
    )


    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )


    print(
        f"AMP: {use_amp}"
    )

    print(
        f"Image size: "
        f"{args.image_size}x{args.image_size}"
    )


    # ========================================================
    # DATA
    # ========================================================

    train_ids, validation_ids = (
        read_split_ids(
            selection_csv=args.selection_csv,
            mask_dir=args.mask_dir,
            allow_partial=args.allow_partial,
        )
    )


    validate_pairs(
        train_ids,
        args.image_dir,
        args.mask_dir,
    )


    validate_pairs(
        validation_ids,
        args.image_dir,
        args.mask_dir,
    )


    print()

    print(
        f"Train: {len(train_ids)}"
    )

    print(
        f"Validation: "
        f"{len(validation_ids)}"
    )


    train_dataset = (
        CeilingSegmentationDataset(
            image_ids=train_ids,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            image_size=args.image_size,
            augment=True,
            normalize=True,
        )
    )


    validation_dataset = (
        CeilingSegmentationDataset(
            image_ids=validation_ids,
            image_dir=args.image_dir,
            mask_dir=args.mask_dir,
            image_size=args.image_size,
            augment=False,
            normalize=True,
        )
    )


    # reproducible shuffle
    generator = torch.Generator()

    generator.manual_seed(
        args.seed
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
        generator=generator,
        drop_last=False,
    )


    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        ),
        drop_last=False,
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = build_model(
        pretrained=(
            not args.no_pretrained
        )
    )


    model = model.to(
        device
    )


    # --------------------------------------------------------
    # Freeze pretrained encoder at beginning.
    # --------------------------------------------------------

    if args.freeze_epochs > 0:

        model.freeze_encoder()

        print(
            f"Encoder frozen for first "
            f"{args.freeze_epochs} epochs."
        )


    # ========================================================
    # LOSS
    # ========================================================

    criterion = CombinedLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = build_optimizer(
        model=model,
        encoder_lr=args.encoder_lr,
        decoder_lr=args.decoder_lr,
        weight_decay=args.weight_decay,
    )


    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )
    )


    # ========================================================
    # AMP
    # ========================================================

    if hasattr(
        torch,
        "amp",
    ):

        scaler = (
            torch.amp.GradScaler(
                "cuda",
                enabled=use_amp,
            )
        )

    else:

        scaler = (
            torch.cuda.amp.GradScaler(
                enabled=use_amp
            )
        )


    # ========================================================
    # OUTPUTS
    # ========================================================

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    best_iou_path = (
        args.output_dir
        / "ceiling_unet_resnet18.best_iou.pt"
    )


    best_loss_path = (
        args.output_dir
        / "ceiling_unet_resnet18.best_loss.pt"
    )


    history_path = (
        args.output_dir
        / "ceiling_unet_resnet18.history.json"
    )


    # ========================================================
    # TRAIN LOOP
    # ========================================================

    best_iou = -1.0

    best_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    history = []


    for epoch in range(
        1,
        args.epochs + 1,
    ):

        print()
        print(
            f"===== EPOCH "
            f"{epoch}/{args.epochs} ====="
        )


        # ----------------------------------------------------
        # Unfreeze encoder.
        # ----------------------------------------------------

        if (
            args.freeze_epochs > 0
            and
            epoch
            == args.freeze_epochs + 1
        ):

            model.unfreeze_encoder()

            print(
                "ResNet18 encoder unfrozen."
            )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val = validate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            use_amp=use_amp,
        )


        scheduler.step(
            val["loss"]
        )


        encoder_lr = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        decoder_lr = (
            optimizer.param_groups[1][
                "lr"
            ]
        )


        epoch_info = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_iou_05": val["iou_05"],
            "val_dice_05": val["dice_05"],
            "best_threshold": (
                val["best_threshold"]
            ),
            "val_best_iou": (
                val["best_iou"]
            ),
            "val_best_dice": (
                val["best_dice"]
            ),
            "encoder_lr": encoder_lr,
            "decoder_lr": decoder_lr,
        }


        history.append(
            epoch_info
        )


        print(
            f"train_loss: "
            f"{train_loss:.5f}"
        )

        print(
            f"val_loss:   "
            f"{val['loss']:.5f}"
        )

        print(
            f"IoU @0.50:  "
            f"{val['iou_05']:.4f}"
        )

        print(
            f"Dice @0.50: "
            f"{val['dice_05']:.4f}"
        )

        print(
            f"Best threshold: "
            f"{val['best_threshold']:.2f}"
        )

        print(
            f"Best IoU:   "
            f"{val['best_iou']:.4f}"
        )

        print(
            f"Best Dice:  "
            f"{val['best_dice']:.4f}"
        )

        print(
            f"LR encoder/decoder: "
            f"{encoder_lr:.2e} / "
            f"{decoder_lr:.2e}"
        )


        # ----------------------------------------------------
        # BEST LOSS
        # ----------------------------------------------------

        if (
            val["loss"]
            < best_loss
        ):

            best_loss = (
                val["loss"]
            )


            save_checkpoint(
                path=best_loss_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val,
                args=args,
            )


            print(
                f"[SAVE BEST LOSS] "
                f"{best_loss_path}"
            )


        # ----------------------------------------------------
        # BEST IOU
        # ----------------------------------------------------

        if (
            val["best_iou"]
            > best_iou
        ):

            best_iou = (
                val["best_iou"]
            )


            epochs_without_improvement = 0


            save_checkpoint(
                path=best_iou_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val,
                args=args,
            )


            print(
                f"[SAVE BEST IOU] "
                f"{best_iou_path}"
            )

        else:

            epochs_without_improvement += 1


        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        with open(
            history_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                {
                    "config": {
                        "image_size": (
                            args.image_size
                        ),
                        "epochs": (
                            args.epochs
                        ),
                        "batch_size": (
                            args.batch_size
                        ),
                        "encoder_lr": (
                            args.encoder_lr
                        ),
                        "decoder_lr": (
                            args.decoder_lr
                        ),
                        "freeze_epochs": (
                            args.freeze_epochs
                        ),
                        "bce_weight": (
                            args.bce_weight
                        ),
                        "dice_weight": (
                            args.dice_weight
                        ),
                        "train_count": (
                            len(train_ids)
                        ),
                        "validation_count": (
                            len(validation_ids)
                        ),
                    },
                    "best_iou": (
                        best_iou
                    ),
                    "best_loss": (
                        best_loss
                    ),
                    "history": history,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )


        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= args.patience
        ):

            print()
            print(
                "EARLY STOPPING"
            )

            print(
                f"IoU не улучшался "
                f"{args.patience} эпох."
            )

            break


    # ========================================================
    # FINISH
    # ========================================================

    print()
    print(
        "=========================================="
    )

    print(
        "TRAINING FINISHED"
    )

    print(
        "=========================================="
    )

    print(
        f"Best validation IoU: "
        f"{best_iou:.4f}"
    )

    print(
        f"Best validation loss: "
        f"{best_loss:.5f}"
    )

    print()

    print(
        f"Best IoU model:"
        f"\n{best_iou_path}"
    )

    print()

    print(
        f"History:"
        f"\n{history_path}"
    )


if __name__ == "__main__":

    main()