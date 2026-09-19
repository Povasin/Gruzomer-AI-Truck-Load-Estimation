from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from .dataset import FloorSegmentationDataset
    from .model import FloorUNet
except ImportError:
    from dataset import FloorSegmentationDataset
    from model import FloorUNet


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        dimensions = (1, 2, 3)
        intersection = (probabilities * targets).sum(dim=dimensions)
        denominator = probabilities.sum(dim=dimensions) + targets.sum(dim=dimensions)
        dice_score = (2.0 * intersection + 1.0) / (denominator + 1.0)
        return bce_loss + (1.0 - dice_score).mean()


class FloorTrainTransform:
    def __init__(self, horizontal_flip_probability: float = 0.0):
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in [0, 1]")
        self.horizontal_flip_probability = horizontal_flip_probability

    def __call__(self, image: np.ndarray, mask: np.ndarray):
        if np.random.random() < self.horizontal_flip_probability:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        return image, mask


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def read_ids(path: str | Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def checkpoint_paths(output_path: str | Path) -> dict[str, Path]:
    output_path = Path(output_path)
    suffix = output_path.suffix or ".pt"
    stem = output_path.stem if output_path.suffix else output_path.name
    return {
        "best_dice": output_path,
        "best_iou": output_path.with_name(f"{stem}.best_iou{suffix}"),
        "best_loss": output_path.with_name(f"{stem}.best_loss{suffix}"),
        "last": output_path.with_name(f"{stem}.last{suffix}"),
    }


def history_path(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    stem = output_path.stem if output_path.suffix else output_path.name
    return output_path.with_name(f"{stem}.history.csv")


def append_history_row(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def segmentation_batch_statistics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float | int]:
    predictions = predictions.flatten(start_dim=1).float()
    targets = targets.flatten(start_dim=1).float()
    intersection = (predictions * targets).sum(dim=1)
    predicted_pixels = predictions.sum(dim=1)
    target_pixels = targets.sum(dim=1)
    dice_denominator = predicted_pixels + target_pixels
    union = dice_denominator - intersection
    eps = 1e-7
    dice = (2.0 * intersection + eps) / (dice_denominator + eps)
    iou = (intersection + eps) / (union + eps)
    dice = torch.where(dice_denominator == 0, torch.ones_like(dice), dice)
    iou = torch.where(union == 0, torch.ones_like(iou), iou)
    positive = target_pixels > 0
    empty = ~positive
    return {
        "count": int(targets.shape[0]),
        "dice_sum": float(dice.sum().item()),
        "iou_sum": float(iou.sum().item()),
        "positive_count": int(positive.sum().item()),
        "positive_dice_sum": float(dice[positive].sum().item()),
        "positive_iou_sum": float(iou[positive].sum().item()),
        "empty_count": int(empty.sum().item()),
        "empty_exact_correct": int((empty & (predicted_pixels == 0)).sum().item()),
        "empty_false_positive_pixels": int(predicted_pixels[empty].sum().item()),
        "true_positive_pixels": int(intersection.sum().item()),
        "predicted_pixels": int(predicted_pixels.sum().item()),
        "target_pixels": int(target_pixels.sum().item()),
    }


def _safe_average(total: float, count: int) -> float:
    return total / count if count else 0.0


def evaluate(model, loader, criterion, device, threshold: float = 0.5):
    model.eval()
    totals = {
        "loss": 0.0,
        "count": 0,
        "dice_sum": 0.0,
        "iou_sum": 0.0,
        "positive_count": 0,
        "positive_dice_sum": 0.0,
        "positive_iou_sum": 0.0,
        "empty_count": 0,
        "empty_exact_correct": 0,
        "empty_false_positive_pixels": 0,
        "true_positive_pixels": 0,
        "predicted_pixels": 0,
        "target_pixels": 0,
    }
    with torch.inference_mode():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            batch_size = images.shape[0]
            totals["loss"] += float(criterion(logits, masks).item()) * batch_size
            predictions = (torch.sigmoid(logits) >= threshold).float()
            batch = segmentation_batch_statistics(predictions, masks)
            for key, value in batch.items():
                totals[key] += value

    count = int(totals["count"])
    positive_count = int(totals["positive_count"])
    empty_count = int(totals["empty_count"])
    true_positive = float(totals["true_positive_pixels"])
    predicted = float(totals["predicted_pixels"])
    target = float(totals["target_pixels"])
    micro_dice_denominator = predicted + target
    micro_union = predicted + target - true_positive
    return {
        "loss": _safe_average(float(totals["loss"]), count),
        "dice": _safe_average(float(totals["dice_sum"]), count),
        "iou": _safe_average(float(totals["iou_sum"]), count),
        "positive_dice": _safe_average(float(totals["positive_dice_sum"]), positive_count),
        "positive_iou": _safe_average(float(totals["positive_iou_sum"]), positive_count),
        "empty_accuracy": _safe_average(float(totals["empty_exact_correct"]), empty_count),
        "empty_mean_false_positive_pixels": _safe_average(
            float(totals["empty_false_positive_pixels"]), empty_count
        ),
        "micro_dice": 2.0 * true_positive / micro_dice_denominator
        if micro_dice_denominator
        else 1.0,
        "micro_iou": true_positive / micro_union if micro_union else 1.0,
        "count": count,
        "positive_count": positive_count,
        "empty_count": empty_count,
        "empty_exact_correct": int(totals["empty_exact_correct"]),
    }


def _checkpoint(model, optimizer, scheduler, epoch, train_loss, metrics, best, args):
    return {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "train_loss": train_loss,
        "valid_loss": metrics["loss"],
        "valid_dice": metrics["dice"],
        "valid_iou": metrics["iou"],
        "valid_metrics": metrics,
        "best_valid_loss": best["loss"],
        "best_valid_dice": best["dice"],
        "best_valid_iou": best["iou"],
        "threshold": args.threshold,
        "size": args.size,
        "seed": args.seed,
        "lr": args.lr,
    }


def train(args: argparse.Namespace) -> None:
    set_reproducibility(args.seed)
    train_ids = read_ids(args.train_ids)
    valid_ids = read_ids(args.valid_ids)
    if not train_ids or not valid_ids:
        raise ValueError("Train and validation ID lists must both be non-empty")
    overlap = set(train_ids) & set(valid_ids)
    if overlap:
        raise ValueError(f"Train and validation overlap: {len(overlap)} IDs")
    print(f"Train images: {len(train_ids)}")
    print(f"Validation images: {len(valid_ids)}")

    train_dataset = FloorSegmentationDataset(
        args.image_dir,
        args.mask_dir,
        train_ids,
        size=args.size,
        transform=FloorTrainTransform(args.horizontal_flip_probability),
    )
    valid_dataset = FloorSegmentationDataset(
        args.image_dir, args.mask_dir, valid_ids, size=args.size
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_dataset, shuffle=True, generator=generator, **loader_options
    )
    valid_loader = DataLoader(valid_dataset, shuffle=False, **loader_options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    model = FloorUNet(pretrained=args.resume is None).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.model.encoder.parameters(), "lr": args.lr * 0.1},
            {"params": model.model.decoder.parameters(), "lr": args.lr},
            {"params": model.model.segmentation_head.parameters(), "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=args.scheduler_patience
        )

    paths = checkpoint_paths(args.output)
    metrics_history_path = history_path(args.output)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume and metrics_history_path.exists():
        metrics_history_path.unlink()
    start_epoch = 1
    best = {"loss": float("inf"), "dice": -1.0, "iou": -1.0}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint.get("state_dict", checkpoint))
        if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            for group, learning_rate in zip(
                optimizer.param_groups, (args.lr * 0.1, args.lr, args.lr), strict=True
            ):
                group["lr"] = learning_rate
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best["loss"] = float(
            checkpoint.get("best_valid_loss", checkpoint.get("valid_loss", float("inf")))
        )
        best["dice"] = float(
            checkpoint.get("best_valid_dice", checkpoint.get("valid_dice", -1.0))
        )
        best["iou"] = float(
            checkpoint.get("best_valid_iou", checkpoint.get("valid_iou", -1.0))
        )
        print(f"Resume from epoch {start_epoch}; optimizer LR overridden from CLI")

    epochs_without_dice_improvement = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_examples = 0
        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            train_loss_sum += float(loss.item()) * images.shape[0]
            train_examples += images.shape[0]

        train_loss = train_loss_sum / train_examples
        metrics = evaluate(model, valid_loader, criterion, device, args.threshold)
        if scheduler:
            scheduler.step(metrics["dice"])
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        print(f"\n===== EPOCH {epoch}/{args.epochs} =====")
        print(f"train_loss          = {train_loss:.5f}")
        print(f"valid_loss          = {metrics['loss']:.5f}")
        print(f"valid_dice          = {metrics['dice']:.5f}")
        print(f"valid_iou           = {metrics['iou']:.5f}")
        print(f"positive_dice       = {metrics['positive_dice']:.5f}")
        print(f"positive_iou        = {metrics['positive_iou']:.5f}")
        print(f"empty_exact         = {metrics['empty_exact_correct']}/{metrics['empty_count']}")
        print(f"empty_mean_fp_pixels = {metrics['empty_mean_false_positive_pixels']:.1f}")
        print(f"micro_dice          = {metrics['micro_dice']:.5f}")
        print(f"micro_iou           = {metrics['micro_iou']:.5f}")
        print(f"learning_rates      = {learning_rates}")
        append_history_row(
            metrics_history_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **metrics,
                "encoder_lr": learning_rates[0],
                "decoder_lr": learning_rates[1],
                "head_lr": learning_rates[2],
            },
        )

        improved = {
            "loss": metrics["loss"] < best["loss"],
            "dice": metrics["dice"] > best["dice"],
            "iou": metrics["iou"] > best["iou"],
        }
        for name in best:
            if improved[name]:
                best[name] = metrics[name]
        checkpoint = _checkpoint(
            model, optimizer, scheduler, epoch, train_loss, metrics, best, args
        )
        torch.save(checkpoint, paths["last"])
        for name in ("loss", "dice", "iou"):
            if improved[name]:
                torch.save(checkpoint, paths[f"best_{name}"])
                print(f"Saved best {name}: {paths[f'best_{name}']}")

        if improved["dice"]:
            epochs_without_dice_improvement = 0
        else:
            epochs_without_dice_improvement += 1
        if (
            args.early_stopping_patience > 0
            and epochs_without_dice_improvement >= args.early_stopping_patience
        ):
            print("Early stopping: validation Dice did not improve")
            break

    print("\nTRAINING FINISHED")
    print(f"Best validation loss: {best['loss']:.5f}")
    print(f"Best validation Dice: {best['dice']:.5f}")
    print(f"Best validation IoU:  {best['iou']:.5f}")
    print(f"Best Dice checkpoint: {paths['best_dice']}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--train-ids", required=True)
    parser.add_argument("--valid-ids", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=0.0)
    parser.add_argument("--scheduler-patience", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
